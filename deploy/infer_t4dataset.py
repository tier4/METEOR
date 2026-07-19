#!/usr/bin/env python3
"""Run METEOR on a raw t4dataset scene with TensorRT — end to end.

Reads a t4dataset directory directly (annotation/*.json + data/CAM_*), builds
the 8-camera tensor and calibration exactly as training does, streams the
temporal memory on the device, and writes per-frame results (+ an optional
overlay video). No training code, no GT, no PyTorch at inference time.

    python3 deploy/infer_t4dataset.py \
        --engine out/meteor_v29_fp16.engine \
        --scene /data1/dataset/DTSET/<batch>/<scene> \
        --out out/t4_infer --video out/t4_infer.mp4

Outputs per frame (out/<scene>/NNNN.npz):
    boxes   [N,9]  cls, score, x, y, l, w, yaw, stationary, speed
    lane    [800,500] uint8 class map
    tl      int     0 none / 1 green / 2 yellow / 3 red   (+ tl_conf)
    ego     [3,6,2] K=3 path hypotheses (+ ego_conf [3]), steer/accel/brake
    risk    [400,250] uint8 (x255)
    occ     [16,200,200] uint8 class ids
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deploy.runtime import (MeteorRT, decode_boxes, make_warp_theta,  # noqa
                            preprocess_images)

CAMS = ["CAM_FRONT_WIDE", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
        "CAM_BACK_WIDE", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
        "CAM_FRONT_NARROW", "CAM_BACK_NARROW"]
TL_NAMES = ("none", "green", "yellow", "red")
IMG_W, IMG_H = 768, 432


def quat_to_rot(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]],
        np.float64)


def load_scene(root):
    """-> (samples ordered by time, per-sample {cam: sample_data}, calib, poses)"""
    A = os.path.join(root, "annotation")
    j = lambda n: json.load(open(os.path.join(A, n)))
    sample = j("sample.json")
    sd = j("sample_data.json")
    cs = {c["token"]: c for c in j("calibrated_sensor.json")}
    sensor = {s["token"]: s for s in j("sensor.json")}
    ego = {e["token"]: e for e in j("ego_pose.json")}

    # order samples by timestamp (t4dataset next/prev links can be unsorted)
    ordered = sorted(sample, key=lambda s: s["timestamp"])
    by_sample = {}
    calib = {}
    for d in sd:
        if not d.get("is_key_frame", True):
            continue
        c = cs.get(d["calibrated_sensor_token"])
        if c is None:
            continue
        ch = sensor.get(c["sensor_token"], {}).get("channel")
        if ch not in CAMS:
            continue
        by_sample.setdefault(d["sample_token"], {})[ch] = d
        if ch not in calib:
            K = np.array(c["camera_intrinsic"], np.float64)
            R = quat_to_rot(c["rotation"])
            t = np.array(c["translation"], np.float64)
            T_ego_cam = np.eye(4)
            T_ego_cam[:3, :3], T_ego_cam[:3, 3] = R, t
            calib[ch] = (K, np.linalg.inv(T_ego_cam), d)   # d: for image size
    return ordered, by_sample, calib, ego


def scale_K(K, w0, h0):
    K = K.copy()
    K[0] *= IMG_W / w0
    K[1] *= IMG_H / h0
    return K


def build_engine_from_onnx(onnx_path):
    """--onnx convenience: build (or reuse) a cached fp16 engine next to
    the ONNX file via stock trtexec. Returns the engine path."""
    eng = os.path.splitext(onnx_path)[0] + "_fp16.engine"
    if os.path.exists(eng) and \
            os.path.getmtime(eng) >= os.path.getmtime(onnx_path):
        print(f"[engine] reusing cached {eng}", flush=True)
        return eng
    print(f"[engine] building {eng} from {onnx_path} (trtexec --fp16, "
          "one-time, ~minutes)", flush=True)
    import subprocess
    r = subprocess.run(["trtexec", f"--onnx={onnx_path}",
                        f"--saveEngine={eng}", "--fp16"],
                       capture_output=True, text=True)
    if r.returncode or not os.path.exists(eng):
        sys.exit("trtexec failed:\n" + r.stdout[-2000:] + r.stderr[-2000:])
    return eng


def scene_dirs(t4d):
    """--t4d accepts a single scene dir OR a dataset root of scenes."""
    t4d = t4d.rstrip("/")
    if os.path.isdir(os.path.join(t4d, "annotation")):
        return [t4d]
    subs = [os.path.join(t4d, d) for d in sorted(os.listdir(t4d))
            if os.path.isdir(os.path.join(t4d, d, "annotation"))]
    if not subs:
        sys.exit(f"{t4d}: no t4dataset scenes found")
    return subs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default=None,
                    help="prebuilt TensorRT engine")
    ap.add_argument("--onnx", default=None,
                    help="ONNX file: builds/reuses a cached fp16 engine")
    ap.add_argument("--t4d", "--scene", dest="t4d", required=True,
                    help="t4dataset scene directory OR dataset root")
    ap.add_argument("--out", default="out/t4_infer")
    ap.add_argument("--video", default=None)
    ap.add_argument("--display", action="store_true",
                    help="live visualisation window while inferring "
                    "(q quits, space pauses)")
    ap.add_argument("--stride", type=int, default=2, help="keyframe stride")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--thresh", type=float, default=0.45)
    args = ap.parse_args()
    if not args.engine and not args.onnx:
        sys.exit("need --engine or --onnx")
    if not args.engine:
        args.engine = build_engine_from_onnx(args.onnx)
    if args.display and not os.environ.get("DISPLAY"):
        print("[warn] --display requested but no $DISPLAY; continuing "
              "without a window", flush=True)
        args.display = False
    for root in scene_dirs(args.t4d):
        run_scene(args, root)


def run_scene(args, root):
    name = os.path.basename(root)
    ordered, by_sample, calib, egop = load_scene(root)
    missing = [c for c in CAMS if c not in calib]
    if missing:
        sys.exit(f"scene lacks cameras: {missing}")

    # calibration is per-scene constant: build K/T once
    Ks, Ts = [], []
    for c in CAMS:
        K, T_cam_ego, d = calib[c]
        Ks.append(scale_K(K, d.get("width", 2880), d.get("height", 1860)))
        Ts.append(T_cam_ego)
    K_t = np.stack(Ks)[None].astype(np.float32)
    T_t = np.stack(Ts)[None].astype(np.float32)

    rt = MeteorRT(args.engine)
    rt.reset()
    os.makedirs(os.path.join(args.out, name), exist_ok=True)
    vw = None
    if args.video:
        vw = cv2.VideoWriter(args.video.replace(".mp4", "_raw.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), 10,
                             (1920, 1080))

    frames = ordered[::args.stride]
    if args.limit:
        frames = frames[:args.limit]
    prev_xy = None
    prev_t = None
    n = 0
    for fi, s in enumerate(frames):
        cams = by_sample.get(s["token"], {})
        if any(c not in cams for c in CAMS):
            continue
        imgs = []
        ok = True
        for c in CAMS:
            p = os.path.join(root, cams[c]["filename"])
            im = cv2.imread(p)
            if im is None:
                ok = False
                break
            imgs.append(cv2.resize(im, (IMG_W, IMG_H)))
        if not ok:
            continue
        # ego pose + speed from the pose deltas (no CAN in t4dataset)
        ep = egop.get(cams["CAM_FRONT_WIDE"]["ego_pose_token"])
        pose = None
        v0 = 0.0
        if ep is not None:
            R = quat_to_rot(ep["rotation"])
            yaw = float(np.arctan2(R[1, 0], R[0, 0]))
            xy = np.array(ep["translation"][:2], np.float64)
            t_s = cams["CAM_FRONT_WIDE"]["timestamp"] * 1e-6
            pose = (float(xy[0]), float(xy[1]), yaw)
            if prev_xy is not None and t_s > prev_t:
                v0 = float(np.linalg.norm(xy - prev_xy) / (t_s - prev_t))
            prev_xy, prev_t = xy, t_s

        out = rt.infer(preprocess_images(imgs), K_t, T_t, v0, pose=pose)

        boxes = decode_boxes(out["hm"], out["reg"], out.get("stationary"),
                             thresh=args.thresh)
        # attach the winning-mode 3 s speed for each box
        traj = out["traj"]
        for b in boxes:
            ri = int((80.0 - b["x"]) / 0.4)
            ci = int((50.0 - b["y"]) / 0.4)
            if 0 <= ri < traj.shape[-2] and 0 <= ci < traj.shape[-1]:
                v = traj[0, :, ri, ci]
                kb = int(np.argmax(v[36:39])) if v.shape[0] >= 39 else 0
                wp = v[kb * 12:(kb + 1) * 12].reshape(6, 2)
                b["future"] = wp.tolist()
                b["speed"] = float(np.linalg.norm(wp[5]) / 3.0)
        e = out["ego"][0]
        ego_paths = e[:36].reshape(3, 6, 2)
        ego_conf = np.exp(e[36:39]) / np.exp(e[36:39]).sum()
        tl_p = np.exp(out["tl"][0]) / np.exp(out["tl"][0]).sum()
        np.savez_compressed(
            os.path.join(args.out, name, f"{fi:04d}.npz"),
            boxes=np.array([[b["cls"] == "vru", b["score"], b["x"], b["y"],
                             b["l"], b["w"], b["yaw"],
                             float(bool(b.get("stationary"))),
                             b.get("speed", 0.0)] for b in boxes], np.float32),
            lane=out["lane"][0].argmax(0).astype(np.uint8),
            ego_paths=ego_paths.astype(np.float32),
            ego_conf=ego_conf.astype(np.float32),
            ctrl=e[39:42].astype(np.float32),
            tl=int(tl_p.argmax()), tl_conf=float(tl_p.max()),
            risk=(1 / (1 + np.exp(-out["risk"][0, 0])) * 255).astype(np.uint8),
            occ=out["occ"][0].argmax(0).astype(np.uint8))

        if vw is not None or args.display:
            from deploy.visualize import compose_frame
            from bevlane.guardrail import check_path
            vboxes = [(int(b["cls"] == "vru"), b["score"], b["x"], b["y"],
                       b["l"], b["w"], b["yaw"],
                       bool(b.get("stationary"))) for b in boxes]
            guard = None
            try:
                kb_ = int(ego_conf.argmax())
                path6 = ego_paths[kb_]
                tm_ = out["traj"][0]
                offs_ = []
                for b_ in vboxes:
                    rr_ = int((80.0 - b_[2]) / 0.4)
                    cc_ = int((50.0 - b_[3]) / 0.4)
                    o_ = np.zeros((6, 2), np.float32)
                    if 0 <= rr_ < tm_.shape[-2] and 0 <= cc_ < tm_.shape[-1]:
                        v_ = tm_[:, rr_, cc_]
                        if v_.shape[0] >= 39:
                            kk_ = int(v_[36:39].argmax())
                            o_ = v_[kk_ * 12:(kk_ + 1) * 12].reshape(6, 2)
                        else:
                            o_ = v_[:12].reshape(6, 2)
                    offs_.append(o_)
                opz = out["occ"][0]
                oex = np.exp(opz - opz.max(0, keepdims=True))
                opp = oex / oex.sum(0, keepdims=True)
                ocls = (opp[1:].argmax(0) + 1).astype(np.uint8)
                oconf = 1.0 - opp[0]
                othr = np.where((ocls == 7) | (ocls == 8), 0.92, 0.55)
                occ_cls = np.where(oconf > othr, ocls, 0).astype(np.uint8)
                lane_am = out["lane"][0].argmax(0).astype(np.uint8)
                guard = check_path(path6, occ_cls, vboxes, offs_, tl_p,
                                   lane_am, float(v0))
            except Exception:
                guard = None
            g = compose_frame(np.stack(imgs), K_t[0], T_t[0], out,
                              float(v0), vboxes, name, fi, guard=guard)
        if False:
            g = np.zeros((IMG_H * 2, IMG_W * 2, 3), np.uint8)
            g[:IMG_H, :IMG_W] = imgs[0]
            g[:IMG_H, IMG_W:] = imgs[6]
            # BEV panel: lane map + boxes + ego hypotheses
            lane = out["lane"][0].argmax(0).astype(np.uint8)
            bev = cv2.applyColorMap((lane * 28).astype(np.uint8),
                                    cv2.COLORMAP_PARULA)
            bev = cv2.resize(bev[200:600, 125:375], (IMG_W, IMG_H))
            # the lane BEV is 0.2 m/cell (NOT the 0.4 m detection grid):
            # crop rows 200:600 = x +40..-40 m, cols 125:375 = y +25..-25 m
            sx, sy = IMG_W / 250.0, IMG_H / 400.0
            xy2px = lambda x, y: (int(((50.0 - y) / 0.2 - 125) * sx),
                                  int(((80.0 - x) / 0.2 - 200) * sy))
            for b in boxes:
                c, r = xy2px(b["x"], b["y"])
                col = (160, 160, 160) if b.get("stationary") else \
                    ((0, 215, 255) if b["cls"] == "vehicle" else (255, 0, 255))
                cv2.circle(bev, (c, r), 5, col, -1)
            for k in range(3):
                pts = [xy2px(0.0, 0.0)]
                for x, y in ego_paths[k]:
                    pts.append(xy2px(float(x), float(y)))
                thick = 3 if k == int(ego_conf.argmax()) else 1
                cv2.polylines(bev, [np.array(pts, np.int32).reshape(-1, 1, 2)],
                              False, (0, 255, 0) if thick == 3 else (255, 200, 60),
                              thick, cv2.LINE_AA)
            g[IMG_H:, IMG_W:] = bev
            risk = (1 / (1 + np.exp(-out["risk"][0, 0])) * 255).astype(np.uint8)
            g[IMG_H:, :IMG_W] = cv2.resize(
                cv2.applyColorMap(risk, cv2.COLORMAP_TURBO), (IMG_W, IMG_H))
            cv2.putText(g, f"TL {TL_NAMES[int(tl_p.argmax())]} {tl_p.max():.2f}"
                        f" | v0 {v0 * 3.6:.0f} km/h | {len(boxes)} boxes",
                        (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(g, "TensorRT engine | t4dataset raw input", (10, IMG_H * 2 - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1,
                        cv2.LINE_AA)
            pass
        if vw is not None:
            vw.write(g)
        if args.display:
            cv2.imshow("METEOR live", g)
            kq = cv2.waitKey(1) & 0xFF
            if kq == ord('q'):
                break
            if kq == ord(' '):
                cv2.waitKey(0)
        n += 1
        if n % 20 == 0:
            print(f"{n}/{len(frames)} frames", flush=True)
    if vw is not None:
        vw.release()
        import subprocess
        subprocess.run(["ffmpeg", "-y", "-i",
                        args.video.replace(".mp4", "_raw.mp4"), "-c:v",
                        "libx264", "-crf", "24", "-pix_fmt", "yuv420p",
                        args.video], check=True, capture_output=True)
        os.remove(args.video.replace(".mp4", "_raw.mp4"))
    print(f"done {n} frames -> {os.path.join(args.out, name)}"
          + (f" + {args.video}" if args.video else ""), flush=True)


if __name__ == "__main__":
    main()

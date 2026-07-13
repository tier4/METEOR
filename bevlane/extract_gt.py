#!/usr/bin/env python3
"""Extract per-frame training samples: ego-centric BEV GT crops + resized images.

For each keyframe (strided): rotate/crop the scene BEV raster to an ego-centric
window (+x forward = up), cache resized 6-cam images, and record calibration in
a per-scene manifest.
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from autolabel_bev import quat_to_rot  # noqa: E402

cv2.setNumThreads(1)

ROOT = os.environ.get("BEVLANE_ROOT",
                      "/data1/dataset/aisin/converted_valid_delay")
PROD = "out/production"
OUT = "out/bevlane"
CAMS = ["CAM_FRONT_WIDE", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
        "CAM_BACK_WIDE", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
BEV_RES = 0.2
BEV_XH, BEV_YH = 80.0, 50.0        # +-80 m fwd, +-50 m lateral
BEV_H, BEV_W = int(2 * BEV_XH / BEV_RES), int(2 * BEV_YH / BEV_RES)  # 800x500
IMG_W, IMG_H = 768, 432


def load_scene_light(scene_dir):
    ann = os.path.join(scene_dir, "annotation")
    load = lambda n: json.load(open(os.path.join(ann, n + ".json")))
    samples = load("sample")
    by_tok = {s["token"]: s for s in samples}
    first = [s for s in samples if not s["prev"]][0]
    ordered, cur = [], first
    while True:
        ordered.append(cur)
        if not cur["next"]:
            break
        cur = by_tok[cur["next"]]
    frames = {}
    for d in load("sample_data"):
        if d["is_key_frame"]:
            frames.setdefault(d["sample_token"], {})[d["filename"].split("/")[1]] = d
    return (ordered, frames, {c["token"]: c for c in load("calibrated_sensor")},
            {e["token"]: e for e in load("ego_pose")})


def process_scene(args):
    scene, stride, force = args
    try:
        prod = os.path.join(PROD, scene)
        sdir = os.path.join(ROOT, scene)
        meta = json.load(open(os.path.join(prod, "meta.json")))
        bev = np.load(os.path.join(prod, "bev_label_masked.npy"))
        res0 = meta["resolution"]
        x0, y0 = meta["origin"]
        ordered, frames, calib, egop = load_scene_light(sdir)

        out_dir = os.path.join(OUT, scene)
        os.makedirs(os.path.join(out_dir, "gt"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "img"), exist_ok=True)
        manifest = []

        cam_cache = {}
        for fi, s in enumerate(ordered[::stride]):
            frame = frames[s["token"]]
            ld = frame.get("LIDAR_CONCAT")
            if ld is None:
                continue
            ep = egop[ld["ego_pose_token"]]
            tx, ty = ep["translation"][:2]
            R = quat_to_rot(ep["rotation"])
            yaw = np.arctan2(R[1, 0], R[0, 0])
            c, sn = np.cos(yaw), np.sin(yaw)
            # crop px (cx,cy) -> ego (x=XH-cy*res, y=YH-cx*res) -> world -> scene px
            A = np.array([[sn * BEV_RES, -c * BEV_RES],
                          [-c * BEV_RES, -sn * BEV_RES]])  # d(world)/d(cx,cy)
            b = np.array([tx + c * BEV_XH - sn * BEV_YH,
                          ty + sn * BEV_XH + c * BEV_YH])
            M = np.zeros((2, 3))
            M[:, :2] = A / res0
            M[:, 2] = (b - [x0, y0]) / res0
            gt = cv2.warpAffine(bev, M, (BEV_W, BEV_H),
                                flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                                borderValue=0)
            if (gt > 0).mean() < 0.03:   # nearly empty GT: skip
                continue
            gt_name = f"gt/{fi:04d}.png"
            cv2.imwrite(os.path.join(out_dir, gt_name), gt)

            cams = {}
            skip = False
            for ch in CAMS:
                sd = frame.get(ch)
                if sd is None:
                    skip = True
                    break
                img_name = f"img/{fi:04d}_{ch}.jpg"
                if force or not os.path.exists(os.path.join(out_dir, img_name)):
                    img = cv2.imread(os.path.join(sdir, sd["filename"]))
                    if img is None:
                        skip = True
                        break
                    h0, w0 = img.shape[:2]
                    cv2.imwrite(os.path.join(out_dir, img_name),
                                cv2.resize(img, (IMG_W, IMG_H),
                                           interpolation=cv2.INTER_AREA),
                                [cv2.IMWRITE_JPEG_QUALITY, 92])
                else:
                    h0 = sd.get("height", 1860)
                    w0 = sd.get("width", 2880)
                if ch not in cam_cache:
                    cal = calib[sd["calibrated_sensor_token"]]
                    K = np.array(cal["camera_intrinsic"], dtype=np.float64)
                    K[0] *= IMG_W / w0
                    K[1] *= IMG_H / h0
                    T = np.eye(4)
                    T[:3, :3] = quat_to_rot(cal["rotation"])
                    T[:3, 3] = cal["translation"]
                    cam_cache[ch] = {"K": K.tolist(), "T_ego_cam": T.tolist()}
                cams[ch] = img_name
            if skip:
                continue
            manifest.append({"frame": fi, "gt": gt_name, "imgs": cams})

        json.dump({"scene": scene, "bev_res": BEV_RES, "bev_size": BEV_H,
                   "bev_h": BEV_H, "bev_w": BEV_W,
                   "bev_xh": BEV_XH, "bev_yh": BEV_YH,
                   "img_hw": [IMG_H, IMG_W], "cams": cam_cache,
                   "frames": manifest},
                  open(os.path.join(out_dir, "manifest.json"), "w"))
        return f"[ok] {scene} {len(manifest)} frames"
    except Exception as e:
        return f"[fail] {scene}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--scenes", default=None, help="file or comma list; default all")
    ap.add_argument("--force", action="store_true",
                    help="re-extract even if manifest exists")
    args = ap.parse_args()

    if args.scenes and os.path.isfile(args.scenes):
        scenes = [l.strip() for l in open(args.scenes) if l.strip()]
    elif args.scenes:
        scenes = args.scenes.split(",")
    else:
        scenes = sorted(d for d in os.listdir(PROD)
                        if os.path.exists(os.path.join(PROD, d, "bev_label_masked.npy")))
    if not args.force:
        scenes = [s for s in scenes
                  if not os.path.exists(os.path.join(OUT, s, "manifest.json"))]
    print(f"{len(scenes)} scenes to extract", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(process_scene,
                                     [(s, args.stride, args.force) for s in scenes])):
            print(f"{i + 1}/{len(scenes)} {r}", flush=True)


if __name__ == "__main__":
    main()

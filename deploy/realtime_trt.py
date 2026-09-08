#!/usr/bin/env python3
"""Pipelined real-time TensorRT demo: 8-camera RGB + BEV, nothing else.

The existing renderer (bevlane/demo_rgbd_bev.py) builds a 1920x1080 sheet with
depth, occupancy and a pseudo-LiDAR point cloud, and runs everything in one
thread. That is right for inspecting a model and wrong for showing real-time
behaviour on an Orin or a laptop: it spends far longer drawing than the network
spends thinking.

Measured on this machine, per 8-camera frame:

    file read                 0.10 ms
    JPEG decode (cv2)        21.61 ms
    resize                    1.54 ms
    normalise + transpose    24.82 ms    <- CPU float32, then a 32 MB upload
    TRT inference            19.24 ms

Two changes come out of that. First, the normalise moves to the GPU: upload the
uint8 (8 MB instead of 32 MB) and do the divide there. 24.82 -> 0.34 ms, a 73x
cut, because the CPU was touching 8 M pixels serially and quadrupling the
transfer for nothing. (nvJPEG was tried too and is NOT used: torchvision 0.16's
decode_jpeg takes one image per call and re-initialises each time, measured at
4273 ms for eight. It needs 0.17+ for the batched API. On a vehicle the question
is moot -- the ISP hands over YUV in GPU memory and no JPEG exists.)

Second, the stages run concurrently instead of in series, so throughput is set
by the slowest stage rather than their sum: decode threads -> one GPU thread ->
render threads, with bounded queues between them.

    CUDA_VISIBLE_DEVICES=0 python3 deploy/realtime_trt.py \
        --engine out/trt_v52/meteor_v48_int8_partial.engine \
        --root out/bevlane --scenes-file val.lst --out out/realtime.mp4
"""
import argparse
import json
import os
import queue
import sys
import threading
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from autolabel_bev import PALETTE                               # noqa: E402
from bevlane.dataset import CAMS                                # noqa: E402
from bevlane.demo_occ_gt import cube_render_fast as cube_render  # noqa
# The offline demo already solves ego icon + grid, the ground-projected path
# ribbon and 3D-box projection. Importing them keeps the two renderers
# identical instead of approximately similar -- the first cut of this file drew
# its own BEV and got the ego, the heading arrows and the path all wrong.
from bevlane.demo_rgbd_bev import (DEMO_PALETTE, draw_boxes2d,  # noqa
                                   draw_boxes_on_rgb, draw_path_ribbon)
from bevlane.model import DET_RES, EGO_K, MODELS               # noqa: E402
from bevlane.postproc import crop_bev, draw_ego_and_grid        # noqa: E402

VW, VH = 1920, 1080
TILE_W, TILE_H = 320, 180                  # 8 camera tiles, 4x2
BEV_X0 = 4 * TILE_W                        # BEV occupies the right 640 px
BEV_KEEP = (0, 500, 100, 400)              # r0, r1, c0, c1 of the 800x500 grid
PED_TH = 0.70
ZOOM = (20.0, 35.0, 60.0)      # +-m fwd: stopped / town / cruise
_ZOOM_STATE = [60.0, 99.0]
DEP_Y0 = 2 * TILE_H + 130                  # depth block under the RGB block
OCC_Y0 = DEP_Y0 + 2 * TILE_H + 30          # OCC under that
CLS_NAME = {1: "road", 2: "sidewalk", 3: "crosswalk", 4: "laneline",
            5: "stopline", 6: "road_edge", 7: "marking", 8: "parking"}


def palette_lut(n=32):
    lut = np.zeros((n, 3), np.uint8)
    for c in range(n):
        col = PALETTE.get(c) if isinstance(PALETTE, dict) else (
            PALETTE[c] if c < len(PALETTE) else None)
        if col is not None:
            lut[c] = np.array(col[::-1], np.uint8)      # PALETTE is RGB
    return lut


class Engine:
    """One execution context, buffers held as torch tensors on the GPU."""

    def __init__(self, path):
        import tensorrt as trt
        self.trt = trt
        rt = trt.Runtime(trt.Logger(trt.Logger.ERROR))
        rt.engine_host_code_allowed = True
        self.eng = rt.deserialize_cuda_engine(open(path, "rb").read())
        self.ctx = self.eng.create_execution_context()
        self.buf = {}
        for i in range(self.eng.num_io_tensors):
            n = self.eng.get_tensor_name(i)
            dt = {"DataType.FLOAT": torch.float32, "DataType.HALF": torch.float16,
                  "DataType.INT32": torch.int32, "DataType.INT8": torch.int8,
                  "DataType.BOOL": torch.bool}[str(self.eng.get_tensor_dtype(n))]
            t = torch.zeros(*tuple(self.eng.get_tensor_shape(n)), dtype=dt,
                            device="cuda")
            self.buf[n] = t
            self.ctx.set_tensor_address(n, int(t.data_ptr()))
        self.stream = torch.cuda.Stream()
        self.img_shape = tuple(self.buf["imgs"].shape)      # 1,8,3,H,W

    def run(self, imgs_u8, K, T, v0):
        """imgs_u8: pinned uint8 [8,H,W,3] BGR. Normalise on the GPU."""
        with torch.cuda.stream(self.stream):
            x = imgs_u8.to("cuda", non_blocking=True)
            x = x[..., [2, 1, 0]]                       # BGR -> RGB
            x = x.permute(0, 3, 1, 2).float().div_(255.0)
            self.buf["imgs"].copy_(x.unsqueeze(0).to(self.buf["imgs"].dtype))
            for nm, src in (("K", K), ("T_cam_ego", T), ("v0", v0)):
                if nm in self.buf and src is not None:
                    self.buf[nm].copy_(src.to(self.buf[nm].dtype))
            self.ctx.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        return self.buf


def load_frame(root, scene, fr, hw):
    """-> (uint8 [8,H,W,3] BGR, K [8,3,3], T [8,4,4]) or None."""
    H, W = hw
    out = np.zeros((len(CAMS), H, W, 3), np.uint8)
    for i, c in enumerate(CAMS):
        p = fr["imgs"].get(c)
        if not p:
            continue
        im = cv2.imread(os.path.join(root, scene, p), cv2.IMREAD_COLOR)
        if im is None:
            continue
        if im.shape[:2] != (H, W):
            im = cv2.resize(im, (W, H))
        out[i] = im
    return out


def render(rgb, seg, dets, ego_wp, stats, dep, occ, K, T, b2d=None,
           risk=None, unk=None, v0=0.0):
    """Same panels and the same drawing code as the offline demo.

    Everything geometric is delegated to the helpers the offline renderer uses
    (draw_ego_and_grid, draw_path_ribbon, draw_boxes_on_rgb, cube_render) so the
    two agree by construction rather than by eye.
    """
    canvas = np.zeros((VH, VW, 3), np.uint8)
    det_boxes = [(c, sc, x, y, l, w, yw) for (c, sc, x, y, l, w, yw, _, _)
                 in dets]
    for i in range(min(8, rgb.shape[0])):
        tile = rgb[i].copy()
        # 3D boxes projected into the camera, and the planned path as a ground
        # ribbon on the forward wide camera only (it is the one that shows it).
        try:
            draw_boxes_on_rgb(tile, det_boxes, K[i], T[i], tile.shape[1],
                              tile.shape[0], W0=tile.shape[1],
                              H0=tile.shape[0])
            if b2d is not None and i < len(b2d):
                draw_boxes2d(tile, b2d[i], tile.shape[1], tile.shape[0])
            if CAMS[i] == "CAM_FRONT_WIDE" and ego_wp is not None:
                draw_path_ribbon(tile, ego_wp.reshape(-1), K[i], T[i],
                                 tile.shape[1], tile.shape[0],
                                 W0=tile.shape[1], H0=tile.shape[0])
        except Exception:
            pass
        t = cv2.resize(tile, (TILE_W, TILE_H))
        r, c = divmod(i, 4)
        canvas[r * TILE_H:(r + 1) * TILE_H, c * TILE_W:(c + 1) * TILE_W] = t
        cv2.putText(canvas, CAMS[i].replace("CAM_", ""),
                    (c * TILE_W + 6, r * TILE_H + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)

    # ---- BEV: identical crop, palette, ego icon and grid to the demo -------
    pc = crop_bev(seg, xh_m=60.0, yh_m=25.0)
    xh, yh = 60.0, 25.0
    BH2 = VH - 90
    BW2 = min(int(BH2 * pc.shape[1] / pc.shape[0]), VW - BEV_X0 - 8)
    bev = draw_ego_and_grid(DEMO_PALETTE[pc][:, :, ::-1], BH2, BW2,
                            xh_m=xh, yh_m=yh)
    if risk is not None:
        # The risk head covers +-40 m longitudinally; the panel now covers
        # +-xh, which the speed picks. Place it, never stretch it, and gate at
        # 0.3 -- the head's median output is 0.153, so blending everything
        # painted a 10 % wash over the whole BEV and made it muddy. 21.6 % of
        # cells clear 0.3, 6.2 % clear 0.5; that is the part worth showing.
        rh = min(40.0, xh)
        hpx = max(1, int(round(BH2 * rh / xh)))
        y0r = int(round(BH2 * (xh - rh) / (2 * xh)))
        y0r = max(0, min(y0r, BH2 - hpx))
        rm = cv2.resize(risk, (BW2, hpx), interpolation=cv2.INTER_LINEAR)
        rm = np.where(rm > 0.3, rm, 0.0)
        sub = bev[y0r:y0r + hpx]
        heat = cv2.applyColorMap((np.clip(rm, 0, 1) * 255).astype(np.uint8),
                                 cv2.COLORMAP_TURBO)
        al = (np.clip(rm, 0, 1) * 0.65)[..., None]
        bev[y0r:y0r + hpx] = (sub * (1 - al) + heat * al).astype(np.uint8)
    sy2, sx2 = BH2 / (2 * xh), BW2 / (2 * yh)
    for (cls, sc, xe, ye, l, w, yaw, stat, off) in dets:
        if abs(xe) > xh or abs(ye) > yh:
            continue
        cb, sb = np.cos(yaw), np.sin(yaw)
        cor = []
        for lx, wy in ((l / 2, w / 2), (l / 2, -w / 2),
                       (-l / 2, -w / 2), (-l / 2, w / 2)):
            px = xe + lx * cb - wy * sb
            py = ye + lx * sb + wy * cb
            cor.append([int((yh - py) * sx2), int((xh - px) * sy2)])
        col = (128, 128, 128) if stat else (
            (0, 215, 255) if cls == 0 else (255, 0, 255))
        cv2.polylines(bev, [np.array(cor, np.int32)], True, col, 2)
        # heading: a spur from the box centre out through its nose, so a
        # stopped car and a car pointing away are not the same picture
        nx, ny = xe + (l / 2 + 1.2) * cb, ye + (l / 2 + 1.2) * sb
        cv2.arrowedLine(bev,
                        (int((yh - ye) * sx2), int((xh - xe) * sy2)),
                        (int((yh - ny) * sx2), int((xh - nx) * sy2)),
                        col, 2, tipLength=0.35)
        if off is not None and not stat:
            pts = [(int((yh - (ye + o[1])) * sx2),
                    int((xh - (xe + o[0])) * sy2)) for o in off]
            for p, q in zip([(int((yh - ye) * sx2),
                              int((xh - xe) * sy2))] + pts[:-1], pts):
                cv2.line(bev, p, q, (0, 165, 255), 2)
    for (ux, uy) in (unk or []):
        if abs(ux) > xh or abs(uy) > yh:
            continue
        p = (int((yh - uy) * sx2), int((xh - ux) * sy2))
        cv2.drawMarker(bev, p, (255, 255, 255), cv2.MARKER_DIAMOND, 9, 2)
    if ego_wp is not None:
        # Exactly the offline demo's version: a 2 px green polyline from the
        # ego origin through the six waypoints with 3 px dots, and an explicit
        # HOLD ring when the plan collapses onto the ego (under 1 m of travel),
        # which is what a stopped vehicle looks like. No arrow head, no
        # distance readout, no zoom -- the earlier embellishments were mine,
        # added while chasing a path that was missing for a different reason
        # (v0 was being fed as zero, which shortened it by 89 %).
        pts = [(int(yh * sx2), int(xh * sy2))]
        for (xe, ye) in ego_wp:
            if abs(xe) > xh or abs(ye) > yh:
                break
            pts.append((int((yh - ye) * sx2), int((xh - xe) * sy2)))
        trav = float(np.hypot(ego_wp[-1, 0], ego_wp[-1, 1]))
        if trav < 1.0:
            cv2.circle(bev, pts[0], 9, (0, 255, 0), 2)
            cv2.putText(bev, "HOLD", (pts[0][0] + 12, pts[0][1] + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                        cv2.LINE_AA)
        else:
            cv2.polylines(bev, [np.array(pts, np.int32).reshape(-1, 1, 2)],
                          False, (0, 255, 0), 2)
            for p in pts[1:]:
                cv2.circle(bev, p, 3, (0, 255, 0), -1)
    canvas[:bev.shape[0], BEV_X0:BEV_X0 + bev.shape[1]] = bev

    if dep is not None:
        cv2.putText(canvas, "predicted depth (0-80 m)", (8, DEP_Y0 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        for i in range(min(8, dep.shape[0])):
            d = cv2.applyColorMap(dep[i], cv2.COLORMAP_TURBO)
            d = cv2.resize(d, (TILE_W, TILE_H), interpolation=cv2.INTER_NEAREST)
            r, c = divmod(i, 4)
            canvas[DEP_Y0 + r * TILE_H:DEP_Y0 + (r + 1) * TILE_H,
                   c * TILE_W:(c + 1) * TILE_W] = d
    if occ is not None:
        cv2.putText(canvas, "pred OCC voxel grid +-24m (bldg hidden)",
                    (8, OCC_Y0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (220, 220, 220), 1)
        h = min(300, VH - OCC_Y0)
        canvas[OCC_Y0:OCC_Y0 + h, 8:8 + 426] = occ[:h]
    y = 2 * TILE_H + 40
    for k, v in stats.items():
        cv2.putText(canvas, f"{k}: {v}", (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (120, 255, 160), 1)
        y += 26
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--root", default="out/bevlane")
    ap.add_argument("--scenes-file", default="val.lst")
    ap.add_argument("--scenes", type=int, default=6)
    ap.add_argument("--out", default="out/realtime.mp4")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--loaders", type=int, default=4)
    ap.add_argument("--renderers", type=int, default=3)
    ap.add_argument("--model", default="v52", help="decode helpers only")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--thin-bias", type=float, default=0.5)
    ap.add_argument("--thresh2d", type=float, default=0.50,
                    help="2D box score gate. The offline demo defaults to "
                         "0.25, which is 111 boxes a frame across 8 cameras; "
                         "this renderer packs them into 320x180 tiles where "
                         "that reads as noise. 0.50 leaves 24. NOT a "
                         "quantisation issue -- PyTorch emits MORE than the "
                         "INT8 engine at every threshold (111 vs 99 at 0.25).")
    ap.add_argument("--unknown", action="store_true",
                    help="draw unknown-obstacle markers in BEV. "
                         "Off, to match the offline demo, which "
                         "only decodes them for model v30")
    ap.add_argument("--ped-th", type=float, default=0.70,
                    help="separate gate for the pedestrian class")
    ap.add_argument("--risk", action="store_true", default=True,
                    help="overlay the predicted risk map on the BEV")
    ap.add_argument("--no-depth", dest="show_depth",
                    action="store_false")
    ap.add_argument("--no-occ", dest="show_occ",
                    action="store_false")
    ap.add_argument("--occ-every", type=int, default=10,
                    help="redraw the isometric OCC view every N "
                         "frames; it costs 0.3-2.0 s of pure "
                         "Python each time")
    a = ap.parse_args()

    global PED_TH
    PED_TH = a.ped_th
    eng = Engine(a.engine)
    _, N, _, H, W = eng.img_shape
    net = MODELS[a.model](n_seg=21)                 # decode_boxes only, on CPU
    lut = palette_lut()
    scenes = [l.strip() for l in open(a.scenes_file) if l.strip()][:a.scenes]

    work = []
    for s in scenes:
        mf = os.path.join(a.root, s, "manifest.json")
        if not os.path.exists(mf):
            continue
        m = json.load(open(mf))
        K = np.zeros((N, 3, 3), np.float32)
        T = np.zeros((N, 4, 4), np.float32)
        for i, c in enumerate(CAMS):
            cam = m["cams"].get(c)
            if cam is None:
                T[i] = np.eye(4)
                continue
            K[i] = np.array(cam["K"], np.float32)
            T[i] = np.linalg.inv(np.array(cam["T_ego_cam"], np.float32))
        # v0 is a REQUIRED input of the E2E head, not an optional extra.
        # Passing zeros (as this renderer did) shortens the predicted path by
        # 89 % -- median 9.19 m becomes 1.02 m, which on the BEV panel is 8 px
        # and reads as "no path" -- and multiplies ADE by 5.1.
        v0s = None
        ep = os.path.join(a.root, s, "ego_motion.npz")
        if os.path.exists(ep):
            try:
                v0s = np.load(ep)["v0"]
            except Exception:
                v0s = None
        for fr in m["frames"]:
            fi = int(fr.get("frame", 0))
            v0 = float(v0s[fi]) if v0s is not None and fi < len(v0s) else 0.0
            work.append((s, fr, K, T, v0))
    if a.max_frames:
        work = work[:a.max_frames]
    print(f"[data] {len(scenes)} scenes, {len(work)} frames, "
          f"engine input {eng.img_shape}", flush=True)

    q_load = queue.Queue(maxsize=8)
    q_gpu = queue.Queue(maxsize=8)
    q_draw = queue.Queue(maxsize=8)
    tim = {"load": [], "gpu": [], "draw": []}
    lock = threading.Lock()

    def loader(idx):
        for j in range(idx, len(work), a.loaders):
            s, fr, K, T, v0 = work[j]
            t0 = time.time()
            img = load_frame(a.root, s, fr, (H, W))
            with lock:
                tim["load"].append((time.time() - t0) * 1000)
            q_load.put((j, img, K, T, v0))
        q_load.put(None)

    occ_state = {"n": 0, "img": None}

    def gpu_worker():
        # A thread that dies takes the whole pipeline down with it: the
        # renderers block on q_gpu and main blocks on q_draw, and nothing
        # prints -- which is exactly what one missing import (DET_RES) did
        # here, 500 s of silence. Every exception now surfaces and releases
        # the queue instead of hanging.
        done = 0
        pin = torch.empty(N, H, W, 3, dtype=torch.uint8).pin_memory()
        while True:
            it = q_load.get()
            if it is None:
                done += 1
                if done == a.loaders:
                    q_gpu.put(None)
                    return
                continue
            j, img, K, T, v0 = it
            t0 = time.time()
            try:
                pin.copy_(torch.from_numpy(img))
                b = eng.run(pin, torch.from_numpy(K).unsqueeze(0),
                            torch.from_numpy(T).unsqueeze(0),
                            torch.tensor([v0], dtype=torch.float32))
                lg = b["lane"].float()
                if a.thin_bias:
                    # The thin classes come out 2-3x too wide (laneline covers
                    # 2.79x the GT area on val). It is a decision-boundary
                    # offset, not a localisation error, so a constant taken off
                    # those logits before the argmax fixes it: 2.79 -> 1.17x
                    # for -0.4 % of laneline IoU.
                    lg = lg.clone()
                    for c in (4, 5, 6):
                        lg[:, c] -= a.thin_bias
                seg_np = lg.argmax(1)[0].to(torch.uint8).cpu().numpy()

                dep_np = None
                if a.show_depth and "depth" in b:
                    # Reduced on the GPU: raw this tensor is 40.5 MB a frame,
                    # as a display image it is 166 KB. The GPU stage is the
                    # pipeline ceiling, so nothing crosses the bus at full size.
                    dl = b["depth"].float()[0]
                    if dl.shape[1] > 2:
                        p = dl.softmax(1)
                        pk = p.argmax(1, keepdim=True)
                        ar = torch.arange(p.shape[1], device=p.device
                                          ).view(1, -1, 1, 1)
                        pw = p * ((ar - pk).abs() <= 2)
                        bins = (torch.arange(p.shape[1], device=p.device)
                                * net.D_STEP + net.D_MIN).view(1, -1, 1, 1)
                        d_m = (pw * bins).sum(1) / pw.sum(1).clamp(min=1e-6)
                    else:
                        d_m = dl[:, 0]
                    dep_np = (d_m.clamp(0, 80) / 80 * 255).to(
                        torch.uint8).cpu().numpy()

                # Boxes are decoded here, not in the render stage, because the
                # agent forecast and the stationary flag have to be sampled at
                # the box centres from two 400x250 maps; shipping those whole
                # (15.6 MB and 0.4 MB a frame) would cost more than the decode.
                dets = []
                bl = net.decode_boxes(b["hm"].float(), b["reg"].float(),
                                      thresh=0.30, topk=32)[0]
                tj = b["traj"].float()[0] if "traj" in b else None
                st = (b["stationary"].float()[0, 0].sigmoid()
                      if "stationary" in b else None)
                for (cls, sc, xe, ye, l, w, yaw) in bl:
                    rr = int(np.clip((80.0 - xe) / DET_RES, 0, 399))
                    cc = int(np.clip((50.0 - ye) / DET_RES, 0, 249))
                    off = None
                    if tj is not None:
                        v = tj[:, rr, cc]
                        if v.numel() >= 39:      # K modes + logits, like ego
                            kb = int(v[36:39].argmax())
                            off = v[kb * 12:(kb + 1) * 12].view(
                                6, 2).cpu().numpy()
                        elif v.numel() >= 12:
                            off = v[:12].view(6, 2).cpu().numpy()
                    stat = bool(st[rr, cc] > 0.5) if st is not None else False
                    dets.append((cls, sc, xe, ye, l, w, yaw, stat, off))

                risk = None
                if a.risk and "risk" in b:
                    risk = b["risk"].float()[0, 0].sigmoid().cpu().numpy()
                b2d = None
                if all(f"hm2d_s{i}" in b for i in range(3)):
                    raw2d = net.decode_boxes2d_ms(
                        [b[f"hm2d_s{i}"].float()[0] for i in range(3)],
                        [b[f"reg2d_s{i}"].float()[0] for i in range(3)],
                        thresh=min(a.thresh2d, PED_TH))
                    # Measured at 0.50 over 6 frames: 63 pedestrians and 60
                    # cars, and the pedestrians are railings, poles and street
                    # furniture along the kerb -- tall thin verticals that the
                    # class likes. Not a quantisation artefact (PyTorch emits
                    # MORE than the engine at every threshold). Two gates:
                    # a higher bar for the class that is wrong most often, and
                    # an aspect-ratio reject, since a real pedestrian is not
                    # 6x taller than wide at this scale.
                    b2d = []
                    for cam in raw2d:
                        keep = []
                        for (c, sc, cx, cy, w, h) in cam:
                            th = PED_TH if int(c) == 6 else a.thresh2d
                            if sc < th:
                                continue
                            ar = h / max(w, 1e-3)
                            if int(c) == 6 and not (1.2 < ar < 5.0):
                                continue
                            keep.append((c, sc, cx, cy, w, h))
                        b2d.append(keep)
                # No unknown markers. The offline demo gates them on
                # `args.model == "v30"`, so for v52 it draws none at all --
                # this renderer was adding 48 a frame that the reference never
                # shows. --unknown puts them back as BEV-only white diamonds
                # (never on the cameras: folding them into the box list is what
                # buried the tiles in magenta).
                unk = []
                if a.unknown and "unk" in b:
                    unk = [(u[2], u[3]) for u in
                           net.decode_unknown(b["unk"].float())[0]]

                # E2E: commit to the mode the selector picks. Reading ego[:12]
                # always takes candidate 0 whatever the logits say, which is
                # why the path in the first version pointed the wrong way.
                e = b["ego"].float()[0].cpu().numpy()
                # argmax of the logits, not of exp(logits): a large logit
                # overflows float32 exp to inf, and two infs make argmax pick
                # whichever came first rather than the larger one.
                kk = int(e[12 * EGO_K:12 * EGO_K + EGO_K].argmax())
                ego = e[kk * 12:(kk + 1) * 12].reshape(6, 2)

                occ_img = None
                if a.show_occ and "occ" in b:
                    # cube_render is the offline demo's isometric voxel view
                    # and costs 385 ms a call in pure Python -- and being
                    # Python it holds the GIL, so it does not just slow its own
                    # thread, it starves the GPU one. Redrawn every N frames
                    # and held in between: the look the user asked to match, at
                    # a refresh rate a Python voxel painter can actually reach.
                    if occ_state["n"] % max(a.occ_every, 1) == 0:
                        cls_ = b["occ"].float()[0].argmax(0).to(
                            torch.uint8).cpu().numpy()
                        occ_state["img"] = cv2.resize(
                            cube_render(cls_, W=640, H=540), (426, 300))
                    occ_state["n"] += 1
                    occ_img = occ_state["img"]

                with lock:
                    tim["gpu"].append((time.time() - t0) * 1000)
                q_gpu.put((j, img, seg_np, dets, ego, dep_np,
                           occ_img, K, T, b2d, risk, unk, v0))
            except Exception:
                import traceback
                traceback.print_exc()
                q_gpu.put(None)
                return

    def renderer(idx):
        while True:
            it = q_gpu.get()
            if it is None:
                q_gpu.put(None)
                q_draw.put(None)
                return
            j, img, seg, dets, ego, dep, occ, K, T, b2d, risk, unk, v0 = it
            t0 = time.time()
            with lock:
                st = {"pipeline FPS": f"{fps_now[0]:.1f}",
                      "infer": f"{np.mean(tim['gpu'][-30:]):.1f} ms"
                              if tim["gpu"] else "-"}
            fr = render(img, seg, dets, ego, st, dep, occ, K, T,
                        b2d, risk, unk, v0)
            with lock:
                tim["draw"].append((time.time() - t0) * 1000)
            q_draw.put((j, fr))

    fps_now = [0.0]
    vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (VW, VH))
    ths = [threading.Thread(target=loader, args=(i,), daemon=True)
           for i in range(a.loaders)]
    ths.append(threading.Thread(target=gpu_worker, daemon=True))
    ths += [threading.Thread(target=renderer, args=(i,), daemon=True)
            for i in range(a.renderers)]
    t_start = time.time()
    for t in ths:
        t.start()
    pend, nxt, n_out, ends = {}, 0, 0, 0
    while ends < a.renderers:
        it = q_draw.get()
        if it is None:
            ends += 1
            continue
        j, fr = it
        pend[j] = fr
        while nxt in pend:                       # keep the output in order
            vw.write(pend.pop(nxt))
            nxt += 1
            n_out += 1
            fps_now[0] = n_out / max(time.time() - t_start, 1e-6)
            if n_out % 50 == 0:
                print(f"  {n_out}/{len(work)} frames  "
                      f"{fps_now[0]:.1f} FPS", flush=True)
    for j in sorted(pend):
        vw.write(pend[j])
        n_out += 1
    vw.release()
    el = time.time() - t_start
    print(f"\n{n_out} frames in {el:.1f} s = {n_out / el:.1f} FPS -> {a.out}")
    print(f"\n{'stage':>10s} {'mean ms':>9s} {'threads':>8s} "
          f"{'FPS if alone':>13s}")
    for k, nth in (("load", a.loaders), ("gpu", 1), ("draw", a.renderers)):
        if tim[k]:
            mu = float(np.mean(tim[k]))
            print(f"{k:>10s} {mu:9.2f} {nth:8d} {1000 * nth / mu:13.1f}")
    print("\nthe slowest row is the ceiling; the pipeline hides the rest")


if __name__ == "__main__":
    main()

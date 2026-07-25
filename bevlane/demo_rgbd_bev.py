#!/usr/bin/env python3
"""GT-less deployment demo: RGB | predicted Depth | predicted BEV (horizontal).

Layout (1920x1080), left -> right:
  RGB   : 6 surround cameras (2x3, large) + 2 tele NARROW below
  Depth : predicted depth for the 6 cameras (turbo, 2x3)
  BEV   : predicted semantic BEV with ego icon + distance grid (portrait),
          road_edge thinned to its innermost 1-px (post-processing).
No ground truth shown.
"""
import argparse
import os
import subprocess
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from autolabel_bev import PALETTE  # noqa: E402
from bevlane.dataset import CAMS, BevLaneDataset  # noqa: E402
from bevlane.extract_bbox2d import DET10_PAL  # noqa: E402
from bevlane.demo_occ_gt import cube_render  # noqa: E402
from bevlane.guardrail import check_path, risk_pick  # noqa: E402
from bevlane.extract_occ import OCC_PAL  # noqa: E402
from bevlane.model import make_warp_theta  # noqa: E402
from bevlane.model import (DepthGatedIPMNet, DepthSegIPMNet,  # noqa: E402
                           DepthSegIPMNetS4, DepthSegIPMNetV14,
                           DepthSegIPMNetV15, DepthSegIPMNetV16,
                           DepthSegIPMNetV17, DepthSegIPMNetV18,
                           DepthSegIPMNetV19, DepthSegIPMNetV20,
                           DepthSegIPMNetV21, DepthSegIPMNetV22,
                           DepthSegIPMNetV23, DepthSegIPMNetV25,
                           DepthSegIPMNetV26, DepthSegIPMNetV27,
                           DepthSegIPMNetV28, DepthSegIPMNetV29,
                           DepthSegIPMNetV30, DepthSegIPMNetV31,
                           DepthSegIPMNetV32, DepthSegIPMNetV33,
                           DepthSegIPMNetV34, DepthSegIPMNetV35,
                           DepthSegIPMNetV36, DepthSegIPMNetV37,
                           DepthSegIPMNetV38, DepthSegIPMNetV39,
                           DepthSegIPMNetV40, DepthSegIPMNetV41,
                           DepthSegIPMNetV42)

DET10_ABBR = ["obs", "car", "trk", "bus", "bcy", "mcy", "ped", "pnt", "tl", "ts"]


def draw_boxes2d(img, blist, cw, ch):
    """Per-camera 10-class 2D boxes (cls,score,cx,cy,w,h in 768x432 px)."""
    sx, sy = cw / 768.0, ch / 432.0
    for cls, sc, cx, cy, w, h in blist:
        c = tuple(int(v) for v in DET10_PAL[int(cls)][::-1])   # RGB -> BGR
        x1, y1 = int((cx - w / 2) * sx), int((cy - h / 2) * sy)
        x2, y2 = int((cx + w / 2) * sx), int((cy + h / 2) * sy)
        cv2.rectangle(img, (x1, y1), (x2, y2), c, 2)
        tag = f"{DET10_ABBR[int(cls)]}{int(sc * 100):d}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        ty = max(y1, th + 3)
        cv2.rectangle(img, (x1, ty - th - 3), (x1 + tw + 2, ty + 1), c, -1)
        cv2.putText(img, tag, (x1 + 1, ty - 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (0, 0, 0), 1, cv2.LINE_AA)
from bevlane.postproc import crop_bev, draw_ego_and_grid, thin_road_edge  # noqa: E402

# surround order for the 2x3 grids (front row / back row)
SURR = ["CAM_FRONT_LEFT", "CAM_FRONT_WIDE", "CAM_FRONT_RIGHT",
        "CAM_BACK_LEFT", "CAM_BACK_WIDE", "CAM_BACK_RIGHT"]
NARROW = ["CAM_FRONT_NARROW", "CAM_BACK_NARROW"]

# demo palette: hide sidewalk(2) & parking(8) — they are noisy, show as black
DEMO_PALETTE = np.zeros((12, 3), np.uint8)
DEMO_PALETTE[:len(PALETTE)] = PALETTE
DEMO_PALETTE[2] = 0
DEMO_PALETTE[8] = 0
DEMO_PALETTE[10] = (255, 215, 0)    # predicted vehicle box (gold)
DEMO_PALETTE[11] = (255, 0, 255)    # predicted VRU box (magenta)

SEG2D_PAL = np.zeros((256, 3), np.uint8)
SEG2D_PAL[:11] = [(90, 90, 90),     # road
                  (255, 255, 255),  # lane/marking
                  (160, 90, 140),   # sidewalk
                  (255, 200, 0),    # crosswalk (cyan-ish BGR)
                  (0, 140, 255),    # vehicle (orange)
                  (0, 0, 230),      # person (red)
                  (60, 90, 140),    # building
                  (60, 160, 60),    # vegetation
                  (200, 130, 50),   # sky (light blue BGR)
                  (0, 230, 230),    # pole/sign (yellow)
                  (70, 70, 70)]     # freespace



BOX_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]


def draw_boxes_on_rgb(img, det_boxes, K, Tce, cw, ch, W0=768, H0=432):
    """Project decoded BEV boxes (ego frame) into a camera and draw 3D
    wireframes with distance labels. Edges are CLIPPED at the camera near
    plane so nearby vehicles on side cameras are still drawn (partial box)."""
    sx, sy = cw / W0, ch / H0
    EPS = 0.25

    def proj(p):
        return (int((K[0, 0] * p[0] / p[2] + K[0, 2]) * sx),
                int((K[1, 1] * p[1] / p[2] + K[1, 2]) * sy))

    for cls, sc, xe, ye, l, w, yaw in det_boxes:
        hgt = 1.6 if cls == 0 else 1.7
        cb, sb = np.cos(yaw), np.sin(yaw)
        bot = []
        for lx, wy in ((l/2, w/2), (l/2, -w/2), (-l/2, -w/2), (-l/2, w/2)):
            bot.append([xe + lx*cb - wy*sb, ye + lx*sb + wy*cb, 0.0])
        cors = np.array(bot + [[b[0], b[1], hgt] for b in bot])
        pc = cors @ Tce[:3, :3].T + Tce[:3, 3]
        z = pc[:, 2]
        if (z > EPS).sum() == 0:          # entirely behind the camera
            continue
        col = (0, 215, 255) if cls == 0 else (255, 0, 255)
        vis_pts = []
        for a, b in BOX_EDGES:
            pa, pb = pc[a].copy(), pc[b].copy()
            za, zb = pa[2], pb[2]
            if za < EPS and zb < EPS:
                continue
            if za < EPS or zb < EPS:      # clip the edge at the near plane
                t = (EPS - za) / (zb - za)
                pclip = pa + t * (pb - pa)
                if za < EPS:
                    pa = pclip
                else:
                    pb = pclip
            A, B = proj(pa), proj(pb)
            if abs(A[0]) > cw * 8 or abs(B[0]) > cw * 8                     or abs(A[1]) > ch * 8 or abs(B[1]) > ch * 8:
                continue
            cv2.line(img, A, B, col, 1, cv2.LINE_AA)
            vis_pts += [A, B]
        if vis_pts:
            us = [p[0] for p in vis_pts]
            vs = [p[1] for p in vis_pts]
            if max(us) >= 0 and min(us) < cw and max(vs) >= 0 and min(vs) < ch:
                dist = float(np.hypot(xe, ye))
                cv2.putText(img, f"{dist:.0f}m",
                            (max(0, min(us)), max(12, min(vs) - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)


def pseudo_intent(pose, fi, lo=5, hi=25, lat_th=3.0):
    """Navigation-style pseudo Driving Command from the recorded route.

    Lateral displacement of the ego pose 1-5 s ahead (rows at 5 Hz),
    expressed in the CURRENT ego frame -> one-hot [straight,left,right].
    Same convention as training (v37 intent: lat>+th=left, <-th=right)
    but with a LONGER horizon, so the command fires before the turn --
    exactly what a navigation system would provide."""
    x0, y0, yaw0 = pose[fi]
    c, s = np.cos(yaw0), np.sin(yaw0)
    best = 0.0
    for d in range(lo, min(hi, len(pose) - 1 - fi) + 1):
        dx, dy = pose[fi + d][0] - x0, pose[fi + d][1] - y0
        ey = -s * dx + c * dy               # lateral (left +) in ego frame
        if abs(ey) > abs(best):
            best = float(ey)
    oh = np.zeros(3, np.float32)
    if best > lat_th:
        oh[1] = 1.0; lab = "LEFT"
    elif best < -lat_th:
        oh[2] = 1.0; lab = "RIGHT"
    else:
        oh[0] = 1.0; lab = "STRAIGHT"
    return oh, lab


_RIBBON_STATE = {"wps": None}


def draw_path_ribbon(img, ego_pred, K, Tce, cw, ch, W0=768, H0=432,
                     half_w=0.9, color=(60, 255, 120), alpha=0.38,
                     reset=False):
    """Project the predicted trajectory as a vehicle-width ground ribbon
    (filled, semi-transparent) into a camera image.

    Near a stop the ribbon FADES OUT with the predicted travel distance
    (no hard pop-off), and waypoints are EMA-smoothed across frames."""
    wps = np.concatenate([[[0.0, 0.0]], ego_pred[:12].reshape(6, 2)], 0)
    if not np.all(np.isfinite(wps)):
        # NaN frame (bad odometry upstream): draw nothing but DO NOT touch
        # the EMA state -- one bad frame must not poison every later frame
        return
    if (reset or _RIBBON_STATE["wps"] is None
            or not np.all(np.isfinite(_RIBBON_STATE["wps"]))):
        _RIBBON_STATE["wps"] = wps
    else:
        _RIBBON_STATE["wps"] = 0.55 * _RIBBON_STATE["wps"] + 0.45 * wps
    wps = _RIBBON_STATE["wps"].copy()
    trav = float(np.hypot(*(wps[-1] - wps[0])))
    # RETRACT instead of fade: the displayed ribbon length shrinks smoothly
    # with the predicted travel; the tip slides down toward the hood and
    # finally disappears under it (constant opacity throughout)
    s = min(max(trav / 3.0, 0.0), 1.0)
    disp_len = max(trav, 8.0 * s * s * (3 - 2 * s))
    if disp_len < 0.6:
        return
    # densify along the polyline
    seg = np.linalg.norm(np.diff(wps, axis=0), axis=1)
    t = np.concatenate([[0], np.cumsum(seg)])
    if t[-1] < disp_len:                 # extend along the last heading
        d = wps[-1] - wps[-3]
        d = d / max(np.linalg.norm(d), 1e-3)
        wps = np.concatenate([wps, [wps[-1] + d * (disp_len - t[-1])]], 0)
        seg = np.linalg.norm(np.diff(wps, axis=0), axis=1)
        t = np.concatenate([[0], np.cumsum(seg)])
    tt = np.linspace(0, min(t[-1], disp_len), 48)
    px = np.interp(tt, t, wps[:, 0])
    py = np.interp(tt, t, wps[:, 1])
    th = np.arctan2(np.gradient(py), np.gradient(px))
    lx, ly = px - half_w * np.sin(th), py + half_w * np.cos(th)
    rx, ry = px + half_w * np.sin(th), py - half_w * np.cos(th)
    sx, sy = cw / W0, ch / H0

    def proj(xs, ys):
        pts = np.stack([xs, ys, np.zeros_like(xs)], 1)
        pc = pts @ Tce[:3, :3].T + Tce[:3, 3]
        ok = pc[:, 2] > 0.3
        u = (K[0, 0] * pc[:, 0] / np.maximum(pc[:, 2], 0.3) + K[0, 2]) * sx
        v = (K[1, 1] * pc[:, 1] / np.maximum(pc[:, 2], 0.3) + K[1, 2]) * sy
        return np.stack([u, v], 1), ok

    L, okl = proj(lx, ly)
    R, okr = proj(rx, ry)
    ok = okl & okr
    if ok.sum() < 3:
        return
    L, R = L[ok], R[ok]
    poly = np.concatenate([L, R[::-1]], 0).astype(np.int32)
    poly[:, 0] = np.clip(poly[:, 0], -cw, 2 * cw)
    poly[:, 1] = np.clip(poly[:, 1], -ch, 2 * ch)
    ov = img.copy()
    cv2.fillPoly(ov, [poly.reshape(-1, 1, 2)], color)
    cv2.polylines(ov, [L.astype(np.int32).reshape(-1, 1, 2)], False, color, 2)
    cv2.polylines(ov, [R[::-1].astype(np.int32).reshape(-1, 1, 2)], False,
                  color, 2)
    img[:] = cv2.addWeighted(img, 1 - alpha, ov, alpha, 0)


def label(img, txt, color=(255, 255, 255)):
    cv2.putText(img, txt, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


_BEVQ = {}
_SEGACC = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out/bevlane_ckpt_v12/best.pt")
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--out", default="out/demo_rgbd_bev.mp4")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--intent", default="none",
                    choices=["none", "auto", "straight", "left", "right"],
                    help="pseudo Driving Command into the v37+ intent input: "
                         "auto = navigation-style, derived from the RECORDED "
                         "route 1-5 s ahead (command fires BEFORE the turn)")
    ap.add_argument("--no-thin", action="store_true")
    ap.add_argument("--seg-fuse", action="store_true",
                    help="ego-warped log-odds fusion of BEV seg over time "
                         "(<=45 m; measured: stability .83->.93, "
                         "crosswalk 20-40m +.06)")
    ap.add_argument("--refiner-ckpt", default=None,
                    help="apply a trained BEVSegRefiner to the BEV-seg logits "
                         "(far-range completion; measured road 40-80m +.14)")
    ap.add_argument("--model", default="v8", choices=["v8", "v13", "v13d", "v14d", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36", "v37", "v38", "v39", "v40", "v41", "v42"],
                    help="v8=DepthGatedIPMNet(512) / v13=DepthSegIPMNet(768) / v13d=stride-4 depth")
    ap.add_argument("--show-seg2d", action="store_true",
                    help="alpha-blend predicted 2D seg over RGB panels")
    ap.add_argument("--thresh2d", type=float, default=0.25,
                    help="2D bbox score threshold (v17)")
    ap.add_argument("--n-seg2d", type=int, default=12,
                    help="2D seg head classes (21 for seg2d21-trained ckpts; "
                         "uses the csv Cityscapes-like palette)")
    ap.add_argument("--guard", action="store_true",
                    help="L1 safety guardrails: spacetime collision / red "
                    "light / feasibility / drivable checks on the E2E path")
    ap.add_argument("--lidar", action="store_true",
                    help="v31: feed the per-frame LiDAR sparse depth "
                    "(depth_gt4) as the optional input; omit = camera-only")
    ap.add_argument("--infer-hw", default=None,
                    help="resize images to HxW for the model. default: 288x512 for "
                         "v8, none (native 768) for v13. 'none' = use cache res")
    args = ap.parse_args()

    mcls = {"v13": DepthSegIPMNet, "v13d": DepthSegIPMNetS4,
            "v14d": DepthSegIPMNetV14, "v15": DepthSegIPMNetV15,
            "v16": DepthSegIPMNetV16,
            "v17": DepthSegIPMNetV17,
            "v18": DepthSegIPMNetV18,
            "v19": DepthSegIPMNetV19,
            "v20": DepthSegIPMNetV20,
            "v21": DepthSegIPMNetV21,
            "v22": DepthSegIPMNetV22,
            "v23": DepthSegIPMNetV23,
            "v25": DepthSegIPMNetV25,
            "v26": DepthSegIPMNetV26,
            "v27": DepthSegIPMNetV27,
            "v28": DepthSegIPMNetV28,
            "v29": DepthSegIPMNetV29,
            "v30": DepthSegIPMNetV30,
            "v31": DepthSegIPMNetV31,
            "v32": DepthSegIPMNetV32,
            "v33": DepthSegIPMNetV33,
            "v34": DepthSegIPMNetV34,
            "v35": DepthSegIPMNetV35,
            "v36": DepthSegIPMNetV36,
            "v37": DepthSegIPMNetV37,
            "v38": DepthSegIPMNetV38,
            "v39": DepthSegIPMNetV39,
            "v40": DepthSegIPMNetV40, "v41": DepthSegIPMNetV41, "v42": DepthSegIPMNetV42}.get(args.model, DepthGatedIPMNet)
    mkw = {"n_seg": args.n_seg2d} if args.model in (
        "v13", "v13d", "v14d", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36", "v37", "v38", "v39", "v40", "v41", "v42") else {}
    m = mcls(**mkw).cuda().eval()
    if args.n_seg2d == 21:      # csv taxonomy: Cityscapes-like colours (BGR)
        from bevlane.extract_seg2d import SEG21_PAL
        SEG2D_PAL[:] = 0
        SEG2D_PAL[:21] = SEG21_PAL[:, ::-1]
    m.load_state_dict(torch.load(args.ckpt, map_location="cpu")["model"])
    refiner = None
    if args.refiner_ckpt:
        from bevlane.model import (BEVSegRefiner, MultiTaskRefiner,  # noqa
                                   N_CLASSES, EGO_K)
        ck = torch.load(args.refiner_ckpt, map_location="cpu")
        ra = ck.get("args", {})
        sd = ck["refiner"]
        heads = set(k.split(".")[0] for k in sd)
        if heads & {"seg", "box", "e2e"}:        # multi-task ckpt
            refiner = MultiTaskRefiner(
                do_seg="seg" in heads, do_box="box" in heads,
                do_e2e="e2e" in heads, n_cls=N_CLASSES,
                seg_width=ra.get("width", 48), seg_ctx=ra.get("ctx", 0),
                ego_dim=12 * EGO_K + EGO_K + 3).cuda().eval()
            refiner.load_state_dict(sd)
            refiner._multi = True
        else:                                    # legacy single seg head
            refiner = BEVSegRefiner(N_CLASSES, ctx_ch=ra.get("ctx", 0),
                                    width=ra.get("width", 48)).cuda().eval()
            refiner.load_state_dict(sd)
            refiner._multi = False
        refiner._ctx = ra.get("ctx", 0)
        print(f"[refiner] loaded {args.refiner_ckpt} multi={refiner._multi} "
              f"heads={sorted(heads)} epoch={ck.get('epoch')}", flush=True)
    dbins = torch.arange(m.D) * m.D_STEP + m.D_MIN
    infer_hw = args.infer_hw or ("none" if args.model in ("v13", "v13d", "v14d", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36", "v37", "v38", "v39", "v40", "v41", "v42") else "288x512")
    ih, iw = (None, None) if infer_hw == "none" else \
        (int(infer_hw.split("x")[0]), int(infer_hw.split("x")[1]))

    VW, VH = 1920, 1080
    # left = RGB (top) over Depth (below), same 2x4 grid; right = BEV portrait.
    CAM8 = ["CAM_FRONT_LEFT", "CAM_FRONT_WIDE", "CAM_FRONT_RIGHT", "CAM_FRONT_NARROW",
            "CAM_BACK_LEFT", "CAM_BACK_WIDE", "CAM_BACK_RIGHT", "CAM_BACK_NARROW"]
    cw = 375
    ch = cw * 288 // 512                    # 210 per cell (larger)
    rgb_y0 = 40
    dep_y0 = rgb_y0 + 2 * ch + 45           # depth block starts below RGB block
    bx0 = 4 * cw + 8                         # BEV flush to the right edge
    raw = args.out.replace(".mp4", "_raw.mp4")
    vw = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (VW, VH))
    n = 0
    for scene in args.scenes:
        if not os.path.exists(f"out/bevlane/{scene}/manifest.json"):
            continue
        ds = BevLaneDataset("out/bevlane", [scene], gt_key="gt_vec", with_depth=False)
        for i in range(len(ds)):
            imgs, K, T, _ = ds[i]
            # run the model at its training resolution (avoid OOD instability
            # from feeding a 512-trained model the 768 cache).
            if ih is not None and imgs.shape[-2:] != (ih, iw):
                H0, W0 = imgs.shape[-2:]
                imgs_m = F.interpolate(imgs, size=(ih, iw), mode="bilinear",
                                       align_corners=False)
                K = K.clone()
                K[:, 0, :] *= iw / W0
                K[:, 1, :] *= ih / H0
            else:
                imgs_m = imgs
            s_pre, f_pre = ds.items[i]
            v0_t = None
            if args.model in ("v18", "v19", "v20", "v21", "v22", "v23", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36", "v37", "v38", "v39", "v40", "v41", "v42"):  # v0
                try:
                    z = np.load(os.path.join("out/bevlane", s_pre,
                                             "ego_motion.npz"))
                    v0_t = torch.tensor([float(z["v0"][f_pre["frame"]])])
                except Exception:
                    v0_t = torch.zeros(1)
            pb = th = None
            if args.model in ("v22", "v23", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36", "v37", "v38", "v39", "v40", "v41", "v42"):
                fi_cur = f_pre["frame"]
                if _BEVQ.get("scene") != s_pre:
                    _BEVQ.clear(); _BEVQ["scene"] = s_pre
                try:
                    zp = np.load(os.path.join("out/bevlane", s_pre,
                                              "ego_motion.npz"))["pose"]
                    pc_ = zp[fi_cur]

                    def _rel(fj):
                        pp_ = zp[fj]
                        if abs(pc_).sum() == 0 or abs(pp_).sum() == 0:
                            return None
                        dy = float(pc_[2] - pp_[2])
                        cp, sp = np.cos(pp_[2]), np.sin(pp_[2])
                        return torch.tensor([[cp * (pc_[0] - pp_[0])
                                              + sp * (pc_[1] - pp_[1]),
                                              -sp * (pc_[0] - pp_[0])
                                              + cp * (pc_[1] - pp_[1]), dy]],
                                            dtype=torch.float32).cuda()
                    if args.model in ("v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36", "v37", "v38", "v39", "v40", "v41", "v42"):  # 3-slot queue
                        pbs, ths = [], []
                        for off in (2, 6, 14):
                            hb = _BEVQ.get(fi_cur - off)
                            rl = _rel(fi_cur - off) if fi_cur - off >= 0 \
                                else None
                            if hb is None or rl is None:
                                pbs.append(torch.zeros(
                                    1, 96, 800, 500, device="cuda"))
                                ths.append(make_warp_theta(torch.zeros(
                                    1, 3, device="cuda")))
                            else:
                                pbs.append(hb)
                                ths.append(make_warp_theta(rl))
                        pb = torch.stack(pbs, 1)
                        th = torch.stack(ths, 1)
                    else:
                        prev = _BEVQ.get(fi_cur - 2)
                        rl = _rel(fi_cur - 2) if fi_cur >= 2 else None
                        if prev is not None and rl is not None:
                            pb, th = prev, make_warp_theta(rl)
                except Exception:
                    pass
            lid_kw = {}
            if args.lidar and args.model == "v31" and f_pre.get("depth4"):
                try:
                    dz = np.load(os.path.join("out/bevlane", s_pre,
                                              f_pre["depth4"])
                                 )["depth"].astype(np.float32)
                    lt = torch.zeros(1, len(CAMS), dz.shape[1], dz.shape[2])
                    lt[0, :dz.shape[0]] = torch.from_numpy(dz)
                    lid_kw = {"lidar": lt.cuda()}
                except Exception:
                    pass
            # pseudo Driving Command (v37+ intent input)
            intent_lab = None
            if args.intent != "none" and args.model in (
                    "v37", "v38", "v39", "v40", "v41", "v42"):
                if args.intent == "auto":
                    try:
                        pz = np.load(os.path.join("out/bevlane", s_pre,
                                                  "ego_motion.npz"))["pose"]
                        oh, intent_lab = pseudo_intent(pz, f_pre["frame"])
                    except Exception:
                        oh, intent_lab = np.zeros(3, np.float32), None
                else:
                    idx = {"straight": 0, "left": 1, "right": 2}[args.intent]
                    oh = np.zeros(3, np.float32); oh[idx] = 1.0
                    intent_lab = args.intent.upper()
                lid_kw["intent"] = torch.from_numpy(oh)[None].cuda()
            with torch.no_grad(), torch.autocast("cuda", torch.float16):
                if args.model in ("v22", "v23", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36", "v37", "v38", "v39", "v40", "v41", "v42"):
                    out = m(imgs_m[None].cuda(), K[None].cuda(),
                            T[None].cuda(),
                            v0_t.cuda() if v0_t is not None else None, pb, th,
                            **lid_kw)
                    _BEVQ[f_pre["frame"]] = m._last_bev.detach().float()
                    keep = 14 if args.model in ("v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36", "v37", "v38", "v39", "v40", "v41", "v42") else 2
                    for kk in [k for k in _BEVQ
                               if isinstance(k, int)
                               and k < f_pre["frame"] - keep]:
                        _BEVQ.pop(kk)
                else:
                    out = (m(imgs_m[None].cuda(), K[None].cuda(),
                             T[None].cuda(), v0_t.cuda())
                           if v0_t is not None else
                           m(imgs_m[None].cuda(), K[None].cuda(),
                             T[None].cuda()))
            out = list(out)
            if refiner is not None:
                # Apply the refiner(s) to the frozen outputs. MUST be
                # no_grad + detach: this runs outside the model's no_grad
                # block, else the refiner graph (and the seg-fuse accumulator
                # derived from it) chains across frames and leaks to OOM.
                with torch.no_grad(), torch.autocast("cuda", torch.float16):
                    ctx = m.lane_input().float() if refiner._ctx else None
                    if getattr(refiner, "_multi", False):
                        rb = refiner.box is not None
                        re = refiner.e2e is not None
                        v0r = (v0_t.cuda() if v0_t is not None
                               else torch.zeros(1, device="cuda")) if re else None
                        fused = m._fused_bev.float() if re else None
                        r = refiner(
                            seg=out[0].float() if refiner.seg is not None else None,
                            hm=out[3].float() if rb else None,
                            reg=out[4].float() if rb else None,
                            ego=out[7].float() if re else None,
                            v0=v0r, fused=fused, seg_ctx=ctx)
                        if "seg" in r:
                            out[0] = r["seg"].detach()
                        if "hm" in r:
                            out[3] = r["hm"].detach(); out[4] = r["reg"].detach()
                        if "ego" in r:
                            out[7] = r["ego"].detach()
                    else:
                        out[0] = refiner(out[0].float(), ctx).detach()
            seg, dlog = out[0], out[1]        # v13 returns (seg, depth, seg2d)
            pred = seg.argmax(1)[0].cpu().numpy().astype(np.uint8)
            if args.seg_fuse:
                # temporal log-odds fusion in the ego frame (static classes):
                # fused for x<=45 m, raw beyond (pose noise misregisters
                # thin lines at long range).
                # THIN classes (crosswalk/laneline/stopline/road_edge) are
                # 1-2 cells wide: any sub-cell pose error smears them into the
                # dominant road/sidewalk class and the argmax drops the line
                # (measured: laneline 40-80m IoU 0.056->0.012 under fusion).
                # -> keep those pixels from the RAW prediction, fuse only the
                # area classes (measured thin-protected: lanes recover to raw
                # or better, road_edge 0-20m 0.185->0.196, area stability kept)
                try:
                    lp = torch.log_softmax(seg.float(), 1)
                    if _SEGACC.get("scene") == s_pre:
                        rl = _rel(_SEGACC["fi"])
                        if rl is not None:
                            thw = make_warp_theta(rl)
                            gr = F.affine_grid(thw, list(lp.shape),
                                               align_corners=False)
                            lp = lp + 0.7 * F.grid_sample(
                                _SEGACC["acc"], gr, align_corners=False)
                    _SEGACC.update(scene=s_pre, fi=f_pre["frame"],
                                   acc=lp.detach())
                    fpred = lp.argmax(1)[0].cpu().numpy().astype(np.uint8)
                    raw_pred = pred.copy()   # NOT `raw` (that's the mp4 path)
                    pred[175:] = fpred[175:]              # fuse near field
                    thin = np.isin(raw_pred, (3, 4, 5, 6))  # protect raw lines
                    pred[thin] = raw_pred[thin]
                except Exception:
                    pass
            if not args.no_thin:
                pred = thin_road_edge(pred)
            if args.model == "v15" and len(out) > 3:   # box occupancy overlay
                boxp = out[3].argmax(1)[0].cpu().numpy().astype(np.uint8)
                pred[boxp == 1] = 10
                pred[boxp == 2] = 11
            seg2d_pred = None
            if args.show_seg2d and isinstance(out, tuple) and len(out) > 2:
                seg2d_pred = out[2].argmax(2)[0].cpu().numpy().astype(np.uint8)
            det_boxes = None
            if args.model in ("v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36", "v37", "v38", "v39", "v40", "v41", "v42") and len(out) > 4:
                if args.model == "v30" and len(out) >= 18:
                    unk_boxes = m.decode_unknown(out[17].float())[0]
                else:
                    unk_boxes = []
                det_boxes = [d for d in m.decode_boxes(
                    out[3].float(), out[4].float(), thresh=0.25, topk=64)[0]
                    if d[1] > (0.45 if d[0] == 0 else 0.25)] + unk_boxes
            ego_modes = None
            if args.model in ("v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36", "v37", "v38", "v39", "v40", "v41", "v42") and len(out) >= 8:
                _e = out[7][0].float().cpu().numpy()
                _pr = np.exp(_e[36:39]) / np.exp(_e[36:39]).sum()
                _k = int(_pr.argmax())
                if args.guard and len(out) >= 13:      # C1: risk-aware pick
                    _rm = out[12][0, 0].float().sigmoid().cpu().numpy()
                    _k, _, _rk = risk_pick(
                        [_e[j * 12:(j + 1) * 12] for j in range(3)], _pr, _rm)
                ego_modes = [(_e[j * 12:(j + 1) * 12], float(_pr[j]), j == _k)
                             for j in range(3)]
                out = out[:7] + [torch.from_numpy(np.concatenate(
                    [_e[_k * 12:(_k + 1) * 12], _e[39:42]]))[None]] + out[8:]
            ego_pred = out[7][0].float().cpu().numpy() \
                if args.model in ("v18", "v19", "v20", "v21", "v22", "v23", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36", "v37", "v38", "v39", "v40", "v41", "v42") and len(out) >= 8 else None
            traj_map = out[9][0].float().cpu() \
                if args.model in ("v21", "v22", "v23", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36", "v37", "v38", "v39", "v40", "v41", "v42") and len(out) >= 10 else None
            stat_map = out[10][0, 0].float().cpu() \
                if args.model in ("v26", "v27", "v28", "v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36", "v37", "v38", "v39", "v40", "v41", "v42") and len(out) >= 11 else None
            risk_map = out[12][0, 0].float().sigmoid().cpu().numpy() \
                if args.model in ("v28", "v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36", "v37", "v38", "v39", "v40", "v41", "v42") and len(out) >= 13 else None
            tl_state = None
            if args.model in ("v27", "v28", "v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36", "v37", "v38", "v39", "v40", "v41", "v42") and len(out) >= 12:
                p_tl = out[11][0].float().softmax(0)
                tl_state = (int(p_tl.argmax()), float(p_tl.max()))
            occ_pred = None
            if args.model in ("v20", "v21", "v22", "v23", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36", "v37", "v38", "v39", "v40", "v41", "v42") and len(out) >= 9:
                op = out[8][0].float().softmax(0)      # [C,Z,H,W]
                conf = 1.0 - op[0]                      # P(occupied)
                cls = (op[1:].argmax(0) + 1).to(torch.uint8)
                # statics (veg/building) hallucinate in never-observed
                # voxels until the dense-carve GT lands -> stricter gate
                thr = torch.where((cls == 7) | (cls == 8),
                                  torch.full_like(conf, 0.92),
                                  torch.full_like(conf, 0.55))
                occ_pred = torch.where(conf > thr, cls,
                                       torch.zeros_like(cls)) \
                    .cpu().numpy().astype(np.uint8)
            boxes2d = None
            if args.model in ("v17", "v18", "v19", "v20", "v21", "v22", "v23", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36", "v37", "v38", "v39", "v40", "v41", "v42") and len(out) >= 7:
                if isinstance(out[5], (list, tuple)):   # v19 multi-scale
                    boxes2d = m.decode_boxes2d_ms(
                        [t[0].float() for t in out[5]],
                        [t[0].float() for t in out[6]], thresh=args.thresh2d)
                else:
                    boxes2d = m.decode_boxes2d(out[5][0].float(),
                                               out[6][0].float(),
                                               thresh=args.thresh2d)
            if 'det_boxes' not in dir():
                det_boxes = None
            dprob = dlog.float().softmax(2)[0]
            # top-mode expectation (argmax bin +-2, renormalised): the full
            # expectation mixes foreground/background modes at object
            # boundaries into phantom mid-range depths -> soft edges
            pk = dprob.argmax(1, keepdim=True)
            ar = torch.arange(dprob.shape[1], device=dprob.device
                              ).view(1, -1, 1, 1)
            pw = dprob * ((ar - pk).abs() <= 2)
            depth = ((pw.cpu() * dbins.view(1, -1, 1, 1)).sum(1)
                     / pw.sum(1).clamp(min=1e-6).cpu())     # [N,fh,fw]
            s, f = ds.items[i]
            frame = np.zeros((VH, VW, 3), np.uint8)

            # --- RGB block (top, 2x4) and Depth block (below, same grid) ---
            for k, chn in enumerate(CAM8):
                r, c = divmod(k, 4)
                x = c * cw
                p = f["imgs"].get(chn)
                if p:
                    img = cv2.resize(cv2.imread(os.path.join("out/bevlane", s, p)),
                                     (cw, ch))
                    if seg2d_pred is not None:
                        sc = SEG2D_PAL[seg2d_pred[CAMS.index(chn)]]
                        sc = cv2.resize(sc, (cw, ch),
                                        interpolation=cv2.INTER_NEAREST)
                        img = cv2.addWeighted(img, 0.62, sc, 0.38, 0)
                    if "NARROW" in chn:
                        label(img, "NARROW", (0, 255, 0))
                    if det_boxes:
                        ci0 = CAMS.index(chn)
                        draw_boxes_on_rgb(img, det_boxes, K[ci0].numpy(),
                                          T[ci0].numpy(), cw, ch)
                    if boxes2d is not None:
                        draw_boxes2d(img, boxes2d[CAMS.index(chn)], cw, ch)
                    if ego_pred is not None and chn == "CAM_FRONT_WIDE":
                        ci0 = CAMS.index(chn)
                        draw_path_ribbon(img, ego_pred, K[ci0].numpy(),
                                         T[ci0].numpy(), cw, ch,
                                         reset=(i == 0 or f["frame"] == 0))
                    frame[rgb_y0 + r * ch:rgb_y0 + (r + 1) * ch, x:x + cw] = img
                # depth directly under the same camera cell
                ci = CAMS.index(chn)
                d = depth[ci].numpy()
                dc = cv2.applyColorMap(np.clip(d / 80 * 255, 0, 255).astype(np.uint8),
                                       cv2.COLORMAP_TURBO)
                dc = cv2.resize(dc, (cw, ch), interpolation=cv2.INTER_NEAREST)
                label(dc, chn.split("CAM_")[-1], (255, 255, 255))
                frame[dep_y0 + r * ch:dep_y0 + (r + 1) * ch, x:x + cw] = dc
            cv2.putText(frame, "RGB + predicted 2D Seg overlay" if args.show_seg2d
                        else "RGB input (surround + tele NARROW)", (10, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
            dlabel = {"v13": "depth-gate internal repr. (uncalibrated)",
                      "v13d": "predicted Depth (0-80m, calibrated)",
                      "v14d": "predicted Depth (0-80m, calibrated)",
                      "v15": "predicted Depth (0-80m, calibrated)",
                      "v16": "predicted Depth (0-80m, calibrated)",
                      "v17": "predicted Depth (0-80m, calibrated)"}.get(
                          args.model, "predicted Depth (0-80m)")
            cv2.putText(frame, dlabel, (10, dep_y0 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)

            guard = None
            if args.guard and ego_pred is not None and det_boxes is not None:
                path6 = ego_pred[:12].reshape(6, 2)
                offs = []
                for d_ in det_boxes:
                    rr0 = int((80.0 - d_[2]) / 0.4)
                    cc0 = int((50.0 - d_[3]) / 0.4)
                    o_ = np.zeros((6, 2), np.float32)
                    if traj_map is not None and 0 <= rr0 < traj_map.shape[-2] \
                            and 0 <= cc0 < traj_map.shape[-1]:
                        v_ = traj_map[:, rr0, cc0]
                        if v_.numel() >= 39:
                            kb_ = int(v_[36:39].argmax())
                            o_ = v_[kb_ * 12:(kb_ + 1) * 12].view(6, 2).numpy()
                        elif v_.numel() >= 12:
                            o_ = v_[:12].view(6, 2).numpy()
                    offs.append(o_)
                tlp = p_tl.cpu().numpy() if tl_state is not None \
                    else np.array([1.0, 0, 0, 0])
                guard = check_path(path6, occ_pred, det_boxes, offs, tlp,
                                   pred, float(v0_t[0]) if v0_t is not None
                                   else 0.0)
            # --- BEV column: crop to +-25m lat x +-60m long (less blank space)
            pc = crop_bev(pred, xh_m=60.0, yh_m=25.0)      # 600 x 250 (long x lat)
            BH2 = VH - 90                                    # fill height
            BW2 = int(BH2 * pc.shape[1] / pc.shape[0])       # keep aspect
            bev = draw_ego_and_grid(DEMO_PALETTE[pc][:, :, ::-1], BH2, BW2,
                                    xh_m=60.0, yh_m=25.0)
            if risk_map is not None:
                # overlay predicted risk (+-40 x +-25 m) on the BEV panel
                rm = cv2.resize(risk_map, (BW2, int(BH2 * 80.0 / 120.0)),
                                interpolation=cv2.INTER_LINEAR)
                y0r = int(BH2 * (60.0 - 40.0) / 120.0)
                sub = bev[y0r:y0r + rm.shape[0]]
                heat = cv2.applyColorMap((np.clip(rm, 0, 1) * 255
                                          ).astype(np.uint8),
                                         cv2.COLORMAP_TURBO)
                a = (np.clip(rm, 0, 1) * 0.55)[..., None]
                bev[y0r:y0r + rm.shape[0]] = (sub * (1 - a) + heat * a
                                              ).astype(np.uint8)
            if det_boxes:
                sy2 = BH2 / 120.0            # px per metre (2*60m vertical)
                sx2 = BW2 / 50.0             # px per metre (2*25m lateral)
                for cls, sc, xe, ye, l, w, yaw in det_boxes:
                    if abs(xe) > 60 or abs(ye) > 25:
                        continue
                    cb, sb = np.cos(yaw), np.sin(yaw)
                    cor = []
                    for lx, wy in ((l/2, w/2), (l/2, -w/2), (-l/2, -w/2), (-l/2, w/2)):
                        px_ = xe + lx*cb - wy*sb
                        py_ = ye + lx*sb + wy*cb
                        cor.append([int((25.0 - py_) * sx2), int((60.0 - px_) * sy2)])
                    # v26: learned stationary flag; fallback: forecast<1m/3s
                    stationary = False
                    rr0 = int((80.0 - xe) / 0.4)
                    cc0 = int((50.0 - ye) / 0.4)
                    if stat_map is not None:
                        if 0 <= rr0 < stat_map.shape[-2] \
                                and 0 <= cc0 < stat_map.shape[-1]:
                            stationary = float(stat_map[rr0, cc0]) > 0
                    elif traj_map is not None:
                        if 0 <= rr0 < traj_map.shape[-2] \
                                and 0 <= cc0 < traj_map.shape[-1]:
                            _v0 = traj_map[:, rr0, cc0]
                            if _v0.numel() >= 39:
                                _kb0 = int(_v0[36:39].argmax())
                                w6 = _v0[_kb0 * 12:(_kb0 + 1) * 12].view(6, 2)
                            else:
                                w6 = _v0.view(6, 2)
                            stationary = float(w6[5].norm()) < 1.0
                    if stationary:
                        col = (160, 160, 160)          # parked / stopped
                    else:
                        col = ((0, 215, 255), (255, 0, 255),
                           (255, 255, 255))[min(int(cls), 2)]
                    cv2.polylines(bev, [np.array(cor, np.int32).reshape(-1, 1, 2)],
                                  True, col, 2)
                    # heading tick from centre to front edge
                    cxp = int((25.0 - ye) * sx2); cyp = int((60.0 - xe) * sy2)
                    fxp = int((25.0 - (ye + (l/2)*sb)) * sx2)
                    fyp = int((60.0 - (xe + (l/2)*cb)) * sy2)
                    cv2.line(bev, (cxp, cyp), (fxp, fyp), col, 2)
                    if traj_map is not None:      # predicted 3 s agent future
                        rr = int((80.0 - xe) / 0.4)
                        cc2 = int((50.0 - ye) / 0.4)
                        if 0 <= rr < traj_map.shape[-2] \
                                and 0 <= cc2 < traj_map.shape[-1]:
                            _v = traj_map[:, rr, cc2]
                            if _v.numel() >= 39:
                                _kb = int(_v[36:39].argmax())
                                wps = _v[_kb * 12:(_kb + 1) * 12].view(6, 2).numpy()
                            else:
                                wps = _v.view(6, 2).numpy()
                            pts = [(cxp, cyp)]
                            for dx, dy in wps:
                                fx, fy = xe + dx, ye + dy
                                if abs(fx) > 60 or abs(fy) > 25:
                                    break
                                pts.append((int((25.0 - fy) * sx2),
                                            int((60.0 - fx) * sy2)))
                            if len(pts) > 1:
                                cv2.polylines(
                                    bev,
                                    [np.array(pts, np.int32).reshape(-1, 1, 2)],
                                    False, col, 1, cv2.LINE_AA)
                                cv2.circle(bev, pts[-1], 3, col, -1)
            if ego_pred is not None and ego_modes is not None:
                sy2 = BH2 / 120.0
                sx2 = BW2 / 50.0
                for wp_m, pr_m, is_best in ego_modes:
                    if is_best:
                        continue
                    pts_m = [(int(25.0 * sx2), int(60.0 * sy2))]
                    for k in range(6):
                        xe, ye = wp_m[2 * k], wp_m[2 * k + 1]
                        if abs(xe) > 60 or abs(ye) > 25:
                            break
                        pts_m.append((int((25.0 - ye) * sx2),
                                      int((60.0 - xe) * sy2)))
                    cv2.polylines(bev, [np.array(pts_m, np.int32
                                                 ).reshape(-1, 1, 2)],
                                  False, (255, 200, 60), 1, cv2.LINE_AA)
                    if len(pts_m) > 1:
                        cv2.putText(bev, f"{pr_m:.2f}", pts_m[-1],
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                                    (255, 200, 60), 1, cv2.LINE_AA)
            if ego_pred is not None:            # E2E: trajectory + controls
                sy2 = BH2 / 120.0
                sx2 = BW2 / 50.0
                pts = [(int(25.0 * sx2), int(60.0 * sy2))]      # ego origin
                for k in range(6):
                    xe, ye = ego_pred[2 * k], ego_pred[2 * k + 1]
                    if abs(xe) > 60 or abs(ye) > 25:
                        break
                    pts.append((int((25.0 - ye) * sx2), int((60.0 - xe) * sy2)))
                # stationary: waypoints collapse onto ego -> show it explicitly
                trav = float(np.hypot(ego_pred[10], ego_pred[11]))
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
                if guard is not None:
                    sy2 = BH2 / 120.0
                    sx2 = BW2 / 50.0
                    if guard["verdict"] == "OK":
                        cv2.putText(bev, "GUARD OK", (6, BH2 - 118),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                    (0, 255, 0), 2, cv2.LINE_AA)
                    elif guard["verdict"] == "HOLD":
                        cv2.putText(bev, "GUARD OK (holding)", (6, BH2 - 118),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (255, 220, 0), 1, cv2.LINE_AA)
                    else:
                        cv2.putText(bev, "GUARD VETO", (6, BH2 - 138),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    (0, 0, 255), 2, cv2.LINE_AA)
                        cv2.putText(bev, guard["reason"], (6, BH2 - 118),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                                    (0, 80, 255), 1, cv2.LINE_AA)
                        if guard["p_event"] is not None:
                            ex_, ey_ = guard["p_event"]
                            if abs(ex_) < 60 and abs(ey_) < 25:
                                cv2.drawMarker(
                                    bev, (int((25.0 - ey_) * sx2),
                                          int((60.0 - ex_) * sy2)),
                                    (0, 0, 255), cv2.MARKER_TILTED_CROSS,
                                    18, 3)
                        if guard["stop_path"] is not None:
                            sp = [(int(25.0 * sx2), int(60.0 * sy2))]
                            for gx, gy in guard["stop_path"]:
                                if abs(gx) > 60 or abs(gy) > 25:
                                    break
                                sp.append((int((25.0 - gy) * sx2),
                                           int((60.0 - gx) * sy2)))
                            cv2.polylines(
                                bev, [np.array(sp, np.int32).reshape(-1, 1, 2)],
                                False, (0, 200, 255), 2, cv2.LINE_AA)
                            cv2.putText(bev, "MRM stop", sp[-1],
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                        (0, 200, 255), 1, cv2.LINE_AA)
                st_deg = np.degrees(ego_pred[12])
                acc = ego_pred[13]
                brk = 1 / (1 + np.exp(-ego_pred[14]))
                v0v = float(v0_t[0]) if v0_t is not None else 0.0
                for li, txt in enumerate(
                        [f"v0 {v0v * 3.6:5.1f} km/h",
                         f"steer {st_deg:+6.1f} deg",
                         f"accel {acc:+5.2f} m/s2",
                         f"BRAKE {brk:.2f}" if brk > 0.5 else f"brake {brk:.2f}"]):
                    cv2.putText(bev, txt, (6, BH2 - 78 + 22 * li),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                                (0, 80, 255) if "BRAKE" in txt else (0, 255, 0),
                                1, cv2.LINE_AA)
            cv2.putText(bev, "gray box = stationary",
                        (6, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (170, 170, 170), 1, cv2.LINE_AA)
            if tl_state is not None:
                tnm = ("TL none", "TL GREEN", "TL YELLOW", "TL RED")
                tcl = ((150, 150, 150), (80, 220, 80),
                       (0, 210, 235), (60, 60, 235))
                ti, tp_ = tl_state
                cv2.circle(bev, (BW2 - 150, 40), 9, tcl[ti], -1)
                cv2.putText(bev, f"{tnm[ti]} {tp_:.2f}",
                            (BW2 - 136, 46), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, tcl[ti], 2 if ti else 1, cv2.LINE_AA)
            cv2.putText(bev, "pred BEV+bbox+E2E +-25x+-60m"
                        if args.model in ("v18", "v19", "v20", "v21", "v22", "v23", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36", "v37", "v38", "v39", "v40", "v41", "v42") else
                        ("pred BEV+bbox +-25x+-60m" if args.model in ("v15", "v16", "v17")
                         else "pred BEV +-25x+-60m"), (6, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            if intent_lab:
                arr = {"LEFT": "<<", "RIGHT": ">>", "STRAIGHT": "^"}.get(
                    intent_lab, "")
                cv2.putText(bev, f"NAV {arr} {intent_lab}", (6, 48),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 220, 255), 2,
                            cv2.LINE_AA)
            bx = min(bx0, VW - BW2)                          # flush to right edge
            frame[40:40 + BH2, bx:bx + BW2] = bev

            if occ_pred is not None:
                iso = cube_render(occ_pred, W=900, H=760)
                iso = cv2.resize(iso, (426, 360))
                oy0 = VH - 368
                frame[oy0:oy0 + 360, 8:8 + 426] = iso
                cv2.putText(frame, "pred OCC voxel grid +-24m (bldg hidden)",
                            (8, oy0 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1,
                            cv2.LINE_AA)
            mode_tag = "LIDAR ON" if args.lidar else "camera-only"
            if args.seg_fuse:
                mode_tag += "  |  seg-EMA<=45m"
            cv2.putText(frame, f"{scene.split('+0900_')[-1]}  f{f['frame']:03d}  |  "
                        f"6/8-cam -> Depth + BEV (+-25x+-60m)  |  no GT  |  "
                        f"{mode_tag}",
                        (10, VH - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                        (0, 255, 255) if args.lidar else (0, 255, 0), 2,
                        cv2.LINE_AA)
            vw.write(frame)
            n += 1
        print("scene", scene, n, flush=True)
    vw.release()
    subprocess.run(["ffmpeg", "-y", "-i", raw, "-c:v", "libx264", "-crf", "23",
                    "-pix_fmt", "yuv420p", args.out], check=True, capture_output=True)
    os.remove(raw)
    print("done", n, args.out, flush=True)


if __name__ == "__main__":
    main()

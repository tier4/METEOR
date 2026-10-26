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
# history slots forced to zero = the training condition (METEOR_ZERO_HIST=1) and the
# no-history Orin engine; without it the demo feeds real history the model never saw
_ZERO_HIST = os.environ.get("METEOR_ZERO_HIST", "1") == "1"   # default ON: production models are trained with zero history
_EGO_DUMP = os.environ.get("METEOR_EGO_DUMP", "")
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from autolabel_bev import PALETTE  # noqa: E402
from bevlane.dataset import CAMS, BevLaneDataset  # noqa: E402
from bevlane.extract_bbox2d import DET10_PAL  # noqa: E402
from bevlane.demo_occ_gt import cube_render  # noqa: E402
from bevlane.guardrail import check_path, risk_pick  # noqa: E402
from bevlane.extract_occ import OCC_PAL  # noqa: E402
from bevlane.model import BEV_H, BEV_W, BEV_XR, make_warp_theta  # noqa: E402
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
                           DepthSegIPMNetV42, DepthSegIPMNetV43,
                           DepthSegIPMNetV44)

DET10_ABBR = ["obs", "car", "trk", "bus", "bcy", "mcy", "ped", "pnt", "tl", "ts"]


# Rendering lives in deploy/viz_np.py only (2026-08-24). Local and Orin
# used to carry duplicate copies of the same functions, so a mismatch in
# camera order or BEV fore/aft range on the Orin side went unnoticed. One source.
from deploy.viz_np import (draw_boxes2d, draw_boxes_on_rgb,  # noqa: E402
                          draw_path_ribbon)

from bevlane.postproc import crop_bev, draw_ego_and_grid, thin_road_edge  # noqa: E402


def _draw_gt_polyline(img, wp6, K, Tce, cw, ch, W0=768, H0=432, color=(255, 255, 255)):
    """Draw the GT path (ego_motion wp [6,2], ego frame) as a thin white line + dots (leaves draw_path_ribbon's EMA state alone)."""
    wps = np.concatenate([[[0.0, 0.0]], np.asarray(wp6, np.float32).reshape(6, 2)], 0)
    if not np.all(np.isfinite(wps)):
        return
    seg = np.linalg.norm(np.diff(wps, axis=0), axis=1); t = np.concatenate([[0], np.cumsum(seg)])
    if t[-1] < 0.5:
        return
    tt = np.linspace(0, t[-1], 40)
    px, py = np.interp(tt, t, wps[:, 0]), np.interp(tt, t, wps[:, 1])
    pts = np.stack([px, py, np.zeros_like(px)], 1) @ Tce[:3, :3].T + Tce[:3, 3]
    ok = pts[:, 2] > 0.3
    if ok.sum() < 2:
        return
    u = (K[0, 0] * pts[ok, 0] / pts[ok, 2] + K[0, 2]) * (cw / W0); v = (K[1, 1] * pts[ok, 1] / pts[ok, 2] + K[1, 2]) * (ch / H0)
    poly = np.stack([u, v], 1).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [poly], False, (0, 0, 0), 4); cv2.polylines(img, [poly], False, color, 2)
    _draw_wp_dots(img, wps[1:], K, Tce, cw, ch, color, W0, H0, r=4)


def _draw_wp_dots(img, wp6, K, Tce, cw, ch, color, W0=768, H0=432, r=5):
    """Draw each path waypoint (0.5 s steps, ego frame [6,2]) as a black-rimmed dot, to compare prediction (green) and GT (white) point by point."""
    for q in np.asarray(wp6, np.float32).reshape(-1, 2):
        p3 = np.array([q[0], q[1], 0.0]) @ Tce[:3, :3].T + Tce[:3, 3]
        if p3[2] > 0.3:
            c = (int((K[0, 0] * p3[0] / p3[2] + K[0, 2]) * cw / W0), int((K[1, 1] * p3[1] / p3[2] + K[1, 2]) * ch / H0))
            cv2.circle(img, c, r + 2, (0, 0, 0), -1); cv2.circle(img, c, r, color, -1)

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


def pseudo_intent(pose, fi, lo=5, hi=25, lat_th=3.0, dist_hi=None, deg_th=None):
    """Pseudo route intent from the recorded poses.
    Default: lateral offset > lat_th within 1-5 s ahead (in frames). At low urban speed (10-20 km/h) 5 s is only
    15-30 m, so the turn command fires only once the turn has begun.
    With dist_hi/deg_th: look ahead by distance along the path (e.g. 50 m) and output LEFT/RIGHT when the heading
    change within that stretch exceeds deg_th (default 30 deg). The command holds while the heading difference
    remains and returns to STRAIGHT after the turn, closer to a navigation system's advance instruction."""
    if dist_hi is not None:
        x0, y0, yaw0 = pose[fi]
        dist = 0.0; best = 0.0; j = fi
        while j + 1 < len(pose) and dist < dist_hi:
            j += 1
            dist += float(np.hypot(pose[j][0] - pose[j - 1][0], pose[j][1] - pose[j - 1][1]))
            if dist < 8.0:
                continue                                     # ignore heading changes within 8 m (minor corrections)
            d = (float(pose[j][2]) - yaw0 + np.pi) % (2 * np.pi) - np.pi
            if abs(d) > abs(best):
                best = d
        oh = np.zeros(3, np.float32); th = np.radians(deg_th or 30.0)
        if best > th:
            oh[1] = 1.0; lab = "LEFT"
        elif best < -th:
            oh[2] = 1.0; lab = "RIGHT"
        else:
            oh[0] = 1.0; lab = "STRAIGHT"
        return oh, lab
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


def pl_render(pl, W=420, H=720, half_x=50.0, half_y=25.0,
              rgb=None, K=None, T=None, z_above=2.2):
    """Pseudo-LiDAR raster -> an ISOMETRIC 3-D point cloud, painted with the
    camera colours the points project into.

    A top-down tile hides the very thing a predicted sweep is for (height
    structure), so the cloud is drawn in the isometric projection the
    occupancy cube uses. The view auto-fits the tile from the projected
    extent, so the cloud fills the cell whatever the scene looks like.
    Cells no camera sees keep a height colour, so nothing disappears."""
    img = np.zeros((H, W, 3), np.uint8)
    occ = pl[3] > 0.5
    rr, cc = np.nonzero(occ)
    if not len(rr):
        return img
    xm = 80.0 - rr * 0.4                        # forward (m)
    ym = 50.0 - cc * 0.4                        # left (m)
    k = (np.abs(xm) < half_x) & (np.abs(ym) < half_y)
    xm, ym, rr, cc = xm[k], ym[k], rr[k], cc[k]
    if not len(xm):
        return img
    zm = np.clip(pl[1][rr, cc], -1.0, 4.0)
    # Cut at roughly vehicle height. Buildings span the whole 5 m range and,
    # once the height axis is exaggerated, they lean over the road and hide it.
    # The ground is found per frame (10th percentile of the occupied cells'
    # height) so the cut follows slopes instead of assuming z=0.
    if len(zm) > 20:
        ground = float(np.percentile(zm, 10))
        keep_z = zm <= ground + z_above
        if keep_z.sum() > 20:
            xm, ym, rr, cc, zm = (xm[keep_z], ym[keep_z], rr[keep_z],
                                  cc[keep_z], zm[keep_z])

    # ---- colours: camera image where visible, height elsewhere ----
    # normalise over the KEPT height range: after the vehicle-height cut the
    # fixed [-1, 4] mapping compresses everything into one shade of blue
    zlo, zhi = float(zm.min()), float(zm.max())
    z01 = np.clip((zm - zlo) / max(zhi - zlo, 0.5), 0, 1)
    hcol = cv2.applyColorMap((z01 * 255).astype(np.uint8),
                             cv2.COLORMAP_JET).reshape(-1, 3).astype(np.int32)
    col = hcol.copy()
    if rgb is not None and K is not None and T is not None:
        pts = np.stack([xm, ym, zm, np.ones_like(xm)], 1)
        painted = np.zeros(len(xm), bool)
        for ci, im_c in enumerate(rgb):
            if im_c is None or painted.all():
                continue
            pc = (T[ci] @ pts.T).T
            zc = pc[:, 2]
            good = (~painted) & (zc > 0.5)
            if not good.any():
                continue
            u = K[ci][0, 0] * pc[:, 0] / np.maximum(zc, 1e-3) + K[ci][0, 2]
            v = K[ci][1, 1] * pc[:, 1] / np.maximum(zc, 1e-3) + K[ci][1, 2]
            hI, wI = im_c.shape[:2]
            uu = (u / 768.0 * wI).astype(np.int32)
            vv = (v / 432.0 * hI).astype(np.int32)
            good &= (uu >= 0) & (uu < wI) & (vv >= 0) & (vv < hI)
            if not good.any():
                continue
            col[good] = im_c[vv[good], uu[good]].astype(np.int32)
            painted |= good
        # Pure image colour washed the cloud out to grey at this size, so keep
        # a quarter of the height colour as a depth cue and lift a little.
        col[painted] = np.clip(col[painted] * 0.9 + hcol[painted] * 0.25 + 12,
                               0, 255)

    # ---- isometric coordinates, SAME viewpoint as the OCC cube render ----
    # cube_render() uses u ~ (col - row), v ~ (col + row) - z, and with
    # row ~ -x, col ~ -y that is u ~ +x - y, v ~ -(x + y) - z. This renderer
    # had u ~ y - x and v ~ +(x + y): inverted on BOTH axes, so the cloud faced
    # the opposite way from the voxels next to it. Its height exaggeration was
    # also 7.1 per metre against OCC's 2.09; matched here.
    HEX = 0.28 * 2.09

    def iso(x, y, z):
        return (x - y) * 0.5, -(y + x) * 0.28 - (z + 1.0) * HEX

    iu, iv = iso(xm, ym, zm)
    gx = np.arange(-half_x, half_x + 0.1, 10.0)
    gy = np.arange(-half_y, half_y + 0.1, 10.0)
    # Fit the CLOUD, not the grid: a predicted sweep reaches far less far than
    # the +-50 m grid, and fitting the grid shrank it to a third of the cell.
    # The grid is simply clipped by cv2.line where it leaves the tile.
    u0, u1 = iu.min(), iu.max()
    v0_, v1 = iv.min(), iv.max()
    sc = min(W * 0.92 / max(u1 - u0, 8.0), H * 0.92 / max(v1 - v0_, 8.0))
    ox = W * 0.5 - 0.5 * (u0 + u1) * sc
    oy = H * 0.5 - 0.5 * (v0_ + v1) * sc

    def to_px(u, v):
        return (u * sc + ox).astype(np.int32), (v * sc + oy).astype(np.int32)

    # ---- ground grid every 10 m ----
    for x_ in gx:
        a_u, a_v = iso(np.array([x_, x_]), np.array([-half_y, half_y]),
                       np.array([-1.0, -1.0]))
        p = to_px(a_u, a_v)
        cv2.line(img, (int(p[0][0]), int(p[1][0])),
                 (int(p[0][1]), int(p[1][1])), (55, 55, 55), 1, cv2.LINE_AA)
    for y_ in gy:
        a_u, a_v = iso(np.array([-half_x, half_x]), np.array([y_, y_]),
                       np.array([-1.0, -1.0]))
        p = to_px(a_u, a_v)
        cv2.line(img, (int(p[0][0]), int(p[1][0])),
                 (int(p[0][1]), int(p[1][1])), (55, 55, 55), 1, cv2.LINE_AA)

    # ---- points, painter's order far -> near ----
    px, py = to_px(iu, iv)
    inb = (px >= 0) & (px < W) & (py >= 0) & (py < H)
    order = np.argsort(-(xm + ym))          # far (large x+y) drawn first
    rad = np.where(xm < 15, 2, 1)
    if sc > 6.0:                              # zoomed in: fatter points
        rad = rad + 1
    for i in order:
        if inb[i]:
            cv2.circle(img, (int(px[i]), int(py[i])), int(rad[i]),
                       tuple(int(t) for t in col[i]), -1)
    eu, ev = iso(np.zeros(1), np.zeros(1), np.zeros(1))
    ep = to_px(eu, ev)
    cv2.drawMarker(img, (int(ep[0][0]), int(ep[1][0])), (255, 255, 255),
                   cv2.MARKER_TRIANGLE_UP, 14, 2)
    return img


def label(img, txt, color=(255, 255, 255)):
    cv2.putText(img, txt, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


_BEVQ = {}
_SEGACC = {}



def _veh_th(d, args):
    """Box display threshold: vehicles are raised per distance band (--box-thresh-veh-far), VRUs fixed. d = (cls, score, x, y, l, w, yaw)."""
    if d[0] != 0:
        return args.box_thresh_ped
    th = args.box_thresh_veh
    if getattr(args, "box_thresh_veh_far", ""):
        r = (d[2] ** 2 + d[3] ** 2) ** 0.5
        for tok in args.box_thresh_veh_far.split(","):
            dist, t = tok.split(":")
            if r >= float(dist): th = max(th, float(t))
    return th

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out/bevlane_ckpt_v12/best.pt")
    ap.add_argument("--scenes", nargs="+", default=[])
    ap.add_argument("--mite-ckpt", default="",
                    help="METEORite ckpt: forward runs METEORite; rendering/decoding reuse the v52 model from --ckpt")
    ap.add_argument("--out", default="out/demo_rgbd_bev.mp4")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--zero-cams", default="",
                    help="comma-separated cameras to disable (J6 7-cam demo)")
    ap.add_argument("--show-pl", action="store_true",
                    help="v48: draw the PREDICTED (pseudo) LiDAR raster into the BEV panel, height-coloured like a real sweep")
    ap.add_argument("--pseudo-lidar", action="store_true",
                    help="v48: feed the predicted (pseudo) LiDAR "
                         "raster back in (inference ON/OFF switch)")
    ap.add_argument("--no-agent-traj", action="store_true", help="do not draw the other agents' forecast trajectories")
    ap.add_argument("--no-2d-boxes", action="store_true", help="do not draw 2D boxes on the camera images (--unk2d BEV points remain)")
    ap.add_argument("--no-risk", action="store_true", help="do not overlay the risk map")
    ap.add_argument("--show-ped-path", action="store_true",
                    help="draw predicted futures for VRU/pedestrians too")
    ap.add_argument("--unk2d", action="store_true",
                    help="lift 2D 'obs' detections to BEV via predicted "
                         "depth; fixed-size white diamonds on the BEV panel")
    ap.add_argument("--unk2d-thresh", type=float, default=0.35)
    ap.add_argument("--unk2d-draw", action="store_true",
                    help="with --no-2d-boxes, still draw unknown (obs) 2D boxes in white (matching the white BEV diamonds)")
    ap.add_argument("--intent", default="none",
                    choices=["none", "auto", "navsim", "straight", "left", "right"],
                    help="pseudo Driving Command into the v37+ intent input: "
                         "auto = navigation-style, derived from the RECORDED "
                         "route 1-5 s ahead (command fires BEFORE the turn)")
    ap.add_argument("--int8-sim", action="store_true",
                    help="Emulate INT8 inside PyTorch (per-channel weights + static "
                         "activation calibration). For isolating the TensorRT INT8 "
                         "ego-freeze issue (2026-08-22)")
    ap.add_argument("--int8-pct", type=float, default=99.9,
                    help="activation-scale percentile for int8-sim")
    ap.add_argument("--no-thin", action="store_true")
    ap.add_argument("--seg-fuse", action="store_true",
                    help="ego-warped log-odds fusion of BEV seg over time "
                         "(<=45 m; measured: stability .83->.93, "
                         "crosswalk 20-40m +.06)")
    ap.add_argument("--trt-engine", default=None,
                    help="run inference with a TensorRT engine instead of "
                         "PyTorch. The engine must expose raw_bev so the "
                         "temporal queue can be maintained from its output")
    ap.add_argument("--frustum-lift", action="store_true",
                    help="gather BEV cells per camera before sampling: the "
                         "same computation (fp32-exact) with 77%% of the work "
                         "removed. Inference only, no retraining")
    ap.add_argument("--root", default="out/bevlane",
                    help="dataset root: <root>/<scene>/{manifest.json,img,...}")
    ap.add_argument("--frame-stride", type=int, default=1,
                    help="render every Nth frame (the temporal queue still "
                         "sees every frame, only writing is thinned)")
    ap.add_argument("--scenes-file", default=None,
                    help="newline-separated scene list, appended to --scenes")
    ap.add_argument("--shard", default=None,
                    help="i/n: render only shard i of n scenes (parallel GPUs)")
    ap.add_argument("--refine-heads", default=None,
                    help="comma list of refiner heads to APPLY, overriding the "
                         "acceptance gate stored in the ckpt "
                         "(e.g. e2e,stat,pl,box)")
    ap.add_argument("--refiner-ckpt", default=None,
                    help="apply a trained BEVSegRefiner to the BEV-seg logits "
                         "(far-range completion; measured road 40-80m +.14)")
    ap.add_argument("--model", default="v8", choices=["v8", "v13", "v13d", "v14d", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36", "v37", "v38", "v39", "v40", "v41", "v42", "v43", "v44", "v45", "v46", "v47", "v48", "v49", "v50", "v51", "v52", "v52r50", "v53", "v54", "v55", "v56", "v63b"],
                    help="v8=DepthGatedIPMNet(512) / v13=DepthSegIPMNet(768) / v13d=stride-4 depth")
    ap.add_argument("--thin-bias", type=float, default=0.5,
                    help="subtract this from the laneline/stopline/road_edge "
                         "logits before the argmax. 0 restores the raw "
                         "(over-thick) output; 0.5 is the measured optimum")
    ap.add_argument("--show-seg2d", action="store_true",
                    help="alpha-blend predicted 2D seg over RGB panels")
    ap.add_argument("--path-smooth", action="store_true",
                    help="re-fit each E2E mode as a cubic through the ego origin (bevlane/path_smooth.py): "
                         "removes the waypoint zig-zag learned from GT pose noise; decision unchanged")
    ap.add_argument("--box-thresh", type=float, default=0.25,
                    help="3D box score threshold for drawing (0.25 = T4 default; another rig / sensor "
                         "may compress the scores and need its own GT-swept value)")
    ap.add_argument("--box-thresh-veh", type=float, default=0.35, help="display threshold for vehicle boxes (previously fixed at 0.35)")
    ap.add_argument("--box-thresh-veh-far", default="", help="per-distance vehicle display thresholds, e.g. '30:0.5,50:0.6' (reduces far-range false positives)")
    ap.add_argument("--box-thresh-ped", type=float, default=0.15, help="display threshold for pedestrian/VRU boxes (previously fixed at 0.15; 0.30 recommended for METEORite)")
    ap.add_argument("--nav-dist", type=float, default=0.0, help="look-ahead distance [m] for --intent auto (0 = legacy 1-5 s lateral-offset rule; 50 for navigation-like behaviour)")
    ap.add_argument("--nav-deg", type=float, default=30.0, help="heading change [deg] that triggers LEFT/RIGHT with --nav-dist")
    ap.add_argument("--box-post", default="off",
                    help="3D box post-process (bevlane/box_post.py) instead of the display yaw EMA; '+'-joined "
                         "steps in this order: raynms drops same-ray depth duplicates (camera-only input), nms = "
                         "oriented-IoU NMS (BOXPOST_NMS_IOU, default 0.3), track = world-frame tracker "
                         "smoothing heading/size with flip hysteresis (needs ego_motion.npz pose/stamp) and, when "
                         "the track moves, orienting the agent forecast (orient_traj). Low-rate (2 Hz) footage: "
                         "cam-only raynms+nms+track, LiDAR nms+track. none = raw boxes (no display yaw EMA either; "
                         "for before/after comparisons); off = the display yaw EMA")
    ap.add_argument("--thresh2d", type=float, default=0.25,
                    help="2D bbox score threshold (v17)")
    ap.add_argument("--n-seg2d", type=int, default=12,
                    help="2D seg head classes (21 for seg2d21-trained ckpts; "
                         "uses the csv Cityscapes-like palette)")
    ap.add_argument("--guard", action="store_true",
                    help="L1 safety guardrails: spacetime collision / red "
                    "light / feasibility / drivable checks on the E2E path")
    ap.add_argument("--scene-dump", default="",
                    help="also write the per-frame scene record (boxes+ids, paths, forecasts, BEV raster, "
                         "occupancy) under this directory for offline visualisation")
    ap.add_argument("--lidar-bev", action="store_true",
                    help="feed the per-frame LiDAR pillar raster "
                         "(v32+ optional input)")
    ap.add_argument("--sdmap", action="store_true",
                    help="feed the OSM SD-map prior (v46+ optional input)")
    ap.add_argument("--lidar", action="store_true",
                    help="v31: feed the per-frame LiDAR sparse depth "
                    "(depth_gt4) as the optional input; omit = camera-only")
    ap.add_argument("--infer-hw", default=None,
                    help="resize images to HxW for the model. default: 288x512 for "
                         "v8, none (native 768) for v13. 'none' = use cache res")
    args = ap.parse_args()
    if args.scenes_file:
        args.scenes = list(args.scenes) + [l.strip() for l in
                                           open(args.scenes_file) if l.strip()]
    if args.shard:
        _i, _n = (int(x) for x in args.shard.split("/"))
        args.scenes = args.scenes[_i::_n]
        print(f"[shard] {_i}/{_n}: {len(args.scenes)} scenes", flush=True)
    if not args.scenes:
        ap.error("no scenes: pass --scenes and/or --scenes-file")

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
            "v40": DepthSegIPMNetV40, "v41": DepthSegIPMNetV41, "v42": DepthSegIPMNetV42, "v43": DepthSegIPMNetV43, "v44": DepthSegIPMNetV44}.get(args.model) or __import__("bevlane.model", fromlist=["MODELS"]).MODELS[args.model]
    mkw = {"n_seg": args.n_seg2d} if args.model in (
        "v13", "v13d", "v14d", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36", "v37", "v38", "v39", "v40", "v41", "v42", "v43", "v44", "v45", "v46", "v47", "v48", "v49", "v51", "v52", "v52r50", "v53", "v54", "v55", "v56", "v63b") else {}
    m = mcls(**mkw).cuda().eval()
    if args.pseudo_lidar:
        m.pl_feed = True
        print('[pseudo-lidar] feeding predicted raster', flush=True)
    if args.zero_cams:
        m.zero_cams = tuple(CAMS.index(c) for c in args.zero_cams.split(","))
        print(f"[zero-cams] {args.zero_cams} -> {m.zero_cams}", flush=True)
    if args.n_seg2d == 21:      # csv taxonomy: Cityscapes-like colours (BGR)
        from bevlane.extract_seg2d import SEG21_PAL
        SEG2D_PAL[:] = 0
        SEG2D_PAL[:21] = SEG21_PAL[:, ::-1]
    _sd = {k.replace("module.", ""): v for k, v in
           torch.load(args.ckpt, map_location="cpu")["model"].items()}
    if any(k.startswith("lane_branch.") for k in _sd):
        from bevlane.model import enable_lane_branch
        enable_lane_branch(m)
    if any(k.startswith("paint_proj.") for k in _sd):
        # PointPainting ckpt: enable the projection before loading
        _ck_args = torch.load(args.ckpt, map_location="cpu").get("args") or {}
        _pc = _ck_args.get("paint_seg") or "2,3,4,5,6,7"
        m.enable_paint_seg([int(x) for x in str(_pc).split(",")])
        print(f"[paint-seg] classes={_pc}", flush=True)
    if any(k.startswith("delta_stat.") for k in _sd):
        from bevlane.model import enable_delta_stat
        enable_delta_stat(m)
        print("[delta-stat] temporal-delta stat head enabled", flush=True)
    if any(k.startswith("stop_head.") for k in _sd):
        from bevlane.model import enable_stop_mode
        enable_stop_mode(m)
        print("[stop-mode] stop candidate head enabled", flush=True)
    if "vocab.anchors" in _sd:
        from bevlane.model import enable_vocab_planner
        enable_vocab_planner(m, _sd["vocab.anchors"], hidden=int(_sd["vocab.cls.weight"].shape[1]))
        print(f"[vocab] vocabulary planner enabled (K={_sd['vocab.anchors'].shape[0]})", flush=True)
    if "depth_head.0.0.weight" in _sd and hasattr(m, "depth_head"):
        _wck = tuple(_sd[f"depth_head.{i}.0.weight"].shape[0]
                     for i in range(4)
                     if f"depth_head.{i}.0.weight" in _sd)
        _wcur = tuple(m[0].out_channels for m in m.depth_head[:-1])
        if len(_wck) == 4 and _wck != _wcur:
            from bevlane.model import enable_depth_slim
            enable_depth_slim(m, widths=_wck)
            print(f"[depth-slim] detected width {_wck}", flush=True)
    if any(k.startswith("sem_ego.") for k in _sd):
        from bevlane.model import enable_semantic_ego
        enable_semantic_ego(m)
        print("[semantic-ego] semantic ego residual enabled", flush=True)
    if any(k.startswith("lane_sdf.") for k in _sd):
        # v120c line: lane signed-distance aux head (training aid, unused at inference)
        from bevlane.model import enable_lane_sdf
        enable_lane_sdf(m)
        print("[lane-sdf] aux head enabled", flush=True)
    if any(k.startswith("stat_head2.proj.") for k in _sd):
        # v119/v120 line: stationary head is bounded from training (INT8 fix)
        from bevlane.model import enable_quant_stat_head
        enable_quant_stat_head(m, 8.0)
        print("[stat] bounded stat head (trained) enabled", flush=True)
    if any(k.startswith("ctx.0.body.") for k in _sd):
        from bevlane.model import enable_ctx_v2
        enable_ctx_v2(m, verbose=False)
    if any(".body.0.weight" in k and (k.startswith("det_stem.") or ".out." in k
                                      or k.startswith("dec.")) for k in _sd):
        from bevlane.model import enable_head_v2
        enable_head_v2(m, verbose=False)
    m.load_state_dict(_sd)
    _F3 = {}            # METEOR_F3_HEAD / METEOR_F3_ACC_SEL post-process the path (as in eval_unified); acc/gt are filled from ego_motion every frame
    if args.mite_ckpt:
        # METEORite: the v52 net above stays only as the decode/render host; every
        # forward is answered by the METEORite checkpoint, re-packed into the v52 output tuple.
        # depth -> metres (shape [B,N,1,h,w] = the demo's regression branch), 2D heads -> silent
        # (untrained in the mite line), occ -> empty (no occ head).
        from bevlane.meteorite import load_mite_eval
        import bevlane.meteorite as _MT
        _mite = load_mite_eval(args.mite_ckpt)
        if os.environ.get("METEOR_F3_HEAD"):
            from bevlane.f3_refine import enable_f3_refine, f3_apply
            enable_f3_refine(_mite.net, os.environ["METEOR_F3_HEAD"], hook=False)
            _F3["on"] = True
            print(f"[demo] F3 refine on (rule th={os.environ.get('METEOR_F3_ACC_SEL', '0')})", flush=True)

        def _mite_forward(imgs, K, T, v0=None, *a, **kw):
            # Nano (monocular ckpt, N_CAM=1): the display keeps all 8 cameras, inference gets only the first N_CAM
            # (CAM_FRONT_WIDE); per-camera outputs (depth, 2D seg, 2D boxes) are zero-padded back to 8 cameras
            N_all = imgs.shape[1]; NC = min(_MT.N_CAM, N_all)
            if NC < N_all:
                imgs, K, T = imgs[:, :NC], K[:, :NC], T[:, :NC]
            _v0 = v0.view(-1) if v0 is not None else torch.zeros(imgs.shape[0], device=imgs.device)
            o = _mite.net.forward_geo(imgs.float(), K, T, _v0, lidar=kw.get("lidar_bev"))
            if _F3.get("on"):
                _ext = torch.tensor([[float(_F3.get("acc", 0.0)) / 3.0, 0.0]], device=imgs.device)
                o["ego"] = f3_apply(_mite.net, o["ego"], _v0.to(imgs.device), _ext)
            B, N = imgs.shape[:2]
            step = (np.log(_MT.D_MAX) - np.log(_MT.D_MIN)) / (_MT.D - 1)
            centres = torch.exp(np.log(_MT.D_MIN) + step * torch.arange(_MT.D, device=imgs.device)).view(1, 1, -1, 1, 1)
            # Top-mode expectation as in the main model (renormalised over argmax +-2 bins): the all-bin expectation
            # mixes the foreground/background modes at object boundaries into phantom intermediate depths. METEORite
            # depth is 1/8 resolution (54x96), so upsampling to 1/4 is nearest (bilinear recreates the same blend)
            dp = o["dlog"].float().softmax(2)
            pk = dp.argmax(2, keepdim=True)
            ar = torch.arange(dp.shape[2], device=dp.device).view(1, 1, -1, 1, 1)
            pw = dp * ((ar - pk).abs() <= 2)
            dm = (pw * centres).sum(2, keepdim=True) / pw.sum(2, keepdim=True).clamp(min=1e-6)   # [B,N,1,h,w] metres
            up = lambda t, mode="bilinear": F.interpolate(t.flatten(0, 1).float(), (108, 192), mode=mode,
                                                          **({} if mode == "nearest" else {"align_corners": False})
                                                          ).view(B, N, -1, 108, 192)
            # Map METEORite's 2D box head (hm2d 10ch / reg2d 4ch on the FHxFW feature grid, stride = input width / FW)
            # onto the v52 decoder (108x192, DET2D_S=4, 768 px reference). At 720p the grid is resized to 108x192 and
            # the (log) sizes are scaled by 768/IMG_W.
            padc = lambda t: t if t.shape[1] == N_all else torch.cat([t, t.new_zeros(B, N_all - t.shape[1], *t.shape[2:])], 1)
            if "hm2d" in o and os.environ.get("MITE_DET2D", "1") == "1":
                h2 = o["hm2d"].float().view(B, NC, 10, *o["hm2d"].shape[-2:])
                r2 = o["reg2d"].float().view(B, NC, 4, *o["reg2d"].shape[-2:])
                h2 = F.interpolate(h2.flatten(0, 1), (108, 192), mode="bilinear", align_corners=False).view(B, NC, 10, 108, 192)
                r2 = F.interpolate(r2.flatten(0, 1), (108, 192), mode="nearest").view(B, NC, 4, 108, 192)
                r2[:, :, 2:4] += float(np.log(768.0 / _MT.IMG_W))
                hm2d = padc(h2) if NC < N_all else h2
                reg2d_out = padc(r2) if NC < N_all else r2
            else:
                hm2d = torch.full((B, N_all, 10, 108, 192), -20.0, device=imgs.device); reg2d_out = None
            if "occ" in o:                                          # ckpt with occupancy head: [B, 160, 200, 200] -> [B, 10, 16, 200, 200]
                occ = o["occ"].float().view(B, 10, 16, *o["occ"].shape[-2:])
            else:
                occ = torch.zeros(B, 10, 16, 200, 200, device=imgs.device); occ[:, 0] = 20.0
            m._last_bev = torch.zeros(1, device=imgs.device)
            return (o["lane"], padc(up(dm, "nearest")), padc(up(o["seg2d"])), o["hm"], _MT.reg_half_angle(o["reg"]), hm2d,
                    reg2d_out if reg2d_out is not None else torch.zeros(B, N_all, 4, 108, 192, device=imgs.device), o["ego"], occ, o["traj"],
                    o["stationary"], o["tl"], o["risk"])
        m.forward = _mite_forward
        print(f"[demo] METEORite mode: forward answered by {args.mite_ckpt}", flush=True)
    # Every enable_* above (paint-seg, delta-stat, depth-slim rebuild, semantic-ego) attaches NEW modules after
    # the .eval() on construction, so they stayed in training mode: BatchNorm ran on per-frame statistics and
    # the slim depth head mis-placed features -> spurious near-range boxes in the rendered videos (the same
    # weights through probe_net were clean).
    m.eval()
    if args.int8_sim:
        # Reproduce the TensorRT INT8 scheme (per-channel weights + static
        # activation calibration) in PyTorch. Same implementation as probe_int8_sim.py.
        _QMAX = 127
        _SC = {}
        _COL = {"on": True}
        with torch.no_grad():
            for _n, _mod in m.named_modules():
                if isinstance(_mod, (torch.nn.Conv2d, torch.nn.Linear)):
                    _w = _mod.weight.data
                    _d = tuple(range(1, _w.dim()))
                    _s = (_w.abs().amax(dim=_d, keepdim=True)
                          / _QMAX).clamp_min(1e-12)
                    _mod.weight.data = torch.round(_w / _s).clamp(
                        -_QMAX, _QMAX) * _s

        def _fq(mod, inp, out):
            if not torch.is_tensor(out) or not out.is_floating_point():
                return out
            k = id(mod)
            if _COL["on"]:
                v = out.detach().abs().flatten().float()
                if args.int8_pct >= 100.0:
                    mx = float(v.amax())
                else:
                    kk = max(1, int(v.numel() * args.int8_pct / 100.0))
                    mx = float(v.kthvalue(kk).values)
                _SC[k] = max(_SC.get(k, 0.0), mx)
                return out
            mx = _SC.get(k, 0.0)
            if mx <= 0:
                return out
            sc = mx / _QMAX
            return torch.round(out / sc).clamp(-_QMAX, _QMAX) * sc

        for _n, _mod in m.named_modules():
            if isinstance(_mod, (torch.nn.Conv2d, torch.nn.Linear)):
                _mod.register_forward_hook(_fq)
        m._int8_collect = _COL
        print(f"[int8-sim] per-channel INT8 weights + activation pct={args.int8_pct} "
              f"(calibrated on the first few frames)", flush=True)
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
                do_e2e="e2e" in heads, do_traj="traj" in heads,
                do_risk="risk" in heads, do_unk="unk" in heads,
                do_stat="stat" in heads, do_pl="pl" in heads,
                do_depth="depth" in heads, do_seg2d="seg2d" in heads,
                do_det2d="det2d_hm" in heads, do_occ="occ" in heads,
                do_tl="tl" in heads, do_flow="flow" in heads,
                do_lg="lg_pts" in heads, n_seg2d=args.n_seg2d,
                n_cls=N_CLASSES,
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
        # Per-head acceptance gate. The refiner improves E2E/stationary/PL but
        # MEASURABLY hurts val BEV seg (r47: road 40-80m 0.474->0.420,
        # stopline 0-20m 0.251->0.208; r45's 6-head refiner did the same), and
        # BEV seg must never regress. train_refiner now stores which heads
        # actually won; anything listed False is loaded but not applied.
        acc = dict(ck.get("accept") or {})
        if args.refine_heads:
            want = set(args.refine_heads.split(","))
            acc = {k: (k in want) for k in
                   ("seg", "box", "e2e", "stat", "pl", "unk")}
        # NO per-class logit mixing. It was tried and is measurably WRONG:
        # the refiner learns `refined = raw + residual` across ALL channels at
        # once, so swapping a single channel into an otherwise-raw logit field
        # leaves that class systematically below its untouched competitors and
        # the argmax stops choosing it. Measured on 60 val frames with r48's
        # laneline-only gate: laneline IoU 0.125 -> 0.0045 and predicted
        # laneline pixels 0.134 % -> 0.0013 %, while road and road_edge were
        # untouched -- exactly the "road is clean, lanes are gone" report.
        # The seg head is therefore applied whole or not at all; `seg_classes`
        # stays in the ckpt as diagnostics only.
        refiner._segc = None
        refiner._acc = acc
        off = sorted(k for k, v in acc.items() if not v)
        print(f"[refiner] loaded {args.refiner_ckpt} multi={refiner._multi} "
              f"heads={sorted(heads)} epoch={ck.get('epoch')}"
              + (f" NOT-APPLIED={off} (measured worse)" if off else ""),
              flush=True)
    if args.frustum_lift:
        m.frustum_lift = True
        print("[lift] frustum-restricted projection ON", flush=True)
    trt_run = None
    if args.trt_engine:
        # The PyTorch model stays loaded: the demo uses its decode helpers
        # (decode_boxes, pl_activate, D/D_STEP) which hold no weights, while
        # every forward comes from the engine.
        import tensorrt as trt_mod
        _rt = trt_mod.Runtime(trt_mod.Logger(trt_mod.Logger.ERROR))
        _rt.engine_host_code_allowed = True
        _eng = _rt.deserialize_cuda_engine(open(args.trt_engine, "rb").read())
        _ctx = _eng.create_execution_context()
        _buf, _order = {}, []
        for _i in range(_eng.num_io_tensors):
            _n = _eng.get_tensor_name(_i)
            _shp = tuple(_eng.get_tensor_shape(_n))
            _dt = {"DataType.FLOAT": torch.float32,
                   "DataType.HALF": torch.float16,
                   "DataType.INT32": torch.int32,
                   "DataType.INT8": torch.int8}[str(_eng.get_tensor_dtype(_n))]
            _t = torch.zeros(*_shp, dtype=_dt, device="cuda")
            _buf[_n] = _t
            _ctx.set_tensor_address(_n, int(_t.data_ptr()))
            if _eng.get_tensor_mode(_n) == trt_mod.TensorIOMode.OUTPUT:
                _order.append(_n)
        _stream = torch.cuda.Stream()
        _zlg = (torch.zeros(1, 24, 12, 2, device="cuda"),
                torch.zeros(1, 24, 4, device="cuda"),
                torch.full((1, 24, 24), -20.0, device="cuda"))
        if "lg_pts" not in _buf:
            print("[trt] engine has no lane-graph outputs (pruned)",
                  flush=True)
        print(f"[trt] {args.trt_engine} loaded, {len(_order)} outputs",
              flush=True)

        def trt_run(imgs_, K_, T_, v0_, pb_, th_):
            for k, nm in (("imgs", "imgs"), ("K", "K"), ("T", "T_cam_ego"),
                          ("v0", "v0"), ("hist_bev", "hist_bev"),
                          ("hist_theta", "hist_theta")):
                src = {"imgs": imgs_, "K": K_, "T": T_, "v0": v0_,
                       "hist_bev": pb_, "hist_theta": th_}[k]
                if nm in _buf and src is not None:
                    _buf[nm].copy_(src.to(_buf[nm].dtype))
            _ctx.execute_async_v3(_stream.cuda_stream)
            _stream.synchronize()
            g = _buf
            # rebuild the tuple the renderer expects (multi-scale 2D heads are
            # three separate engine outputs)
            return (g["lane"], g["depth"], g["seg2d"], g["hm"], g["reg"],
                    (g["hm2d_s0"], g["hm2d_s1"], g["hm2d_s2"]),
                    (g["reg2d_s0"], g["reg2d_s1"], g["reg2d_s2"]),
                    g["ego"], g["occ"], g["traj"], g["stationary"], g["tl"],
                    g["risk"], g["flow"],
                    # a lane-graph-free engine still has to fill the slots the
                    # renderer indexes; zeros decode to nothing drawn
                    g.get("lg_pts", _zlg[0]), g.get("lg_meta", _zlg[1]),
                    g.get("lg_adj", _zlg[2]),
                    g["unk"], g["pl"], g["raw_bev"])

    dbins = torch.arange(m.D) * m.D_STEP + m.D_MIN
    infer_hw = args.infer_hw or ("none" if args.model in ("v13", "v13d", "v14d", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36", "v37", "v38", "v39", "v40", "v41", "v42", "v43", "v44", "v45", "v46", "v47", "v48", "v49", "v51", "v52", "v52r50", "v53", "v54", "v55", "v56", "v63b") else "288x512")
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
    _dump = None
    if args.scene_dump:
        from bevlane.scene_dump import SceneDump
        _dump = SceneDump(args.scene_dump)
    n = 0
    for scene in args.scenes:
        if not os.path.exists(f"{args.root}/{scene}/manifest.json"):
            continue
        # Native-resolution input via METEOR_HR_ROOT / METEOR_IMG_SCALE (same env as the evaluator). Scenes missing
        # from hr_root fall back to upscaled 768 images (same rule as dataset.py); K follows img_scale.
        ds = BevLaneDataset(args.root, [scene], gt_key="gt_vec", with_depth=False,
                            gt_optional=bool(args.scene_dump) or bool(args.mite_ckpt),
                            img_scale=float(os.environ.get("METEOR_IMG_SCALE", "1") or 1),
                            hr_root=os.environ.get("METEOR_HR_ROOT") or None)
        # A camera the RECORDING does not have is treated exactly like one
        # zeroed by --zero-cams: blank tile, and the freed depth cell taken
        # over by the OCC render. x2gen2 is a 7-camera rig (no
        # CAM_BACK_NARROW) and the dataset already feeds zeros there.
        blank = set(args.zero_cams.split(",")) if args.zero_cams else set()
        blank |= {CAMS[i] for i in ds.absent.get(scene, ())}
        if blank:
            print(f"[cams] blank: {sorted(blank)}", flush=True)
        for i in range(len(ds)):
            imgs, K, T, _ = ds[i]
            # with native-resolution input (img_scale != 1), overlays on camera images (boxes, path, unk2d) use the 768x432-based K
            K_draw = K.clone(); K_draw[:, :2] /= float(getattr(ds, 'img_scale', 1) or 1)
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
            if args.model in ("v18", "v19", "v20", "v21", "v22", "v23", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36", "v37", "v38", "v39", "v40", "v41", "v42", "v43", "v44", "v45", "v46", "v47", "v48", "v49", "v51", "v52", "v52r50", "v53", "v54", "v55", "v56", "v63b"):  # v0
                try:
                    z = np.load(os.path.join(args.root, s_pre,
                                             "ego_motion.npz"))
                    v0_t = torch.tensor([float(z["v0"][f_pre["frame"]])])
                    _fi = int(f_pre["frame"])                        # current accel and GT path (for the thin-line overlay)
                    from bevlane.dataset import causal_acc
                    _F3["acc"] = causal_acc({k: z[k] for k in z.files}, _fi)   # causal (z["acc"] looks ahead; target only)
                    _F3["gt"] = np.asarray(z["wp"][_fi], np.float32) if "wp" in z.files and _fi < len(z["wp"]) and bool(z["valid"][_fi]) else None
                except Exception:
                    v0_t = torch.zeros(1); _F3["acc"] = 0.0; _F3["gt"] = None
            pb = th = None
            if args.model in ("v22", "v23", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36", "v37", "v38", "v39", "v40", "v41", "v42", "v43", "v44", "v45", "v46", "v47", "v48", "v49", "v51", "v52", "v52r50", "v53", "v54", "v55", "v56", "v63b"):
                fi_cur = f_pre["frame"]
                if _BEVQ.get("scene") != s_pre:
                    _BEVQ.clear(); _BEVQ["scene"] = s_pre
                try:
                    zp = np.load(os.path.join(args.root, s_pre,
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
                    if args.model in ("v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36", "v37", "v38", "v39", "v40", "v41", "v42", "v43", "v44", "v45", "v46", "v47", "v48", "v49", "v51", "v52", "v52r50", "v53", "v54", "v55", "v56", "v63b"):  # 3-slot queue
                        pbs, ths = [], []
                        for off in (2, 6, 14):
                            hb = _BEVQ.get(fi_cur - off)
                            rl = _rel(fi_cur - off) if fi_cur - off >= 0 \
                                else None
                            if _ZERO_HIST or hb is None or rl is None:
                                pbs.append(torch.zeros(
                                    1, 96, BEV_H, BEV_W, device="cuda"))
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
            # v32+ take the LiDAR pillar raster and the OSM SD-map prior as
            # OPTIONAL inputs: feeding zeros is bit-identical to not having
            # them, which is how the camera-only path stays honest. The demo
            # never passed either, so every video so far has been camera-only
            # even on scenes that have both on disk.
            if args.lidar_bev and f_pre.get("lidar_bev"):
                try:
                    lb = np.load(os.path.join(args.root, s_pre,
                                              f_pre["lidar_bev"]))["lb"]
                    lid_kw["lidar_bev"] = torch.from_numpy(
                        lb.astype(np.float32))[None].cuda()
                except Exception as e:
                    print(f"[lidar_bev] {type(e).__name__}", flush=True)
            if args.sdmap and f_pre.get("sdmap"):
                try:
                    sd = np.load(os.path.join(args.root, s_pre,
                                              f_pre["sdmap"]))["sd"]
                    lid_kw["sdmap"] = torch.from_numpy(
                        sd.astype(np.float32))[None].cuda()
                except Exception as e:
                    print(f"[sdmap] {type(e).__name__}", flush=True)
            if args.lidar and args.model == "v31" and f_pre.get("depth4"):
                try:
                    dz = np.load(os.path.join(args.root, s_pre,
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
                    "v37", "v38", "v39", "v40", "v41", "v42", "v43", "v44", "v45", "v46", "v47", "v48", "v49", "v51", "v52", "v52r50", "v53", "v54", "v55", "v56", "v63b"):
                if args.intent == "navsim":
                    # public_eval/navsim_to_meteor.py frames carry NAVSIM's driving_command
                    # [left, straight, right, unknown]; METEOR's one-hot is [straight, left, right]
                    oh, intent_lab = np.zeros(3, np.float32), None
                    nc = f_pre.get("nav_cmd")
                    if nc is not None and int(np.argmax(nc)) < 3 and max(nc) > 0:
                        k_ = int(np.argmax(nc)); oh[{0: 1, 1: 0, 2: 2}[k_]] = 1.0
                        intent_lab = ["LEFT", "STRAIGHT", "RIGHT"][k_]
                elif args.intent == "auto":
                    try:
                        pz = np.load(os.path.join(args.root, s_pre,
                                                  "ego_motion.npz"))["pose"]
                        oh, intent_lab = pseudo_intent(pz, f_pre["frame"], dist_hi=args.nav_dist if args.nav_dist > 0 else None, deg_th=args.nav_deg)
                    except Exception:
                        oh, intent_lab = np.zeros(3, np.float32), None
                else:
                    idx = {"straight": 0, "left": 1, "right": 2}[args.intent]
                    oh = np.zeros(3, np.float32); oh[idx] = 1.0
                    intent_lab = args.intent.upper()
                lid_kw["intent"] = torch.from_numpy(oh)[None].cuda()
            if getattr(m, "_int8_collect", None) is not None \
                    and m._int8_collect["on"]:
                # Calibration phase: collect activation scales over the first 8 frames,
                # then quantize with fixed scales (same flow as TensorRT)
                m._int8_collect.setdefault("n", 0)
                m._int8_collect["n"] += 1
                if m._int8_collect["n"] > 8:
                    m._int8_collect["on"] = False
                    print("[int8-sim] calibration done -> starting quantized inference", flush=True)
            with torch.no_grad(), torch.autocast("cuda", torch.float16):
                if args.model in ("v22", "v23", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36", "v37", "v38", "v39", "v40", "v41", "v42", "v43", "v44", "v45", "v46", "v47", "v48", "v49", "v51", "v52", "v52r50", "v53", "v54", "v55", "v56", "v63b"):
                    if trt_run is not None:
                        # engine path: the temporal queue is filled from the
                        # engine's raw_bev output, so the memory keeps working
                        # exactly as in the PyTorch path
                        _o = trt_run(imgs_m[None], K[None], T[None],
                                     v0_t if v0_t is not None
                                     else torch.zeros(1), pb, th)
                        out = _o[:-1]
                        _BEVQ[f_pre["frame"]] = _o[-1].detach().float()
                    else:
                        out = m(imgs_m[None].cuda(), K[None].cuda(),
                                T[None].cuda(),
                                v0_t.cuda() if v0_t is not None else None,
                                pb, th, **lid_kw)
                        _BEVQ[f_pre["frame"]] = m._last_bev.detach().float()
                    keep = 14 if args.model in ("v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36", "v37", "v38", "v39", "v40", "v41", "v42", "v43", "v44", "v45", "v46", "v47", "v48", "v49", "v51", "v52", "v52r50", "v53", "v54", "v55", "v56", "v63b") else 2
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
                        ga = getattr(refiner, "_acc", {})
                        def _on(k):
                            return ga.get(k, True)
                        rb = refiner.box is not None and _on("box")
                        re = refiner.e2e is not None and _on("e2e")
                        v0r = (v0_t.cuda() if v0_t is not None
                               else torch.zeros(1, device="cuda")) if re else None
                        fused = m._fused_bev.float() if re else None
                        r = refiner(
                            seg=out[0].float() if (refiner.seg is not None
                                                  and _on("seg")) else None,
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
                        if getattr(refiner, "pl", None) is not None \
                                and _on("pl") \
                                and len(out) >= 19 and out[18] is not None:
                            out[18] = refiner.pl(
                                out[18].float().clamp(-15, 15)).detach()
                        if getattr(refiner, "stat", None) is not None \
                                and _on("stat") and len(out) >= 11:
                            out[10] = refiner.stat(
                                out[10].float().clamp(-15, 15)).detach()
                    elif getattr(refiner, "_acc", {}).get("seg", True):
                        out[0] = refiner(out[0].float(), ctx).detach()
            seg, dlog = out[0], out[1]        # v13 returns (seg, depth, seg2d)
            if args.thin_bias:
                # The thin classes come out far too fat: measured on val with
                # r54, laneline covers 2.79x the GT area, stopline 2.40x,
                # road_edge 1.94x. The cause is the training objective, not the
                # features -- Tversky runs at beta 0.8, so a miss is punished
                # four times as hard as a false positive and the network learns
                # to paint wide. IoU hides it (a fatter line grows the union
                # about as fast as the intersection), which is how r54 set a
                # best-ever mIoU while getting visibly worse.
                #
                # Since it is a decision-boundary bias and not a localisation
                # error, subtracting a constant from those logits before the
                # argmax fixes it at no cost: at 0.5 the laneline area ratio
                # goes 2.79 -> 1.17 and road_edge 1.94 -> 0.91, for -0.4 % of
                # laneline IoU and -0.0002 mIoU.
                seg = seg.clone()
                for _c in (4, 5, 6):          # laneline, stopline, road_edge
                    # Per-class calibration (2026-08-18, unified with Orin): measured
                    # optimum from probe_seg_bias: lane 0.75 / stop 1.0 / edge 0.5. Applied
                    # only when --thin-bias is the default (0.5); explicit value keeps old behavior.
                    _bmap = {4: 0.75, 5: 1.0, 6: 0.5}
                    seg[:, _c] -= (_bmap.get(_c, args.thin_bias)
                                   if abs(args.thin_bias - 0.5) < 1e-9
                                   else args.thin_bias)
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
            if args.model in ("v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36", "v37", "v38", "v39", "v40", "v41", "v42", "v43", "v44", "v45", "v46", "v47", "v48", "v49", "v51", "v52", "v52r50", "v53", "v54", "v55", "v56", "v63b") and len(out) > 4:
                if args.model == "v30" and len(out) >= 18:
                    unk_boxes = m.decode_unknown(out[17].float())[0]
                else:
                    unk_boxes = []
                det_boxes = [d for d in m.decode_boxes(
                    out[3].float(), out[4].float(), thresh=args.box_thresh, topk=128)[0]
                    if d[1] > _veh_th(d, args) ] + unk_boxes
                if os.environ.get("METEOR_BOX_DUMP"):
                    # debug: raw decoded boxes within 20 m before any display-side processing
                    _tr_mods = [n for n, mm in m.named_modules() if mm.training]
                    print(f"[boxdump mode f{f_pre['frame']:03d}] training_modules={len(_tr_mods)} {_tr_mods[:6]} "
                          f"autocast={torch.is_autocast_enabled()} grad={torch.is_grad_enabled()}", flush=True)
                    print(f"[boxdump raw f{f_pre['frame']:03d}] n_out={len(out)} hm={tuple(out[3].shape)} "
                          + str([(int(d[0]), round(float(d[1]), 2), round(float(d[2]), 1), round(float(d[3]), 1),
                                  round(float(d[4]), 1), round(float(d[5]), 1)) for d in det_boxes
                                 if np.hypot(float(d[2]), float(d[3])) < 20]), flush=True)
                # --- display-side stabilization (2026-08-13) ----------------
                # 1) Display clip (fixed 2026-08-18): the fixed front-50/rear-28 window
                #    was a workaround for the light model (det supervision front 50 / rear 28);
                #    on the mainline full grid the teacher covers everything, yet boxes beyond
                #    28 m rear / 50 m front were dropped. Follow the real grid extent (+-2 m margin).
                import bevlane.model as _M
                _xr = _M.BEV_H * 0.2 - 80.0
                det_boxes = [d for d in det_boxes
                             if -(_xr - 2.0) <= d[2] <= 78.0]
                _bp = set(args.box_post.split("+"))
                main._bp_vel = None            # per-box (vx, vy, moving) from the tracker, aligned with det_boxes
                if not _bp & {"off", "none"}:
                    from bevlane.box_post import NMS_IOU_DEFAULT, Tracker, boxnms, raynms
                    if "raynms" in _bp:
                        det_boxes = raynms(det_boxes)
                    if "nms" in _bp:
                        det_boxes = boxnms(det_boxes, NMS_IOU_DEFAULT)
                    if "fpnms" in _bp:                  # footprint NMS for near-range duplicates (parked rows, large vehicles)
                        from bevlane.box_post import fpnms
                        det_boxes = [d for d in fpnms([d for d in det_boxes if len(d) == 7])] + [d for d in det_boxes if len(d) != 7]
                    if "track" in _bp:
                        if getattr(main, "_bp_scene", None) != scene:
                            main._bp_scene, main._bp_trk, main._bp_t = scene, Tracker(), None
                        _em = np.load(os.path.join(args.root, s_pre, "ego_motion.npz"))
                        if "stamp" in _em.files:
                            _t = float(_em["stamp"][f_pre["frame"]])
                        else:
                            # no timestamps (released demo scenes): frame interval from odometry step / speed
                            if getattr(main, "_bp_dt_scene", None) != scene:
                                _p, _v = _em["pose"], _em["v0"]
                                _d = np.hypot(*(_p[1:, :2] - _p[:-1, :2]).T)
                                _ok = _v[1:] > 3.0
                                main._bp_dt = float(np.median(_d[_ok] / _v[1:][_ok])) if _ok.sum() >= 5 else 0.1
                                main._bp_dt_scene = scene
                            _t = f_pre["frame"] * main._bp_dt
                        det_boxes = main._bp_trk.step(det_boxes, _em["pose"][f_pre["frame"]],
                                                      0.0 if main._bp_t is None else _t - main._bp_t)
                        main._bp_t = _t
                        main._bp_vel = list(main._bp_trk.out_vel)
                # 2) Temporal yaw smoothing: match boxes to the previous frame within 2.5 m,
                #    suppress 180-degree flips, then EMA (a=0.6). Directly damps the per-frame
                #    spinning of boxes with weak features (broadside / directly behind).
                _tr = getattr(main, "_yaw_tracks", None)
                if _bp & {"track", "none"}:
                    _tr = []                  # the tracker already smooths heading / raw boxes requested
                    main._yaw_scene = scene
                elif _tr is None or getattr(main, "_yaw_scene", None) != scene:
                    _tr = []
                    main._yaw_scene = scene
                _sm = []
                for d in det_boxes:
                    d = list(d)
                    best = None
                    for (px, py, pyaw) in _tr:
                        dd = (d[2] - px) ** 2 + (d[3] - py) ** 2
                        if dd < 6.25 and (best is None or dd < best[0]):
                            best = (dd, pyaw)
                    if best is not None and len(d) > 6:
                        py_ = best[1]
                        dy_ = (d[6] - py_ + np.pi) % (2 * np.pi) - np.pi
                        if abs(dy_) > np.pi / 2:      # flip suppression
                            d[6] = d[6] + (np.pi if dy_ < 0 else -np.pi)
                            dy_ = (d[6] - py_ + np.pi) % (2 * np.pi) - np.pi
                        d[6] = py_ + 0.4 * dy_        # EMA a=0.6
                    _sm.append(tuple(d))
                det_boxes = _sm
                main._yaw_tracks = [(d[2], d[3], d[6]) for d in det_boxes
                                    if len(d) > 6]
                if os.environ.get("METEOR_BOX_DUMP"):
                    print(f"[boxdump post f{f_pre['frame']:03d}] "
                          + str([(int(d[0]), round(float(d[1]), 2), round(float(d[2]), 1), round(float(d[3]), 1))
                                 for d in det_boxes if np.hypot(float(d[2]), float(d[3])) < 20]), flush=True)
            ego_modes = None
            if args.model in ("v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36", "v37", "v38", "v39", "v40", "v41", "v42", "v43", "v44", "v45", "v46", "v47", "v48", "v49", "v51", "v52", "v52r50", "v53", "v54", "v55", "v56", "v63b") and len(out) >= 8:
                _e = out[7][0].float().cpu().numpy()
                if args.path_smooth:      # cubic-through-origin re-fit of every mode
                    from bevlane.path_smooth import smooth_modes
                    _e = smooth_modes(_e)
                    out = list(out)
                    out[7] = torch.from_numpy(_e)[None]
                _pr = np.exp(_e[36:39]) / np.exp(_e[36:39]).sum()
                _k = int(_pr.argmax())
                if args.guard and len(out) >= 13:      # C1: risk-aware pick
                    _rm = out[12][0, 0].float().sigmoid().cpu().numpy()
                    _k, _, _rk = risk_pick(
                        [_e[j * 12:(j + 1) * 12] for j in range(3)], _pr, _rm)
                # Unified rendering (2026-08-18): apply the same selection rule as the
                # Orin renderer after risk_pick. (1) +0.35 for the previous frame's mode
                # (flicker suppression), (2) keep straight unless a turn mode beats it by
                # 1.0 (suppresses lunging in the no-command demo).
                _lg = np.log(np.maximum(_pr, 1e-9))
                _prev = getattr(main, "_prev_mode", None)
                if getattr(main, "_mode_scene", None) != scene:
                    _prev = None
                    main._mode_scene = scene
                if _prev is not None:
                    _lg[_prev] += 0.35
                _k2 = int(_lg.argmax())
                # straight preference retired: mode 0 is no longer the straight mode
                # (in v210 the argmax is mode 1 on most frames), so the rule pinned
                # every frame to mode 0 and slowed the path (more false stops)
                if os.environ.get("METEOR_STRAIGHT_PREF", "0") == "1" and \
                        _k2 != 0 and (_lg[_k2] - _lg[0]) < 1.0:
                    _k2 = 0
                if _k2 != _k:
                    _k = _k2
                main._prev_mode = _k
                ego_modes = [(_e[j * 12:(j + 1) * 12], float(_pr[j]), j == _k)
                             for j in range(3)]
                out = out[:7] + [torch.from_numpy(np.concatenate(
                    [_e[_k * 12:(_k + 1) * 12], _e[39:42]]))[None]] + out[8:]
            ego_pred = out[7][0].float().cpu().numpy() \
                if args.model in ("v18", "v19", "v20", "v21", "v22", "v23", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36", "v37", "v38", "v39", "v40", "v41", "v42", "v43", "v44", "v45", "v46", "v47", "v48", "v49", "v51", "v52", "v52r50", "v53", "v54", "v55", "v56", "v63b") and len(out) >= 8 else None
            traj_map = out[9][0].float().cpu() \
                if args.model in ("v21", "v22", "v23", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36", "v37", "v38", "v39", "v40", "v41", "v42", "v43", "v44", "v45", "v46", "v47", "v48", "v49", "v51", "v52", "v52r50", "v53", "v54", "v55", "v56", "v63b") and len(out) >= 10 else None
            stat_map = out[10][0, 0].float().cpu() \
                if args.model in ("v26", "v27", "v28", "v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36", "v37", "v38", "v39", "v40", "v41", "v42", "v43", "v44", "v45", "v46", "v47", "v48", "v49", "v51", "v52", "v52r50", "v53", "v54", "v55", "v56", "v63b") and len(out) >= 11 else None
            risk_map = out[12][0, 0].float().sigmoid().cpu().numpy() \
                if args.model in ("v28", "v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36", "v37", "v38", "v39", "v40", "v41", "v42", "v43", "v44", "v45", "v46", "v47", "v48", "v49", "v51", "v52", "v52r50", "v53", "v54", "v55", "v56", "v63b") and len(out) >= 13 else None
            tl_state = None
            if args.model in ("v27", "v28", "v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36", "v37", "v38", "v39", "v40", "v41", "v42", "v43", "v44", "v45", "v46", "v47", "v48", "v49", "v51", "v52", "v52r50", "v53", "v54", "v55", "v56", "v63b") and len(out) >= 12:
                p_tl = out[11][0].float().softmax(0)
                tl_state = (int(p_tl.argmax()), float(p_tl.max()))
            occ_pred = None
            if args.model in ("v20", "v21", "v22", "v23", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36", "v37", "v38", "v39", "v40", "v41", "v42", "v43", "v44", "v45", "v46", "v47", "v48", "v49", "v51", "v52", "v52r50", "v53", "v54", "v55", "v56", "v63b") and len(out) >= 9:
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
            if args.model in ("v17", "v18", "v19", "v20", "v21", "v22", "v23", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36", "v37", "v38", "v39", "v40", "v41", "v42", "v43", "v44", "v45", "v46", "v47", "v48", "v49", "v51", "v52", "v52r50", "v53", "v54", "v55", "v56", "v63b") and len(out) >= 7:
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
            if dlog.shape[2] == 1:
                # v50: the regression head already emits metres per pixel,
                # so there is no distribution to take an expectation over
                depth = dlog.float()[0, :, 0].cpu()
            else:
              dprob = dlog.float().softmax(2)[0]
              # top-mode expectation (argmax bin +-2, renormalised): the full
              # expectation mixes foreground/background modes at object
              # boundaries into phantom mid-range depths -> soft edges
              pk = dprob.argmax(1, keepdim=True)
              ar = torch.arange(dprob.shape[1], device=dprob.device
                                ).view(1, -1, 1, 1)
              pw = dprob * ((ar - pk).abs() <= 2)
              depth = ((pw.cpu() * dbins.view(1, -1, 1, 1)).sum(1)
                       / pw.sum(1).clamp(min=1e-6).cpu())   # [N,fh,fw]
            # --- 2D 'obs' unknown -> BEV via predicted depth (--unk2d) ---
            unk2d = []
            if args.unk2d and boxes2d is not None and depth is not None:
                fh, fw = depth.shape[-2:]
                for ci in range(len(CAMS)):
                    if ci >= len(boxes2d):
                        break
                    for cls2, sc2, cx2, cy2, w2, h2 in boxes2d[ci]:
                        if int(cls2) != 0 or sc2 < args.unk2d_thresh:
                            continue      # class 0 = 'obs' (unknown obstacle)
                        # depth sampled at the lower-centre of the box
                        # (closest to the ground-contact point)
                        u = int(np.clip(cx2 / 768.0 * fw, 0, fw - 1))
                        v = int(np.clip((cy2 + 0.25 * h2) / 432.0 * fh,
                                        0, fh - 1))
                        dm = float(depth[ci, v, u])
                        if not (1.5 < dm < 50.0):
                            continue
                        Kc = K_draw[ci].cpu().numpy()
                        Tce = T[ci].cpu().numpy()
                        pcam = np.array([(cx2 - Kc[0, 2]) / Kc[0, 0] * dm,
                                         (cy2 - Kc[1, 2]) / Kc[1, 1] * dm,
                                         dm])
                        pe = Tce[:3, :3].T @ (pcam - Tce[:3, 3])
                        if abs(pe[0]) > 60 or abs(pe[1]) > 25:
                            continue
                        unk2d.append((float(pe[0]), float(pe[1]),
                                      float(sc2)))
                # cross-camera dedupe: greedy 1.5 m clustering, keep max score
                unk2d.sort(key=lambda t: -t[2])
                kept = []
                for xe_, ye_, sc_ in unk2d:
                    if all((xe_ - a) ** 2 + (ye_ - b) ** 2 > 1.5 ** 2
                           for a, b, _ in kept):
                        kept.append((xe_, ye_, sc_))
                unk2d = kept
            s, f = ds.items[i]
            frame = np.zeros((VH, VW, 3), np.uint8)

            # --- RGB block (top, 2x4) and Depth block (below, same grid) ---
            pl_rgb = None
            if args.show_pl:
                pl_rgb = []
                for chn_ in CAMS:
                    p_ = f["imgs"].get(chn_)
                    pl_rgb.append(cv2.imread(os.path.join(args.root, s, p_))
                                  if p_ else None)
            pl_ras = None
            if args.show_pl and len(out) >= 19 and out[18] is not None:
                # raw pseudo-LiDAR logits -> the real raster's units
                # (log-count, max z, mean z, occupancy), occupancy-gated
                pl_ras = m.pl_activate(out[18][:1].float(),
                                       hard=True)[0].cpu().numpy()
            for k, chn in enumerate(CAM8):
                r, c = divmod(k, 4)
                x = c * cw
                p = f["imgs"].get(chn)
                if p:
                    img = cv2.resize(cv2.imread(os.path.join(args.root, s, p)),
                                     (cw, ch))
                    if seg2d_pred is not None:
                        sc = SEG2D_PAL[seg2d_pred[CAMS.index(chn)]]
                        sc = cv2.resize(sc, (cw, ch),
                                        interpolation=cv2.INTER_NEAREST)
                        img = cv2.addWeighted(img, 0.62, sc, 0.38, 0)
                    if chn in blank:
                        img = np.zeros((ch, cw, 3), np.uint8)   # blank tile
                    if "NARROW" in chn:
                        label(img, "NARROW", (0, 255, 0))
                    if det_boxes:
                        ci0 = CAMS.index(chn)
                        draw_boxes_on_rgb(img, det_boxes, K_draw[ci0].numpy(),
                                          T[ci0].numpy(), cw, ch)
                    if boxes2d is not None and not args.no_2d_boxes:
                        draw_boxes2d(img, boxes2d[CAMS.index(chn)], cw, ch)
                    elif args.unk2d_draw and boxes2d is not None:
                        # draw only unknown (class 0 'obs') boxes, in white (other 2D classes are not drawn)
                        sx_, sy_ = cw / 768.0, ch / 432.0
                        for cls2, sc2, cx2, cy2, w2, h2 in boxes2d[CAMS.index(chn)]:
                            if int(cls2) != 0 or sc2 < args.unk2d_thresh:
                                continue
                            cv2.rectangle(img, (int((cx2 - w2 / 2) * sx_), int((cy2 - h2 / 2) * sy_)),
                                          (int((cx2 + w2 / 2) * sx_), int((cy2 + h2 / 2) * sy_)), (255, 255, 255), 2)
                    if ego_pred is not None and chn == "CAM_FRONT_WIDE":
                        ci0 = CAMS.index(chn)
                        draw_path_ribbon(img, ego_pred, K_draw[ci0].numpy(),
                                         T[ci0].numpy(), cw, ch,
                                         reset=(i == 0 or f["frame"] == 0))
                        if os.environ.get("METEOR_DEMO_GT_PATH") == "1" and _F3.get("gt") is not None:   # GT path as a thin white line (visual comparison)
                            _draw_gt_polyline(img, _F3["gt"], K_draw[ci0].numpy(), T[ci0].numpy(), cw, ch)
                    frame[rgb_y0 + r * ch:rgb_y0 + (r + 1) * ch, x:x + cw] = img
                # depth directly under the same camera cell
                ci = CAMS.index(chn)
                if chn in blank:
                    if occ_pred is not None:      # OCC takes the free cell
                        dc = cv2.resize(cube_render(occ_pred, W=900, H=760),
                                        (cw, ch))
                        label(dc, "pred OCC voxel +-24m", (220, 220, 220))
                    else:
                        dc = np.zeros((ch, cw, 3), np.uint8)
                elif args.show_pl and chn in NARROW:
                    dc = None            # the tall PL view covers both cells
                else:
                    d = depth[ci].numpy()
                    dc = cv2.applyColorMap(
                        np.clip(d / 80 * 255, 0, 255).astype(np.uint8),
                        cv2.COLORMAP_TURBO)
                    dc = cv2.resize(dc, (cw, ch),
                                    interpolation=cv2.INTER_NEAREST)
                    label(dc, chn.split("CAM_")[-1], (255, 255, 255))
                if dc is not None:
                    frame[dep_y0 + r * ch:dep_y0 + (r + 1) * ch,
                          x:x + cw] = dc
            # Front-only layout (env METEOR_DEMO_FRONT=1, for Nano): redraw the front wide RGB and depth large and
            #   replace the frame with a 3-panel layout [front RGB / front depth | front half of BEV].
            _front_rgb = _front_dep = None
            if os.environ.get("METEOR_DEMO_FRONT") == "1" and f["imgs"].get("CAM_FRONT_WIDE"):
                FWb, FHb = 960, 540
                _chn = "CAM_FRONT_WIDE"; _ci = CAMS.index(_chn); _p = f["imgs"][_chn]
                _hp = os.path.join(os.environ.get("METEOR_HR_ROOT") or "", s, _p)
                _im = cv2.imread(_hp) if os.environ.get("METEOR_HR_ROOT") and os.path.isfile(_hp) else None
                if _im is None:
                    _im = cv2.imread(os.path.join(args.root, s, _p))
                _im = cv2.resize(_im, (FWb, FHb), interpolation=cv2.INTER_AREA)
                if seg2d_pred is not None:
                    _sc = cv2.resize(SEG2D_PAL[seg2d_pred[_ci]], (FWb, FHb), interpolation=cv2.INTER_NEAREST)
                    _im = cv2.addWeighted(_im, 0.70, _sc, 0.30, 0)
                if det_boxes:
                    draw_boxes_on_rgb(_im, det_boxes, K_draw[_ci].numpy(), T[_ci].numpy(), FWb, FHb)
                if ego_pred is not None:
                    draw_path_ribbon(_im, ego_pred, K_draw[_ci].numpy(), T[_ci].numpy(), FWb, FHb,
                                     reset=(i == 0 or f["frame"] == 0))
                    if os.environ.get("METEOR_DEMO_GT_PATH") == "1":          # predicted points (green) vs GT (white line + dots), 1.5 s / 3 s errors and current accel
                        _draw_wp_dots(_im, ego_pred[:12], K_draw[_ci].numpy(), T[_ci].numpy(), FWb, FHb, (60, 255, 120))
                        _txt = f"acc {float(_F3.get('acc', 0.0)):+.2f} m/s2"
                        if _F3.get("gt") is not None:
                            _draw_gt_polyline(_im, _F3["gt"], K_draw[_ci].numpy(), T[_ci].numpy(), FWb, FHb)
                            _g6 = np.asarray(_F3["gt"], np.float32).reshape(6, 2); _p6 = np.asarray(ego_pred[:12], np.float32).reshape(6, 2)
                            _txt += f"   path err vs GT (white)  1.5 s: {float(np.hypot(*(_p6[2] - _g6[2]))):.2f} m   3 s: {float(np.hypot(*(_p6[5] - _g6[5]))):.2f} m"
                        cv2.putText(_im, _txt, (14, FHb - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
                        cv2.putText(_im, _txt, (14, FHb - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                _front_rgb = _im
                _d = depth[_ci].numpy()
                _front_dep = cv2.resize(cv2.applyColorMap(np.clip(_d / 80 * 255, 0, 255).astype(np.uint8),
                                                          cv2.COLORMAP_TURBO), (FWb, FHb), interpolation=cv2.INTER_LINEAR)
            if args.show_pl and pl_ras is not None:
                # column 3 of the depth grid = FRONT_NARROW + BACK_NARROW,
                # used as one tall cell so the 3-D cloud is actually legible
                px0 = 3 * cw
                pv = pl_render(pl_ras, W=cw, H=2 * ch, rgb=pl_rgb,
                               K=K_draw.numpy(), T=T.numpy())
                label(pv, "pseudo-LiDAR 3D (image-coloured)", (200, 230, 255))
                frame[dep_y0:dep_y0 + 2 * ch, px0:px0 + cw] = pv
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
            draw_ego_and_grid.xr_m = min(60.0, BEV_XR)
            bev = draw_ego_and_grid(DEMO_PALETTE[pc][:, :, ::-1], BH2, BW2,
                                    xh_m=60.0, yh_m=25.0)
            # The BEV panel spans +60 m ahead to -min(60, BEV_XR) behind.
            # Every metre->pixel conversion below used the literal 120 m span
            # of the symmetric grid; on a rear-truncated grid that drew the
            # E2E path (and every overlay) shifted toward the panel centre.
            _XB = min(60.0, BEV_XR)
            _SPAN = 60.0 + _XB
            if risk_map is not None and not args.no_risk:
                # overlay predicted risk (+-40 x +-25 m) on the BEV panel
                rm = cv2.resize(risk_map, (BW2, int(BH2 * (40.0 + _XB) / _SPAN)),
                                interpolation=cv2.INTER_LINEAR)
                y0r = int(BH2 * (60.0 - 40.0) / _SPAN)
                sub = bev[y0r:y0r + rm.shape[0]]
                heat = cv2.applyColorMap((np.clip(rm, 0, 1) * 255
                                          ).astype(np.uint8),
                                         cv2.COLORMAP_TURBO)
                a = (np.clip(rm, 0, 1) * 0.55)[..., None]
                bev[y0r:y0r + rm.shape[0]] = (sub * (1 - a) + heat * a
                                              ).astype(np.uint8)
            if det_boxes:
                sy2 = BH2 / _SPAN            # px per metre (2*60m vertical)
                sx2 = BW2 / 50.0             # px per metre (2*25m lateral)
                for _bi, (cls, sc, xe, ye, l, w, yaw) in enumerate(det_boxes):
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
                    # predicted 3 s agent future (pedestrian/VRU paths are
                    # suppressed by default: too jittery on screen for now)
                    if traj_map is not None and not args.no_agent_traj and (int(cls) != 1
                                                 or args.show_ped_path):
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
                            _vel = getattr(main, "_bp_vel", None)
                            if _vel is not None and _bi < len(_vel):
                                # the forecast head points nearly every car along the traffic flow (so oncoming
                                # cars are often reversed); the track's displacement decides
                                from bevlane.box_post import orient_traj
                                wps = orient_traj(wps, _vel[_bi])
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
            if args.unk2d and unk2d:
                sy2 = BH2 / _SPAN
                sx2 = BW2 / 50.0
                sp = max(3, int(0.6 * sx2))       # fixed 1.2 m marker
                for xe_, ye_, sc_ in unk2d:
                    cxp = int((25.0 - ye_) * sx2)
                    cyp = int((60.0 - xe_) * sy2)
                    dia = np.array([[cxp, cyp - sp], [cxp + sp, cyp],
                                    [cxp, cyp + sp], [cxp - sp, cyp]],
                                   np.int32)
                    cv2.polylines(bev, [dia.reshape(-1, 1, 2)], True,
                                  (255, 255, 255), 2, cv2.LINE_AA)
            if ego_pred is not None and ego_modes is not None:
                sy2 = BH2 / _SPAN
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
                sy2 = BH2 / _SPAN
                sx2 = BW2 / 50.0
                pts = [(int(25.0 * sx2), int(60.0 * sy2))]      # ego origin
                for k in range(6):
                    xe, ye = ego_pred[2 * k], ego_pred[2 * k + 1]
                    if abs(xe) > 60 or abs(ye) > 25:
                        break
                    pts.append((int((25.0 - ye) * sx2), int((60.0 - xe) * sy2)))
                # stationary: waypoints collapse onto ego -> show it explicitly
                trav = float(np.hypot(ego_pred[10], ego_pred[11]))
                if _EGO_DUMP:
                    main._ego_rows = getattr(main, "_ego_rows", [])
                    main._ego_rows.append((scene, int(f_pre["frame"]), np.asarray(ego_pred[:12], np.float32), float(v0_t) if v0_t is not None else float("nan")))
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
                    sy2 = BH2 / _SPAN
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
                        if args.model in ("v18", "v19", "v20", "v21", "v22", "v23", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36", "v37", "v38", "v39", "v40", "v41", "v42", "v43", "v44", "v45", "v46", "v47", "v48", "v49", "v51", "v52", "v52r50", "v53", "v54", "v55", "v56", "v63b") else
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
            if _front_rgb is not None:
                # BEV from 60 m ahead to 6 m behind the ego (front sensing range) fills the right half
                _ego_row = int(BH2 * 60.0 / _SPAN); _bf = bev[:min(BH2, _ego_row + int(BH2 * 6.0 / _SPAN))]
                _bh = VH; _bw = int(_bh * _bf.shape[1] / _bf.shape[0])
                if _bw > VW - 960:
                    _bw = VW - 960; _bh = int(_bw * _bf.shape[0] / _bf.shape[1])
                frame = np.zeros((VH, VW, 3), np.uint8)
                frame[0:540, 0:960] = _front_rgb; frame[540:1080, 0:960] = _front_dep
                _bx = 960 + (VW - 960 - _bw) // 2
                frame[(VH - _bh) // 2:(VH - _bh) // 2 + _bh, _bx:_bx + _bw] = cv2.resize(_bf, (_bw, _bh), interpolation=cv2.INTER_AREA)
                for _t, _y in (("Front camera + 3D boxes + ego path", 30), ("Predicted depth (0-80 m)", 570)):
                    cv2.putText(frame, _t, (14, _y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
                    cv2.putText(frame, _t, (14, _y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                if args.frame_stride <= 1 or i % args.frame_stride == 0:
                    vw.write(frame); n += 1
                continue

            if occ_pred is not None and not blank:
                iso = cube_render(occ_pred, W=900, H=760)
                iso = cv2.resize(iso, (426, 360))
                oy0 = VH - 368
                frame[oy0:oy0 + 360, 8:8 + 426] = iso
                cv2.putText(frame, "pred OCC voxel grid +-24m (bldg hidden)",
                            (8, oy0 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1,
                            cv2.LINE_AA)
            _mods = ([] + (["LIDAR"] if args.lidar_bev else [])
                     + (["SDMAP"] if args.sdmap else []))
            mode_tag = ("+".join(_mods) + " ON") if _mods \
                else ("LIDAR ON" if args.lidar else "camera-only")
            if args.seg_fuse:
                mode_tag += "  |  seg-EMA<=45m"
            cv2.putText(frame, f"{scene.split('+0900_')[-1]}  f{f['frame']:03d}  |  "
                        f"{len(CAMS) - len(blank)}/{len(CAMS)}-cam -> Depth "
                        f"+ BEV (+-25x+-60m)  |  no GT  |  "
                        f"{mode_tag}",
                        (10, VH - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                        (0, 255, 255) if args.lidar else (0, 255, 0), 2,
                        cv2.LINE_AA)
            # The temporal queue must still see every frame, so thinning
            # happens at WRITE time, not by skipping the forward pass.
            if args.frame_stride <= 1 or i % args.frame_stride == 0:
                vw.write(frame)
                n += 1
                if _dump is not None:
                    _lv = locals()
                    try:
                        _pose = np.load(os.path.join(args.root, s_pre, "ego_motion.npz"))["pose"][f_pre["frame"]]
                    except Exception:
                        _pose = None
                    from bevlane.box_post import orient_traj
                    _trk = getattr(main, "_bp_trk", None) if "track" in args.box_post.split("+") else None
                    _dump.write(scene, f_pre["frame"], _pose,
                                None if v0_t is None else float(v0_t[0]), _lv.get("intent_lab"),
                                _lv.get("ego_pred"), _lv.get("ego_modes"), _lv.get("tl_state"), _lv.get("guard"),
                                det_boxes, None if _trk is None else _trk.out_info,
                                getattr(main, "_bp_vel", None) if _trk is not None else None,
                                _lv.get("traj_map"), _lv.get("stat_map"), _lv.get("pred"), _lv.get("occ_pred"),
                                orient=orient_traj, unk=(_lv.get("unk2d") if args.unk2d else None))
        print("scene", scene, n, flush=True)
    vw.release()
    if _dump is not None:
        _dump.close()
    if _EGO_DUMP and getattr(main, "_ego_rows", None):
        np.save(_EGO_DUMP, np.array(main._ego_rows, dtype=object), allow_pickle=True)
    if torch.cuda.is_available():
        print(f"[mem] peak GPU allocated {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB "
              f"(reserved {torch.cuda.max_memory_reserved() / 2**20:.0f} MiB)",
              flush=True)
    if n == 0:
        # a shard can legitimately be empty: 22 of the 279 test scenes ship an
        # empty manifest (unconverted). ffmpeg on a 0-frame file exits 234 and
        # took the whole parallel render down with it.
        print(f"[demo] no frames rendered -> not encoding {args.out}",
              flush=True)
        os.remove(raw) if os.path.exists(raw) else None
        return
    subprocess.run(["ffmpeg", "-y", "-i", raw, "-c:v", "libx264", "-crf", "23",
                    "-pix_fmt", "yuv420p", args.out], check=True, capture_output=True)
    os.remove(raw)
    print("done", n, args.out, flush=True)


if __name__ == "__main__":
    main()

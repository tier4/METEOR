#!/usr/bin/env python3
"""GT demo for the lane-departure RECOVERY augmentation (idea 1, v45).

For each frame of a scene, shows side by side:
  LEFT  : original ego frame  -- GT BEV raster + the recorded future (3 s)
  RIGHT : PERTURBED ego frame (lateral offset dy + yaw dpsi) -- the same GT
          re-expressed from the departed viewpoint; the recorded future,
          re-expressed and spline-connected from the new origin, becomes a
          RECOVERY-to-lane target "for free".

No model, no new data: everything is a geometric transform of existing
auto-labels. The same transform applied to T_cam_ego at train time gives the
camera-feature BEV from the departed viewpoint (depth-gated IPM re-projects
geometrically), so this is exactly what the planner would be trained on.

Usage:
  python3 bevlane/demo_recovery_gt.py --scenes <SCENE ...> \
      --out out/demo_recovery_gt.mp4
"""
import argparse
import os
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from autolabel_bev import PALETTE  # noqa: E402
from bevlane.postproc import crop_bev, draw_ego_and_grid  # noqa: E402

RES = 0.2                       # m/cell, raster 800x500 (+-80 x +-50)
PAL = np.zeros((12, 3), np.uint8)
PAL[:len(PALETTE)] = PALETTE


def perturb_raster(gt, dy, dpsi):
    """re-render the GT raster as seen from an ego displaced (0, dy) with
    heading dpsi. p_orig = R(dpsi) @ p_new + (0, dy)."""
    H, W = gt.shape
    # cell centres in the NEW frame
    xs = 80.0 - (np.arange(H) + 0.5) * RES
    ys = 50.0 - (np.arange(W) + 0.5) * RES
    Xn, Yn = np.meshgrid(xs, ys, indexing="ij")
    c, s = np.cos(dpsi), np.sin(dpsi)
    Xo = c * Xn - s * Yn
    Yo = s * Xn + c * Yn + dy
    r = ((80.0 - Xo) / RES - 0.5).round().astype(np.int32)
    q = ((50.0 - Yo) / RES - 0.5).round().astype(np.int32)
    ok = (r >= 0) & (r < H) & (q >= 0) & (q < W)
    out = np.zeros_like(gt)
    out[ok] = gt[r[ok], q[ok]]
    return out


def transform_wp(wp, dy, dpsi):
    """future waypoints [K,2] (original ego frame) -> perturbed frame.
    p_new = R(-dpsi) @ (p_orig - (0, dy))."""
    c, s = np.cos(dpsi), np.sin(dpsi)
    x = wp[:, 0]; y = wp[:, 1] - dy
    return np.stack([c * x + s * y, -s * x + c * y], 1)


def recovery_target(wp_t, v0=0.0):
    # speed-dependent rejoin horizon: at highway speed take 2 s
    # (soft, lateral-accel-friendly) instead of 1.5 s
    rejoin_k = 4 if v0 > 15.0 else 3
    """smooth spline from the perturbed origin (heading straight ahead in
    the new frame) that rejoins the transformed trajectory at waypoint
    rejoin_k -- removes the t=0 kink, exactly what training would use."""
    P0 = np.zeros(2); T0 = np.array([max(wp_t[0, 0], 2.0), 0.0])
    P1 = wp_t[rejoin_k]
    if rejoin_k + 1 < len(wp_t):
        T1 = wp_t[rejoin_k + 1] - wp_t[rejoin_k - 1]
    else:
        T1 = wp_t[rejoin_k] - wp_t[rejoin_k - 1]
    out = wp_t.copy()
    for i in range(rejoin_k):
        t = (i + 1) / (rejoin_k + 1)
        h00 = 2*t**3 - 3*t**2 + 1; h10 = t**3 - 2*t**2 + t
        h01 = -2*t**3 + 3*t**2;    h11 = t**3 - t**2
        out[i] = h00 * P0 + h10 * T0 + h01 * P1 + h11 * T1
    return out


def ego_centerline(pts, n):
    """lanegraph polylines are lane BOUNDARIES; the ego-lane centerline is
    the midline between the nearest left and right laneline. Returns
    [K,2] on a 1 m forward grid or None."""
    xg = np.arange(0.0, 46.0, 1.0)
    offs = []
    for i in range(int(n)):
        P = pts[i].astype(np.float64)
        P = P[np.isfinite(P).all(1)]
        if len(P) < 4:
            continue
        o = np.argsort(P[:, 0])
        P = P[o]
        if P[-1, 0] - P[0, 0] < 8 or P[-1, 0] < 15:
            continue
        y = np.interp(xg, P[:, 0], P[:, 1],
                      left=np.nan, right=np.nan)
        y[(xg < P[0, 0]) | (xg > P[-1, 0])] = np.nan
        offs.append(y)
    left = right = None
    bl = br = 1e9
    for y in offs:
        near = y[:16]
        if np.isnan(near).mean() > 0.6:
            continue
        med = np.nanmedian(near)
        if 0.3 < med < 3.2 and med < bl:
            left, bl = y, med
        if -3.2 < med < -0.3 and -med < br:
            right, br = y, -med
    if left is None or right is None or not (2.0 < bl + br < 6.0):
        return None
    yc = (left + right) / 2.0
    m = ~np.isnan(yc)
    if m.sum() < 12:
        return None
    return np.stack([xg[m], yc[m]], 1)


def pursuit_target(cl_t, v0, K=6, dt=0.5):
    """idea 2: rejoin the (perturbed-frame) centerline with a speed-aware
    lookahead, then FOLLOW it -- curvature of the lane is baked into the
    target, so a curve-lag state gets a curve-aware catch-up path."""
    seg = np.linalg.norm(np.diff(cl_t, axis=0), axis=1)
    t = np.concatenate([[0], np.cumsum(seg)])
    L = float(np.clip(1.2 * v0, 8.0, 30.0))     # lookahead [m]
    # arc position of the closest point to origin (projection start)
    i0 = int(np.argmin(np.hypot(cl_t[:, 0], cl_t[:, 1])))
    s0 = t[i0]
    def at(a):
        return np.array([np.interp(a, t, cl_t[:, 0]),
                         np.interp(a, t, cl_t[:, 1])])
    P1 = at(s0 + L)
    d1 = (at(s0 + L + 1.0) - at(s0 + L - 1.0)); d1 /= max(np.linalg.norm(d1), 1e-6)
    step = max(v0 * dt, 1.0)
    kj = max(1, min(K - 1, int(round(L / step))))
    out = np.zeros((K, 2))
    P0 = np.zeros(2); T0 = np.array([max(step, 2.0), 0.0]); T1 = d1 * step * 2
    for i in range(kj):                          # hermite: origin -> rejoin
        u = (i + 1) / (kj + 1)
        h00 = 2*u**3-3*u**2+1; h10 = u**3-2*u**2+u
        h01 = -2*u**3+3*u**2;  h11 = u**3-u**2
        out[i] = h00*P0 + h10*T0 + h01*P1 + h11*T1
    for i in range(kj, K):                       # then follow the lane
        out[i] = at(s0 + L + step * (i - kj + 1))
    return out


def draw_panel(gt, wps, title, sub, path_col, dy=None, centerline=None):
    pc = crop_bev(gt, xh_m=60.0, yh_m=25.0)
    BH, BW = 900, int(900 * pc.shape[1] / pc.shape[0])
    bev = draw_ego_and_grid(PAL[pc][:, :, ::-1].copy(), BH, BW,
                            xh_m=60.0, yh_m=25.0)
    sx, sy = BW / 50.0, BH / 120.0
    if centerline is not None:
        cpts = [(int((25.0 - y) * sx), int((60.0 - x) * sy))
                for x, y in centerline if abs(x) <= 60 and abs(y) <= 25]
        if len(cpts) > 1:
            cv2.polylines(bev, [np.array(cpts, np.int32).reshape(-1, 1, 2)],
                          False, (255, 230, 80), 2, cv2.LINE_AA)
    pts = [(int(25.0 * sx), int(60.0 * sy))]
    for x, y in wps:
        if abs(x) > 60 or abs(y) > 25:
            break
        pts.append((int((25.0 - y) * sx), int((60.0 - x) * sy)))
    if len(pts) > 1:
        cv2.polylines(bev, [np.array(pts, np.int32).reshape(-1, 1, 2)],
                      False, path_col, 3, cv2.LINE_AA)
        cv2.circle(bev, pts[-1], 6, path_col, -1)
    if dy is not None:                     # true lane position marker
        gx = int((25.0 - (-dy)) * sx)
        cv2.line(bev, (gx, int(55.0 * sy)), (gx, int(65.0 * sy)),
                 (0, 200, 255), 2)
    cv2.putText(bev, title, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(bev, sub, (10, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (80, 220, 255), 1, cv2.LINE_AA)
    return bev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--out", default="out/demo_recovery_gt.mp4")
    ap.add_argument("--root", default="out/bevlane")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--min-v0", type=float, default=0.0,
                    help="skip frames slower than this [m/s]")
    ap.add_argument("--mode", default="record",
                    choices=["record", "pursuit"],
                    help="recovery target: transformed recorded future "
                         "(idea 1) or lane-centerline pursuit (idea 2)")
    args = ap.parse_args()

    import json
    vw = None
    rng = np.random.default_rng(45)
    n = 0
    for scene in args.scenes:
        man = json.load(open(os.path.join(args.root, scene,
                                          "manifest.json")))
        eg = np.load(os.path.join(args.root, scene, "ego_motion.npz"))
        wp_all, valid = eg["wp"], eg["valid"]
        v0_all = eg["v0"]
        lg = (np.load(os.path.join(args.root, scene, "lanegraph.npz"))
              if args.mode == "pursuit" else None)
        dy = dpsi = 0.0
        for f in man["frames"]:
            fi = f["frame"]
            if fi >= len(wp_all) or valid[fi] < 0.5 or not f.get("gt"):
                continue
            if v0_all[fi] < args.min_v0:
                continue
            if n % 30 == 0 or dy == 0.0:   # new departure every 3 s
                dy = float(rng.uniform(0.5, 1.5)) * rng.choice([-1, 1])
                dpsi = float(rng.uniform(-8, 8)) * np.pi / 180
            gt = cv2.imread(os.path.join(args.root, scene, f["gt"]),
                            cv2.IMREAD_GRAYSCALE)
            if gt is None:
                continue
            wp = wp_all[fi].reshape(6, 2).astype(np.float64)
            pass
            cl0 = None
            if lg is not None and fi < len(lg["n"]):
                cl0 = ego_centerline(lg["pts"][fi], lg["n"][fi])
            left = draw_panel(gt, wp, "ORIGINAL (recorded)",
                              f"GT BEV + driven future 3s | "
                              f"{v0_all[fi]*3.6:.0f} km/h", (60, 255, 120),
                              centerline=cl0)
            gt_p = perturb_raster(gt, dy, dpsi)
            wp_t = transform_wp(wp, dy, dpsi)
            cl = cl_t = None
            if lg is not None and fi < len(lg["n"]):
                cl = ego_centerline(lg["pts"][fi], lg["n"][fi])
            if args.mode == "pursuit" and cl is None:
                continue                          # no usable centerline
            if cl is not None:
                cl_t = transform_wp(cl, dy, dpsi)
            if args.mode == "pursuit":
                rec = pursuit_target(cl_t, float(v0_all[fi]))
                tag = "PURSUIT(centerline) RECOVERY"
            else:
                rec = recovery_target(wp_t, float(v0_all[fi]))
                tag = "RECOVERY target"
            right = draw_panel(
                gt_p, rec, "PERTURBED = departed viewpoint",
                f"dy={dy:+.2f}m dpsi={np.degrees(dpsi):+.1f}deg -> " + tag,
                (0, 80, 255), dy=dy, centerline=cl_t)
            frame = np.hstack([left, np.full((left.shape[0], 8, 3), 60,
                                             np.uint8), right])
            frame = cv2.resize(frame, (1280, 720))
            if vw is None:
                vw = cv2.VideoWriter(args.out.replace(".mp4", "_raw.mp4"),
                                     cv2.VideoWriter_fourcc(*"mp4v"),
                                     args.fps, (1280, 720))
            vw.write(frame)
            n += 1
        print(f"{scene}: total {n} frames", flush=True)
    vw.release()
    subprocess.run(["ffmpeg", "-y", "-i", args.out.replace(".mp4", "_raw.mp4"),
                    "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p",
                    args.out], check=True, capture_output=True)
    os.remove(args.out.replace(".mp4", "_raw.mp4"))
    print(f"done -> {args.out}", flush=True)


if __name__ == "__main__":
    main()

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


def recovery_target(wp_t, rejoin_k=3):
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


def draw_panel(gt, wps, title, sub, path_col, dy=None):
    pc = crop_bev(gt, xh_m=60.0, yh_m=25.0)
    BH, BW = 900, int(900 * pc.shape[1] / pc.shape[0])
    bev = draw_ego_and_grid(PAL[pc][:, :, ::-1].copy(), BH, BW,
                            xh_m=60.0, yh_m=25.0)
    sx, sy = BW / 50.0, BH / 120.0
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
        dy = dpsi = 0.0
        for f in man["frames"]:
            fi = f["frame"]
            if fi >= len(wp_all) or valid[fi] < 0.5 or not f.get("gt"):
                continue
            if n % 30 == 0:                # new departure every 3 s
                dy = float(rng.uniform(0.5, 1.5)) * rng.choice([-1, 1])
                dpsi = float(rng.uniform(-8, 8)) * np.pi / 180
            gt = cv2.imread(os.path.join(args.root, scene, f["gt"]),
                            cv2.IMREAD_GRAYSCALE)
            if gt is None:
                continue
            wp = wp_all[fi].reshape(6, 2).astype(np.float64)
            left = draw_panel(gt, wp, "ORIGINAL (recorded)",
                              "GT BEV + driven future 3s", (60, 255, 120))
            gt_p = perturb_raster(gt, dy, dpsi)
            wp_t = transform_wp(wp, dy, dpsi)
            rec = recovery_target(wp_t)
            right = draw_panel(
                gt_p, rec, "PERTURBED = departed viewpoint",
                f"dy={dy:+.2f}m dpsi={np.degrees(dpsi):+.1f}deg -> "
                "RECOVERY target", (0, 80, 255), dy=dy)
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

"""Yaw-bias fix: rotate existing gt_cons about the ego origin -> gt_cons_yf, and
wp by the same rotation -> ego_motion_yf.npz. Existing files untouched (new keys only).

The rotation direction is switchable via --sign (+1/-1) and fixed empirically (the
direction that kills the range-proportional component of the GT-box lane-center
offset). Rasters and wp share the same corrective rotation.
"""
import argparse
import json
import os

import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--list", required=True)
ap.add_argument("--bias", required=True, help="JSON from estimate_yaw_bias.py")
ap.add_argument("--sign", type=float, default=1.0)
ap.add_argument("--gt-key", default="gt_cons")
ap.add_argument("--out-key", default="gt_cons_yf")
ap.add_argument("--ledger", default="out/yawfix_created.txt")
a = ap.parse_args()

bias = json.load(open(a.bias))
scenes = [l.strip() for l in open(a.list) if l.strip()]
led = open(a.ledger, "a")
n_sc = n_png = 0
for s in scenes:
    if s not in bias:
        continue
    b_deg = a.sign * bias[s]["bias_deg"]
    src = os.path.join(a.root, s, a.gt_key)
    dst = os.path.join(a.root, s, a.out_key)
    if not os.path.isdir(src):
        continue
    os.makedirs(dst, exist_ok=True)
    # Rotation about the ego origin (col=250, row=400). angle is CCW degrees in the
    # cv2 image plane (x=col, y=row). Direction correctness is verified empirically by the caller.
    M = cv2.getRotationMatrix2D((250.0, 400.0), b_deg, 1.0)
    for f in sorted(os.listdir(src)):
        if not f.endswith(".png"):
            continue
        g = cv2.imread(os.path.join(src, f), 0)
        if g is None:
            continue
        r = cv2.warpAffine(g, M, (g.shape[1], g.shape[0]),
                           flags=cv2.INTER_NEAREST,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=255)
        cv2.imwrite(os.path.join(dst, f), r)
        n_png += 1
    led.write(dst + "\n")
    # Same rotation for wp (point set in ego coords). Apply the same corrective rotation as the rasters.
    # The raster moves pixel values, so when the content rotates by +phi, rotate the points by +phi too.
    emo_p = os.path.join(a.root, s, "ego_motion.npz")
    if os.path.exists(emo_p):
        try:
            z = dict(np.load(emo_p))
            wp = z["wp"].copy()                       # [T,6,2] (x,y)
            # +b_deg CCW in the cv2 image plane (col~-y, row~-x) equals a +b_deg rotation
            # in the ego (x,y) plane too (flipping both axes preserves rotation direction).
            th = np.radians(b_deg)
            c0, s0 = np.cos(th), np.sin(th)
            x, y = wp[..., 0].copy(), wp[..., 1].copy()
            z["wp"] = np.stack([c0 * x - s0 * y, s0 * x + c0 * y],
                               -1).astype(wp.dtype)
            np.savez_compressed(os.path.join(a.root, s,
                                             "ego_motion_yf.npz"), **z)
            led.write(os.path.join(a.root, s, "ego_motion_yf.npz") + "\n")
        except Exception as e:
            print(f"[warn] {s} wp failed: {e}")
    n_sc += 1
led.close()
print(f"done: {n_sc} scenes / {n_png} rasters (sign={a.sign})")
print("APPLY_YAW_FIX_DONE")

"""Signed lateral error of 3D BBoxes (+y=left) per range band.

Quantifies the holdout-video impression that other vehicles beyond 20 m hug the
left lane line. Lateral shift growing with range = rotational bias (calib yaw /
far bias of the det head); constant shift = parallel offset.
Only predictions matched to GT boxes are used (avoids recall contamination; paired-comparison rule).
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bevlane.dataset import BevLaneDataset                        # noqa: E402
from bevlane.model import MODELS                                  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--model", default="v52")
ap.add_argument("--list", default=None, help="scene-name list file")
ap.add_argument("--scenes", nargs="*", default=None)
ap.add_argument("--root", default="out/bevlane")
ap.add_argument("--frames", type=int, default=200)
ap.add_argument("--thresh", type=float, default=0.25)
ap.add_argument("--cls-max", type=float, default=1.5,
                help="only classes below this (default: vehicles only)")
ap.add_argument("--tag", default="")
a = ap.parse_args()

scenes = a.scenes or [l.strip() for l in open(a.list) if l.strip()]
ds = BevLaneDataset(a.root, scenes, gt_key="gt_cons", with_boxdet=True,
                    max_per_scene=8, n_cams=8, trim_start=3, trim_end=10)
m = MODELS[a.model](n_seg=21).cuda().eval()
sd = torch.load(a.ckpt, map_location="cpu")
sd = {k.replace("module.", ""): v for k, v in sd.get("model", sd).items()}
cur = m.state_dict()
m.load_state_dict({k: v for k, v in sd.items()
                   if k in cur and cur[k].shape == v.shape}, strict=False)

BANDS = [("front 0-10m", 0, 10, 1), ("front 10-20m", 10, 20, 1),
         ("front 20-30m", 20, 30, 1), ("front 30-40m", 30, 40, 1),
         ("front 40-60m", 40, 60, 1),
         ("rear 0-20m", 0, 20, -1), ("rear 20-40m", 20, 40, -1)]
dy_s = {b[0]: [] for b in BANDS}       # signed lateral error (+=left)
dx_s = {b[0]: [] for b in BANDS}       # signed longitudinal error (+=farther)
step = max(1, len(ds) // a.frames)
done = 0
for i in range(0, len(ds), step):
    b = ds[i]
    if b is None:
        continue
    with torch.no_grad(), torch.autocast("cuda", torch.float16):
        out = m(b[0][None].cuda(), b[1][None].cuda(), b[2][None].cuda())
    dets = m.decode_boxes(out[3].float().cpu(), out[4].float().cpu(),
                          thresh=a.thresh)[0]
    pred = [(float(d[2]), float(d[3])) for d in dets
            if float(d[0]) < a.cls_max]
    bx, nb = b[4], int(b[5])
    for k in range(max(nb, 0)):
        cls, xe, ye, ln, wd, yw = [float(v) for v in bx[k][:6]]
        if ln <= 0 or cls >= a.cls_max:
            continue
        r = (xe * xe + ye * ye) ** 0.5
        best = None
        for px, py in pred:
            d2 = (xe - px) ** 2 + (ye - py) ** 2
            if d2 < 9.0 and (best is None or d2 < best[0]):
                best = (d2, px, py)
        if best is None:
            continue
        for nm, lo, hi, sgn in BANDS:
            if lo <= r < hi and (xe > 0) == (sgn > 0):
                dy_s[nm].append(best[2] - ye)
                dx_s[nm].append((best[1] - xe) * (1 if xe > 0 else -1))
    done += 1
    if done >= a.frames:
        break

print(f"\n=== {a.tag or a.ckpt} ({done} frames) signed error (+y=left) ===")
print("band        n    lateral(y) mean±SD  range mean±SD")
for nm, *_ in BANDS:
    e, d = np.array(dy_s[nm]), np.array(dx_s[nm])
    if len(e) >= 3:
        print(f"  {nm:<9} {len(e):4d}  {e.mean():+6.3f}±{e.std():5.3f} m"
              f"   {d.mean():+6.3f}±{d.std():5.3f} m")
print("PROBE_LAT_BAND_DONE")

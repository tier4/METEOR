"""Measure the pred-GT offset of the left/right road edges per range band.

Quantifies the holdout impression that the predicted road spreads left and grows an
extra lane. Per row, take the leftmost/rightmost road column and accumulate pred-GT
per side (+=left). A large + on the left edge alone confirms road overflow to the left.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bevlane.dataset import BevLaneDataset                        # noqa: E402
from bevlane.model import MODELS, BEV_H                           # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--model", default="v52")
ap.add_argument("--list", default=None)
ap.add_argument("--root", default="out/bevlane")
ap.add_argument("--frames", type=int, default=150)
ap.add_argument("--tag", default="")
a = ap.parse_args()

scenes = [l.strip() for l in open(a.list) if l.strip()]
ds = BevLaneDataset(a.root, scenes, gt_key="gt_cons", max_per_scene=6,
                    n_cams=8, trim_start=3, trim_end=10)
m = MODELS[a.model](n_seg=21).cuda().eval()
sd = torch.load(a.ckpt, map_location="cpu")
sd = {k.replace("module.", ""): v for k, v in sd.get("model", sd).items()}
cur = m.state_dict()
m.load_state_dict({k: v for k, v in sd.items()
                   if k in cur and cur[k].shape == v.shape}, strict=False)

BANDS = [("front 0-20m", 0, 20), ("front 20-40m", 20, 40), ("front 40-60m", 40, 60),
         ("rear 0-20m", -20, 0), ("rear 20-40m", -40, -20)]
dl = {b[0]: [] for b in BANDS}       # left edge pred-GT (+=expands left)
dr = {b[0]: [] for b in BANDS}       # right edge pred-GT (+=left = shrinks on the right)
wr = {b[0]: [] for b in BANDS}       # width ratio pred/GT
step = max(1, len(ds) // a.frames)
done = 0
ROAD = 1
for i in range(0, len(ds), step):
    b = ds[i]
    if b is None:
        continue
    with torch.no_grad(), torch.autocast("cuda", torch.float16):
        out = m(b[0][None].cuda(), b[1][None].cuda(), b[2][None].cuda())
    pred = out[0].float().argmax(1)[0].cpu().numpy()
    gt = b[3].numpy()
    pm = np.isin(pred, (1, 3, 4, 5))     # road+crosswalk+lane+stop
    gm = np.isin(np.where(gt == 255, 0, gt), (1, 3, 4, 5))
    for nm, lo, hi in BANDS:
        r0 = max(0, int((80.0 - hi) / 0.2))
        r1 = min(BEV_H, int((80.0 - lo) / 0.2))
        for r in range(r0, r1, 10):
            gc = np.flatnonzero(gm[r])
            pc = np.flatnonzero(pm[r])
            if len(gc) < 5 or len(pc) < 5:
                continue
            # column index decreases toward +y (left)
            dl[nm].append((gc[0] - pc[0]) * 0.2)
            dr[nm].append((gc[-1] - pc[-1]) * 0.2)
            wr[nm].append(len(pc) / len(gc))
    done += 1
    if done >= a.frames:
        break

print(f"\n=== {a.tag or a.ckpt} road edge offset ({done} frames, +=left) ===")
print("band       n     left pred-GT      right pred-GT    width pred/GT")
for nm, *_ in BANDS:
    L, R, W = map(np.array, (dl[nm], dr[nm], wr[nm]))
    if len(L) >= 10:
        print(f"  {nm:<8} {len(L):5d}  {np.median(L):+6.2f} m"
              f"           {np.median(R):+6.2f} m        "
              f"{np.median(W):5.2f}")
print("PROBE_ROAD_EDGE_DONE")

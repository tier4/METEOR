"""Signed lateral offset of predicted lane lines (class 4) vs GT, per range band.

Re the holdout-video impression that far vehicles hug the left lane line:
boxes were unbiased vs GT, so the remaining hypothesis is that the lane line is
drawn shifted right (-y). Per row, match each GT line component center to the
nearest predicted line center and accumulate the pred-GT y difference per band (+=left).
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
ap.add_argument("--cls", type=int, default=4, help="4=laneline")
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


def runs(row_mask):
    """Return the center column of each connected component in a row."""
    cols = np.flatnonzero(row_mask)
    if len(cols) == 0:
        return []
    br = np.flatnonzero(np.diff(cols) > 1)
    out, s0 = [], 0
    for b in list(br) + [len(cols) - 1]:
        out.append(cols[s0:b + 1].mean())
        s0 = b + 1
    return out


BANDS = [("front 0-10m", 0, 10), ("front 10-20m", 10, 20), ("front 20-30m", 20, 30),
         ("front 30-40m", 30, 40), ("front 40-60m", 40, 60),
         ("rear 0-20m", -20, 0), ("rear 20-40m", -40, -20)]
dy = {b[0]: [] for b in BANDS}
step = max(1, len(ds) // a.frames)
done = 0
for i in range(0, len(ds), step):
    b = ds[i]
    if b is None:
        continue
    with torch.no_grad(), torch.autocast("cuda", torch.float16):
        out = m(b[0][None].cuda(), b[1][None].cuda(), b[2][None].cuda())
    pred = out[0].float().argmax(1)[0].cpu().numpy()
    gt = b[3].numpy() if torch.is_tensor(b[3]) else np.asarray(b[3])
    if gt.shape != pred.shape:
        continue
    pm, gm = pred == a.cls, gt == a.cls
    for nm, lo, hi in BANDS:
        r0 = max(0, int((80.0 - hi) / 0.2))
        r1 = min(BEV_H, int((80.0 - lo) / 0.2))
        for r in range(r0, r1, 5):
            gc = runs(gm[r])
            if not gc:
                continue
            pc = runs(pm[r])
            if not pc:
                continue
            for g in gc:
                d = min(pc, key=lambda p: abs(p - g)) - g
                if abs(d) <= 7:                       # match only within 1.4 m
                    # column index decreases toward +y (left): y = 50 - col*0.2
                    dy[nm].append(-d * 0.2)
    done += 1
    if done >= a.frames:
        break

print(f"\n=== {a.tag or a.ckpt} lane line pred-GT lateral offset ({done} frames, +y=left) ===")
for nm, *_ in BANDS:
    e = np.array(dy[nm])
    if len(e) >= 10:
        print(f"  {nm:<9} n={len(e):5d}  {e.mean():+6.3f}±{e.std():5.3f} m")
print("PROBE_LANE_LAT_DONE")

"""Lateral distance from GT vehicle box centers to the nearest lane line: predicted lines vs GT lines.

A vehicle in the lane center should be ~1.5-1.7 m from the line. If that distance
collapses to ~0.3-0.5 m for predicted lines, it quantitatively backs the "boxes hug
the lane line" impression (boxes are right, line position is off).
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
ds = BevLaneDataset(a.root, scenes, gt_key="gt_cons", with_boxdet=True,
                    max_per_scene=6, n_cams=8, trim_start=3, trim_end=10)
m = MODELS[a.model](n_seg=21).cuda().eval()
sd = torch.load(a.ckpt, map_location="cpu")
sd = {k.replace("module.", ""): v for k, v in sd.get("model", sd).items()}
cur = m.state_dict()
m.load_state_dict({k: v for k, v in sd.items()
                   if k in cur and cur[k].shape == v.shape}, strict=False)

BANDS = [("0-15m", 0, 15), ("15-30m", 15, 30), ("30-45m", 30, 45)]
d_pred = {b[0]: [] for b in BANDS}
d_gt = {b[0]: [] for b in BANDS}
step = max(1, len(ds) // a.frames)
done = 0
for i in range(0, len(ds), step):
    b = ds[i]
    if b is None:
        continue
    with torch.no_grad(), torch.autocast("cuda", torch.float16):
        out = m(b[0][None].cuda(), b[1][None].cuda(), b[2][None].cuda())
    pred = out[0].float().argmax(1)[0].cpu().numpy()
    gt = np.where(b[3].numpy() == 255, 0, b[3].numpy())
    pm, gm = pred == 4, gt == 4
    bx, nb = b[4], int(b[5])
    for k in range(max(nb, 0)):
        cls, xe, ye = [float(v) for v in bx[k][:3]]
        ln = float(bx[k][3])
        if ln <= 0 or cls >= 1.5 or abs(ye) > 12:
            continue
        r = int((80.0 - xe) / 0.2)
        if not (0 <= r < BEV_H):
            continue
        c = (50.0 - ye) / 0.2
        for nm, lo, hi in BANDS:
            if lo <= abs(xe) < hi:
                for mask, acc in ((pm, d_pred), (gm, d_gt)):
                    cols = np.flatnonzero(mask[max(0, r - 2):r + 3].any(0))
                    if len(cols):
                        acc[nm].append(np.abs(cols - c).min() * 0.2)
    done += 1
    if done >= a.frames:
        break

print(f"\n=== {a.tag or a.ckpt} vehicle box center -> nearest lane line distance ({done} frames) ===")
print("band     n(pred/gt)   to pred line (median)   to GT line (median)")
for nm, *_ in BANDS:
    P, G = np.array(d_pred[nm]), np.array(d_gt[nm])
    if len(P) >= 10 and len(G) >= 10:
        print(f"  {nm:<7} {len(P):4d}/{len(G):<4d}   {np.median(P):5.2f} m"
              f"              {np.median(G):5.2f} m")
print("PROBE_BOX_TO_LINE_DONE")

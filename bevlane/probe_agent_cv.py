#!/usr/bin/env python3
"""Physics baselines for agent trajectories (2026-08-27).

Judge how much of the model's agentADE 2.14 m (veh 2.45 / vru 1.95) is actually
learned. Two baselines computed from GT alone:
  zero      : predict zero displacement (everyone stationary)
  oracle-CV : take the first 0.5 s displacement as velocity and extrapolate at
              constant velocity (zero observation error = upper bound of CV models)
If the model does not approach oracle-CV, the head has effectively learned only
constant velocity and a structural lever (history, interaction, CV residual) is needed.
"""
import argparse
import math
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from bevlane.dataset import BevLaneDataset                      # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="out/bevlane")
ap.add_argument("--val-list", default="val.lst")
ap.add_argument("--scenes", type=int, default=40)
ap.add_argument("--max-per-scene", type=int, default=8)
a = ap.parse_args()

sc = [l.strip() for l in open(a.val_list) if l.strip()][:a.scenes]
ds = BevLaneDataset(a.root, sc, gt_key="gt_cons",
                    with_boxdet=True, with_agenttraj=True,
                    max_per_scene=a.max_per_scene)

stats = {}


def add(key, val):
    stats.setdefault(key, []).append(float(val))


n_agents = 0
for i in range(len(ds)):
    b = ds[i]
    if b is None:
        continue
    # with_agenttraj returns 4 trailing tensors (boxes[64,6], count, traj[64,6,2],
    # tvalid[64,6]) in a row (matches the order in dataset.py)
    tensors = [t for t in b if torch.is_tensor(t)]
    tj = None
    for j in range(len(tensors) - 3):
        t2 = tensors[j + 2]
        if t2.dim() == 3 and tuple(t2.shape[1:]) == (6, 2) and \
                tensors[j].dim() == 2 and tensors[j].shape[0] == t2.shape[0]:
            bx, nbT, tj, tv = tensors[j], tensors[j + 1], t2, tensors[j + 3]
            break
    if tj is None:
        continue
    N = max(0, int(nbT))
    for k in range(N):
        if float(bx[k, 3]) <= 0:
            continue
        v = tv[k] if tv is not None else torch.ones(6)
        if v.sum() == 0:
            continue
        g = tj[k]                                   # [6,2] displacement (cumulative)
        cls = "veh" if float(bx[k, 0]) < 1.5 else "vru"
        disp3 = float(g[5].norm())
        moving = disp3 > 1.0
        # zero baseline
        dz = g.norm(dim=1)
        add(f"zero_{cls}", (dz * v).sum() / v.sum())
        # oracle-CV: v = g[0] (first 0.5s), pred_k = v*(k+1)
        steps = torch.arange(1, 7).view(6, 1).float()
        pcv = g[0].view(1, 2) * steps
        dcv = (pcv - g).norm(dim=1)
        add(f"cv_{cls}", (dcv * v).sum() / v.sum())
        add(f"cv_{cls}_{'mov' if moving else 'stat'}",
            (dcv * v).sum() / v.sum())
        if v[5] > 0:
            add(f"cv_fde_{cls}", dcv[5])
            if moving and float(pcv[5].norm()) > 0.3:
                ga = math.atan2(float(g[5, 1]), float(g[5, 0]))
                pa = math.atan2(float(pcv[5, 1]), float(pcv[5, 0]))
                add(f"cv_head_{cls}",
                    math.degrees(abs((pa - ga + math.pi) % (2 * math.pi)
                                     - math.pi)))
        add(f"disp3_{cls}", disp3)
        add(f"moving_{cls}", moving)
        n_agents += 1

print(f"agents={n_agents}  (scenes={len(sc)})")
for k in sorted(stats):
    v = np.array(stats[k])
    print(f"  {k:<18} mean={v.mean():.3f}  n={len(v)}")

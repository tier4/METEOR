"""Measure the near-ego road dropout on intersection approach, per band and over time (2026-08-22).

Lumping 0-20 m together made the area appear to grow at intersections and missed the
user-observed shrinking of the nearby road. The immediate vicinity (0-5 m) has steep
viewing angles even with 8 cameras and is prone to blind spots, so split into 0-5 / 5-10 /
10-20 m and relate to the distance to the intersection (nearest GT crosswalk ahead).
"""
import argparse, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bevlane.dataset import BevLaneDataset
from bevlane.model import MODELS

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--list", default="val.lst")
ap.add_argument("--root", default="out/bevlane")
ap.add_argument("--scenes", type=int, default=10)
ap.add_argument("--tag", default="")
a = ap.parse_args()
scenes = [l.strip() for l in open(a.list) if l.strip()][:a.scenes]
ds = BevLaneDataset(a.root, scenes, gt_key="gt_cons", max_per_scene=200,
                    n_cams=8, trim_start=3, trim_end=10)
m = MODELS["v52"](n_seg=21).cuda().eval()
sd = torch.load(a.ckpt, map_location="cpu"); sd = sd.get("model", sd)
sd = {k.replace("module.", ""): v for k, v in sd.items()}
cur = m.state_dict()
m.load_state_dict({k: v for k, v in sd.items()
                   if k in cur and cur[k].shape == v.shape}, strict=False)
ROAD = (1, 3, 4, 5)
COLS = slice(200, 300)               # |y| <= 10 m
BANDS = [("0-5m", 375, 400), ("5-10m", 350, 375), ("10-20m", 300, 350)]
DIST = [("ix 0-10m", 0, 10), ("10-20m", 10, 20),
        ("20-40m", 20, 40), ("no ix", 40, 999)]
acc = {(d[0], b[0]): [] for d in DIST for b in BANDS}
by_scene = {}
for i, (s, f) in enumerate(ds.items):
    by_scene.setdefault(s, []).append((int(f["frame"]), i))
for s, lst in list(by_scene.items())[:a.scenes]:
    lst.sort()
    for fi, di in lst[::3]:
        b = ds[di]
        if b is None: continue
        gt = np.where(b[3].numpy() == 255, 0, b[3].numpy())
        cw = np.argwhere(gt[:400, COLS] == 3)      # crosswalks ahead
        d_ix = 999.0
        if len(cw):
            d_ix = float((400 - cw[:, 0].max()) * 0.2)   # nearest crosswalk
        with torch.no_grad(), torch.autocast("cuda", torch.float16):
            out = m(b[0][None].cuda(), b[1][None].cuda(), b[2][None].cuda())
        pred = out[0].float().argmax(1)[0].cpu().numpy()
        for dn, lo, hi in DIST:
            if not (lo <= d_ix < hi): continue
            for bn, r0, r1 in BANDS:
                p = np.isin(pred[r0:r1, COLS], ROAD).mean()
                g = np.isin(gt[r0:r1, COLS], ROAD).mean()
                acc[(dn, bn)].append((p, g))
print(f"\n=== {a.tag} distance to intersection x road coverage per band (pred / GT) ===")
print("ix distance       0-5m              5-10m             10-20m")
for dn, _, _ in DIST:
    row = f"  {dn:12s}"
    for bn, _, _ in BANDS:
        v = acc[(dn, bn)]
        if len(v) < 3:
            row += "     --            "
            continue
        arr = np.array(v)
        row += f"  {arr[:,0].mean():.3f}/{arr[:,1].mean():.3f}(n={len(v):3d})"
    print(row)
print("PROBE_NEAR_DONE")

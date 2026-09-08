"""VRU (cls GT=2 / pred=1) recall@score0.25 per range band (for the paint gate decision)."""
import argparse, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bevlane.dataset import BevLaneDataset
from bevlane.model import MODELS

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--model", default="v52")
ap.add_argument("--list", default="val.lst")
ap.add_argument("--root", default="out/bevlane")
ap.add_argument("--frames", type=int, default=250)
ap.add_argument("--tag", default="")
a = ap.parse_args()
scenes = [l.strip() for l in open(a.list) if l.strip()][:80]
ds = BevLaneDataset(a.root, scenes, gt_key="gt_cons", with_boxdet=True,
                    max_per_scene=6, n_cams=8, trim_start=3, trim_end=10)
m = MODELS[a.model](n_seg=21).cuda().eval()
sd = torch.load(a.ckpt, map_location="cpu")
sd = {k.replace("module.", ""): v for k, v in sd.get("model", sd).items()}
cur = m.state_dict()
m.load_state_dict({k: v for k, v in sd.items()
                   if k in cur and cur[k].shape == v.shape}, strict=False)
BANDS = [(0, 10), (10, 20), (20, 30), (30, 40)]
gt_n = {b: 0 for b in BANDS}; hit = {b: 0 for b in BANDS}
step = max(1, len(ds) // a.frames); done = 0
for i in range(0, len(ds), step):
    b = ds[i]
    if b is None: continue
    with torch.no_grad(), torch.autocast("cuda", torch.float16):
        out = m(b[0][None].cuda(), b[1][None].cuda(), b[2][None].cuda())
    dets = m.decode_boxes(out[3].float().cpu(), out[4].float().cpu(), thresh=0.25)[0]
    pred = [(float(d[2]), float(d[3])) for d in dets if 0.5 <= float(d[0]) < 1.5]
    bx, nb = b[4], int(b[5])
    for k in range(max(nb, 0)):
        cls, xe, ye, ln = [float(v) for v in bx[k][:4]]
        if ln <= 0 or not (1.5 <= cls < 2.5): continue
        r = (xe*xe + ye*ye) ** 0.5
        ok = any((xe-px)**2 + (ye-py)**2 < 4.0 for px, py in pred)
        for lo, hi in BANDS:
            if lo <= r < hi:
                gt_n[(lo, hi)] += 1
                if ok: hit[(lo, hi)] += 1
    done += 1
    if done >= a.frames: break
print(f"=== {a.tag or a.ckpt} VRU recall@0.25 ({done} frames) ===")
for b in BANDS:
    if gt_n[b]:
        print(f"  {b[0]}-{b[1]}m  n={gt_n[b]:4d}  R={hit[b]/gt_n[b]:.3f}")
print("PROBE_VRU_DONE")

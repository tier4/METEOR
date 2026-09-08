"""Quantify the near-field road shrinking when approaching an intersection (2026-08-22).

Per frame, measure (a) road area within 0-20 m of ego and (b) crosswalk pixels
within 30 m ahead (proxy for intersection proximity), for both pred and GT, and
relate them. Shrinking in pred only = model problem; shrinking in GT too =
label / occlusion origin.
"""
import argparse, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bevlane.dataset import BevLaneDataset
from bevlane.model import MODELS, BEV_H

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--list", default="val.lst")
ap.add_argument("--root", default="out/bevlane")
ap.add_argument("--scenes", type=int, default=12)
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

# rows: x = 80 - 0.2*row. near 0-20m = rows 300..400, ahead 30m = rows 250..400
NEAR = slice(300, 400)       # 0-20 m ahead of ego
FWD30 = slice(250, 400)      # 0-30 m ahead
COLS = slice(175, 325)       # |y| <= 15 m
ROAD = (1, 3, 4, 5)          # road + crosswalk + laneline + stopline
rows = []
by_scene = {}
for i, (s, f) in enumerate(ds.items):
    by_scene.setdefault(s, []).append((int(f["frame"]), i))
for s, lst in list(by_scene.items())[:a.scenes]:
    lst.sort()
    for fi, di in lst[::3]:
        b = ds[di]
        if b is None: continue
        with torch.no_grad(), torch.autocast("cuda", torch.float16):
            out = m(b[0][None].cuda(), b[1][None].cuda(), b[2][None].cuda())
        pred = out[0].float().argmax(1)[0].cpu().numpy()
        gt = np.where(b[3].numpy() == 255, 0, b[3].numpy())
        p_road = int(np.isin(pred[NEAR, COLS], ROAD).sum())
        g_road = int(np.isin(gt[NEAR, COLS], ROAD).sum())
        p_cw = int((pred[FWD30, COLS] == 3).sum())
        g_cw = int((gt[FWD30, COLS] == 3).sum())
        rows.append((s, fi, p_road, g_road, p_cw, g_cw))
if not rows:
    print("no data"); sys.exit()
arr = np.array([[r[2], r[3], r[4], r[5]] for r in rows], float)
# intersection frames = top 25% of GT crosswalk
thr = np.percentile(arr[:, 3], 75)
near_ix = arr[:, 3] >= max(thr, 50)
far_ix = arr[:, 3] < max(thr, 50) * 0.2
print(f"\n=== {a.tag} near-field road area on intersection approach ({len(rows)} frames) ===")
print(f"intersection frames {int(near_ix.sum())} / non-intersection {int(far_ix.sum())}")
for nm, msk in (("near ix", near_ix), ("straight", far_ix)):
    if msk.sum() < 3: continue
    p, g = arr[msk, 0].mean(), arr[msk, 1].mean()
    print(f"  {nm:8s} pred road {p:8.0f} px | GT road {g:8.0f} px | "
          f"pred/GT {p/max(g,1):.3f}")
if near_ix.sum() >= 3 and far_ix.sum() >= 3:
    pr = arr[near_ix, 0].mean() / max(arr[far_ix, 0].mean(), 1)
    gr = arr[near_ix, 1].mean() / max(arr[far_ix, 1].mean(), 1)
    print(f"\n  intersection/straight ratio: pred {pr:.3f} / GT {gr:.3f}")
    print("  -> small in pred only = model-specific; small in GT too = label/occlusion origin")
print("PROBE_IX_DONE")

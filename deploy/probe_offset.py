"""Measure systematic lateral offset of BEV outputs against GT (no torch needed; not
shared engine/ckpt, engine only). Quantifies the report "rtv_r73 is offset slightly right".

  1) Detection boxes: mean (pred y - GT y) of vehicles matched to GT within 3 m.
     Screen right = negative y, so a rightward offset yields a negative mean.
  2) Road raster: shift the road class of lane argmax column-wise against GT
     (gt_cons), take IoU at each shift, and pick the best (0.2 m/column).
"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deploy.runtime import MeteorRT, decode_boxes
from deploy.eval_deploy import frames

ap = argparse.ArgumentParser()
ap.add_argument("--engine", required=True)
ap.add_argument("--root", default="fast")
ap.add_argument("--scenes-file", default="out/eval_scenes.txt")
ap.add_argument("--limit", type=int, default=120)
ap.add_argument("--tag", default="")
a = ap.parse_args()
scenes = [l.strip() for l in open(a.scenes_file) if l.strip()]
rt = MeteorRT(a.engine, n_out_slots=1)
dys, shifts = [], []
for imgs, K, Tc, v0, pose, gt, bx in frames(a.root, scenes, 8, 2, a.limit):
    o = rt.infer(imgs, K, Tc, v0, pose=pose)
    dets = decode_boxes(o["hm"], o["reg"], thresh=0.25)
    pred = [(d["x"], d["y"]) for d in dets if d["cls"] == "vehicle"]
    for r in bx:
        if int(r[0]) != 1 or float(r[3]) <= 0: continue
        gx, gy = float(r[1]), float(r[2])
        if (gx*gx + gy*gy) ** 0.5 > 45 or abs(gy) > 40: continue
        best = None
        for px, py in pred:
            d2 = (gx-px)**2 + (gy-py)**2
            if d2 < 9.0 and (best is None or d2 < best[0]): best = (d2, py)
        if best is not None: dys.append(best[1] - gy)
    lane = o["lane"][0]
    if lane.ndim == 3: lane = lane.argmax(0)
    pr = (lane == 1); gr = (gt == 1)
    if gr.sum() < 500: continue
    ious = []
    for sh in range(-5, 6):
        p2 = np.roll(pr, sh, axis=1)
        ious.append(((p2 & gr).sum() / max((p2 | gr).sum(), 1), sh))
    shifts.append(max(ious)[1])
dys = np.array(dys); shifts = np.array(shifts)
print(f"=== {a.tag or a.engine} ===")
print(f"box lateral offset (n={len(dys)}): mean {dys.mean():+.3f} m / median {np.median(dys):+.3f} m"
      f"  (negative = offset to the right)")
print(f"road raster best column shift (n={len(shifts)}): mean {shifts.mean():+.2f} cols"
      f" = {0.2*shifts.mean():+.2f} m (positive = shifting pred right fits = pred is left-biased)")

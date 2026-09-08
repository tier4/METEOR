"""Dump per-GT-box yaw error for one model (one side of a paired comparison).

Why one model per process:
  the BEV rear range (METEOR_BEV_XR) is a constant read at import time, so one
  process cannot correctly build both the light (rear-40) and full-range models.
  The 2026-08-14 comparison ignored this, ran the light model with the wrong
  geometry, and compared numbers with detections collapsed to 90.

Why paired:
  yaw error is heavily contaminated by recall. A model that only detects easy boxes
  (ego-parallel, near) shows a small mean yaw error. Phase A 2.7 deg (121 boxes) vs
  phase B 7.4 deg (832 boxes) is exactly this contamination. Only the same GT boxes
  detected by both models give a real heading-accuracy comparison.

The output npz holds yaw error keyed by (frame index, box index).
probe_yaw_join.py does the join.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bevlane.dataset import BevLaneDataset                        # noqa: E402
from bevlane.model import MODELS                                  # noqa: E402


def load_model(ckpt, name):
    m = MODELS[name](n_seg=21).cuda().eval()
    sd = torch.load(ckpt, map_location="cpu")
    sd = sd.get("model", sd)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    cur = m.state_dict()
    keep = {k: v for k, v in sd.items()
            if k in cur and cur[k].shape == v.shape}
    m.load_state_dict(keep, strict=False)
    print(f"[load] {ckpt}: {len(keep)}/{len(cur)} tensors restored", flush=True)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-cams", type=int, default=8)
    ap.add_argument("--list", default="val.lst")
    ap.add_argument("--root", default="out/bevlane")
    ap.add_argument("--scenes", type=int, default=60)
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--thresh", type=float, default=0.25)
    ap.add_argument("--match-r", type=float, default=2.0,
                    help="GT-to-prediction matching radius [m]")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    scenes = [l.strip() for l in open(a.list) if l.strip()][:a.scenes]
    ds = BevLaneDataset(a.root, scenes, gt_key="gt_cons", with_boxdet=True,
                        max_per_scene=4, n_cams=8, trim_start=3, trim_end=10)
    m = load_model(a.ckpt, a.model)

    rows = []
    step = max(1, len(ds) // a.frames)
    done = 0
    for i in range(0, len(ds), step):
        b = ds[i]
        if b is None:
            continue
        with torch.no_grad(), torch.autocast("cuda", torch.float16):
            out = m(b[0][None][:, :a.n_cams].cuda(),
                    b[1][None][:, :a.n_cams].cuda(),
                    b[2][None][:, :a.n_cams].cuda())
        dets = m.decode_boxes(out[3].float().cpu(), out[4].float().cpu(),
                              thresh=a.thresh)[0]
        pred = [(float(d[2]), float(d[3]), float(d[6]))
                for d in dets if int(d[0]) == 0]
        bx, nb = b[4], int(b[5])
        for k in range(max(nb, 0)):
            cls, xe, ye, ln, wd, yaw = [float(v) for v in bx[k][:6]]
            if ln <= 0 or cls >= 1.5:
                continue
            r = (xe * xe + ye * ye) ** 0.5
            if not (0 < xe <= 50 or (-20 <= xe <= 0 and r <= 50)):
                continue
            best = None
            for dx, dy, dyaw in pred:
                d2 = (xe - dx) ** 2 + (ye - dy) ** 2
                if d2 < a.match_r ** 2 and (best is None or d2 < best[0]):
                    best = (d2, dyaw)
            if best is None:
                continue
            de = abs((best[1] - yaw + np.pi) % (2 * np.pi) - np.pi)
            rows.append((i, k, xe, ye, r,
                         min(de, np.pi - de),                 # 180-degree symmetry
                         abs((np.degrees(yaw) + 90) % 180 - 90)))
        done += 1
        if done >= a.frames:
            break

    arr = np.array(rows, dtype=np.float64) if rows else np.zeros((0, 7))
    np.savez(a.out, rows=arr)
    print(f"[out] {a.out}: {len(arr)} boxes detected ({done} frames)", flush=True)


if __name__ == "__main__":
    main()

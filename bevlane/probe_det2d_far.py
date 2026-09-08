#!/usr/bin/env python3
"""Premise check for paint-det: does 2D detection really see far vehicles?

Cam-only BEV vehicle recall is stuck around 0.50 at 20-40 m. Is that because 2D
does not see them, or because it does but the wide depth distribution fails to
vote into the right BEV cell? In the latter case injecting the 2D heatmap before the
lift (paint-det) makes sense; in the former there is no information to inject.

Method: for each vehicle box in bbox2d GT (image coords), read the peak of hm2d near
the center at the scale assigned by its size. Peak > thr = detected in 2D.
Range is approximated as Z ~ f * H_real / h_px (vehicle height 1.5 m, f from K), so it
lines up with the BEV bands (20-40 / 40-60 m).
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bevlane.dataset import BevLaneDataset          # noqa: E402
from bevlane.model import MODELS                    # noqa: E402

VEH = (1, 2, 3)          # car / truck / bus
VRU = (4, 5, 6)          # bicycle / bike / person
H_REAL = {1: 1.5, 2: 3.0, 3: 3.2, 4: 1.7, 5: 1.7, 6: 1.7}
BANDS = ((0, 20), (20, 40), (40, 60), (60, 80))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--list", default="val.lst")
    ap.add_argument("--scenes", type=int, default=12)
    ap.add_argument("--per-scene", type=int, default=6)
    ap.add_argument("--thr", type=float, default=0.3)
    ap.add_argument("--root", default="out/bevlane")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    from bevlane.probe_net import load_full
    net = load_full(a.ckpt, device=dev)   # 2026-09-06: lossless build (silent failure #10)

    scenes = [l.strip() for l in open(a.list) if l.strip()][:a.scenes]
    ds = BevLaneDataset(a.root, scenes, gt_key="gt_cons",
                        max_per_scene=a.per_scene, n_cams=8, with_bbox2d=True)

    split = net.DET2D_SPLIT
    strides = net.DET2D_STRIDES
    hit = {g: {b: [0, 0] for b in BANDS} for g in ("veh", "vru")}
    peaks = {g: {b: [] for b in BANDS} for g in ("veh", "vru")}

    for i in range(len(ds)):
        x = ds[i]
        if x is None:
            continue
        img, K, T, _gt, b2, c2 = x
        with torch.no_grad(), torch.autocast("cuda", torch.float16):
            net(img[None].to(dev), K[None].to(dev), T[None].to(dev))
            f = net._last_f
            hms, _ = net.det2d_forward(f, 1, img.shape[0])
        for ci in range(img.shape[0]):
            fx = float(K[ci][0, 0])
            for bi in range(int(c2[ci])):
                cls, cx, cy, w, h = [float(v) for v in b2[ci, bi]]
                cls = int(cls)
                if cls in VEH:
                    g = "veh"
                elif cls in VRU:
                    g = "vru"
                else:
                    continue
                if h <= 1 or w <= 1:
                    continue
                z = fx * H_REAL[cls] / h            # approximate range [m]
                band = next((b for b in BANDS if b[0] <= z < b[1]), None)
                if band is None:
                    continue
                sz = max(w, h)
                si = 0 if sz < split[0] else (1 if sz < split[1] else 2)
                hm = hms[si][0, ci, cls]            # [fh, fw]
                st = strides[si]
                gx, gy = int(cx / st), int(cy / st)
                fh, fw = hm.shape
                y0, y1 = max(0, gy - 1), min(fh, gy + 2)
                x0, x1 = max(0, gx - 1), min(fw, gx + 2)
                if y0 >= y1 or x0 >= x1:
                    continue
                p = float(hm[y0:y1, x0:x1].float().sigmoid().max())
                hit[g][band][1] += 1
                hit[g][band][0] += int(p >= a.thr)
                peaks[g][band].append(p)

    THRS = (0.05, 0.1, 0.2, 0.3, 0.5)
    print("\n2D detection hit rate / range approximated from h_px")
    print("    band      n     peak median  " +
          "  ".join(f"thr{t:g}" for t in THRS))
    for g in ("veh", "vru"):
        print(f"  [{g}]")
        for b in BANDS:
            pk = np.array(peaks[g][b], dtype=np.float64)
            if pk.size == 0:
                continue
            rs = "  ".join(f"{float((pk >= t).mean()):.3f} " for t in THRS)
            print(f"    {b[0]:>2}-{b[1]:<2}m  {pk.size:>4}  "
                  f"{float(np.median(pk)):.3f}       {rs}")


if __name__ == "__main__":
    main()

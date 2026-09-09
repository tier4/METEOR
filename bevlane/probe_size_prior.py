#!/usr/bin/env python3
"""Is the geometric prior "range from box pixel height" more accurate than the current depth head?

Paint-style self-injection fails because the injected quantity is a function of the
same backbone feature f (no new information). Z = f_x * H_real / h_px, by contrast,
  - depends on the box extent h_px, a **non-local quantity**
  - and on the intrinsic f_x, a **quantity absent from the feature map**
so the depth head cannot compute it on its own. Before adding it, measure how
accurate that formula really is against LiDAR depth GT.

Comparison: Z_geo (box pixel height) vs Z_pred (depth-distribution expectation) vs Z_gt (LiDAR).
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bevlane.dataset import BevLaneDataset          # noqa: E402
from bevlane.model import MODELS                    # noqa: E402
from bevlane.ckpt_load import load_net              # noqa: E402

# real-world height [m] of the 10 2D instance classes (boxes are image-space AABBs, so vehicle height + a bit)
H_REAL = {1: 1.55, 2: 3.2, 3: 3.3, 4: 1.7, 5: 1.7, 6: 1.7}
BANDS = ((0, 20), (20, 40), (40, 60), (60, 80))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--list", default="val.lst")
    ap.add_argument("--scenes", type=int, default=10)
    ap.add_argument("--per-scene", type=int, default=5)
    ap.add_argument("--root", default="out/bevlane")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = MODELS["v52"](n_seg=21).to(dev).eval()
    load_net(net, a.ckpt)

    ds = BevLaneDataset(a.root, [l.strip() for l in open(a.list) if l.strip()][:a.scenes],
                        gt_key="gt_cons", max_per_scene=a.per_scene, n_cams=8,
                        with_depth=True, with_bbox2d=True)
    x0 = ds[0]
    print(f"[ds] {len(x0)} elements: " +
          ", ".join(str(tuple(t.shape)) for t in x0 if torch.is_tensor(t)))

    D_MIN, D_STEP = net.D_MIN, net.D_STEP
    centers = torch.arange(net.D, device=dev).float() * D_STEP + D_MIN
    err = {b: {"geo": [], "pred": []} for b in BANDS}

    for i in range(len(ds)):
        x = ds[i]
        if x is None:
            continue
        img, K, T, _gt, dgt, b2, c2 = x[0], x[1], x[2], x[3], x[4], x[-2], x[-1]
        if dgt.dim() != 3:
            print("[skip] unexpected depth GT shape", dgt.shape)
            break
        with torch.no_grad(), torch.autocast("cuda", torch.float16):
            out = net(img[None].to(dev), K[None].to(dev), T[None].to(dev))
        dlog = out[1]
        p = dlog.float().flatten(0, 1).softmax(1)            # [N,D,h,w]
        zpred = (p * centers.view(1, -1, 1, 1)).sum(1)        # [N,h,w]
        nC, dh, dw = dgt.shape
        for ci in range(min(nC, img.shape[0])):
            fx = float(K[ci][0, 0])
            sy, sx = dh / img.shape[-2], dw / img.shape[-1]
            for bi in range(int(c2[ci])):
                cls, cx, cy, w, h = [float(v) for v in b2[ci, bi]]
                cls = int(cls)
                if cls not in H_REAL or h <= 1:
                    continue
                z_geo = fx * H_REAL[cls] / h
                gy, gx = int(cy * sy), int(cx * sx)
                if not (0 <= gy < dh and 0 <= gx < dw):
                    continue
                y0, y1 = max(0, gy - 1), min(dh, gy + 2)
                x0_, x1 = max(0, gx - 1), min(dw, gx + 2)
                win = dgt[ci, y0:y1, x0_:x1]
                win = win[win > 0.5]
                if win.numel() == 0:
                    continue
                z_gt = float(win.median())
                band = next((b for b in BANDS if b[0] <= z_gt < b[1]), None)
                if band is None:
                    continue
                z_pr = float(zpred[ci, y0:y1, x0_:x1].median())
                err[band]["geo"].append(abs(z_geo - z_gt))
                err[band]["pred"].append(abs(z_pr - z_gt))

    print("\nmedian range error at object centers [m] (upper bound using GT 2D boxes)")
    print("   band      n    geom Z_geo   depth head Z_pred   winner")
    for b in BANDS:
        g, p_ = err[b]["geo"], err[b]["pred"]
        if not g:
            continue
        mg, mp = float(np.median(g)), float(np.median(p_))
        win = "geom" if mg < mp else "depth head"
        print(f"  {b[0]:>2}-{b[1]:<2}m  {len(g):>4}   {mg:>7.2f}      "
              f"{mp:>7.2f}        {win} ({abs(mg-mp):.2f} m diff)")


if __name__ == "__main__":
    main()

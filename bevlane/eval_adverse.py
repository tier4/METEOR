#!/usr/bin/env python3
"""Generalization evaluation under adverse conditions (Cosmos transfer_2).

Question: does mixing Cosmos data into training improve robustness to adverse conditions?
On the normal-condition val, v116a (real data only) vs v116c (+Cosmos) scores
0.351 vs 0.348, a slight loss; the real goal of Cosmos (adverse-condition
robustness) cannot be measured on a mostly-clear val. Here we measure on **2 fully
unseen base scenes x 6 adverse conditions** (transfer_2; no model saw the converted images).

Metrics: BEV Seg mIoU (drivable surface / lane classes) and vehicle recall (3 m match).
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bevlane.dataset import BevLaneDataset          # noqa: E402
from bevlane.model import MODELS                    # noqa: E402
from bevlane.ckpt_load import load_net              # noqa: E402

ROAD = (1, 3, 4, 5)
LANE = (2, 6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--tag", default="")
    ap.add_argument("--lidar", action="store_true",
                    help="feed the scene's lidar_bev/NNNN.npz (pillar raster) as optional input (LiDAR-ON evaluation, 2026-09-09)")
    a = ap.parse_args()

    net = MODELS["v52"](n_seg=21).cuda().eval()
    load_net(net, a.ckpt, verbose=False)
    ds = BevLaneDataset(a.root, a.scenes, gt_key="gt", with_boxdet=True,
                        n_cams=8, trim_start=3, trim_end=5)
    inter = {k: 0 for k in ("road", "lane")}
    union = {k: 0 for k in ("road", "lane")}
    gt_n = hit = 0
    n = 0
    for i in range(0, len(ds), a.stride):
        x = ds[i]
        if x is None:
            continue
        lb = None
        if a.lidar:
            s_, f_ = ds.items[i]
            lp = f_.get("lidar_bev") or f"lidar_bev/{int(f_['frame']):04d}.npz"
            try:
                lb = torch.from_numpy(np.load(os.path.join(a.root, s_, lp))["lb"].astype(np.float32))[None].cuda()
            except Exception:
                lb = None
        with torch.no_grad(), torch.autocast("cuda", torch.float16):
            out = net(x[0][None].cuda(), x[1][None].cuda(), x[2][None].cuda(),
                      **({"lidar_bev": lb} if lb is not None else {}))
        p = out[0].float().argmax(1)[0].cpu().numpy()
        g = x[3].numpy()
        valid = g != 255
        for nm, ks in (("road", ROAD), ("lane", LANE)):
            pm = np.isin(p, ks) & valid
            gm = np.isin(g, ks) & valid
            inter[nm] += int((pm & gm).sum())
            union[nm] += int((pm | gm).sum())
        # vehicle recall (bev_box GT, 3m)
        bx, nb = x[4], int(x[5])
        dets = net.decode_boxes(out[3].float().cpu(), out[4].float().cpu(),
                                thresh=0.25)[0]
        pv = [(float(d[2]), float(d[3])) for d in dets if int(d[0]) == 0]
        for k in range(nb):
            cls, xe, ye, ln = [float(v) for v in bx[k][:4]]
            if ln <= 0 or cls >= 1.5:
                continue
            gt_n += 1
            hit += int(any((xe - px) ** 2 + (ye - py) ** 2 < 9.0
                           for px, py in pv))
        n += 1
    r = {nm: inter[nm] / max(union[nm], 1) for nm in inter}
    print(f"{a.tag}\t{n}\t{r['road']:.4f}\t{r['lane']:.4f}\t"
          f"{hit / max(gt_n, 1):.4f}\t{gt_n}")


if __name__ == "__main__":
    main()

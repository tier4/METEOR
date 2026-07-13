#!/usr/bin/env python3
"""Run BEVLane inference on frames and save visualization panels."""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from autolabel_bev import PALETTE  # noqa: E402
from bevlane.dataset import CAMS, BevLaneDataset  # noqa: E402
from bevlane.model import MODELS  # noqa: E402


def panel(imgs_raw, gt, pred):
    """6 cams (2x3) on the left, GT / prediction BEV on the right."""
    tile_w, tile_h = 512, 288
    grid = np.zeros((tile_h * 2, tile_w * 3, 3), np.uint8)
    order = [1, 0, 2, 4, 3, 5]  # FL, FW, FR / BL, BW, BR
    for k, idx in enumerate(order):
        r, c = divmod(k, 3)
        grid[r * tile_h:(r + 1) * tile_h, c * tile_w:(c + 1) * tile_w] = imgs_raw[idx]
    bev_h = tile_h * 2
    gt_v = cv2.resize(PALETTE[gt][:, :, ::-1], (bev_h, bev_h),
                      interpolation=cv2.INTER_NEAREST)
    pr_v = cv2.resize(PALETTE[pred][:, :, ::-1], (bev_h, bev_h),
                      interpolation=cv2.INTER_NEAREST)
    for v, t in ((gt_v, "GT (autolabel)"), (pr_v, "prediction")):
        cv2.putText(v, t, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2, cv2.LINE_AA)
    out = np.hstack([grid, gt_v, pr_v])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out/bevlane_ckpt/last.pt")
    ap.add_argument("--root", default="out/bevlane")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", default="out/bevlane_pred")
    ap.add_argument("--num", type=int, default=6)
    ap.add_argument("--gt-key", default="gt")
    ap.add_argument("--model", default="v1")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ds = BevLaneDataset(args.root, [args.scene], gt_key=args.gt_key)
    model = MODELS[args.model]().to(args.device)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu")["model"])
    model.eval()

    step = max(1, len(ds) // args.num)
    for i in range(0, len(ds), step):
        imgs, K, Tc, gt = ds[i]
        with torch.no_grad(), torch.autocast("cuda", torch.float16):
            logits = model(imgs[None].to(args.device), K[None].to(args.device),
                           Tc[None].to(args.device))
        pred = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
        s, f = ds.items[i]
        raws = [cv2.imread(os.path.join(args.root, s, f["imgs"][c])) for c in CAMS]
        img = panel(raws, gt.numpy().astype(np.uint8), pred)
        name = f"{args.scene.split('+0900_')[-1]}_{f['frame']:04d}.jpg"
        cv2.imwrite(os.path.join(args.out, name), img)
        print("[ok]", name, flush=True)


if __name__ == "__main__":
    main()

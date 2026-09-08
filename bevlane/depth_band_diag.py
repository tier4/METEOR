#!/usr/bin/env python3
"""Per-camera depth MAE and 2D-seg IoU: is the light model's side weakness
real, and is it light-specific?

The report "Depth and 2D seg look weak on the side cameras of the light model"
has three candidate explanations, and per-camera numbers against the SAME
frames separate them:

  1. light-specific capacity: the light line halved the depth tower
     (DEPTH_MULT 0.5). If that is the cause, the baseline's side cameras hold
     up and the light model's do not, and the gap should be widest on depth.
  2. it was always like this: side cameras carry sparser accumulated-LiDAR
     depth GT and weaker auto-labels than the front. Then BOTH models sag on
     the sides by a similar factor, and the fix is data-side, not model-side.
  3. sparsity damage (v60 only): mid-layers of seg_head were maskable. v59
     predates the sparse round, so v59-vs-v60 separates this.

    python3 bevlane/side_diag.py   (run each model in its own process/env)
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bevlane.dataset import BevLaneDataset, CAMS                # noqa: E402
from bevlane.model import MODELS                                # noqa: E402

D_MIN, D_STEP = 1.0, 1.0        # depth bin geometry (64 bins)


def run(tag, ckpt, model, n_cams):
    scenes = [l.strip() for l in open("val.lst") if l.strip()][:50]
    ds = BevLaneDataset("out/bevlane", scenes, gt_key="gt_cons",
                        with_depth=True, with_seg2d=True,
                        seg2d_key="seg2d21", n_cams=n_cams,
                        max_per_scene=4, trim_start=3, trim_end=10)
    m = MODELS[model](n_seg=21).cuda().eval()
    sd = {k.replace("module.", ""): v for k, v in
          torch.load(ckpt, map_location="cpu")["model"].items()}
    cur = m.state_dict()
    m.load_state_dict({k: v for k, v in sd.items()
                       if k in cur and cur[k].shape == v.shape}, strict=False)
    n_cam = n_cams
    dep_ae = np.zeros(n_cam)
    dep_n = np.zeros(n_cam)
    BANDS = ((0.1, 10), (10, 20), (20, 40), (40, 90))
    band_ae = np.zeros((n_cam, len(BANDS)))
    band_n = np.zeros((n_cam, len(BANDS)))
    seg_i = np.zeros((n_cam, 21))
    seg_u = np.zeros((n_cam, 21))
    n = 0
    for i in range(0, len(ds), max(1, len(ds) // 150)):
        b = ds[i]
        if b is None:
            continue
        with torch.no_grad(), torch.autocast("cuda", torch.float16):
            out = m(b[0][None].cuda(), b[1][None].cuda(), b[2][None].cuda())
        dprob = out[1].float()[0]                       # [N,D,h,w]
        Dn, Dm, Ds = m.D, m.D_MIN, m.D_STEP
        dexp = (dprob.softmax(1)
                * (torch.arange(Dn, device=dprob.device)
                   .view(1, Dn, 1, 1) * Ds + Dm)).sum(1)
        dgt = b[4].cuda()                               # [N,h,w] metres
        seg = out[2].float()[0].argmax(1)               # [N,h,w]
        sgt = b[5].cuda()                               # [N,h,w] 255=ignore
        for c in range(n_cam):
            v = dgt[c] > 0.1
            if v.any():
                dep_ae[c] += float((dexp[c][v] - dgt[c][v]).abs().sum())
                dep_n[c] += float(v.sum())
            for bi, (lo, hi) in enumerate(BANDS):
                vb = (dgt[c] > lo) & (dgt[c] <= hi)
                if vb.any():
                    band_ae[c, bi] += float((dexp[c][vb] - dgt[c][vb]).abs().sum())
                    band_n[c, bi] += float(vb.sum())
            val = sgt[c] != 255
            for cls in range(21):
                p = (seg[c] == cls) & val
                g = (sgt[c] == cls) & val
                seg_i[c, cls] += float((p & g).sum())
                seg_u[c, cls] += float((p | g).sum())
        n += 1
        if n >= 150:
            break
    print(f"\n=== {tag}  ({n} frames) ===")
    print(f"{'camera':18s} {'depth MAE':>10s} {'seg2d mIoU':>11s}")
    for c in range(n_cam):
        iou = seg_i[c] / np.maximum(seg_u[c], 1)
        miou = iou[seg_u[c] > 100].mean()
        print(f"{CAMS[c]:18s} {dep_ae[c] / max(dep_n[c], 1):10.2f} "
              f"{miou:11.3f}")
    print("  depth MAE per range band (all cameras):")
    for bi, (lo, hi) in enumerate(BANDS):
        mae = band_ae[:, bi].sum() / max(band_n[:, bi].sum(), 1)
        print(f"    {lo:>4.0f}-{hi:<3.0f}m: {mae:6.2f} m  (n={int(band_n[:, bi].sum())})")
    front = [0]
    side = [1, 2, 4, 5]
    print(f"  side/front ratio: depth "
          f"{(dep_ae[side].sum() / max(dep_n[side].sum(), 1)) / max(dep_ae[front].sum() / max(dep_n[front].sum(), 1), 1e-6):.2f}x  ")
    del m
    torch.cuda.empty_cache()


if __name__ == "__main__":
    run("v63b light latest", "out/v63b_best_e2e.pt", "v63b", 7)
    run("r64 baseline", "out/bevlane_ckpt_r64/best_e2e.pt", "v52", 8)

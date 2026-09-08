#!/usr/bin/env python3
"""How well does motion parallax recover depth at 20-60 m (premise check before implementing).

The bottleneck now is not depth sharpness but **center accuracy** (5.26 m error at
20-40 m, beyond the 3 m 3D-box match radius). A single monocular frame has plateaued,
so test a quantity the backbone cannot compute in principle: temporal parallax.

Method: for the object-center pixel in the current frame, sweep depth Z over 5-80 m,
project the 3D point into the frame 0.4 s earlier and take the normalized cross-
correlation (NCC) of the pixel patches. The Z with max correlation is the parallax depth.
Compare it with LiDAR depth GT and the model's depth expectation. If it loses here, do not implement.
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bevlane.dataset import BevLaneDataset, MEAN, STD   # noqa: E402
from bevlane.model import MODELS                        # noqa: E402
from bevlane.ckpt_load import load_net                  # noqa: E402

BANDS = ((0, 20), (20, 40), (40, 60), (60, 80))
H_CLS = (1, 2, 3)          # car / truck / bus


def rotz(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--list", default="val.lst")
    ap.add_argument("--scenes", type=int, default=10)
    ap.add_argument("--per-scene", type=int, default=4)
    ap.add_argument("--root", default="out/bevlane")
    ap.add_argument("--patch", type=int, default=9)
    a = ap.parse_args()

    net = MODELS["v52"](n_seg=21).cuda().eval()
    load_net(net, a.ckpt, verbose=False)
    cen = torch.arange(net.D).cuda().float() * net.D_STEP + net.D_MIN

    ds = BevLaneDataset(a.root, [l.strip() for l in open(a.list) if l.strip()][:a.scenes],
                        gt_key="gt_cons", max_per_scene=a.per_scene, n_cams=8,
                        with_depth=True, with_bbox2d=True, with_temporal=True)
    x0 = ds[0]
    print("[ds] elements:", ", ".join(str(tuple(t.shape)) for t in x0
                                  if torch.is_tensor(t)))

    ZS = np.arange(5.0, 80.0, 1.0)
    err = {b: {"par": [], "pred": []} for b in BANDS}
    base = []
    R = a.patch // 2

    for i in range(len(ds)):
        x = ds[i]
        if x is None:
            continue
        img, K, T = x[0], x[1], x[2]
        dgt, b2, c2 = x[4], None, None
        # from the end: pimgs, rel, pv (with_temporal), preceded by bbox2d
        pimgs, rel, pv = x[-3], x[-2], x[-1]
        for t in x:
            if torch.is_tensor(t) and t.dim() == 3 and t.shape[-1] == 5:
                b2 = t
            if torch.is_tensor(t) and t.dim() == 1 and t.dtype == torch.int64 \
                    and t.numel() == img.shape[0]:
                c2 = t
        if b2 is None or c2 is None or float(pv.sum()) < 0.5:
            continue
        with torch.no_grad(), torch.autocast("cuda", torch.float16):
            out = net(img[None].cuda(), K[None].cuda(), T[None].cuda())
        zp = ((out[1].float().flatten(0, 1).softmax(1))
              * cen.view(1, -1, 1, 1)).sum(1).cpu().numpy()
        dx, dy, dyaw = [float(v) for v in rel]
        base.append((dx ** 2 + dy ** 2) ** 0.5)
        Rz = rotz(dyaw)
        tvec = np.array([dx, dy, 0.0])
        nC, dh, dw = dgt.shape
        for ci in range(min(nC, img.shape[0], 3)):     # 3 front cameras
            Kc = K[ci].numpy().astype(np.float64)
            Tce = T[ci].numpy().astype(np.float64)     # ego -> cam
            Tec = np.linalg.inv(Tce)
            # back to uint8 grayscale (undo normalization)
            def gray(t):
                im = t.numpy().transpose(1, 2, 0) * STD + MEAN
                return cv2.cvtColor((np.clip(im, 0, 1) * 255).astype(np.uint8),
                                    cv2.COLOR_RGB2GRAY).astype(np.float32)
            g_cur, g_prv = gray(img[ci]), gray(pimgs[ci])
            H_, W_ = g_cur.shape
            sy, sx = dh / H_, dw / W_
            for bi in range(int(c2[ci])):
                cls, cx, cy, w, h = [float(v) for v in b2[ci, bi]]
                if int(cls) not in H_CLS or h <= 1:
                    continue
                u, v = int(round(cx)), int(round(cy))
                if not (R <= u < W_ - R and R <= v < H_ - R):
                    continue
                gy, gx = int(v * sy), int(u * sx)
                if not (0 <= gy < dh and 0 <= gx < dw):
                    continue
                win = dgt[ci, max(0, gy-1):gy+2, max(0, gx-1):gx+2]
                win = win[win > 0.5]
                if win.numel() == 0:
                    continue
                z_gt = float(win.median())
                band = next((b for b in BANDS if b[0] <= z_gt < b[1]), None)
                if band is None:
                    continue
                ref = g_cur[v-R:v+R+1, u-R:u+R+1]
                ref = ref - ref.mean()
                rn = np.linalg.norm(ref) + 1e-6
                ray = np.linalg.inv(Kc) @ np.array([u, v, 1.0])
                best, bz = -2.0, np.nan
                for Z in ZS:
                    Xc = ray * (Z / ray[2])              # cam coords (z=Z)
                    Xe = (Tec @ np.append(Xc, 1.0))[:3]  # current ego
                    Xp = Rz @ Xe + tvec                  # previous ego
                    Xpc = (Tce @ np.append(Xp, 1.0))[:3]
                    if Xpc[2] <= 0.5:
                        continue
                    uv = Kc @ Xpc
                    u2, v2 = uv[0] / uv[2], uv[1] / uv[2]
                    u2i, v2i = int(round(u2)), int(round(v2))
                    if not (R <= u2i < W_ - R and R <= v2i < H_ - R):
                        continue
                    pat = g_prv[v2i-R:v2i+R+1, u2i-R:u2i+R+1]
                    pat = pat - pat.mean()
                    nc = float((ref * pat).sum() / (rn * (np.linalg.norm(pat) + 1e-6)))
                    if nc > best:
                        best, bz = nc, Z
                if not np.isfinite(bz):
                    continue
                err[band]["par"].append(abs(bz - z_gt))
                err[band]["pred"].append(
                    abs(float(zp[ci, max(0, gy-1):gy+2, max(0, gx-1):gx+2].mean()) - z_gt))

    print(f"\nbaseline (translation over 0.4 s) median {np.median(base):.2f} m "
          f"(n={len(base)})")
    print("median depth error at vehicle centers [m]")
    print("   band      n    parallax NCC   depth head   winner")
    for b in BANDS:
        p_, q_ = err[b]["par"], err[b]["pred"]
        if len(p_) < 5:
            continue
        mp, mq = float(np.median(p_)), float(np.median(q_))
        print(f"  {b[0]:>2}-{b[1]:<2}m  {len(p_):>4}   {mp:>7.2f}   {mq:>7.2f}"
              f"     {'parallax' if mp < mq else 'depth head'}")


if __name__ == "__main__":
    main()

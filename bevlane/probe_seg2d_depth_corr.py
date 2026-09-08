#!/usr/bin/env python3
"""課題2 の実測: 2D Seg の乱れと Depth 誤差は相関するか。

フレームごとに (a) seg2d の GT 一致率、(b) 深度の |誤差| 中央値を測り、
相関係数を出す。悪条件 (Cosmos transfer_2) と通常 (val) の両方で見る。
相関が強ければ「共有バックボーン特徴の乱れが両方を壊す」ことの証拠になり、
対処は特徴の頑健化 (悪条件データで深度まで教える) が本命になる。
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    net = MODELS["v52"](n_seg=21).cuda().eval()
    load_net(net, a.ckpt, verbose=False)
    cen = torch.arange(net.D).cuda().float() * net.D_STEP + net.D_MIN
    ds = BevLaneDataset(a.root, a.scenes, gt_key="gt", with_depth=True,
                        with_seg2d=True, seg2d_key="seg2d21", n_cams=8)
    segq, derr = [], []
    for i in range(0, len(ds), a.stride):
        x = ds[i]
        if x is None:
            continue
        img, K, T = x[0], x[1], x[2]
        dgt, sgt = x[4], x[5]
        with torch.no_grad(), torch.autocast("cuda", torch.float16):
            out = net(img[None].cuda(), K[None].cuda(), T[None].cuda())
        s2 = out[2].float()[0].argmax(1).cpu()       # [N,h,w]
        p = out[1].float().flatten(0, 1).softmax(1)
        zp = (p * cen.view(1, -1, 1, 1)).sum(1).cpu()
        sv = sgt != 255
        if sv.sum() < 100:
            continue
        segq.append(float((s2[sv] == sgt[sv]).float().mean()))
        dv = dgt > 0.5
        if dv.sum() < 100:
            segq.pop()
            continue
        derr.append(float((zp[dv] - dgt[dv]).abs().median()))
    sq, de = np.array(segq), np.array(derr)
    r = float(np.corrcoef(sq, de)[0, 1]) if len(sq) > 4 else float("nan")
    print(f"{a.tag}\t{len(sq)}\t{sq.mean():.4f}\t{de.mean():.3f}\t{r:+.3f}")
    # 乱れの大きいフレームの深度悪化量 (下位 1/4 対 上位 1/4)
    if len(sq) > 8:
        o = np.argsort(sq)
        lo, hi = de[o[:len(o)//4]].mean(), de[o[-len(o)//4:]].mean()
        print(f"  seg2d 下位1/4 の深度誤差 {lo:.3f}m 対 上位1/4 {hi:.3f}m "
              f"(比 {lo/max(hi,1e-9):.2f})")


if __name__ == "__main__":
    main()

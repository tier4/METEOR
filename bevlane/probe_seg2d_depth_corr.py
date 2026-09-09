#!/usr/bin/env python3
"""Measurement for issue 2: do 2D Seg disruption and depth error correlate?

Per frame, measure (a) seg2d agreement with GT and (b) median |depth error|, then
compute the correlation. Checked on both adverse (Cosmos transfer_2) and normal (val).
A strong correlation is evidence that disrupted shared backbone features break both,
making feature robustification (teaching depth on adverse data too) the main fix.
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
    # depth degradation on heavily disrupted frames (bottom quarter vs top quarter)
    if len(sq) > 8:
        o = np.argsort(sq)
        lo, hi = de[o[:len(o)//4]].mean(), de[o[-len(o)//4:]].mean()
        print(f"  depth error on seg2d bottom quarter {lo:.3f}m vs top quarter {hi:.3f}m "
              f"(ratio {lo/max(hi,1e-9):.2f})")


if __name__ == "__main__":
    main()

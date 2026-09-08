#!/usr/bin/env python3
"""Measure whether the paint branches are actually used, via contribution ratio.

Injection through a zero-initialized 1x1 projection is function-preserving and safe,
but **training may never use the branch** and leave the weights near zero. v103
paint-seg was exactly that: the addition was 0.06% of the ctx output and dropping
the branch left detection recall identical to 3 decimals. Every injection lever gets this liveness check.

Contribution ratio = mean |injection| / mean |ctx output|.
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bevlane.dataset import BevLaneDataset          # noqa: E402
from bevlane.model import MODELS                    # noqa: E402
from bevlane.ckpt_load import load_net              # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--list", default="val.lst")
    ap.add_argument("--scenes", type=int, default=6)
    ap.add_argument("--per-scene", type=int, default=4)
    ap.add_argument("--root", default="out/bevlane")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = MODELS["v52"](n_seg=21).to(dev).eval()
    load_net(net, a.ckpt)

    acc = {"ctx": [], "seg": [], "det": []}

    def probe(_m, _i, _o):
        d_seg = d_det = None
        if hasattr(net, "paint_proj") and getattr(net, "_paint_buf", None) is not None:
            pb = net._paint_buf.float().softmax(1)[:, net._paint_cls]
            if pb.shape[-2:] != _o.shape[-2:]:
                pb = F.interpolate(pb, size=_o.shape[-2:], mode="bilinear",
                                   align_corners=False)
            d_seg = net.paint_proj(pb.to(_o.dtype))
        if hasattr(net, "paint_det_proj"):
            f = getattr(net, "_last_f", None)
            if f is not None and f.shape[0] == _o.shape[0]:
                hm = net.hm2d_head(net.det2d_stem(f))
                p = hm.sigmoid()[:, net._paint_det_cls].to(_o.dtype)
                if p.shape[-2:] != _o.shape[-2:]:
                    p = F.interpolate(p, size=_o.shape[-2:], mode="bilinear",
                                      align_corners=False)
                d_det = net.paint_det_proj(p)
        base = _o
        for d in (d_seg, d_det):
            if d is not None:
                base = base - d
        acc["ctx"].append(float(base.abs().mean()))
        if d_seg is not None:
            acc["seg"].append(float(d_seg.abs().mean()))
        if d_det is not None:
            acc["det"].append(float(d_det.abs().mean()))

    net.ctx.register_forward_hook(probe)

    scenes = [l.strip() for l in open(a.list) if l.strip()][:a.scenes]
    ds = BevLaneDataset(a.root, scenes, gt_key="gt_cons",
                        max_per_scene=a.per_scene, n_cams=8)
    for i in range(0, len(ds), 2):
        x = ds[i]
        if x is None:
            continue
        with torch.no_grad(), torch.autocast("cuda", torch.float16):
            net(x[0][None].to(dev), x[1][None].to(dev), x[2][None].to(dev))

    c = float(np.mean(acc["ctx"])) if acc["ctx"] else float("nan")
    print(f"\nmean |x| of ctx output = {c:.5f}  (n={len(acc['ctx'])} calls)")
    for nm, key in (("paint-seg", "seg"), ("paint-det", "det")):
        if not acc[key]:
            print(f"  {nm}: no branch")
            continue
        d = float(np.mean(acc[key]))
        r = d / c * 100
        verdict = "alive" if r >= 1.0 else "**dead (< 1%)**"
        print(f"  {nm}: mean |d| added {d:.5f} -> contribution {r:.2f} %  {verdict}")


if __name__ == "__main__":
    main()

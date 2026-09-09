#!/usr/bin/env python3
"""Measure the outlier ratio (max / p99.9) of ReLU outputs. Intermediate check on PACT effectiveness.

The per-tensor INT8 scale is set by the max, so the larger this ratio, the more the
main signal is crushed. Effective INT8 steps = 127 / ratio is the real resolution.
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bevlane.dataset import BevLaneDataset          # noqa: E402
from bevlane.model import MODELS, PACTReLU          # noqa: E402
from bevlane.ckpt_load import load_net              # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--list", default="val.lst")
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--scenes", type=int, default=12)
    ap.add_argument("--prefix", default="", help="show only layers with this prefix")
    ap.add_argument("--pact", default="", help="layers to enable PACT on (not needed if "
                                               "the ckpt has alpha)")
    ap.add_argument("--root", default="out/bevlane")
    a = ap.parse_args()

    net = MODELS["v52"](n_seg=21).cuda().eval()
    sd = torch.load(a.ckpt, map_location="cpu")
    sd = {k.replace("module.", ""): v for k, v in sd.get("model", sd).items()}
    if any(k.endswith(".alpha") for k in sd):
        pats = sorted({k.rsplit(".alpha", 1)[0] for k in sd
                       if k.endswith(".alpha") and "lid_alpha" not in k})
        net.enable_pact(pats, alpha_init={}, verbose=False)
        print(f"[probe] ckpt has alpha -> loading {len(pats)} layers as PACT")
    elif a.pact:
        net.enable_pact(a.pact, alpha_init={}, verbose=False)
    load_net(net, a.ckpt, verbose=False)

    st = {}

    def mk(n):
        def h(_m, _i, _o):
            o = _o.detach().float().flatten()
            if o.numel() > 300000:
                o = o[torch.randperm(o.numel(), device=o.device)[:300000]]
            d = st.setdefault(n, {"p999": [], "max": []})
            d["p999"].append(float(torch.quantile(o, 0.999)))
            d["max"].append(float(o.max()))
        return h

    for n, m in net.named_modules():
        if isinstance(m, (nn.ReLU, PACTReLU)):
            if not a.prefix or n.startswith(tuple(a.prefix.split(","))):
                m.register_forward_hook(mk(n))

    ds = BevLaneDataset(a.root, [l.strip() for l in open(a.list) if l.strip()][:a.scenes],
                        gt_key="gt_cons", max_per_scene=3, n_cams=8)
    c = 0
    for i in range(0, len(ds), 2):
        x = ds[i]
        if x is None:
            continue
        with torch.no_grad(), torch.autocast("cuda", torch.float16):
            net(x[0][None].cuda(), x[1][None].cuda(), x[2][None].cuda())
        c += 1
        if c >= a.frames:
            break

    alphas = {n: float(m.alpha.abs()) for n, m in net.named_modules()
              if isinstance(m, PACTReLU)}
    rows = []
    for n, d in st.items():
        mx, p = float(np.mean(d["max"])), float(np.mean(d["p999"]))
        rows.append((mx / max(p, 1e-6), p, mx, alphas.get(n), n))
    rows.sort(reverse=True)
    print(f"\n{c} frames / {len(rows)} layers. outlier ratio = max / p99.9")
    print(f"{'ratio':>7} {'INT8 steps':>10} {'p99.9':>9} {'max':>9} {'alpha':>9}  layer")
    for r in rows[:20]:
        al = f"{r[3]:9.2f}" if r[3] is not None else "        -"
        print(f"{r[0]:>7.1f} {127 / max(r[0], 1e-9):>10.1f} {r[1]:>9.2f} "
              f"{r[2]:>9.2f} {al}  {r[4]}")


if __name__ == "__main__":
    main()

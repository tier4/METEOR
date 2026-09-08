#!/usr/bin/env python3
"""ReLU 出力の外れ値比 (max / p99.9) を測る。PACT の効きを見る中間チェック用。

per-tensor の INT8 スケールは max で決まるので、この比が大きい層ほど
本体の信号が潰れる。「INT8 実効段階数 = 127 / 比」が実質の分解能。
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
    ap.add_argument("--prefix", default="", help="この接頭辞の層だけ表示")
    ap.add_argument("--pact", default="", help="PACT を有効化する層 (ckpt に "
                                               "alpha があるなら不要)")
    ap.add_argument("--root", default="out/bevlane")
    a = ap.parse_args()

    net = MODELS["v52"](n_seg=21).cuda().eval()
    sd = torch.load(a.ckpt, map_location="cpu")
    sd = {k.replace("module.", ""): v for k, v in sd.get("model", sd).items()}
    if any(k.endswith(".alpha") for k in sd):
        pats = sorted({k.rsplit(".alpha", 1)[0] for k in sd
                       if k.endswith(".alpha") and "lid_alpha" not in k})
        net.enable_pact(pats, alpha_init={}, verbose=False)
        print(f"[probe] ckpt に alpha あり -> {len(pats)} 層を PACT 化して読む")
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
    print(f"\n{c} フレーム / {len(rows)} 層。外れ値比 = max / p99.9")
    print(f"{'比':>7} {'INT8実効段':>10} {'p99.9':>9} {'max':>9} {'alpha':>9}  層")
    for r in rows[:20]:
        al = f"{r[3]:9.2f}" if r[3] is not None else "        -"
        print(f"{r[0]:>7.1f} {127 / max(r[0], 1e-9):>10.1f} {r[1]:>9.2f} "
              f"{r[2]:>9.2f} {al}  {r[4]}")


if __name__ == "__main__":
    main()

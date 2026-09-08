#!/usr/bin/env python3
"""ego のプール出力のチャネル毎 平均/分散 を測って JSON に保存する。

convert_ego_pool はこの統計を BN の running stats に入れ、ego_mlp の第 1 層で
打ち消すことで **変換直後の出力を一致させる** (機能保存)。統計が無いと
BN が恒等にならず、変換した瞬間に ego の出力が桁で変わってしまう。
"""
import argparse
import json
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bevlane.dataset import BevLaneDataset          # noqa: E402
from bevlane.model import MODELS                    # noqa: E402
from bevlane.ckpt_load import load_net              # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--list", default="val.lst")
    ap.add_argument("--scenes", type=int, default=8)
    ap.add_argument("--frames", type=int, default=48)
    ap.add_argument("--root", default="out/bevlane")
    ap.add_argument("--out", default="out/ego_pool_stats.json")
    a = ap.parse_args()

    net = MODELS["v52"](n_seg=21).cuda().eval()
    load_net(net, a.ckpt, verbose=False)
    seq = net.ego_stem
    pi = [i for i, m in enumerate(seq) if isinstance(m, nn.AdaptiveAvgPool2d)]
    assert pi, "AdaptiveAvgPool が無い (既に変換済み?)"
    buf = {}
    seq[pi[-1] - 1].register_forward_hook(
        lambda m, i, o: buf.__setitem__("x", o.detach().float()))

    ds = BevLaneDataset(a.root, [l.strip() for l in open(a.list) if l.strip()][:a.scenes],
                        gt_key="gt_cons", max_per_scene=8, n_cams=8)
    P = []
    for i in range(len(ds)):
        x = ds[i]
        if x is None:
            continue
        with torch.no_grad(), torch.autocast("cuda", torch.float16):
            net(x[0][None].cuda(), x[1][None].cuda(), x[2][None].cuda())
        P.append(F.adaptive_avg_pool2d(buf["x"], 1).flatten(1))
        if len(P) >= a.frames:
            break
    P = torch.cat(P, 0)
    hw = tuple(buf["x"].shape[-2:])
    mu, va = P.mean(0), P.var(0, unbiased=False)
    json.dump({"hw": [int(hw[0]), int(hw[1])],
               "mean": mu.cpu().tolist(), "var": va.cpu().tolist(),
               "frames": int(P.shape[0]), "ckpt": a.ckpt},
              open(a.out, "w"))
    step_in = float(buf["x"].max() - buf["x"].min()) / 127
    ac = float(P.std(0).mean())
    print(f"{P.shape[0]} フレーム / プール手前 {hw} / ch {P.shape[1]}")
    print(f"  現行 (入力の物差し {step_in:.4f}): 変動 {ac:.5f} = "
          f"{ac / step_in:.3f} 段階")
    Q = (P - mu) / va.clamp(min=1e-8).sqrt()
    print(f"  conv+BN 後: 変動 {float(Q.std(0).mean()):.4f} / 1段階 "
          f"{float(Q.max()-Q.min())/127:.5f} = "
          f"{float(Q.std(0).mean())/(float(Q.max()-Q.min())/127):.2f} 段階")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()

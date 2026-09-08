#!/usr/bin/env python3
"""Rescale conv->BN pairs so the conv output stops overflowing fp16.

This is a repair, not a tuning knob: the transform leaves the network's function
unchanged to within float rounding.

A convolution that feeds straight into a BatchNorm has a FREE scale. BN divides
by the (batch or running) standard deviation of its input, so multiplying that
conv's weight by any s > 0 and the BN's running_mean by s and running_var by s^2
gives bit-comparable outputs. Nothing in the loss pushes s toward 1, so it
drifts -- the same flat-direction pathology that let the depth head walk its MAE
from 3.1 m to 30 m.

Here it drifted far enough to break training outright. `seg_head.out.1`
(BatchNorm2d, 96ch) carried running_var 4.3e6 after r58 and 2.6e7 after r59,
i.e. the conv before it emits activations with std ~5000. In autocast fp16
anything past 65504 is inf, so `seg_head.out.0` started returning inf on finite
input, `clamp(-30, 30)` could not help (clamp of NaN is NaN, and inf-inf in the
BN gives NaN), and the training loop discarded the step. r59 skipped 0 % of its
steps to step 8k and then 98-100 % of every step after that -- 69 % of the whole
round, 23,533 of 34,280 steps -- while still reporting plausible val numbers,
because the weights simply stopped moving.

    python3 bevlane/renorm_convbn.py --ckpt out/bevlane_ckpt_r59/best_e2e.pt \
        --out out/bevlane_ckpt_r59/best_e2e_renorm.pt --target-std 1.0
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bevlane  # noqa: F401,E402  enables the torch>=2.6 weights_only compatibility shim


def pairs(sd):
    """-> [(conv weight key, conv bias key or None, bn prefix)] for conv->BN.

    Matched by name: a BatchNorm at <p>.<i> whose immediately preceding index
    <i-1> holds a conv weight. That is how every block in this model is built
    (Conv2d, BatchNorm2d, ReLU as consecutive Sequential entries), and it avoids
    having to instantiate and trace the graph just to repair a state dict.
    """
    out = []
    for k in sd:
        if not k.endswith(".running_var"):
            continue
        p = k[: -len(".running_var")]
        head, _, idx = p.rpartition(".")
        if not idx.isdigit():
            continue
        cw = f"{head}.{int(idx) - 1}.weight"
        if cw not in sd or sd[cw].dim() != 4:
            continue
        cb = f"{head}.{int(idx) - 1}.bias"
        out.append((cw, cb if cb in sd else None, p))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-std", type=float, default=1.0,
                    help="std the conv output is rescaled to, per BN")
    ap.add_argument("--min-var", type=float, default=100.0,
                    help="leave a pair alone unless its running_var exceeds "
                         "this; healthy pairs are near 1 and rescaling them "
                         "would be churn for nothing")
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu")
    sd = ck["model"]
    pfx = "module." if any(k.startswith("module.") for k in sd) else ""
    plain = {k[len(pfx):]: v for k, v in sd.items()} if pfx else sd

    n = 0
    for cw, cb, bn in pairs(plain):
        rv = plain[f"{bn}.running_var"]
        mx = float(rv.max())
        if mx <= a.min_var:
            continue
        s = (mx ** 0.5) / a.target_std
        plain[cw] = plain[cw] / s
        if cb is not None:
            plain[cb] = plain[cb] / s
        plain[f"{bn}.running_mean"] = plain[f"{bn}.running_mean"] / s
        plain[f"{bn}.running_var"] = rv / (s * s)
        print(f"  {bn:44s} running_var {mx:11.4g} -> "
              f"{float(plain[f'{bn}.running_var'].max()):8.4g}  (s={s:.4g})")
        n += 1
    if n == 0:
        print(f"no conv->BN pair has running_var above {a.min_var}")
    ck["model"] = {f"{pfx}{k}": v for k, v in plain.items()} if pfx else plain
    torch.save(ck, a.out)
    print(f"\n{n} pairs rescaled -> {a.out}")


if __name__ == "__main__":
    main()

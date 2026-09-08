#!/usr/bin/env python3
"""Uniform weight soup over checkpoints that share one architecture.

Averaging the weights of runs fine-tuned from nearby starts is free and often
worth a few hundredths (Wortsman et al., "Model soups"). r59 / r60 / r61 are all
v52 and each warm-started from the previous one's best, so their soup is
well-posed: they sit in the same basin and the average is not interpolating
between unrelated solutions.

The reason to expect anything at all is the same reason an EMA helps here --
consecutive epochs bounce. r61 went ADE 0.63 / 0.66 / 0.62 / 0.63 over its last
four evaluations while nothing about the data changed, which is weights orbiting
the basin rather than sitting in it.

BatchNorm running statistics are averaged along with the weights. That is the
standard recipe and is defensible when the inputs are identical across the
runs -- which they are, same corpus, same augmentation. It is also the part most
likely to misbehave, so the soup is judged like any other candidate: measured on
val, never assumed.

    python3 bevlane/make_soup.py --out out/soup_596061.pt \\
        out/bevlane_ckpt_r59/best_e2e_renorm.pt \\
        out/bevlane_ckpt_r60/best_e2e.pt \\
        out/bevlane_ckpt_r61/best_e2e.pt
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bevlane  # noqa: F401,E402  torch>=2.6 の weights_only 互換シム


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--key", default="model")
    a = ap.parse_args()

    acc = None
    n_int = 0
    for c in a.ckpts:
        sd = torch.load(c, map_location="cpu")[a.key]
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        if acc is None:
            acc = {k: (v.double().clone() if v.is_floating_point() else v.clone())
                   for k, v in sd.items()}
            ref_keys = set(acc)
        else:
            if set(sd) != ref_keys:
                raise SystemExit(f"{c} のキー集合が違います "
                                 f"(+{len(set(sd) - ref_keys)} "
                                 f"-{len(ref_keys - set(sd))})")
            for k in acc:
                if acc[k].is_floating_point():
                    acc[k] += sd[k].double()
                else:
                    # num_batches_tracked and friends: keep the last, do not
                    # average an integer counter into a fraction
                    acc[k] = sd[k].clone()
                    n_int += 1
        print(f"[soup] + {c}")

    m = len(a.ckpts)
    out = {k: (v / m).to(torch.float32) if v.is_floating_point() else v
           for k, v in acc.items()}
    ck = torch.load(a.ckpts[0], map_location="cpu")
    ck[a.key] = out
    ck["soup_of"] = a.ckpts
    torch.save(ck, a.out)
    print(f"\n{m} 個を平均 ({n_int // max(m - 1, 1)} 個の整数バッファは最後の値を採用)"
          f" -> {a.out}")


if __name__ == "__main__":
    main()

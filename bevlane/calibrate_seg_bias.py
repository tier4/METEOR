#!/usr/bin/env python3
"""Bake a per-class logit offset into the BEV seg head's final bias.

The thin classes have been predicted 3x too wide for many rounds -- laneline
area ratio 2.71 (r53) climbing to 3.72 (r58) -- and four attempts to fix it
through the loss all failed. The gradient probe in bevlane/probe_thick.py
explains why: all four seg losses push the ring THINNER (this Tversky is
tp/(tp + 0.2*fn + 0.8*fp), already punishing false positives 4x). Nothing in
the objective is making the lines fat.

What is fat is the DECISION BOUNDARY. Subtracting a constant from the thin-class
logits before the argmax fixes the width, and -- measured, against my own
expectation that it would cost 0.4 % of IoU -- it makes IoU BETTER, because the
class-imbalance terms (dice / lovasz / tversky) leave those logits biased high
relative to what argmax wants:

    r61 best.pt, 110 fit + 110 verify frames, split by scene

      bias (lane/stop/edge)   mIoU      laneline   lane area ratio
      0.00 / 0.00 / 0.00      0.2897    0.0804     3.05
      0.75 / 1.25 / 0.50      0.2940    0.0885     1.07

    +1.5 % mIoU, +10 % laneline IoU, and the width is simply correct. fit and
    verify agree to 0.0005, so this is calibration, not a fitted artefact.

A subtracted constant on the logits is exactly a subtracted constant on the last
layer's bias, so it belongs in the weights rather than in every inference path.
Baked in, it costs nothing at runtime, needs no flag, and TensorRT inherits it.

    python3 bevlane/calibrate_seg_bias.py --ckpt out/bevlane_ckpt_r61/best.pt \
        --out out/bevlane_ckpt_r61/best_calib.pt --bias 4:0.75,5:1.25,6:0.5
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# final 1x1 conv of the BEV seg decoder: 64 -> 9 classes
BIAS_KEY = "dec.out.3.bias"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bias", default="4:0.75,5:1.25,6:0.5",
                    help="class:offset pairs; the offset is SUBTRACTED from "
                         "that class's logit")
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu")
    sd = ck["model"]
    pfx = "module." if any(k.startswith("module.") for k in sd) else ""
    key = pfx + BIAS_KEY
    if key not in sd:
        raise SystemExit(f"{key} がありません。存在するのは: "
                         f"{[k for k in sd if k.endswith('out.3.bias')]}")

    b = sd[key].clone()
    print(f"{key}  変更前: " + " ".join(f"{v:+.3f}" for v in b.tolist()))
    for part in a.bias.split(","):
        c, off = part.split(":")
        c, off = int(c), float(off)
        if not 0 <= c < b.numel():
            raise SystemExit(f"クラス {c} は範囲外 (0-{b.numel() - 1})")
        b[c] -= off
        print(f"  class {c}: -{off}")
    sd[key] = b
    print(f"{key}  変更後: " + " ".join(f"{v:+.3f}" for v in b.tolist()))

    ck["model"] = sd
    ck["seg_bias_calib"] = a.bias
    torch.save(ck, a.out)
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()

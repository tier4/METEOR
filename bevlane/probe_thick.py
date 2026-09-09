#!/usr/bin/env python3
"""Which loss term fattens the thin classes? Attribute it by gradient.

laneline has been getting steadily wider every round -- predicted area over GT
area went 2.71 (r53) -> 2.96 -> 3.30 -> 3.60 -> 3.72 (r58) -- and four guesses
at the cause have missed. Raising --tversky-w from 0.6 to 1.2 did nothing, and
reading the implementation showed why the first guess was backwards: this
Tversky is ti = tp / (tp + alpha*fn + beta*fp) with alpha=0.2, beta=0.8, so it
already punishes false positives four times harder than misses and is pushing
THINNER, not fatter.

So stop guessing and measure. A constant subtracted from the thin-class logits
at inference fixes the width almost for free (2.79 -> 1.17 area ratio for -0.4 %
of laneline IoU), which says the spatial profile is about right and the decision
boundary sits too low -- i.e. some term is pushing the thin-class logit UP in
the ring of cells just outside the true line. This backprops each seg loss on
its own and reports the mean gradient it puts on the laneline channel in three
places:

    core : cells the GT calls laneline
    ring : cells within 2 of a GT laneline cell that the GT calls something else
    far  : everything else

A term that fattens shows a NEGATIVE gradient on the ring -- gradient descent
then raises that logit, and the argmax region grows outward.

    CUDA_VISIBLE_DEVICES=7 python3 bevlane/probe_thick.py \
        --ckpt out/bevlane_ckpt_r58/last.pt --model v52
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bevlane.dataset import BevLaneDataset                      # noqa: E402
from bevlane.model import MODELS                                # noqa: E402
from bevlane.train import (DICE_CLASSES, LINE_CLASSES,          # noqa: E402
                           boundary_weight, dice_loss,
                           lovasz_softmax, tversky_loss)

CLS = {4: "laneline", 5: "stopline", 6: "road_edge"}


def rings(gt, c, r=2):
    """core / ring / far masks for class c."""
    core = (gt == c)
    k = 2 * r + 1
    near = F.max_pool2d(core.float()[None, None], k, 1, r)[0, 0] > 0.5
    return core, near & ~core, ~near


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", default="v52")
    ap.add_argument("--root", default="out/bevlane")
    ap.add_argument("--val-list", default="val.lst")
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--n-seg2d", type=int, default=21)
    # the weights the rounds actually run with
    ap.add_argument("--seg-w", type=float, default=1.0)
    ap.add_argument("--dice-w", type=float, default=0.5)
    ap.add_argument("--lovasz-w", type=float, default=0.5)
    ap.add_argument("--tversky-w", type=float, default=1.2)
    ap.add_argument("--boundary-w", type=float, default=3.0)
    ap.add_argument("--far-w", type=float, default=1.0)
    a = ap.parse_args()

    ds = BevLaneDataset(a.root, [l.strip() for l in open(a.val_list)][:12],
                        gt_key="gt_cons", max_per_scene=4)
    net = MODELS[a.model](n_seg=a.n_seg2d).cuda().eval()
    sd = torch.load(a.ckpt, map_location="cpu")["model"]
    net.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()},
                        strict=False)

    acc = {}
    n = 0
    for i in range(0, len(ds), max(1, len(ds) // a.frames)):
        b = ds[i]
        if b is None:
            continue
        gt = b[3][None].cuda()
        with torch.no_grad(), torch.autocast("cuda", torch.float16):
            o = net(b[0][None].cuda(), b[1][None].cuda(), b[2][None].cuda())
        base = (o[0] if isinstance(o, tuple) else o).float().detach()
        H2, W2 = base.shape[-2:]

        terms = {}
        ig = gt.clone()
        terms["CE(+boundary,far)"] = lambda lg: _ce(lg, ig, a, H2)
        terms["dice"] = lambda lg: a.dice_w * dice_loss(lg, ig,
                                                        classes=DICE_CLASSES)
        terms["lovasz"] = lambda lg: a.lovasz_w * lovasz_softmax(lg, ig,
                                                                 ignore=255)
        terms["tversky"] = lambda lg: a.tversky_w * tversky_loss(
            lg, ig, classes=LINE_CLASSES, alpha=0.2, beta=0.8)

        for name, fn in terms.items():
            lg = base.clone().requires_grad_(True)
            loss = fn(lg)
            g, = torch.autograd.grad(loss, lg)
            for c in CLS:
                core, ring, far = rings(gt[0], c)
                d = acc.setdefault((name, c), np.zeros(3))
                gc = g[0, c]
                for j, m in enumerate((core, ring, far)):
                    if m.any():
                        d[j] += float(gc[m].mean())
        n += 1
        if n >= a.frames:
            break

    print(f"\n{os.path.basename(a.ckpt)}  n={n} frames")
    print("gradient sign: negative ring = pushes that cell's logit up = thickens\n")
    for c in CLS:
        print(f"--- {CLS[c]} (class {c}) ---")
        print(f"{'loss term':22s} {'core':>12s} {'ring':>12s} {'far':>12s}")
        tot = np.zeros(3)
        for name in ("CE(+boundary,far)", "dice", "lovasz", "tversky"):
            d = acc.get((name, c))
            if d is None:
                continue
            d = d / n
            tot += d
            flag = "  * thickens" if d[1] < -1e-9 else ""
            print(f"{name:22s} {d[0]:12.3e} {d[1]:12.3e} {d[2]:12.3e}{flag}")
        print(f"{'total':22s} {tot[0]:12.3e} {tot[1]:12.3e} {tot[2]:12.3e}\n")


def _ce(lg, gt, a, H2):
    ce = F.cross_entropy(lg, gt, ignore_index=255, reduction="none")
    w = torch.ones_like(ce)
    if a.far_w > 0:
        rows = torch.arange(H2, device=ce.device, dtype=ce.dtype)
        wrow = 1 + a.far_w * (rows - (H2 - 1) / 2).abs() / ((H2 - 1) / 2)
        w = w * wrow.view(1, -1, 1)
    if a.boundary_w > 0:
        w = w * boundary_weight(gt, radius=2, w=1 + a.boundary_w)
    return a.seg_w * (ce * w).mean()


if __name__ == "__main__":
    main()

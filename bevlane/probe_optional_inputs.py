#!/usr/bin/env python3
"""A/B probe for v47's optional inputs on the val split.

Same weights, same frames, three arms:
  A: camera-only (sdmap=None, tl=None)
  B: + SD map
  C: + SD map + traffic-light boxes
Reports road/crosswalk IoU (full range and the intersection-vicinity cells
beyond 20 m -- the "beyond the cross-road" region the SD map should help),
plus E2E ADE. History is zeroed identically in all arms, so deltas are
attributable to the inputs.
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bevlane.dataset import BevLaneDataset                    # noqa: E402
from bevlane.model import MODELS, EGO_K                       # noqa: E402
from bevlane.train import split_scenes                        # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out/bevlane_ckpt_r45/last.pt")
    ap.add_argument("--root", default="out/bevlane")
    ap.add_argument("--batches", type=int, default=120)
    a = ap.parse_args()

    dev = "cuda"
    m = MODELS["v47"](n_seg=21).to(dev).eval()
    m.load_state_dict(torch.load(a.ckpt, map_location="cpu")["model"],
                      strict=False)

    _, val_s = split_scenes(a.root)
    ds = BevLaneDataset(a.root, val_s, gt_key="gt_cons", max_per_scene=2,
                        with_ego=True, with_sdmap=True, with_tlin=True)
    dl = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False,
                                     num_workers=8)
    print(f"val scenes={len(val_s)} samples={len(ds)}", flush=True)

    arms = ["cam", "+sd", "+sd+tl"]
    inter = {k: np.zeros(2) for k in arms}   # [road, crosswalk]
    union = {k: np.zeros(2) for k in arms}
    inter_ix = {k: np.zeros(1) for k in arms}  # road IoU in isec>20m region
    union_ix = {k: np.zeros(1) for k in arms}
    ade = {k: 0.0 for k in arms}
    n_ade = 0

    with torch.no_grad():
        for bi, batch in enumerate(dl):
            if bi >= a.batches:
                break
            imgs, K, Tc, gt = [t.to(dev) for t in batch[:4]]
            ego_gt = batch[4].to(dev)
            sd = batch[5].to(dev)
            tl = batch[6].to(dev)
            v0 = ego_gt[:, 12]
            has_sd = sd.abs().sum((1, 2, 3)) > 0
            # intersection-vicinity mask beyond 20 m: sdmap ch2 dilated
            ix = F.max_pool2d(sd[:, 2:3], 51, 1, 25)[:, 0] > 0
            ix = F.interpolate(ix[:, None].float(), gt.shape[-2:]
                               )[:, 0] > 0.5
            far = torch.zeros_like(ix)
            far[:, :int(gt.shape[-2] * (60 / 160))] = True  # x > +20 m
            ixfar = ix & far & has_sd.view(-1, 1, 1)

            valid_ego = ego_gt[:, 16] > 0.5
            gtw = ego_gt[:, :12].view(-1, 6, 2)
            for arm, kw in (("cam", {}),
                            ("+sd", {"sdmap": sd}),
                            ("+sd+tl", {"sdmap": sd, "tl": tl})):
                out = m(imgs.float(), K.float(), Tc.float(), v0.float(),
                        **kw)
                pred = out[0].argmax(1)
                mval = gt != 255
                for ci, cls in enumerate((1, 3)):
                    p = (pred == cls) & mval
                    g = (gt == cls) & mval
                    inter[arm][ci] += (p & g).sum().item()
                    union[arm][ci] += (p | g).sum().item()
                p = (pred == 1) & mval & ixfar
                g = (gt == 1) & mval & ixfar
                inter_ix[arm][0] += (p & g).sum().item()
                union_ix[arm][0] += (p | g).sum().item()
                if valid_ego.any():
                    wp = out[7][:, :12 * EGO_K].view(-1, EGO_K, 6, 2)
                    d = (wp - gtw[:, None]).pow(2).sum(-1).sqrt().mean(2)
                    ade[arm] += float(d.min(1).values[valid_ego].mean())
            if valid_ego.any():
                n_ade += 1

    print(f"\n=== A/B probe ({min(a.batches, bi + 1)} batches) ===")
    print(f"{'arm':8s} {'roadIoU':>8s} {'xwalkIoU':>9s} "
          f"{'road@isec>20m':>14s} {'E2E ADE':>8s}")
    for k in arms:
        r = inter[k][0] / max(union[k][0], 1)
        c = inter[k][1] / max(union[k][1], 1)
        rix = inter_ix[k][0] / max(union_ix[k][0], 1)
        print(f"{k:8s} {r:8.4f} {c:9.4f} {rix:14.4f} "
              f"{ade[k] / max(n_ade, 1):8.3f}")


if __name__ == "__main__":
    main()

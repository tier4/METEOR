#!/usr/bin/env python3
"""BEV seg + E2E on a val slice that actually spans the val set.

The epoch-end val inside train.py used to read the FIRST `max_batches` batches
of the val loader -- 10 of 270 scenes at val_batch 2, 5 (and zero turn frames,
so ADEc=nan) at val_batch 1. Numbers from that slice describe a handful of
scenes, so any checkpoint comparison drawn from it is unreliable. This script
walks the same evaluation budget with a stride, covering every val scene, and
is the tool to compare checkpoints with.

    python3 bevlane/eval_val_spread.py --ckpt A.pt B.pt --samples 240
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bevlane.dataset import BevLaneDataset                      # noqa: E402
from bevlane.model import MODELS, N_CLASSES                    # noqa: E402
from bevlane.train import CLASS_NAMES                          # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--root", default="out/bevlane")
    ap.add_argument("--val-list", default="val.lst")
    ap.add_argument("--samples", type=int, default=240)
    ap.add_argument("--gt-key", default="gt_cons")
    ap.add_argument("--n-seg2d", type=int, default=21)
    a = ap.parse_args()

    scenes = [l.strip() for l in open(a.val_list) if l.strip()]
    ds = BevLaneDataset(a.root, scenes, gt_key=a.gt_key, max_per_scene=8,
                        with_ego=True)
    st = max(1, len(ds) // a.samples)
    idx = list(range(0, len(ds), st))[:a.samples]
    cov = {ds.items[i][0] for i in idx}
    print(f"val {len(ds)} samples / {len(scenes)} scenes -> slice "
          f"{len(idx)} samples over {len(cov)} scenes (stride {st})")

    for p in a.ckpt:
        ck = torch.load(p, map_location="cpu")
        mv = a.model or (ck.get("args") or {}).get("model") or "v48"
        m = MODELS[mv](n_seg=a.n_seg2d).cuda().eval()
        m.load_state_dict({k.replace("module.", ""): v
                           for k, v in ck["model"].items()}, strict=False)
        inter = np.zeros(N_CLASSES)
        union = np.zeros(N_CLASSES)
        ade = fde = adec = 0.0
        n = nc = 0
        with torch.no_grad():
            for i in idx:
                b = ds[i]
                if b is None:
                    continue
                # v0 (current speed) is a REQUIRED input of the E2E head:
                # without it the predicted path has no scale and ADE reads
                # ~10 m. train.py's evaluate_ego passes eg[:, 12]; do the same.
                eg0 = b[4] if len(b) > 4 else None
                v0 = (eg0[12].view(1).cuda().float()
                      if eg0 is not None and eg0.numel() >= 17 else None)
                with torch.autocast("cuda", torch.float16):
                    o = m(b[0][None].cuda(), b[1][None].cuda(),
                          b[2][None].cuda(), v0)
                seg = (o[0] if isinstance(o, tuple) else o)
                pred = seg.argmax(1)[0].cpu().numpy()
                g = b[3].numpy()
                msk = g > 0
                for c in range(1, N_CLASSES):
                    pi, gi = (pred == c) & msk, g == c
                    inter[c] += (pi & gi).sum()
                    union[c] += (pi | gi).sum()
                if isinstance(o, tuple) and len(o) > 7 and len(b) > 4:
                    eg = b[4]
                    if eg.numel() >= 17 and float(eg[16]) > 0.5:
                        wp = o[7][0, :12].float().view(6, 2).cpu().numpy()
                        gw = eg[:12].view(6, 2).numpy()
                        d = np.linalg.norm(wp - gw, axis=1)
                        ade += d.mean()
                        fde += d[-1]
                        n += 1
                        if abs(float(eg[11])) > 2.0:
                            adec += d.mean()
                            nc += 1
        iou = {CLASS_NAMES[c]: (inter[c] / union[c] if union[c] else float("nan"))
               for c in range(1, N_CLASSES)}
        miou = float(np.nanmean(list(iou.values())))
        print(f"\n{os.path.basename(os.path.dirname(p))}/{os.path.basename(p)} "
              f"(epoch {ck.get('epoch')})")
        print(f"  mIoU {miou:.4f}  " +
              "  ".join(f"{k}={v:.3f}" for k, v in iou.items()))
        if n:
            print(f"  E2E  ADE={ade / n:.3f}m FDE={fde / n:.3f}m "
                  f"ADEc={adec / nc if nc else float('nan'):.3f}m "
                  f"(n={n} nc={nc})")
        del m
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

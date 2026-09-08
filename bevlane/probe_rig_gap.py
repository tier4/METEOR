#!/usr/bin/env python3
"""Two user-visible symptoms, measured separately.

  1. "lanes disappear when the 8-camera rig is cut to 7"
  2. "on x2gen2 the BEV is coarse and NO 3D boxes appear at all"

Both were observed in one video where the 7-camera scenes are ALSO the x2gen2
scenes, so rig and domain are confounded. This probe separates them:

    A) JP 8-camera, full rig
    B) JP 8-camera, CAM_BACK_NARROW zeroed   -> the rig effect alone
    C) x2gen2 7-camera as delivered          -> rig + domain

and reports, per arm: laneline/stopline/road_edge IoU, and the 3D detection
head's score distribution (max score, boxes over the demo's 0.25 threshold),
which tells a threshold problem apart from a head that predicts nothing.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bevlane.dataset import BevLaneDataset                      # noqa: E402
from bevlane.model import MODELS, N_CLASSES                     # noqa: E402
from bevlane.train import CLASS_NAMES                           # noqa: E402


@torch.no_grad()
def arm(model, ds, idx, zero7=False, thr=0.25):
    inter = np.zeros(N_CLASSES)
    union = np.zeros(N_CLASSES)
    smax, nbox, nfr, nany = [], [], 0, 0
    for i in idx:
        b = ds[i]
        if b is None:
            continue
        imgs = b[0].clone()
        if zero7:
            imgs[7] = 0.0
        v0 = None
        with torch.autocast("cuda", torch.float16):
            out = model(imgs[None].cuda(), b[1][None].cuda(),
                        b[2][None].cuda(), v0)
        seg = (out[0] if isinstance(out, tuple) else out)
        pred = seg.argmax(1)[0].cpu().numpy()
        g = b[3].numpy()
        msk = g > 0
        for c in range(1, N_CLASSES):
            pi, gi = (pred == c) & msk, g == c
            inter[c] += (pi & gi).sum()
            union[c] += (pi | gi).sum()
        if isinstance(out, tuple) and len(out) > 4:
            hm = out[3].float().sigmoid()
            smax.append(float(hm.max()))
            k = int((hm > thr).sum())
            nbox.append(k)
            nany += int(k > 0)
        nfr += 1
    iou = {CLASS_NAMES[c]: (inter[c] / union[c] if union[c] else float("nan"))
           for c in range(1, N_CLASSES)}
    return iou, nfr, (np.mean(smax) if smax else float("nan"),
                      np.max(smax) if smax else float("nan"),
                      np.mean(nbox) if nbox else float("nan"),
                      nany)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--root", default="out/bevlane")
    ap.add_argument("--jp-list", default="val.lst")
    ap.add_argument("--x2-list", default="out/x2gen2_test.txt")
    ap.add_argument("--scenes", type=int, default=14)
    ap.add_argument("--frames", type=int, default=80)
    ap.add_argument("--thr", type=float, default=0.25)
    ap.add_argument("--n-seg2d", type=int, default=21)
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu")
    mv = a.model or (ck.get("args") or {}).get("model") or "v48"
    m = MODELS[mv](n_seg=a.n_seg2d).cuda().eval()
    m.load_state_dict({k.replace("module.", ""): v
                       for k, v in ck["model"].items()}, strict=False)
    print(f"ckpt {a.ckpt} | {mv} | epoch {ck.get('epoch')} | thr {a.thr}")

    def mk(lst):
        sc = [l.strip() for l in open(lst) if l.strip()][:a.scenes]
        sc = [s for s in sc
              if os.path.exists(os.path.join(a.root, s, "manifest.json"))]
        d = BevLaneDataset(a.root, sc, gt_key="gt_cons", max_per_scene=8)
        st = max(1, len(d) // a.frames)
        return d, list(range(0, len(d), st))[:a.frames]

    djp, ijp = mk(a.jp_list)
    dx2, ix2 = mk(a.x2_list)
    arms = [("A JP 8-cam full ", djp, ijp, False),
            ("B JP cam7 zeroed", djp, ijp, True),
            ("C x2gen2 7-cam  ", dx2, ix2, False)]
    print(f"\n{'arm':17s} {'n':>4s} {'laneline':>9s} {'stopline':>9s} "
          f"{'road_edge':>10s} {'road':>7s} {'marking':>8s} | "
          f"{'hm mean':>8s} {'hm max':>7s} {'>thr':>6s} {'frames w/ box':>14s}")
    for nm, d, idx, z in arms:
        iou, n, (sm, sx, nb, nany) = arm(m, d, idx, z, a.thr)
        print(f"{nm} {n:4d} {iou['laneline']:9.4f} {iou['stopline']:9.4f} "
              f"{iou['road_edge']:10.4f} {iou['road']:7.4f} "
              f"{iou['marking']:8.4f} | {sm:8.3f} {sx:7.3f} {nb:6.1f} "
              f"{100 * nany / max(n, 1):13.0f}%")


if __name__ == "__main__":
    main()

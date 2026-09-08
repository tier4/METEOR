#!/usr/bin/env python3
"""Measure what dropping CAM_BACK_NARROW costs, on both rigs.

Three questions, all answered by measurement on the same weights:

  A) 8-camera scenes, full rig vs CAM_BACK_NARROW zeroed
     -> what the 7-camera configuration costs on data that HAS the camera.
  B) 7-camera scenes (x2gen2) as delivered
     -> that the 7-camera path runs at all and what it scores.
  C) bit-equality: an 8-camera sample with the 8th image zeroed by the
     dataset must give the SAME output as zeroing it in the model
     (model.zero_cams), i.e. the two ways of expressing "no camera there"
     agree. If they diverge, train and inference disagree.

Usage:
  python3 bevlane/probe_cam_config.py --ckpt out/bevlane_ckpt_r48/last.pt \
      --scenes8 val.lst --scenes7 out/x2gen2_test.txt --frames 60
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bevlane.dataset import BevLaneDataset, CAMS      # noqa: E402
from bevlane.model import MODELS, N_CLASSES           # noqa: E402


def load_list(p, n):
    s = [l.strip() for l in open(p) if l.strip()]
    return s[:n]


@torch.no_grad()
def run(model, ds, frames, zero8=False, device="cuda"):
    """Mean per-class IoU over `frames` samples; optionally zero camera 7."""
    inter = np.zeros(N_CLASSES)
    union = np.zeros(N_CLASSES)
    n = 0
    for i in range(min(frames, len(ds))):
        b = ds[i]
        if b is None:
            continue
        imgs, K, T, gt = b[0], b[1], b[2], b[3]
        if zero8:
            imgs = imgs.clone()
            imgs[7] = 0.0
        with torch.autocast("cuda", torch.float16):
            out = model(imgs[None].to(device), K[None].to(device),
                        T[None].to(device))
        seg = out[0] if isinstance(out, tuple) else out
        pred = seg.argmax(1)[0].cpu().numpy()
        g = gt.numpy()
        for c in range(N_CLASSES):
            pi, gi = pred == c, g == c
            inter[c] += (pi & gi).sum()
            union[c] += (pi | gi).sum()
        n += 1
    iou = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
    return iou, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--root", default="out/bevlane")
    ap.add_argument("--scenes8", default="val.lst")
    ap.add_argument("--scenes7", default="out/x2gen2_test.txt")
    ap.add_argument("--scenes", type=int, default=12)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--gt-key", default="gt_cons")
    ap.add_argument("--n-seg2d", type=int, default=21)
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu")
    ca = ck.get("args", {})
    mv = a.model or ca.get("model", "v48")
    print(f"ckpt {a.ckpt} | model {mv} | epoch {ck.get('epoch')} "
          f"| cam_drop at train time {ca.get('cam_drop')}")
    model = MODELS[mv](n_seg=a.n_seg2d).cuda().eval()
    sd = {k.replace("module.", ""): v for k, v in ck["model"].items()}
    miss = model.load_state_dict(sd, strict=False)
    print(f"  load: missing {len(miss.missing_keys)} "
          f"unexpected {len(miss.unexpected_keys)}")

    dkw = dict(gt_key=a.gt_key, with_depth=False)
    s8 = load_list(a.scenes8, a.scenes)
    d8 = BevLaneDataset(a.root, s8, **dkw)
    print(f"\n8-camera set: {len(d8)} samples / {len(s8)} scenes "
          f"(absent cams: {sum(len(v) for v in d8.absent.values())})")
    i_full, n1 = run(model, d8, a.frames, zero8=False)
    i_zero, _ = run(model, d8, a.frames, zero8=True)
    print(f"  A) same scenes, full rig vs CAM_BACK_NARROW zeroed  (n={n1})")
    print(f"     mIoU {np.nanmean(i_full):.4f} -> {np.nanmean(i_zero):.4f} "
          f"({np.nanmean(i_zero) - np.nanmean(i_full):+.4f})")
    for c, nm in ((1, "road"), (3, "crosswalk"), (4, "laneline"),
                  (5, "stopline"), (6, "road_edge")):
        print(f"     {nm:10s} {i_full[c]:.4f} -> {i_zero[c]:.4f} "
              f"({i_zero[c] - i_full[c]:+.4f})")

    s7 = load_list(a.scenes7, a.scenes)
    d7 = BevLaneDataset(a.root, s7, **dkw)
    nab = sum(len(v) for v in d7.absent.values())
    print(f"\n7-camera set: {len(d7)} samples / {len(s7)} scenes "
          f"(absent cam slots: {nab})")
    if len(d7):
        i7, n2 = run(model, d7, a.frames)
        print(f"  B) as delivered (n={n2})  mIoU {np.nanmean(i7):.4f}")
        for c, nm in ((1, "road"), (3, "crosswalk"), (4, "laneline"),
                      (5, "stopline"), (6, "road_edge")):
            print(f"     {nm:10s} {i7[c]:.4f}")

    # ---- C) the two ways of saying "no camera" must agree ----
    b = d8[0]
    imgs, K, T = b[0], b[1], b[2]
    z = imgs.clone()
    z[7] = 0.0
    with torch.no_grad(), torch.autocast("cuda", torch.float16):
        o_data = model(z[None].cuda(), K[None].cuda(), T[None].cuda())
        model.zero_cams = (7,)
        o_model = model(imgs[None].cuda(), K[None].cuda(), T[None].cuda())
        model.zero_cams = ()
    s_d = (o_data[0] if isinstance(o_data, tuple) else o_data).float()
    s_m = (o_model[0] if isinstance(o_model, tuple) else o_model).float()
    d = (s_d - s_m).abs().max().item()
    print(f"\n  C) dataset-zeroed vs model.zero_cams: max |seg logit diff| "
          f"{d:.3e}  -> {'IDENTICAL' if d < 1e-3 else 'DIVERGENT'}")


if __name__ == "__main__":
    main()

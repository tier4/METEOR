#!/usr/bin/env python3
"""C2: failure auto-mining — find the train scenes the model is worst on.

Runs the given checkpoint over a sample of TRAIN scenes, scores each scene
by (E2E ADE + vehicle-detection miss rate), and writes the worst
--frac fraction to out/mined_scenes.txt; the trainer oversamples them via
--mined-oversample. Our substitute for fleet shadow mode.

Usage:
  python3 bevlane/mine_failures.py --ckpt out/bevlane_ckpt_r27/last.pt \
      --model v33 --list out/round28_scenes.txt --sample 240 --frac 0.15
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bevlane.dataset import BevLaneDataset  # noqa: E402
from bevlane.model import MODELS, make_warp_theta  # noqa: E402


def scene_score(m, scene, root="out/bevlane", mode="combo", vru_w=1.0):
    try:
        ds = BevLaneDataset(root, [scene], gt_key="gt_vec",
                            with_ego=True, with_agenttraj=True,
                            with_temporal=True, temporal_hist=3)
    except Exception:
        return None
    if len(ds) == 0:
        return None
    ades, misses = [], []
    for i in range(2, len(ds), 12):
        b = ds[i]
        try:
            with torch.no_grad(), torch.autocast("cuda", torch.float16):
                K, Tc = b[1][None].cuda(), b[2][None].cuda()
                hi, hr, hv = (b[-3][None].cuda(), b[-2][None].cuda(),
                              b[-1][None].cuda())
                pbs = [m.compute_bev(hi[:, s_], K, Tc)
                       * hv[:, s_].view(-1, 1, 1, 1) for s_ in range(3)]
                ths = [make_warp_theta(hr[:, s_]) for s_ in range(3)]
                out = m(b[0][None].cuda(), K, Tc,
                        torch.tensor([0.0], device="cuda"),
                        torch.stack(pbs, 1).float(), torch.stack(ths, 1))
        except Exception:
            continue
        # batch order with (agenttraj, ego, temporal):
        # 4=boxes 5=count 6=traj 7=tvalid 8=ego_vec
        e = out[7][0].float().cpu().numpy()
        ego_v = b[8]
        gt = np.asarray(ego_v[:12], np.float32).reshape(6, 2) \
            if ego_v.numel() >= 12 else None
        if gt is not None and np.abs(gt).sum() > 0:
            best = None
            for k in range(3):
                wp = e[k * 12:(k + 1) * 12].reshape(6, 2)
                a = float(np.linalg.norm(wp - gt, axis=1).mean())
                best = a if best is None or a < best else best
            ades.append(best)
        # detection miss rate vs GT boxes (near corridor)
        boxes, nb = b[4], int(b[5])
        dets = m.decode_boxes(out[3].float(), out[4].float(),
                              thresh=0.3, topk=64)[0]
        miss = hit = 0.0
        for k in range(nb):
            cls, xe, ye = (float(boxes[k, 0]), float(boxes[k, 1]),
                           float(boxes[k, 2]))
            if boxes[k, 3] <= 0 or not (abs(xe) < 30 and abs(ye) < 12):
                continue
            ok = any((d[2] - xe) ** 2 + (d[3] - ye) ** 2 < 4.0
                     for d in dets)
            # VRU misses can be up-weighted (--vru-w): the vru_diag showed the
            # rear-40 line lost near-field VRU recall and mining is the
            # training-side lever (decode-threshold calibration is the other)
            w = vru_w if cls >= 1.5 else 1.0
            hit += w * ok
            miss += w * (not ok)
        if hit + miss:
            misses.append(miss / (hit + miss))
    if not ades and not misses:
        return None
    if mode == "e2e":
        arr = np.array(ades) if ades else np.zeros(1)
        return float(arr.mean() + 2.0 * (arr > 1.5).mean())
    return (np.mean(ades) if ades else 0.0) / 2.0 \
        + (np.mean(misses) if misses else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--root", default="out/bevlane")
    ap.add_argument("--model", default="v33")
    ap.add_argument("--list", required=True)
    ap.add_argument("--sample", type=int, default=240)
    ap.add_argument("--vru-w", type=float, default=1.0,
                    help="weight on VRU misses in the mining score")
    ap.add_argument("--frac", type=float, default=0.15)
    ap.add_argument("--mode", default="combo", choices=["combo", "e2e"],
                    help="e2e: longitudinal-ADE + tail emphasis (P3)")
    args = ap.parse_args()
    m = MODELS[args.model](n_seg=21).cuda().eval()
    sd = torch.load(args.ckpt, map_location="cpu")["model"]
    cur = m.state_dict()
    m.load_state_dict({k: v for k, v in sd.items()
                       if k in cur and cur[k].shape == v.shape}, strict=False)
    scenes = open(args.list).read().split()
    step = max(1, len(scenes) // args.sample)
    sample = scenes[::step][:args.sample]
    scored = []
    for i, s in enumerate(sample):
        sc = scene_score(m, s, root=args.root, mode=args.mode,
                         vru_w=args.vru_w)
        if sc is not None:
            scored.append((sc, s))
        if i % 40 == 0:
            print(f"{i + 1}/{len(sample)}", flush=True)
    scored.sort(reverse=True)
    n = max(1, int(len(scored) * args.frac))
    # 学習と並行で回すため、書きかけを読まれないよう原子的に置換する
    with open("out/mined_scenes.txt.tmp", "w") as f:
        f.write("\n".join(s for _, s in scored[:n]))
    import os as _os
    _os.replace("out/mined_scenes.txt.tmp", "out/mined_scenes.txt")
    print(f"mined {n}/{len(scored)} scenes "
          f"(worst score {scored[0][0]:.2f}, cut {scored[n - 1][0]:.2f})",
          flush=True)


if __name__ == "__main__":
    main()

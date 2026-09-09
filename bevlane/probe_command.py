"""Measure whether the driving command moves the trajectory and whether the K=3 modes separate.

Same question as the reference implementation (VLA eval_vla_command.py):
  force a left / right command on straight-driving GT frames and read the lateral offset at 3 s.
  If the command works, the path should be pushed sideways even where the scene allows straight.
  Reference CNN target is >= 1 m; measured 0.2 m before mode fusion.

Also checks selector health:
  spread between modes (are the 3 distinct), selection hit rate (is the best mode picked),
  selection loss (error of the chosen mode - error of the best mode).
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bevlane.dataset import BevLaneDataset                        # noqa: E402
from bevlane.model import MODELS, EGO_K                           # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", default="v52")
    ap.add_argument("--n-cams", type=int, default=8)
    ap.add_argument("--list", default="val.lst")
    ap.add_argument("--root", default="out/bevlane")
    ap.add_argument("--scenes", type=int, default=60)
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--straight", type=float, default=0.5,
                    help="max lateral offset at 3 s to count as straight [m]")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    scenes = [l.strip() for l in open(a.list) if l.strip()][:a.scenes]
    ds = BevLaneDataset(a.root, scenes, gt_key="gt_cons", with_ego=True,
                        max_per_scene=4, n_cams=8, trim_start=3, trim_end=10)
    m = MODELS[a.model](n_seg=21).cuda().eval()
    sd = torch.load(a.ckpt, map_location="cpu")
    sd = {k.replace("module.", ""): v for k, v in sd.get("model", sd).items()}
    cur = m.state_dict()
    m.load_state_dict({k: v for k, v in sd.items()
                       if k in cur and cur[k].shape == v.shape}, strict=False)

    CMD = {"none": None, "straight": 0, "left": 1, "right": 2}
    lat = {k: [] for k in CMD}
    mode_sel = {k: [] for k in CMD}
    spread, sel_gap, sel_hit = [], [], []
    step = max(1, len(ds) // a.frames)
    done = 0
    for i in range(0, len(ds), step):
        b = ds[i]
        if b is None:
            continue
        # ego GT: first 12 elements are 6 (x,y) points. Straight = lateral position of the last point
        eg = b[4] if len(b) > 4 and torch.is_tensor(b[4]) and b[4].numel() >= 12 else None
        if eg is None:
            continue
        gt_lat = float(eg[11])
        if abs(gt_lat) > a.straight:            # straight frames only
            continue
        ims = b[0][None][:, :a.n_cams].cuda()
        Kk = b[1][None][:, :a.n_cams].cuda()
        Tc = b[2][None][:, :a.n_cams].cuda()
        for name, idx in CMD.items():
            oh = None
            if idx is not None:
                oh = F.one_hot(torch.tensor([idx]), 3).float().cuda()
            with torch.no_grad(), torch.autocast("cuda", torch.float16):
                out = m(ims, Kk, Tc, intent=oh)
            e = out[7].float()[0]
            wp = e[:12 * EGO_K].view(EGO_K, 6, 2)
            lg = e[12 * EGO_K:12 * EGO_K + EGO_K]
            k = int(lg.argmax())
            lat[name].append(float(wp[k, -1, 1]))
            mode_sel[name].append(k)
            if idx is None:
                # mode spread (max difference in final lateral position)
                ys = wp[:, -1, 1].cpu().numpy()
                spread.append(float(ys.max() - ys.min()))
                # selection quality: ADE of each mode vs GT
                g = eg[:12].view(6, 2).cuda()
                ade = ((wp - g[None]) ** 2).sum(-1).sqrt().mean(1)
                best = int(ade.argmin())
                sel_hit.append(1.0 if best == k else 0.0)
                sel_gap.append(float(ade[k] - ade[best]))
        done += 1
        if done >= a.frames:
            break

    print(f"\n=== {a.tag or a.ckpt} ({done} straight frames) ===")
    base = np.mean(lat["none"]) if lat["none"] else 0.0
    print("command     lateral@3s (mean)    delta vs none      selected-mode histogram")
    for name in CMD:
        if not lat[name]:
            continue
        v = np.mean(lat[name])
        cnt = np.bincount(mode_sel[name], minlength=EGO_K)
        print(f"  {name:<10} {v:+6.2f} m            {v - base:+6.2f} m        "
              f"{list(cnt)}")
    if spread:
        print(f"\nmode spread (max difference in final lateral position): "
              f"median {np.median(spread):.2f} m / mean {np.mean(spread):.2f} m")
        print(f"selection hit rate: {100 * np.mean(sel_hit):.1f}% "
              f"(random 3-way guess is 33.3%)")
        print(f"selection loss (ADE of chosen mode - best mode): "
              f"{np.mean(sel_gap):.3f} m")
    dl = np.mean(lat["left"]) - base if lat["left"] else float("nan")
    dr = base - np.mean(lat["right"]) if lat["right"] else float("nan")
    print(f"\ncommand response: left {dl:+.2f} m / right {dr:+.2f} m "
          f"(reference target >= 1 m; measured 0.2 m before mode fusion)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""What does the RL-trained mode selector actually buy at inference time?

Training reports "pick", the fraction of frames where the selector chose the
highest-reward candidate, and it rose from 0.63 to ~0.77 against a 1/3 chance
baseline. That measures the RL against its own objective and is very nearly
circular. This asks the question the deployed system cares about: given the SAME
network and the SAME three candidate trajectories, does committing to the
selector's choice drive better than the alternatives?

Four policies over the K=3 candidates, scored on identical frames:

  selector : argmax of the mode logits -- what the vehicle would actually do
  oracle   : the candidate closest to the logged future (upper bound; needs GT,
             not available at run time)
  fixed    : always candidate 0 -- what you get with no selector at all
  worst    : the candidate the reward likes least (lower bound)

Reported per policy: ADE/FDE against the logged future, and the rule terms the
reward is built from -- drivable-area occupancy, agent collision, red-light
compliance, comfort, progress -- so a policy that games the distance metric
while driving off the road cannot hide.

    CUDA_VISIBLE_DEVICES=7 python3 bevlane/probe_rl.py \
        --ckpt out/bevlane_ckpt_r53/best_depthswap.pt --model v50 --samples 400
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bevlane.dataset import BevLaneDataset                      # noqa: E402
from bevlane.e2e_reward import candidate_rewards                # noqa: E402
from bevlane.model import EGO_K, MODELS                         # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", default="v50")
    ap.add_argument("--root", default="out/bevlane")
    ap.add_argument("--val-list", default="val.lst")
    ap.add_argument("--scenes", type=int, default=40)
    ap.add_argument("--samples", type=int, default=400)
    ap.add_argument("--n-seg2d", type=int, default=21)
    # The reward MUST match the one training optimised, or "pick" here measures
    # a different objective than the selector was taught. The first run left the
    # traffic-light raster out and used default term weights, and reported a
    # pick of 0.50 against training's 0.73 -- not comparable.
    ap.add_argument("--rl-imit-w", type=float, default=0.5)
    ap.add_argument("--rl-tl-w", type=float, default=1.5)
    ap.add_argument("--no-tl", action="store_true")
    a = ap.parse_args()

    sc = [l.strip() for l in open(a.val_list) if l.strip()][:a.scenes]
    ds = BevLaneDataset(a.root, sc, gt_key="gt_cons", with_ego=True,
                        with_boxdet=True, with_agenttraj=True,
                        with_tlin=not a.no_tl, max_per_scene=6)
    net = MODELS[a.model](n_seg=a.n_seg2d).cuda().eval()
    sd = torch.load(a.ckpt, map_location="cpu")["model"]
    net.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()},
                        strict=False)

    step = max(1, len(ds) // a.samples)
    POL = ["selector", "oracle", "fixed", "worst"]
    ade = {p: [] for p in POL}
    fde = {p: [] for p in POL}
    rew = {p: [] for p in POL}
    parts = {p: {} for p in POL}
    picks = []
    n = 0
    for i in range(0, len(ds), step):
        b = ds[i]
        if b is None:
            continue
        # locate the optional tensors by shape: box list (N,6) + its count,
        # then the ego vector (17,)
        bx = nbx = eg = tj = tv = tl = None
        for j, t in enumerate(b):
            if not torch.is_tensor(t):
                continue
            if t.dim() == 2 and t.shape[-1] == 6 and t.dtype == torch.float32 \
                    and bx is None:
                bx, nbx = t, (b[j + 1] if j + 1 < len(b) else None)
            elif t.dim() == 1 and t.numel() == 17:
                eg = t
            elif t.dim() == 3 and t.shape[-1] == 2 and t.shape[-2] == 6:
                tj = t
            elif t.dim() == 4 and t.shape[0] == 8 and t.shape[1] == 7:
                tl = t
        if eg is None or float(eg[16]) < 0.5:
            continue
        with torch.no_grad(), torch.autocast("cuda", torch.float16):
            o = net(b[0][None].cuda(), b[1][None].cuda(), b[2][None].cuda(),
                    eg[12].view(1).cuda().float())
        if not (isinstance(o, tuple) and len(o) > 7):
            continue
        e = o[7][0].float()
        wp = e[:12 * EGO_K].view(1, EGO_K, 6, 2)
        lg = e[12 * EGO_K:12 * EGO_K + EGO_K].view(1, EGO_K)
        gw = eg[:12].view(1, 6, 2).cuda()

        r, _ = candidate_rewards(
            wp, gt=b[3][None].cuda(),
            boxes=(bx[None].cuda() if bx is not None else None),
            nbox=(nbx.view(1).clamp(min=0).cuda()
                  if torch.is_tensor(nbx) else None),
            traj=(tj[None].cuda() if tj is not None else None),
            tvalid=tv,
            tl=(tl[None].cuda() if tl is not None else None),
            v0=eg[12].view(1).cuda(), ego_gt=eg[None].cuda(),
            w={"imit": a.rl_imit_w, "tl": a.rl_tl_w})
        d = torch.linalg.norm(wp[0] - gw, dim=-1)          # [K,6]
        choice = {"selector": int(lg.argmax()),
                  "oracle": int(d.mean(1).argmin()),
                  "fixed": 0,
                  "worst": int(r[0].argmin())}
        picks.append(int(choice["selector"] == int(r[0].argmax())))
        for p, k in choice.items():
            ade[p].append(float(d[k].mean()))
            fde[p].append(float(d[k][-1]))
            rew[p].append(float(r[0, k]))
            # per-term rewards for THIS candidate alone
            r1, pp = candidate_rewards(
                wp[:, k:k + 1], gt=b[3][None].cuda(),
                boxes=(bx[None].cuda() if bx is not None else None),
                nbox=(nbx.view(1).clamp(min=0).cuda()
                      if torch.is_tensor(nbx) else None),
                traj=(tj[None].cuda() if tj is not None else None),
                tvalid=tv,
                tl=(tl[None].cuda() if tl is not None else None),
                v0=eg[12].view(1).cuda(), ego_gt=eg[None].cuda(),
                w={"imit": a.rl_imit_w, "tl": a.rl_tl_w})
            for kk, vv in pp.items():
                parts[p].setdefault(kk, []).append(float(vv))
        n += 1
        if n >= a.samples:
            break

    print(f"\n{os.path.basename(a.ckpt)}  n={n} フレーム  "
          f"K={EGO_K} 候補  selector的中率 {np.mean(picks):.2f}")
    print(f"\n{'方策':>10s} {'ADE [m]':>9s} {'FDE [m]':>9s} {'報酬':>9s}")
    for p in POL:
        print(f"{p:>10s} {np.mean(ade[p]):9.3f} {np.mean(fde[p]):9.3f} "
              f"{np.mean(rew[p]):9.3f}")
    keys = [k for k in parts["selector"] if isinstance(
        parts["selector"][k][0], float)]
    if keys:
        print(f"\n報酬の内訳（高いほど良い）")
        print(f"{'方策':>10s} " + " ".join(f"{k:>9s}" for k in keys))
        for p in POL:
            print(f"{p:>10s} " + " ".join(
                f"{np.mean(parts[p][k]):9.3f}" for k in keys))
    s, f, o = np.mean(ade["selector"]), np.mean(ade["fixed"]), \
        np.mean(ade["oracle"])
    print(f"\n選択器の効果: fixed {f:.3f} -> selector {s:.3f} m "
          f"({f - s:+.3f} m)、oracle は {o:.3f} m")
    if f > o:
        print(f"  oracle との差の {100 * (f - s) / (f - o):.0f} % を回収")


if __name__ == "__main__":
    main()

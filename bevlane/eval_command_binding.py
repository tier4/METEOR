#!/usr/bin/env python3
"""Does a Driving Command actually move the planned path?

Two different questions, deliberately separated (docs/FIX_COMMAND_BINDING.md):

  A) within ONE forward, how far apart are the K hypotheses?    (lat@3s spread)
  B) how far apart are the SELECTED paths when the command changes?

If A == B the command buys nothing beyond picking one of K similar paths. The
decisive number is the sign-reversal rate: on frames whose GT turns right, how
often does commanding "left" produce a left-going path.

    python3 bevlane/eval_command_binding.py --ckpt out/.../best.pt --frames 50
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bevlane.dataset import BevLaneDataset                     # noqa: E402
from bevlane.model import MODELS, EGO_K                        # noqa: E402
from bevlane.train import split_scenes                         # noqa: E402

LAT = 11          # ego_gt column: lateral offset of the 3 s waypoint
CMDS = (("none", None), ("left", 1), ("right", 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", default=None,
                    help="default: read args.model from the checkpoint")
    ap.add_argument("--root", default="out/bevlane")
    ap.add_argument("--frames", type=int, default=50)
    ap.add_argument("--turn-m", type=float, default=2.0,
                    help="|lat@3s| above this counts as a turn frame")
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu")
    mv = a.model or (ck.get("args") or {}).get("model") or "v48"
    print(f"ckpt {a.ckpt} | model {mv} | epoch {ck.get('epoch')} "
          f"| git {(ck.get('git') or '?')[:8]}", flush=True)
    if ck.get("args"):
        ia = {k: v for k, v in ck["args"].items() if "intent" in k}
        print("  intent flags in ckpt:", ia, flush=True)

    dev = "cuda"
    m = MODELS[mv](n_seg=21).to(dev).eval()
    m.load_state_dict(ck["model"], strict=False)

    _, val_s = split_scenes(a.root)
    ds = BevLaneDataset(a.root, val_s, gt_key="gt_cons", with_ego=True,
                        max_per_scene=4)
    dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False,
                                     num_workers=6)

    spreadA, sel = [], {c: [] for c, _ in CMDS}
    gt_lat, speeds = [], []
    with torch.no_grad():
        for batch in dl:
            if len(spreadA) >= a.frames:
                break
            imgs, K, Tc = [t.to(dev) for t in batch[:3]]
            ego_gt = batch[4].to(dev)
            if ego_gt[0, 16] < 0.5:
                continue
            lat = float(ego_gt[0, LAT])
            if abs(lat) < a.turn_m:            # turn frames only
                continue
            v0 = ego_gt[:, 12]
            per_cmd = {}
            for name, idx in CMDS:
                intent = None
                if idx is not None:
                    intent = torch.zeros(1, 3, device=dev)
                    intent[0, idx] = 1.0
                with torch.autocast("cuda", torch.float16):
                    out = m(imgs, K, Tc, v0,
                            **({"intent": intent} if intent is not None else {}))
                e = out[7].float()
                wp = e[0, :12 * EGO_K].view(EGO_K, 6, 2)
                lg = e[0, 12 * EGO_K:12 * EGO_K + EGO_K]
                lats = wp[:, 5, 1]                       # lat@3s per mode
                if name == "none":
                    spreadA.append(float(lats.max() - lats.min()))
                    probs = torch.softmax(lg, 0).cpu().numpy()
                per_cmd[name] = float(lats[int(lg.argmax())])
            for name in per_cmd:
                sel[name].append(per_cmd[name])
            gt_lat.append(lat)
            speeds.append(float(v0[0]) * 3.6)

    n = len(spreadA)
    if not n:
        print("no turn frames found", flush=True)
        return
    A = float(np.mean(spreadA))
    sel = {k: np.array(v) for k, v in sel.items()}
    stack = np.stack([sel[c] for c, _ in CMDS])
    B = float(np.mean(stack.max(0) - stack.min(0)))
    gt_lat = np.array(gt_lat)
    print(f"\n=== TURN FRAMES: {n} frames, mean speed {np.mean(speeds):.0f} km/h")
    print(f"  A) K={EGO_K} spread within one forward (lat@3s): mean {A:.2f}  "
          f"median {np.median(spreadA):.2f}  p90 {np.percentile(spreadA, 90):.2f}")
    print(f"     collapsed (<0.5 m): {np.mean(np.array(spreadA) < 0.5):.0%}")
    print("  B) selected path per command (none/left/right): "
          + " / ".join(f"{sel[c].mean():+.2f}" for c, _ in CMDS)
          + f"   spread {B:.2f} m")
    for tag, mask, want, cmd in (("GT left ", gt_lat > 0, -1, "right"),
                                 ("GT right", gt_lat < 0, +1, "left")):
        if not mask.any():
            continue
        vals = " / ".join(f"{sel[c][mask].mean():+.2f}" for c, _ in CMDS)
        rev = float(np.mean(np.sign(sel[cmd][mask]) == want))
        print(f"     {tag} ({int(mask.sum()):3d}): {vals}   "
              f"opposite command reverses the sign on {rev:.0%}")
    print(f"\n  PASS lines: B >= 5.0 m, reversal > 60%  ->  "
          f"B={B:.2f} m", flush=True)


if __name__ == "__main__":
    main()

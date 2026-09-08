#!/usr/bin/env python3
"""Per-module inference latency of the training model, on ONE GPU.

The TensorRT numbers in docs/perf_analysis_plan.md are from v42 (r37). v48
added three optional-input stems, the pseudo-LiDAR head and the dense unknown
head, and the deployment wrapper in deploy/export_onnx.py is still built around
v29's head set — so before touching the engine we need to know where v48's time
actually goes, measured on the very weights we ship.

Timing method: forward hooks with CUDA events around each registered module,
after a warmup, with the whole forward run under `torch.autocast(fp16)` and
`no_grad` exactly as inference does. Children are subtracted from parents so
the column sums to the wall time (a module's "self" time).

    CUDA_VISIBLE_DEVICES=7 python3 bevlane/profile_torch.py \
        --ckpt out/bevlane_ckpt_r48/last.pt --iters 30
"""
import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bevlane.dataset import BevLaneDataset                      # noqa: E402
from bevlane.model import MODELS                                # noqa: E402

# name prefix -> reported group (first match wins)
GROUPS = [
    ("backbone", ("stem", "layer1", "layer2", "layer3", "layer4",
                  "lat1", "lat2", "lat3", "lat4", "fuse")),
    ("depth head", ("depth_head",)),
    ("2D seg head", ("seg_head", "dec2d")),
    ("BEV projection", ("ipm", "proj", "bev_stem")),
    ("temporal fuse", ("tfuse", "tgate", "ctx")),
    ("BEV seg decoder", ("dec",)),
    ("3D det", ("det_stem", "hm_head", "reg_head", "stat_head")),
    ("2D det", ("det2d",)),
    ("E2E", ("ego_stem", "ego_mlp")),
    ("OCC", ("occ_stem", "occ_head")),
    ("traj / flow", ("traj_stem", "traj_head", "flow_head")),
    ("TL", ("tl_head", "tl_fc")),
    ("risk", ("risk_head",)),
    ("lane graph", ("lg_",)),
    ("unknown dense", ("unk_dense",)),
    ("pseudo-LiDAR", ("pl_head",)),
    ("optional stems", ("lidar_stem", "sdmap_stem", "tl_stem", "lid_alpha")),
]


def group_of(name):
    for g, keys in GROUPS:
        if any(k in name for k in keys):
            return g
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--root", default="out/bevlane")
    ap.add_argument("--val-list", default="val.lst")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--n-seg2d", type=int, default=21)
    ap.add_argument("--zero-cams", default="",
                    help="e.g. CAM_BACK_NARROW to time the 7-camera rig")
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu")
    mv = a.model or (ck.get("args") or {}).get("model") or "v48"
    m = MODELS[mv](n_seg=a.n_seg2d).cuda().eval()
    m.load_state_dict({k.replace("module.", ""): v
                       for k, v in ck["model"].items()}, strict=False)
    if a.zero_cams:
        from bevlane.dataset import CAMS
        m.zero_cams = tuple(CAMS.index(c) for c in a.zero_cams.split(","))
        print(f"[cams] zeroed {a.zero_cams} -> {m.zero_cams}")

    sc = [l.strip() for l in open(a.val_list) if l.strip()][:2]
    ds = BevLaneDataset(a.root, sc, gt_key="gt_cons", with_ego=True)
    b = ds[0]
    imgs, K, T = (b[0][None].cuda(), b[1][None].cuda(), b[2][None].cuda())
    v0 = b[4][12].view(1).cuda().float()

    # ---- hooks: CUDA events per module ----
    tot = defaultdict(float)
    child = defaultdict(float)
    parent = {}
    ev = {}
    named = [(n, mod) for n, mod in m.named_modules() if n]
    for n, mod in named:
        parent[n] = n.rsplit(".", 1)[0] if "." in n else ""

    def pre(n):
        def f(_mod, _in):
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            ev[n] = e
        return f

    def post(n):
        def f(_mod, _in, _out):
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            ev[n + "#end"] = e
        return f

    hs = []
    for n, mod in named:
        hs.append(mod.register_forward_pre_hook(pre(n)))
        hs.append(mod.register_forward_hook(post(n)))

    def run():
        with torch.no_grad(), torch.autocast("cuda", torch.float16):
            return m(imgs, K, T, v0)

    for _ in range(a.warmup):
        run()
    torch.cuda.synchronize()

    walls = []
    for _ in range(a.iters):
        ev.clear()
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        run()
        t1.record()
        torch.cuda.synchronize()
        walls.append(t0.elapsed_time(t1))
        for n, _mod in named:
            s, e = ev.get(n), ev.get(n + "#end")
            if s is None or e is None:
                continue
            dt = s.elapsed_time(e)
            tot[n] += dt
            if parent[n]:
                child[parent[n]] += dt
    for h in hs:
        h.remove()

    wall = float(np.mean(walls))
    it = a.iters
    self_ms = {n: (tot[n] - child[n]) / it for n in tot}
    per_group = defaultdict(float)
    for n, ms in self_ms.items():
        per_group[group_of(n)] += ms

    print(f"\n{mv} inference, batch 1, {imgs.shape[1]} cameras, fp16 autocast, "
          f"workstation GPU")
    print(f"wall {wall:.1f} ms  ({1000 / wall:.1f} FPS)   "
          f"peak mem {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    print(f"\n{'module group':22s} {'ms':>7s} {'% of wall':>10s}")
    acc = 0.0
    for g, ms in sorted(per_group.items(), key=lambda kv: -kv[1]):
        acc += ms
        print(f"{g:22s} {ms:7.2f} {100 * ms / wall:9.1f}%")
    print(f"{'-' * 40}\n{'sum of modules':22s} {acc:7.2f} "
          f"{100 * acc / wall:9.1f}%")
    print(f"{'unattributed':22s} {wall - acc:7.2f} "
          f"{100 * (wall - acc) / wall:9.1f}%   (python/launch overhead)")

    # ---- stage timing: the BEV projection is FUNCTIONAL (grid_sample), so it
    # never shows up as a module and lands in "unattributed". Time the pipeline
    # stages directly; this is what the TRT profile groups as bev-projection.
    def stage_times():
        st = {}

        def t(fn, n=15):
            for _ in range(4):
                fn()
            torch.cuda.synchronize()
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            for _ in range(n):
                fn()
            e1.record()
            torch.cuda.synchronize()
            return e0.elapsed_time(e1) / n

        with torch.no_grad(), torch.autocast("cuda", torch.float16):
            st["image_feats (backbone+depth+2Dseg)"] = t(
                lambda: m.image_feats(imgs))
            st["compute_bev (feats + depth-gated lift)"] = t(
                lambda: m.compute_bev(imgs, K, T))
            st["full forward (all 19 heads)"] = t(
                lambda: m(imgs, K, T, v0), n=10)
        return st

    print("\nstage timing (functional code included)")
    for k, v in stage_times().items():
        print(f"  {v:7.2f} ms  {100 * v / wall:5.1f}%  {k}")

    print(f"\ntop 12 individual modules")
    for n, ms in sorted(self_ms.items(), key=lambda kv: -kv[1])[:12]:
        print(f"  {ms:7.2f} ms  {100 * ms / wall:5.1f}%  {n}")


if __name__ == "__main__":
    main()

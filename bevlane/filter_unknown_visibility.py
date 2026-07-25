#!/usr/bin/env python3
"""Camera-visibility filter for the dense unknown GT (unknown_v2 -> v3).

The LiDAR-accumulated unknown_v2 mask marks obstacles confirmed over the
whole drive, so a frame's mask contains obstacles the cameras CANNOT see in
that frame (occluded by vehicles/structures). Training counts them as
positives and eval counts them as misses -> the head is punished for
physically impossible predictions (r36/r37: obj-recall stuck ~0).

Per frame, per GT-positive cell: march the BEV ray ego -> cell over the
per-frame LiDAR raster (lidar_bev ch1 = max z, ch3 = occupancy). The sight
line from camera height HC down to the obstacle top HO is blocked when an
intervening occupied cell's max z rises above the line -> relabel that cell
2 (= don't-care; dataset maps it to -1, the loss/eval ignore it).

Output: unknown_v3/<fi>.npz {"mask": uint8 [400,250], 0 free / 1 obstacle /
2 occluded-don't-care}; frame key "unknown_v3", manifest flag unknown_v3=1.
unknown_v2 files are left untouched (switch back anytime).

Usage: python3 bevlane/filter_unknown_visibility.py [--workers 16]
       [--scenes file]      # default: every scene flagged unknown_v2=1
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

B = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(B, "out/bevlane")
GH, GW, RES = 400, 250, 0.4
R0, C0 = 200, 125                     # ego cell (x=0, y=0)
HC = 1.75                             # camera height [m]
HO = 0.60                             # assumed obstacle top height [m]
HOOD_M = 3.5                          # ignore blockers this close to ego
TGT_M = 1.6                           # ignore blockers this close to target
K = 64                                # samples per ray


def filter_frame(mask, zmax, occ):
    """mask uint8 [400,250]; zmax/occ float [400,250] -> v3 mask uint8."""
    rr, cc = np.nonzero(mask > 0)
    if len(rr) == 0:
        return mask.copy()
    P = len(rr)
    dr = rr.astype(np.float32) - R0
    dc = cc.astype(np.float32) - C0
    dist = np.sqrt(dr ** 2 + dc ** 2) * RES            # [P] metres
    t = np.linspace(0.0, 1.0, K, dtype=np.float32)[None]  # [1,K]
    sr = (R0 + dr[:, None] * t).round().astype(np.int32).clip(0, GH - 1)
    sc = (C0 + dc[:, None] * t).round().astype(np.int32).clip(0, GW - 1)
    d = dist[:, None] * t                              # [P,K] metres
    zb = zmax[sr, sc]                                  # blocker max z
    ob = occ[sr, sc] > 0.5
    # sight line height at distance d toward a target at dist
    hline = HC + (HO - HC) * (d / np.maximum(dist[:, None], 1e-3))
    blocking = (ob & (zb >= hline + 0.05)
                & (d > HOOD_M) & (d < (dist[:, None] - TGT_M)))
    occluded = blocking.any(1)                         # [P]
    out = mask.copy()
    out[rr[occluded], cc[occluded]] = 2
    return out


def process_scene(scene):
    try:
        sd = os.path.join(OUT, scene)
        man = json.load(open(os.path.join(sd, "manifest.json")))
        if man.get("unknown_v3") == 1:
            return f"[skip] {scene}"
        if man.get("unknown_v2") != 1:
            return f"[nov2] {scene}"
        os.makedirs(os.path.join(sd, "unknown_v3"), exist_ok=True)
        npos = ndc = 0
        for fr in man["frames"]:
            p2 = fr.get("unknown_v2")
            plb = fr.get("lidar_bev")
            if not p2:
                continue
            m = np.load(os.path.join(sd, p2))["mask"]
            if plb and os.path.exists(os.path.join(sd, plb)):
                lb = np.load(os.path.join(sd, plb))["lb"]
                m3 = filter_frame(m, lb[1], lb[3])
            else:
                m3 = m.copy()                # no lidar raster: keep as-is
            fi = fr["frame"]
            np.savez_compressed(
                os.path.join(sd, f"unknown_v3/{fi:04d}.npz"), mask=m3)
            fr["unknown_v3"] = f"unknown_v3/{fi:04d}.npz"
            npos += int((m > 0).sum()); ndc += int((m3 == 2).sum())
        man["unknown_v3"] = 1
        json.dump(man, open(os.path.join(sd, "manifest.json"), "w"))
        pct = 100.0 * ndc / max(npos, 1)
        return f"[ok] {scene} pos={npos} dontcare={ndc} ({pct:.0f}%)"
    except Exception as e:
        return f"[fail] {scene}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--scenes", default=None)
    args = ap.parse_args()
    if args.scenes:
        scenes = open(args.scenes).read().split()
    else:
        scenes = sorted(
            d for d in os.listdir(OUT)
            if os.path.exists(os.path.join(OUT, d, "manifest.json")))
    print(f"{len(scenes)} scenes", flush=True)
    ok = fail = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(process_scene, scenes)):
            ok += r.startswith(("[ok]", "[skip]"))
            fail += r.startswith("[fail]")
            if i % 200 == 0 or r.startswith("[fail]"):
                print(f"{i + 1}/{len(scenes)} {r}", flush=True)
    print(f"DONE ok={ok} fail={fail}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Vector lane-graph GT from the autolabel vector maps (DESIGN_v29 §3).

vector_map.json stores connected polylines per class in map-frame metres.
Per frame: transform to ego, clip to the ROI (x -10..60, |y| <= 25 m),
resample each piece to P=12 points, keep the M=24 nearest chains over
{laneline, road_edge, stopline}; adjacency links consecutive pieces of one
source chain and nearby endpoints (< 1 m).

Saved per scene: lanegraph.npz {pts[F,M,P,2] f16, cls[F,M] u8 (255 empty),
n[F] u8, adj[F,M,M] u8}; manifest key "lanegraph".
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bevlane.extract_gt import OUT  # noqa: E402

PROD = "out/production"
M, P = 24, 12
XMIN, XMAX, YH = -10.0, 60.0, 25.0
CLS = {"laneline": 0, "road_edge": 1, "stopline": 2}


def resample(pts, n=P):
    d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(d)])
    if s[-1] < 1e-3:
        return None
    t = np.linspace(0, s[-1], n)
    return np.stack([np.interp(t, s, pts[:, 0]), np.interp(t, s, pts[:, 1])], 1)


def clip_chain(pts):
    """split an ego-frame polyline into maximal in-ROI pieces."""
    inside = (pts[:, 0] > XMIN) & (pts[:, 0] < XMAX) & (np.abs(pts[:, 1]) < YH)
    pieces, cur = [], []
    for p, ok in zip(pts, inside):
        if ok:
            cur.append(p)
        elif len(cur) >= 2:
            pieces.append(np.array(cur)); cur = []
        else:
            cur = []
    if len(cur) >= 2:
        pieces.append(np.array(cur))
    return pieces


def process_scene(args):
    scene, _ = args
    try:
        out_dir = os.path.join(OUT, scene)
        mf = os.path.join(out_dir, "manifest.json")
        man = json.load(open(mf))
        vm = json.load(open(os.path.join(PROD, scene, "vector_map.json")))
        chains = []                              # (cls, pts[N,2] map metres)
        for nm, ci in CLS.items():
            for pl in vm["classes"].get(nm, []):
                a = np.asarray(pl, np.float64)
                if len(a) >= 2:
                    chains.append((ci, a))
        ego = np.load(os.path.join(out_dir, "ego_motion.npz"))["pose"]
        F = 1 + max(fr["frame"] for fr in man["frames"])
        PT = np.zeros((F, M, P, 2), np.float16)
        CL = np.full((F, M), 255, np.uint8)
        N = np.zeros(F, np.uint8)
        AD = np.zeros((F, M, M), np.uint8)
        for fr in man["frames"]:
            fi = fr["frame"]
            if fi >= len(ego) or np.abs(ego[fi]).sum() == 0:
                continue
            px, py, yaw = ego[fi]
            c, s = np.cos(yaw), np.sin(yaw)
            slots = []                           # (dist, cls, src, pts[P,2])
            for src, (ci, a) in enumerate(chains):
                dx, dy = a[:, 0] - px, a[:, 1] - py
                e = np.stack([c * dx + s * dy, -s * dx + c * dy], 1)
                if e[:, 0].max() < XMIN or e[:, 0].min() > XMAX \
                        or np.abs(e[:, 1]).min() > YH:
                    continue
                for piece in clip_chain(e):
                    r = resample(piece)
                    if r is None or np.linalg.norm(r[-1] - r[0]) < 1.5:
                        continue
                    slots.append((float(np.abs(r).sum(1).min()), ci, src, r))
            slots.sort(key=lambda t: t[0])
            slots = slots[:M]
            for i, (_, ci, src, r) in enumerate(slots):
                PT[fi, i] = r
                CL[fi, i] = ci
            N[fi] = len(slots)
            for i in range(len(slots)):
                for j in range(i + 1, len(slots)):
                    same = slots[i][2] == slots[j][2]
                    ei = np.array([slots[i][3][0], slots[i][3][-1]])
                    ej = np.array([slots[j][3][0], slots[j][3][-1]])
                    dmin = min(np.linalg.norm(a - b)
                               for a in ei for b in ej)
                    if same or dmin < 1.0:
                        AD[fi, i, j] = AD[fi, j, i] = 1
        np.savez_compressed(os.path.join(out_dir, "lanegraph.npz"),
                            pts=PT, cls=CL, n=N, adj=AD)
        man["lanegraph"] = "lanegraph.npz"
        json.dump(man, open(mf, "w"))
        return f"[ok] {scene} chains={int(N.sum())}"
    except Exception as e:
        return f"[fail] {scene}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--scenes", default=None)
    ap.add_argument("--stride", type=int, default=2)   # unused, stage-uniform
    args = ap.parse_args()
    if args.scenes:
        scenes = (open(args.scenes).read().split() if os.path.isfile(args.scenes)
                  else args.scenes.split(","))
    else:
        scenes = sorted(d for d in os.listdir(OUT)
                        if os.path.exists(os.path.join(OUT, d, "manifest.json")))
    print(f"{len(scenes)} scenes; lane-graph GT", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(process_scene,
                                     [(s, 0) for s in scenes])):
            if i % 200 == 0 or not r.startswith("[ok"):
                print(f"{i + 1}/{len(scenes)} {r}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

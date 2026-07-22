#!/usr/bin/env python3
"""Unknown-object GT v2 (roadmap: fundamental unknown-detector redesign).

Replaces the fragile single-frame occ-blob GT (extract_unknown.py) with a
LiDAR-accumulated, known-box-residual DENSE occupancy mask of small static
obstacles (cones, posts, bollards, barriers, debris -- the "unknown" class the
annotation set does not contain).

Per scene:
  1. per-frame small-obstacle candidates from the LiDAR BEV raster
       cand = occ AND (Z_LOW < max_z < Z_HIGH)           # short things above
              AND NOT known-box footprint (bev_box png)   # not veh/VRU
              AND range < R_MAX AND outside the ego hood
  2. DRIVE ACCUMULATION: splat each frame's candidates into a global 0.4 m grid
     via the ego pose; a global cell is a confirmed obstacle when it is hit in
     >= MIN_HITS frames AND hits/seen >= RATIO (transient noise / moving objects
     wash out, real static obstacles reinforce -- the gt_cons trick).
  3. project the confirmed global obstacles back into every frame's ego grid ->
     a dense [400,250] uint8 mask on the detection grid (0.4 m).

Saved as unknown_v2/<fi>.npz {"mask": uint8[400,250]}; manifest key
"unknown_v2" + flag unknown_v2=1. Existing GT untouched.
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bevlane.extract_gt import OUT  # noqa: E402

GH, GW, RES = 400, 250, 0.4          # det grid: rows=(80-x)/.4, cols=(50-y)/.4
X0, Y0 = 80.0, 50.0
Z_LOW, Z_HIGH = 0.20, 1.6            # small-obstacle height band (m)
R_MAX = 45.0                         # LiDAR-reliable range (m)
HOOD_R = 3.0                         # exclude ego-proximal cells (m)
BOX_DILATE = 2                       # cells; grow known-box footprint
MIN_HITS = 3                         # frames a global cell must be hit
RATIO = 0.30                         # hits / seen to confirm
GRES = 0.4                           # global accumulation grid (m)
MAX_OBST_AREA = 30                   # cells (~4.8 m2): drop walls/buildings


def _cell_xy():
    """ego metric (x fwd, y left) at each det-grid cell centre."""
    xs = X0 - (np.arange(GH) + 0.5) * RES
    ys = Y0 - (np.arange(GW) + 0.5) * RES
    return np.meshgrid(xs, ys, indexing="ij")   # [GH,GW] each


CX, CY = _cell_xy()
CR = np.sqrt(CX ** 2 + CY ** 2)                  # range at each cell
HOOD = CR < HOOD_R
INRANGE = CR < R_MAX


def per_frame_candidates(out_dir, fr):
    """small-obstacle candidate mask [GH,GW] bool for one frame, or None."""
    lbp = os.path.join(out_dir, fr.get("lidar_bev", "_"))
    if not os.path.exists(lbp):
        return None
    lb = np.load(lbp)["lb"].astype(np.float32)    # [4,GH,GW]
    occ = lb[3] > 0.5
    maxz = lb[1]
    cand = occ & (maxz > Z_LOW) & (maxz < Z_HIGH) & INRANGE & (~HOOD)
    # subtract known-box footprints (vehicles / VRUs), rasterised 800x500
    bp = os.path.join(out_dir, fr.get("bev_box", "_"))
    box = cv2.imread(bp, cv2.IMREAD_UNCHANGED) if os.path.exists(bp) else None
    if box is not None and box.shape == (2 * GH, 2 * GW):
        # 800x500 -> 400x250 by 2x2 max (keep thin box footprints)
        box = (box > 0).astype(np.uint8).reshape(GH, 2, GW, 2).max((1, 3))
        if BOX_DILATE:
            box = cv2.dilate(box, np.ones((2 * BOX_DILATE + 1,) * 2, np.uint8))
        cand &= (box == 0)
    return cand


def process_scene(scene):
    try:
        out_dir = os.path.join(OUT, scene)
        man = json.load(open(os.path.join(out_dir, "manifest.json")))
        if man.get("unknown_v2") == 1:
            return f"[skip] {scene}"
        try:
            pose = np.load(os.path.join(out_dir, "ego_motion.npz"))["pose"]
        except Exception:
            return f"[nolidarpose] {scene}"
        frames = [f for f in man["frames"] if "lidar_bev" in f]
        if not frames:
            return f"[nolidar] {scene}"

        # ---- pass 1: candidates + accumulate into a global grid
        cand_by_fi, glob_pts = {}, []
        for fr in frames:
            fi = fr["frame"]
            if fi >= len(pose) or np.abs(pose[fi]).sum() == 0:
                continue
            cand = per_frame_candidates(out_dir, fr)
            if cand is None:
                continue
            cand_by_fi[fi] = cand
            px, py, pa = pose[fi]
            ca, sa = np.cos(pa), np.sin(pa)
            r, c = np.nonzero(cand)
            xe, ye = CX[r, c], CY[r, c]
            gx = px + xe * ca - ye * sa
            gy = py + xe * sa + ye * ca
            glob_pts.append(np.stack([gx, gy], 1))
        if not cand_by_fi:
            return f"[empty] {scene}"

        # ---- global hit / seen histograms over the drive extent
        allg = np.concatenate(glob_pts, 0) if glob_pts else np.zeros((0, 2))
        # seen extent = every frame's in-range footprint centre positions
        gminx = min(pose[fi][0] for fi in cand_by_fi) - X0
        gminy = min(pose[fi][1] for fi in cand_by_fi) - X0
        gmaxx = max(pose[fi][0] for fi in cand_by_fi) + X0
        gmaxy = max(pose[fi][1] for fi in cand_by_fi) + X0
        HH = int((gmaxx - gminx) / GRES) + 2
        WW = int((gmaxy - gminy) / GRES) + 2
        HH, WW = min(HH, 6000), min(WW, 6000)
        hits = np.zeros((HH, WW), np.int32)
        seen = np.zeros((HH, WW), np.int32)

        def to_g(gx, gy):
            gr = ((gx - gminx) / GRES).astype(np.int32)
            gc = ((gy - gminy) / GRES).astype(np.int32)
            ok = (gr >= 0) & (gr < HH) & (gc >= 0) & (gc < WW)
            return gr[ok], gc[ok]

        if len(allg):
            gr, gc = to_g(allg[:, 0], allg[:, 1])
            np.add.at(hits, (gr, gc), 1)
        # seen: splat each frame's near-range (<R_MAX) footprint
        for fi in cand_by_fi:
            px, py, pa = pose[fi]
            ca, sa = np.cos(pa), np.sin(pa)
            m = INRANGE & (~HOOD)
            xe, ye = CX[m], CY[m]
            gx = px + xe * ca - ye * sa
            gy = py + xe * sa + ye * ca
            gr, gc = to_g(gx, gy)
            np.add.at(seen, (gr, gc), 1)

        conf = (hits >= MIN_HITS) & (hits >= RATIO * np.maximum(seen, 1))
        # keep only DISCRETE small obstacles: drop connected components larger
        # than a small barrier (walls / buildings / long guardrails / hedges
        # are continuous and huge; cones/posts/bollards are a few cells).
        conf = conf.astype(np.uint8)
        ncomp, lbls, stats, _ = cv2.connectedComponentsWithStats(conf, 8)
        keep = np.zeros(ncomp, bool)
        for ci in range(1, ncomp):
            keep[ci] = stats[ci, cv2.CC_STAT_AREA] <= MAX_OBST_AREA
        conf = keep[lbls]

        # ---- pass 2: project confirmed obstacles back into each ego frame
        n = 0
        os.makedirs(os.path.join(out_dir, "unknown_v2"), exist_ok=True)
        for fr in frames:
            fi = fr["frame"]
            if fi not in cand_by_fi:
                continue
            px, py, pa = pose[fi]
            ca, sa = np.cos(pa), np.sin(pa)
            # ego cell -> global -> lookup conf
            gx = px + CX * ca - CY * sa
            gy = py + CX * sa + CY * ca
            gr = ((gx - gminx) / GRES).astype(np.int32)
            gc = ((gy - gminy) / GRES).astype(np.int32)
            ok = (gr >= 0) & (gr < HH) & (gc >= 0) & (gc < WW) & INRANGE
            mask = np.zeros((GH, GW), np.uint8)
            v = np.zeros((GH, GW), bool)
            v[ok] = conf[gr[ok], gc[ok]]
            mask[v] = 1
            np.savez_compressed(
                os.path.join(out_dir, f"unknown_v2/{fi:04d}.npz"), mask=mask)
            fr["unknown_v2"] = f"unknown_v2/{fi:04d}.npz"
            n += 1
        man["unknown_v2"] = 1
        json.dump(man, open(os.path.join(out_dir, "manifest.json"), "w"))
        npos = int(conf.sum())
        return f"[ok] {scene} frames={n} globobst={npos}"
    except Exception as e:
        import traceback
        return f"[fail] {scene}: {e} {traceback.format_exc()[-200:]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--scenes", default=None)
    args = ap.parse_args()
    if args.scenes:
        scenes = (open(args.scenes).read().split()
                  if os.path.isfile(args.scenes) else args.scenes.split(","))
    else:
        scenes = sorted(d for d in os.listdir(OUT)
                        if os.path.exists(os.path.join(OUT, d, "manifest.json")))
    print(f"{len(scenes)} scenes; unknown GT v2", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(process_scene, scenes)):
            if i % 100 == 0 or r.startswith(("[fail", "[ok")):
                print(f"{i + 1}/{len(scenes)} {r}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Filter dynamic-object 'shadows' out of the occupancy GT.

Dynamic classes are accumulated from ±1 frames, so a moving car at 10 m/s
smears up to ~4 m of vehicle voxels along its motion; measured on the train
corpus, 18% of near-ego vehicle GT voxels lie OUTSIDE every camera-confirmed
3D box — phantom supervision that teaches near-range vehicle false
positives (user-visible, especially beside the ego).

Filter: for dynamic classes {2 vehicle, 3 two-wheeler, 4 pedestrian}, any
voxel whose ground cell falls outside every (0.6 m-dilated) GT box footprint
becomes 255 (ignore) — NOT free, because far outside-box voxels are often
real-but-unconfirmed vehicles and asserting free would teach false
negatives. Idempotent per scene via manifest["occ_shadow_filtered"].
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

DYN = (2, 3, 4)


def footprint_mask(boxes, count):
    m = np.zeros((200, 200), np.uint8)
    for k in range(int(count)):
        cls, xe, ye, l, w, yaw = boxes[k]
        if l <= 0 or abs(xe) > 44 or abs(ye) > 44:
            continue
        c_, s_ = np.cos(yaw), np.sin(yaw)
        pts = [[int((40 - (ye + lx * s_ + wy * c_)) / 0.4),
                int((40 - (xe + lx * c_ - wy * s_)) / 0.4)]
               for lx, wy in ((l / 2, w / 2), (l / 2, -w / 2),
                              (-l / 2, -w / 2), (-l / 2, w / 2))]
        cv2.fillPoly(m, [np.array(pts, np.int32).reshape(-1, 1, 2)], 1)
    return cv2.dilate(m, np.ones((5, 5), np.uint8))   # ~0.6 m margin


def process_scene(scene):
    try:
        out_dir = os.path.join(OUT, scene)
        mf = os.path.join(out_dir, "manifest.json")
        man = json.load(open(mf))
        if man.get("occ_shadow_filtered"):
            return f"[skip] {scene}: already filtered"
        n_fix = 0
        for fr in man["frames"]:
            if not fr.get("occ") or not fr.get("agent_traj"):
                continue
            op = os.path.join(out_dir, fr["occ"])
            try:
                occ = np.load(op)["occ"]
                z = np.load(os.path.join(out_dir, fr["agent_traj"]))
            except Exception:
                continue
            m = footprint_mask(z["boxes"], z["count"])
            dyn = np.isin(occ, DYN)
            shadow = dyn & (m[None] == 0)
            if shadow.any():
                occ = occ.copy()
                occ[shadow] = 255
                np.savez_compressed(op.replace(".npz", ""), occ=occ)
                n_fix += int(shadow.sum())
        man["occ_shadow_filtered"] = 1
        json.dump(man, open(mf, "w"))
        return f"[ok] {scene} ignored={n_fix}"
    except Exception as e:
        return f"[fail] {scene}: {e}"


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
    print(f"{len(scenes)} scenes; occ shadow filter", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(process_scene, scenes)):
            if i % 200 == 0 or r.startswith("[fail"):
                print(f"{i + 1}/{len(scenes)} {r}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

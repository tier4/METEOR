#!/usr/bin/env python3
"""Unknown-object (cone / pole / debris) centre GT for BEV 3D detection.

The annotation boxes carry no 'unknown' class, but the occupancy GT already
localises them: class 1 (obstacle/unknown, from the seg2d21 LUT: cones,
guide posts, generic obstacles) in 3D voxels. Small, low blobs of that
class are exactly the objects we want a fixed-size detection for.

Per frame: ground-project class-1 voxels, connected components, keep blobs
with area <= 2.4 m^2 (15 cells @0.4 m) and height <= 1.6 m (4 layers),
centre = blob centroid. Saved per scene as unknown_obj.npz
{centers [F,32,2] float16 (x,y ego metres), n [F] uint8}.
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

KMAX = 32
MAX_AREA = 15          # cells (2.4 m^2)
MAX_LAYERS = 4         # 1.6 m


def process_scene(scene):
    try:
        out_dir = os.path.join(OUT, scene)
        mf = os.path.join(out_dir, "manifest.json")
        man = json.load(open(mf))
        F = 1 + max(fr["frame"] for fr in man["frames"])
        C = np.zeros((F, KMAX, 2), np.float16)
        N = np.zeros(F, np.uint8)
        tot = 0
        for fr in man["frames"]:
            if not fr.get("occ"):
                continue
            try:
                occ = np.load(os.path.join(out_dir, fr["occ"]))["occ"]
            except Exception:
                continue
            obs = (occ == 1)
            ground = obs.any(0).astype(np.uint8)
            nlab, lab, stats, cent = cv2.connectedComponentsWithStats(ground)
            cands = []
            for j in range(1, nlab):
                if stats[j, cv2.CC_STAT_AREA] > MAX_AREA:
                    continue
                m = lab == j
                if int(obs[:, m].any(1).sum()) > MAX_LAYERS:
                    continue
                cy, cx = cent[j][1], cent[j][0]        # row, col
                xe = 40.0 - cy * 0.4
                ye = 40.0 - cx * 0.4
                cands.append((xe * xe + ye * ye, xe, ye))
            cands.sort()
            fi = fr["frame"]
            for k, (_, xe, ye) in enumerate(cands[:KMAX]):
                C[fi, k] = (xe, ye)
            N[fi] = min(len(cands), KMAX)
            tot += int(N[fi])
        np.savez_compressed(os.path.join(out_dir, "unknown_obj.npz"),
                            centers=C, n=N)
        man["unknown_obj"] = "unknown_obj.npz"
        json.dump(man, open(mf, "w"))
        return f"[ok] {scene} unk={tot}"
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
    print(f"{len(scenes)} scenes; unknown-object centres", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(process_scene, scenes)):
            if i % 300 == 0 or r.startswith("[fail"):
                print(f"{i + 1}/{len(scenes)} {r}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

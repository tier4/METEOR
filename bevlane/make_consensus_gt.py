#!/usr/bin/env python3
"""Consensus BEV-lane GT (roadmap 3a/3e): agreement of two GT generations.

The two independently generated lane rasters ('gt' accumulation render and
'gt_vec' vector re-render) agree at only mIoU 0.452 on val (laneline 0.19)
— i.e. much of the thin-class "error" the model is punished for is label
noise. This pass writes a THIRD, versioned GT ('gt_cons'):

    both agree  -> that class        disagree -> 255 (ignore)

Existing GT files are untouched; training opts in via --gt-key gt_cons.
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


def process_scene(scene):
    try:
        out_dir = os.path.join(OUT, scene)
        mf = os.path.join(out_dir, "manifest.json")
        man = json.load(open(mf))
        if man.get("gt_cons_v") == 2:
            return f"[skip] {scene}: already v2"
        os.makedirs(os.path.join(out_dir, "gt_cons"), exist_ok=True)
        n = 0
        for fr in man["frames"]:
            if "gt" not in fr or "gt_vec" not in fr:
                continue
            dst = f"gt_cons/{fr['frame']:04d}.png"

            a = cv2.imread(os.path.join(out_dir, fr["gt"]), 0)
            b = cv2.imread(os.path.join(out_dir, fr["gt_vec"]), 0)
            if a is None or b is None or a.shape != b.shape:
                continue
            cons = np.where(a == b, b, 255).astype(np.uint8)
            # road_edge (6): thin line, half-cell offsets make the two
            # generations disagree almost everywhere (measured IoU 0.14)
            # -> inherit the vector render's edge instead of erasing it
            cons[b == 6] = 6
            cv2.imwrite(os.path.join(out_dir, dst), cons)
            fr["gt_cons"] = dst
            n += 1
        man["gt_cons_v"] = 2
        json.dump(man, open(mf, "w"))
        return f"[ok] {scene} n={n}"
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
                        if os.path.exists(os.path.join(OUT, d,
                                                       "manifest.json")))
    print(f"{len(scenes)} scenes; consensus GT", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(process_scene, scenes)):
            if i % 300 == 0 or r.startswith("[fail"):
                print(f"{i + 1}/{len(scenes)} {r}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

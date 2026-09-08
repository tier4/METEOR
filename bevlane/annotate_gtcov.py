#!/usr/bin/env python3
"""Annotate per-frame GT coverage into manifests: f["gtcov"] = [core, fwd].

core = labeled fraction of the +-30 m band (rows 250:550);
fwd  = labeled fraction of the far-forward band +30..+80 m (rows 0:250).
Used to exclude frames with weak GT (long-stationary spots, sparse ends)
from training.
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import cv2

cv2.setNumThreads(1)
OUT = "out/bevlane"


def process_scene(scene):
    try:
        mp = os.path.join(OUT, scene, "manifest.json")
        man = json.load(open(mp))
        changed = False
        for f in man["frames"]:
            if "gtcov" in f or "gt_vec" not in f:
                continue
            g = cv2.imread(os.path.join(OUT, scene, f["gt_vec"]), 0)
            if g is None:
                continue
            core = float((g[250:550] > 0).mean())
            fwd = float((g[:250] > 0).mean())
            f["gtcov"] = [round(core, 4), round(fwd, 4)]
            changed = True
        if changed:
            json.dump(man, open(mp, "w"))
        return f"[ok] {scene}"
    except Exception as e:
        return f"[fail] {scene}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default=None)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    if args.scenes:
        scenes = (open(args.scenes).read().split() if os.path.isfile(args.scenes)
                  else args.scenes.split(","))
    else:
        scenes = sorted(d for d in os.listdir(OUT)
                        if os.path.exists(os.path.join(OUT, d, "manifest.json")))
    print(f"{len(scenes)} scenes", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(process_scene, scenes)):
            if i % 100 == 0 or r.startswith("[fail"):
                print(f"{i + 1}/{len(scenes)} {r}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

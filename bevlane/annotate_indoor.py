#!/usr/bin/env python3
"""Flag indoor / GNSS-dead scenes (underground garages, covered parking).

Ego pose in these scenes is a smooth fiction (GNSS dead, drift): kinematic
sanity checks pass while the path is wrong, so E2E / temporal / trajectory
GT is silently corrupted. Detector: mean SKY fraction of the FRONT_WIDE
2D-seg autolabel over the scene — indoor scenes measure ~0.000, outdoor
0.35+ even at night. Threshold 0.02.

Writes manifest["indoor"] = 0/1 and appends flagged scenes to
out/indoor_scenes.txt. Round scene lists exclude flagged scenes.
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

SKY_CLS = 19          # seg2d21 taxonomy
THRESH = 0.02


def process_scene(scene):
    try:
        out_dir = os.path.join(OUT, scene)
        mf = os.path.join(out_dir, "manifest.json")
        man = json.load(open(mf))
        fracs = []
        for fr in man["frames"][::5][:24]:
            if not fr.get("seg2d21"):
                continue
            try:
                sg = np.load(os.path.join(out_dir, fr["seg2d21"]))["seg"]
                fracs.append(float((sg[0] == SKY_CLS).mean()))   # FRONT_WIDE
            except Exception:
                pass
        if not fracs:
            return f"[skip] {scene}: no seg2d21"
        sky = float(np.mean(fracs))
        man["indoor"] = int(sky < THRESH)
        json.dump(man, open(mf, "w"))
        return f"[{'INDOOR' if man['indoor'] else 'ok'}] {scene} sky={sky:.4f}"
    except Exception as e:
        return f"[fail] {scene}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--scenes", default=None)
    args = ap.parse_args()
    if args.scenes:
        scenes = (open(args.scenes).read().split() if os.path.isfile(args.scenes)
                  else args.scenes.split(","))
    else:
        scenes = sorted(d for d in os.listdir(OUT)
                        if os.path.exists(os.path.join(OUT, d, "manifest.json")))
    print(f"{len(scenes)} scenes; indoor/GNSS-dead detector", flush=True)
    flagged = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(process_scene, scenes)):
            if r.startswith("[INDOOR]"):
                flagged.append(r.split()[1])
            if i % 300 == 0 or r.startswith(("[INDOOR", "[fail")):
                print(f"{i + 1}/{len(scenes)} {r}", flush=True)
    # merge with the existing list (single-scene convert_dtset invocations
    # must not clobber the global registry)
    prev = set()
    if os.path.exists("out/indoor_scenes.txt"):
        prev = set(open("out/indoor_scenes.txt").read().split())
    prev.update(flagged)
    with open("out/indoor_scenes.txt", "w") as f:
        f.write("\n".join(sorted(prev)))
    print(f"DONE flagged={len(flagged)} -> out/indoor_scenes.txt", flush=True)


if __name__ == "__main__":
    main()

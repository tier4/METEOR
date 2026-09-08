#!/usr/bin/env python3
"""One-off scan: which scenes have NO 3D-box annotation at all?

x2gen2 ships bev_box_p / agent_traj files that are empty in every frame (2D
boxes are there, only the BEV/3D conversion never ran). An empty file is
indistinguishable from "no objects in this frame", so those samples trained the
detector to stay silent on that whole rig — measured heatmap score 0.919 on the
Japanese rig vs 0.080 on x2gen2, zero boxes over the demo threshold.

Doing this check inside BevLaneDataset made every process open ~10 npz files
per scene; across 8,736 scenes and 8 ranks that turned a 3-minute dataset scan
into hours (r49 hung twice). So it is precomputed here, once, into
`out/nobox_scenes.txt`, which the dataset reads like indoor_scenes.txt.

    python3 bevlane/scan_nobox.py                 # all scenes under out/bevlane
    python3 bevlane/scan_nobox.py --scenes-file out/round49_scenes.txt
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

PROBE = 5          # frames per scene, spread out


def scene_has_boxes(args):
    root, s = args
    mf = os.path.join(root, s, "manifest.json")
    try:
        m = json.load(open(mf))
    except Exception:
        return s, None                       # unreadable -> say nothing
    fr = m.get("frames", [])
    if not fr:
        return s, None
    probe = fr[::max(1, len(fr) // PROBE)][:PROBE]
    seen = 0
    checked = 0
    for f in probe:
        for k in ("bev_box_p", "agent_traj"):
            q = f.get(k)
            if not q:
                continue
            p = os.path.join(root, s, q)
            if not os.path.exists(p):
                continue
            checked += 1
            try:
                z = np.load(p)
                seen += (len(z["boxes"]) if k == "bev_box_p"
                         else int(z["count"]))
            except Exception:
                pass
    if not checked:
        return s, None                       # nothing to judge
    return s, seen > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="out/bevlane")
    ap.add_argument("--scenes-file", default=None)
    ap.add_argument("--out", default="out/nobox_scenes.txt")
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()

    if a.scenes_file:
        scenes = [l.strip() for l in open(a.scenes_file) if l.strip()]
    else:
        scenes = sorted(os.listdir(a.root))
    print(f"scanning {len(scenes)} scenes with {a.workers} workers")
    nobox, unknown, ok = [], [], 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, (s, has) in enumerate(
                ex.map(scene_has_boxes, ((a.root, s) for s in scenes),
                       chunksize=16)):
            if has is None:
                unknown.append(s)
            elif has:
                ok += 1
            else:
                nobox.append(s)
            if (i + 1) % 2000 == 0:
                print(f"  {i + 1}/{len(scenes)} …", flush=True)
    open(a.out, "w").write("\n".join(sorted(nobox)) + "\n")
    print(f"annotated {ok} | NO 3D boxes {len(nobox)} | undecidable "
          f"{len(unknown)} -> {a.out}")
    x2 = set()
    for f in ("out/x2gen2_train.txt", "out/x2gen2_test.txt"):
        if os.path.exists(f):
            x2 |= {l.strip() for l in open(f) if l.strip()}
    if x2:
        print(f"  of the no-box scenes, {len(set(nobox) & x2)} are x2gen2 "
              f"and {len(set(nobox) - x2)} are not")


if __name__ == "__main__":
    main()

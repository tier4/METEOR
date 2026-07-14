#!/usr/bin/env python3
"""Ego-relevant traffic-light state GT from CoMET TLR autolabels.

CoMET's TLR subnet writes green/yellow/red bboxes (+prob) into the per-image
fastlabel JSONs. This stage reduces them to ONE whole-image label per frame —
the state of the traffic light the EGO vehicle should obey:

  ego-relevance heuristic:
    1. CAM_FRONT_NARROW first — the telephoto looks straight down the travel
       corridor, so any TL it sees is (almost always) ours; take the largest.
    2. else CAM_FRONT_WIDE, but only boxes whose centre lies in the middle
       50 % of the image (side TLs are for crossing roads); take the largest.
    3. else: none.

  label per frame: 0=none, 1=green, 2=yellow, 3=red  (+ prob of the pick)
  A 3-frame temporal median removes single-frame TLR flickers.

Saved once per scene as tl_state.npz {label[F], conf[F]}; manifest gains a
scene-level "tl_state" key.
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
from bevlane.extract_gt import OUT, ROOT  # noqa: E402

COLORS = {"green": 1, "yellow": 2, "red": 3}


def frame_label(raw_dir, stem_by_cam):
    """-> (label, conf) for one frame from its front-camera fastlabel JSONs."""
    best = (0, 0.0, 0.0)                       # (label, area_rank, prob)
    for cam, central in (("CAM_FRONT_NARROW", False), ("CAM_FRONT_WIDE", True)):
        stem = stem_by_cam.get(cam)
        if stem is None:
            continue
        fj = os.path.join(raw_dir, "fastlabel", stem + ".json")
        if not os.path.exists(fj):
            continue
        try:
            anns = json.load(open(fj))[0].get("annotations", [])
        except Exception:
            continue
        W = 2880.0                              # only used for centrality
        for a in anns:
            if a.get("title") not in COLORS or a.get("type") != "bbox":
                continue
            p = a.get("points", [])
            if len(p) != 4:
                continue
            x1, y1, x2, y2 = [float(v) for v in p]
            if central:
                cx = (x1 + x2) / 2
                if not (0.25 * W < cx < 0.75 * W):
                    continue
            area = max(x2 - x1, 0) * max(y2 - y1, 0)
            prob = float(a.get("prob", 0.5))
            if area > best[1]:
                best = (COLORS[a["title"]], area, prob)
        if best[0]:                             # NARROW hit wins outright
            break
    return best[0], best[2]


def process_scene(args):
    scene, stride = args
    try:
        out_dir = os.path.join(OUT, scene)
        mf = os.path.join(out_dir, "manifest.json")
        man = json.load(open(mf))
        raw_dir = os.path.join(ROOT, scene)
        if not os.path.isdir(os.path.join(raw_dir, "fastlabel")):
            return f"[skip] {scene}: no fastlabel dir"
        F = 1 + max(fr["frame"] for fr in man["frames"])
        lab = np.zeros(F, np.uint8)
        cf = np.zeros(F, np.float32)
        for fr in man["frames"]:
            fi = fr["frame"]
            # raw per-camera sample index = manifest frame id * stride
            stems = {cam: f"{scene}_{cam}_{fi * stride:05d}.jpg"
                     for cam in ("CAM_FRONT_NARROW", "CAM_FRONT_WIDE")}
            lab[fi], cf[fi] = frame_label(raw_dir, stems)
        # 3-frame median over the sampled frames (kills 1-frame flickers)
        fis = sorted(fr["frame"] for fr in man["frames"])
        sm = lab.copy()
        for j in range(1, len(fis) - 1):
            a, b, c = lab[fis[j - 1]], lab[fis[j]], lab[fis[j + 1]]
            if a == c and b != a:
                sm[fis[j]] = a
        np.savez_compressed(os.path.join(out_dir, "tl_state.npz"),
                            label=sm, conf=cf)
        man["tl_state"] = "tl_state.npz"
        json.dump(man, open(mf, "w"))
        n = int((sm[fis] > 0).sum())
        return f"[ok] {scene} tl-frames={n}/{len(fis)}"
    except Exception as e:
        return f"[fail] {scene}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--scenes", default=None)
    args = ap.parse_args()
    if args.scenes:
        scenes = (open(args.scenes).read().split() if os.path.isfile(args.scenes)
                  else args.scenes.split(","))
    else:
        scenes = sorted(d for d in os.listdir(OUT)
                        if os.path.exists(os.path.join(OUT, d, "manifest.json")))
    print(f"{len(scenes)} scenes; ego-relevant TL state", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(process_scene,
                                     [(s, args.stride) for s in scenes])):
            if i % 200 == 0 or not r.startswith("[ok"):
                print(f"{i + 1}/{len(scenes)} {r}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Per-camera 10-class 2D bounding-box GT from object_ann.

Class taxonomy comes from fastlabel_2510_instance.csv (id 0..9; colour =
first row per id). Boxes are stored in cached-image coordinates (768x432)
as bbox2d/<fi>.npz: boxes float32 [8, KMAX, 5] = (cls, cx, cy, w, h) padded,
counts uint8 [8]; manifest frames gain a "bbox2d" key. Cheap (JSON only,
no RLE decode).
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bevlane.extract_gt import OUT, ROOT, IMG_H, IMG_W, load_scene_light  # noqa: E402
from bevlane.extract_seg2d import CAMS  # noqa: E402

N_DET10 = 10
KMAX = 96
_CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "fastlabel_2510_instance.csv")


def _load_taxonomy():
    cat2det, pal = {}, np.zeros((N_DET10, 3), np.uint8)
    for ln in open(_CSV).read().splitlines()[1:]:
        i, nm, r, g, b = ln.split(",")
        cat2det[nm] = int(i)
        if not pal[int(i)].any():
            pal[int(i)] = (int(r), int(g), int(b))
    return cat2det, pal


CAT2DET, DET10_PAL = _load_taxonomy()             # DET10_PAL is RGB


def load_anns(scene_dir):
    ann = os.path.join(scene_dir, "annotation")
    cats = {c["token"]: c["name"] for c in
            json.load(open(os.path.join(ann, "category.json")))}
    by_sd = defaultdict(list)
    for a in json.load(open(os.path.join(ann, "object_ann.json"))):
        bb = a.get("bbox")
        nm = cats.get(a["category_token"], "")
        if bb and nm in CAT2DET:
            by_sd[a["sample_data_token"]].append((CAT2DET[nm], bb))
    return by_sd


def process_scene(args):
    scene, stride = args
    try:
        out_dir = os.path.join(OUT, scene)
        man = json.load(open(os.path.join(out_dir, "manifest.json")))
        sdir = os.path.join(ROOT, scene)
        ordered, frames, _, _ = load_scene_light(sdir)
        strided = ordered[::stride]
        anns = load_anns(sdir)
        os.makedirs(os.path.join(out_dir, "bbox2d"), exist_ok=True)
        for fr in man["frames"]:
            fi = fr["frame"]
            path = os.path.join(out_dir, f"bbox2d/{fi:04d}.npz")
            if os.path.exists(path):
                fr["bbox2d"] = f"bbox2d/{fi:04d}.npz"
                continue
            frame = frames[strided[fi]["token"]]
            boxes = np.zeros((len(CAMS), KMAX, 5), np.float32)
            counts = np.zeros(len(CAMS), np.uint8)
            for ci, ch in enumerate(CAMS):
                sd = frame.get(ch)
                if sd is None:
                    continue
                lst = anns.get(sd["token"], [])
                if not lst:
                    continue
                w0 = sd.get("width") or 2880
                h0 = sd.get("height") or 1860
                sx, sy = IMG_W / w0, IMG_H / h0
                cand = []
                for cls, (x1, y1, x2, y2) in lst:
                    cx, cy = (x1 + x2) / 2 * sx, (y1 + y2) / 2 * sy
                    w, h = (x2 - x1) * sx, (y2 - y1) * sy
                    if w < 2 or h < 2:            # sub-2px boxes: unlearnable
                        continue
                    cand.append((w * h, cls, cx, cy, w, h))
                cand.sort(reverse=True)           # biggest first when > KMAX
                for k, (_, cls, cx, cy, w, h) in enumerate(cand[:KMAX]):
                    boxes[ci, k] = (cls, cx, cy, w, h)
                counts[ci] = min(len(cand), KMAX)
            np.savez_compressed(path, boxes=boxes, counts=counts)
            fr["bbox2d"] = f"bbox2d/{fi:04d}.npz"
        json.dump(man, open(os.path.join(out_dir, "manifest.json"), "w"))
        return f"[ok] {scene}"
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
    print(f"{len(scenes)} scenes; bbox2d KMAX={KMAX}", flush=True)
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(process_scene,
                                     [(s, args.stride) for s in scenes])):
            done += 1
            if i % 100 == 0 or r.startswith("[fail"):
                print(f"{i + 1}/{len(scenes)} {r}", flush=True)
    print(f"DONE total={done}", flush=True)


if __name__ == "__main__":
    main()

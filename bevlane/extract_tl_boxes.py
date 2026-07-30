#!/usr/bin/env python3
"""Per-frame, per-camera BOX-LEVEL traffic-light states for the v47 input.

Source (dummy recognizer) = dataset color_shape annotations: lamp elements
are separate categories (red_circle / green_arrow / red_pedestrian ...);
arrows carry a row-level `orientation` (0=up, +pi/2=right, clockwise).

Writes per converted frame  tl/{fi:04d}.npz {"boxes": float32 [N,12]}:
  [cam_idx, x1,y1,x2,y2 (normalized), red, yel, grn, is_ped, is_arrow,
   sin(orient), cos(orient)]   and manifest key "tl".
Converted frame fi corresponds to raw keyframe 2*fi (pixel-verified).
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

CAMS = ["CAM_FRONT_WIDE", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
        "CAM_BACK_WIDE", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
        "CAM_FRONT_NARROW", "CAM_BACK_NARROW"]
COLORS = ("red", "yellow", "green")
OUT = "out/bevlane"


def process_scene(args):
    scene, raw = args
    if not os.path.isdir(os.path.join(raw, "annotation")):
        raw = os.path.join(raw, scene)          # raw root given
    A = os.path.join(raw, "annotation")
    man_p = os.path.join(OUT, scene, "manifest.json")
    try:
        man = json.load(open(man_p))
        cats = {c["token"]: c["name"] for c in
                json.load(open(A + "/category.json"))}
        sdl = json.load(open(A + "/sample_data.json"))
        fn = {s["token"]: (s.get("filename", ""), s.get("width", 2880),
                           s.get("height", 1860)) for s in sdl}
        per_kf = {}
        for o in json.load(open(A + "/object_ann.json")):
            name = cats.get(o["category_token"], "")
            col = name.split("_")[0]
            if col not in COLORS:
                continue
            f, w, h = fn.get(o["sample_data_token"], ("", 2880, 1860))
            cam = next((c for c in CAMS if c in f), None)
            if cam is None:
                continue
            try:
                ki = int(os.path.splitext(os.path.basename(f))[0])
            except ValueError:
                continue
            shape = name.split("_", 1)[1] if "_" in name else "circle"
            x1, y1, x2, y2 = o["bbox"]
            orient = o.get("orientation")
            row = [CAMS.index(cam), x1 / w, y1 / h, x2 / w, y2 / h,
                   float(col == "red"), float(col == "yellow"),
                   float(col == "green"), float("pedestrian" in shape),
                   float("arrow" in shape),
                   float(np.sin(orient)) if orient is not None else 0.0,
                   float(np.cos(orient)) if orient is not None else 0.0]
            per_kf.setdefault(ki, []).append(row)

        od = os.path.join(OUT, scene, "tl")
        os.makedirs(od, exist_ok=True)
        n = 0
        for f in man["frames"]:
            fi = f["frame"]
            rows = per_kf.get(2 * fi, [])
            np.savez_compressed(os.path.join(od, f"{fi:04d}.npz"),
                                boxes=np.array(rows, np.float32).reshape(-1, 12))
            f["tl"] = f"tl/{fi:04d}.npz"
            n += len(rows)
        man["tl"] = 1
        json.dump(man, open(man_p, "w"))
        return f"[ok] {scene} {n} elements"
    except Exception as e:
        return f"[err] {scene}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root",
                    default="/data1/dataset/aisin/converted_valid_delay")
    ap.add_argument("--scenes", default=None, help="file or comma list")
    ap.add_argument("--index", default=None,
                    help="scene|rawdir index file (overrides --scenes)")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    if a.index:
        pairs = [l.strip().split("|") for l in open(a.index) if "|" in l]
        print(f"{len(pairs)} scenes (index)", flush=True)
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for i, r in enumerate(ex.map(process_scene, pairs)):
                if i % 200 == 0 or r.startswith("[err]"):
                    print(f"{i + 1}/{len(pairs)} {r}", flush=True)
        print("TL_BOXES_DONE", flush=True)
        return
    if a.scenes and os.path.isfile(a.scenes):
        scenes = [l.strip() for l in open(a.scenes) if l.strip()]
    elif a.scenes:
        scenes = a.scenes.split(",")
    else:
        scenes = sorted(
            s for s in os.listdir(a.raw_root)
            if os.path.isfile(os.path.join(OUT, s, "manifest.json"))
            and os.path.isfile(os.path.join(a.raw_root, s,
                                            "annotation/object_ann.json")))
    print(f"{len(scenes)} scenes", flush=True)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, r in enumerate(ex.map(
                process_scene,
                [(s, os.path.join(a.raw_root, s)) for s in scenes])):
            if i % 50 == 0 or r.startswith("[err]"):
                print(f"{i + 1}/{len(scenes)} {r}", flush=True)
    print("TL_BOXES_DONE", flush=True)


if __name__ == "__main__":
    main()

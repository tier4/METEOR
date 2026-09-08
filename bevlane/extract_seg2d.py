#!/usr/bin/env python3
"""Per-camera 2D semantic-segmentation GT (stride-4) from the 2D panoptic masks.

Class taxonomy comes from comlops-21cls-autolabel-2504.csv (id 1..20 with the
Cityscapes-like colours listed there; id 0 = background = anything not listed,
15 unused). All 8 cameras. Saved as seg2d21/<fi>.npz (uint8 [8, sh, sw],
255 = ignore: no annotations for the camera, ego_vehicle, Invalid); manifest
frames gain a "seg2d21" key. The old 12-class "seg2d" files are left untouched
so a running 12-class training keeps working.
"""
import argparse
import base64
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import cv2
import numpy as np
from pycocotools import mask as cocomask

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bevlane.extract_gt import OUT, ROOT, IMG_H, IMG_W, load_scene_light  # noqa: E402
CAMS = ["CAM_FRONT_WIDE", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
        "CAM_BACK_WIDE", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
        "CAM_FRONT_NARROW", "CAM_BACK_NARROW"]

cv2.setNumThreads(1)
# all 8 cameras (direct 2D supervision for side/narrow too)
SEG_CAMS = set(CAMS)
STRIDE = 4
SH, SW = IMG_H // STRIDE, IMG_W // STRIDE     # 108 x 192

N_SEG21 = 21
_CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "comlops-21cls-autolabel-2504.csv")


def _load_taxonomy():
    cat2seg, pal = {}, np.zeros((N_SEG21, 3), np.uint8)
    for ln in open(_CSV).read().splitlines()[1:]:
        i, nm, r, g, b = ln.split(",")
        cat2seg[nm] = int(i)
        if not pal[int(i)].any():                 # first row per id = palette
            pal[int(i)] = (int(r), int(g), int(b))
    # dataset spelling variants / near-synonyms not in the csv
    cat2seg["striped_road_marking"] = cat2seg["striped_road_markings"]
    cat2seg["wheelchair"] = cat2seg["other_pedestrian"]
    # ego vehicle is always visible: supervise as background (0) so the hood
    # doesn't stay unsupervised noise; only broken annotations are ignored
    cat2seg["ego_vehicle"] = 0
    cat2seg["Invalid"] = 255
    return cat2seg, pal


CAT2SEG, SEG21_PAL = _load_taxonomy()             # SEG21_PAL is RGB
# thin/small classes: a nearest-neighbour resize to 108x192 makes sub-pixel
# lane lines vanish from the GT. Instead, mark a low-res cell as the thin
# class when its full-res footprint covers > THIN_FRAC of the cell.
# Priority: later wins (lanes on top).
THIN_IDS = (9, 10, 20, 8, 13)     # light, sign, pole, marking, lane
THIN_FRAC = 0.12
# paint order (later overwrites earlier): background surfaces first, thin last,
# ego/Invalid on top of everything (they occlude)
PAINT_ORDER = {c: i for i, c in enumerate(
    ["sky", "vegetation_terrain", "building", "wall_fence", "gate",
     "road", "parking_lot", "sidewalk", "crosswalk",
     "construction", "car", "truck", "bus", "motorcycle", "bicycle",
     "animal", "pedestrian", "stroller", "other_pedestrian", "wheelchair",
     "unknown", "road_debris", "cone", "guide_post", "pole",
     "traffic_sign", "traffic_light",
     "marking_other", "striped_road_markings", "striped_road_marking",
     "laneline_solid_white", "dashed_lane_marking", "deceleration_line",
     "stopline", "marking_character", "marking_arrow",
     "ego_vehicle", "Invalid"])}


def load_anns(scene_dir):
    ann = os.path.join(scene_dir, "annotation")
    cats = {c["token"]: c["name"] for c in
            json.load(open(os.path.join(ann, "category.json")))}
    by_sd = defaultdict(list)
    for name in ("surface_ann", "object_ann"):
        for a in json.load(open(os.path.join(ann, name + ".json"))):
            if a.get("mask"):
                nm = cats.get(a["category_token"], "")
                if nm in CAT2SEG:
                    by_sd[a["sample_data_token"]].append((nm, a["mask"]))
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
        os.makedirs(os.path.join(out_dir, "seg2d21"), exist_ok=True)
        for fr in man["frames"]:
            fi = fr["frame"]
            path = os.path.join(out_dir, f"seg2d21/{fi:04d}.npz")
            if os.path.exists(path):
                fr["seg2d21"] = f"seg2d21/{fi:04d}.npz"
                continue
            frame = frames[strided[fi]["token"]]
            seg = np.full((len(CAMS), SH, SW), 255, np.uint8)
            for ci, ch in enumerate(CAMS):
                if ch not in SEG_CAMS:
                    continue
                sd = frame.get(ch)
                if sd is None:
                    continue
                lst = anns.get(sd["token"], [])
                if not lst:
                    continue
                hw = tuple(lst[0][1]["size"])
                # base 0 = background: pixels of unlisted categories
                # (freespace, road_edge, obstacle_edge, ...) are none-of-21cls
                lab = np.zeros(hw, np.uint8)
                for nm, m in sorted(lst, key=lambda a: PAINT_ORDER.get(a[0], -1)):
                    rle = {"size": m["size"], "counts": base64.b64decode(m["counts"])}
                    lab[cocomask.decode(rle).astype(bool)] = CAT2SEG[nm]
                small = cv2.resize(lab, (SW, SH), interpolation=cv2.INTER_NEAREST)
                keep = small != 255                    # never overwrite ignore
                for tid in THIN_IDS:
                    frac = cv2.resize((lab == tid).astype(np.float32), (SW, SH),
                                      interpolation=cv2.INTER_AREA)
                    small[(frac > THIN_FRAC) & keep] = tid
                seg[ci] = small
            np.savez_compressed(path, seg=seg)
            fr["seg2d21"] = f"seg2d21/{fi:04d}.npz"
        json.dump(man, open(os.path.join(out_dir, "manifest.json"), "w"))
        return f"[ok] {scene}"
    except Exception as e:
        return f"[fail] {scene}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--scenes", default=None)
    args = ap.parse_args()
    if args.scenes:
        scenes = (open(args.scenes).read().split() if os.path.isfile(args.scenes)
                  else args.scenes.split(","))
    else:
        scenes = sorted(d for d in os.listdir(OUT)
                        if os.path.exists(os.path.join(OUT, d, "manifest.json")))
    print(f"{len(scenes)} scenes; seg2d {SH}x{SW}", flush=True)
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(process_scene,
                                     [(s, args.stride) for s in scenes])):
            done += 1
            if i % 50 == 0 or r.startswith("[fail"):
                print(f"{i + 1}/{len(scenes)} {r}", flush=True)
    print(f"DONE total={done}", flush=True)


if __name__ == "__main__":
    main()

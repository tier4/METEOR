#!/usr/bin/env python3
"""Per-frame ego-centric vector GT clips (MapTR-style) from scene vector maps.

For each manifest frame: transform vector polylines/polygons to the ego frame
(x fwd, y left) and clip to the +-30 m ROI. Written as frames_vector.json per
scene: {frame: {lines: {...}, polygons: {...}}} with metre coordinates.
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
from autolabel_bev import quat_to_rot  # noqa: E402
from bevlane.extract_gt import ROOT, load_scene_light  # noqa: E402

PROD = "out/production"
OUT = "out/bevlane"
LINE_CLASSES = ["laneline", "stopline", "road_edge"]
POLY_CLASSES = ["crosswalk", "sidewalk"]
HX, HY = 80.0, 50.0


def clip_polyline(p, margin=2.0):
    """Split an ego-frame polyline into segments inside the ROI box."""
    inside = (np.abs(p[:, 0]) <= HX + margin) & (np.abs(p[:, 1]) <= HY + margin)
    segs, cur = [], []
    for pt, ok in zip(p, inside):
        if ok:
            cur.append(pt)
        elif cur:
            if len(cur) >= 2:
                segs.append(np.array(cur))
            cur = []
    if len(cur) >= 2:
        segs.append(np.array(cur))
    return segs


def process_scene(args):
    scene, stride = args
    try:
        out_dir = os.path.join(OUT, scene)
        man = json.load(open(os.path.join(out_dir, "manifest.json")))
        vec = json.load(open(os.path.join(PROD, scene, "vector_map.json")))
        ordered, frames, _, egop = load_scene_light(os.path.join(ROOT, scene))
        strided = ordered[::stride]
        polys = {c: [np.asarray(p) for p in vec["classes"].get(c, [])]
                 for c in LINE_CLASSES + POLY_CLASSES}

        out = {}
        for fr in man["frames"]:
            fi = fr["frame"]
            s = strided[fi]
            ld = frames[s["token"]]["LIDAR_CONCAT"]
            ep = egop[ld["ego_pose_token"]]
            tx, ty = ep["translation"][:2]
            R = quat_to_rot(ep["rotation"])
            yaw = np.arctan2(R[1, 0], R[0, 0])
            c, sn = np.cos(yaw), np.sin(yaw)

            def to_ego(pw):
                dx, dy = pw[:, 0] - tx, pw[:, 1] - ty
                return np.stack([c * dx + sn * dy, -sn * dx + c * dy], 1)

            rec = {"lines": {}, "polygons": {}}
            for cname in LINE_CLASSES:
                segs = []
                for pw in polys[cname]:
                    segs += [np.round(s, 2).tolist()
                             for s in clip_polyline(to_ego(pw))]
                rec["lines"][cname] = segs
            for cname in POLY_CLASSES:
                pl = []
                for pw in polys[cname]:
                    pe = to_ego(pw)
                    if ((np.abs(pe[:, 0]) <= HX + 5)
                            & (np.abs(pe[:, 1]) <= HY + 5)).any():
                        pl.append(np.round(pe, 2).tolist())
                rec["polygons"][cname] = pl
            out[fi] = rec
        json.dump(out, open(os.path.join(out_dir, "frames_vector.json"), "w"))
        return f"[ok] {scene} {len(out)}"
    except Exception as e:
        return f"[fail] {scene}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    scenes = sorted(d for d in os.listdir(OUT)
                    if os.path.exists(os.path.join(OUT, d, "manifest.json"))
                    and not os.path.exists(os.path.join(OUT, d, "frames_vector.json")))
    print(f"{len(scenes)} scenes", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(process_scene,
                                     [(s, args.stride) for s in scenes])):
            if i % 100 == 0 or r.startswith("[fail"):
                print(f"{i + 1}/{len(scenes)} {r}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Per-frame LiDAR BEV raster for the optional pillar branch (roadmap C6b).

Points (ego frame, current sweep only) are rasterised onto the model's BEV
extent (+-80 x +-50 m) at 0.4 m into 4 channels:
  0: log1p(point count)         1: max z (m, clipped -1..4, 0 when empty)
  2: mean z                     3: occupancy (any point)
Saved per scene as lidar_bev/<fi>.npz {"lb": float16 [4,400,250]}.

Deployment computes the same raster from the raw pcd in the runtime; the
engine consumes it as a plain input, so LiDAR stays a host-side, optional
modality (feed zeros -> bit-equal camera-only, see model v32).
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
from autolabel_bev import Transform  # noqa: E402
from bevlane.extract_gt import OUT, ROOT, load_scene_light  # noqa: E402

GH, GW, RES = 400, 250, 0.4          # rows = (80-x)/0.4, cols = (50-y)/0.4
Z0, Z1 = -1.0, 4.0


def raster(pe):
    """ego-frame points [N,3] -> [4,GH,GW] float32."""
    lb = np.zeros((4, GH, GW), np.float32)
    r = ((80.0 - pe[:, 0]) / RES).astype(np.int32)
    c = ((50.0 - pe[:, 1]) / RES).astype(np.int32)
    ok = (r >= 0) & (r < GH) & (c >= 0) & (c < GW)
    r, c, z = r[ok], c[ok], np.clip(pe[ok, 2], Z0, Z1)
    if len(r) == 0:
        return lb
    flat = r * GW + c
    cnt = np.bincount(flat, minlength=GH * GW).astype(np.float32)
    zsum = np.bincount(flat, weights=z, minlength=GH * GW)
    zmax = np.full(GH * GW, Z0, np.float32)
    np.maximum.at(zmax, flat, z)
    occ = cnt > 0
    lb[0] = np.log1p(cnt).reshape(GH, GW)
    lb[1] = np.where(occ, zmax, 0.0).reshape(GH, GW)
    lb[2] = np.where(occ, zsum / np.maximum(cnt, 1), 0.0).reshape(GH, GW)
    lb[3] = occ.reshape(GH, GW).astype(np.float32)
    return lb


def process_scene(args):
    scene, stride = args
    try:
        out_dir = os.path.join(OUT, scene)
        mf = os.path.join(out_dir, "manifest.json")
        man = json.load(open(mf))
        sdir = os.path.join(ROOT, scene)
        ordered, frames, calib, egop = load_scene_light(sdir)
        os.makedirs(os.path.join(out_dir, "lidar_bev"), exist_ok=True)
        n = 0
        for fr in man["frames"]:
            fi = fr["frame"]
            path = os.path.join(out_dir, f"lidar_bev/{fi:04d}.npz")
            if os.path.exists(path):
                fr["lidar_bev"] = f"lidar_bev/{fi:04d}.npz"
                continue
            ri = fi * stride
            if ri >= len(ordered):
                continue
            ld = frames[ordered[ri]["token"]].get("LIDAR_CONCAT")
            if ld is None:
                continue
            pts = np.fromfile(os.path.join(sdir, ld["filename"]),
                              dtype=np.float32).reshape(-1, 5)[:, :3]
            cal = calib[ld["calibrated_sensor_token"]]
            pe = Transform(cal["rotation"], cal["translation"]) \
                .apply(pts.astype(np.float64)).astype(np.float32)
            np.savez_compressed(path, lb=raster(pe).astype(np.float16))
            fr["lidar_bev"] = f"lidar_bev/{fi:04d}.npz"
            n += 1
        json.dump(man, open(mf, "w"))
        return f"[ok] {scene} n={n}"
    except Exception as e:
        return f"[fail] {scene}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--scenes", default=None)
    args = ap.parse_args()
    if args.scenes:
        scenes = (open(args.scenes).read().split()
                  if os.path.isfile(args.scenes) else args.scenes.split(","))
    else:
        scenes = sorted(d for d in os.listdir(OUT)
                        if os.path.exists(os.path.join(OUT, d, "manifest.json")))
    print(f"{len(scenes)} scenes; lidar BEV raster", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(process_scene,
                                     [(s, args.stride) for s in scenes])):
            if i % 200 == 0 or r.startswith("[fail"):
                print(f"{i + 1}/{len(scenes)} {r}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

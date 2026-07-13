#!/usr/bin/env python3
"""Per-camera depth GT for LSS-style depth supervision.

For each manifest frame and camera: project the LiDAR sweep into the camera and
min-pool depth onto a stride-8 grid (64x36 for 512x288 input). Saved per frame
as depth_gt/<fi>.npz (float16 [6,36,64], 0 = invalid).
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
from autolabel_bev import Transform, quat_to_rot  # noqa: E402
from bevlane.extract_gt import CAMS, ROOT, load_scene_light  # noqa: E402

cv2.setNumThreads(1)
OUT = "out/bevlane"
DH, DW = 36, 64          # stride-8 grid for 512x288
MAX_DEPTH = 60.0


def process_scene(args):
    scene, stride = args
    try:
        out_dir = os.path.join(OUT, scene)
        man = json.load(open(os.path.join(out_dir, "manifest.json")))
        sdir = os.path.join(ROOT, scene)
        ordered, frames, calib, egop = load_scene_light(sdir)
        strided = ordered[::stride]

        # camera projection setup (K scaled to cached res; depth grid /8)
        cams = []
        for ch in CAMS:
            K = np.array(man["cams"][ch]["K"])
            T_ego_cam = np.array(man["cams"][ch]["T_ego_cam"])
            T_cam_ego = np.linalg.inv(T_ego_cam)
            cams.append((K / 8.0, T_cam_ego))

        os.makedirs(os.path.join(out_dir, "depth_gt"), exist_ok=True)
        done = 0
        for fr in man["frames"]:
            fi = fr["frame"]
            path = os.path.join(out_dir, f"depth_gt/{fi:04d}.npz")
            if os.path.exists(path):
                done += 1
                continue
            s = strided[fi]
            ld = frames[s["token"]]["LIDAR_CONCAT"]
            pts = np.fromfile(os.path.join(sdir, ld["filename"]),
                              dtype=np.float32).reshape(-1, 5)[:, :3].astype(np.float64)
            cal_l = calib[ld["calibrated_sensor_token"]]
            pts_ego = Transform(cal_l["rotation"], cal_l["translation"]).apply(pts)

            depth = np.zeros((len(CAMS), DH, DW), np.float32)
            for ci, (K8, T_cam_ego) in enumerate(cams):
                pc = pts_ego @ T_cam_ego[:3, :3].T + T_cam_ego[:3, 3]
                z = pc[:, 2]
                m = (z > 0.5) & (z < MAX_DEPTH)
                u = (K8[0, 0] * pc[m, 0] / z[m] + K8[0, 2]).astype(np.int32)
                v = (K8[1, 1] * pc[m, 1] / z[m] + K8[1, 2]).astype(np.int32)
                zm = z[m].astype(np.float32)
                ok = (u >= 0) & (u < DW) & (v >= 0) & (v < DH)
                d = np.full(DH * DW, np.inf, np.float32)
                np.minimum.at(d, v[ok] * DW + u[ok], zm[ok])
                d[np.isinf(d)] = 0.0
                depth[ci] = d.reshape(DH, DW)
            np.savez_compressed(path, depth=depth.astype(np.float16))
            fr["depth"] = f"depth_gt/{fi:04d}.npz"
            done += 1
        # record depth key in manifest
        for fr in man["frames"]:
            fr.setdefault("depth", f"depth_gt/{fr['frame']:04d}.npz")
        json.dump(man, open(os.path.join(out_dir, "manifest.json"), "w"))
        return f"[ok] {scene} {done}"
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
    print(f"{len(scenes)} scenes", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(process_scene,
                                     [(s, args.stride) for s in scenes])):
            if i % 50 == 0 or r.startswith("[fail"):
                print(f"{i + 1}/{len(scenes)} {r}", flush=True)


if __name__ == "__main__":
    main()

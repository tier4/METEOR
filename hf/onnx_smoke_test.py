#!/usr/bin/env python3
"""Run the plain (plugin-free) METEOR ONNX on one frame of a demo scene with onnxruntime.

This is the minimal "does the download work" check for the Hugging Face release and
doubles as the reference for how to feed the graph:

    imgs      uint8   [1, 8, 3, 432, 768]   RGB, 0..255, camera order = CAMS below
    K         float32 [1, 8, 3, 3]          intrinsics at 768x432 (manifest.json "K")
    T_cam_ego float32 [1, 8, 4, 4]          inverse of manifest.json "T_ego_cam"
    v0        float32 [1]                   ego speed [m/s] (ego_motion.npz "v0")

The engine variants with the LiDAR input additionally take
    lidar_bev  float32 [1, 4, 400, 250]     pillar raster (scene/lidar_bev/NNNN.npz)
    lidar_flag float32 [1]                  1.0 when lidar_bev is real, 0.0 otherwise

Usage:
    python3 hf/onnx_smoke_test.py --onnx meteor_v157c3Z.onnx --root data/valday [--frame 0]
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
import onnxruntime as ort

CAMS = ["CAM_FRONT_WIDE", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
        "CAM_BACK_WIDE", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
        "CAM_FRONT_NARROW", "CAM_BACK_NARROW"]


def load_frame(scene_dir, frame):
    m = json.load(open(os.path.join(scene_dir, "manifest.json")))
    f = m["frames"][frame]
    imgs, K, T = [], [], []
    for c in CAMS:
        bgr = cv2.imread(os.path.join(scene_dir, f["imgs"][c]))
        if bgr is None:
            raise FileNotFoundError(f["imgs"][c])
        if bgr.shape[:2] != (432, 768):
            bgr = cv2.resize(bgr, (768, 432), interpolation=cv2.INTER_AREA)
        imgs.append(bgr[:, :, ::-1].transpose(2, 0, 1))          # BGR -> RGB, HWC -> CHW
        K.append(np.array(m["cams"][c]["K"], np.float32))
        T.append(np.linalg.inv(np.array(m["cams"][c]["T_ego_cam"], np.float32)))
    v0 = np.load(os.path.join(scene_dir, "ego_motion.npz"))["v0"][frame]
    feed = {"imgs": np.stack(imgs)[None].astype(np.uint8),
            "K": np.stack(K)[None], "T_cam_ego": np.stack(T)[None],
            "v0": np.array([v0], np.float32)}
    lb = os.path.join(scene_dir, f.get("lidar_bev") or f"lidar_bev/{frame:04d}.npz")
    if os.path.isfile(lb):
        z = np.load(lb)
        feed["lidar_bev"] = z[z.files[0]].astype(np.float32)[None]
        feed["lidar_flag"] = np.array([1.0], np.float32)
    return feed, m["scene"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--root", required=True, help="demo root: <root>/<scene>/manifest.json")
    ap.add_argument("--scene", default=None, help="scene name (default: first in root)")
    ap.add_argument("--frame", type=int, default=0)
    a = ap.parse_args()

    scene_dir = os.path.join(a.root, a.scene) if a.scene else \
        sorted(glob.glob(os.path.join(a.root, "*", "manifest.json")))[0].rsplit("/", 1)[0]
    feed, name = load_frame(scene_dir, a.frame)

    s = ort.InferenceSession(a.onnx, providers=["CPUExecutionProvider"])
    names = {i.name for i in s.get_inputs()}
    for k in list(feed):
        if k not in names:
            feed.pop(k)                                  # camera-only graph: drop lidar
    if "lidar_bev" in names and "lidar_bev" not in feed:
        shp = [d if isinstance(d, int) else 1 for d in
               next(i for i in s.get_inputs() if i.name == "lidar_bev").shape]
        feed["lidar_bev"] = np.zeros(shp, np.float32)
        feed["lidar_flag"] = np.array([0.0], np.float32)
    outs = s.run(None, feed)
    print(f"scene {name} frame {a.frame}  v0={float(feed['v0'][0]):.2f} m/s")
    ok = True
    for o, v in zip(s.get_outputs(), outs):
        line = f"  {o.name:12s} {str(v.dtype):8s} {str(list(v.shape)):26s}"
        if o.name in ("lane", "seg2d", "depth"):
            nz = float((v > 0).mean())
            line += f" non-background {nz:.3f}"
            if o.name == "seg2d" and nz < 0.10:
                ok = False
        elif v.dtype != np.uint8:
            line += f" mean {float(v.astype(np.float32).mean()):+.4f}"
        print(line)
    print("SMOKE", "PASS" if ok else "FAIL (seg2d nearly empty)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

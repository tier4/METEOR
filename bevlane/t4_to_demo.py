#!/usr/bin/env python3
"""Convert a raw t4dataset scene directly into the format the demo reads (2026-08-25).

Unlike the training conversion (run_batch: LiDAR accumulation + GT generation,
tens of minutes per scene), the demo only needs images, K/T and poses, so this takes seconds:

    <out>/<scene>/manifest.json    cams (K, T_ego_cam) + frames (imgs)
    <out>/<scene>/img/*.jpg        8 camera images downscaled to 768x432
    <out>/<scene>/ego_motion.npz   pose [N,3] (x,y,yaw) and v0 [N]

Usage:
    python3 bevlane/t4_to_demo.py \
        --scene data/batchA/converted_valid_delay/Pct6CqsV_... \
        --out out/t4demo
    ./demo.sh --root out/t4demo   (or orin_render --root out/t4demo)

t4 gotcha (observed): ego_pose.json is not always in time order.
Always sort by timestamp before computing velocity.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bevlane.extract_gt import (IMG_H, IMG_W,                # noqa: E402
                                load_scene_light, quat_to_rot)
from bevlane.dataset import CAMS                             # noqa: E402  8-camera definition


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, help="t4 scene directory")
    ap.add_argument("--out", default="out/t4demo")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--cams", default=",".join(CAMS),
                    help="cameras to use (default = the 8 from training)")
    a = ap.parse_args()

    sdir = a.scene.rstrip("/")
    scene = os.path.basename(sdir)
    cams = [c for c in a.cams.split(",") if c]
    out_dir = os.path.join(a.out, scene)
    os.makedirs(os.path.join(out_dir, "img"), exist_ok=True)

    ordered, frames, calib, egop = load_scene_light(sdir)
    cam_cache, manifest = {}, []
    poses, stamps = [], []
    for fi, s in enumerate(ordered[::a.stride]):
        frame = frames.get(s["token"], {})
        imgs, skip = {}, False
        for ch in cams:
            sd = frame.get(ch)
            if sd is None:
                skip = True
                break
            name = f"img/{fi:04d}_{ch}.jpg"
            img = cv2.imread(os.path.join(sdir, sd["filename"]))
            if img is None:
                skip = True
                break
            h0, w0 = img.shape[:2]
            cv2.imwrite(os.path.join(out_dir, name),
                        cv2.resize(img, (IMG_W, IMG_H),
                                   interpolation=cv2.INTER_AREA),
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            if ch not in cam_cache:
                cal = calib[sd["calibrated_sensor_token"]]
                K = np.array(cal["camera_intrinsic"], dtype=np.float64)
                K[0] *= IMG_W / w0
                K[1] *= IMG_H / h0
                T = np.eye(4)
                T[:3, :3] = quat_to_rot(cal["rotation"])
                T[:3, 3] = cal["translation"]
                cam_cache[ch] = {"K": K.tolist(), "T_ego_cam": T.tolist()}
            imgs[ch] = name
        if skip:
            continue
        sd0 = frame[cams[0]]
        ep = egop[sd0["ego_pose_token"]]
        R = quat_to_rot(ep["rotation"])
        poses.append([float(ep["translation"][0]), float(ep["translation"][1]),
                      float(np.arctan2(R[1, 0], R[0, 0]))])
        stamps.append(float(sd0["timestamp"]) * 1e-6)
        manifest.append({"frame": len(manifest), "imgs": imgs})

    pose = np.array(poses, np.float32)
    ts = np.array(stamps, np.float64)
    v0 = np.zeros(len(pose), np.float32)
    if len(pose) > 1:
        d = np.linalg.norm(np.diff(pose[:, :2], axis=0), axis=1)
        dt = np.clip(np.diff(ts), 1e-3, None)
        v0[1:] = d / dt
        v0[0] = v0[1]
    np.savez(os.path.join(out_dir, "ego_motion.npz"),
             pose=pose, v0=v0, stamp=ts)
    json.dump({"scene": scene, "img_hw": [IMG_H, IMG_W],
               "cams": cam_cache, "frames": manifest},
              open(os.path.join(out_dir, "manifest.json"), "w"))
    print(f"{scene}: {len(manifest)} frames / {len(cam_cache)} cameras "
          f"/ v0 median {np.median(v0)*3.6:.1f} km/h -> {out_dir}")


if __name__ == "__main__":
    main()

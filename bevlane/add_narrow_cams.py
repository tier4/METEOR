#!/usr/bin/env python3
"""Add CAM_FRONT_NARROW / CAM_BACK_NARROW (telephoto) to the image cache.

Extracts resized narrow-cam images for every manifest frame and registers
their K / T_ego_cam + image paths in the manifest (existing entries untouched).
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
from autolabel_bev import quat_to_rot  # noqa: E402
from bevlane.extract_gt import IMG_H, IMG_W, OUT, ROOT, load_scene_light  # noqa: E402

cv2.setNumThreads(1)
NARROW = ["CAM_FRONT_NARROW", "CAM_BACK_NARROW"]


def process_scene(args):
    scene, stride = args
    try:
        out_dir = os.path.join(OUT, scene)
        mf = os.path.join(out_dir, "manifest.json")
        man = json.load(open(mf))
        if all(c in man["cams"] for c in NARROW) and \
                all(all(c in fr["imgs"] for c in NARROW) for fr in man["frames"]):
            return f"[skip] {scene}"
        sdir = os.path.join(ROOT, scene)
        ordered, frames, calib, egop = load_scene_light(sdir)
        strided = ordered[::stride]
        for fr in man["frames"]:
            frame = frames[strided[fr["frame"]]["token"]]
            for ch in NARROW:
                sd = frame.get(ch)
                if sd is None:
                    continue
                img_name = f"img/{fr['frame']:04d}_{ch}.jpg"
                dst = os.path.join(out_dir, img_name)
                img = cv2.imread(os.path.join(sdir, sd["filename"]))
                if img is None:
                    continue
                cv2.imwrite(dst, cv2.resize(img, (IMG_W, IMG_H),
                                            interpolation=cv2.INTER_AREA),
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
                if ch not in man["cams"]:
                    cal = calib[sd["calibrated_sensor_token"]]
                    K = np.array(cal["camera_intrinsic"], dtype=np.float64)
                    K[0] *= IMG_W / sd.get("width", 2880)
                    K[1] *= IMG_H / sd.get("height", 1860)
                    T = np.eye(4)
                    T[:3, :3] = quat_to_rot(cal["rotation"])
                    T[:3, 3] = cal["translation"]
                    man["cams"][ch] = {"K": K.tolist(), "T_ego_cam": T.tolist()}
                fr["imgs"][ch] = img_name
        json.dump(man, open(mf, "w"))
        return f"[ok] {scene}"
    except Exception as e:
        return f"[fail] {scene}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()
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

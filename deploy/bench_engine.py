#!/usr/bin/env python3
"""Latency / throughput benchmark for a METEOR TensorRT engine.

Measures, on real t4dataset frames (default: first scene of --t4d):
  1. pure engine execute latency (input tensors pre-staged)
  2. end-to-end per-frame latency = JPEG decode + resize + normalize + infer
Reports mean / p50 / p99 and FPS.

Usage:
  CUDA_VISIBLE_DEVICES=1 python3 deploy/bench_engine.py \
      --engine out/meteor_v41_fp16.engine \
      --t4d /data1/dataset/aisin/converted_valid_delay/<scene> [--iters 200]
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deploy.infer_t4dataset import (CAMS, IMG_H, IMG_W, load_scene,  # noqa
                                    preprocess_images, quat_to_rot, scale_K)
from deploy.runtime import MeteorRT  # noqa: E402


def pct(a, p):
    return float(np.percentile(np.asarray(a) * 1000.0, p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--t4d", required=True, help="scene dir for real frames")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    args = ap.parse_args()

    ordered, by_sample, calib, egop = load_scene(args.t4d)
    Ks, Ts = [], []
    for c in CAMS:
        K, T_cam_ego, d = calib[c]
        Ks.append(scale_K(K, d.get("width", 2880), d.get("height", 1860)))
        Ts.append(T_cam_ego)
    K_t = np.stack(Ks)[None].astype(np.float32)
    T_t = np.stack(Ts)[None].astype(np.float32)

    # collect real frames (jpeg paths) for the e2e path
    frames = []
    for s in ordered[::2]:
        cams = by_sample.get(s["token"], {})
        if any(c not in cams for c in CAMS):
            continue
        frames.append([os.path.join(args.t4d, cams[c]["filename"])
                       for c in CAMS])
        if len(frames) >= 40:
            break
    assert frames, "no complete frames in scene"

    rt = MeteorRT(args.engine)
    rt.reset()

    # ---- 1. pure engine execute (pre-staged input) ----
    imgs = [cv2.resize(cv2.imread(p), (IMG_W, IMG_H)) for p in frames[0]]
    x = preprocess_images(imgs)
    for _ in range(args.warmup):
        rt.infer(x, K_t, T_t, 5.0, pose=(0.0, 0.0, 0.0))
    lat = []
    for i in range(args.iters):
        t0 = time.perf_counter()
        rt.infer(x, K_t, T_t, 5.0, pose=(float(i) * 0.5, 0.0, 0.0))
        lat.append(time.perf_counter() - t0)
    print(f"[engine-only]  mean {pct(lat,50)*0+np.mean(lat)*1000:6.1f} ms | "
          f"p50 {pct(lat,50):6.1f} ms | p99 {pct(lat,99):6.1f} ms | "
          f"{1.0/np.mean(lat):5.1f} FPS", flush=True)

    # ---- 2. end-to-end: decode + resize + normalize + infer ----
    rt.reset()
    lat2, tdec, tpre, tinf = [], [], [], []
    n = 0
    while n < args.iters:
        for paths in frames:
            if n >= args.iters:
                break
            t0 = time.perf_counter()
            raw = [cv2.imread(p) for p in paths]
            t1 = time.perf_counter()
            imgs = [cv2.resize(im, (IMG_W, IMG_H)) for im in raw]
            x = preprocess_images(imgs)
            t2 = time.perf_counter()
            rt.infer(x, K_t, T_t, 5.0, pose=(float(n) * 0.5, 0.0, 0.0))
            t3 = time.perf_counter()
            lat2.append(t3 - t0)
            tdec.append(t1 - t0); tpre.append(t2 - t1); tinf.append(t3 - t2)
            n += 1
    print(f"[end-to-end]   mean {np.mean(lat2)*1000:6.1f} ms | "
          f"p50 {pct(lat2,50):6.1f} ms | p99 {pct(lat2,99):6.1f} ms | "
          f"{1.0/np.mean(lat2):5.1f} FPS", flush=True)
    print(f"  breakdown: jpeg-decode {np.mean(tdec)*1000:5.1f} ms | "
          f"preprocess {np.mean(tpre)*1000:5.1f} ms | "
          f"infer {np.mean(tinf)*1000:5.1f} ms", flush=True)


if __name__ == "__main__":
    main()

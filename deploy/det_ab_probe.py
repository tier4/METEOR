"""Paired INT8 vs fp16 detection comparison (on-device Orin, 2026-08-27).

Runs the same frame sequence through both engines and matches decode_boxes
results per class by center distance. Without GT on the device, the diff
against the fp16 reference is the primary signal of how INT8 changed detection:
  many b_only (boxes only INT8 emits) -> suspected precision drop
  many a_only (boxes only fp16 emits) -> suspected recall drop
Usage: python3 det_ab_probe.py <fp16.engine> <int8.engine> [root] [stride]
"""
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, "/home/nvidia/meteor")
from deploy.runtime import MeteorRT, decode_boxes

ORD = ["CAM_FRONT_WIDE", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK_WIDE",
       "CAM_BACK_LEFT", "CAM_BACK_RIGHT", "CAM_FRONT_NARROW", "CAM_BACK_NARROW"]

ENG_A, ENG_B = sys.argv[1], sys.argv[2]
ROOT = sys.argv[3] if len(sys.argv) > 3 else "calib"
STRIDE = int(sys.argv[4]) if len(sys.argv) > 4 else 4
TH = 0.25          # same threshold as the demo
MATCH_R = 1.5      # center distance [m]


def load_scene(d):
    man = json.load(open(os.path.join(d, "manifest.json")))
    K = np.stack([np.array(man["cams"][c]["K"], np.float32)
                  for c in ORD])[None]
    T = np.stack([np.linalg.inv(np.array(man["cams"][c]["T_ego_cam"],
                                         np.float32)) for c in ORD])[None]
    return man, K, T


def frame_img(d, f):
    return np.stack([cv2.imread(os.path.join(d, f["imgs"][c]))[:, :, ::-1]
                     .transpose(2, 0, 1) for c in ORD])[None].astype(np.uint8)


def match(da, db):
    """Per-class greedy match. Returns: matched, a_only, b_only (box lists)."""
    used = set()
    matched, a_only = [], []
    for a in da:
        best, bi = None, -1
        for i, b in enumerate(db):
            if i in used or b["cls"] != a["cls"]:
                continue
            dist = ((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2) ** 0.5
            if dist <= MATCH_R and (best is None or dist < best):
                best, bi = dist, i
        if bi >= 0:
            used.add(bi)
            matched.append((a, db[bi]))
        else:
            a_only.append(a)
    b_only = [b for i, b in enumerate(db) if i not in used]
    return matched, a_only, b_only


rt_a = MeteorRT(ENG_A, n_out_slots=1)
rt_b = MeteorRT(ENG_B, n_out_slots=1)

acc = {c: {"a": 0, "b": 0, "m": 0, "ao": 0, "bo": 0,
           "bo_sc": [], "ao_sc": [], "bo_far": 0} for c in ("vehicle", "vru")}
nfr = 0
scenes = sorted(os.listdir(ROOT))
for sc in scenes:
    d = os.path.join(ROOT, sc)
    if not os.path.isfile(os.path.join(d, "manifest.json")):
        continue
    man, K, T = load_scene(d)
    for f in man["frames"][::STRIDE]:
        im = frame_img(d, f)
        oa = rt_a.infer(im, K, T, 8.0, pose=(0., 0., 0.))
        ob = rt_b.infer(im, K, T, 8.0, pose=(0., 0., 0.))
        da = decode_boxes(np.asarray(oa["hm"], np.float32),
                          np.asarray(oa["reg"], np.float32), thresh=TH)
        db = decode_boxes(np.asarray(ob["hm"], np.float32),
                          np.asarray(ob["reg"], np.float32), thresh=TH)
        nfr += 1
        for cls in ("vehicle", "vru"):
            ca = [x for x in da if x["cls"] == cls]
            cb = [x for x in db if x["cls"] == cls]
            m, ao, bo = match(ca, cb)
            A = acc[cls]
            A["a"] += len(ca); A["b"] += len(cb); A["m"] += len(m)
            A["ao"] += len(ao); A["bo"] += len(bo)
            A["ao_sc"] += [x["score"] for x in ao]
            A["bo_sc"] += [x["score"] for x in bo]
            A["bo_far"] += sum(1 for x in bo if abs(x["x"]) > 40)

print(f"engines A={os.path.basename(ENG_A)} (reference) "
      f"B={os.path.basename(ENG_B)}  root={ROOT} frames={nfr} th={TH}")
for cls in ("vehicle", "vru"):
    A = acc[cls]
    agree = A["m"] / max(A["a"], 1)
    print(f"  {cls:8s} A={A['a']:5d} B={A['b']:5d} matched={A['m']:5d} "
          f"(agreement vs A {agree:.2f})  A_only={A['ao']:4d} "
          f"B_only={A['bo']:4d} (of which beyond 40 m {A['bo_far']})")
    for tag in ("ao", "bo"):
        s = A[tag + "_sc"]
        if s:
            print(f"    {tag}_score: mean {np.mean(s):.3f} "
                  f"p50 {np.median(s):.3f} max {np.max(s):.3f} "
                  f">0.4: {sum(1 for x in s if x > 0.4)}")
print("DET_AB_DONE")

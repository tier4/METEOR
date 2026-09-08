"""2D BBox (カメラ面) の INT8 vs fp16 ペア比較 (Orin 実機, 2026-08-27)。

det_ab_probe.py の 2D 版。同一フレームを 2 エンジンに流し、カメラ別・
クラス別に IoU>=0.3 で greedy マッチ。fp16 基準で
  b_only 多 → INT8 の precision 低下疑い / a_only 多 → recall 低下疑い。
使い方: python3 det2d_ab_probe.py <fp16.engine> <int8.engine> [root] [stride]
"""
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, "/home/nvidia/meteor")
from deploy.runtime import MeteorRT
from deploy.viz_np import decode_boxes2d_ms_np

ORD = ["CAM_FRONT_WIDE", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK_WIDE",
       "CAM_BACK_LEFT", "CAM_BACK_RIGHT", "CAM_FRONT_NARROW", "CAM_BACK_NARROW"]

ENG_A, ENG_B = sys.argv[1], sys.argv[2]
ROOT = sys.argv[3] if len(sys.argv) > 3 else "calib"
STRIDE = int(sys.argv[4]) if len(sys.argv) > 4 else 4
TH = 0.50            # demo (orin_render) と同じ閾値
IOU_TH = 0.3


def iou(a, b):
    ax0, ay0 = a[2] - a[4] / 2, a[3] - a[5] / 2
    ax1, ay1 = a[2] + a[4] / 2, a[3] + a[5] / 2
    bx0, by0 = b[2] - b[4] / 2, b[3] - b[5] / 2
    bx1, by1 = b[2] + b[4] / 2, b[3] + b[5] / 2
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    ua = a[4] * a[5] + b[4] * b[5] - inter
    return inter / max(ua, 1e-6)


def match(da, db):
    used = set()
    m, ao = 0, []
    for a in da:
        best, bi = 0.0, -1
        for i, b in enumerate(db):
            if i in used or b[0] != a[0]:
                continue
            v = iou(a, b)
            if v >= IOU_TH and v > best:
                best, bi = v, i
        if bi >= 0:
            used.add(bi)
            m += 1
        else:
            ao.append(a)
    bo = [b for i, b in enumerate(db) if i not in used]
    return m, ao, bo


def load_scene(d):
    man = json.load(open(os.path.join(d, "manifest.json")))
    K = np.stack([np.array(man["cams"][c]["K"], np.float32)
                  for c in ORD])[None]
    T = np.stack([np.linalg.inv(np.array(man["cams"][c]["T_ego_cam"],
                                         np.float32)) for c in ORD])[None]
    return man, K, T


rt_a = MeteorRT(ENG_A, n_out_slots=1)
rt_b = MeteorRT(ENG_B, n_out_slots=1)

acc = {}
nfr = 0
for sc in sorted(os.listdir(ROOT)):
    d = os.path.join(ROOT, sc)
    if not os.path.isfile(os.path.join(d, "manifest.json")):
        continue
    man, K, T = load_scene(d)
    for f in man["frames"][::STRIDE]:
        im = np.stack([cv2.imread(os.path.join(d, f["imgs"][c]))[:, :, ::-1]
                       .transpose(2, 0, 1) for c in ORD])[None] \
            .astype(np.uint8)
        oa = rt_a.infer(im, K, T, 8.0, pose=(0., 0., 0.))
        ob = rt_b.infer(im, K, T, 8.0, pose=(0., 0., 0.))
        da = decode_boxes2d_ms_np(
            [np.asarray(oa[f"hm2d_s{i}"], np.float32)[0] for i in range(3)],
            [np.asarray(oa[f"reg2d_s{i}"], np.float32)[0] for i in range(3)],
            thresh=TH)
        db = decode_boxes2d_ms_np(
            [np.asarray(ob[f"hm2d_s{i}"], np.float32)[0] for i in range(3)],
            [np.asarray(ob[f"reg2d_s{i}"], np.float32)[0] for i in range(3)],
            thresh=TH)
        nfr += 1
        for ci in range(len(ORD)):
            for a_or_b, boxes in (("a", da[ci]), ("b", db[ci])):
                for x in boxes:
                    acc.setdefault(x[0], {"a": 0, "b": 0, "m": 0, "ao": 0,
                                          "bo": 0, "ao_sc": [], "bo_sc": []})
            classes = {x[0] for x in da[ci]} | {x[0] for x in db[ci]}
            for cls in classes:
                A = acc.setdefault(cls, {"a": 0, "b": 0, "m": 0, "ao": 0,
                                         "bo": 0, "ao_sc": [], "bo_sc": []})
                ca = [x for x in da[ci] if x[0] == cls]
                cb = [x for x in db[ci] if x[0] == cls]
                m, ao, bo = match(ca, cb)
                A["a"] += len(ca); A["b"] += len(cb); A["m"] += m
                A["ao"] += len(ao); A["bo"] += len(bo)
                A["ao_sc"] += [x[1] for x in ao]
                A["bo_sc"] += [x[1] for x in bo]

print(f"2D-BB  A={os.path.basename(ENG_A)} (基準) "
      f"B={os.path.basename(ENG_B)}  root={ROOT} frames={nfr} "
      f"th={TH} IoU>={IOU_TH}")
for cls in sorted(acc):
    A = acc[cls]
    if A["a"] + A["b"] == 0:
        continue
    agree = A["m"] / max(A["a"], 1)
    print(f"  cls{cls} A={A['a']:5d} B={A['b']:5d} 一致={A['m']:5d} "
          f"(A基準一致率 {agree:.2f})  A_only={A['ao']:4d} B_only={A['bo']:4d}")
    for tag in ("ao", "bo"):
        s = A[tag + "_sc"]
        if s:
            print(f"    {tag}_score: mean {np.mean(s):.3f} "
                  f"p50 {np.median(s):.3f} max {np.max(s):.3f} "
                  f">0.65: {sum(1 for x in s if x > 0.65)}")
print("DET2D_AB_DONE")

"""3D 検出の GT 付き fp16 vs INT8 判定 (ローカル GPU, 2026-08-27)。

Orin ペア比較で「INT8 が 40m 超の車両箱を +2.4/frame 出す」を検出したが
GT なしでは幻影か実物か判定不能だった。ローカル val シーン (bev_box GT) で
P/R をゾーン別に測って白黒つける。
使い方: python3 deploy/det3d_gt_probe.py <engine> [n_scenes] [stride]
GT: boxes [N,6] = (cls, x_fwd, y_left, L, W, yaw)、cls1=vehicle。
"""
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, ".")
from deploy.runtime import MeteorRT, decode_boxes

ORD = ["CAM_FRONT_WIDE", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK_WIDE",
       "CAM_BACK_LEFT", "CAM_BACK_RIGHT", "CAM_FRONT_NARROW", "CAM_BACK_NARROW"]
ROOT = "out/bevlane"
ENG = sys.argv[1]
NSC = int(sys.argv[2]) if len(sys.argv) > 2 else 10
STRIDE = int(sys.argv[3]) if len(sys.argv) > 3 else 4
TH = 0.25
MATCH_R = 2.0


def zone(x):
    if x >= 0:
        return 0 if x < 40 else (1 if x <= 80 else None)
    return 2 if x > -40 else (3 if x >= -80 else None)


rt = MeteorRT(ENG, n_out_slots=1)
tp = [0, 0, 0, 0]; fp = [0, 0, 0, 0]; fn = [0, 0, 0, 0]
nfr = 0
scenes = [s for s in open("val.lst").read().split()][:NSC]
for sc in scenes:
    d = os.path.join(ROOT, sc)
    mp = os.path.join(d, "manifest.json")
    if not os.path.isfile(mp):
        continue
    man = json.load(open(mp))
    K = np.stack([np.array(man["cams"][c]["K"], np.float32)
                  for c in ORD])[None]
    T = np.stack([np.linalg.inv(np.array(man["cams"][c]["T_ego_cam"],
                                         np.float32)) for c in ORD])[None]
    for fi, f in enumerate(man["frames"]):
        if fi % STRIDE:
            continue
        gtp = os.path.join(d, "bev_box", f"{fi:04d}.npz")
        if not os.path.isfile(gtp):
            continue
        gt = np.load(gtp)["boxes"]
        gt = gt[gt[:, 0] == 1] if len(gt) else gt   # vehicle のみ
        im = np.stack([cv2.imread(os.path.join(d, f["imgs"][c]))[:, :, ::-1]
                       .transpose(2, 0, 1) for c in ORD])[None] \
            .astype(np.uint8)
        o = rt.infer(im, K, T, 8.0, pose=(0., 0., 0.))
        det = [b for b in decode_boxes(np.asarray(o["hm"], np.float32),
                                       np.asarray(o["reg"], np.float32),
                                       thresh=TH) if b["cls"] == "vehicle"]
        nfr += 1
        used = set()
        for b in det:
            zi = zone(b["x"])
            best, gi = None, -1
            for i in range(len(gt)):
                if i in used:
                    continue
                dist = ((b["x"] - gt[i, 1]) ** 2
                        + (b["y"] - gt[i, 2]) ** 2) ** 0.5
                if dist <= MATCH_R and (best is None or dist < best):
                    best, gi = dist, i
            if gi >= 0:
                used.add(gi)
                if zi is not None:
                    tp[zi] += 1
            elif zi is not None:
                fp[zi] += 1
        for i in range(len(gt)):
            if i in used:
                continue
            zi = zone(float(gt[i, 1]))
            if zi is not None:
                fn[zi] += 1

names = ("F0-40", "F40-80", "R0-40", "R40-80")
print(f"engine={os.path.basename(ENG)} scenes={len(scenes)} frames={nfr} "
      f"th={TH} match<={MATCH_R}m (vehicle)")
TPa = sum(tp); FPa = sum(fp); FNa = sum(fn)
print(f"  ALL    P={TPa / max(TPa + FPa, 1):.3f} "
      f"R={TPa / max(TPa + FNa, 1):.3f}  tp={TPa} fp={FPa} fn={FNa}")
for i, nm in enumerate(names):
    print(f"  {nm:6s} P={tp[i] / max(tp[i] + fp[i], 1):.3f} "
          f"R={tp[i] / max(tp[i] + fn[i], 1):.3f}  "
          f"tp={tp[i]} fp={fp[i]} fn={fn[i]}")
print("DET3D_GT_DONE")

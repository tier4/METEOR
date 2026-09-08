"""2D BBox の GT 付き fp16 vs INT8 判定 (ローカル, 2026-08-27)。
使い方: python3 deploy/det2d_gt_probe.py <engine> [n_scenes] [stride]
GT: bbox2d/%04d.npz boxes[8,96,5]=(cls,cx,cy,w,h), counts[8]。
"""
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, ".")
from deploy.runtime import MeteorRT
from deploy.viz_np import decode_boxes2d_ms_np, DET10_ABBR

ORD = ["CAM_FRONT_WIDE", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK_WIDE",
       "CAM_BACK_LEFT", "CAM_BACK_RIGHT", "CAM_FRONT_NARROW", "CAM_BACK_NARROW"]
ROOT = "out/bevlane"
ENG = sys.argv[1]
NSC = int(sys.argv[2]) if len(sys.argv) > 2 else 10
STRIDE = int(sys.argv[3]) if len(sys.argv) > 3 else 4
TH = float(os.environ.get("TH2D", "0.50"))
IOU = 0.3


def iou(a, b):
    ax0, ay0, ax1, ay1 = a[0]-a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2
    bx0, by0, bx1, by1 = b[0]-b[2]/2, b[1]-b[3]/2, b[0]+b[2]/2, b[1]+b[3]/2
    iw = max(0., min(ax1, bx1) - max(ax0, bx0))
    ih = max(0., min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    return inter / max(a[2]*a[3] + b[2]*b[3] - inter, 1e-6)


rt = MeteorRT(ENG, n_out_slots=1)
acc = {}
nfr = 0
for sc in open("val.lst").read().split()[:NSC]:
    d = os.path.join(ROOT, sc)
    mp = os.path.join(d, "manifest.json")
    if not os.path.isfile(mp):
        continue
    man = json.load(open(mp))
    K = np.stack([np.array(man["cams"][c]["K"], np.float32) for c in ORD])[None]
    T = np.stack([np.linalg.inv(np.array(man["cams"][c]["T_ego_cam"],
                                         np.float32)) for c in ORD])[None]
    for fi, f in enumerate(man["frames"]):
        if fi % STRIDE:
            continue
        gp = os.path.join(d, "bbox2d", f"{fi:04d}.npz")
        if not os.path.isfile(gp):
            continue
        z = np.load(gp)
        gtb, cnt = z["boxes"], z["counts"]
        im = np.stack([cv2.imread(os.path.join(d, f["imgs"][c]))[:, :, ::-1]
                       .transpose(2, 0, 1) for c in ORD])[None].astype(np.uint8)
        o = rt.infer(im, K, T, 8.0, pose=(0., 0., 0.))
        det = decode_boxes2d_ms_np(
            [np.asarray(o[f"hm2d_s{i}"], np.float32)[0] for i in range(3)],
            [np.asarray(o[f"reg2d_s{i}"], np.float32)[0] for i in range(3)],
            thresh=TH)
        nfr += 1
        for ci in range(8):
            gts = [(int(gtb[ci, j, 0]), gtb[ci, j, 1:5])
                   for j in range(int(cnt[ci]))]
            used = set()
            for x in sorted(det[ci], key=lambda x: -x[1]):
                cls = x[0]
                A = acc.setdefault(cls, {"tp": 0, "fp": 0, "fn": 0})
                best, gi = 0.0, -1
                for j, (gc, gb) in enumerate(gts):
                    if j in used or gc != cls:
                        continue
                    v = iou(x[2:6], gb)
                    if v >= IOU and v > best:
                        best, gi = v, j
                if gi >= 0:
                    used.add(gi)
                    A["tp"] += 1
                else:
                    A["fp"] += 1
            for j, (gc, gb) in enumerate(gts):
                if j not in used:
                    acc.setdefault(gc, {"tp": 0, "fp": 0, "fn": 0})["fn"] += 1

print(f"engine={os.path.basename(ENG)} frames={nfr} th={TH} IoU>={IOU}")
for cls in sorted(acc):
    A = acc[cls]
    if A["tp"] + A["fp"] + A["fn"] < 20:
        continue
    nm = DET10_ABBR[cls] if cls < len(DET10_ABBR) else str(cls)
    print(f"  {nm:4s} P={A['tp']/max(A['tp']+A['fp'],1):.3f} "
          f"R={A['tp']/max(A['tp']+A['fn'],1):.3f} "
          f"tp={A['tp']} fp={A['fp']} fn={A['fn']}")
print("DET2D_GT_DONE")

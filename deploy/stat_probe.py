"""Directly measure whether the stationary head collapses under INT8.
Rendering samples stationary at det positions to emit "stopped", so look at
both the logit distribution and the values actually read at box positions."""
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, "/home/nvidia/meteor")
from deploy.runtime import MeteorRT, decode_boxes

ORD = ["CAM_FRONT_WIDE", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK_WIDE",
       "CAM_BACK_LEFT", "CAM_BACK_RIGHT", "CAM_FRONT_NARROW", "CAM_BACK_NARROW"]
d = os.path.join("sample", sorted(os.listdir("sample"))[0])
man = json.load(open(os.path.join(d, "manifest.json")))
K = np.stack([np.array(man["cams"][c]["K"], np.float32) for c in ORD])[None]
T = np.stack([np.linalg.inv(np.array(man["cams"][c]["T_ego_cam"], np.float32))
              for c in ORD])[None]
frames = [np.stack([cv2.imread(os.path.join(d, f["imgs"][c]))[:, :, ::-1]
                    .transpose(2, 0, 1) for c in ORD])[None].astype(np.uint8)
          for f in man["frames"][5:25:4]]


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


for eng in sys.argv[1:]:
    rt = MeteorRT(eng, n_out_slots=1)
    lo, hi, nb, nstat = [], [], 0, 0
    for im in frames:
        o = rt.infer(im, K, T, 8.0, pose=(0., 0., 0.))
        st = np.asarray(o["stationary"], np.float32).reshape(-1)
        lo.append(st.std()); hi.append(float(np.abs(st).max()))
        hm = np.asarray(o["hm"], np.float32)
        rg = np.asarray(o["reg"], np.float32)
        dets = decode_boxes(hm, rg, thresh=0.25,
                            stationary=np.asarray(o["stationary"], np.float32))
        for b in dets:
            nb += 1
            nstat += int(bool(b.get("stationary")))
    print(f"{os.path.basename(eng):<26} logit std {np.mean(lo):6.3f} "
          f"|max| {np.mean(hi):6.2f}  boxes {nb} stationary {nstat} "
          f"({nstat / max(nb, 1) * 100:.0f}%)")
    del rt

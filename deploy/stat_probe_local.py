"""stationary (停止判定) ヘッドが INT8 で潰れていないかを直接測る。
描画は det の位置で stationary をサンプルして「停止」を出すので、
ロジットの分布と、実際に箱の位置で読んだ値の両方を見る。"""
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, ".")
from deploy.runtime import MeteorRT, decode_boxes

ORD = ["CAM_FRONT_WIDE", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK_WIDE",
       "CAM_BACK_LEFT", "CAM_BACK_RIGHT", "CAM_FRONT_NARROW", "CAM_BACK_NARROW"]
d = os.path.join("out/bevlane", open("val.lst").readline().strip())
man = json.load(open(os.path.join(d, "manifest.json")))
_z = np.load(os.path.join(d, "ego_motion.npz"))
_poses = _z["pose"] if "pose" in _z else None
_v0s = _z["v0"] if "v0" in _z else None
K = np.stack([np.array(man["cams"][c]["K"], np.float32) for c in ORD])[None]
T = np.stack([np.linalg.inv(np.array(man["cams"][c]["T_ego_cam"], np.float32))
              for c in ORD])[None]
_sel = man["frames"][5:45:2]     # 時間連続 (stride2) -- delta-stat のワープに必要
frames = [(np.stack([cv2.imread(os.path.join(d, f["imgs"][c]))[:, :, ::-1]
                     .transpose(2, 0, 1) for c in ORD])[None].astype(np.uint8),
           tuple(float(x) for x in _poses[f["frame"]]) if _poses is not None else (0., 0., 0.),
           float(_v0s[f["frame"]]) if _v0s is not None else 8.0)
          for f in _sel]


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


for eng in sys.argv[1:]:
    rt = MeteorRT(eng, n_out_slots=1)
    lo, hi, nb, nstat = [], [], 0, 0
    for im, po, v0 in frames:
        o = rt.infer(im, K, T, v0, pose=po)
        st = np.asarray(o["stationary"], np.float32).reshape(-1)
        lo.append(st.std()); hi.append(float(np.abs(st).max()))
        hm = np.asarray(o["hm"], np.float32)
        rg = np.asarray(o["reg"], np.float32)
        dets = decode_boxes(hm, rg, thresh=0.25,
                            stationary=np.asarray(o["stationary"], np.float32))
        for b in dets:
            nb += 1
            nstat += int(bool(b.get("stationary")))
    print(f"{os.path.basename(eng):<26} ロジット std {np.mean(lo):6.3f} "
          f"|最大| {np.mean(hi):6.2f}  箱 {nb} 個中 停止判定 {nstat} "
          f"({nstat / max(nb, 1) * 100:.0f}%)")
    del rt

"""torch なしでランタイム込みのレイテンシを測る (Orin 用)。

trtexec の数字は H2D 12ms / D2H 5.8ms を含むが、実ランタイムは pinned
memory + 非同期転送なのでそのままは出ない。90ms 目標に対して「実機で
本当に何 ms か」を出すのが目的。
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/home/nvidia/meteor")
from deploy.runtime import MeteorRT

CAMS = ["CAM_FRONT_LEFT", "CAM_FRONT_WIDE", "CAM_FRONT_RIGHT", "CAM_FRONT_NARROW",
        "CAM_BACK_LEFT", "CAM_BACK_WIDE", "CAM_BACK_RIGHT", "CAM_BACK_NARROW"]

root = "fast"
s = sorted(os.listdir(root))[0]
d = os.path.join(root, s)
man = json.load(open(os.path.join(d, "manifest.json")))
emo = np.load(os.path.join(d, "ego_motion.npz"))
K = np.stack([np.array(man["cams"][c]["K"], np.float32) for c in CAMS])[None]
T = np.stack([np.linalg.inv(np.array(man["cams"][c]["T_ego_cam"], np.float32))
              for c in CAMS])[None]
import cv2
frames = []
for f in man["frames"][3:23]:
    fi = int(f["frame"])
    ims = np.stack([cv2.imread(os.path.join(d, f["imgs"][c]))[:, :, ::-1]
                    .transpose(2, 0, 1) for c in CAMS])
    frames.append((ims[None].astype(np.uint8), float(emo["v0"][fi]),
                   tuple(float(x) for x in emo["pose"][fi])))
print(f"{len(frames)} フレーム読み込み")

# METEOR_SKIP_OUT: D2H しない出力名 (カンマ区切り)。描画に使わない大物
# (depth 42MB / seg2d 14MB / 2D 検出ヘッド) を止めると転送とホストコピーが
# 丸ごと消える。METEOR_OUT_SLOTS>1 なら出力の .copy() も省ける。
_SKIP = tuple(x for x in os.environ.get("METEOR_SKIP_OUT", "").split(",") if x)
_SLOTS = int(os.environ.get("METEOR_OUT_SLOTS", "1"))
if _SKIP:
    print(f"[bench] D2H を止める出力: {list(_SKIP)}")
print(f"[bench] out_slots = {_SLOTS}")
for eng in sys.argv[1:]:
    rt = MeteorRT(eng, skip_outputs=_SKIP, n_out_slots=_SLOTS)
    for i in range(8):                       # warmup
        rt.infer(*frames[i % len(frames)][:1], K, T, frames[0][1],
                 pose=frames[0][2])
    ts = []
    for i in range(40):
        ims, v0, po = frames[i % len(frames)]
        t = time.perf_counter()
        rt.infer(ims, K, T, v0, pose=po, out_slot=i % _SLOTS)
        ts.append((time.perf_counter() - t) * 1000)
    a = np.array(ts)
    try:
        from deploy.runtime import rt_profile_report
        rep = rt_profile_report()
    except Exception as e:
        rep = f"(内訳取得不可: {e})"
    print(f"{eng}: 中央値 {np.median(a):7.2f} ms  平均 {a.mean():7.2f} ms  "
          f"最小 {a.min():7.2f}  p90 {np.percentile(a, 90):7.2f}  "
          f"-> {1000 / np.median(a):.1f} FPS")
    print(rep)
    del rt
print("BENCH_RT_DONE")

"""raw_bev (BEV 特徴) がフレーム間で変化しているかを INT8/fp16 で比較。
ego 凍結の原因が上流 (特徴) か下流 (ego ヘッド) かを切り分ける。"""
import sys, os, json
sys.path.insert(0, "/home/nvidia/meteor")
import numpy as np, cv2
from deploy.runtime import MeteorRT
CAMS = ["CAM_FRONT_LEFT","CAM_FRONT_WIDE","CAM_FRONT_RIGHT","CAM_FRONT_NARROW",
        "CAM_BACK_LEFT","CAM_BACK_WIDE","CAM_BACK_RIGHT","CAM_BACK_NARROW"]
root = "fast"
s = sorted(os.listdir(root))[0]
d = os.path.join(root, s)
man = json.load(open(os.path.join(d, "manifest.json")))
emo = np.load(os.path.join(d, "ego_motion.npz"))
K = np.stack([np.array(man["cams"][c]["K"], np.float32) for c in CAMS])[None]
T = np.stack([np.linalg.inv(np.array(man["cams"][c]["T_ego_cam"], np.float32)) for c in CAMS])[None]
fr = []
for f in man["frames"][3:35:10]:
    fi = int(f["frame"])
    ims = np.stack([cv2.imread(os.path.join(d, f["imgs"][c]))[:, :, ::-1].transpose(2,0,1) for c in CAMS])
    fr.append((fi, ims[None].astype(np.uint8), float(emo["v0"][fi]),
               tuple(float(x) for x in emo["pose"][fi])))
for eng in sys.argv[1].split(","):
    rt = MeteorRT(eng, n_out_slots=1)
    print(f"\n=== {eng}")
    prev_bev = prev_ego = None
    for fi, ims, v0, po in fr:
        o = rt.infer(ims, K, T, v0, pose=po)
        bev = np.concatenate([np.asarray(o[k], np.float32).reshape(-1) for k in ("occ","traj")])
        ego = np.asarray(o["ego"], np.float32).reshape(-1)
        db = np.abs(bev - prev_bev).mean() if prev_bev is not None else 0.0
        de = np.abs(ego - prev_ego).mean() if prev_ego is not None else 0.0
        print(f"  f{fi:03d} occ+traj(std {bev.std():7.4f} 前frame差 {db:7.5f}) "
              f"ego(std {ego.std():7.4f} 前frame差 {de:7.5f})")
        prev_bev, prev_ego = bev, ego
    del rt
print("BEV_FROZEN_TEST_DONE")

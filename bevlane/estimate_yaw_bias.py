"""シーンごとのヨーバイアス b̂ = median(進行方位 − pose ヨー) を推定して保存。

直進区間 (0.4 s で 2 m 以上移動・ヨーレート ≈0) のみ使用。サンプルが
少ないシーンは同じ root の全シーン中央値で代替 (車両/日付単位の取り付け差)。
出力: JSON {scene: {"bias_deg": float, "n": int, "fallback": bool}}
"""
import argparse
import json
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--list", required=True)
ap.add_argument("--min-n", type=int, default=30)
ap.add_argument("--out", required=True)
a = ap.parse_args()

scenes = [l.strip() for l in open(a.list) if l.strip()]
per = {}
for s in scenes:
    p = os.path.join(a.root, s, "ego_motion.npz")
    diffs = []
    if os.path.exists(p):
        try:
            pose = np.load(p)["pose"]
            for i in range(2, len(pose) - 3):
                p0, p1 = pose[i], pose[i + 2]
                if abs(p0).sum() == 0 or abs(p1).sum() == 0:
                    continue
                dx, dy = p1[0] - p0[0], p1[1] - p0[1]
                dyaw = abs((p1[2] - p0[2] + np.pi) % (2 * np.pi) - np.pi)
                if np.hypot(dx, dy) < 2.0 or dyaw > 0.005:
                    continue
                mid = p0[2] + ((p1[2] - p0[2] + np.pi) % (2 * np.pi)
                               - np.pi) / 2
                d = (np.arctan2(dy, dx) - mid + np.pi) % (2 * np.pi) - np.pi
                if abs(d) < 0.1:
                    diffs.append(d)
        except Exception:
            pass
    per[s] = diffs

all_d = [d for v in per.values() for d in v]
root_med = float(np.median(all_d)) if all_d else 0.0
out = {}
n_fb = 0
for s, diffs in per.items():
    if len(diffs) >= a.min_n:
        out[s] = {"bias_deg": float(np.degrees(np.median(diffs))),
                  "n": len(diffs), "fallback": False}
    else:
        out[s] = {"bias_deg": float(np.degrees(root_med)),
                  "n": len(diffs), "fallback": True}
        n_fb += 1
json.dump(out, open(a.out, "w"), indent=1, ensure_ascii=False)
bs = [v["bias_deg"] for v in out.values()]
print(f"{a.out}: {len(out)} シーン (代替 {n_fb}) "
      f"root中央値 {np.degrees(root_med):+.3f}° "
      f"分布 {np.mean(bs):+.3f}±{np.std(bs):.3f}°")
print("YAW_BIAS_DONE")

"""BEV 出力の系統的な横ずれを GT 基準で測る (torch 不要、エンジン/ckpt 共用は不可,
エンジン専用)。指摘「rtv_r73 がやや右にオフセット」の定量化。

  1) 検出箱: GT と 3 m でマッチした車両の (予測 y - GT y) の平均。
     画面の右 = y 負方向なので、右オフセットなら負の平均が出る。
  2) 路面ラスタ: lane argmax の road クラスを GT (gt_cons) と列方向に
     ずらしながら IoU を取り、最良シフトを求める (0.2 m/列)。
"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deploy.runtime import MeteorRT, decode_boxes
from deploy.eval_deploy import frames

ap = argparse.ArgumentParser()
ap.add_argument("--engine", required=True)
ap.add_argument("--root", default="fast")
ap.add_argument("--scenes-file", default="out/eval_scenes.txt")
ap.add_argument("--limit", type=int, default=120)
ap.add_argument("--tag", default="")
a = ap.parse_args()
scenes = [l.strip() for l in open(a.scenes_file) if l.strip()]
rt = MeteorRT(a.engine, n_out_slots=1)
dys, shifts = [], []
for imgs, K, Tc, v0, pose, gt, bx in frames(a.root, scenes, 8, 2, a.limit):
    o = rt.infer(imgs, K, Tc, v0, pose=pose)
    dets = decode_boxes(o["hm"], o["reg"], thresh=0.25)
    pred = [(d["x"], d["y"]) for d in dets if d["cls"] == "vehicle"]
    for r in bx:
        if int(r[0]) != 1 or float(r[3]) <= 0: continue
        gx, gy = float(r[1]), float(r[2])
        if (gx*gx + gy*gy) ** 0.5 > 45 or abs(gy) > 40: continue
        best = None
        for px, py in pred:
            d2 = (gx-px)**2 + (gy-py)**2
            if d2 < 9.0 and (best is None or d2 < best[0]): best = (d2, py)
        if best is not None: dys.append(best[1] - gy)
    lane = o["lane"][0]
    if lane.ndim == 3: lane = lane.argmax(0)
    pr = (lane == 1); gr = (gt == 1)
    if gr.sum() < 500: continue
    ious = []
    for sh in range(-5, 6):
        p2 = np.roll(pr, sh, axis=1)
        ious.append(((p2 & gr).sum() / max((p2 | gr).sum(), 1), sh))
    shifts.append(max(ious)[1])
dys = np.array(dys); shifts = np.array(shifts)
print(f"=== {a.tag or a.engine} ===")
print(f"検出箱の横ずれ (n={len(dys)}): 平均 {dys.mean():+.3f} m / 中央値 {np.median(dys):+.3f} m"
      f"  (負 = 右へオフセット)")
print(f"路面ラスタの最良列シフト (n={len(shifts)}): 平均 {shifts.mean():+.2f} 列"
      f" = {0.2*shifts.mean():+.2f} m (正 = 予測を右へずらすと合う = 予測は左寄り)")

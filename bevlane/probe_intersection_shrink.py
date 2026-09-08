"""交差点接近時に近傍 road がしぼむ現象の定量化 (2026-08-22)。

各フレームで (a) 自車近傍 0-20m の road 面積、(b) 前方 30m 以内の
crosswalk 画素数 (= 交差点への近さの代理) を pred / GT 双方で測り、
交差点接近と road 面積の関係を見る。pred だけで縮むならモデル側の問題、
GT でも縮むなら教師/オクルージョン由来。
"""
import argparse, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bevlane.dataset import BevLaneDataset
from bevlane.model import MODELS, BEV_H

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--list", default="val.lst")
ap.add_argument("--root", default="out/bevlane")
ap.add_argument("--scenes", type=int, default=12)
ap.add_argument("--tag", default="")
a = ap.parse_args()

scenes = [l.strip() for l in open(a.list) if l.strip()][:a.scenes]
ds = BevLaneDataset(a.root, scenes, gt_key="gt_cons", max_per_scene=200,
                    n_cams=8, trim_start=3, trim_end=10)
m = MODELS["v52"](n_seg=21).cuda().eval()
sd = torch.load(a.ckpt, map_location="cpu"); sd = sd.get("model", sd)
sd = {k.replace("module.", ""): v for k, v in sd.items()}
cur = m.state_dict()
m.load_state_dict({k: v for k, v in sd.items()
                   if k in cur and cur[k].shape == v.shape}, strict=False)

# 行: x = 80 - 0.2*row。近傍 0-20m = row 300..400、前方 30m = row 250..400
NEAR = slice(300, 400)       # 自車前方 0-20 m
FWD30 = slice(250, 400)      # 前方 0-30 m
COLS = slice(175, 325)       # |y| <= 15 m
ROAD = (1, 3, 4, 5)          # road + crosswalk + laneline + stopline
rows = []
by_scene = {}
for i, (s, f) in enumerate(ds.items):
    by_scene.setdefault(s, []).append((int(f["frame"]), i))
for s, lst in list(by_scene.items())[:a.scenes]:
    lst.sort()
    for fi, di in lst[::3]:
        b = ds[di]
        if b is None: continue
        with torch.no_grad(), torch.autocast("cuda", torch.float16):
            out = m(b[0][None].cuda(), b[1][None].cuda(), b[2][None].cuda())
        pred = out[0].float().argmax(1)[0].cpu().numpy()
        gt = np.where(b[3].numpy() == 255, 0, b[3].numpy())
        p_road = int(np.isin(pred[NEAR, COLS], ROAD).sum())
        g_road = int(np.isin(gt[NEAR, COLS], ROAD).sum())
        p_cw = int((pred[FWD30, COLS] == 3).sum())
        g_cw = int((gt[FWD30, COLS] == 3).sum())
        rows.append((s, fi, p_road, g_road, p_cw, g_cw))
if not rows:
    print("データなし"); sys.exit()
arr = np.array([[r[2], r[3], r[4], r[5]] for r in rows], float)
# 交差点フレーム = GT crosswalk が上位 25%
thr = np.percentile(arr[:, 3], 75)
near_ix = arr[:, 3] >= max(thr, 50)
far_ix = arr[:, 3] < max(thr, 50) * 0.2
print(f"\n=== {a.tag} 交差点接近時の近傍 road 面積 ({len(rows)} フレーム) ===")
print(f"交差点フレーム {int(near_ix.sum())} / 非交差点 {int(far_ix.sum())}")
for nm, msk in (("交差点付近", near_ix), ("直線路", far_ix)):
    if msk.sum() < 3: continue
    p, g = arr[msk, 0].mean(), arr[msk, 1].mean()
    print(f"  {nm:8s} pred road {p:8.0f} px | GT road {g:8.0f} px | "
          f"pred/GT {p/max(g,1):.3f}")
if near_ix.sum() >= 3 and far_ix.sum() >= 3:
    pr = arr[near_ix, 0].mean() / max(arr[far_ix, 0].mean(), 1)
    gr = arr[near_ix, 1].mean() / max(arr[far_ix, 1].mean(), 1)
    print(f"\n  交差点/直線 の比: pred {pr:.3f} / GT {gr:.3f}")
    print("  -> pred だけ小さければモデル固有、GT も小さければ教師・遮蔽由来")
print("PROBE_IX_DONE")

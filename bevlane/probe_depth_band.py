"""深度の距離帯別誤差と分布の鋭さを測る (2026-08-21)。

LiDAR とカメラの遠方検出差 (recall 0.50 vs 0.89) の根本原因が
「深度の距離劣化」なら、遠方帯で誤差と分布のボケが急増するはず。
改善余地を定量化してから対策 (遠方重み付け / ビン再設計) を決める。
"""
import argparse, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bevlane.dataset import BevLaneDataset
from bevlane.model import MODELS

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--list", default="val.lst")
ap.add_argument("--root", default="out/bevlane")
ap.add_argument("--frames", type=int, default=60)
ap.add_argument("--tag", default="")
a = ap.parse_args()

scenes = [l.strip() for l in open(a.list) if l.strip()][:40]
ds = BevLaneDataset(a.root, scenes, gt_key="gt_cons", with_depth=True,
                    max_per_scene=4, n_cams=8, trim_start=3, trim_end=10)
m = MODELS["v52"](n_seg=21).cuda().eval()
sd = torch.load(a.ckpt, map_location="cpu"); sd = sd.get("model", sd)
sd = {k.replace("module.", ""): v for k, v in sd.items()}
cur = m.state_dict()
m.load_state_dict({k: v for k, v in sd.items()
                   if k in cur and cur[k].shape == v.shape}, strict=False)
D_MIN, D_STEP, D = m.D_MIN, m.D_STEP, m.D
BANDS = [(0, 15), (15, 30), (30, 45), (45, 60), (60, 80)]
err = {b: [] for b in BANDS}
ent = {b: [] for b in BANDS}
top1 = {b: [] for b in BANDS}
step = max(1, len(ds) // a.frames); done = 0
for i in range(0, len(ds), step):
    b = ds[i]
    if b is None: continue
    with torch.no_grad(), torch.autocast("cuda", torch.float16):
        out = m(b[0][None].cuda(), b[1][None].cuda(), b[2][None].cuda())
    dlog = out[1].float()[0]                       # [N,D,h,w]
    dgt = b[3 if not torch.is_tensor(b[3]) else 4]
    dgt = b[4] if torch.is_tensor(b[4]) and b[4].dim() == 3 else None
    if dgt is None: continue
    dgt = dgt.cuda()
    n = min(dlog.shape[0], dgt.shape[0])
    p = dlog[:n].softmax(1)
    bins = (torch.arange(D, device=p.device) * D_STEP + D_MIN).view(1, -1, 1, 1)
    exp_d = (p * bins).sum(1)                      # [n,h,w] 期待距離
    e = -(p.clamp_min(1e-6) * p.clamp_min(1e-6).log()).sum(1)   # エントロピー
    t1 = p.max(1).values
    g = dgt[:n]
    if g.shape[-2:] != exp_d.shape[-2:]:
        g = torch.nn.functional.interpolate(g[None].float(), size=exp_d.shape[-2:],
                                            mode="nearest")[0]
    valid = g > 0.5
    for lo, hi in BANDS:
        m_ = valid & (g >= lo) & (g < hi)
        if m_.sum() < 10: continue
        err[(lo, hi)].append(float((exp_d[m_] - g[m_]).abs().mean()))
        ent[(lo, hi)].append(float(e[m_].mean()))
        top1[(lo, hi)].append(float(t1[m_].mean()))
    done += 1
    if done >= a.frames: break
print(f"\n=== {a.tag} 深度の距離帯別 ({done} 枚, {D} ビン x {D_STEP} m) ===")
print("帯        平均絶対誤差   相対誤差   分布エントロピー  最大確率")
for lo, hi in BANDS:
    if not err[(lo, hi)]: continue
    ae = np.mean(err[(lo, hi)])
    mid = (lo + hi) / 2
    print(f"  {lo:2d}-{hi:2d}m   {ae:6.2f} m    {100*ae/mid:5.1f}%    "
          f"{np.mean(ent[(lo,hi)]):6.3f}        {np.mean(top1[(lo,hi)]):.3f}")
print("PROBE_DEPTH_BAND_DONE")

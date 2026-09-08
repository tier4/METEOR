"""後方 20-40m の GT 箱に対し「最近傍の予測」までの距離と方向を測る。

深度バイアス説の検証: 検出自体はされているが縦方向に系統的にずれている
なら、リコール低下 (2m マッチで外れる) と、マッチ集合の誤差が 0.7m で
頭打ち (生存者バイアス) の両方が一度に説明される。
"""
import argparse, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bevlane.dataset import BevLaneDataset
from bevlane.model import MODELS

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--model", default="v52")
ap.add_argument("--frames", type=int, default=150)
a = ap.parse_args()
scenes = [l.strip() for l in open("val.lst") if l.strip()][:60]
ds = BevLaneDataset("out/bevlane", scenes, gt_key="gt_cons", with_boxdet=True,
                    n_cams=8, trim_start=25, trim_end=10)
m = MODELS[a.model](n_seg=21).cuda().eval()
sd = torch.load(a.ckpt, map_location="cpu")
sd = {k.replace("module.", ""): v for k, v in sd.get("model", sd).items()}
cur = m.state_dict()
m.load_state_dict({k: v for k, v in sd.items() if k in cur and cur[k].shape == v.shape}, strict=False)

rows = {"前": [], "後": []}
step = max(1, len(ds)//a.frames); done = 0
for i in range(0, len(ds), step):
    b = ds[i]
    if b is None: continue
    with torch.no_grad(), torch.autocast("cuda", torch.float16):
        out = m(b[0][None].cuda(), b[1][None].cuda(), b[2][None].cuda())
    # 弱い検出も含める (変位か不在かを見分けるため threshold を 0.10 まで下げる)
    dets = m.decode_boxes(out[3].float().cpu(), out[4].float().cpu(), thresh=0.10)[0]
    pred = [(float(d[2]), float(d[3])) for d in dets if int(d[0]) == 0]
    bx, nb = b[4], int(b[5])
    for k in range(max(nb, 0)):
        cls, xe, ye, ln, wd, yw = [float(v) for v in bx[k][:6]]
        if ln <= 0 or cls >= 1.5 or abs(ye) > 45: continue
        r = (xe*xe + ye*ye) ** 0.5
        if not (20 <= r < 40): continue
        best = None
        for dx, dy in pred:
            d2 = (xe-dx)**2 + (ye-dy)**2
            if best is None or d2 < best[0]: best = (d2, dx, dy)
        side = "前" if xe > 0 else "後"
        if best is None:
            rows[side].append((99.0, 0.0)); continue
        dist = best[0] ** 0.5
        # 半径方向 (奥行き) の符号付きずれ: + = 予測が GT より遠い
        ux, uy = xe / max(r, 1e-6), ye / max(r, 1e-6)
        dr = (best[1]-xe)*ux + (best[2]-ye)*uy
        rows[side].append((dist, dr))
    done += 1
    if done >= a.frames: break

print(f"=== 20-40m 帯: GT 箱から最近傍予測まで (しきい値 0.10, {done} フレーム) ===")
for side in ("前", "後"):
    A = np.array(rows[side])
    d, dr = A[:,0], A[:,1]
    print(f"\n{side}方 (GT {len(A)} 箱)")
    for lo, hi, nm in ((0,1,"0-1m"),(1,2,"1-2m"),(2,3,"2-3m"),(3,5,"3-5m"),(5,98,"5m超"),(98,100,"予測なし")):
        s = (d>=lo)&(d<hi)
        print(f"  最近傍 {nm:<6}: {100*s.mean():5.1f}%")
    near = d < 5
    if near.sum() >= 5:
        print(f"  5m 以内の箱の奥行きずれ: 中央値 {np.median(dr[near]):+.2f}m "
              f"(+=遠く予測) / 遠寄り {100*(dr[near]>1).mean():.0f}% 近寄り {100*(dr[near]<-1).mean():.0f}%")

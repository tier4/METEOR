"""road/crosswalk の logit オフセット較正フィット (2026-08-27)。

ユーザー要望「BEV Seg の road / crosswalk の precision を高めたい」。
細線クラスで実証済みの決定境界較正 (calibrate_seg_bias.py, r61 で
IoU +1.5% と precision 同時改善) を面クラスへ拡張するためのフィット。
val フレームで (road, crosswalk) オフセット格子の P/R/IoU を測る。

使い方: python3 bevlane/fit_seg_bias_area.py --ckpt out/v128_best_e2e.pt
"""
import argparse
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, ".")
from bevlane.dataset import BevLaneDataset          # noqa: E402
from bevlane.model import DepthSegIPMNetV52         # noqa: E402
from bevlane.ckpt_load import load_net              # noqa: E402
from bevlane.train import _temporal_inputs          # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="out/v128_best_e2e.pt")
ap.add_argument("--root", default="out/bevlane")
ap.add_argument("--scenes", type=int, default=24)
ap.add_argument("--max-per-scene", type=int, default=5)
ap.add_argument("--skip", type=int, default=0,
                help="検証用に別シーン帯を使う (fit/verify 分離)")
a = ap.parse_args()

dev = "cuda"
sc = [l.strip() for l in open("val.lst") if l.strip()][a.skip:a.skip + a.scenes]
ds = BevLaneDataset(a.root, sc, gt_key="gt_cons", with_temporal=True,
                    max_per_scene=a.max_per_scene)
dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)
net = DepthSegIPMNetV52(n_seg=21).to(dev).eval()
load_net(net, a.ckpt)

ROAD, XW, LANE, STOP, EDGE = 1, 3, 4, 5, 6
# (road_off, xw_off, lane_off)
GRID = [(0.0, 0.0, 0.0, 0.0, 0.0), (0.5, 1.5, 1.0, 0.6, 0.5)]
# combo -> class -> [tp, fp, fn]
acc = {g: {c: [0, 0, 0] for c in (ROAD, XW, LANE, STOP, EDGE)} for g in GRID}
nfr = 0
with torch.no_grad():
    for batch in dl:
        imgs = batch[0].to(dev)
        K = batch[1].to(dev)
        Tc = batch[2].to(dev)
        gt = None
        for t in batch[3:]:
            if torch.is_tensor(t) and t.dim() == 3 \
                    and tuple(t.shape[1:]) == (800, 500) \
                    and t.dtype == torch.int64:
                gt = t.to(dev)[0]
                break
        if gt is None:
            continue
        pb, th = _temporal_inputs(net, batch, dev, None)
        with torch.autocast("cuda", torch.float16):
            out = net(imgs, K, Tc, None, pb, th)
        lg = out[0][0].float()                    # [9,800,500]
        valid = gt != 255
        nfr += 1
        for (r_off, x_off, l_off, s_off, e_off) in GRID:
            lg2 = lg.clone()
            lg2[ROAD] -= r_off
            lg2[XW] -= x_off
            lg2[LANE] -= l_off
            lg2[STOP] -= s_off
            lg2[EDGE] -= e_off
            pred = lg2.argmax(0)
            for c in (ROAD, XW, LANE, STOP, EDGE):
                p = (pred == c) & valid
                t = gt == c
                tp = int((p & t).sum())
                acc[(r_off, x_off, l_off, s_off, e_off)][c][0] += tp
                acc[(r_off, x_off, l_off, s_off, e_off)][c][1] += int(p.sum()) - tp
                acc[(r_off, x_off, l_off, s_off, e_off)][c][2] += int(t.sum()) - tp

print(f"frames={nfr} scenes={len(sc)} ckpt={a.ckpt}")
print(f"{'road':>5} {'xw':>5} {'lane':>5} | road P/R/IoU | xwalk P/R/IoU | lane P/R/IoU/面積比")
for g in GRID:
    line = f"{g[0]:4.1f} {g[1]:4.1f} {g[2]:4.1f} {g[3]:4.1f} {g[4]:4.1f} |"
    for c in (ROAD, XW, LANE, STOP, EDGE):
        tp, fp, fn = acc[g][c]
        P = tp / max(tp + fp, 1)
        R = tp / max(tp + fn, 1)
        iou = tp / max(tp + fp + fn, 1)
        line += f" {P:.3f}/{R:.3f}/{iou:.3f}"
        if c in (LANE, STOP, EDGE):
            line += f"/{(tp + fp) / max(tp + fn, 1):.2f}"
        line += " |"
    print(line)
print("FIT_SEG_BIAS_DONE")

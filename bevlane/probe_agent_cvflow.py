#!/usr/bin/env python3
"""CV-from-flow プローブ (2026-08-27): 学習なしで「CV+残差」レバーの上限を測る。

probe_agent_cv.py の続き。モデル (v125) のエージェント軌跡 2.14m は
oracle-CV 0.8m に大差で負けている。では flow ヘッドの推定速度から作る
「現実の CV」はどこまで出るか? これが良ければ、traj ヘッドを
CV(flow)+残差 に再パラメータ化するラウンド (v128 候補) の期待値が立つ。
"""
import argparse
import math
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, ".")
from bevlane.dataset import BevLaneDataset                      # noqa: E402
from bevlane.model import DepthSegIPMNetV52                     # noqa: E402
from bevlane.ckpt_load import load_net                          # noqa: E402
from bevlane.train import _temporal_inputs                      # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="out/v125_final_best_e2e.pt")
ap.add_argument("--root", default="out/bevlane")
ap.add_argument("--val-list", default="val.lst")
ap.add_argument("--scenes", type=int, default=24)
ap.add_argument("--max-per-scene", type=int, default=5)
a = ap.parse_args()

dev = "cuda"
sc = [l.strip() for l in open(a.val_list) if l.strip()][:a.scenes]
ds = BevLaneDataset(a.root, sc, gt_key="gt_cons",
                    with_boxdet=True, with_agenttraj=True,
                    with_temporal=True, max_per_scene=a.max_per_scene)
dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)
net = DepthSegIPMNetV52(n_seg=21).to(dev).eval()
load_net(net, a.ckpt)

S = {}


def add(k, v):
    S.setdefault(k, []).append(float(v))


steps = torch.arange(1, 7, device=dev).view(6, 1).float()
n_ag = 0
with torch.no_grad():
    for batch in dl:
        imgs = batch[0].to(dev)
        K = batch[1].to(dev)
        Tc = batch[2].to(dev)
        # agenttraj 4 連 (boxes[64,6], count, traj[64,6,2], tvalid[64,6])
        tj = bx = None
        tensors = [t for t in batch if torch.is_tensor(t)]
        for j in range(len(tensors) - 3):
            t2 = tensors[j + 2]
            if t2.dim() == 4 and tuple(t2.shape[2:]) == (6, 2) and \
                    tensors[j].dim() == 3 and tensors[j].shape[1] == t2.shape[1]:
                bx, nbT, tj, tv = (tensors[j][0], tensors[j + 1],
                                   tensors[j + 2][0], tensors[j + 3][0])
                break
        if tj is None:
            continue
        pb, th = _temporal_inputs(net, batch, dev, None)
        with torch.autocast("cuda", torch.float16):
            out = net(imgs, K, Tc, None, pb, th)
        tp = out[9][0].float()                       # [39,H,W] det 格子
        # eager 出力順: ...traj9 stationary10 tl11 risk12 flow13。
        # 「C==2 の最初のテンソル」だと hm (2ch ヒートマップ) を掴む誤り。
        flow = out[13]
        assert torch.is_tensor(flow) and flow.dim() == 4 \
            and flow.shape[1] == 2, f"flow 位置ズレ: {flow.shape}"
        flow = flow[0].float()                       # [2,FH,FW] ±40m 窓
        Hh, Ww = tp.shape[-2:]
        FH, FW = flow.shape[-2:]
        for k in range(max(0, int(nbT))):
            if float(bx[k, 3]) <= 0:
                continue
            v = tv[k]
            if v.sum() == 0:
                continue
            x, y = float(bx[k, 1]), float(bx[k, 2])
            ri = int((80.0 - x) / 0.4)
            ci = int((50.0 - y) / 0.4)
            if not (0 <= ri < Hh and 0 <= ci < Ww):
                continue
            g = tj[k].to(dev)
            vv = v.to(dev)
            cls = "veh" if float(bx[k, 0]) < 1.5 else "vru"
            # --- モデル予測 (勝者モード)
            vec = tp[:, ri, ci]
            kb = int(vec[36:39].argmax())
            p = vec[kb * 12:(kb + 1) * 12].view(6, 2)
            add(f"model_{cls}", ((p - g).norm(dim=1) * vv).sum() / vv.sum())
            # --- CV(flow): flow 窓 ±40m
            rf = int((40.0 - x) / (80.0 / FH))
            cf = int((40.0 - y) / (80.0 / FW))
            if 0 <= rf < FH and 0 <= cf < FW:
                vel = flow[:, rf, cf]                # 0.5s あたり変位
                pcv = steps * vel.view(1, 2)
                add(f"cvflow_{cls}",
                    ((pcv - g).norm(dim=1) * vv).sum() / vv.sum())
                if vv[5] > 0 and float(g[5].norm()) > 1.0:
                    if float(pcv[5].norm()) > 0.3:
                        ga = math.atan2(float(g[5, 1]), float(g[5, 0]))
                        pa = math.atan2(float(pcv[5, 1]), float(pcv[5, 0]))
                        add(f"cvflow_head_{cls}", math.degrees(
                            abs((pa - ga + math.pi) % (2 * math.pi) - math.pi)))
                    # 速度推定そのものの誤差 (0.5s 変位)
                    add(f"flowerr_{cls}", float((vel - g[0].to(dev)).norm()))
            n_ag += 1

print(f"agents={n_ag} scenes={len(sc)} ckpt={a.ckpt}")
for k in sorted(S):
    vv = np.array(S[k])
    print(f"  {k:<18} mean={vv.mean():.3f}  n={len(vv)}")

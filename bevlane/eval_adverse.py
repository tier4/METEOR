#!/usr/bin/env python3
"""悪条件 (Cosmos transfer_2) での汎化評価。

問い: Cosmos データを学習に混ぜると、悪条件への頑健性は上がるのか。
通常条件の val では v116a (実データのみ) 対 v116c (+Cosmos) は score が
0.351 対 0.348 とむしろ僅差で負けており、Cosmos の本来の狙い (悪条件の
頑健化) は晴天中心の val では測れない。ここでは**完全未見の 2 ベース
シーン × 6 悪条件** (transfer_2、変換画像は全モデル未学習) で測る。

指標: BEV Seg mIoU (走行面/レーン系) と 車両 recall (3m マッチ)。
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bevlane.dataset import BevLaneDataset          # noqa: E402
from bevlane.model import MODELS                    # noqa: E402
from bevlane.ckpt_load import load_net              # noqa: E402

ROAD = (1, 3, 4, 5)
LANE = (2, 6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    net = MODELS["v52"](n_seg=21).cuda().eval()
    load_net(net, a.ckpt, verbose=False)
    ds = BevLaneDataset(a.root, a.scenes, gt_key="gt", with_boxdet=True,
                        n_cams=8, trim_start=3, trim_end=5)
    inter = {k: 0 for k in ("road", "lane")}
    union = {k: 0 for k in ("road", "lane")}
    gt_n = hit = 0
    n = 0
    for i in range(0, len(ds), a.stride):
        x = ds[i]
        if x is None:
            continue
        with torch.no_grad(), torch.autocast("cuda", torch.float16):
            out = net(x[0][None].cuda(), x[1][None].cuda(), x[2][None].cuda())
        p = out[0].float().argmax(1)[0].cpu().numpy()
        g = x[3].numpy()
        valid = g != 255
        for nm, ks in (("road", ROAD), ("lane", LANE)):
            pm = np.isin(p, ks) & valid
            gm = np.isin(g, ks) & valid
            inter[nm] += int((pm & gm).sum())
            union[nm] += int((pm | gm).sum())
        # 車両 recall (bev_box GT, 3m)
        bx, nb = x[4], int(x[5])
        dets = net.decode_boxes(out[3].float().cpu(), out[4].float().cpu(),
                                thresh=0.25)[0]
        pv = [(float(d[2]), float(d[3])) for d in dets if int(d[0]) == 0]
        for k in range(nb):
            cls, xe, ye, ln = [float(v) for v in bx[k][:4]]
            if ln <= 0 or cls >= 1.5:
                continue
            gt_n += 1
            hit += int(any((xe - px) ** 2 + (ye - py) ** 2 < 9.0
                           for px, py in pv))
        n += 1
    r = {nm: inter[nm] / max(union[nm], 1) for nm in inter}
    print(f"{a.tag}\t{n}\t{r['road']:.4f}\t{r['lane']:.4f}\t"
          f"{hit / max(gt_n, 1):.4f}\t{gt_n}")


if __name__ == "__main__":
    main()

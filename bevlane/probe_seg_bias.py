"""BEV Seg の細クラス (laneline/stopline/road_edge) のロジット・バイアスを掃引する。

r61 で「細クラスのロジットから定数を引くと線幅が正しくなり IoU も上がる」
ことが実測されている (calibrate_seg_bias.py の記録: mIoU +1.5%, laneline
+10%, 線幅比 3.05->1.07)。その較正が現行ラウンドにも効くのか、効くなら
どの値が最適かを、適合用と検証用でシーンを分けて確かめる。

1 フレームにつきロジットを 1 回だけ計算し、候補バイアスごとに argmax を
取り直すので、掃引を増やしても推論コストは増えない。
"""
import argparse
import itertools
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bevlane.dataset import BevLaneDataset                        # noqa: E402
from bevlane.model import MODELS                                  # noqa: E402

NAMES = ["背景", "road", "sidewalk", "crosswalk", "laneline",
         "stopline", "road_edge", "marking", "parking"]
THIN = [4, 5, 6]                       # laneline / stopline / road_edge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", default="v52")
    ap.add_argument("--n-cams", type=int, default=8)
    ap.add_argument("--list", default="val.lst")
    ap.add_argument("--root", default="out/bevlane")
    ap.add_argument("--scenes", type=int, default=60)
    ap.add_argument("--frames", type=int, default=160)
    ap.add_argument("--grid", default="0,0.5,0.75,1.0,1.25")
    a = ap.parse_args()

    cand = [float(x) for x in a.grid.split(",")]
    combos = [c for c in itertools.product(cand, repeat=3)]
    scenes = [l.strip() for l in open(a.list) if l.strip()][:a.scenes]
    ds = BevLaneDataset(a.root, scenes, gt_key="gt_cons", max_per_scene=4,
                        n_cams=8, trim_start=3, trim_end=10)
    m = MODELS[a.model](n_seg=21).cuda().eval()
    sd = torch.load(a.ckpt, map_location="cpu")
    sd = {k.replace("module.", ""): v for k, v in sd.get("model", sd).items()}
    cur = m.state_dict()
    m.load_state_dict({k: v for k, v in sd.items()
                       if k in cur and cur[k].shape == v.shape}, strict=False)

    C = 9
    # 適合用 / 検証用でシーンを分ける (較正か過学習かを見分けるため)
    conf = {"fit": {c: torch.zeros(C, C, dtype=torch.long, device="cuda")
                    for c in combos},
            "ver": {c: torch.zeros(C, C, dtype=torch.long, device="cuda")
                    for c in combos}}
    area = {"fit": {c: [0, 0] for c in combos},
            "ver": {c: [0, 0] for c in combos}}
    step = max(1, len(ds) // a.frames)
    done = 0
    for i in range(0, len(ds), step):
        b = ds[i]
        if b is None:
            continue
        split = "fit" if (done % 2 == 0) else "ver"
        with torch.no_grad(), torch.autocast("cuda", torch.float16):
            out = m(b[0][None][:, :a.n_cams].cuda(),
                    b[1][None][:, :a.n_cams].cuda(),
                    b[2][None][:, :a.n_cams].cuda())
        lg = out[0][0].float()                        # [C,H,W]
        gt = b[3].cuda() if torch.is_tensor(b[3]) else torch.as_tensor(b[3]).cuda()
        gt = gt.long()
        ok = gt < C
        gtv = gt[ok]
        for cb in combos:
            adj = lg.clone()
            for ci, off in zip(THIN, cb):
                adj[ci] -= off
            pr = adj.argmax(0)[ok]
            idx = gtv * C + pr
            conf[split][cb] += torch.bincount(idx, minlength=C * C).view(C, C)
            area[split][cb][0] += int((pr == 4).sum())
            area[split][cb][1] += int((gtv == 4).sum())
        done += 1
        if done >= a.frames:
            break

    def miou(cm):
        cm = cm.double()
        inter = cm.diag()
        union = cm.sum(1) + cm.sum(0) - inter
        iou = torch.where(union > 0, inter / union.clamp(min=1),
                          torch.full_like(inter, float("nan")))
        return iou

    rows = []
    for cb in combos:
        f, v = miou(conf["fit"][cb]), miou(conf["ver"][cb])
        rows.append((float(np.nanmean(v.cpu().numpy())), cb,
                     float(np.nanmean(f.cpu().numpy())),
                     float(v[4]), float(v[5]), float(v[6]),
                     area["ver"][cb][0] / max(area["ver"][cb][1], 1)))
    rows.sort(reverse=True)
    print(f"=== {a.ckpt} ({done} フレーム, 適合/検証で交互分割) ===")
    print("バイアス(lane/stop/edge)  検証mIoU  適合mIoU  laneline  stopline  "
          "road_edge  lane面積比")
    base = [r for r in rows if r[1] == (0.0, 0.0, 0.0)]
    for sc, cb, fmi, i4, i5, i6, ar in rows[:8]:
        mark = " <- 現状" if cb == (0.0, 0.0, 0.0) else ""
        print(f"  {cb[0]:.2f}/{cb[1]:.2f}/{cb[2]:.2f}        {sc:.4f}   "
              f"{fmi:.4f}   {i4:.4f}    {i5:.4f}    {i6:.4f}     "
              f"{ar:5.2f}{mark}")
    if base and rows[0][1] != (0.0, 0.0, 0.0):
        b0 = base[0]
        print(f"\n  現状 (0/0/0): 検証mIoU {b0[0]:.4f} laneline {b0[3]:.4f} "
              f"面積比 {b0[6]:.2f}")
        print(f"  最良       : 検証mIoU {rows[0][0]:.4f} "
              f"({100*(rows[0][0]-b0[0])/max(b0[0],1e-9):+.1f}%) "
              f"laneline {rows[0][3]:.4f} "
              f"({100*(rows[0][3]-b0[3])/max(b0[3],1e-9):+.1f}%)")


if __name__ == "__main__":
    main()

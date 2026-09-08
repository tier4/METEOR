"""laneline の失点がどこから来ているかを分解する。

線は幅 1-2 セル (0.2 m/セル) しかないので、1 セルずれるだけで IoU は半減する。
「形は合っているが位置が微妙にずれている」のか「そもそも出ていない/余計に
出ている」のかで、打つ手がまったく変わる:

  許容 0 セル IoU が低く、許容 1-2 セルにすると跳ね上がる
      -> サブセル精度の問題。SDF/オフセット回帰や高解像度ヘッドが効く
  許容を広げても上がらない
      -> 検出そのものの問題。特徴量・損失・GT 品質を見る必要がある

あわせて再現率と適合率を許容距離別に出し、太さ (面積比) も測る。
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bevlane.dataset import BevLaneDataset                        # noqa: E402
from bevlane.model import MODELS                                  # noqa: E402
from bevlane.ckpt_load import load_net                            # noqa: E402

LANE = 4                                # laneline のクラス番号
TOL = [0, 1, 2, 3]                      # 許容セル数 (0.2 m/セル)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", default="v52")
    ap.add_argument("--n-cams", type=int, default=8)
    ap.add_argument("--list", default="val.lst")
    ap.add_argument("--root", default="out/bevlane")
    ap.add_argument("--scenes", type=int, default=60)
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--bias", type=float, default=0.75,
                    help="laneline ロジットから引く定数 (較正)")
    ap.add_argument("--cls", type=int, default=LANE)
    ap.add_argument("--trim-start", type=int, default=3)
    ap.add_argument("--trim-end", type=int, default=10)
    a = ap.parse_args()

    scenes = [l.strip() for l in open(a.list) if l.strip()][:a.scenes]
    ds = BevLaneDataset(a.root, scenes, gt_key="gt_cons",
                        n_cams=8, trim_start=a.trim_start,
                        trim_end=a.trim_end)
    m = MODELS[a.model](n_seg=21).cuda().eval()
    load_net(m, a.ckpt)

    tp = {t: 0 for t in TOL}            # 予測のうち GT の許容内にあるもの
    hit = {t: 0 for t in TOL}           # GT のうち予測の許容内にあるもの
    n_pred = n_gt = 0
    row_gt = row_hit = row_gt_far = row_hit_far = 0
    gap_runs = []
    step = max(1, len(ds) // a.frames)
    done = 0
    for i in range(0, len(ds), step):
        b = ds[i]
        if b is None:
            continue
        with torch.no_grad(), torch.autocast("cuda", torch.float16):
            out = m(b[0][None][:, :a.n_cams].cuda(),
                    b[1][None][:, :a.n_cams].cuda(),
                    b[2][None][:, :a.n_cams].cuda())
        lg = out[0][0].float()
        lg[a.cls] -= a.bias
        pr = (lg.argmax(0) == a.cls).cpu().numpy().astype(np.uint8)
        raw_gt = b[3].numpy()
        valid = raw_gt != 255
        pr &= valid.astype(np.uint8)   # predictions in don't-care are not FP
        gt = (raw_gt == a.cls).astype(np.uint8)
        if gt.sum() == 0 and pr.sum() == 0:
            continue
        n_pred += int(pr.sum())
        n_gt += int(gt.sum())
        for t in TOL:
            if t == 0:
                gd, pd = gt, pr
            else:
                k = np.ones((2 * t + 1, 2 * t + 1), np.uint8)
                gd = cv2.dilate(gt, k)
                pd = cv2.dilate(pr, k)
            tp[t] += int((pr & gd).sum())      # 予測が GT の許容内
            hit[t] += int((gt & pd).sum())     # GT が予測の許容内
            if t == 1:
                gr = gt.any(1)
                hr = (gt & pd).any(1)
                rows = np.arange(len(gr))
                far = (80.0 - (rows + 0.5) * 0.2) > 30.0
                row_gt += int(gr.sum()); row_hit += int((gr & hr).sum())
                row_gt_far += int((gr & far).sum())
                row_hit_far += int((gr & hr & far).sum())
                miss = gr & ~hr
                edge = np.diff(np.pad(miss.astype(np.int8), (1, 1)))
                starts = np.flatnonzero(edge == 1)
                stops = np.flatnonzero(edge == -1)
                gap_runs.extend((stops - starts).tolist())
        done += 1
        if done >= a.frames:
            break

    print(f"=== {a.ckpt} (laneline, {done} フレーム, バイアス {a.bias}) ===")
    print(f"予測セル数 {n_pred}  GT セル数 {n_gt}  "
          f"面積比 {n_pred / max(n_gt, 1):.2f}")
    print("許容    適合率(予測がGT近傍)  再現率(GTが予測近傍)  F1")
    for t in TOL:
        p = tp[t] / max(n_pred, 1)
        r = hit[t] / max(n_gt, 1)
        f = 2 * p * r / max(p + r, 1e-9)
        print(f"  {t}セル ({t * 0.2:.1f}m)      {p:5.3f}            "
              f"{r:5.3f}          {f:5.3f}")
    gaps = np.asarray(gap_runs, np.float32) * 0.2
    print(f"行連続性@0.2m: recall={row_hit / max(row_gt, 1):.3f} "
          f"far(+30m)={row_hit_far / max(row_gt_far, 1):.3f} "
          f"欠落run mean/p95/max="
          f"{(gaps.mean() if len(gaps) else 0):.2f}/"
          f"{(np.percentile(gaps, 95) if len(gaps) else 0):.2f}/"
          f"{(gaps.max() if len(gaps) else 0):.2f}m")


if __name__ == "__main__":
    main()

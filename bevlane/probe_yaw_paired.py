"""1 モデル分の yaw 誤差を GT 箱ごとに書き出す (ペア比較用の片側)。

なぜ 1 プロセス 1 モデルなのか:
  BEV の後方レンジ (METEOR_BEV_XR) はインポート時に読まれる定数なので、
  1 プロセスで軽量版 (rear-40) と全域版を同時に正しく構成できない。
  2026-08-14 の比較はこれを守らず、軽量版を誤ったジオメトリで動かして
  検出数が 90 個まで落ちた状態の数値を比べていた。

なぜペアにするのか:
  yaw 誤差はリコールに強く汚染される。検出が易しい箱 (正対・近距離) しか
  出せないモデルは平均 yaw 誤差が小さく出る。フェーズ A 2.7 度 (121 箱) と
  フェーズ B 7.4 度 (832 箱) はこの汚染そのもの。両モデルが検出できた
  同一 GT 箱だけを突き合わせて初めて姿勢精度の比較になる。

出力 npz は (フレーム index, 箱 index) をキーに yaw 誤差を持つ。
突き合わせは probe_yaw_join.py が行う。
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bevlane.dataset import BevLaneDataset                        # noqa: E402
from bevlane.model import MODELS                                  # noqa: E402


def load_model(ckpt, name):
    m = MODELS[name](n_seg=21).cuda().eval()
    sd = torch.load(ckpt, map_location="cpu")
    sd = sd.get("model", sd)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    cur = m.state_dict()
    keep = {k: v for k, v in sd.items()
            if k in cur and cur[k].shape == v.shape}
    m.load_state_dict(keep, strict=False)
    print(f"[load] {ckpt}: {len(keep)}/{len(cur)} テンソルを復元", flush=True)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-cams", type=int, default=8)
    ap.add_argument("--list", default="val.lst")
    ap.add_argument("--root", default="out/bevlane")
    ap.add_argument("--scenes", type=int, default=60)
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--thresh", type=float, default=0.25)
    ap.add_argument("--match-r", type=float, default=2.0,
                    help="GT と予測の対応付け半径 [m]")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    scenes = [l.strip() for l in open(a.list) if l.strip()][:a.scenes]
    ds = BevLaneDataset(a.root, scenes, gt_key="gt_cons", with_boxdet=True,
                        max_per_scene=4, n_cams=8, trim_start=3, trim_end=10)
    m = load_model(a.ckpt, a.model)

    rows = []
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
        dets = m.decode_boxes(out[3].float().cpu(), out[4].float().cpu(),
                              thresh=a.thresh)[0]
        pred = [(float(d[2]), float(d[3]), float(d[6]))
                for d in dets if int(d[0]) == 0]
        bx, nb = b[4], int(b[5])
        for k in range(max(nb, 0)):
            cls, xe, ye, ln, wd, yaw = [float(v) for v in bx[k][:6]]
            if ln <= 0 or cls >= 1.5:
                continue
            r = (xe * xe + ye * ye) ** 0.5
            if not (0 < xe <= 50 or (-20 <= xe <= 0 and r <= 50)):
                continue
            best = None
            for dx, dy, dyaw in pred:
                d2 = (xe - dx) ** 2 + (ye - dy) ** 2
                if d2 < a.match_r ** 2 and (best is None or d2 < best[0]):
                    best = (d2, dyaw)
            if best is None:
                continue
            de = abs((best[1] - yaw + np.pi) % (2 * np.pi) - np.pi)
            rows.append((i, k, xe, ye, r,
                         min(de, np.pi - de),                 # 180 度対称
                         abs((np.degrees(yaw) + 90) % 180 - 90)))
        done += 1
        if done >= a.frames:
            break

    arr = np.array(rows, dtype=np.float64) if rows else np.zeros((0, 7))
    np.savez(a.out, rows=arr)
    print(f"[out] {a.out}: {len(arr)} 箱を検出 ({done} フレーム)", flush=True)


if __name__ == "__main__":
    main()

"""E2E 軌道の「右寄り」が定数バイアスかどうかを切り分ける。

問い: 予測軌道が右レーン端を沿うように見えるのはバグ (系統オフセット) か、
場面依存の誤差か。3 つを分けて測る:
  (1) GT 自体の横位置分布 — 教師が右に寄っていれば学習データ起因
  (2) 予測 - GT の平均と分散 — 平均が非ゼロで分散が小さければ定数バイアス
  (3) 横方向レンジ別 — 直進場面に限定しても残るか

y の符号: ego 座標系で +y = 左、-y = 右。
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bevlane.dataset import BevLaneDataset                        # noqa: E402
from bevlane.model import MODELS, EGO_K                           # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", default="v52")
    ap.add_argument("--n-cams", type=int, default=8)
    ap.add_argument("--list", default="val.lst")
    ap.add_argument("--root", default="out/bevlane")
    ap.add_argument("--scenes", type=int, default=80)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    scenes = [l.strip() for l in open(a.list) if l.strip()][:a.scenes]
    ds = BevLaneDataset(a.root, scenes, gt_key="gt_cons", with_ego=True,
                        max_per_scene=4, n_cams=8, trim_start=3, trim_end=10)
    m = MODELS[a.model](n_seg=21).cuda().eval()
    sd = torch.load(a.ckpt, map_location="cpu")
    sd = {k.replace("module.", ""): v for k, v in sd.get("model", sd).items()}
    cur = m.state_dict()
    m.load_state_dict({k: v for k, v in sd.items()
                       if k in cur and cur[k].shape == v.shape}, strict=False)

    HOR = [1, 3, 5]                     # 6 点中の添字 (0.5s 刻みなら 1/2/3 秒相当)
    gt_y, pr_y, df_y = [], [], []      # [フレーム][horizon]
    gt_last = []
    step = max(1, len(ds) // a.frames)
    done = 0
    for i in range(0, len(ds), step):
        b = ds[i]
        if b is None:
            continue
        eg = b[4] if len(b) > 4 and torch.is_tensor(b[4]) and b[4].numel() >= 12 else None
        if eg is None:
            continue
        g = eg[:12].view(6, 2)
        ims = b[0][None][:, :a.n_cams].cuda()
        Kk = b[1][None][:, :a.n_cams].cuda()
        Tc = b[2][None][:, :a.n_cams].cuda()
        with torch.no_grad(), torch.autocast("cuda", torch.float16):
            out = m(ims, Kk, Tc)
        e = out[7].float()[0]
        wp = e[:12 * EGO_K].view(EGO_K, 6, 2)
        k = int(e[12 * EGO_K:12 * EGO_K + EGO_K].argmax())
        gt_y.append([float(g[h, 1]) for h in HOR])
        pr_y.append([float(wp[k, h, 1]) for h in HOR])
        df_y.append([p - t for p, t in zip(pr_y[-1], gt_y[-1])])
        gt_last.append(float(g[-1, 1]))
        done += 1
        if done >= a.frames:
            break

    gt_y, pr_y, df_y = map(np.array, (gt_y, pr_y, df_y))
    gt_last = np.array(gt_last)
    straight = np.abs(gt_last) < 0.5

    print(f"\n=== {a.tag or a.ckpt} ({done} 枚, 直進 {straight.sum()} 枚) ===")
    print("(+y = 左 / -y = 右)")
    for j, h in enumerate(HOR):
        print(f"[点{h+1}/6]"
              f" GT y {gt_y[:, j].mean():+.3f}±{gt_y[:, j].std():.3f}"
              f" | 予測 y {pr_y[:, j].mean():+.3f}±{pr_y[:, j].std():.3f}"
              f" | 予測-GT {df_y[:, j].mean():+.3f}±{df_y[:, j].std():.3f}"
              f" | 右寄り率 {(df_y[:, j] < 0).mean() * 100:.0f}%")
    j = len(HOR) - 1
    s, c = df_y[straight, j], df_y[~straight, j]
    print(f"\n最終点の 予測-GT: 直進のみ {s.mean():+.3f}±{s.std():.3f} "
          f"(n={len(s)}) / 旋回含む {c.mean():+.3f}±{c.std():.3f} (n={len(c)})")
    print(f"GT 自体の最終点 y: 全体 {gt_last.mean():+.3f}±{gt_last.std():.3f} / "
          f"直進のみ {gt_last[straight].mean():+.3f}±{gt_last[straight].std():.3f}")
    print("PROBE_RIGHT_BIAS_DONE")


if __name__ == "__main__":
    main()

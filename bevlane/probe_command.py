"""運転コマンドが軌道を動かしているか、K=3 モードが分離しているかを測る。

参照実装 (VLA 版 eval_vla_command.py) と同じ問い方をする:
  直進する GT フレームに「左」「右」コマンドを強制入力し、3 秒後の横変位を見る。
  コマンドが効いていれば、その場面が直進を許していても横に押されるはず。
  参照実装の CNN 目標は 1 m 以上、モード結合前の実測は 0.2 m。

あわせて選択器の健全性も測る:
  モード間の距離 (3 本が似ていないか)、選択の的中率 (最良モードを選べているか)、
  選択ロス (選んだモードの誤差 - 最良モードの誤差)。
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
    ap.add_argument("--scenes", type=int, default=60)
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--straight", type=float, default=0.5,
                    help="直進とみなす 3 秒後横変位の上限 [m]")
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

    CMD = {"コマンド無し": None, "直進": 0, "左": 1, "右": 2}
    lat = {k: [] for k in CMD}
    mode_sel = {k: [] for k in CMD}
    spread, sel_gap, sel_hit = [], [], []
    step = max(1, len(ds) // a.frames)
    done = 0
    for i in range(0, len(ds), step):
        b = ds[i]
        if b is None:
            continue
        # ego GT: 先頭 12 要素が 6 点の (x,y)。最終点の横位置で直進判定
        eg = b[4] if len(b) > 4 and torch.is_tensor(b[4]) and b[4].numel() >= 12 else None
        if eg is None:
            continue
        gt_lat = float(eg[11])
        if abs(gt_lat) > a.straight:            # 直進フレームだけを使う
            continue
        ims = b[0][None][:, :a.n_cams].cuda()
        Kk = b[1][None][:, :a.n_cams].cuda()
        Tc = b[2][None][:, :a.n_cams].cuda()
        for name, idx in CMD.items():
            oh = None
            if idx is not None:
                oh = F.one_hot(torch.tensor([idx]), 3).float().cuda()
            with torch.no_grad(), torch.autocast("cuda", torch.float16):
                out = m(ims, Kk, Tc, intent=oh)
            e = out[7].float()[0]
            wp = e[:12 * EGO_K].view(EGO_K, 6, 2)
            lg = e[12 * EGO_K:12 * EGO_K + EGO_K]
            k = int(lg.argmax())
            lat[name].append(float(wp[k, -1, 1]))
            mode_sel[name].append(k)
            if idx is None:
                # モードの散らばり (最終点の横位置の最大差)
                ys = wp[:, -1, 1].cpu().numpy()
                spread.append(float(ys.max() - ys.min()))
                # 選択の良否: GT に対する各モードの ADE
                g = eg[:12].view(6, 2).cuda()
                ade = ((wp - g[None]) ** 2).sum(-1).sqrt().mean(1)
                best = int(ade.argmin())
                sel_hit.append(1.0 if best == k else 0.0)
                sel_gap.append(float(ade[k] - ade[best]))
        done += 1
        if done >= a.frames:
            break

    print(f"\n=== {a.tag or a.ckpt} (直進フレーム {done} 枚) ===")
    base = np.mean(lat["コマンド無し"]) if lat["コマンド無し"] else 0.0
    print("コマンド      3秒後の横変位(平均)   コマンド無しとの差   選択モード分布")
    for name in CMD:
        if not lat[name]:
            continue
        v = np.mean(lat[name])
        cnt = np.bincount(mode_sel[name], minlength=EGO_K)
        print(f"  {name:<10} {v:+6.2f} m            {v - base:+6.2f} m        "
              f"{list(cnt)}")
    if spread:
        print(f"\nモード間の散らばり (最終点横位置の最大差): "
              f"中央値 {np.median(spread):.2f} m / 平均 {np.mean(spread):.2f} m")
        print(f"選択の的中率: {100 * np.mean(sel_hit):.1f}% "
              f"(3択のあてずっぽうは 33.3%)")
        print(f"選択ロス (選んだモード - 最良モードの ADE): "
              f"{np.mean(sel_gap):.3f} m")
    dl = np.mean(lat["左"]) - base if lat["左"] else float("nan")
    dr = base - np.mean(lat["右"]) if lat["右"] else float("nan")
    print(f"\nコマンド応答量: 左 {dl:+.2f} m / 右 {dr:+.2f} m "
          f"(参照実装の目標は 1 m 以上、モード結合前の実測は 0.2 m)")


if __name__ == "__main__":
    main()

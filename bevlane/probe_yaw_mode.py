"""yaw が外れるとき、予測がどの向きに引き寄せられているかを調べる。

仮説の切り分け:
  A. 自車平行 (0 度) に張り付く  -> GT の姿勢分布の偏りに引きずられている
     (正対車が大多数なので、迷ったら 0 度と答えるのが損失的に得)
  B. 視線方向 (自車から見た方位) に張り付く -> リフトの深度ビンが粗く、
     足跡がレイ方向に潰れて姿勢の手がかりが消えている
どちらが優勢かで対策が変わる (A なら重み付け/損失、B なら深度分解能)。
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bevlane.dataset import BevLaneDataset                        # noqa: E402
from bevlane.model import MODELS                                  # noqa: E402


def fold(a):
    """180 度対称の軸誤差 [rad]。"""
    d = abs((a + np.pi) % (2 * np.pi) - np.pi)
    return min(d, np.pi - d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-cams", type=int, default=7)
    ap.add_argument("--list", default="val.lst")
    ap.add_argument("--root", default="out/bevlane")
    ap.add_argument("--scenes", type=int, default=60)
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--thresh", type=float, default=0.25)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    scenes = [l.strip() for l in open(a.list) if l.strip()][:a.scenes]
    ds = BevLaneDataset(a.root, scenes, gt_key="gt_cons", with_boxdet=True,
                        max_per_scene=4, n_cams=8, trim_start=3, trim_end=10)
    m = MODELS[a.model](n_seg=21).cuda().eval()
    sd = torch.load(a.ckpt, map_location="cpu")
    sd = {k.replace("module.", ""): v for k, v in sd.get("model", sd).items()}
    cur = m.state_dict()
    m.load_state_dict({k: v for k, v in sd.items()
                       if k in cur and cur[k].shape == v.shape}, strict=False)

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
                if d2 < 4.0 and (best is None or d2 < best[0]):
                    best = (d2, dyaw)
            if best is None:
                continue
            py = best[1]
            bearing = np.arctan2(ye, xe)          # 自車から見た方位
            rows.append((fold(py - yaw), fold(py - 0.0), fold(py - bearing),
                         fold(yaw - 0.0), fold(yaw - bearing), r))
        done += 1
        if done >= a.frames:
            break

    A = np.degrees(np.array(rows))
    if not len(A):
        sys.exit("箱が取れなかった")
    print(f"\n===== {a.tag or a.ckpt} ({len(A)} 箱) =====")
    print("列: 予測-GT / 予測-0度 / 予測-視線 / GT-0度 / GT-視線")
    print(f"全体 中央値: {np.median(A[:, 0]):5.1f} / {np.median(A[:, 1]):5.1f}"
          f" / {np.median(A[:, 2]):5.1f} / {np.median(A[:, 3]):5.1f}"
          f" / {np.median(A[:, 4]):5.1f} 度")
    # GT が斜めの箱だけを見る (ここが壊れている帯)
    ob = A[A[:, 3] >= 15.0]
    if len(ob) >= 5:
        print(f"\nGT が斜め (自車平行から 15 度以上) の {len(ob)} 箱:")
        print(f"  予測-GT   中央値 {np.median(ob[:, 0]):5.1f} 度")
        print(f"  予測-0度  中央値 {np.median(ob[:, 1]):5.1f} 度  "
              f"(小さいほど自車平行に張り付いている)")
        print(f"  予測-視線 中央値 {np.median(ob[:, 2]):5.1f} 度  "
              f"(小さいほど視線方向に張り付いている)")
        print(f"  参考: GT-0度 {np.median(ob[:, 3]):5.1f} 度 / "
              f"GT-視線 {np.median(ob[:, 4]):5.1f} 度")
        n0 = int((ob[:, 1] < ob[:, 0]).sum())
        nb = int((ob[:, 2] < ob[:, 0]).sum())
        print(f"  GT より 0 度に近い箱: {n0}/{len(ob)}  "
              f"GT より視線に近い箱: {nb}/{len(ob)}")


if __name__ == "__main__":
    main()

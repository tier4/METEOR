"""後方 3D BBox の距離精度が動かない原因を 3 つの仮説に分解して測る。

仮説 A: 深度ヘッドの推定そのものが後方カメラで悪い
    -> depth_gt4/depth_gt4n を正解に、カメラ別・距離帯別の深度誤差を直接測る。
       前方カメラと同じ精度なら深度ヘッドは無罪。

仮説 B: 幾何 (画素分解能) の上限
    -> K から各カメラの焦点距離を取り、30 m の車両が何画素に写るかを計算。
       後方カメラの px/m が前方より低ければ、学習では埋まらない構造差。

仮説 C: 特徴はあるのに検出が沈んでいる (しきい値/較正の問題)
    -> 後方 20-40 m の GT 箱位置でヒートマップのピーク値を測る。
       ピークが立っていれば較正の問題、立っていなければ特徴の問題。

1 パスで A と C を同時に測る (B はマニフェストから計算するだけ)。
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bevlane.model import MODELS                                  # noqa: E402

CAMS = ["CAM_FRONT_WIDE", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
        "CAM_BACK_WIDE", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
        "CAM_FRONT_NARROW", "CAM_BACK_NARROW"]
D_MIN, D_STEP = 1.0, 1.25              # v52 の深度ビン
BANDS = [(10, 20), (20, 40), (40, 80)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", default="v52")
    ap.add_argument("--list", default="val.lst")
    ap.add_argument("--root", default="out/bevlane")
    ap.add_argument("--scenes", type=int, default=40)
    ap.add_argument("--frames", type=int, default=120)
    a = ap.parse_args()

    # ---- 仮説 B: 画素分解能 (フォワード不要) ----
    scenes = [l.strip() for l in open(a.list) if l.strip()][:a.scenes]
    m0 = json.load(open(os.path.join(a.root, scenes[0], "manifest.json")))
    print("=== 仮説 B: 30 m 先の車両 (幅 1.8 m) が写る画素数 ===")
    for c in CAMS:
        if c not in m0["cams"]:
            continue
        fx = float(np.array(m0["cams"][c]["K"])[0][0])
        print(f"  {c:<18} fx={fx:7.1f}px   30m の車幅 {fx * 1.8 / 30:6.1f}px"
              f"  1px の距離分解能@30m {30 * 30 / (fx * 1.55):.2f}m")

    net = MODELS[a.model](n_seg=21).cuda().eval()
    sd = torch.load(a.ckpt, map_location="cpu")
    sd = {k.replace("module.", ""): v for k, v in sd.get("model", sd).items()}
    cur = net.state_dict()
    net.load_state_dict({k: v for k, v in sd.items()
                         if k in cur and cur[k].shape == v.shape}, strict=False)

    # 集計器
    derr = {(c, b): [] for c in range(8) for b in BANDS}     # 仮説 A
    dbias = {(c, b): [] for c in range(8) for b in BANDS}
    peaks = {("前", b): [] for b in BANDS} | {("後", b): [] for b in BANDS}

    nfr = 0
    for s in scenes:
        d = os.path.join(a.root, s)
        try:
            man = json.load(open(os.path.join(d, "manifest.json")))
        except Exception:
            continue
        if any(c not in man["cams"] for c in CAMS):
            continue
        K = np.stack([np.array(man["cams"][c]["K"], np.float32) for c in CAMS])
        Tc = np.stack([np.linalg.inv(np.array(man["cams"][c]["T_ego_cam"],
                                              np.float32)) for c in CAMS])
        fr = man["frames"]
        if len(fr) > 20:
            fr = fr[3:len(fr) - 10]
        for f in fr[::12]:
            if "depth4" not in f or "bev_box_p" not in f:
                continue
            try:
                ims = np.stack([cv2.imread(os.path.join(d, f["imgs"][c])
                                           )[:, :, ::-1] for c in CAMS])
                dgt = np.load(os.path.join(d, f["depth4"]))["depth"]
                dgn = np.load(os.path.join(d, f["depth4n"]))["depth"]
                bx = np.load(os.path.join(d, f["bev_box_p"]))["boxes"]
            except Exception:
                continue
            dall = np.concatenate([dgt, dgn], 0).astype(np.float32)  # [8,108,192]
            t = torch.from_numpy(
                np.ascontiguousarray(ims.transpose(0, 3, 1, 2))
            )[None].float().cuda() / 255.0
            with torch.no_grad(), torch.autocast("cuda", torch.float16):
                out = net(t, torch.from_numpy(K)[None].cuda(),
                          torch.from_numpy(Tc)[None].cuda())
            dep = out[1].float()
            if dep.dim() == 5:                     # [B,N,D,h,w]
                dep = dep[0]
            elif dep.dim() == 4 and dep.shape[0] == 8:
                pass
            else:
                dep = dep.view(8, -1, dep.shape[-2], dep.shape[-1])
            pred_m = (dep.argmax(1).cpu().numpy() * D_STEP + D_MIN)  # [8,h,w]
            if pred_m.shape[-2:] != dall.shape[-2:]:
                pred_m = np.stack([cv2.resize(p, dall.shape[-1:-3:-1],
                                              interpolation=cv2.INTER_NEAREST)
                                   for p in pred_m])
            # ---- 仮説 A: カメラ別・帯別の深度誤差 ----
            for ci in range(8):
                g = dall[ci]
                ok = g > 0.5
                for b in BANDS:
                    sel = ok & (g >= b[0]) & (g < b[1])
                    if sel.sum() < 20:
                        continue
                    e = pred_m[ci][sel] - g[sel]
                    derr[(ci, b)].append(np.median(np.abs(e)))
                    dbias[(ci, b)].append(np.median(e))
            # ---- 仮説 C: GT 箱位置のヒートマップピーク ----
            hm = torch.sigmoid(out[3].float())[0, 0].cpu().numpy()  # 車両ch
            res = 160.0 / hm.shape[0]              # 前後 160 m / 行数
            for r in bx:
                if int(r[0]) != 1 or float(r[3]) <= 0:
                    continue
                x, y = float(r[1]), float(r[2])
                rr = (x * x + y * y) ** 0.5
                band = next((b for b in BANDS if b[0] <= rr < b[1]), None)
                if band is None or abs(y) > 45:
                    continue
                ri = int((80.0 - x) / res)
                ci_ = int((50.0 - y) / res)
                w = max(1, int(2.0 / res))
                r0, r1 = max(0, ri - w), min(hm.shape[0], ri + w + 1)
                c0, c1 = max(0, ci_ - w), min(hm.shape[1], ci_ + w + 1)
                if r1 <= r0 or c1 <= c0:
                    continue
                peaks[("前" if x > 0 else "後", band)].append(
                    float(hm[r0:r1, c0:c1].max()))
            nfr += 1
            if nfr >= a.frames:
                break
        if nfr >= a.frames:
            break

    print(f"\n=== 仮説 A: 深度ヘッドの誤差 (対 LiDAR 深度 GT, {nfr} フレーム) ===")
    print(f"{'カメラ':<18} " + "  ".join(f"{b[0]}-{b[1]}m 誤差/偏り" for b in BANDS))
    for ci, c in enumerate(CAMS):
        row = []
        for b in BANDS:
            v, w = derr[(ci, b)], dbias[(ci, b)]
            row.append(f"{np.median(v):5.2f}/{np.median(w):+5.2f}m"
                       if v else "   —      ")
        print(f"  {c:<18} " + "  ".join(row))

    print("\n=== 仮説 C: GT 箱位置のヒートマップピーク (車両ch) ===")
    for side in ("前", "後"):
        for b in BANDS:
            p = peaks[(side, b)]
            if not p:
                continue
            p = np.array(p)
            print(f"  {side} {b[0]:2d}-{b[1]:2d}m (n={len(p):4d}): "
                  f"ピーク中央値 {np.median(p):.3f}  "
                  f"0.25未満 {100 * (p < 0.25).mean():4.1f}%  "
                  f"0.10未満 {100 * (p < 0.10).mean():4.1f}%")


if __name__ == "__main__":
    main()

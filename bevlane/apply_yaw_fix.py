"""ヨーバイアス補正: 既存 gt_cons を自車原点まわりに回転 → gt_cons_yf、
wp を同回転 → ego_motion_yf.npz。既存ファイルは無修正 (新キーのみ生成)。

回転の向きは --sign (+1/-1) で切替可能にし、実測 (GT 箱の車線中央
オフセットの距離比例成分が潰れる向き) で確定する。ラスタと wp は
同一の補正回転を共有する。
"""
import argparse
import json
import os

import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--list", required=True)
ap.add_argument("--bias", required=True, help="estimate_yaw_bias.py の JSON")
ap.add_argument("--sign", type=float, default=1.0)
ap.add_argument("--gt-key", default="gt_cons")
ap.add_argument("--out-key", default="gt_cons_yf")
ap.add_argument("--ledger", default="out/yawfix_created.txt")
a = ap.parse_args()

bias = json.load(open(a.bias))
scenes = [l.strip() for l in open(a.list) if l.strip()]
led = open(a.ledger, "a")
n_sc = n_png = 0
for s in scenes:
    if s not in bias:
        continue
    b_deg = a.sign * bias[s]["bias_deg"]
    src = os.path.join(a.root, s, a.gt_key)
    dst = os.path.join(a.root, s, a.out_key)
    if not os.path.isdir(src):
        continue
    os.makedirs(dst, exist_ok=True)
    # 自車原点 (col=250, row=400) まわりの回転。angle は cv2 の画像平面
    # (x=col, y=row) での CCW 度数。向きの正しさは呼び出し側の実測で保証。
    M = cv2.getRotationMatrix2D((250.0, 400.0), b_deg, 1.0)
    for f in sorted(os.listdir(src)):
        if not f.endswith(".png"):
            continue
        g = cv2.imread(os.path.join(src, f), 0)
        if g is None:
            continue
        r = cv2.warpAffine(g, M, (g.shape[1], g.shape[0]),
                           flags=cv2.INTER_NEAREST,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=255)
        cv2.imwrite(os.path.join(dst, f), r)
        n_png += 1
    led.write(dst + "\n")
    # wp の同回転 (ego 座標系の点集合)。ラスタと同じ補正回転を適用する。
    # ラスタは「画素値の移動」なので content が +φ 回るとき、点も +φ 回す。
    emo_p = os.path.join(a.root, s, "ego_motion.npz")
    if os.path.exists(emo_p):
        try:
            z = dict(np.load(emo_p))
            wp = z["wp"].copy()                       # [T,6,2] (x,y)
            # cv2 画像平面 (col~-y, row~-x) での +b_deg CCW は、ego (x,y)
            # 平面でも +b_deg の回転に一致する (両軸反転は回転向きを保存)。
            th = np.radians(b_deg)
            c0, s0 = np.cos(th), np.sin(th)
            x, y = wp[..., 0].copy(), wp[..., 1].copy()
            z["wp"] = np.stack([c0 * x - s0 * y, s0 * x + c0 * y],
                               -1).astype(wp.dtype)
            np.savez_compressed(os.path.join(a.root, s,
                                             "ego_motion_yf.npz"), **z)
            led.write(os.path.join(a.root, s, "ego_motion_yf.npz") + "\n")
        except Exception as e:
            print(f"[warn] {s} wp 失敗: {e}")
    n_sc += 1
led.close()
print(f"完了: {n_sc} シーン / {n_png} ラスタ (sign={a.sign})")
print("APPLY_YAW_FIX_DONE")

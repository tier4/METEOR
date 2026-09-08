"""probe_yaw_paired.py の出力 2 本を突き合わせ、共通箱で yaw を比較する。"""
import sys

import numpy as np

A_TAG, A_NPZ, B_TAG, B_NPZ = sys.argv[1:5]
a = np.load(A_NPZ)["rows"]
b = np.load(B_NPZ)["rows"]


def keyed(x):
    return {(int(r[0]), int(r[1])): r for r in x}


ka, kb = keyed(a), keyed(b)
common = sorted(set(ka) & set(kb))
print(f"{A_TAG}: 検出 {len(ka)} 箱 / {B_TAG}: 検出 {len(kb)} 箱 "
      f"-> 共通 {len(common)} 箱")
if not common:
    sys.exit("共通箱が無い")

# 同じキーが同じ箱を指しているかの検算 (データセット構成のズレ検知)
dx = np.array([abs(ka[k][2] - kb[k][2]) + abs(ka[k][3] - kb[k][3])
               for k in common])
if dx.max() > 0.01:
    print(f"警告: 同一キーの GT 位置が最大 {dx.max():.3f} m ずれている "
          f"(フレームリストか箱の並びが一致していない)")

ea = np.degrees([ka[k][5] for k in common])
eb = np.degrees([kb[k][5] for k in common])
ang = np.array([ka[k][6] for k in common])       # GT の斜め度 (0=正対)
rad = np.array([ka[k][4] for k in common])

print(f"\n=== 共通 {len(common)} 箱での yaw 誤差 (中央値 / 平均) ===")
print(f"{A_TAG}: {np.median(ea):5.1f} / {ea.mean():5.1f} 度")
print(f"{B_TAG}: {np.median(eb):5.1f} / {eb.mean():5.1f} 度")
print(f"差 (B-A): 中央値 {np.median(eb) - np.median(ea):+5.1f} 度 / "
      f"箱ごとの差の中央値 {np.median(eb - ea):+5.1f} 度")

print("\n--- GT 姿勢の斜め度別 ---")
for lo, hi in ((0, 15), (15, 45), (45, 75), (75, 90)):
    s = (ang >= lo) & (ang < hi)
    if s.sum() >= 3:
        print(f"  {lo:2d}-{hi:2d}度 (n={s.sum():4d}): "
              f"{A_TAG} {np.median(ea[s]):5.1f} / "
              f"{B_TAG} {np.median(eb[s]):5.1f} 度")

print("\n--- 距離別 ---")
for lo, hi in ((0, 20), (20, 40), (40, 80)):
    s = (rad >= lo) & (rad < hi)
    if s.sum() >= 3:
        print(f"  {lo:2d}-{hi:2d}m (n={s.sum():4d}): "
              f"{A_TAG} {np.median(ea[s]):5.1f} / "
              f"{B_TAG} {np.median(eb[s]):5.1f} 度")

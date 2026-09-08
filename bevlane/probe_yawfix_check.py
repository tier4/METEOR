"""yawfix の受け入れ判定: GT 箱の車線中央オフセット (距離比例成分) を
指定 gt キーで測る。dataset を介さず png / bev_box を直接読む。"""
import argparse
import os

import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--list", required=True)
ap.add_argument("--gt-key", default="gt_cons")
ap.add_argument("--tag", default="")
a = ap.parse_args()

scenes = [l.strip() for l in open(a.list) if l.strip()]
BANDS = [("前 0-15m", 0, 15, 1), ("前 15-30m", 15, 30, 1),
         ("前 30-45m", 30, 45, 1), ("後 0-15m", 0, 15, -1),
         ("後 15-30m", 15, 30, -1)]
off = {b[0]: [] for b in BANDS}
n_fr = 0
for s in scenes:
    gd = os.path.join(a.root, s, a.gt_key)
    bd = os.path.join(a.root, s, "bev_box")
    if not (os.path.isdir(gd) and os.path.isdir(bd)):
        continue
    for f in sorted(x for x in os.listdir(gd) if x.endswith(".png"))[3:-10:3]:
        g = cv2.imread(os.path.join(gd, f), 0)
        try:
            bx = np.load(os.path.join(bd, f[:-4] + ".npz"))["boxes"]
        except Exception:
            continue
        if g is None:
            continue
        gm = (np.where(g == 255, 0, g) == 4)
        n_fr += 1
        for r in bx:
            cls, xe, ye, ln = r[0], r[1], r[2], r[3]
            if ln <= 0 or cls >= 1.5 or abs(ye) > 10:
                continue
            rr = int((80.0 - xe) / 0.2)
            if not (2 <= rr < 798):
                continue
            c = (50.0 - ye) / 0.2
            cols = np.flatnonzero(gm[rr - 2:rr + 3].any(0))
            lf, rt = cols[cols < c - 2], cols[cols > c + 2]
            if len(lf) == 0 or len(rt) == 0:
                continue
            cl, cr = lf.max(), rt.min()
            if not (2.2 <= (cr - cl) * 0.2 <= 5.0):
                continue
            d = ((cl + cr) / 2.0 - c) * 0.2
            for nm, lo, hi, sgn in BANDS:
                if lo <= abs(xe) < hi and (xe > 0) == (sgn > 0):
                    off[nm].append(d)
print(f"--- {a.tag or a.gt_key} ({n_fr} 枚): 箱の車線中央オフセット (+=左)")
means = {}
for nm, *_ in BANDS:
    o = np.array(off[nm])
    if len(o) >= 8:
        means[nm] = o.mean()
        print(f"  {nm:<9} n={len(o):4d}  {o.mean():+.3f}±{o.std():.3f} m")
if "前 15-30m" in means and "前 0-15m" in means:
    print(f"  距離比例成分 (前 15-30 − 前 0-15): "
          f"{means['前 15-30m'] - means['前 0-15m']:+.3f} m")
print("YAWFIX_CHECK_DONE")

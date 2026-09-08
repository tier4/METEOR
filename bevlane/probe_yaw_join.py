"""Join two probe_yaw_paired.py outputs and compare yaw on the common boxes."""
import sys

import numpy as np

A_TAG, A_NPZ, B_TAG, B_NPZ = sys.argv[1:5]
a = np.load(A_NPZ)["rows"]
b = np.load(B_NPZ)["rows"]


def keyed(x):
    return {(int(r[0]), int(r[1])): r for r in x}


ka, kb = keyed(a), keyed(b)
common = sorted(set(ka) & set(kb))
print(f"{A_TAG}: {len(ka)} boxes detected / {B_TAG}: {len(kb)} boxes detected "
      f"-> {len(common)} common")
if not common:
    sys.exit("no common boxes")

# sanity check that equal keys refer to the same box (detects dataset-config drift)
dx = np.array([abs(ka[k][2] - kb[k][2]) + abs(ka[k][3] - kb[k][3])
               for k in common])
if dx.max() > 0.01:
    print(f"warning: GT positions for the same key differ by up to {dx.max():.3f} m "
          f"(frame list or box order does not match)")

ea = np.degrees([ka[k][5] for k in common])
eb = np.degrees([kb[k][5] for k in common])
ang = np.array([ka[k][6] for k in common])       # GT obliqueness (0 = ego-parallel)
rad = np.array([ka[k][4] for k in common])

print(f"\n=== yaw error on {len(common)} common boxes (median / mean) ===")
print(f"{A_TAG}: {np.median(ea):5.1f} / {ea.mean():5.1f} deg")
print(f"{B_TAG}: {np.median(eb):5.1f} / {eb.mean():5.1f} deg")
print(f"diff (B-A): median {np.median(eb) - np.median(ea):+5.1f} deg / "
      f"median of per-box diff {np.median(eb - ea):+5.1f} deg")

print("\n--- by GT heading obliqueness ---")
for lo, hi in ((0, 15), (15, 45), (45, 75), (75, 90)):
    s = (ang >= lo) & (ang < hi)
    if s.sum() >= 3:
        print(f"  {lo:2d}-{hi:2d}deg (n={s.sum():4d}): "
              f"{A_TAG} {np.median(ea[s]):5.1f} / "
              f"{B_TAG} {np.median(eb[s]):5.1f} deg")

print("\n--- by range ---")
for lo, hi in ((0, 20), (20, 40), (40, 80)):
    s = (rad >= lo) & (rad < hi)
    if s.sum() >= 3:
        print(f"  {lo:2d}-{hi:2d}m (n={s.sum():4d}): "
              f"{A_TAG} {np.median(ea[s]):5.1f} / "
              f"{B_TAG} {np.median(eb[s]):5.1f} deg")

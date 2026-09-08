"""Scan for scenes rich in far vehicles (for the D2' lever, 2026-08-27).

val3D-zone shows 40-80 m recall (0.19-0.28) at half the near-range value (0.46-0.53).
As a cheap alternative to GT-paste (D2), build a scene list for oversampling scenes
with many far vehicles (class=1, 40<|x|<=80).
Output: out/farveh_scene_scores.tsv (scene\tfrac\tmean_far).
Thresholding into a list is done separately after inspecting the distribution.
"""
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = sys.argv[1] if len(sys.argv) > 1 else "out/bevlane"
LIST = sys.argv[2] if len(sys.argv) > 2 else "out/round74_scenes.txt"
STRIDE = 8


def scan(scene):
    fs = sorted(glob.glob(os.path.join(ROOT, scene, "bev_box", "*.npz")))
    if not fs:
        return scene, -1.0, -1.0
    n_rich = 0
    tot_far = 0
    n = 0
    for f in fs[::STRIDE]:
        try:
            b = np.load(f)["boxes"]
        except Exception:
            continue
        n += 1
        if len(b) == 0:
            continue
        far = int(((b[:, 0] == 1) & (np.abs(b[:, 1]) > 40)
                   & (np.abs(b[:, 1]) <= 80)).sum())
        tot_far += far
        if far >= 2:
            n_rich += 1
    if n == 0:
        return scene, -1.0, -1.0
    return scene, n_rich / n, tot_far / n


def main():
    scenes = open(LIST).read().split()
    with ProcessPoolExecutor(max_workers=16) as ex:
        rows = list(ex.map(scan, scenes, chunksize=32))
    with open("out/farveh_scene_scores.tsv", "w") as fh:
        fh.write("scene\tfrac_rich\tmean_far\n")
        for s, fr, mf in rows:
            fh.write(f"{s}\t{fr:.3f}\t{mf:.2f}\n")
    ok = [r for r in rows if r[1] >= 0]
    fr = np.array([r[1] for r in ok])
    print(f"scenes={len(rows)} with_boxdir={len(ok)}")
    for t in (0.3, 0.5, 0.7):
        print(f"frac_rich>={t}: {(fr >= t).sum()}")
    print("SCAN_FARVEH_DONE")


if __name__ == "__main__":
    main()

"""When yaw is wrong, find out which direction the prediction is pulled toward.

Hypotheses:
  A. snaps to ego-parallel (0 deg)  -> dragged by the skewed GT heading distribution
     (most vehicles face along the road, so answering 0 deg when unsure is cheapest for the loss)
  B. snaps to the line of sight (bearing from ego) -> lift depth bins are coarse,
     footprints smear along the ray and the heading cue is lost
The fix differs (A: weighting/loss, B: depth resolution).
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
    """Axis error with 180-degree symmetry [rad]."""
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
            bearing = np.arctan2(ye, xe)          # bearing from ego
            rows.append((fold(py - yaw), fold(py - 0.0), fold(py - bearing),
                         fold(yaw - 0.0), fold(yaw - bearing), r))
        done += 1
        if done >= a.frames:
            break

    A = np.degrees(np.array(rows))
    if not len(A):
        sys.exit("no boxes obtained")
    print(f"\n===== {a.tag or a.ckpt} ({len(A)} boxes) =====")
    print("columns: pred-GT / pred-0deg / pred-LOS / GT-0deg / GT-LOS")
    print(f"overall median: {np.median(A[:, 0]):5.1f} / {np.median(A[:, 1]):5.1f}"
          f" / {np.median(A[:, 2]):5.1f} / {np.median(A[:, 3]):5.1f}"
          f" / {np.median(A[:, 4]):5.1f} deg")
    # only boxes with oblique GT (this is the broken band)
    ob = A[A[:, 3] >= 15.0]
    if len(ob) >= 5:
        print(f"\n{len(ob)} boxes with oblique GT (>= 15 deg from ego-parallel):")
        print(f"  pred-GT    median {np.median(ob[:, 0]):5.1f} deg")
        print(f"  pred-0deg  median {np.median(ob[:, 1]):5.1f} deg  "
              f"(smaller = more snapped to ego-parallel)")
        print(f"  pred-LOS   median {np.median(ob[:, 2]):5.1f} deg  "
              f"(smaller = more snapped to line of sight)")
        print(f"  ref: GT-0deg {np.median(ob[:, 3]):5.1f} deg / "
              f"GT-LOS {np.median(ob[:, 4]):5.1f} deg")
        n0 = int((ob[:, 1] < ob[:, 0]).sum())
        nb = int((ob[:, 2] < ob[:, 0]).sum())
        print(f"  boxes closer to 0 deg than GT: {n0}/{len(ob)}  "
              f"boxes closer to LOS than GT: {nb}/{len(ob)}")


if __name__ == "__main__":
    main()

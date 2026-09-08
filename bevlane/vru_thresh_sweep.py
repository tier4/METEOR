#!/usr/bin/env python3
"""Why did the rear-40 line halve VRU recall while vehicles held?

v55 (full grid): vru R50 0.29 / veh 0.45.  v59 (rear-40): vru 0.18 / veh 0.45.
The det loss already carries a x5 VRU channel weight and x3 near boost, so
"weight it harder" is not the untried lever -- this measures which of three
mechanisms actually moved, same frames, one process:

  A  eval denominator: GT behind the rear extent counts against recall the
     truncated model cannot see. Both models are scored twice: all GT vs GT
     inside the front window only.
  B  range structure: recall per distance band, front-only.
  C  score scale: max heatmap score per matched/missed VRU -- v60's P 0.78 /
     R 0.13 pattern smells like scores slid under the decode threshold, which
     is calibration, not capacity.

    METEOR_BEV_XR=40.0 python3 bevlane/vru_diag.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bevlane.dataset import BevLaneDataset                      # noqa: E402
from bevlane.model import MODELS                                # noqa: E402

BANDS = ((0, 10), (10, 20), (20, 35), (35, 50))


def run(tag, ckpt, model, n_cams, scenes, THRESH=0.25):
    ds = BevLaneDataset("out/bevlane", scenes, gt_key="gt_cons",
                        with_boxdet=True, max_per_scene=4, n_cams=n_cams,
                        trim_start=3, trim_end=10)
    m = MODELS[model](n_seg=21).cuda().eval()
    sd = {k.replace("module.", ""): v for k, v in
          torch.load(ckpt, map_location="cpu")["model"].items()}
    cur = m.state_dict()
    m.load_state_dict({k: v for k, v in sd.items()
                       if k in cur and cur[k].shape == v.shape}, strict=False)
    DP = {0: [0, 0], 1: [0, 0]}
    tp = np.zeros((2, len(BANDS)))
    fn = np.zeros((2, len(BANDS)))
    miss_sc, hit_sc = [], []
    n = 0
    bx_idx = 4                     # (imgs,K,T,gt, boxes, nbox) with_boxdet
    for i in range(0, len(ds), max(1, len(ds) // 200)):
        b = ds[i]
        if b is None:
            continue
        with torch.no_grad(), torch.autocast("cuda", torch.float16):
            out = m(b[0][None].cuda(), b[1][None].cuda(), b[2][None].cuda())
        hm = out[3].float()
        dets = m.decode_boxes(hm, out[4].float(), thresh=THRESH)[0]
        det_by_cls = {0: [], 1: []}
        for d in dets:
            det_by_cls[int(d[0])].append((d[2], d[3], d[1]))
        # precision 側: 各検出が GT (前方窓) に 2m 以内でマッチするか
        gt_by_cls = {0: [], 1: []}
        for k in range(max(int(b[bx_idx + 1]), 0)):
            cls_, xe_, ye_, l_ = [float(v) for v in b[bx_idx][k][:4]]
            if l_ <= 0:
                continue
            gt_by_cls[0 if cls_ < 1.5 else 1].append((xe_, ye_))
        for ci in (0, 1):
            for dx, dy, _ in det_by_cls[ci]:
                if not (0 < dx <= 50 or (-20 <= dx <= 0 and (dx*dx+dy*dy) <= 2500)):
                    continue
                ok = any((dx-gx)**2 + (dy-gy)**2 < 4.0 for gx, gy in gt_by_cls[ci])
                DP[ci][0] += ok
                DP[ci][1] += 1
        hm_vru = hm.sigmoid()[0, 1].cpu().numpy()
        bx, nb = b[bx_idx], int(b[bx_idx + 1])
        for k in range(max(nb, 0)):
            cls, xe, ye, l, w, yaw = [float(v) for v in bx[k][:6]]
            if l <= 0:
                continue
            ci = 0 if cls < 1.5 else 1
            r = (xe * xe + ye * ye) ** 0.5
            if not (0 < xe <= 50 or (-20 <= xe <= 0 and r <= 50)):
                continue                     # front window, <=50 m
            band = next((bi for bi, (lo, hi) in enumerate(BANDS)
                         if lo <= r < hi), None)
            if band is None:
                continue
            matched = any((xe - dx) ** 2 + (ye - dy) ** 2 < 4.0
                          for dx, dy, _ in det_by_cls[ci])
            (tp if matched else fn)[ci, band] += 1
            if ci == 1:
                rr = int((80.0 - xe) / 0.4)
                cc = int((50.0 - ye) / 0.4)
                if 0 <= rr < hm_vru.shape[0] and 0 <= cc < hm_vru.shape[1]:
                    sc = float(hm_vru[max(0, rr - 2):rr + 3,
                                      max(0, cc - 2):cc + 3].max())
                    (hit_sc if matched else miss_sc).append(sc)
        n += 1
        if n >= 200:
            break
    print(f"\n=== {tag}  ({n} frames, front window <=50 m) ===")
    for ci, nm in ((0, "veh"), (1, "vru")):
        row = " ".join(
            f"{lo}-{hi}m {tp[ci, bi] / max(tp[ci, bi] + fn[ci, bi], 1):.2f}"
            f"({int(tp[ci, bi] + fn[ci, bi])})"
            for bi, (lo, hi) in enumerate(BANDS))
        tot = tp[ci].sum() / max(tp[ci].sum() + fn[ci].sum(), 1)
        print(f"  {nm}: R={tot:.2f}  " + row)
    for ci, nm in ((0, "veh"), (1, "vru")):
        p = DP[ci][0] / max(DP[ci][1], 1)
        print(f"  {nm} P={p:.2f} ({DP[ci][0]}/{DP[ci][1]})")
    if miss_sc or hit_sc:
        print(f"  vru hm score: hit median "
              f"{np.median(hit_sc) if hit_sc else float('nan'):.2f}  "
              f"miss median "
              f"{np.median(miss_sc) if miss_sc else float('nan'):.2f}  "
              f"(decode thresh 0.25; miss>=0.15 band = calibration headroom: "
              f"{np.mean([s >= 0.15 for s in miss_sc]) if miss_sc else 0:.0%})")
    del m
    torch.cuda.empty_cache()


if __name__ == "__main__":
    scenes = [l.strip() for l in open("val.lst") if l.strip()][:60]
    for t in (0.25, 0.20, 0.15, 0.10):
        run(f"v63b thresh={t}", "out/v63b_best_e2e.pt", "v63b", 7, scenes,
            THRESH=t)

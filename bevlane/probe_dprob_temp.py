#!/usr/bin/env python3
"""Does sharpening the lift depth distribution at inference raise far recall (no retraining)?

Trigger (measured 2026-08-22): the depth **expectation** is good even at 40-60 m
(4.7 m error), but the **distribution** is spread out: max prob 0.09, entropy 3.5
= +-17 m. The lift distributes features to BEV cells by this distribution, so
the model knows the right range yet smears its evidence thin. Concentrating the
distribution around the expectation (temperature T<1) should raise far peaks --
testable without touching training.

Same frames, one process, only the temperature toggled (optional-input A/B rule).
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bevlane.dataset import BevLaneDataset          # noqa: E402
from bevlane.model import MODELS                    # noqa: E402
from bevlane.ckpt_load import load_net              # noqa: E402

BANDS = [(0, 20), (20, 40), (40, 60), (60, 80)]

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--list", default="val.lst")
ap.add_argument("--frames", type=int, default=120)
ap.add_argument("--scenes", type=int, default=20)
ap.add_argument("--temps", default="1.0,0.7,0.5,0.3")
ap.add_argument("--root", default="out/bevlane")
a = ap.parse_args()

TEMPS = [t if (t in ("exp", "max") or t.startswith("cal")) else float(t)
         for t in a.temps.split(",")]

net = MODELS["v52"](n_seg=21).cuda().eval()
load_net(net, a.ckpt)

# replace sharpen_dprob with a temperature-aware version (wraps the original)
_orig = type(net).sharpen_dprob
_T = {"t": 1.0}


def _sharp(self, dprob):
    """Numeric t = temperature; "exp" = triangular kernel around the expectation; "max" = one-hot.

    exp: the depth expectation is good (4.7 m error at 40-60 m), so place the same
    triangular kernel (1 bin wide) as the LiDAR sharpen on that point and drop the smear.
    """
    dp = _orig(self, dprob)
    t = _T["t"]
    if t == 1.0:
        return dp
    p32 = dp.float().clamp_min(1e-8)
    if isinstance(t, str) and t.startswith("cal"):
        BN, D, fh, fw = p32.shape
        q = torch.einsum("ji,bihw->bjhw", _CAL, p32)
        q = q / q.sum(1, keepdim=True).clamp_min(1e-8)
        if "+" in t:                       # e.g. cal+0.3 also applies a temperature
            tt = float(t.split("+")[1])
            q = q.clamp_min(1e-8) ** (1.0 / tt)
            q = q / q.sum(1, keepdim=True)
        return q.to(dp.dtype)
    if t in ("exp", "max"):
        D = p32.shape[1]
        idx = torch.arange(D, device=p32.device, dtype=p32.dtype).view(1, D, 1, 1)
        if t == "max":
            k = p32.argmax(1, keepdim=True).to(p32.dtype)
        else:
            k = (p32 * idx).sum(1, keepdim=True)
        tri = (1.0 - (idx - k).abs()).clamp(min=0)
        return (tri / tri.sum(1, keepdim=True).clamp_min(1e-8)).to(dp.dtype)
    q = p32 ** (1.0 / float(t))
    return (q / q.sum(1, keepdim=True)).to(dp.dtype)


# --- depth calibration ("cal"): apply the monotone map pred z -> true z along the depth axis ---
# Over all 3.92M pixels the prediction is consistently too far (-7.1 m in the 45-50 m band).
# That is an axis offset, not a sharpness issue, so redistribute mass to the mapped bins.
# Implementation is one fixed D x D matrix (64x64) = essentially free. For production,
# rewriting the lift bin centers themselves removes even that.
_CAL = None
if os.path.exists("out/depth_calib_table.npy"):
    _t = np.load("out/depth_calib_table.npy")
    _zc = (net.D_MIN + torch.arange(net.D).float() * net.D_STEP)
    _tgt = torch.from_numpy(
        np.interp(_zc.numpy(), _t[:, 0], _t[:, 1])).float()
    _idx = ((_tgt - net.D_MIN) / net.D_STEP).clamp(0, net.D - 1)
    _lo = _idx.floor().long()
    _w = (_idx - _lo.float())
    _M = torch.zeros(net.D, net.D)
    for _i in range(net.D):
        _M[_lo[_i], _i] += 1 - _w[_i]
        _M[min(_lo[_i] + 1, net.D - 1), _i] += _w[_i]
    _CAL = _M.cuda()
    print(f"[cal] calibration table loaded: {float(_zc[20]):.1f}m -> "
          f"{float(_tgt[20]):.1f}m, {float(_zc[40]):.1f}m -> {float(_tgt[40]):.1f}m")


type(net).sharpen_dprob = _sharp

ds = BevLaneDataset(a.root, [l.strip() for l in open(a.list) if l.strip()][:a.scenes],
                    gt_key="gt_cons", with_boxdet=True, with_ego=True,
                    max_per_scene=8, n_cams=8, trim_start=3, trim_end=10)

res = {t: dict(gt={b: 0 for b in BANDS}, hit={b: 0 for b in BANDS},
               nd=[], dy=[], tp=0, fp=0) for t in TEMPS}
step = max(1, len(ds) // a.frames)
done = 0
for i in range(0, len(ds), step):
    b = ds[i]
    if b is None:
        continue
    bx, nb, eg = b[4], int(b[5]), b[6]
    v0t = torch.tensor([float(eg[12]) if torch.is_tensor(eg) else 8.0]).cuda()
    for t in TEMPS:
        _T["t"] = t
        with torch.no_grad(), torch.autocast("cuda", torch.float16):
            out = net(b[0][None].cuda(), b[1][None].cuda(), b[2][None].cuda(),
                      v0=v0t)
        dets = net.decode_boxes(out[3].float().cpu(), out[4].float().cpu(),
                                thresh=0.25)[0]
        pv = [(float(d[2]), float(d[3])) for d in dets if int(d[0]) == 0]
        R = res[t]
        R["nd"].append(len(pv))
        # precision: does the predicted box hit a GT vehicle (within 3 m)?
        # Required: recall alone rises simply because sharpening adds boxes.
        gtv = [(float(bx[kk][1]), float(bx[kk][2]))
               for kk in range(max(nb, 0))
               if float(bx[kk][3]) > 0 and float(bx[kk][0]) < 1.5]
        for px2, py2 in pv:
            ok = any((px2 - gx) ** 2 + (py2 - gy) ** 2 < 9.0 for gx, gy in gtv)
            R["tp" if ok else "fp"] += 1
        for kk in range(max(nb, 0)):
            cls, xe, ye, ln = [float(v) for v in bx[kk][:4]]
            if ln <= 0 or cls >= 1.5:
                continue
            r_ = (xe * xe + ye * ye) ** 0.5
            best = None
            for px2, py2 in pv:
                d2 = (xe - px2) ** 2 + (ye - py2) ** 2
                if d2 < 9.0 and (best is None or d2 < best[0]):
                    best = (d2, px2, py2)
            for lo, hi in BANDS:
                if lo <= r_ < hi:
                    R["gt"][(lo, hi)] += 1
                    if best:
                        R["hit"][(lo, hi)] += 1
            if best:
                R["dy"].append(best[2] - ye)
    done += 1
    if done >= a.frames:
        break

print(f"\n=== depth distribution temperature A/B ({done} frames, same frames, cam-only) ===")
print("   T     " + "  ".join(f"{lo}-{hi}m" for lo, hi in BANDS)
      + "   boxes/frame  precision   F1(20-40m)  lat err")
for t in TEMPS:
    R = res[t]
    rec = "  ".join(
        f"{R['hit'][b] / max(R['gt'][b], 1):.3f} " for b in BANDS)
    dy = np.array(R["dy"]) if R["dy"] else np.zeros(1)
    prec = R["tp"] / max(R["tp"] + R["fp"], 1)
    r2040 = R["hit"][(20, 40)] / max(R["gt"][(20, 40)], 1)
    f1 = 2 * prec * r2040 / max(prec + r2040, 1e-9)
    print(f"  {str(t):<4}  {rec}   {np.mean(R['nd']):>5.1f}   {prec:.3f}   "
          f"{f1:.3f}      {dy.mean():+.3f}±{dy.std():.3f}")

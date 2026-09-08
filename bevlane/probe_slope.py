#!/usr/bin/env python3
"""Does the flat-ground BEV lift mis-place vehicles on a slope? Measure it.

The lift samples the image at BEV points that all sit on z=0 in the ego frame
(bev_pts has a single unique z, checked: 0.0). Which pixel a BEV cell reads is
therefore fixed by K, T_cam_ego and the assumption that the road is flat; the
depth head only re-weights cameras, it does not move the sample point. So a
vehicle standing on a rising road should land in the WRONG BEV cell:

    a ray from a camera at height h through a point at forward distance X on a
    road of grade tan(theta) crosses z=0 at   X_flat = X * h / (h - X tan theta)

which is farther than X uphill, nearer downhill, and does not exist at all
beyond X = h / tan(theta), where the road has risen to the camera's own height
and the ray leaves through the horizon.

The GT boxes are horizontal footprints (cls, x, y, l, w, yaw) taken from the
annotation's true ego-frame position, so they are slope-independent and can
referee this directly. Road grade is measured per frame from the LiDAR depth GT
itself: back-project the front camera's depth into the ego frame, take a low
percentile of height per forward-distance bin as the road surface, and fit its
slope over 10..40 m.

    CUDA_VISIBLE_DEVICES=7 python3 bevlane/probe_slope.py \
        --ckpt out/distill_v50/last_regh.pt --model v50 --frames 400
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bevlane.dataset import BevLaneDataset                      # noqa: E402
from bevlane.model import MODELS                                # noqa: E402


def road_grade(depth, K, T_cam_ego, img_w, cam=0,
               x_lo=6.0, x_hi=32.0):
    """Grade of the road ahead, from the depth GT. -> (grade, camera height)."""
    d = depth[cam].astype(np.float32)
    h, w = d.shape
    vv, uu = np.mgrid[0:h, 0:w].astype(np.float32)
    # the depth map is a resized copy of the image, so K must be scaled to it
    K = K[cam].astype(np.float64).copy()
    K[:2] *= float(w) / float(img_w)
    m = (d > 1.0) & (d < 50.0)
    if m.sum() < 200:
        return None, None
    # unproject: the stored value is range along the ray in the camera frame
    zc = np.ones_like(d)
    xn = (uu - K[0, 2]) / K[0, 0]
    yn = (vv - K[1, 2]) / K[1, 1]
    n = np.sqrt(xn * xn + yn * yn + 1.0)
    pc = np.stack([xn / n * d, yn / n * d, zc / n * d], -1)      # [h,w,3]
    T = T_cam_ego[cam].astype(np.float64)
    R, t = T[:3, :3], T[:3, 3]
    pe = (pc.reshape(-1, 3) - t) @ R                            # ego frame
    pe = pe[m.reshape(-1)]
    # OWN LANE ONLY, and below waist height. A first attempt took a low
    # percentile of height over the full width and read a spurious +2.5 % grade
    # on almost every frame: LiDAR ground returns thin out with range while
    # returns from walls, parked cars and sidewalks do not, so the percentile
    # climbs with distance all by itself. Restricting to |y| < 3.5 m and
    # z < 1 m keeps the surface the ego is actually driving on.
    x, y, z = pe[:, 0], pe[:, 1], pe[:, 2]
    lane = (np.abs(y) < 3.5) & (z < 1.0) & (z > -3.0)
    x, z = x[lane], z[lane]
    edges = np.arange(x_lo, x_hi + 2.0, 2.0)
    xs, zs = [], []
    for a_, b_ in zip(edges[:-1], edges[1:]):
        s = (x >= a_) & (x < b_)
        if s.sum() < 60:            # enough points for a low quantile to mean
            continue                # something
        xs.append((a_ + b_) * 0.5)
        # 5th, not 20th: with the LLS point count per bin falling off with
        # range, a higher quantile sits above the surface by a distance that
        # grows with range, and reads as a phantom uphill. At the 20th the
        # median grade over 884 val frames came out +2.99 %, which no real road
        # network is.
        zs.append(np.percentile(z[s], 5))
    # the depth GT reaches ~27 m at the 95th percentile, so the fit window is
    # 6..32 m and five bins of it must be populated
    if len(xs) < 5:
        return None, None
    xs, zs = np.array(xs), np.array(zs)
    # one round of outlier rejection about the fit
    g, c = np.polyfit(xs, zs, 1)
    res = np.abs(zs - (g * xs + c))
    keep = res < max(0.25, 2.0 * np.median(res))
    if keep.sum() >= 5:
        g, c = np.polyfit(xs[keep], zs[keep], 1)
    # camera height comes from the extrinsics, not from the cloud: the ego
    # origin already sits on the road, so a percentile of z is ~0 by definition
    cam_h = float((-R.T @ t)[2])
    return float(g), (float(cam_h) if cam_h is not None else None)


def road_grade_lidar(lb, x_lo=6.0, x_hi=34.0, half_y=3.0, res=0.4):
    """Grade of the road ahead from the LiDAR BEV raster. -> grade or None.

    Preferred over the depth GT: lidar_bev is a direct rasterisation of the
    sweep (rows = (80-x)/0.4, cols = (50-y)/0.4, ch2 = mean z of the points in
    the cell), so no value is ever interpolated across a depth discontinuity.
    Deriving the profile from the 1/4-resized depth GT instead put a +3 % uphill
    bias on the median val frame -- resizing a depth map blends road with the
    car in front of it and the blend lands above the road.
    """
    occ, zm = lb[3] > 0.5, lb[2]
    c0 = int(round((50.0 - half_y) / res))
    c1 = int(round((50.0 + half_y) / res))
    xs, zs = [], []
    for x in np.arange(x_lo, x_hi, 2.0):
        r1 = int(round((80.0 - x) / res))
        r0 = int(round((80.0 - (x + 2.0)) / res))
        v = zm[r0:r1, c0:c1][occ[r0:r1, c0:c1]]
        if v.size < 12:
            continue
        xs.append(x + 1.0)
        zs.append(float(np.percentile(v, 20)))
    if len(xs) < 6:
        return None
    xs, zs = np.array(xs), np.array(zs)
    g, c = np.polyfit(xs, zs, 1)
    res_ = np.abs(zs - (g * xs + c))
    keep = res_ < max(0.20, 2.0 * np.median(res_))
    if keep.sum() >= 5:
        g = np.polyfit(xs[keep], zs[keep], 1)[0]
    return float(g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", default="v50")
    ap.add_argument("--scenes", default="val.lst")
    ap.add_argument("--root", default="out/bevlane")
    ap.add_argument("--frames", type=int, default=400)
    ap.add_argument("--n-scenes", type=int, default=40)
    ap.add_argument("--match", type=float, default=6.0,
                    help="max lateral+longitudinal gate for GT<->pred pairing")
    ap.add_argument("--out", default="out/slope_probe.npz")
    a = ap.parse_args()

    sc = [l.strip() for l in open(a.scenes) if l.strip()][:a.n_scenes]
    ds = BevLaneDataset(a.root, sc, gt_key="gt_cons", with_depth=True,
                        depth_hw=(108, 192), with_boxdet=True,
                        with_lidarbev=True, augment=False)
    net = MODELS[a.model](n_seg=21).cuda().eval()
    sd = torch.load(a.ckpt, map_location="cpu")["model"]
    net.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()},
                        strict=False)

    step = max(1, len(ds) // a.frames)
    rows = []                                   # (grade, x_gt, dx, cls)
    cam_hs = []
    for i in range(0, len(ds), step):
        b = ds[i]
        if b is None:
            continue
        imgs, K, Tc = b[0], b[1], b[2]
        dep = b[4].numpy()
        # the box GT is the first (N,6) float tensor after depth, followed by
        # its valid count
        bx = nbx = None
        for j, t in enumerate(b):
            if (torch.is_tensor(t) and t.dim() == 2 and t.shape[-1] == 6
                    and t.dtype == torch.float32):
                bx = t.numpy()
                nbx = int(b[j + 1]) if j + 1 < len(b) else bx.shape[0]
                break
        if bx is None:
            continue
        bx = bx[:max(nbx, 0)]
        lb = None
        for t in b:
            if torch.is_tensor(t) and t.dim() == 3 and t.shape[0] == 4 \
                    and t.shape[1] == 400:
                lb = t.numpy()
                break
        g = road_grade_lidar(lb) if lb is not None else None
        if g is None:
            continue
        _, ch = road_grade(dep, K.numpy(), Tc.numpy(), imgs.shape[-1])
        if ch is not None:
            cam_hs.append(ch)
        with torch.no_grad(), torch.autocast("cuda", torch.float16):
            out = net(imgs[None].cuda(), K[None].cuda(), Tc[None].cuda())
        det = net.decode_boxes(out[3].float(), out[4].float(),
                               thresh=0.3, topk=64)[0]
        for k in range(bx.shape[0]):
            c, xg, yg, ln = bx[k, 0], bx[k, 1], bx[k, 2], bx[k, 3]
            if ln <= 0 or xg < 5.0 or xg > 60.0:
                continue                        # ahead only: the slope is ahead
            best, bd = None, 1e9
            for (dc, s, xe, ye, l, w, yaw) in det:
                if dc != int(c):
                    continue
                d = abs(xe - xg) + abs(ye - yg)
                if d < bd:
                    bd, best = d, (xe, ye)
            if best is None or bd > a.match:
                continue
            rows.append((g, xg, best[0] - xg, int(c)))
        if len(rows) > 20000:
            break

    r = np.array(rows, np.float64)
    np.savez(a.out, rows=r, cam_h=np.array(cam_hs))
    ch = float(np.median(cam_hs)) if cam_hs else float("nan")
    print(f"\ncamera height from extrinsics (median of {len(cam_hs)}): "
          f"{ch:.2f} m")
    print(f"matched GT/prediction pairs: {len(r)}")
    if len(r):
        gg = r[:, 0]
        print(f"road grade over the pairs: median {np.median(gg):+.2%}, "
              f"10th {np.percentile(gg, 10):+.2%}, "
              f"90th {np.percentile(gg, 90):+.2%}")
    if not len(r):
        return
    print(f"\n{'road grade':>16s} {'pairs':>6s} {'signed dx':>10s} "
          f"{'|dx|':>7s} {'predicted X_flat/X':>19s}")
    bins = [(-1.0, -0.025, "downhill < -2.5%"), (-0.025, -0.01, "-2.5..-1%"),
            (-0.01, 0.01, "flat |g| < 1%"), (0.01, 0.025, "+1..+2.5%"),
            (0.025, 1.0, "uphill > +2.5%")]
    for lo, hi, lbl in bins:
        s = (r[:, 0] >= lo) & (r[:, 0] < hi)
        if s.sum() < 10:
            print(f"{lbl:>16s} {int(s.sum()):6d}   (too few)")
            continue
        dx = r[s, 2]
        ratio = ((r[s, 1] + dx) / r[s, 1]).mean()
        print(f"{lbl:>16s} {int(s.sum()):6d} {dx.mean():+9.2f}m "
              f"{np.abs(dx).mean():6.2f}m {ratio:18.3f}")
    # the geometric prediction, for the same grades, at the mean GT distance
    if ch == ch:
        print(f"\ngeometry for h={ch:.2f} m: X_flat/X = h / (h - X tan(theta))")
        xm = r[:, 1].mean()
        for gr in (-0.03, -0.015, 0.0, 0.015, 0.03):
            den = ch - xm * gr
            v = ch / den if den > 0.05 else float("inf")
            print(f"  grade {gr:+.1%} at X={xm:.0f} m -> "
                  + (f"{v:.2f}x" if np.isfinite(v)
                     else "above the horizon, no z=0 cell exists"))
        print(f"  the road reaches camera height at X = h/tan(theta): "
              + ", ".join(f"{gr:+.1%}: {ch / abs(gr):.0f} m"
                          for gr in (0.02, 0.03, 0.05)))


if __name__ == "__main__":
    main()

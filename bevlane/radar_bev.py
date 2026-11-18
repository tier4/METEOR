"""Radar detections -> BEV raster (optional radar input of METEOR2).

Input: per-frame radar points `pts` [N, 8] = x, y, z, rcs, range_rate, vr_comp, dt, sensor in the ego frame, with
about 0.5 s of sweeps motion-compensated into the current frame (written by tools/kitscenes/kits_radar_to_bevlane.py).
Output: a raster on the same 0.4 m grid as the LiDAR raster (extract_lidar_bev.raster), [C, 400, 250]
(row 0 = 80 m ahead, column 0 = 50 m to the left).

Channels:
  0  log1p(number of points)
  1  maximum RCS (/30)
  2  mean ground-relative radial velocity (/10 m/s)
  3  maximum |ground-relative radial velocity| (/10; moving objects)
  4  occupancy
  5  recency (maximum of 1 + dt / 0.5; the current sweep = 1)

Training and evaluation read METEOR_RADAR_ROOT/<scene>/<frame, 4 digits>.npz and append the raster as extra channels
after the LiDAR raster (bevlane/dataset.py).

Environment:
  METEOR_RADAR_ROOT     root directory of the per-frame radar point files (unset = no radar)
  METEOR_RADAR_SENSORS  comma-separated sensor indices to keep (e.g. "0" = the front radar only; an index that no
                        unit has, e.g. "99", removes all points). Unset = all radars.
                        KITScenes: 0 front / 1 left / 2 right.
                        nuScenes order: 0 front / 1 front-left / 2 front-right / 3 back-left / 4 back-right.
  METEOR_RADAR_FRONT_P  training only: with probability p keep only sensor 0 (so that a single-front-radar
                        configuration is covered by the same weights). Ignored when METEOR_RADAR_SENSORS is set.
"""
import os

import numpy as np

C = 6
GH, GW, RES = 400, 250, 0.4


def raster(pts):
    rb = np.zeros((C, GH, GW), np.float32)
    if pts is None or len(pts) == 0:
        return rb
    p = pts.astype(np.float32)
    r = ((80.0 - p[:, 0]) / RES).astype(np.int32)
    c = ((50.0 - p[:, 1]) / RES).astype(np.int32)
    ok = (r >= 0) & (r < GH) & (c >= 0) & (c < GW) & np.isfinite(p).all(1)
    if not ok.any():
        return rb
    p = p[ok]; flat = r[ok] * GW + c[ok]
    n = GH * GW
    cnt = np.bincount(flat, minlength=n).astype(np.float32)
    occ = cnt > 0
    rcs = np.full(n, -1e9, np.float32); np.maximum.at(rcs, flat, p[:, 3])
    vs = np.bincount(flat, weights=p[:, 5], minlength=n)
    va = np.zeros(n, np.float32); np.maximum.at(va, flat, np.abs(p[:, 5]))
    rec = np.zeros(n, np.float32); np.maximum.at(rec, flat, np.clip(1.0 + p[:, 6] / 0.5, 0.0, 1.0))
    rb[0] = np.log1p(cnt).reshape(GH, GW)
    rb[1] = np.where(occ, np.clip(rcs, -30, 60) / 30.0, 0.0).reshape(GH, GW)
    rb[2] = np.where(occ, np.clip(vs / np.maximum(cnt, 1) / 10.0, -5, 5), 0.0).reshape(GH, GW)
    rb[3] = np.clip(va / 10.0, 0, 5).reshape(GH, GW)
    rb[4] = occ.reshape(GH, GW).astype(np.float32)
    rb[5] = rec.reshape(GH, GW)
    return rb


def load(scene, frame, root=None):
    """METEOR_RADAR_ROOT/<scene>/<frame:04d>.npz -> [C, 400, 250]. Zeros when the file is missing (= no radar)."""
    root = root or os.environ.get("METEOR_RADAR_ROOT", "")
    if not root:
        return np.zeros((C, GH, GW), np.float32)
    p = os.path.join(root, scene, f"{int(frame):04d}.npz")
    try:
        pts = np.load(p)["pts"]
    except Exception:
        return np.zeros((C, GH, GW), np.float32)
    sel = os.environ.get("METEOR_RADAR_SENSORS", "")
    front_p = float(os.environ.get("METEOR_RADAR_FRONT_P", "0") or 0)
    if front_p > 0 and not sel and np.random.rand() < front_p:
        sel = "0"
    if sel and len(pts):
        pts = pts[np.isin(pts[:, 7].astype(np.int64), [int(v) for v in sel.split(",")])]
    return raster(pts)

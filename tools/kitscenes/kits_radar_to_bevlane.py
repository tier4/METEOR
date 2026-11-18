#!/usr/bin/env python3
"""Add radar points to converted KITScenes scenes (KITS_<uuid>/manifest.json).

Input: the per-sequence extract written by tools/kitscenes/kits_radar_extract.py (RAW/<uuid>/).

- Converted frame fj = original frame fi / stride (kitscenes_to_bevlane.py uses stride 2 and keeps the frame numbers).
- Points are given in each radar's sensor frame (x forward) -> calib T_to_reference -> ego frame (the reference
  frame raised by ground_offset_m, i.e. origin on the road surface).
- 0.5 s of sweeps (the current one plus the SWEEPS - 1 previous ones at the original 10 Hz) are motion-compensated
  into the current ego frame with poses.txt (TUM format, map-local) and stacked.
- Velocity: range_rate (rate of change of the range seen from the sensor, positive when moving away) plus the
  line-of-sight component of the ego velocity gives the ground-relative radial velocity vr_comp. The lever-arm
  velocity from the ego rotation is ignored (the radars are within 1 m of the reference frame).
- Output: OUT/<scene>/<fj:04d>.npz with pts float16 [N, 8] = x, y, z, rcs, range_rate, vr_comp, dt (s, negative for
  older sweeps), sensor (0 front / 1 left / 2 right). The scene manifests are not modified; training and evaluation
  read METEOR_RADAR_ROOT/<scene>/<fj>.npz (bevlane/radar_bev.py).

usage: kits_radar_to_bevlane.py --root KITS_ROOT --raw RAW_ROOT --out OUT [--scenes a,b | --all] [--sweeps 5]
           [--stride 2] [--workers 16] [--dry]
"""
import argparse
import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pyarrow.parquet as pq

RADARS = ("radar_front", "radar_left", "radar_right")


def quat_mat(qx, qy, qz, qw):
    return np.array([[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                     [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                     [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])


def poses_T(pz):
    T = np.tile(np.eye(4), (len(pz), 1, 1))
    for i, p in enumerate(pz):
        T[i, :3, :3] = quat_mat(*p[4:8]); T[i, :3, 3] = p[1:4]
    return T                                                        # map <- reference


def load_radar(raw, fi, calib, T_ego_ref):
    out = []
    for k, name in enumerate(RADARS):
        p = f"{raw}/{name}/{fi:010d}.parquet"
        if not os.path.isfile(p) or name not in calib:
            continue
        t = pq.read_table(p, columns=["x", "y", "z", "rcs", "range_rate"])
        xyz = np.stack([t["x"].to_numpy(), t["y"].to_numpy(), t["z"].to_numpy()], 1).astype(np.float64)
        if not len(xyz):
            continue
        T = T_ego_ref @ np.array(calib[name]["T_to_reference"], np.float64)            # ego <- sensor
        pe = xyz @ T[:3, :3].T + T[:3, 3]
        u = (pe - T[:3, 3]) / np.linalg.norm(pe - T[:3, 3], axis=1, keepdims=True).clip(min=1e-3)   # sensor -> point direction (ego)
        out.append((pe, t["rcs"].to_numpy().astype(np.float64), t["range_rate"].to_numpy().astype(np.float64), u, np.full(len(pe), k)))
    return out


def process(args):
    scene_dir, raw_root, out_root, sweeps, stride, dry = args
    try:
        man = json.load(open(f"{scene_dir}/manifest.json"))
        uid = man.get("kitscenes_uuid") or os.path.basename(scene_dir)[5:]
        raw = f"{raw_root}/{uid}"
        if not os.path.exists(f"{raw}/.ok"):
            return f"[noraw] {uid}"
        calib = json.load(open(f"{raw}/calibration/calib.json"))
        pz = np.loadtxt(f"{raw}/poses.txt", ndmin=2); ts = pz[:, 0]
        Tm = poses_T(pz)
        h = float(man.get("ground_offset_m", 0.0))
        T_ego_ref = np.eye(4); T_ego_ref[2, 3] = h
        # ego velocity in the reference frame (central difference)
        vel = np.zeros((len(pz), 3))
        for i in range(len(pz)):
            a, b = max(0, i - 1), min(len(pz) - 1, i + 1)
            if b > a:
                vw = (Tm[b, :3, 3] - Tm[a, :3, 3]) / max(ts[b] - ts[a], 1e-3)
                vel[i] = Tm[i, :3, :3].T @ vw
        cache = {}
        od = os.path.join(out_root, os.path.basename(scene_dir.rstrip("/")))
        os.makedirs(od, exist_ok=True)
        stats = [0, 0]
        for fr in man["frames"]:
            fj = int(fr["frame"]); fi = fj * stride
            if fi >= len(pz):
                continue
            rows = []
            for j in range(fi, max(-1, fi - sweeps), -1):
                if j not in cache:
                    cache[j] = load_radar(raw, j, calib, T_ego_ref)
                # ego frame of sweep j -> ego frame of the current frame fi (ego = reference + h)
                Tij = np.linalg.inv(Tm[fi]) @ Tm[j]
                Tij_e = T_ego_ref @ Tij @ np.linalg.inv(T_ego_ref)
                for pe, rcs, rr, u, sid in cache[j]:
                    vc = rr + u @ (vel[j])                                           # ground-relative radial velocity (sensor velocity ~ ego velocity)
                    p2 = pe @ Tij_e[:3, :3].T + Tij_e[:3, 3]
                    rows.append(np.column_stack([p2, rcs, rr, vc, np.full(len(pe), ts[j] - ts[fi]), sid]))
            P = np.concatenate(rows).astype(np.float16) if rows else np.zeros((0, 8), np.float16)
            stats[0] += 1; stats[1] += len(P)
            if not dry:
                tmp = f"{od}/{fj:04d}.tmp.npz"
                np.savez_compressed(tmp, pts=P); os.replace(tmp, f"{od}/{fj:04d}.npz")
        if not dry:
            json.dump({"sweeps": sweeps, "frames": stats[0], "fields": "x,y,z,rcs,range_rate,vr_comp,dt,sensor",
                       "sensor": list(RADARS)}, open(f"{od}/meta.json", "w"))
        return f"[ok] {uid} frames={stats[0]} pts/frame={stats[1] / max(stats[0], 1):.0f}"
    except Exception as e:
        return f"[fail] {scene_dir}: {type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="root of the converted scenes (KITS_<uuid>/manifest.json)")
    ap.add_argument("--raw", required=True, help="output directory of kits_radar_extract.py")
    ap.add_argument("--out", required=True, help="radar point root (use it as METEOR_RADAR_ROOT)")
    ap.add_argument("--scenes", default="", help="comma-separated scene names")
    ap.add_argument("--all", action="store_true", help="all KITS_* scenes under --root")
    ap.add_argument("--sweeps", type=int, default=5, help="number of 10 Hz sweeps to stack (5 = 0.5 s)")
    ap.add_argument("--stride", type=int, default=2, help="frame stride used by kitscenes_to_bevlane.py")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--dry", action="store_true", help="count points only, write nothing")
    a = ap.parse_args()
    dirs = sorted(glob.glob(f"{a.root}/KITS_*")) if a.all else [f"{a.root}/{s}" for s in a.scenes.split(",") if s]
    with ProcessPoolExecutor(a.workers) as ex:
        for i, r in enumerate(ex.map(process, [(d, a.raw, a.out, a.sweeps, a.stride, a.dry) for d in dirs])):
            if i % 50 == 0 or not r.startswith("[ok]"):
                print(f"{i + 1}/{len(dirs)} {r}", flush=True)
    print("RADAR_DONE", flush=True)


if __name__ == "__main__":
    main()

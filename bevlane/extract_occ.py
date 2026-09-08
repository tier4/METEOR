#!/usr/bin/env python3
"""3D semantic occupancy GT (ego frame) from accumulated labeled LiDAR.

Grid: x fwd [-40,40) x y left [-40,40) x z [-1.0,5.4) @ 0.4 m
      -> uint8 [Z=16, 200, 200], row=(40-x)/0.4, col=(40-y)/0.4.
Classes: 0 free, 1 obstacle/unknown, 2 vehicle, 3 two-wheeler, 4 pedestrian,
         5 road surface, 6 sidewalk, 7 vegetation, 8 building/wall,
         9 pole/sign/light, 255 unknown (never observed).

Recipe per output frame:
  1. label every strided keyframe's LiDAR once by sampling the CACHED
     seg2d21 maps (108x192, no RLE decode); scenes without seg2d21 are
     skipped (rolling dependency on extract_seg2d),
  2. accumulate labeled points from +-ACC neighbouring strided frames into
     the current ego frame via ego_pose,
  3. voxelize (majority class per voxel),
  4. carve free space by ray-stepping the CURRENT frame's returns,
  5. everything else stays 255 (unknown).
Saved as occ/<fi>.npz {"occ": uint8 [16,200,200]}; manifest key "occ".
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from autolabel_bev import Transform, quat_to_rot  # noqa: E402
from bevlane.extract_gt import OUT, ROOT, load_scene_light  # noqa: E402
from bevlane.extract_seg2d import CAMS  # noqa: E402

VOX = 0.4
XH = YH = 40.0
Z0, Z1 = -1.0, 5.4
GX = GY = int(2 * XH / VOX)      # 200
GZ = int((Z1 - Z0) / VOX)        # 16
ACC = 8                          # +- strided frames accumulated (~1.6 s)
DECIM = 3                        # point decimation (0.4 m voxels are coarse)
N_OCC = 10                       # 0 free + 9 occupied classes (255 = unknown)

# 21-class seg2d21 id -> OCC class (0 = unlabeled/drop). Index 0..20.
#  seg21: 0 bg,1 misc,2 car,3 truck,4 bus,5 moto,6 bicycle,7 ped,8 marking,
#         9 light,10 sign,11 road,12 sidewalk,13 lane,14 crosswalk,15 unused,
#         16 wall,17 building,18 vegetation,19 sky,20 pole
LUT21 = np.array([0, 1, 2, 2, 2, 3, 3, 4, 5, 9, 9, 5, 6, 5, 5, 0, 8, 8, 7,
                  0, 9], np.uint8)
OCC_PRIO = np.array([0, 3, 5, 6, 7, 1, 1, 2, 2, 4], np.int8)
OCC_PAL = np.array([[0, 0, 0], [244, 244, 244], [0, 0, 255], [119, 11, 32],
                    [220, 20, 60], [128, 64, 128], [244, 35, 232],
                    [107, 142, 35], [70, 70, 70], [220, 220, 0]], np.uint8)
OCC_NAMES = ["free", "obstacle", "vehicle", "2wheel", "pedestrian", "road",
             "sidewalk", "vegetation", "building", "pole/sign"]


def label_points(pts_ego, seg21, cams):
    """OCC class per point (0 = unlabeled) by sampling cached seg2d21 maps
    (uint8 [8,108,192], 255 = ignore). Higher OCC_PRIO wins across cameras."""
    cls = np.zeros(len(pts_ego), np.uint8)
    for ci in range(len(CAMS)):
        lab = seg21[ci]
        h4, w4 = lab.shape
        K4, T_cam_ego = cams[ci]
        pc = pts_ego @ T_cam_ego[:3, :3].T + T_cam_ego[:3, 3]
        z = pc[:, 2]
        m = z > 0.5
        u = (K4[0, 0] * pc[m, 0] / z[m] + K4[0, 2]).astype(np.int32)
        v = (K4[1, 1] * pc[m, 1] / z[m] + K4[1, 2]).astype(np.int32)
        ok = (u >= 0) & (u < w4) & (v >= 0) & (v < h4)
        s = np.full(int(m.sum()), 255, np.uint8)
        s[ok] = lab[v[ok], u[ok]]
        c = np.where(s == 255, 0, LUT21[np.minimum(s, 20)])
        cur = cls[m]
        upd = OCC_PRIO[c] > OCC_PRIO[cur]
        cur[upd] = c[upd]
        cls[m] = cur
    return cls


def pose_of(egop, frames, s):
    ld = frames[s["token"]].get("LIDAR_CONCAT")
    if ld is None:
        return None
    ep = egop[ld["ego_pose_token"]]
    R = quat_to_rot(ep["rotation"])
    return np.asarray(R), np.asarray(ep["translation"]), ld


def process_scene(args):
    scene, stride = args
    try:
        out_dir = os.path.join(OUT, scene)
        man = json.load(open(os.path.join(out_dir, "manifest.json")))
        sdir = os.path.join(ROOT, scene)
        ordered, frames, calib, egop = load_scene_light(sdir)
        strided = ordered[::stride]
        cams = []
        for ch in CAMS:
            K = np.array(man["cams"][ch]["K"])
            T_cam_ego = np.linalg.inv(np.array(man["cams"][ch]["T_ego_cam"]))
            cams.append((K / 4.0, T_cam_ego))
        by_fi = {fr["frame"]: fr for fr in man["frames"]}

        # pass 1: per strided frame, labeled+decimated points (ego) + pose
        labeled = {}
        for j, s in enumerate(strided):
            frj = by_fi.get(j)
            if frj is None or "seg2d21" not in frj:
                continue
            info = pose_of(egop, frames, s)
            if info is None:
                continue
            R, tr, ld = info
            try:
                seg21 = np.load(os.path.join(out_dir, frj["seg2d21"]))["seg"]
            except Exception:
                continue
            pts = np.fromfile(os.path.join(sdir, ld["filename"]),
                              dtype=np.float32).reshape(-1, 5)[:, :3]
            cal = calib[ld["calibrated_sensor_token"]]
            pe = Transform(cal["rotation"], cal["translation"]) \
                .apply(pts.astype(np.float64))[::DECIM]
            rng = np.hypot(pe[:, 0], pe[:, 1])
            keep = (rng > 2.5) & (rng < 75.0) \
                & (pe[:, 2] > Z0) & (pe[:, 2] < Z1 + 2)
            pe = pe[keep]
            cls = label_points(pe, seg21, cams)
            m = cls > 0
            labeled[j] = (pe[m].astype(np.float32), cls[m],
                          R.astype(np.float32), tr.astype(np.float32),
                          pe.astype(np.float32))     # all pts for carving

        os.makedirs(os.path.join(out_dir, "occ"), exist_ok=True)
        n_done = 0
        for fr in man["frames"]:
            fi = fr["frame"]
            path = os.path.join(out_dir, f"occ/{fi:04d}.npz")
            if os.path.exists(path):
                fr["occ"] = f"occ/{fi:04d}.npz"
                n_done += 1
                continue
            if fi not in labeled:
                continue
            _, _, R0, t0, rays = labeled[fi]
            allp, allc = [], []
            for j in range(max(0, fi - ACC), min(len(strided), fi + ACC + 1)):
                if j not in labeled:
                    continue
                p, c, R, tr, _ = labeled[j]
                if abs(j - fi) > 1:
                    # dynamic classes (vehicle/2wheel/pedestrian) smear when
                    # accumulated far in time -> keep them only from the
                    # current +-1 frames (<=0.4 s: sub-voxel smear, but 3x
                    # the points for sparse peds/bikes)
                    stat = (c != 2) & (c != 3) & (c != 4)
                    p, c = p[stat], c[stat]
                pw = p @ R.T + tr
                allp.append((pw - t0) @ R0)
                allc.append(c)
            if not allp:
                continue
            P = np.concatenate(allp)
            C = np.concatenate(allc)
            occ = np.full((GZ, GX, GY), 255, np.uint8)
            r = ((XH - P[:, 0]) / VOX).astype(np.int32)
            cco = ((YH - P[:, 1]) / VOX).astype(np.int32)
            zz = ((P[:, 2] - Z0) / VOX).astype(np.int32)
            ok = (r >= 0) & (r < GX) & (cco >= 0) & (cco < GY) \
                & (zz >= 0) & (zz < GZ)
            lin = (zz[ok] * GX + r[ok]) * GY + cco[ok]
            cnt = np.bincount(lin * N_OCC + C[ok],
                              minlength=GZ * GX * GY * N_OCC).reshape(-1, N_OCC)
            hit = cnt.sum(1) > 0
            occ.reshape(-1)[hit] = cnt[hit].argmax(1).astype(np.uint8)
            # free space: ray-step current frame's returns (subsampled)
            q = rays[::2]
            d = np.linalg.norm(q, axis=1)
            for tfrac in np.linspace(0.03, 0.97, 96):   # ~0.4m ray steps
                s_ = q[(tfrac * d) < (d - 0.6)] * tfrac
                rr = ((XH - s_[:, 0]) / VOX).astype(np.int32)
                cc2 = ((YH - s_[:, 1]) / VOX).astype(np.int32)
                zz2 = ((s_[:, 2] - Z0) / VOX).astype(np.int32)
                ok2 = (rr >= 0) & (rr < GX) & (cc2 >= 0) & (cc2 < GY) \
                    & (zz2 >= 0) & (zz2 < GZ)
                sel = occ[zz2[ok2], rr[ok2], cc2[ok2]]
                w = sel == 255                    # never overwrite occupied
                occ[zz2[ok2][w], rr[ok2][w], cc2[ok2][w]] = 0
            np.savez_compressed(path, occ=occ)
            fr["occ"] = f"occ/{fi:04d}.npz"
            n_done += 1
        json.dump(man, open(os.path.join(out_dir, "manifest.json"), "w"))
        return f"[ok] {scene} occ={n_done}/{len(man['frames'])}"
    except Exception as e:
        return f"[fail] {scene}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--scenes", default=None)
    args = ap.parse_args()
    if args.scenes:
        scenes = (open(args.scenes).read().split() if os.path.isfile(args.scenes)
                  else args.scenes.split(","))
    else:
        scenes = sorted(d for d in os.listdir(OUT)
                        if os.path.exists(os.path.join(OUT, d, "manifest.json")))
    print(f"{len(scenes)} scenes; occ {GZ}x{GX}x{GY} @{VOX}m", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(process_scene,
                                     [(s, args.stride) for s in scenes])):
            if i % 20 == 0 or not r.startswith("[ok"):
                print(f"{i + 1}/{len(scenes)} {r}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

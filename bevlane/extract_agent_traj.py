#!/usr/bin/env python3
"""Agent-trajectory GT: camera-confirmed 3D boxes + per-instance futures.

For every frame, the SAME boxes as extract_bev_box (class, ego-frame pose,
size, yaw; occluded ones dropped via 2D confirmation) plus, per box, the
instance's future positions at +0.5..+3.0 s (tracked by instance_token
through sample_annotation, transformed into the CURRENT ego frame).

agent_traj/<fi>.npz:
  boxes  float32 [K,6]   (cls, xe, ye, l, w, yaw)   K = KMAX padded
  count  int64
  traj   float32 [K,6,2] future (xe,ye) OFFSETS from the box centre
  tvalid float32 [K,6]   1 where the instance exists at that horizon
Manifest frames gain "agent_traj".
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
from autolabel_bev import quat_to_rot  # noqa: E402
from bevlane.extract_bev_box import (CAMS, box_corners_ego,  # noqa: E402
                                     load_boxes, visible_in_any_cam)
from bevlane.extract_gt import OUT, ROOT, load_scene_light  # noqa: E402

KMAX = 64
HORIZON = 6
RAW_STEP = 5                 # keyframes @10 Hz -> 5 frames = 0.5 s
BEV_XH, BEV_YH = 80.0, 50.0


def process_scene(args):
    scene, stride = args
    try:
        out_dir = os.path.join(OUT, scene)
        man = json.load(open(os.path.join(out_dir, "manifest.json")))
        sdir = os.path.join(ROOT, scene)
        ordered, frames, _, egop = load_scene_light(sdir)
        boxes_by_sample, ann2d_by_sd, sd_size = load_boxes(sdir)
        # instance -> map position per raw sample index (for future lookup)
        pos_by_raw = []
        for s in ordered:
            pos_by_raw.append({inst: tr for inst, _, tr, _, _ in
                               boxes_by_sample.get(s["token"], [])})
        cams = {}
        for ch in CAMS:
            if ch in man["cams"]:
                cams[ch] = (np.array(man["cams"][ch]["K"]),
                            np.linalg.inv(np.array(man["cams"][ch]["T_ego_cam"])))
        os.makedirs(os.path.join(out_dir, "agent_traj"), exist_ok=True)
        for fr in man["frames"]:
            fi = fr["frame"]
            path = os.path.join(out_dir, f"agent_traj/{fi:04d}.npz")
            if os.path.exists(path):
                fr["agent_traj"] = f"agent_traj/{fi:04d}.npz"
                continue
            ri = fi * stride
            s = ordered[ri]
            frame = frames[s["token"]]
            ld = frame.get("LIDAR_CONCAT")
            if ld is None:
                continue
            ep = egop[ld["ego_pose_token"]]
            te = np.array(ep["translation"])
            Re = quat_to_rot(ep["rotation"])
            yaw_e = np.arctan2(Re[1, 0], Re[0, 0])
            c, sn = np.cos(-yaw_e), np.sin(-yaw_e)

            def to_ego(tr):
                dx, dy = tr[0] - te[0], tr[1] - te[1]
                return c * dx - sn * dy, sn * dx + c * dy

            B = np.zeros((KMAX, 6), np.float32)
            T = np.zeros((KMAX, HORIZON, 2), np.float32)
            V = np.zeros((KMAX, HORIZON), np.float32)
            # nearest agents first (crowded scenes exceed KMAX)
            cand = sorted(boxes_by_sample.get(s["token"], []),
                          key=lambda a: (a[2][0] - te[0]) ** 2
                          + (a[2][1] - te[1]) ** 2)
            k = 0
            for inst, cls, tr, size, rot in cand:
                if k >= KMAX:
                    break
                corners = box_corners_ego(tr, size, rot, te, Re)
                # VRUs have tiny 2D boxes; projection error dominates the
                # overlap check -> relax their confirmation threshold
                vis_th = 0.15 if cls >= 1.5 else 0.3
                if not visible_in_any_cam(corners, cls, frame, cams,
                                          ann2d_by_sd, sd_size,
                                          thresh=vis_th):
                    continue
                xe, ye = to_ego(tr)
                if not (-BEV_XH - 5 < xe < BEV_XH + 5
                        and -BEV_YH - 5 < ye < BEV_YH + 5):
                    continue
                Rb = quat_to_rot(rot)
                yaw_b = np.arctan2(Rb[1, 0], Rb[0, 0]) - yaw_e
                B[k] = (cls, xe, ye, size[1], size[0], yaw_b)
                for h in range(HORIZON):
                    rj = ri + RAW_STEP * (h + 1)
                    if rj >= len(ordered):
                        break
                    trf = pos_by_raw[rj].get(inst)
                    if trf is None:
                        continue
                    fxe, fye = to_ego(trf)
                    T[k, h] = (fxe - xe, fye - ye)
                    V[k, h] = 1.0
                k += 1
            np.savez_compressed(path, boxes=B, count=np.int64(k),
                                traj=T, tvalid=V)
            fr["agent_traj"] = f"agent_traj/{fi:04d}.npz"
        json.dump(man, open(os.path.join(out_dir, "manifest.json"), "w"))
        return f"[ok] {scene}"
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
    print(f"{len(scenes)} scenes; agent_traj {HORIZON}wp @0.5s", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(process_scene,
                                     [(s, args.stride) for s in scenes])):
            if i % 100 == 0 or not r.startswith("[ok"):
                print(f"{i + 1}/{len(scenes)} {r}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

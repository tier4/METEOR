#!/usr/bin/env python3
"""E2E ego-motion GT from ego_pose (no CAN in these datasets).

Per manifest frame:
  wp    [6,2]  future ego positions at +0.5..+3.0 s in the CURRENT ego frame
               (x fwd, y left, metres) - trajectory target
  v0    [1]    current speed (m/s) - model INPUT (E2E standard conditioning)
  acc   [1]    longitudinal acceleration (m/s^2, smoothed +-0.5 s)
  steer [1]    bicycle-model steering angle (rad):
               atan(WHEELBASE * yaw_rate / v); 0 & masked when v < 0.5 m/s
  brake [1]    binary: acc < -0.5 m/s^2
  valid [1]    1 when full 3 s of future poses exist (scene end -> 0)
Saved once per scene as ego_motion.npz (arrays indexed by manifest frame
order); manifest gains a scene-level "ego_motion" key.
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
from bevlane.extract_gt import OUT, ROOT, load_scene_light  # noqa: E402

WHEELBASE = 2.8          # [m] bicycle-model assumption (no CAN available)
HORIZON = 6              # waypoints
WP_DT = 0.5              # [s] waypoint spacing
V_MIN_STEER = 0.5        # [m/s] steering undefined below this speed


def pose_track(ordered, frames, egop):
    """(t [s], x, y, yaw) per keyframe from the LiDAR ego pose."""
    ts, xs, ys, yaws = [], [], [], []
    for s in ordered:
        ld = frames.get(s["token"], {}).get("LIDAR_CONCAT")
        if ld is None:
            continue
        ep = egop[ld["ego_pose_token"]]
        R = quat_to_rot(ep["rotation"])
        ts.append(ld["timestamp"] * 1e-6)
        xs.append(ep["translation"][0])
        ys.append(ep["translation"][1])
        yaws.append(np.arctan2(R[1, 0], R[0, 0]))
    return (np.array(ts), np.array(xs), np.array(ys),
            np.unwrap(np.array(yaws)))


def process_scene(args):
    scene, stride = args
    try:
        out_dir = os.path.join(OUT, scene)
        mf = os.path.join(out_dir, "manifest.json")
        man = json.load(open(mf))
        ordered, frames, _, egop = load_scene_light(os.path.join(ROOT, scene))
        t, x, y, yaw = pose_track(ordered, frames, egop)
        if len(t) < 5:
            return f"[skip] {scene}: too few poses"
        # smoothed speed / accel / yaw-rate on the raw 10 Hz track
        dt = np.gradient(t)
        vx, vy = np.gradient(x) / dt, np.gradient(y) / dt
        v = np.hypot(vx, vy)
        k = 5                                        # ~0.5 s box smoothing
        ker = np.ones(k) / k
        vs = np.convolve(v, ker, "same")
        acc = np.convolve(np.gradient(vs) / dt, ker, "same")
        yr = np.convolve(np.gradient(yaw) / dt, ker, "same")

        strided = ordered[::stride]
        tok2raw = {s["token"]: i for i, s in enumerate(ordered)
                   if frames.get(s["token"], {}).get("LIDAR_CONCAT")}
        # map raw index over the pose arrays (they skip pose-less samples)
        pose_idx = {}
        pi = 0
        for i, s in enumerate(ordered):
            if frames.get(s["token"], {}).get("LIDAR_CONCAT"):
                pose_idx[i] = pi
                pi += 1

        F = 1 + max(fr["frame"] for fr in man["frames"])   # index by frame id
        wp = np.zeros((F, HORIZON, 2), np.float32)
        v0 = np.zeros(F, np.float32)
        ac = np.zeros(F, np.float32)
        st = np.zeros(F, np.float32)
        br = np.zeros(F, np.float32)
        vd = np.zeros(F, np.float32)
        for fr in man["frames"]:
            fj = fr["frame"]
            s = strided[fj]
            ri = tok2raw.get(s["token"])
            if ri is None or ri not in pose_idx:
                continue
            i = pose_idx[ri]
            # instantaneous signals need no future -> write for EVERY frame
            # (v0 conditions the model at inference; a bogus 0 at scene tails
            # makes it predict "hold" at speed)
            v0[fj] = vs[i]
            ac[fj] = acc[i]
            st[fj] = (np.arctan(WHEELBASE * yr[i] / vs[i])
                      if vs[i] > V_MIN_STEER else 0.0)
            br[fj] = float(acc[i] < -0.5)
            t0 = t[i]
            tq = t0 + WP_DT * np.arange(1, HORIZON + 1)
            if tq[-1] > t[-1]:                       # not enough future:
                continue                             # wp stays 0, valid=0
            xq = np.interp(tq, t, x)
            yq = np.interp(tq, t, y)
            c, sn = np.cos(yaw[i]), np.sin(yaw[i])
            dx, dy = xq - x[i], yq - y[i]
            wp[fj, :, 0] = c * dx + sn * dy          # fwd
            wp[fj, :, 1] = -sn * dx + c * dy         # left
            vd[fj] = 1.0
        np.savez_compressed(os.path.join(out_dir, "ego_motion.npz"),
                            wp=wp, v0=v0, acc=ac, steer=st, brake=br, valid=vd)
        man["ego_motion"] = "ego_motion.npz"
        json.dump(man, open(mf, "w"))
        return f"[ok] {scene} valid={int(vd.sum())}/{F}"
    except Exception as e:
        return f"[fail] {scene}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--scenes", default=None)
    args = ap.parse_args()
    if args.scenes:
        scenes = (open(args.scenes).read().split() if os.path.isfile(args.scenes)
                  else args.scenes.split(","))
    else:
        scenes = sorted(d for d in os.listdir(OUT)
                        if os.path.exists(os.path.join(OUT, d, "manifest.json")))
    print(f"{len(scenes)} scenes; ego {HORIZON}wp @ {WP_DT}s", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(process_scene,
                                     [(s, args.stride) for s in scenes])):
            if i % 100 == 0 or not r.startswith("[ok"):
                print(f"{i + 1}/{len(scenes)} {r}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

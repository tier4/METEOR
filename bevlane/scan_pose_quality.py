#!/usr/bin/env python3
"""Flag scenes with broken ego localization before/after conversion.

Three failure modes:
  - TELEPORT: NDT/GNSS pose jumps (consecutive high-rate poses metres apart) ->
    a smeared, unusable BEV. maxstep > JUMP_M is unambiguous (>>physically
    possible at the pose rate). Excluded.
  - YAW-BIAS: a *consistent* signed offset between pose yaw and travel direction
    (|median signed| > BIAS_DEG). Turning noise averages to ~0, so a non-zero
    signed median means the heading is miscalibrated/drifted and the whole scene's
    ego-frame GT (BEV, box yaw, trajectory) is rotated by that angle. Excluded.
    (Use the SIGNED median, not |mean|: a ~10 deg constant bias hides under a 20 deg
    absolute-error threshold but stands out in the signed statistic.)
  - DRIFT: large median |heading error| with no jump/bias (variable, e.g. heavy
    low-speed maneuvering) -> mild ghosting. Watch-list, not auto-excluded.

Reads the (convention-corrected) poses from BEVLANE_ROOT so the r7/Vr +90 fix is
not re-flagged as drift. Prints two lists; writes the teleport list to --out.
"""
import argparse
import json
import os

import numpy as np

ROOT = os.environ.get("BEVLANE_ROOT",
                      "/data6/dataset/transfer_pp/group2_meteor/allroot")
JUMP_M = 8.0        # consecutive-pose step above this = teleport
BIAS_DEG = 6.0      # |median SIGNED heading offset| above this = constant yaw bias
DRIFT_DEG = 20.0    # median |heading-vs-motion| above this = drift suspect


def yaw_of(q):
    w, x, y, z = q
    return np.arctan2(2 * (x * y + z * w), 1 - 2 * (y * y + z * z))


def scan(scene):
    ep = json.load(open(os.path.join(ROOT, scene, "annotation", "ego_pose.json")))
    ep.sort(key=lambda e: e["timestamp"])
    tr = np.array([e["translation"][:2] for e in ep])
    t = np.array([e["timestamp"] for e in ep]) / 1e6
    step = np.linalg.norm(np.diff(tr, axis=0), axis=1)
    maxstep = float(step.max()) if len(step) else 0.0
    njump = int((step > JUMP_M).sum())
    sg = []
    for i in range(1, len(ep) - 1):
        d = tr[i + 1] - tr[i - 1]
        dt = t[i + 1] - t[i - 1]
        sp = np.linalg.norm(d) / dt if dt > 0 else 0
        if 2 < sp < 40:
            sg.append(np.degrees((np.arctan2(d[1], d[0]) - yaw_of(ep[i]["rotation"])
                                  + np.pi) % (2 * np.pi) - np.pi))
    hm = float(np.median(np.abs(sg))) if len(sg) >= 20 else None
    # constant yaw bias: a consistent SIGNED offset (turning noise averages to ~0,
    # a miscalibrated/drifted heading does not) -> the whole scene's BEV is rotated.
    bias = float(np.median(sg)) if len(sg) >= 30 else None
    return njump, maxstep, hm, bias


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--out", help="write teleport-broken scene list here")
    args = ap.parse_args()
    scenes = [s for s in open(args.scenes).read().split() if s]
    teleport, biased, drift = [], [], []
    for sc in scenes:
        nj, ms, hm, bias = scan(sc)
        if nj >= 1:
            teleport.append((sc, nj, ms, hm))
        elif bias is not None and abs(bias) > BIAS_DEG:
            biased.append((sc, ms, bias))
        elif hm is not None and hm > DRIFT_DEG:
            drift.append((sc, ms, hm))
    print(f"TELEPORT-broken (maxstep>{JUMP_M}m): {len(teleport)}")
    for sc, nj, ms, hm in sorted(teleport, key=lambda r: -r[2]):
        print(f"  {sc} jumps={nj} maxstep={ms:.1f}m headErr={hm and round(hm, 1)}")
    print(f"YAW-BIASED (|signed heading offset|>{BIAS_DEG}deg): {len(biased)}")
    for sc, ms, b in sorted(biased, key=lambda r: -abs(r[2])):
        print(f"  {sc} signed={b:+.1f}deg")
    print(f"DRIFT-suspect (headErr>{DRIFT_DEG}deg, no jump/bias): {len(drift)}")
    for sc, ms, hm in sorted(drift, key=lambda r: -r[2]):
        print(f"  {sc} maxstep={ms:.1f}m headErr={round(hm, 1)}")
    if args.out:
        # teleport + constant-bias are both unusable -> exclude
        open(args.out, "w").write("\n".join([r[0] for r in teleport]
                                            + [r[0] for r in biased]) + "\n")


if __name__ == "__main__":
    main()

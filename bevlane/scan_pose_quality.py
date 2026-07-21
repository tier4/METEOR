#!/usr/bin/env python3
"""Flag scenes with broken ego localization before/after conversion.

Two failure modes, both fatal to the map accumulation:
  - TELEPORT: NDT/GNSS pose jumps (consecutive high-rate poses metres apart) ->
    a smeared, unusable BEV. maxstep > JUMP_M is unambiguous (>>physically
    possible at the pose rate). This is the "completely broken" class.
  - DRIFT: no teleport, but pose yaw disagrees with the direction of travel by a
    large median angle over well-moving frames -> ghosted lane lines. Milder;
    reported as a watch-list, not auto-excluded.

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
DRIFT_DEG = 20.0    # median heading-vs-motion error above this = suspect


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
    er = []
    for i in range(1, len(ep) - 1):
        d = tr[i + 1] - tr[i - 1]
        dt = t[i + 1] - t[i - 1]
        sp = np.linalg.norm(d) / dt if dt > 0 else 0
        if 2 < sp < 40:
            er.append(abs(np.degrees((np.arctan2(d[1], d[0]) - yaw_of(ep[i]["rotation"])
                                      + np.pi) % (2 * np.pi) - np.pi)))
    hm = float(np.median(er)) if len(er) >= 20 else None
    return njump, maxstep, hm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--out", help="write teleport-broken scene list here")
    args = ap.parse_args()
    scenes = [s for s in open(args.scenes).read().split() if s]
    teleport, drift = [], []
    for sc in scenes:
        nj, ms, hm = scan(sc)
        if nj >= 1:
            teleport.append((sc, nj, ms, hm))
        elif hm is not None and hm > DRIFT_DEG:
            drift.append((sc, ms, hm))
    print(f"TELEPORT-broken (maxstep>{JUMP_M}m): {len(teleport)}")
    for sc, nj, ms, hm in sorted(teleport, key=lambda r: -r[2]):
        print(f"  {sc} jumps={nj} maxstep={ms:.1f}m headErr={hm and round(hm, 1)}")
    print(f"DRIFT-suspect (headErr>{DRIFT_DEG}deg, no jump): {len(drift)}")
    for sc, ms, hm in sorted(drift, key=lambda r: -r[2]):
        print(f"  {sc} maxstep={ms:.1f}m headErr={round(hm, 1)}")
    if args.out:
        open(args.out, "w").write("\n".join(r[0] for r in teleport) + "\n")


if __name__ == "__main__":
    main()

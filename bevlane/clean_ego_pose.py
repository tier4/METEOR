#!/usr/bin/env python3
"""Data-cleansing: fix the r7ZRDFWf (2024) / VrJnyZb4 ego-pose convention.

These recordings express the ego *orientation* with the vehicle forward axis
along body +y instead of the standard +x, so the pose yaw sits a constant +90 deg
off the true heading (verified: median heading-vs-motion error 89.4 deg raw ->
0.73 deg after a body-frame +90 deg yaw correction; healthy scenes sit at ~1.7).
Left uncorrected the LiDAR map accumulation and every ego-frame crop are rotated
90 deg, so all GT for moving scenes is garbage.

Non-destructive: the source tree is never modified. For each scene we build a
clean mirror under OUTROOT/<scene> whose annotation/ symlinks every original json
except ego_pose.json, which is rewritten with the corrected rotation. data/ and
the rest are symlinked through. The allroot entry METEOR reads from is repointed
to the mirror.
"""
import argparse
import json
import os
import re

import numpy as np

SRC = "/data6/dataset/transfer_pp/group2"
OUTROOT = "/data6/dataset/transfer_pp/group2_clean"
ALLROOT = "/data6/dataset/transfer_pp/group2_meteor/allroot"
# body-frame +90 deg about z, quaternion [w,x,y,z]
QZ90 = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])


def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2]


def yaw_of(q):
    w, x, y, z = q
    return np.arctan2(2 * (x * y + z * w), 1 - 2 * (y * y + z * z))


def measure_offset(scene):
    """Median (motion-direction - pose-yaw) over moving frames, in degrees, or
    None when the scene never moves fast enough to measure."""
    ep = json.load(open(os.path.join(SRC, scene, "annotation", "ego_pose.json")))
    ep.sort(key=lambda e: e["timestamp"])
    tr = np.array([e["translation"][:2] for e in ep])
    t = np.array([e["timestamp"] for e in ep]) / 1e6
    off = []
    for i in range(1, len(ep) - 1):
        d = tr[i + 1] - tr[i - 1]
        dt = t[i + 1] - t[i - 1]
        if dt <= 0 or np.linalg.norm(d) / dt < 2.0:
            continue
        off.append(np.degrees((np.arctan2(d[1], d[0]) - yaw_of(ep[i]["rotation"])
                               + np.pi) % (2 * np.pi) - np.pi))
    return (float(np.median(off)), len(off)) if off else (None, 0)


def needed_correction_deg(scene, session_hint=0.0):
    """Per-scene yaw correction, rounded to a multiple of 90 deg. Scenes with a
    measurable offset drive their own correction; unmeasurable (stationary)
    scenes fall back to session_hint (the correction of moving scenes recorded in
    the same session)."""
    med, n = measure_offset(scene)
    off = session_hint if med is None else med
    return round(off / 90.0) * 90.0


def clean_scene(scene, correction_deg):
    src = os.path.join(SRC, scene)
    dst = os.path.join(OUTROOT, scene)
    ann_dst = os.path.join(dst, "annotation")
    os.makedirs(ann_dst, exist_ok=True)

    # top-level entries: symlink everything except annotation (rebuilt below)
    for name in os.listdir(src):
        if name == "annotation":
            continue
        link = os.path.join(dst, name)
        if not os.path.lexists(link):
            os.symlink(os.path.join(src, name), link)

    # annotation: symlink every json but ego_pose.json
    for name in os.listdir(os.path.join(src, "annotation")):
        if name == "ego_pose.json":
            continue
        link = os.path.join(ann_dst, name)
        if not os.path.lexists(link):
            os.symlink(os.path.join(src, "annotation", name), link)

    # body-frame yaw correction, rounded to a multiple of 90 deg
    k = int(round(correction_deg / 90.0)) % 4
    qcorr = [np.cos(k * np.pi / 4), 0.0, 0.0, np.sin(k * np.pi / 4)]
    ep = json.load(open(os.path.join(src, "annotation", "ego_pose.json")))
    for e in ep:
        e["rotation"] = qmul(e["rotation"], qcorr)
    json.dump(ep, open(os.path.join(ann_dst, "ego_pose.json"), "w"))

    # repoint the root METEOR reads from
    link = os.path.join(ALLROOT, scene)
    if os.path.lexists(link):
        os.remove(link)
    os.symlink(dst, link)
    return len(ep), k * 90


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", required=True, help="file, one scene per line")
    args = ap.parse_args()
    scenes = [s for s in open(args.scenes).read().split() if s]
    os.makedirs(OUTROOT, exist_ok=True)

    # session hint: for stationary (unmeasurable) scenes, inherit the correction
    # of moving scenes recorded in the same session (scene.json name up to the
    # trailing "_NNN" index).
    def session(sc):
        # group by recording = everything up to and including the tz offset,
        # dropping the trailing segment/clip indices (r7: "_034"; Vr: "_91_0")
        n = json.load(open(os.path.join(SRC, sc, "annotation",
                                        "scene.json")))[0]["name"]
        m = re.search(r"[+-]\d{4}", n)
        return n[:m.end()] if m else n.rsplit("_", 1)[0]

    sess_off = {}
    for sc in scenes:
        med, _ = measure_offset(sc)
        if med is not None:
            sess_off.setdefault(session(sc), []).append(med)
    sess_corr = {s: round(float(np.median(v)) / 90.0) * 90.0
                 for s, v in sess_off.items()}

    corrected = skipped = 0
    for i, sc in enumerate(scenes):
        hint = sess_corr.get(session(sc), 0.0)
        cdeg = needed_correction_deg(sc, hint)
        if cdeg % 360 == 0:
            # already correct in raw form: point straight at the source, no mirror
            link = os.path.join(ALLROOT, sc)
            if os.path.lexists(link):
                os.remove(link)
            os.symlink(os.path.join(SRC, sc), link)
            skipped += 1
            print(f"{i + 1}/{len(scenes)} {sc} already-OK -> raw (no correction)",
                  flush=True)
            continue
        n, applied = clean_scene(sc, cdeg)
        corrected += 1
        print(f"{i + 1}/{len(scenes)} cleaned {sc} ({n} poses, +{applied} deg)",
              flush=True)
    print(f"DONE corrected={corrected} already-ok={skipped}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Scan scenes for autolabel suitability.

Outputs per scene: travel distance, median speed, max pose jump, sky-annotation
count (indoor/garage detection -> broken localization), verdict.
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = "/data1/dataset/aisin/converted_valid_delay"
CHUNK = 10 * 1024 * 1024


def scan_one(name):
    ann = os.path.join(ROOT, name, "annotation")
    try:
        ep = json.load(open(os.path.join(ann, "ego_pose.json")))
        cats = {c["name"]: c["token"] for c in
                json.load(open(os.path.join(ann, "category.json")))}
    except Exception:
        return None
    ep.sort(key=lambda e: e["timestamp"])
    t = np.array([e["timestamp"] for e in ep]) / 1e6
    tr = np.array([e["translation"][:2] for e in ep])
    idx = np.searchsorted(t, np.arange(t[0], t[-1], 1.0))
    steps = np.linalg.norm(np.diff(tr[idx], axis=0), axis=1)
    dist = float(steps.sum())
    med = float(np.median(steps)) if len(steps) else 0.0
    mx = float(steps.max()) if len(steps) else 0.0

    sky = 0
    tok = cats.get("sky")
    if tok:
        try:
            with open(os.path.join(ann, "surface_ann.json"), "rb") as f:
                size = os.fstat(f.fileno()).st_size
                sky += f.read(CHUNK).count(tok.encode())
                if size > 2 * CHUNK:
                    f.seek(size - CHUNK)
                    sky += f.read(CHUNK).count(tok.encode())
        except Exception:
            sky = -1

    outdoor = sky >= 300
    clean = mx < 30.0
    moving = med >= 1.0
    ok = outdoor and clean and moving
    return (name, dist, med, mx, sky, ok)


def main():
    scenes = sorted(d for d in os.listdir(ROOT)
                    if os.path.isdir(os.path.join(ROOT, d)) and d.startswith("Pct"))
    if len(sys.argv) > 1:
        scenes = [s for s in scenes if sys.argv[1] in s]
    rows = []
    with ProcessPoolExecutor(max_workers=16) as ex:
        for r in ex.map(scan_one, scenes, chunksize=4):
            if r:
                rows.append(r)
                print(f"{r[0]:55s} {r[1]:8.1f} {r[2]:7.2f} {r[3]:8.1f} {r[4]:6d} {'OK' if r[5] else '--'}",
                      flush=True)
    nok = sum(r[5] for r in rows)
    print(f"# {len(rows)} scenes, {nok} OK (outdoor, clean poses, moving)")


if __name__ == "__main__":
    main()

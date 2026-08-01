#!/usr/bin/env python3
"""Incremental ingest of the growing US batch into out/bevlane.

Idempotent: run it again whenever more scenes finish converting and it only
processes what is new. Steps per new scene:
  1. symlink  /data1/dataset/bevlane/us_1/out/bevlane/<scene> -> out/bevlane/
  2. gt_cons (consensus GT) -- rounds train with --gt-key gt_cons, so a scene
     without it contributes nothing
  3. quality gates, both learned the hard way:
       * moth-eaten gt_cons (road retention < 50% of gt) -> reject
       * frames whose imgs dict misses a camera (one such frame killed the
         whole 8-GPU r47 run with a KeyError) -> scene kept, frame dropped by
         the dataset filter; scenes with many such frames are rejected
  4. write out/us_ingested.txt (all accepted) and out/round48_scenes.txt
     (= round45 + accepted US), leaving the list the running round uses alone
"""
import argparse
import json
import os
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bevlane.dataset import CAMS                              # noqa: E402

SRC = "/data1/dataset/bevlane/us_1/out"
OUT = "out/bevlane"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--base-list", default="out/round45_scenes.txt")
    ap.add_argument("--round-list", default="out/round48_scenes.txt")
    a = ap.parse_args()

    done_f = os.path.join(a.src, "us_done_scenes.txt")
    done = [l.strip() for l in open(done_f) if l.strip()]
    src_bev = os.path.join(a.src, "bevlane")
    bad_prev = set()
    if os.path.exists("out/us1_bad_scenes.txt"):
        bad_prev = {l.strip() for l in open("out/us1_bad_scenes.txt")
                    if l.strip()}

    # ---- 1. symlink what is new -------------------------------------
    new = []
    for s in done:
        if s in bad_prev:
            continue
        sp = os.path.join(src_bev, s)
        dp = os.path.join(OUT, s)
        if not os.path.isdir(sp) or os.path.exists(dp):
            continue
        os.symlink(os.path.abspath(sp), dp)
        new.append(s)
    print(f"done-list {len(done)} | newly symlinked {len(new)}", flush=True)

    # ---- 2. consensus GT for the new ones ---------------------------
    if new:
        lst = "out/us_new_scenes.txt"
        open(lst, "w").write("\n".join(new) + "\n")
        subprocess.run([sys.executable, "bevlane/make_consensus_gt.py",
                        "--scenes", lst, "--workers", str(a.workers)],
                       check=False)

    # ---- 3. quality gates over every US scene present ---------------
    present = [s for s in done if os.path.isdir(os.path.join(OUT, s))]
    ok, rej = [], []
    for s in present:
        mp = os.path.join(OUT, s, "manifest.json")
        try:
            man = json.load(open(mp))
        except Exception:
            rej.append((s, "no manifest")); continue
        fr = [f for f in man["frames"] if "gt_cons" in f]
        if len(fr) < 20:
            rej.append((s, f"gt_cons frames {len(fr)}")); continue
        incomplete = sum(1 for f in man["frames"]
                         if set(CAMS) - set(f.get("imgs", {})))
        if incomplete > 0.1 * len(man["frames"]):
            rej.append((s, f"{incomplete} camera-incomplete frames")); continue
        f = fr[len(fr) // 2]
        g = cv2.imread(os.path.join(OUT, s, f["gt"]), 0)
        gc = cv2.imread(os.path.join(OUT, s, f["gt_cons"]), 0)
        if g is None or gc is None:
            rej.append((s, "unreadable GT")); continue
        r0, r1 = float((g == 1).mean()), float((gc == 1).mean())
        if r0 > 0.02 and r1 / max(r0, 1e-6) < 0.5:
            rej.append((s, f"moth-eaten gt_cons {r1 / r0:.2f}")); continue
        ok.append(s)
    print(f"accepted {len(ok)} | rejected {len(rej)}", flush=True)
    for s, why in rej[:8]:
        print(f"  reject {s[:20]} {why}", flush=True)

    open("out/us_ingested.txt", "w").write("\n".join(sorted(ok)) + "\n")
    base = [l.strip() for l in open(a.base_list) if l.strip()]
    merged = base + [s for s in sorted(ok) if s not in set(base)]
    open(a.round_list, "w").write("\n".join(merged) + "\n")
    print(f"{a.round_list}: {len(merged)} scenes "
          f"(+{len(merged) - len(base)} US vs {a.base_list})", flush=True)


if __name__ == "__main__":
    main()

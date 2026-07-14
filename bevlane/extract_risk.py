#!/usr/bin/env python3
"""Area risk-map GT extraction (definition: bevlane/risk_field.py).

Per scene: for every manifest frame, combine agent_traj GT (dynamic lobes)
and occupancy GT (static falloff) into the approved potential-field risk
map. Saved as risk_map.npz {risk [F,400,250] uint8 (x255)}; manifest gains
a scene-level "risk_map" key.
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
from bevlane.extract_gt import OUT  # noqa: E402
from bevlane.risk_field import RH, RW, risk_field  # noqa: E402


def process_scene(args):
    scene, _stride = args
    try:
        out_dir = os.path.join(OUT, scene)
        mf = os.path.join(out_dir, "manifest.json")
        man = json.load(open(mf))
        F = 1 + max(fr["frame"] for fr in man["frames"])
        risk = np.zeros((F, RH, RW), np.uint8)
        n_dyn = 0
        for fr in man["frames"]:
            fi = fr["frame"]
            boxes = np.zeros((64, 6), np.float32)
            count = 0
            traj = np.zeros((64, 6, 2), np.float32)
            tvalid = np.zeros((64, 6), np.float32)
            atp = os.path.join(out_dir, fr.get("agent_traj", "_"))
            if os.path.exists(atp):
                z = np.load(atp)
                boxes, count = z["boxes"], int(z["count"])
                traj, tvalid = z["traj"], z["tvalid"]
                n_dyn += count
            occ_arr = None
            if fr.get("occ"):
                try:
                    occ_arr = np.load(os.path.join(out_dir, fr["occ"]))["occ"]
                except Exception:
                    pass
            r = risk_field(boxes, count, traj, tvalid, occ_arr)
            risk[fi] = (np.clip(r, 0, 1) * 255).astype(np.uint8)
        np.savez_compressed(os.path.join(out_dir, "risk_map.npz"), risk=risk)
        man["risk_map"] = "risk_map.npz"
        json.dump(man, open(mf, "w"))
        return f"[ok] {scene} agents={n_dyn}"
    except Exception as e:
        return f"[fail] {scene}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--scenes", default=None)
    args = ap.parse_args()
    if args.scenes:
        scenes = (open(args.scenes).read().split() if os.path.isfile(args.scenes)
                  else args.scenes.split(","))
    else:
        scenes = sorted(d for d in os.listdir(OUT)
                        if os.path.exists(os.path.join(OUT, d, "manifest.json")))
    print(f"{len(scenes)} scenes; area risk map GT", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(process_scene,
                                     [(s, args.stride) for s in scenes])):
            if i % 100 == 0 or not r.startswith("[ok"):
                print(f"{i + 1}/{len(scenes)} {r}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

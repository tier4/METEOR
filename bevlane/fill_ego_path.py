#!/usr/bin/env python3
"""Post-process fix: stamp ROAD under the ego path in the accumulated BEV.

Near the ego the LiDAR is blind (< ~1.4 m) and the ground just beyond projects
onto the hood / into the nadir gap between the outward cameras, so the driven
strip gets no panoptic label and reads as a hole in the BEV road — severe wherever
the ego dwells (lights, congestion, parking). The ego is by definition on drivable
surface, so we stamp ROAD into UNLABELED cells within half a lane of the trajectory.

Patches `bev_label_masked.npy` (+ the PNGs) in place for the given scenes. Run
`render_vector_gt` + `extract_gt` + `annotate_gtcov` afterwards to propagate.
Equivalent to the in-accumulation fill now in autolabel_bev.process_scene; this path
avoids re-running the expensive autolabel on already-converted scenes.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from autolabel_bev import EGO_FILL_R, PALETTE, ROAD, UNLABELED  # noqa: E402

PROD = "out/production"
ROOT = os.environ.get("BEVLANE_ROOT",
                      "/data6/dataset/transfer_pp/group2_meteor/allroot")


def load_traj(scene):
    ep = json.load(open(os.path.join(ROOT, scene, "annotation", "ego_pose.json")))
    ep.sort(key=lambda e: e["timestamp"])
    return np.array([e["translation"][:2] for e in ep], np.float64)


def fill_scene(scene):
    prod = os.path.join(PROD, scene)
    meta = json.load(open(os.path.join(prod, "meta.json")))
    bev = np.load(os.path.join(prod, "bev_label_masked.npy"))
    H, W = bev.shape
    x0, y0 = meta["origin"]
    res = meta["resolution"]
    traj = load_traj(scene)

    mask = np.zeros((H, W), np.uint8)
    tix = ((traj[:, 0] - x0) / res).astype(np.int32)
    tiy = ((traj[:, 1] - y0) / res).astype(np.int32)
    r_px = max(1, int(round(EGO_FILL_R / res)))
    for x, y in zip(tix, tiy):
        if 0 <= x < W and 0 <= y < H:
            cv2.circle(mask, (int(x), int(y)), r_px, 1, -1)
    fillm = (mask > 0) & (bev == UNLABELED)
    n = int(fillm.sum())
    bev[fillm] = ROAD

    np.save(os.path.join(prod, "bev_label_masked.npy"), bev)
    vis = np.ascontiguousarray(PALETTE[bev][::-1])
    cv2.imwrite(os.path.join(prod, "bev_label_masked.png"), vis[:, :, ::-1])
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", required=True, help="file, one scene per line")
    args = ap.parse_args()
    scenes = [s for s in open(args.scenes).read().split() if s]
    for i, sc in enumerate(scenes):
        try:
            n = fill_scene(sc)
            tag = f"filled {n} cells"
        except Exception as e:
            tag = f"FAIL {e}"
        if i % 20 == 0 or "FAIL" in tag:
            print(f"{i + 1}/{len(scenes)} {sc} {tag}", flush=True)
    print("DONE", len(scenes), flush=True)


if __name__ == "__main__":
    main()

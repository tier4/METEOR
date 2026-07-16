#!/usr/bin/env python3
"""OCC GT before/after the dynamic-shadow filter — side-by-side video.

Left: camera. Middle: UNFILTERED occupancy GT (top view). Right: FILTERED
GT. GT 3D boxes are overlaid on both so the shadows are attributable:
vehicle/pedestrian colour outside a box on the left that turns dark on the
right is exactly what the filter removed (ignored, not asserted free).
"""
import argparse
import json
import os
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bevlane.demo_occ_gt import top_view  # noqa: E402


def draw_boxes(img, z, size):
    s = size / 200.0
    for k in range(int(z["count"])):
        cls, xe, ye, l, w, yaw = z["boxes"][k]
        if l <= 0 or abs(xe) > 40 or abs(ye) > 40:
            continue
        c_, s_ = np.cos(yaw), np.sin(yaw)
        cor = [[int((40 - (ye + lx * s_ + wy * c_)) / 0.4 * s),
                int((40 - (xe + lx * c_ - wy * s_)) / 0.4 * s)]
               for lx, wy in ((l/2, w/2), (l/2, -w/2),
                              (-l/2, -w/2), (-l/2, w/2))]
        col = (255, 255, 255) if cls < 1.5 else (255, 0, 255)
        cv2.polylines(img, [np.array(cor, np.int32).reshape(-1, 1, 2)],
                      True, col, 2)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--before-dir", required=True,
                    help="dir holding the UNFILTERED occ/ copy for the scene")
    ap.add_argument("--out", default="out/demo_occ_filter_gt.mp4")
    ap.add_argument("--fps", type=int, default=10)
    args = ap.parse_args()
    scene = args.scene
    root = f"out/bevlane/{scene}"
    man = json.load(open(f"{root}/manifest.json"))
    VW, VH = 1920, 800
    raw = args.out.replace(".mp4", "_raw.mp4")
    vw = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
                         (VW, VH))
    SIZE = 700
    n = 0
    for f in man["frames"]:
        if not f.get("occ") or not f.get("agent_traj"):
            continue
        bpath = os.path.join(args.before_dir, scene, f["occ"])
        apath = os.path.join(root, f["occ"])
        if not os.path.exists(bpath):
            continue
        try:
            ob = np.load(bpath)["occ"]
            oa = np.load(apath)["occ"]
            z = np.load(f"{root}/" + f["agent_traj"])
        except Exception:
            continue
        frame = np.zeros((VH, VW, 3), np.uint8)
        p = f["imgs"].get("CAM_FRONT_WIDE")
        img = cv2.imread(f"{root}/" + p) if p else None
        if img is not None:
            img = cv2.resize(img, (480, 270))
            frame[40:310, 8:488] = img
        p2 = f["imgs"].get("CAM_FRONT_LEFT")
        img2 = cv2.imread(f"{root}/" + p2) if p2 else None
        if img2 is not None:
            img2 = cv2.resize(img2, (480, 270))
            frame[350:620, 8:488] = img2
            cv2.putText(frame, "CAM_FRONT_LEFT", (14, 344),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1,
                        cv2.LINE_AA)
        tb = draw_boxes(top_view(ob, size=SIZE), z, SIZE)
        ta = draw_boxes(top_view(oa, size=SIZE), z, SIZE)
        frame[60:60 + SIZE, 500:500 + SIZE] = tb
        frame[60:60 + SIZE, 1215:1215 + SIZE] = ta
        cv2.putText(frame, "OCC GT BEFORE (raw accumulation)",
                    (500, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (0, 200, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "OCC GT AFTER shadow filter",
                    (1215, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "white/magenta = GT boxes (veh / VRU) | "
                    "dyn voxels outside boxes -> ignored (black)",
                    (500, VH - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, f"{scene.split('+0900_')[-1]} f{f['frame']:03d} "
                    "| GT only", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 0), 1, cv2.LINE_AA)
        vw.write(frame)
        n += 1
    vw.release()
    subprocess.run(["ffmpeg", "-y", "-i", raw, "-c:v", "libx264", "-crf",
                    "24", "-pix_fmt", "yuv420p", args.out], check=True,
                   capture_output=True)
    os.remove(raw)
    print("done", n, args.out, flush=True)


if __name__ == "__main__":
    main()

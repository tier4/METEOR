#!/usr/bin/env python3
"""Survey montage of the converted BEV GT: ~1 s per scene.

Each scene contributes N sampled frames; panel = FRONT_WIDE RGB (left) +
BEV GT with 3D-box outlines (right), scene id caption. 1280x720 @ 10 fps.
"""
import argparse
import json
import os
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from autolabel_bev import PALETTE  # noqa: E402
from bevlane.postproc import crop_bev, draw_ego_and_grid, thin_road_edge  # noqa: E402

P = np.zeros((12, 3), np.uint8)
P[:len(PALETTE)] = PALETTE
P[2] = 0
P[8] = 0
P[10] = (255, 215, 0)
P[11] = (255, 0, 255)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--out", default="out/montage_bev200.mp4")
    ap.add_argument("--per-scene", type=int, default=10)
    ap.add_argument("--fps", type=int, default=10)
    args = ap.parse_args()
    scenes = open(args.scenes).read().split()
    VW, VH = 1280, 720
    raw = args.out.replace(".mp4", "_raw.mp4")
    vw = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (VW, VH))
    n = 0
    for si, s in enumerate(scenes):
        try:
            man = json.load(open(f"out/bevlane/{s}/manifest.json"))
        except Exception:
            continue
        frames = [f for f in man["frames"] if "gt_vec" in f]
        if not frames:
            continue
        step = max(1, len(frames) // args.per_scene)
        for f in frames[10::step][:args.per_scene]:
            gt = cv2.imread(f"out/bevlane/{s}/" + f["gt_vec"], 0)
            if gt is None:
                continue
            g = thin_road_edge(gt)
            bx = cv2.imread(f"out/bevlane/{s}/" + f.get("bev_box", "_"), 0)
            if bx is not None:
                g[bx == 1] = 10
                g[bx == 2] = 11
            pc = crop_bev(g, xh_m=60.0, yh_m=25.0)
            BH2 = VH - 40
            BW2 = int(BH2 * pc.shape[1] / pc.shape[0])
            bev = draw_ego_and_grid(P[pc][:, :, ::-1], BH2, BW2,
                                    xh_m=60.0, yh_m=25.0)
            # box outlines from params
            bpp = f"out/bevlane/{s}/" + f.get("bev_box_p", "_")
            if os.path.exists(bpp):
                try:
                    bxs = np.load(bpp)["boxes"]
                except Exception:
                    bxs = np.zeros((0, 6), np.float32)
                sy2, sx2 = BH2 / 120.0, BW2 / 50.0
                for cls, xe, ye, l, w, yaw in bxs:
                    if abs(xe) > 60 or abs(ye) > 25:
                        continue
                    cb, sb = np.cos(yaw), np.sin(yaw)
                    cor = [[int((25 - (ye + lx * sb + wy * cb)) * sx2),
                            int((60 - (xe + lx * cb - wy * sb)) * sy2)]
                           for lx, wy in ((l/2, w/2), (l/2, -w/2),
                                          (-l/2, -w/2), (-l/2, w/2))]
                    col = (0, 215, 255) if cls < 1.5 else (255, 0, 255)
                    cv2.polylines(bev, [np.array(cor, np.int32).reshape(-1, 1, 2)],
                                  True, col, 2)
            frame = np.zeros((VH, VW, 3), np.uint8)
            rgbp = f["imgs"].get("CAM_FRONT_WIDE")
            if rgbp:
                img = cv2.imread(f"out/bevlane/{s}/" + rgbp)
                if img is not None:
                    rw = VW - BW2 - 30
                    rh = int(rw * 432 / 768)
                    frame[40:40 + rh, 10:10 + rw] = cv2.resize(img, (rw, rh))
            frame[30:30 + BH2, VW - BW2 - 10:VW - 10] = bev
            tag = "batchB" if s.startswith("6yb") else "batchA"
            cv2.putText(frame, f"[{si+1}/{len(scenes)}] {tag}  {s[-40:]}",
                        (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 255, 0), 1, cv2.LINE_AA)
            vw.write(frame)
            n += 1
        if si % 20 == 0:
            print(f"{si+1}/{len(scenes)}", flush=True)
    vw.release()
    subprocess.run(["ffmpeg", "-y", "-i", raw, "-c:v", "libx264", "-crf", "24",
                    "-pix_fmt", "yuv420p", args.out], check=True,
                   capture_output=True)
    os.remove(raw)
    print("done", n, args.out, flush=True)


if __name__ == "__main__":
    main()

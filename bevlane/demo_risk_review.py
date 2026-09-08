#!/usr/bin/env python3
"""Review video for the extracted risk-map GT: renders the STORED
risk_map.npz (not a recomputation) for a sampled subset of scenes."""
import argparse
import json
import os
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from autolabel_bev import PALETTE  # noqa: E402
from bevlane.risk_field import RH, RW, XH, YH  # noqa: E402

GTP = np.zeros((256, 3), np.uint8)
GTP[:len(PALETTE)] = PALETTE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--out", default="out/demo_risk_review.mp4")
    ap.add_argument("--fps", type=int, default=15)
    args = ap.parse_args()
    scenes = sorted(s for s in os.listdir("out/bevlane")
                    if os.path.exists(f"out/bevlane/{s}/risk_map.npz"))
    scenes = scenes[::args.every]
    print(f"{len(scenes)} scenes in review", flush=True)
    VW, VH = 1280, 720
    raw = args.out.replace(".mp4", "_raw.mp4")
    vw = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
                         (VW, VH))
    n = 0
    PH = VH - 50
    PW = int(PH * RW / RH)
    for si, scene in enumerate(scenes):
        root = f"out/bevlane/{scene}"
        try:
            man = json.load(open(f"{root}/manifest.json"))
            risk_all = np.load(f"{root}/risk_map.npz")["risk"]
        except Exception:
            continue
        ego = None
        try:
            ego = np.load(f"{root}/ego_motion.npz")
        except Exception:
            pass
        for f in man["frames"]:
            fi = f["frame"]
            if fi >= len(risk_all):
                continue
            risk = risk_all[fi].astype(np.float32) / 255.0
            frame = np.zeros((VH, VW, 3), np.uint8)
            p = f["imgs"].get("CAM_FRONT_WIDE")
            img = cv2.imread(f"{root}/" + p) if p else None
            if img is not None:
                img = cv2.resize(img, (760, 428))
                frame[140:568, 10:770] = img
            gt = cv2.imread(f"{root}/" + f.get("gt_vec", "_"), 0)
            if gt is not None:
                r0 = int((80 - XH) / 0.2)
                c0 = int((50 - YH) / 0.2)
                under = (GTP[gt[r0:r0 + RH, c0:c0 + RW]][:, :, ::-1]
                         * 0.45).astype(np.uint8)
            else:
                under = np.zeros((RH, RW, 3), np.uint8)
            heat = cv2.applyColorMap((risk * 255).astype(np.uint8),
                                     cv2.COLORMAP_TURBO)
            a = (risk * 0.85)[..., None]
            bev = (under * (1 - a) + heat * a).astype(np.uint8)
            if ego is not None and fi < len(ego["v0"]) and ego["valid"][fi] > 0:
                pts = [(RW // 2, RH // 2)]
                for xe, ye in ego["wp"][fi]:
                    if abs(xe) > XH or abs(ye) > YH:
                        break
                    pts.append((int((YH - ye) / 0.2), int((XH - xe) / 0.2)))
                cv2.polylines(bev, [np.array(pts, np.int32).reshape(-1, 1, 2)],
                              False, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.drawMarker(bev, (RW // 2, RH // 2), (255, 255, 255),
                           cv2.MARKER_TRIANGLE_UP, 12, 2)
            bev = cv2.resize(bev, (PW, PH), interpolation=cv2.INTER_NEAREST)
            x0 = VW - PW - 12
            frame[36:36 + PH, x0:x0 + PW] = bev
            cv2.putText(frame, "GT RISK MAP +-40x+-25m (stored)", (x0, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
                        cv2.LINE_AA)
            cv2.putText(frame, f"[{si + 1}/{len(scenes)}] "
                        f"{scene}  f{fi:03d}", (10, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                        cv2.LINE_AA)
            vw.write(frame)
            n += 1
        if si % 20 == 0:
            print(f"{si + 1}/{len(scenes)} scenes, {n} frames", flush=True)
    vw.release()
    subprocess.run(["ffmpeg", "-y", "-i", raw, "-c:v", "libx264", "-crf", "26",
                    "-pix_fmt", "yuv420p", args.out], check=True,
                   capture_output=True)
    os.remove(raw)
    print("done", n, args.out, flush=True)


if __name__ == "__main__":
    main()

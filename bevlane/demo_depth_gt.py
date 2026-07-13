#!/usr/bin/env python3
"""GT-side companion of demo_rgbd_bev: RGB | depth GT | BEV GT, same layout.

Shows exactly the supervision targets: dense depth GT (LiDAR + panoptic
densify, stride-4, 0=invalid shown black) under each camera, and the gt_vec
BEV on the right (same +-25 x +-60 m crop, ego icon, grid).
"""
import argparse
import os
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from autolabel_bev import PALETTE  # noqa: E402
from bevlane.dataset import CAMS, BevLaneDataset  # noqa: E402
from bevlane.postproc import crop_bev, draw_ego_and_grid, thin_road_edge  # noqa: E402

DEMO_PALETTE = np.zeros((12, 3), np.uint8)
DEMO_PALETTE[:len(PALETTE)] = PALETTE
DEMO_PALETTE[2] = 0
DEMO_PALETTE[8] = 0
DEMO_PALETTE[10] = (255, 215, 0)    # vehicle box (RGB gold)
DEMO_PALETTE[11] = (255, 0, 255)    # VRU box (magenta)


def label(img, txt, color=(255, 255, 255)):
    cv2.putText(img, txt, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--out", default="out/demo_depth_gt.mp4")
    ap.add_argument("--fps", type=int, default=15)
    args = ap.parse_args()

    VW, VH = 1920, 1080
    CAM8 = ["CAM_FRONT_LEFT", "CAM_FRONT_WIDE", "CAM_FRONT_RIGHT", "CAM_FRONT_NARROW",
            "CAM_BACK_LEFT", "CAM_BACK_WIDE", "CAM_BACK_RIGHT", "CAM_BACK_NARROW"]
    cw = 375
    ch = cw * 288 // 512
    rgb_y0 = 40
    dep_y0 = rgb_y0 + 2 * ch + 45
    bx0 = 4 * cw + 8
    raw = args.out.replace(".mp4", "_raw.mp4")
    vw = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (VW, VH))
    n = 0
    for scene in args.scenes:
        if not os.path.exists(f"out/bevlane/{scene}/manifest.json"):
            continue
        ds = BevLaneDataset("out/bevlane", [scene], gt_key="gt_vec",
                            with_depth=True, depth_hw=(108, 192))
        for i in range(len(ds)):
            imgs, K, T, gt, dep = ds[i]
            s, f = ds.items[i]
            frame = np.zeros((VH, VW, 3), np.uint8)
            for k, chn in enumerate(CAM8):
                r, c = divmod(k, 4)
                x = c * cw
                p = f["imgs"].get(chn)
                if p:
                    img = cv2.resize(cv2.imread(os.path.join("out/bevlane", s, p)),
                                     (cw, ch))
                    if "NARROW" in chn:
                        label(img, "NARROW", (0, 255, 0))
                    frame[rgb_y0 + r * ch:rgb_y0 + (r + 1) * ch, x:x + cw] = img
                d = dep[CAMS.index(chn)].numpy()
                dc = cv2.applyColorMap(np.clip(d / 80 * 255, 0, 255).astype(np.uint8),
                                       cv2.COLORMAP_TURBO)
                dc[d <= 0.1] = (0, 0, 0)          # invalid = black (don't-care)
                dc = cv2.resize(dc, (cw, ch), interpolation=cv2.INTER_NEAREST)
                label(dc, chn.split("CAM_")[-1], (255, 255, 255))
                frame[dep_y0 + r * ch:dep_y0 + (r + 1) * ch, x:x + cw] = dc
            cv2.putText(frame, "RGB input (surround + tele NARROW)", (10, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
            cv2.putText(frame, "depth GT (LiDAR + panoptic densify, 0-80m, black=invalid)",
                        (10, dep_y0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (220, 220, 220), 1, cv2.LINE_AA)

            g = thin_road_edge(gt.numpy().astype(np.uint8))
            # overlay camera-visible 3D-box GT (vehicle=10, vru=11 sentinel ids)
            bxp = os.path.join("out/bevlane", s, f.get("bev_box", "_"))
            bx = cv2.imread(bxp, 0)
            if bx is not None:
                g = g.copy()
                g[bx == 1] = 10
                g[bx == 2] = 11
            pc = crop_bev(g, xh_m=60.0, yh_m=25.0)
            BH2 = VH - 90
            BW2 = int(BH2 * pc.shape[1] / pc.shape[0])
            bev = draw_ego_and_grid(DEMO_PALETTE[pc][:, :, ::-1], BH2, BW2,
                                    xh_m=60.0, yh_m=25.0)
            # oriented outlines + heading ticks from the box params (npz)
            bpp = os.path.join("out/bevlane", s, f.get("bev_box_p", "_"))
            if os.path.exists(bpp):
                try:
                    bxs = np.load(bpp)["boxes"]
                except Exception:
                    bxs = np.zeros((0, 6), np.float32)
                sy2 = BH2 / 120.0
                sx2 = BW2 / 50.0
                for cls, xe, ye, l, w, yaw in bxs:
                    if abs(xe) > 60 or abs(ye) > 25:
                        continue
                    cb, sb = np.cos(yaw), np.sin(yaw)
                    cor = []
                    for lx, wy in ((l/2, w/2), (l/2, -w/2), (-l/2, -w/2), (-l/2, w/2)):
                        px_ = xe + lx*cb - wy*sb
                        py_ = ye + lx*sb + wy*cb
                        cor.append([int((25.0 - py_) * sx2), int((60.0 - px_) * sy2)])
                    col = (0, 215, 255) if cls < 1.5 else (255, 0, 255)
                    cv2.polylines(bev, [np.array(cor, np.int32).reshape(-1, 1, 2)],
                                  True, col, 2)
                    cxp = int((25.0 - ye) * sx2); cyp = int((60.0 - xe) * sy2)
                    fxp = int((25.0 - (ye + (l/2)*sb)) * sx2)
                    fyp = int((60.0 - (xe + (l/2)*cb)) * sy2)
                    cv2.line(bev, (cxp, cyp), (fxp, fyp), col, 2)
            cv2.putText(bev, "GT BEV+3D-box +-25x+-60m", (6, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            bx = min(bx0, VW - BW2)
            frame[40:40 + BH2, bx:bx + BW2] = bev

            cv2.putText(frame, f"{scene.split('+0900_')[-1]}  f{f['frame']:03d}  |  "
                        f"GROUND TRUTH (depth + BEV autolabel)",
                        (10, VH - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                        (0, 200, 255), 2, cv2.LINE_AA)
            vw.write(frame)
            n += 1
        print("scene", scene, n, flush=True)
    vw.release()
    subprocess.run(["ffmpeg", "-y", "-i", raw, "-c:v", "libx264", "-crf", "23",
                    "-pix_fmt", "yuv420p", args.out], check=True, capture_output=True)
    os.remove(raw)
    print("done", n, args.out, flush=True)


if __name__ == "__main__":
    main()

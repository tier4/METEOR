#!/usr/bin/env python3
"""Full GT visualisation video: all 8 supervision targets, no model.

Top 2x4 grid: cached RGB + 21-class seg2d21 GT overlay + 10-class 2D bbox GT.
Middle 2x4 grid: dense metric depth GT (stride-4, TURBO 0-80 m, black=invalid).
Bottom: 3D occupancy GT (voxel-cube isometric + top-down) with class legend.
Right column: BEV gt_vec (+-25 x +-60 m) + oriented 3D-box GT outlines +
E2E GT (green trajectory waypoints, v0/steer/accel/brake gauges).
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
from bevlane.demo_occ_gt import cube_render  # noqa: E402
from bevlane.extract_bbox2d import CAT2DET, DET10_PAL  # noqa: E402
from bevlane.extract_occ import OCC_NAMES, OCC_PAL  # noqa: E402
from bevlane.extract_seg2d import CAMS, SEG21_PAL  # noqa: E402
from bevlane.postproc import crop_bev, draw_ego_and_grid, thin_road_edge  # noqa: E402

DET10_ABBR = ["obs", "car", "trk", "bus", "bcy", "mcy", "ped", "pnt", "tl", "ts"]
CAM8 = ["CAM_FRONT_LEFT", "CAM_FRONT_WIDE", "CAM_FRONT_RIGHT", "CAM_FRONT_NARROW",
        "CAM_BACK_LEFT", "CAM_BACK_WIDE", "CAM_BACK_RIGHT", "CAM_BACK_NARROW"]
DCAMS = ["CAM_FRONT_WIDE", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
         "CAM_BACK_WIDE", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
         "CAM_FRONT_NARROW", "CAM_BACK_NARROW"]
GTP = np.zeros((256, 3), np.uint8)
GTP[:len(PALETTE)] = PALETTE


def occ_topdown(occ, size):
    """Top-down occupancy: highest occupied voxel wins, +x forward = up."""
    img = np.zeros((occ.shape[1], occ.shape[2], 3), np.uint8)
    img[(occ != 255).any(0)] = (35, 35, 35)          # observed free
    for z in range(occ.shape[0]):
        o = occ[z]
        m = (o > 0) & (o != 255)
        img[m] = OCC_PAL[o[m]][:, ::-1]
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)
    cv2.drawMarker(img, (size // 2, size // 2), (0, 255, 0),
                   cv2.MARKER_TRIANGLE_UP, 14, 2)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+")
    ap.add_argument("--out", default="out/demo_gt_full.mp4")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--frame-stride", type=int, default=1,
                    help="sample every Nth manifest frame (dataset-wide overview)")
    ap.add_argument("--scenes-file", help="read scene list from a file instead")
    args = ap.parse_args()
    if args.scenes_file:
        args.scenes = [s for s in open(args.scenes_file).read().split() if s]
    if not args.scenes:
        ap.error("provide --scenes or --scenes-file")
    VW, VH = 1920, 1080
    raw = args.out.replace(".mp4", "_raw.mp4")
    vw = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (VW, VH))
    n = 0
    for scene in args.scenes:
        root = f"out/bevlane/{scene}"
        man = json.load(open(f"{root}/manifest.json"))
        try:
            ego = np.load(f"{root}/ego_motion.npz")
        except Exception:
            ego = None
        cw, ch = 373, 210
        for f in man["frames"][::args.frame_stride]:
            fi = f["frame"]
            seg = None
            if f.get("seg2d21"):
                try:
                    seg = np.load(f"{root}/" + f["seg2d21"])["seg"]
                except Exception:
                    pass
            bb2 = None
            if f.get("bbox2d"):
                try:
                    z = np.load(f"{root}/" + f["bbox2d"])
                    bb2 = (z["boxes"], z["counts"])
                except Exception:
                    pass
            dep = None
            if f.get("depth4"):
                try:
                    d6 = np.load(f"{root}/" + f["depth4"])["depth"].astype(np.float32)
                    if f.get("depth4n"):
                        dn = np.load(f"{root}/" + f["depth4n"])["depth"].astype(np.float32)
                    else:
                        dn = np.zeros((2,) + d6.shape[1:], np.float32)
                    dep = np.concatenate([d6, dn], 0)
                except Exception:
                    pass
            occ = None
            if f.get("occ"):
                try:
                    occ = np.load(f"{root}/" + f["occ"])["occ"]
                except Exception:
                    pass
            frame = np.zeros((VH, VW, 3), np.uint8)
            for k, chn in enumerate(CAM8):
                r, c = divmod(k, 4)
                p = f["imgs"].get(chn)
                if not p:
                    continue
                img = cv2.imread(f"{root}/" + p)
                if img is None:
                    continue
                img = cv2.resize(img, (cw, ch))
                ci = CAMS.index(chn)
                if seg is not None:
                    lab = seg[ci]
                    col = SEG21_PAL[np.where(lab == 255, 0, lab)][:, :, ::-1]
                    col = cv2.resize(col, (cw, ch), interpolation=cv2.INTER_NEAREST)
                    m3 = cv2.resize((lab != 255).astype(np.uint8), (cw, ch),
                                    interpolation=cv2.INTER_NEAREST)[..., None]
                    img = np.where(m3 > 0,
                                   (0.55 * img + 0.45 * col).astype(np.uint8), img)
                if bb2 is not None:
                    bx, cnt = bb2
                    sx, sy = cw / 768.0, ch / 432.0
                    for j in range(int(cnt[ci])):
                        cls, cx, cy, w, h = bx[ci, j]
                        col2 = tuple(int(v) for v in DET10_PAL[int(cls)][::-1])
                        x1, y1 = int((cx - w / 2) * sx), int((cy - h / 2) * sy)
                        x2, y2 = int((cx + w / 2) * sx), int((cy + h / 2) * sy)
                        cv2.rectangle(img, (x1, y1), (x2, y2), col2, 1)
                        cv2.putText(img, DET10_ABBR[int(cls)], (x1, max(y1 - 2, 9)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, col2, 1,
                                    cv2.LINE_AA)
                if "NARROW" in chn:
                    cv2.putText(img, "NARROW", (6, 18), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 1, cv2.LINE_AA)
                frame[40 + r * ch:40 + (r + 1) * ch, 8 + c * cw:8 + (c + 1) * cw] = img
            cv2.putText(frame, "RGB + GT: 21cls 2D Seg overlay + 10cls 2D BBox",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (220, 220, 220), 1, cv2.LINE_AA)

            # ---- depth GT grid (2x4, same cam layout as RGB) ----
            dw, dh, dy0 = 280, 158, 505
            if dep is not None:
                for k, chn in enumerate(CAM8):
                    r, c = divmod(k, 4)
                    d = dep[DCAMS.index(chn)]
                    dc = cv2.applyColorMap(
                        np.clip(d / 80 * 255, 0, 255).astype(np.uint8),
                        cv2.COLORMAP_TURBO)
                    dc[d <= 0.1] = (0, 0, 0)          # invalid = black
                    dc = cv2.resize(dc, (dw, dh), interpolation=cv2.INTER_NEAREST)
                    cv2.putText(dc, chn.split("CAM_")[-1], (6, 16),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                (255, 255, 255), 1, cv2.LINE_AA)
                    frame[dy0 + r * dh:dy0 + (r + 1) * dh,
                          8 + c * dw:8 + (c + 1) * dw] = dc
            cv2.putText(frame, "depth GT (stride-4 dense, TURBO 0-80m, black=invalid)",
                        (10, dy0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (220, 220, 220), 1, cv2.LINE_AA)

            # ---- 3D occupancy GT: top-down + voxel-cube isometric ----
            if occ is not None:
                td = occ_topdown(occ, 316)
                frame[dy0:dy0 + 316, 1150:1150 + 316] = td
                cv2.putText(frame, "occ GT top-down +-40m", (1150, dy0 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (220, 220, 220), 1, cv2.LINE_AA)
                cube = cube_render(occ, W=700, H=238, rng_m=24.0)
                frame[832:832 + 238, 8:8 + 700] = cube
                cv2.putText(frame, "occ GT voxels (iso, +-24m, z<3m, bldg hidden)",
                            (10, 852), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (220, 220, 220), 1, cv2.LINE_AA)
                for li, nm in enumerate(OCC_NAMES):
                    r, c = divmod(li, 4)
                    x0, y0 = 740 + c * 190, 880 + r * 34
                    col = tuple(int(v) for v in OCC_PAL[li][::-1])
                    cv2.rectangle(frame, (x0, y0), (x0 + 20, y0 + 20), col, -1)
                    cv2.rectangle(frame, (x0, y0), (x0 + 20, y0 + 20),
                                  (90, 90, 90), 1)
                    cv2.putText(frame, nm, (x0 + 28, y0 + 16),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                                (200, 200, 200), 1, cv2.LINE_AA)

            # ---- BEV GT column ----
            gt = cv2.imread(f"{root}/" + f.get("gt_vec", "_"), 0)
            if gt is not None:
                g = thin_road_edge(gt)
                pc = crop_bev(g, xh_m=60.0, yh_m=25.0)
                BH2 = VH - 90
                BW2 = int(BH2 * pc.shape[1] / pc.shape[0])
                bev = draw_ego_and_grid(GTP[pc][:, :, ::-1], BH2, BW2,
                                        xh_m=60.0, yh_m=25.0)
                sy2, sx2 = BH2 / 120.0, BW2 / 50.0
                bpp = f"{root}/" + f.get("bev_box_p", "_")
                if os.path.exists(bpp):
                    try:
                        bxs = np.load(bpp)["boxes"]
                    except Exception:
                        bxs = np.zeros((0, 6), np.float32)
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
                # ---- agent trajectory GT (future 3 s per box) ----
                atp = f"{root}/" + f.get("agent_traj", "_")
                if os.path.exists(atp):
                    try:
                        z2 = np.load(atp)
                        for k in range(int(z2["count"])):
                            cls, xe, ye = z2["boxes"][k][:3]
                            if abs(xe) > 60 or abs(ye) > 25:
                                continue
                            pts = [(int((25 - ye) * sx2),
                                    int((60 - xe) * sy2))]
                            for h in range(6):
                                if z2["tvalid"][k, h] < 0.5:
                                    break
                                fx = xe + z2["traj"][k, h, 0]
                                fy = ye + z2["traj"][k, h, 1]
                                if abs(fx) > 60 or abs(fy) > 25:
                                    break
                                pts.append((int((25 - fy) * sx2),
                                            int((60 - fx) * sy2)))
                            if len(pts) > 1:
                                col = (255, 255, 0) if cls < 1.5                                     else (255, 0, 255)
                                cv2.polylines(
                                    bev,
                                    [np.array(pts, np.int32).reshape(-1, 1, 2)],
                                    False, col, 1, cv2.LINE_AA)
                                cv2.circle(bev, pts[-1], 3, col, -1)
                    except Exception:
                        pass
                # ---- E2E GT ----
                if ego is not None and fi < len(ego["v0"]) \
                        and ego["valid"][fi] == 0:
                    cv2.putText(bev, "E2E GT: none (scene end, no 3s future)",
                                (6, BH2 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                                (0, 200, 255), 1, cv2.LINE_AA)
                if ego is not None and fi < len(ego["v0"]) and ego["valid"][fi] > 0:
                    wp = ego["wp"][fi]
                    pts = [(int(25.0 * sx2), int(60.0 * sy2))]
                    for xe, ye in wp:
                        if abs(xe) > 60 or abs(ye) > 25:
                            break
                        pts.append((int((25.0 - ye) * sx2),
                                    int((60.0 - xe) * sy2)))
                    cv2.polylines(bev, [np.array(pts, np.int32).reshape(-1, 1, 2)],
                                  False, (0, 255, 0), 2)
                    for p in pts[1:]:
                        cv2.circle(bev, p, 3, (0, 255, 0), -1)
                    txts = [f"v0 {ego['v0'][fi] * 3.6:5.1f} km/h",
                            f"steer {np.degrees(ego['steer'][fi]):+6.1f} deg",
                            f"accel {ego['acc'][fi]:+5.2f} m/s2",
                            "BRAKE" if ego["brake"][fi] > 0 else "brake 0"]
                    for li, txt in enumerate(txts):
                        cv2.putText(bev, txt, (6, BH2 - 78 + 22 * li),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                                    (0, 80, 255) if txt == "BRAKE" else (0, 255, 0),
                                    1, cv2.LINE_AA)
                cv2.putText(bev, "GT BEV + 3D box + agent traj + E2E", (6, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
                            cv2.LINE_AA)
                frame[40:40 + BH2, VW - BW2 - 8:VW - 8] = bev
            cv2.putText(frame, f"{scene.split('+0900_')[-1]}  f{fi:03d}  |  "
                        "GROUND TRUTH (all 8 tasks)", (740, VH - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 0), 1, cv2.LINE_AA)
            vw.write(frame)
            n += 1
        print(f"scene {scene} done ({n} frames total)", flush=True)
    vw.release()
    subprocess.run(["ffmpeg", "-y", "-i", raw, "-c:v", "libx264", "-crf", "24",
                    "-pix_fmt", "yuv420p", args.out], check=True,
                   capture_output=True)
    os.remove(raw)
    print("done", n, args.out, flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Near-range RISK MAP ground truth — sample video (no training involved).

Definition (per BEV cell, +-40 m x +-25 m @ 0.2 m):
  risk = max over sources of
    - dynamic agents (agent_traj GT): the agent's oriented footprint swept
      along its GT 3 s future at 0.25 s steps; weight
        severity * exp(-t / TAU)          TAU = 1.5 s
      severity: pedestrian/2-wheeler 1.0, moving vehicle 0.75.
      Footprint inflated by 0.15 m per second of horizon (uncertainty).
    - stationary vehicles (|GT disp@3s| < 0.5 m): constant 0.55 footprint,
      no future sweep (they are obstacles, not moving hazards).
    - static world from occupancy GT: obstacle/wall/building/pole 0.5,
      vegetation 0.3 (any occupied voxel in the column).
  Ego's own footprint is zeroed.

Right panel: risk heatmap (TURBO) over the BEV lane GT, with GT boxes,
futures and the ego GT path. Left: front cameras for context.
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

XH, YH, RES = 40.0, 25.0, 0.2          # risk grid: x +40..-40, y +25..-25
RH, RW = int(2 * XH / RES), int(2 * YH / RES)          # 400 x 250
TAU = 1.5
SEV_VRU, SEV_VEH, SEV_PARK = 1.0, 0.75, 0.55
STATIC_RISK = {1: 0.5, 8: 0.5, 9: 0.5, 7: 0.3}         # occ classes
GTP = np.zeros((256, 3), np.uint8)
GTP[:len(PALETTE)] = PALETTE


def cell(x, y):
    return int((XH - x) / RES), int((YH - y) / RES)


# grid coordinate maps (metres), built once
_GX = XH - (np.arange(RH, dtype=np.float32) + 0.5) * RES     # rows -> x
_GY = YH - (np.arange(RW, dtype=np.float32) + 0.5) * RES     # cols -> y
GX = np.repeat(_GX[:, None], RW, 1)
GY = np.repeat(_GY[None, :], RH, 0)


def gauss_lobe(x, y, yaw, s_long, s_lat, amp, lead=0.0):
    """Anisotropic Gaussian potential centred at (x,y) (+ lead metres along
    heading), axes aligned with yaw. Returns [RH,RW] float."""
    cx = x + lead * np.cos(yaw)
    cy = y + lead * np.sin(yaw)
    dx, dy = GX - cx, GY - cy
    c, s = np.cos(yaw), np.sin(yaw)
    u = c * dx + s * dy                 # along heading
    v = -s * dx + c * dy                # lateral
    return amp * np.exp(-0.5 * ((u / s_long) ** 2 + (v / s_lat) ** 2))


def risk_frame(root, f, occ_arr):
    """Smooth AREA risk field around the ego (potential-field style)."""
    inten = np.zeros((RH, RW), np.float32)   # hazard intensity (saturating)
    # ---- static world: distance falloff from occupied cells ----
    if occ_arr is not None:
        occ = occ_arr
        stat = np.zeros((200, 200), np.float32)
        for cls_id, rv in STATIC_RISK.items():
            stat = np.maximum(stat, rv * (occ == cls_id).any(0))
        up = cv2.resize(stat, (400, 400), interpolation=cv2.INTER_NEAREST)
        c0 = int((40.0 - YH) / 0.2)
        stat = up[:, c0:c0 + RW]
        d = cv2.distanceTransform((stat < 0.05).astype(np.uint8),
                                  cv2.DIST_L2, 3) * RES
        amp = cv2.dilate(stat, np.ones((9, 9), np.uint8))
        inten += amp * np.exp(-d / 1.2)
    # ---- dynamic agents: anisotropic potential lobes ----
    atp = os.path.join(root, f.get("agent_traj", "_"))
    if os.path.exists(atp):
        z = np.load(atp)
        for k in range(int(z["count"])):
            cls, xe, ye, l, w, yaw = z["boxes"][k]
            if l <= 0 or abs(xe) > XH + 10 or abs(ye) > YH + 10:
                continue
            tj, tv = z["traj"][k], z["tvalid"][k]
            vru = cls >= 1.5
            # mean speed over the first 1.5 s of GT future
            v = 0.0
            hd = float(yaw)
            if tv[2] > 0.5:
                v = float(np.linalg.norm(tj[2])) / 1.5
                if np.linalg.norm(tj[2]) > 0.4:
                    hd = float(np.arctan2(tj[2, 1], tj[2, 0]))
            if vru:
                sig = 1.3 + 0.8 * v          # isotropic, grows with speed
                inten += gauss_lobe(xe, ye, hd, sig + 0.6 * v, sig,
                                    SEV_VRU, lead=0.5 * v)
            elif v < 0.35:                   # parked / stopped vehicle
                inten += gauss_lobe(xe, ye, yaw, l * 0.55, w * 0.7,
                                    SEV_PARK)
            else:                            # moving vehicle: comet lobe
                s_long = l * 0.6 + 1.2 * v   # stretches with speed
                s_lat = w * 0.7 + 0.3
                inten += gauss_lobe(xe, ye, hd, s_long, s_lat,
                                    SEV_VEH, lead=0.75 * v)
    # saturate overlapping hazards, emphasise the ego neighbourhood
    risk = 1.0 - np.exp(-1.6 * inten)
    d_ego = np.sqrt(GX ** 2 + GY ** 2)
    risk *= np.exp(-d_ego / 30.0)
    # ego footprint out
    r0, c0 = cell(3.8, 1.1)
    r1, c1 = cell(-1.2, -1.1)
    risk[max(r0, 0):r1, max(c0, 0):c1] = 0.0
    return risk.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--out", default="out/demo_risk_gt.mp4")
    ap.add_argument("--fps", type=int, default=15)
    args = ap.parse_args()
    VW, VH = 1600, 900
    raw = args.out.replace(".mp4", "_raw.mp4")
    vw = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
                         (VW, VH))
    n = 0
    for scene in args.scenes:
        root = f"out/bevlane/{scene}"
        man = json.load(open(f"{root}/manifest.json"))
        try:
            ego = np.load(f"{root}/ego_motion.npz")
        except Exception:
            ego = None
        for f in man["frames"]:
            fi = f["frame"]
            occ_arr = None
            if f.get("occ"):
                try:
                    occ_arr = np.load(f"{root}/" + f["occ"])["occ"]
                except Exception:
                    pass
            risk = risk_frame(root, f, occ_arr)

            frame = np.zeros((VH, VW, 3), np.uint8)
            # ---- left: front cameras ----
            for row, cam in enumerate(("CAM_FRONT_WIDE", "CAM_FRONT_NARROW")):
                p = f["imgs"].get(cam)
                img = cv2.imread(f"{root}/" + p) if p else None
                if img is None:
                    continue
                img = cv2.resize(img, (760, 428))
                y0 = 20 + row * 440
                frame[y0:y0 + 428, 12:772] = img
                cv2.putText(frame, cam, (18, y0 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (240, 240, 240), 1, cv2.LINE_AA)
            # ---- right: risk map over lane GT ----
            PH = VH - 60                       # panel height
            PW = int(PH * RW / RH)
            gt = cv2.imread(f"{root}/" + f.get("gt_vec", "_"), 0)
            under = None
            if gt is not None:
                # gt_vec: 800x500 (+-80 x +-50 @0.2) -> crop +-40 x +-25
                r0 = int((80 - XH) / 0.2)
                c0 = int((50 - YH) / 0.2)
                crop = gt[r0:r0 + RH, c0:c0 + RW]
                under = (GTP[crop][:, :, ::-1] * 0.45).astype(np.uint8)
            else:
                under = np.zeros((RH, RW, 3), np.uint8)
            heat = cv2.applyColorMap((np.clip(risk, 0, 1) * 255).astype(np.uint8),
                                     cv2.COLORMAP_TURBO)
            a = (risk * 0.85)[..., None]
            bev = (under * (1 - a) + heat * a).astype(np.uint8)
            # ego GT path
            if ego is not None and fi < len(ego["v0"]) and ego["valid"][fi] > 0:
                pts = [cell(0, 0)[::-1]]
                for xe, ye in ego["wp"][fi]:
                    if abs(xe) > XH or abs(ye) > YH:
                        break
                    r, c = cell(xe, ye)
                    pts.append((c, r))
                cv2.polylines(bev, [np.array(pts, np.int32).reshape(-1, 1, 2)],
                              False, (255, 255, 255), 2, cv2.LINE_AA)
            # ego marker
            er, ec = cell(0, 0)
            cv2.drawMarker(bev, (ec, er), (255, 255, 255),
                           cv2.MARKER_TRIANGLE_UP, 14, 2)
            bev = cv2.resize(bev, (PW, PH), interpolation=cv2.INTER_NEAREST)
            x0 = VW - PW - 16
            frame[30:30 + PH, x0:x0 + PW] = bev
            cv2.putText(frame, "GT RISK MAP  +-40m x +-25m", (x0, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                        cv2.LINE_AA)
            # legend
            for i in range(200):
                col = cv2.applyColorMap(
                    np.full((1, 1), int(i / 199 * 255), np.uint8),
                    cv2.COLORMAP_TURBO)[0, 0]
                frame[VH - 24:VH - 10, 790 + i] = col
            cv2.putText(frame, "risk 0", (730, VH - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            cv2.putText(frame, "1  (area potential field: lobes grow/lead with speed, "
                        "VRU>veh>parked>static, ego-proximity weighted)",
                        (995, VH - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (200, 200, 200), 1)
            cv2.putText(frame, f"{scene.split('+0900_')[-1]}  f{fi:03d}  |  "
                        "GT only - no model", (12, VH - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1,
                        cv2.LINE_AA)
            vw.write(frame)
            n += 1
        print(f"scene {scene} done ({n} frames)", flush=True)
    vw.release()
    subprocess.run(["ffmpeg", "-y", "-i", raw, "-c:v", "libx264", "-crf", "24",
                    "-pix_fmt", "yuv420p", args.out], check=True,
                   capture_output=True)
    os.remove(raw)
    print("done", n, args.out, flush=True)


if __name__ == "__main__":
    main()

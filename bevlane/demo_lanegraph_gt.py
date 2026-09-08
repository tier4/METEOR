#!/usr/bin/env python3
"""GT demo for the v29 additions: vector lane graph + occupancy flow.

Right panel (BEV ROI x -10..60, |y|<=25): lane-graph chains as polylines
(laneline cyan, road_edge orange, stopline red) with adjacency links,
plus agent boxes with velocity arrows (occ-flow GT source). Left: camera.
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

XMIN, XMAX, YH, RES = -10.0, 60.0, 25.0, 0.2
RH, RW = int((XMAX - XMIN) / RES), int(2 * YH / RES)     # 350 x 250
CCOL = {0: (255, 255, 0), 1: (0, 165, 255), 2: (0, 0, 255)}
CNAME = {0: "laneline", 1: "road_edge", 2: "stopline"}
GTP = np.zeros((256, 3), np.uint8)
GTP[:len(PALETTE)] = PALETTE


def cell(x, y):
    return int((XMAX - x) / RES), int((YH - y) / RES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--out", default="out/demo_lanegraph_gt.mp4")
    ap.add_argument("--fps", type=int, default=15)
    args = ap.parse_args()
    VW, VH = 1600, 900
    raw = args.out.replace(".mp4", "_raw.mp4")
    vw = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
                         (VW, VH))
    n = 0
    PH = VH - 70
    PW = int(PH * RW / RH)
    for scene in args.scenes:
        root = f"out/bevlane/{scene}"
        man = json.load(open(f"{root}/manifest.json"))
        try:
            lg = np.load(f"{root}/lanegraph.npz")
        except Exception:
            print("no lanegraph:", scene)
            continue
        for f in man["frames"]:
            fi = f["frame"]
            if fi >= len(lg["n"]):
                continue
            frame = np.zeros((VH, VW, 3), np.uint8)
            for row, cam in enumerate(("CAM_FRONT_WIDE", "CAM_FRONT_NARROW")):
                p = f["imgs"].get(cam)
                img = cv2.imread(f"{root}/" + p) if p else None
                if img is None:
                    continue
                img = cv2.resize(img, (760, 428))
                y0 = 16 + row * 440
                frame[y0:y0 + 428, 10:770] = img
                cv2.putText(frame, cam, (16, y0 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (240, 240, 240), 1, cv2.LINE_AA)
            # BEV underlay from lane raster GT
            gt = cv2.imread(f"{root}/" + f.get("gt_vec", "_"), 0)
            if gt is not None:
                r0 = int((80 - XMAX) / 0.2)
                c0 = int((50 - YH) / 0.2)
                bev = (GTP[gt[r0:r0 + RH, c0:c0 + RW]][:, :, ::-1]
                       * 0.35).astype(np.uint8)
            else:
                bev = np.zeros((RH, RW, 3), np.uint8)
            nch = int(lg["n"][fi])
            pts = lg["pts"][fi].astype(np.float32)
            cls = lg["cls"][fi]
            adj = lg["adj"][fi]
            mids = {}
            for i in range(nch):
                poly = [cell(x, y)[::-1] for x, y in pts[i]]
                cv2.polylines(bev, [np.array(poly, np.int32).reshape(-1, 1, 2)],
                              False, CCOL.get(int(cls[i]), (200, 200, 200)), 2,
                              cv2.LINE_AA)
                cv2.circle(bev, poly[0], 3, (255, 255, 255), -1)
                mids[i] = poly[len(poly) // 2]
            for i in range(nch):
                for j in range(i + 1, nch):
                    if adj[i, j]:
                        cv2.line(bev, mids[i], mids[j], (255, 255, 255), 1,
                                 cv2.LINE_AA)
            # agents + velocity arrows (occ-flow GT source)
            atp = f"{root}/" + f.get("agent_traj", "_")
            if os.path.exists(atp):
                z = np.load(atp)
                for k in range(int(z["count"])):
                    cl_, xe, ye, l, w, yaw = z["boxes"][k]
                    if not (XMIN < xe < XMAX and abs(ye) < YH):
                        continue
                    cb, sb = np.cos(yaw), np.sin(yaw)
                    cor = [cell(xe + lx * cb - wy * sb, ye + lx * sb + wy * cb)[::-1]
                           for lx, wy in ((l/2, w/2), (l/2, -w/2),
                                          (-l/2, -w/2), (-l/2, w/2))]
                    col = (0, 215, 255) if cl_ < 1.5 else (255, 0, 255)
                    cv2.polylines(bev, [np.array(cor, np.int32).reshape(-1, 1, 2)],
                                  True, col, 1)
                    if z["tvalid"][k, 0] > 0.5:
                        vx, vy = z["traj"][k, 0] / 0.5
                        sp = float(np.hypot(vx, vy))
                        if sp > 0.4:
                            p0 = cell(xe, ye)[::-1]
                            p1 = cell(xe + vx * 1.2, ye + vy * 1.2)[::-1]
                            cv2.arrowedLine(bev, p0, p1, (0, 255, 0), 2,
                                            cv2.LINE_AA, tipLength=0.3)
            # ego GT trajectory (3 s)
            try:
                eg = np.load(f"{root}/ego_motion.npz")
                if fi < len(eg["v0"]) and eg["valid"][fi] > 0:
                    pp = [cell(0, 0)[::-1]]
                    for xe, ye in eg["wp"][fi]:
                        if not (XMIN < xe < XMAX and abs(ye) < YH):
                            break
                        pp.append(cell(xe, ye)[::-1])
                    cv2.polylines(bev,
                                  [np.array(pp, np.int32).reshape(-1, 1, 2)],
                                  False, (255, 255, 255), 2, cv2.LINE_AA)
            except Exception:
                pass
            er, ec = cell(0, 0)
            cv2.drawMarker(bev, (ec, er), (255, 255, 255),
                           cv2.MARKER_TRIANGLE_UP, 14, 2)
            bev = cv2.resize(bev, (PW, PH), interpolation=cv2.INTER_NEAREST)
            x0 = VW - PW - 14
            frame[34:34 + PH, x0:x0 + PW] = bev
            cv2.putText(frame, "GT: LANE GRAPH (vector) + FLOW (arrows)",
                        (x0, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "cyan=laneline orange=road_edge red=stopline "
                        "white=adjacency/ego-path green=velocity", (x0, VH - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1,
                        cv2.LINE_AA)
            cv2.putText(frame, f"{scene.split('+0900_')[-1]}  f{fi:03d}  "
                        f"chains={nch}  |  GT only - no model", (12, VH - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                        cv2.LINE_AA)
            vw.write(frame)
            n += 1
        print(f"{scene} done ({n})", flush=True)
    vw.release()
    subprocess.run(["ffmpeg", "-y", "-i", raw, "-c:v", "libx264", "-crf", "24",
                    "-pix_fmt", "yuv420p", args.out], check=True,
                   capture_output=True)
    os.remove(raw)
    print("done", n, args.out, flush=True)


if __name__ == "__main__":
    main()

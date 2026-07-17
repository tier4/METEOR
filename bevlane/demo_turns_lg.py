#!/usr/bin/env python3
"""Intersection-turn demo: camera + BEV (K=3 ego paths) + LANE GRAPH panel.

Layout (1920x1080):
  left   : CAM_FRONT_WIDE (top) + CAM_FRONT_NARROW (bottom)
  middle : predicted BEV lane map +-40x+-25 m with 3D boxes and the K=3
           ego hypotheses (best = bold green, others = cyan + confidence)
  right  : predicted VECTOR LANE GRAPH (ROI x -10..60, |y|<=25): chains
           coloured by class, brightness by existence confidence, white
           adjacency links; faint grey = GT chains for reference.
Scene banner marks TRAIN / VAL / UNUSED so evaluation stays honest.
"""
import argparse
import json
import os
import subprocess
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from autolabel_bev import PALETTE  # noqa: E402
from bevlane.dataset import BevLaneDataset  # noqa: E402
from bevlane.model import MODELS, make_warp_theta  # noqa: E402

GTP = np.zeros((256, 3), np.uint8)
GTP[:len(PALETTE)] = PALETTE
CCOL = {0: (255, 255, 0), 1: (0, 165, 255), 2: (0, 0, 255)}


def membership(scene, train_list):
    if "2026-01-23T15-26-01" in scene:
        return "VAL (held-out)"
    return "TRAIN" if scene in train_list else "UNUSED"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--out", default="out/demo_turns_lg.mp4")
    ap.add_argument("--train-list", default="out/round21_scenes.txt")
    ap.add_argument("--fps", type=int, default=15)
    args = ap.parse_args()
    train_list = set(open(args.train_list).read().split()) \
        if os.path.exists(args.train_list) else set()

    m = MODELS["v29"](n_seg=21).cuda().eval()
    sd = torch.load(args.ckpt, map_location="cpu")["model"]
    cur = m.state_dict()
    sd = {k: v for k, v in sd.items() if k in cur and cur[k].shape == v.shape}
    m.load_state_dict(sd, strict=False)

    VW, VH = 1920, 1080
    raw = args.out.replace(".mp4", "_raw.mp4")
    vw = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
                         (VW, VH))
    n = 0
    for scene in args.scenes:
        ds = BevLaneDataset("out/bevlane", [scene], gt_key="gt_vec",
                            with_temporal=True, temporal_hist=3)
        mem = membership(scene, train_list)
        lg_gt = None
        try:
            lg_gt = np.load(f"out/bevlane/{scene}/lanegraph.npz")
        except Exception:
            pass
        ego_np = None
        try:
            ego_np = np.load(f"out/bevlane/{scene}/ego_motion.npz")
        except Exception:
            pass
        for i in range(len(ds)):
            b = ds[i]
            imgs = b[0][None].cuda()
            K, Tc = b[1][None].cuda(), b[2][None].cuda()
            hi, hr, hv = b[-3][None].cuda(), b[-2][None].cuda(), b[-1][None].cuda()
            with torch.no_grad(), torch.autocast("cuda", torch.float16):
                pbs, ths = [], []
                for s_ in range(3):
                    pbs.append(m.compute_bev(hi[:, s_], K, Tc)
                               * hv[:, s_].view(-1, 1, 1, 1))
                    ths.append(make_warp_theta(hr[:, s_]))
                fi0 = ds.items[i][1]["frame"]
                v0 = 0.0
                if ego_np is not None and fi0 < len(ego_np["v0"]):
                    v0 = float(ego_np["v0"][fi0])
                out = m(imgs, K, Tc,
                        torch.tensor([v0], device="cuda"),
                        torch.stack(pbs, 1).float(), torch.stack(ths, 1))
            frame = np.zeros((VH, VW, 3), np.uint8)
            # ---- cameras ----
            s_, f_ = ds.items[i]
            man_f = f_
            for row, cam in enumerate(("CAM_FRONT_WIDE", "CAM_FRONT_NARROW")):
                p = man_f["imgs"].get(cam)
                img = cv2.imread(f"out/bevlane/{scene}/" + p) if p else None
                if img is None:
                    continue
                img = cv2.resize(img, (700, 394))
                y0 = 60 + row * 410
                frame[y0:y0 + 394, 10:710] = img
                cv2.putText(frame, cam, (16, y0 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240),
                            1, cv2.LINE_AA)
            # ---- middle: BEV +-40 x +-25 with K=3 paths ----
            lane = out[0][0].float().argmax(0).cpu().numpy().astype(np.uint8)
            bev = GTP[lane][:, :, ::-1].copy()
            bev = bev[200:600, 125:375]                     # 400x250
            PH = 860
            PW = int(PH * 250 / 400)
            bev = cv2.resize(bev, (PW, PH), interpolation=cv2.INTER_NEAREST)
            sx, sy = PW / 250.0, PH / 400.0
            xy2px = lambda x, y: (int(((50.0 - y) / 0.2 - 125) * sx),
                                  int(((80.0 - x) / 0.2 - 200) * sy))
            dets = m.decode_boxes(out[3].float(), out[4].float(),
                                  thresh=0.3, topk=64)[0]
            stat = out[10][0, 0].float().cpu()
            for cls, sc, xe, ye, l, w, yaw in dets:
                if sc < (0.45 if cls == 0 else 0.3):
                    continue
                if not (-40 < xe < 40 and abs(ye) < 25):
                    continue
                cb, sb = np.cos(yaw), np.sin(yaw)
                cor = [xy2px(xe + lx * cb - wy * sb, ye + lx * sb + wy * cb)
                       for lx, wy in ((l/2, w/2), (l/2, -w/2),
                                      (-l/2, -w/2), (-l/2, w/2))]
                ri0 = int((80.0 - xe) / 0.4)
                ci0 = int((50.0 - ye) / 0.4)
                st = 0 <= ri0 < 400 and 0 <= ci0 < 250 and \
                    float(stat[ri0, ci0]) > 0
                col = (160, 160, 160) if st else \
                    ((0, 215, 255) if cls == 0 else (255, 0, 255))
                cv2.polylines(bev, [np.array(cor, np.int32).reshape(-1, 1, 2)],
                              True, col, 2)
            e = out[7][0].float().cpu().numpy()
            paths = e[:36].reshape(3, 6, 2)
            conf = np.exp(e[36:39]) / np.exp(e[36:39]).sum()
            kb = int(conf.argmax())
            for k in range(3):
                pts = [xy2px(0, 0)]
                for x, y in paths[k]:
                    if not (-40 < x < 40 and abs(y) < 25):
                        break
                    pts.append(xy2px(float(x), float(y)))
                if len(pts) < 2:
                    continue
                col = (0, 255, 0) if k == kb else (255, 200, 60)
                th = 4 if k == kb else 2
                cv2.polylines(bev, [np.array(pts, np.int32).reshape(-1, 1, 2)],
                              False, col, th, cv2.LINE_AA)
                cv2.putText(bev, f"{conf[k]:.2f}", pts[-1],
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2,
                            cv2.LINE_AA)
            er, ec = xy2px(0, 0)
            cv2.drawMarker(bev, (er, ec), (255, 255, 255),
                           cv2.MARKER_TRIANGLE_UP, 16, 2)
            x_b = 730
            frame[70:70 + PH, x_b:x_b + PW] = bev
            cv2.putText(frame, "pred BEV +-40x+-25m  |  E2E K=3 paths",
                        (x_b, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2, cv2.LINE_AA)
            # ---- right: lane-graph panel (ROI x -10..60, |y|<=25) ----
            LH, LW = 350, 250
            PH2 = 860
            PW2 = int(PH2 * LW / LH)
            lg = np.zeros((LH * 2, LW * 2, 3), np.uint8)   # 2x supersample
            g2px = lambda x, y: (int((25.0 - y) / 0.2 * 2),
                                 int((60.0 - x) / 0.2 * 2))
            if lg_gt is not None and man_f["frame"] < len(lg_gt["n"]):
                fi_ = man_f["frame"]
                for j in range(int(lg_gt["n"][fi_])):
                    poly = [g2px(x, y) for x, y in
                            lg_gt["pts"][fi_][j].astype(np.float32)]
                    cv2.polylines(lg, [np.array(poly, np.int32
                                                ).reshape(-1, 1, 2)],
                                  False, (70, 70, 70), 2, cv2.LINE_AA)
            pts_p = out[14][0].float().cpu().numpy()       # [24,12,2]
            meta = out[15][0].float().cpu().numpy()        # [24,4]
            adj = out[16][0].float().cpu().numpy()
            ex = 1 / (1 + np.exp(-meta[:, 0]))
            mids = {}
            for j in range(24):
                if ex[j] < 0.3:
                    continue
                ci_ = int(meta[j, 1:].argmax())
                col = CCOL.get(ci_, (200, 200, 200))
                a = 0.4 + 0.6 * min(float(ex[j]), 1.0)
                col = tuple(int(c * a) for c in col)
                poly = [g2px(x, y) for x, y in pts_p[j]]
                cv2.polylines(lg, [np.array(poly, np.int32).reshape(-1, 1, 2)],
                              False, col, 3, cv2.LINE_AA)
                mids[j] = poly[6]
            for j in mids:
                for j2 in mids:
                    if j2 > j and 1 / (1 + np.exp(-adj[j, j2])) > 0.5:
                        cv2.line(lg, mids[j], mids[j2], (255, 255, 255), 1,
                                 cv2.LINE_AA)
            er2 = g2px(0, 0)
            cv2.drawMarker(lg, er2, (255, 255, 255), cv2.MARKER_TRIANGLE_UP,
                           20, 2)
            lg = cv2.resize(lg, (PW2, PH2), interpolation=cv2.INTER_AREA)
            x_l = x_b + PW + 20
            frame[70:70 + PH2, x_l:x_l + PW2] = lg
            cv2.putText(frame, "pred LANE GRAPH (x -10..60m)",
                        (x_l, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "cyan=laneline orange=edge red=stopline "
                        "white=adjacency grey=GT", (x_l, VH - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1,
                        cv2.LINE_AA)
            cv2.putText(frame, f"{scene.split('+0900_')[-1]}  "
                        f"f{man_f['frame']:03d}  [{mem}]  "
                        f"v0={v0 * 3.6:.0f}km/h", (12, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (0, 255, 0) if "VAL" in mem else (0, 200, 255), 2,
                        cv2.LINE_AA)
            vw.write(frame)
            n += 1
        print(f"{scene} done ({n})", flush=True)
    vw.release()
    subprocess.run(["ffmpeg", "-y", "-i", raw, "-c:v", "libx264", "-crf",
                    "24", "-pix_fmt", "yuv420p", args.out], check=True,
                   capture_output=True)
    os.remove(raw)
    print("done", n, args.out, flush=True)


if __name__ == "__main__":
    main()

"""GT BEV Seg + 推論/GT BBox に FRONT_WIDE / FRONT_NARROW を並べた検証動画。

左列: FRONT_WIDE (上) / FRONT_NARROW (下) に 3D 箱を投影
      (黄 = 推論、白 = GT。z=0 接地・高さ 1.8 m の近似ワイヤーフレーム)
右列: GT BEV Seg + 同じ箱
「GT Seg 上でも箱が左車線寄りに見える」件を、実画像の車両位置と
突き合わせて確認するための映像。
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bevlane.dataset import BevLaneDataset                        # noqa: E402
from bevlane.model import MODELS                                  # noqa: E402
from deploy.viz_np import PALETTE                                 # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--model", default="v52")
ap.add_argument("--list", required=True)
ap.add_argument("--root", required=True)
ap.add_argument("--n-scenes", type=int, default=12)
ap.add_argument("--stride", type=int, default=2)
ap.add_argument("--fps", type=int, default=8)
ap.add_argument("--box-h", type=float, default=1.8)
ap.add_argument("--out", default="out/gtseg_predbox_cam.mp4")
a = ap.parse_args()

scenes = [l.strip() for l in open(a.list) if l.strip()][:a.n_scenes]
ds = BevLaneDataset(a.root, scenes, gt_key="gt_cons", with_boxdet=True,
                    max_per_scene=200, n_cams=8)
m = MODELS[a.model](n_seg=21).cuda().eval()
sd = torch.load(a.ckpt, map_location="cpu")
sd = {k.replace("module.", ""): v for k, v in sd.get("model", sd).items()}
cur = m.state_dict()
m.load_state_dict({k: v for k, v in sd.items()
                   if k in cur and cur[k].shape == v.shape}, strict=False)

by_scene = {}
for i, (s, f) in enumerate(ds.items):
    by_scene.setdefault(s, []).append((int(f["frame"]), i))


def corners3d(xe, ye, ln, wd, yw, h):
    c, s2 = np.cos(yw), np.sin(yw)
    pts = []
    for dx, dy in ((ln / 2, wd / 2), (ln / 2, -wd / 2),
                   (-ln / 2, -wd / 2), (-ln / 2, wd / 2)):
        x, y = xe + c * dx - s2 * dy, ye + s2 * dx + c * dy
        pts.append((x, y, 0.0))
        pts.append((x, y, h))
    return np.array(pts)                       # [8,3] 下上交互


EDGES = [(0, 2), (2, 4), (4, 6), (6, 0),       # 底面
         (1, 3), (3, 5), (5, 7), (7, 1),       # 天面
         (0, 1), (2, 3), (4, 5), (6, 7)]       # 柱


def draw_cam_box(img, pts_ego, K, T_cam_ego, col, th=2):
    P = (T_cam_ego @ np.concatenate(
        [pts_ego, np.ones((8, 1))], 1).T).T[:, :3]
    if (P[:, 2] < 0.5).all():
        return
    uv = (K @ P.T).T
    uv = uv[:, :2] / np.clip(uv[:, 2:3], 1e-3, None)
    Hh, Ww = img.shape[:2]
    for i, j in EDGES:
        if P[i, 2] < 0.5 or P[j, 2] < 0.5:
            continue
        p1 = (int(uv[i, 0]), int(uv[i, 1]))
        p2 = (int(uv[j, 0]), int(uv[j, 1]))
        if (max(abs(p1[0]), abs(p2[0])) > 4 * Ww
                or max(abs(p1[1]), abs(p2[1])) > 4 * Hh):
            continue
        cv2.line(img, p1, p2, col, th, cv2.LINE_AA)


def draw_bev_box(img, xe, ye, ln, wd, yw, col, th=2):
    c, s2 = np.cos(yw), np.sin(yw)
    pts = []
    for dx, dy in ((ln / 2, wd / 2), (ln / 2, -wd / 2),
                   (-ln / 2, -wd / 2), (-ln / 2, wd / 2)):
        x, y = xe + c * dx - s2 * dy, ye + s2 * dx + c * dy
        pts.append((int((50.0 - y) / 0.2), int((80.0 - x) / 0.2)))
    cv2.polylines(img, [np.array(pts, np.int32)], True, col, th)


CAM2 = ["CAM_FRONT_WIDE", "CAM_FRONT_NARROW"]
vw = None
n_out = 0
for si, s in enumerate(scenes):
    man = json.load(open(os.path.join(a.root, s, "manifest.json")))
    frames_by_fi = {int(f["frame"]): f for f in man["frames"]}
    cams = {}
    for c in CAM2:
        cam = man["cams"].get(c)
        if cam is not None:
            cams[c] = (np.array(cam["K"], np.float64),
                       np.linalg.inv(np.array(cam["T_ego_cam"], np.float64)))
    lst = sorted(by_scene.get(s, []))
    for fi, di in lst[::a.stride]:
        b = ds[di]
        if b is None or fi not in frames_by_fi:
            continue
        with torch.no_grad(), torch.autocast("cuda", torch.float16):
            out = m(b[0][None].cuda(), b[1][None].cuda(), b[2][None].cuda())
        dets = m.decode_boxes(out[3].float().cpu(), out[4].float().cpu(),
                              thresh=0.25)[0]
        pred_boxes = [[float(v) for v in d[2:7]] for d in dets
                      if float(d[0]) < 1.5]
        bx, nb = b[4], int(b[5])
        gt_boxes = []
        for k in range(max(nb, 0)):
            cls, xe, ye, ln, wd, yw = [float(v) for v in bx[k][:6]]
            if ln > 0 and cls < 1.5:
                gt_boxes.append([xe, ye, ln, wd, yw])
        # --- カメラパネル
        cam_imgs = []
        fmeta = frames_by_fi[fi]
        for c in CAM2:
            p = os.path.join(a.root, s, fmeta["imgs"].get(c, "_"))
            im = cv2.imread(p)
            if im is None or c not in cams:
                im = np.zeros((432, 768, 3), np.uint8)
            else:
                K, Tce = cams[c]
                for bb in pred_boxes:
                    draw_cam_box(im, corners3d(*bb, a.box_h), K, Tce,
                                 (0, 255, 255), 2)
                for bb in gt_boxes:
                    draw_cam_box(im, corners3d(*bb, a.box_h), K, Tce,
                                 (255, 255, 255), 1)
            cv2.putText(im, c.replace("CAM_", ""), (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cam_imgs.append(im)
        cam_col = np.concatenate(cam_imgs, 0)          # 768x864
        # --- BEV パネル (GT seg)
        gt = b[3].numpy()
        gt = np.where(gt == 255, 0, gt).astype(np.uint8)
        bev = PALETTE[gt][:, :, ::-1].astype(np.uint8).copy()
        for bb in pred_boxes:
            draw_bev_box(bev, *bb, (0, 255, 255), 2)
        for bb in gt_boxes:
            draw_bev_box(bev, *bb, (255, 255, 255), 1)
        for x_m in (-40, -20, 20, 40):
            r = int((80.0 - x_m) / 0.2)
            if 0 <= r < bev.shape[0]:
                cv2.line(bev, (0, r), (bev.shape[1], r), (80, 80, 80), 1)
        cv2.drawMarker(bev, (250, 400), (0, 255, 0),
                       cv2.MARKER_TRIANGLE_UP, 14, 2)
        # --- 合成
        H = max(cam_col.shape[0], bev.shape[0])
        canvas = np.zeros((H + 52, cam_col.shape[1] + bev.shape[1] + 8, 3),
                          np.uint8)
        cv2.putText(canvas, f"{s[:44]}  f{fi:03d}", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        cv2.putText(canvas, "yellow = pred BBox | white(thin) = GT BBox | "
                    "right = GT BEV Seg", (8, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        canvas[52:52 + cam_col.shape[0], :cam_col.shape[1]] = cam_col
        canvas[52:52 + bev.shape[0], cam_col.shape[1] + 8:] = bev
        if vw is None:
            vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"),
                                 a.fps, (canvas.shape[1], canvas.shape[0]))
        vw.write(canvas)
        n_out += 1
    print(f"scene {si+1}/{len(scenes)} {s} 済 (計 {n_out} フレーム)",
          flush=True)
if vw is not None:
    vw.release()
print(f"saved {a.out} ({n_out} frames)")
print("GTSEG_CAM_DONE")

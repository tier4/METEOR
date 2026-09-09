"""Verification video overlaying predicted BBoxes on the GT BEV Seg raster (single panel).

Base = the GT (gt_cons) lane map. On top, draw only
  yellow = predicted 3D BBox / white = GT 3D BBox
so that whether predicted boxes sit in the lane center of the GT lanes (= are the boxes
right) can be checked directly on the GT map.
"""
import argparse
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
ap.add_argument("--out", default="out/gtseg_predbox.mp4")
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


def draw_box(img, xe, ye, ln, wd, yw, col, th=2):
    c, s2 = np.cos(yw), np.sin(yw)
    pts = []
    for dx, dy in ((ln / 2, wd / 2), (ln / 2, -wd / 2),
                   (-ln / 2, -wd / 2), (-ln / 2, wd / 2)):
        x, y = xe + c * dx - s2 * dy, ye + s2 * dx + c * dy
        pts.append((int((50.0 - y) / 0.2), int((80.0 - x) / 0.2)))
    cv2.polylines(img, [np.array(pts, np.int32)], True, col, th)


vw = None
n_out = 0
for si, s in enumerate(scenes):
    lst = sorted(by_scene.get(s, []))
    for fi, di in lst[::a.stride]:
        b = ds[di]
        if b is None:
            continue
        with torch.no_grad(), torch.autocast("cuda", torch.float16):
            out = m(b[0][None].cuda(), b[1][None].cuda(), b[2][None].cuda())
        gt = b[3].numpy()
        gt = np.where(gt == 255, 0, gt).astype(np.uint8)
        img = PALETTE[gt][:, :, ::-1].astype(np.uint8).copy()
        dets = m.decode_boxes(out[3].float().cpu(), out[4].float().cpu(),
                              thresh=0.25)[0]
        for d in dets:
            if float(d[0]) < 1.5:
                draw_box(img, *[float(v) for v in d[2:7]], (0, 255, 255), 2)
        bx, nb = b[4], int(b[5])
        for k in range(max(nb, 0)):
            cls, xe, ye, ln, wd, yw = [float(v) for v in bx[k][:6]]
            if ln > 0 and cls < 1.5:
                draw_box(img, xe, ye, ln, wd, yw, (255, 255, 255), 1)
        for x_m in (-40, -20, 20, 40):
            r = int((80.0 - x_m) / 0.2)
            if 0 <= r < img.shape[0]:
                cv2.line(img, (0, r), (img.shape[1], r), (80, 80, 80), 1)
        # ego marker
        cv2.drawMarker(img, (250, 400), (0, 255, 0),
                       cv2.MARKER_TRIANGLE_UP, 14, 2)
        head = np.zeros((52, img.shape[1], 3), np.uint8)
        cv2.putText(head, f"{s[:40]}  f{fi:03d}", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
        cv2.putText(head, "base = GT BEV Seg | yellow = pred BBox | "
                    "white = GT BBox", (8, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        frame = np.concatenate([head, img], 0)
        if vw is None:
            vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"),
                                 a.fps, (frame.shape[1], frame.shape[0]))
        vw.write(frame)
        n_out += 1
    print(f"scene {si+1}/{len(scenes)} {s} done ({n_out} frames total)",
          flush=True)
if vw is not None:
    vw.release()
print(f"saved {a.out} ({n_out} frames)")
print("GTSEG_PREDBOX_DONE")

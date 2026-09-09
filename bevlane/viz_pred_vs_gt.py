"""Verification image showing the pred BEV and GT of one frame side by side.

Left: predicted seg + predicted boxes (yellow) + GT boxes (white)
Right: GT raster + the same boxes
Settles by eye whether the "other vehicles hug the left lane line" effect comes
from pred or from GT.
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
ap.add_argument("--scene", required=True)
ap.add_argument("--root", required=True)
ap.add_argument("--frames", type=int, nargs="*", default=[40, 80, 120])
ap.add_argument("--out", default="out/viz_pred_vs_gt")
a = ap.parse_args()

ds = BevLaneDataset(a.root, [a.scene], gt_key="gt_cons", with_boxdet=True,
                    max_per_scene=200, n_cams=8)
m = MODELS[a.model](n_seg=21).cuda().eval()
sd = torch.load(a.ckpt, map_location="cpu")
sd = {k.replace("module.", ""): v for k, v in sd.get("model", sd).items()}
cur = m.state_dict()
m.load_state_dict({k: v for k, v in sd.items()
                   if k in cur and cur[k].shape == v.shape}, strict=False)

idx = {int(f["frame"]): i for i, (s, f) in enumerate(ds.items)}


def draw_box(img, xe, ye, ln, wd, yw, col):
    c, s2 = np.cos(yw), np.sin(yw)
    pts = []
    for dx, dy in ((ln / 2, wd / 2), (ln / 2, -wd / 2),
                   (-ln / 2, -wd / 2), (-ln / 2, wd / 2)):
        x, y = xe + c * dx - s2 * dy, ye + s2 * dx + c * dy
        pts.append((int((50.0 - y) / 0.2), int((80.0 - x) / 0.2)))
    cv2.polylines(img, [np.array(pts, np.int32)], True, col, 2)


for fi in a.frames:
    if fi not in idx:
        continue
    b = ds[idx[fi]]
    if b is None:
        continue
    with torch.no_grad(), torch.autocast("cuda", torch.float16):
        out = m(b[0][None].cuda(), b[1][None].cuda(), b[2][None].cuda())
    pred = out[0].float().argmax(1)[0].cpu().numpy().astype(np.uint8)
    gt = b[3].numpy().astype(np.int64)
    gt_img = np.where(gt == 255, 0, gt).astype(np.uint8)
    L = PALETTE[pred][:, :, ::-1].astype(np.uint8).copy()
    R = PALETTE[gt_img][:, :, ::-1].astype(np.uint8).copy()
    dets = m.decode_boxes(out[3].float().cpu(), out[4].float().cpu(),
                          thresh=0.25)[0]
    for d in dets:
        if float(d[0]) < 1.5:
            draw_box(L, *[float(v) for v in d[2:7]], (0, 255, 255))
            draw_box(R, *[float(v) for v in d[2:7]], (0, 255, 255))
    bx, nb = b[4], int(b[5])
    for k in range(max(nb, 0)):
        cls, xe, ye, ln, wd, yw = [float(v) for v in bx[k][:6]]
        if ln > 0 and cls < 1.5:
            draw_box(L, xe, ye, ln, wd, yw, (255, 255, 255))
            draw_box(R, xe, ye, ln, wd, yw, (255, 255, 255))
    for img, t in ((L, "pred seg"), (R, "GT")):
        for x_m in (20, 40):
            r = int((80.0 - x_m) / 0.2)
            cv2.line(img, (0, r), (img.shape[1], r), (90, 90, 90), 1)
            r = int((80.0 + x_m) / 0.2)
            if r < img.shape[0]:
                cv2.line(img, (0, r), (img.shape[1], r), (90, 90, 90), 1)
        cv2.putText(img, t, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)
    cat = np.concatenate([L, np.full((L.shape[0], 6, 3), 60, np.uint8), R], 1)
    p = f"{a.out}_{fi:03d}.png"
    cv2.imwrite(p, cat)
    print("saved", p, f"(pred boxes={len(dets)} GT boxes={nb})")
print("VIZ_DONE")

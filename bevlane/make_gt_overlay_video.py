"""予測 BEV に GT レーン線・GT 箱を重ねた検証動画。

「他車両が左レーン線スレスレに見える」件の証拠映像:
  左パネル: 予測 seg + 予測箱(黄) + GT 箱(白)   … デモ動画と同じ見え方
  右パネル: 同じ予測 seg に GT レーン線(緑)・GT 停止線(赤)を上書き
右パネルで、白= GT 箱が緑= GT 線の車線中央に収まる一方、予測の白線が
そこからずれている(=線側の誤差)ことを 1 本の動画で確認できる。
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
ap.add_argument("--out", default="out/gt_overlay.mp4")
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
        pred = out[0].float().argmax(1)[0].cpu().numpy().astype(np.uint8)
        gt = b[3].numpy()
        gt = np.where(gt == 255, 0, gt).astype(np.uint8)
        base = PALETTE[pred][:, :, ::-1].astype(np.uint8)
        L, R = base.copy(), base.copy()
        # 右パネル: GT 線を上書き (緑=レーン線, 赤=停止線, 橙=道路端)
        R[gt == 4] = (0, 255, 0)
        R[gt == 5] = (0, 0, 255)
        R[gt == 6] = (0, 165, 255)
        dets = m.decode_boxes(out[3].float().cpu(), out[4].float().cpu(),
                              thresh=0.25)[0]
        for d in dets:
            if float(d[0]) < 1.5:
                for img in (L, R):
                    draw_box(img, *[float(v) for v in d[2:7]],
                             (0, 255, 255), 1)
        bx, nb = b[4], int(b[5])
        for k in range(max(nb, 0)):
            cls, xe, ye, ln, wd, yw = [float(v) for v in bx[k][:6]]
            if ln > 0 and cls < 1.5:
                for img in (L, R):
                    draw_box(img, xe, ye, ln, wd, yw, (255, 255, 255), 2)
        for img in (L, R):
            for x_m in (-40, -20, 20, 40):
                r = int((80.0 - x_m) / 0.2)
                if 0 <= r < img.shape[0]:
                    cv2.line(img, (0, r), (img.shape[1], r), (80, 80, 80), 1)
        cat = np.concatenate(
            [L, np.full((L.shape[0], 8, 3), 50, np.uint8), R], 1)
        head = np.zeros((56, cat.shape[1], 3), np.uint8)
        cv2.putText(head, f"{s[:44]}  f{fi:03d}", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        cv2.putText(head, "pred only", (8, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(head, "+ GT lines: green=lane red=stop orange=edge | "
                    "white box=GT yellow=pred", (516, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        frame = np.concatenate([head, cat], 0)
        if vw is None:
            vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"),
                                 a.fps, (frame.shape[1], frame.shape[0]))
        vw.write(frame)
        n_out += 1
    print(f"scene {si+1}/{len(scenes)} {s} 済 (計 {n_out} フレーム)",
          flush=True)
if vw is not None:
    vw.release()
print(f"saved {a.out} ({n_out} frames)")
print("GT_OVERLAY_DONE")

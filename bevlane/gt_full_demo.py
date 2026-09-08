#!/usr/bin/env python3
"""GT だけで Pred デモと同一フォーマットの動画を作る (2026-08-29)。

レイアウトは実機デモ準拠: 8 カメラ (GT 2Dセグ + GT 箱) / 8 深度 (GT) /
OCC ボクセル (GT, 左下) / BEV (gt_cons + GT 箱 + GT 他車軌跡 + GT 自車経路)。
モデル推論は一切使わない。

  python3 bevlane/gt_full_demo.py --root out/bevlane_okinawa \
      --list scenes.txt --out out/demo_okinawa_gtfull.mp4 [--stride 2]
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deploy.viz_np import DEMO_PALETTE, crop_bev_np, draw_boxes_on_rgb  # noqa
from deploy.occ_iso import cube_render_fast                             # noqa

ORD = ["CAM_FRONT_WIDE", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
       "CAM_FRONT_NARROW", "CAM_BACK_LEFT", "CAM_BACK_WIDE",
       "CAM_BACK_RIGHT", "CAM_BACK_NARROW"]          # 表示タイル順
CAMS6 = ["CAM_FRONT_WIDE", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
         "CAM_BACK_WIDE", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]  # depth_gt4 順
VW, VH = 1920, 1080
CW, CH = 358, 200


def turbo_depth(d):
    """metric depth [108,192] -> TURBO (無効=暗)。"""
    m = np.asarray(d, np.float32)
    valid = m > 0.5
    b = ((m - 1.0) / 1.25).clip(0, 63) / 63.0 * 255.0
    img = cv2.applyColorMap(b.astype(np.uint8), cv2.COLORMAP_TURBO)
    img[~valid] = (40, 20, 20)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--list", required=True)
    ap.add_argument("--out", default="out/demo_gtfull.mp4")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--n-scenes", type=int, default=4)
    a = ap.parse_args()

    scenes = [l.strip() for l in open(a.list) if l.strip()][:a.n_scenes]
    vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), 10,
                         (VW, VH))
    for sc in scenes:
        d = os.path.join(a.root, sc)
        man = json.load(open(os.path.join(d, "manifest.json")))
        try:
            ego = np.load(os.path.join(d, "ego_motion.npz"))
        except Exception:
            ego = None
        K = {c: np.array(man["cams"][c]["K"], np.float32)
             for c in man["cams"]}
        Tce = {c: np.linalg.inv(np.array(man["cams"][c]["T_ego_cam"],
                                         np.float32))
               for c in man["cams"]}
        for f in man["frames"][::a.stride]:
            fi = f["frame"]
            canvas = np.full((VH, VW, 3), 16, np.uint8)
            # --- GT 各種を読む (無いものは黙って省略) ---
            def _npz(sub, key):
                try:
                    return np.load(os.path.join(d, sub, f"{fi:04d}.npz"))[key]
                except Exception:
                    return None
            seg2d = _npz("seg2d21", "seg")
            dep6 = _npz("depth_gt4", "depth")
            dep2 = _npz("depth_gt4n", "depth")
            occ = _npz("occ", "occ")
            bev_box = _npz("bev_box", "boxes")
            atr = None
            try:
                atr = np.load(os.path.join(d, "agent_traj", f"{fi:04d}.npz"))
            except Exception:
                pass
            gt = cv2.imread(os.path.join(d, "gt_cons", f"{fi:04d}.png"), 0)
            if gt is None:
                gt = cv2.imread(os.path.join(d, f["gt"]), 0)
            # GT 3D 箱 -> dict (draw_boxes_on_rgb / BEV 共用)
            boxes = []
            if bev_box is not None:
                for b in bev_box:
                    if b[3] <= 0:
                        continue
                    boxes.append({"cls": "vehicle" if b[0] == 1 else "vru",
                                  "score": 1.0, "x": float(b[1]),
                                  "y": float(b[2]), "l": float(b[3]),
                                  "w": float(b[4]), "yaw": float(b[5]),
                                  "stationary": False})
            # --- 8 カメラ (GT seg2d 重畳 + GT 箱投影) ---
            seg_order = {c: i for i, c in enumerate(
                man.get("seg2d_cams", list(man["cams"].keys())))}
            for ti, ch in enumerate(ORD):
                r, c_ = divmod(ti, 4)
                x0, y0 = 8 + c_ * (CW + 8), 26 + r * (CH + 16)
                if ch not in man["cams"] or ch not in f["imgs"]:
                    cv2.putText(canvas, f"{ch} (blank)", (x0, y0 + 20), 0,
                                0.5, (120, 120, 120), 1)
                    continue
                img = cv2.imread(os.path.join(d, f["imgs"][ch]))
                if img is None:
                    continue
                img = cv2.resize(img, (CW, CH))
                if seg2d is not None and ch in seg_order \
                        and seg_order[ch] < seg2d.shape[0]:
                    s2 = seg2d[seg_order[ch]]
                    pal = np.zeros((22, 3), np.uint8)
                    np.random.seed(3)
                    pal[1:] = np.random.randint(50, 255, (21, 3))
                    ov = cv2.resize(pal[s2][:, :, ::-1], (CW, CH),
                                    interpolation=cv2.INTER_NEAREST)
                    m_ = cv2.resize((s2 > 0).astype(np.uint8), (CW, CH),
                                    interpolation=cv2.INTER_NEAREST) > 0
                    img[m_] = (img[m_] * 0.55 + ov[m_] * 0.45).astype(np.uint8)
                if boxes and ch in K:
                    tup = [(b["cls"], 1.0, b["x"], b["y"], b["l"], b["w"],
                            b["yaw"]) for b in boxes]
                    draw_boxes_on_rgb(img, tup, K[ch], Tce[ch], CW, CH)
                canvas[y0:y0 + CH, x0:x0 + CW] = img
                cv2.putText(canvas, ch, (x0, y0 - 6), 0, 0.42,
                            (200, 200, 200), 1)
            # --- 8 深度 (GT) ---
            deps = {}
            if dep6 is not None:
                for i, ch in enumerate(CAMS6[:dep6.shape[0]]):
                    deps[ch] = dep6[i]
            if dep2 is not None:
                for i, ch in enumerate(["CAM_FRONT_NARROW",
                                        "CAM_BACK_NARROW"][:dep2.shape[0]]):
                    deps[ch] = dep2[i]
            for ti, ch in enumerate(ORD):
                r, c_ = divmod(ti, 4)
                x0 = 8 + c_ * (CW + 8)
                y0 = 26 + 2 * (CH + 16) + r * (CH + 12)
                if ti == 4:      # 左下スロットは OCC ボクセル (実機と同配置)
                    continue
                if ch in deps:
                    canvas[y0:y0 + CH, x0:x0 + CW] = cv2.resize(
                        turbo_depth(deps[ch]), (CW, CH))
            # --- OCC ボクセル (GT, 左下) ---
            if occ is not None:
                oc = occ[:10, 40:160, 40:160]
                iso = cube_render_fast(oc, W=CW, H=CH)
                x0, y0 = 8, 26 + 2 * (CH + 16) + (CH + 12)
                canvas[y0:y0 + CH, x0:x0 + CW] = iso
                cv2.putText(canvas, "GT OCC voxel +-24m", (x0, y0 - 4), 0,
                            0.42, (220, 220, 220), 1)
            # --- BEV (gt_cons + GT 箱 + GT 軌跡 + 自車経路) ---
            if gt is not None:
                g = gt.copy()
                g[g == 255] = 0
                pc = crop_bev_np(g, 80.0, 80.0, 60.0, 60.0, 25.0)
                BH2 = VH - 60
                BW2 = int(BH2 * pc.shape[1] / pc.shape[0])
                bev = cv2.resize(DEMO_PALETTE[pc][:, :, ::-1].astype(np.uint8),
                                 (BW2, BH2), interpolation=cv2.INTER_NEAREST)
                sy2 = BH2 / 120.0
                sx2 = BW2 / 50.0
                def to_px(x, y):
                    return int((25.0 - y) * sx2), int((60.0 - x) * sy2)
                for b in boxes:
                    col = (0, 215, 255) if b["cls"] == "vehicle" \
                        else (255, 0, 255)
                    cb, sb = np.cos(b["yaw"]), np.sin(b["yaw"])
                    pts = []
                    for dx, dy in ((b["l"]/2, b["w"]/2), (b["l"]/2, -b["w"]/2),
                                   (-b["l"]/2, -b["w"]/2), (-b["l"]/2, b["w"]/2)):
                        pts.append(to_px(b["x"] + cb*dx - sb*dy,
                                         b["y"] + sb*dx + cb*dy))
                    cv2.polylines(bev, [np.array(pts, np.int32)], True, col, 2)
                if atr is not None:
                    n_ag = int(atr["count"])
                    for i in range(min(n_ag, 64)):
                        bx = atr["boxes"][i]
                        tr = atr["traj"][i]; tv = atr["tvalid"][i]
                        pts = [to_px(bx[1], bx[2])]
                        for j in range(6):
                            if tv[j] < 0.5:
                                break
                            pts.append(to_px(bx[1] + tr[j, 0],
                                             bx[2] + tr[j, 1]))
                        if len(pts) > 1:
                            cv2.polylines(bev, [np.array(pts, np.int32)],
                                          False, (0, 200, 255), 1)
                if ego is not None and fi < len(ego["wp"]):
                    wp = ego["wp"][fi]
                    pts = [to_px(0, 0)] + [to_px(x, y) for x, y in wp]
                    cv2.polylines(bev, [np.array(pts, np.int32)], False,
                                  (0, 255, 0), 2)
                x0 = VW - BW2 - 8
                canvas[30:30 + BH2, x0:x0 + BW2] = bev
                if ego is not None and fi < len(ego["v0"]):
                    hud = [f"v0 {ego['v0'][fi]*3.6:5.1f} km/h",
                           f"steer {np.degrees(ego['steer'][fi]):+5.1f} deg",
                           f"accel {ego['acc'][fi]:+5.2f} m/s2",
                           f"brake {ego['brake'][fi]:4.2f}"]
                    for li, t_ in enumerate(hud):
                        cv2.putText(canvas, t_, (x0 + 10, VH - 130 + li * 26),
                                    0, 0.55, (240, 240, 240), 1)
            cv2.putText(canvas, f"METEOR GT-only  {sc[:28]}  f{fi:04d}"
                        "  (no prediction)", (8, 18), 0, 0.5,
                        (120, 220, 120), 1)
            vw.write(canvas)
        print(f"[ok] {sc}", flush=True)
    vw.release()
    print(f"[done] {a.out}")


if __name__ == "__main__":
    main()

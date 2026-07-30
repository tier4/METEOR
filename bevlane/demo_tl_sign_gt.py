"""GT demo: traffic-light state (from dataset color_shape annotations) +
road signs (from OSM SD map) rendered together.

Left pane: CAM_FRONT_WIDE with per-element traffic-light bboxes
(red/yellow/green, arrows get an orientation glyph). Right pane: GT BEV +
OSM SD-map overlay. HUD: ego traffic-light icon, speed-limit roundel
(OSM maxspeed of the nearest way), STOP sign and crossing proximity icons.

Converted frame i corresponds to raw keyframe 2*i (verified by pixel diff).
"""
import argparse
import json
import os
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from extract_sdmap import (GH, GW, build_scene_map, refine_alignment,
                           render_frame)

COLOR_BGR = {"red": (0, 0, 255), "yellow": (0, 200, 255),
             "green": (0, 210, 60)}


def load_tl(raw):
    """frame(front-cam keyframe idx) -> list of light elements."""
    A = os.path.join(raw, "annotation")
    cats = {c["token"]: c["name"] for c in json.load(open(A + "/category.json"))}
    sd = json.load(open(A + "/sample_data.json"))
    fn = {s["token"]: s.get("filename", "") for s in sd}
    out = {}
    for o in json.load(open(A + "/object_ann.json")):
        name = cats.get(o["category_token"], "")
        col = name.split("_")[0]
        if col not in COLOR_BGR:
            continue
        f = fn.get(o["sample_data_token"], "")
        cam = next((c for c in ("CAM_FRONT_WIDE", "CAM_FRONT_NARROW")
                    if c in f), None)
        if cam is None:
            continue
        ki = int(os.path.splitext(os.path.basename(f))[0])
        shape = name.split("_", 1)[1] if "_" in name else "circle"
        out.setdefault(ki, []).append(
            {"cam": cam, "bbox": o["bbox"], "color": col, "shape": shape,
             "orient": o.get("orientation")})
    return out


def draw_arrow_glyph(img, cx, cy, r, ang, color):
    """Annotation orientation convention (verified on real arrow crops):
    0 = up, +pi/2 = right (clockwise). Screen y grows down."""
    dx, dy = np.sin(ang), -np.cos(ang)
    p0 = (int(cx - dx * r), int(cy - dy * r))
    p1 = (int(cx + dx * r), int(cy + dy * r))
    cv2.arrowedLine(img, p0, p1, color, max(2, r // 4), tipLength=0.5)


def hud_traffic_light(img, x, y, elems, state=None, arrow=None):
    """3-lamp housing lit from CIRCLE lamps only; an arrow is a LIMITED
    permission (e.g. red + green-arrow = stop except that direction), so it
    is drawn separately and never lights the main lamps."""
    lamps = ["green", "yellow", "red"]
    cv2.rectangle(img, (x, y), (x + 150, y + 56), (60, 60, 60), -1)
    cv2.rectangle(img, (x, y), (x + 150, y + 56), (140, 140, 140), 2)
    ped = [e for e in elems if "pedestrian" in e["shape"]]
    act = state
    for i, c in enumerate(lamps):
        cc = (x + 28 + i * 48, y + 28)
        on = (act == c)
        col = COLOR_BGR[c] if on else tuple(v // 5 for v in COLOR_BGR[c])
        cv2.circle(img, cc, 18, col, -1)
        cv2.circle(img, cc, 18, (150, 150, 150), 1)
    if arrow is not None and arrow.get("orient") is not None:
        cv2.rectangle(img, (x + 160, y), (x + 200, y + 56), (60, 60, 60), -1)
        cv2.rectangle(img, (x + 160, y), (x + 200, y + 56), (140, 140, 140), 2)
        draw_arrow_glyph(img, x + 180, y + 28, 14, arrow["orient"],
                         COLOR_BGR[arrow["color"]])
        cv2.putText(img, "arrow only", (x + 160, y + 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 210, 60), 1)
    if ped:
        col = COLOR_BGR[max(ped, key=lambda e: (e["bbox"][2] - e["bbox"][0])
                            * (e["bbox"][3] - e["bbox"][1]))["color"]]
        cv2.putText(img, "PED", (x + 210, y + 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
    if act is None and not ped:
        cv2.putText(img, "no light", (x + 8, y + 76),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)


def hud_speed(img, x, y, kmh, r=26):
    cv2.circle(img, (x, y), r, (255, 255, 255), -1)
    cv2.circle(img, (x, y), r, (0, 0, 220), max(3, r // 5))
    txt = str(int(kmh))
    fs = r / 32.0
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
    cv2.putText(img, txt, (x - tw // 2, y + th // 2),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (10, 10, 10), 2)


def hud_stop(img, x, y, r=26, label=True):
    pts = []
    for k in range(8):
        a = np.pi / 8 + k * np.pi / 4
        pts.append([int(x + r * np.cos(a)), int(y + r * np.sin(a))])
    cv2.fillPoly(img, [np.array(pts)], (0, 0, 200))
    cv2.polylines(img, [np.array(pts)], True, (255, 255, 255), 2)
    if label:
        cv2.putText(img, "STOP", (x - int(r * 0.88), y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, r / 52.0, (255, 255, 255), 2)


def bev_signal_icon(img, x, y, state=None, arrow=None):
    """Mini 3-lamp housing at an OSM signal node. Circle state lights the
    lamp; an arrow (limited permission, NOT a green light) is drawn as a
    separate glyph beside the housing."""
    cv2.rectangle(img, (x - 14, y - 7), (x + 14, y + 7), (60, 60, 60), -1)
    cv2.rectangle(img, (x - 14, y - 7), (x + 14, y + 7), (230, 230, 230), 1)
    for i, c in enumerate(("green", "yellow", "red")):
        cc = (x - 9 + i * 9, y)
        if state == c:
            cv2.circle(img, cc, 4, COLOR_BGR[c], -1)
        else:
            cv2.circle(img, cc, 4, tuple(v // 4 for v in COLOR_BGR[c]), -1)
    if arrow is not None and arrow.get("orient") is not None:
        cv2.rectangle(img, (x + 16, y - 7), (x + 32, y + 7), (60, 60, 60), -1)
        draw_arrow_glyph(img, x + 24, y, 6, arrow["orient"],
                         COLOR_BGR[arrow["color"]])


def to_ego(pts, pose):
    x0, y0, yaw = pose
    c, s = np.cos(-yaw), np.sin(-yaw)
    P = np.asarray(pts, np.float64) - [x0, y0]
    return np.stack([c * P[:, 0] - s * P[:, 1],
                     s * P[:, 0] + c * P[:, 1]], 1)  # x fwd, y left


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--root", default="out/bevlane")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from autolabel_bev import PALETTE
    PAL = np.zeros((256, 3), np.uint8)
    PAL[:len(PALETTE)] = PALETTE

    ways, widths, inters, cross, err, meta = build_scene_map(args.raw)
    man = json.load(open(os.path.join(args.root, args.scene, "manifest.json")))
    eg = np.load(os.path.join(args.root, args.scene, "ego_motion.npz"))
    pose = eg["pose"].astype(np.float64)
    dx, dy, dth, iou = refine_alignment(ways, widths, pose, args.root,
                                        args.scene, man, cross=cross)
    print(f"[refine] iou={iou:.3f}", flush=True)
    pose = pose.copy()
    pose[:, 0] += dx
    pose[:, 1] += dy
    pose[:, 2] += dth

    tl = load_tl(args.raw)
    sigs = np.array(meta["signals"]) if meta["signals"] else np.zeros((0, 2))
    stops = np.array(meta["stops"]) if meta["stops"] else np.zeros((0, 2))

    BH = GH * 2                                   # 800
    vw = None
    last_state = [None, 999]
    last_arrow = [None, 999]
    for f in man["frames"]:
        fi = f["frame"]
        # --- BEV pane ---
        g = cv2.imread(os.path.join(args.root, args.scene, f["gt"]), 0)
        bev = PAL[np.where(g == 255, 0, g)][:, :, ::-1].copy()
        bev = cv2.resize(bev, (GW * 2, BH), interpolation=cv2.INTER_NEAREST)
        sd = render_frame(ways, widths, inters, cross, pose[fi])
        up = lambda a: cv2.resize(a * 255, (GW * 2, BH),
                                  interpolation=cv2.INTER_NEAREST)
        m = up(sd[0]) > 0
        bev[m] = bev[m] * 0.55 + np.array([80, 40, 0]) * 0.45
        bev[up(sd[1]) > 0] = (0, 200, 255)
        bev[up(sd[2]) > 0] = (255, 120, 255)
        bev[up(sd[3]) > 0] = (0, 255, 80)
        def bev_px(x_, y_):
            return int((50 - y_) / 0.4) * 2, int((80 - x_) / 0.4) * 2

        elems = tl.get(2 * fi, [])
        veh_el = [e for e in elems if "pedestrian" not in e["shape"]]

        def ego_score(e):
            # ego's own signal: large AND near the image centre column.
            # Pure max-area picks the cross street's light at intersections.
            x1, y1, x2, y2 = e["bbox"]
            area = (x2 - x1) * (y2 - y1)
            w_img = 2880.0
            off = abs((x1 + x2) / 2 - w_img / 2) / (w_img / 2)
            return area * (1.0 - 0.7 * off)

        circles = [e for e in veh_el if "circle" in e["shape"]]
        arrows = [e for e in veh_el if "arrow" in e["shape"]]
        ego_state = max(circles, key=ego_score)["color"] if circles else None
        ego_arrow = max(arrows, key=ego_score) if arrows else None
        # Hold the last state while a signal is still nearby: on approach
        # the lamps leave the camera FOV a few frames before the stop line.
        if ego_state is not None:
            last_state[0], last_state[1] = ego_state, 0
        else:
            last_state[1] += 1
        if ego_arrow is not None:
            last_arrow[0], last_arrow[1] = ego_arrow, 0
        else:
            last_arrow[1] += 1
            if last_arrow[0] is not None and last_arrow[1] <= 40:
                ego_arrow = last_arrow[0]
            near = False
            if len(sigs):
                e_s = to_ego(sigs, pose[fi])
                d_s = np.hypot(e_s[:, 0], e_s[:, 1])
                near = bool((d_s[e_s[:, 0] > -25] < 50).any())
            if last_state[0] and last_state[1] <= 40 and near:
                ego_state = last_state[0]

        # road signs in BEV: STOP octagons + speed roundels on tagged ways
        if len(stops):
            for x_, y_ in to_ego(stops, pose[fi]):
                q, r = bev_px(x_, y_)
                if 12 <= r < BH - 12 and 12 <= q < GW * 2 - 12:
                    hud_stop(bev, q, r, r=13, label=False)
        for seg in meta["maxspeed"]:
            e = to_ego(seg["pts"], pose[fi])
            # resample along the polyline every ~60 m for icon placement
            d = np.hypot(*np.diff(e, axis=0).T)
            cum = np.concatenate([[0], np.cumsum(d)])
            if cum[-1] < 1:
                continue
            for s_ in np.arange(30, cum[-1], 60):
                i = int(np.searchsorted(cum, s_)) - 1
                tt = (s_ - cum[i]) / max(cum[i + 1] - cum[i], 1e-9)
                x_, y_ = e[i] + tt * (e[i + 1] - e[i])
                q, r = bev_px(x_, y_)
                if 16 <= r < BH - 16 and 16 <= q < GW * 2 - 16:
                    hud_speed(bev, q, r, seg["kmh"], r=14)
        # signals in BEV: nearest-ahead one shows the annotated ego state
        if len(sigs):
            e = to_ego(sigs, pose[fi])
            dist = np.hypot(e[:, 0], e[:, 1])
            # keep the ego signal selected while passing under it: OSM
            # nodes sit at the intersection centre, slightly behind the
            # stop line when the ego is already braking at it.
            ahead = np.where(e[:, 0] > -15, dist, 1e9)
            k_ego = int(np.argmin(ahead)) if ahead.min() < 80 else -1
            for k, (x_, y_) in enumerate(e):
                q, r = bev_px(x_, y_)
                if 16 <= r < BH - 16 and 16 <= q < GW * 2 - 16:
                    bev_signal_icon(bev, q, r,
                                    ego_state if k == k_ego else None,
                                    arrow=ego_arrow if k == k_ego else None)
        # ego marker + range ticks: without them viewers read the ego at the
        # pane bottom and everything looks tens of meters too close
        er, ec = BH // 2, GW  # ego at (x=0,y=0) -> row 400, col 250
        cv2.fillPoly(bev, [np.array([[ec, er - 16], [ec - 9, er + 10],
                                     [ec + 9, er + 10]])], (255, 255, 255))
        cv2.polylines(bev, [np.array([[ec, er - 16], [ec - 9, er + 10],
                                      [ec + 9, er + 10]])], True, (0, 0, 0), 2)
        for dist in (20, 40, 60):
            rr = er - dist * 5              # 0.4 m/px, x2 upscale
            cv2.line(bev, (0, rr), (12, rr), (200, 200, 200), 2)
            cv2.putText(bev, f"+{dist}m", (14, rr + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(bev, "EGO", (ec - 18, er + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(bev, "GT BEV + OSM SD-map", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # --- camera pane ---
        img = cv2.imread(os.path.join(args.root, args.scene,
                                      f["imgs"]["CAM_FRONT_WIDE"]))
        elems = tl.get(2 * fi, [])
        # bbox coords are in RAW resolution; converted img may be resized
        raw0 = cv2.imread(os.path.join(args.raw,
                                       f"data/CAM_FRONT_WIDE/{2*fi:05d}.jpg"))
        sx = img.shape[1] / raw0.shape[1] if raw0 is not None else 1.0
        sy = img.shape[0] / raw0.shape[0] if raw0 is not None else 1.0
        for e in elems:
            if e["cam"] != "CAM_FRONT_WIDE":
                continue
            x1, y1, x2, y2 = e["bbox"]
            p1 = (int(x1 * sx), int(y1 * sy))
            p2 = (int(x2 * sx), int(y2 * sy))
            col = COLOR_BGR[e["color"]]
            cv2.rectangle(img, p1, p2, col, 2)
            if "arrow" in e["shape"] and e.get("orient") is not None:
                draw_arrow_glyph(img, p2[0] + 18, p1[1] + 9, 12,
                                 e["orient"], col)
        cam = cv2.resize(img, (int(img.shape[1] * BH / img.shape[0]), BH))

        # --- HUD ---
        hud_traffic_light(cam, 12, 12, elems, ego_state, ego_arrow)
        hx = 12
        ex_sig = to_ego(sigs, pose[fi]) if len(sigs) else np.zeros((0, 2))
        ex_stop = to_ego(stops, pose[fi]) if len(stops) else np.zeros((0, 2))
        near_ms = None
        best = 25.0
        for seg in meta["maxspeed"]:
            e = to_ego(seg["pts"], pose[fi])
            a, b = e[:-1], e[1:]
            ab = b - a
            L2 = (ab ** 2).sum(1).clip(min=1e-9)
            tt = (-(a * ab).sum(1) / L2).clip(0, 1)
            d = float(np.hypot(*(a + tt[:, None] * ab).T).min())
            if d < best:
                best, near_ms = d, seg["kmh"]
        if near_ms is not None:
            hud_speed(cam, 40, BH - 44, near_ms)
            hx = 90
        if len(ex_stop) and (np.hypot(ex_stop[:, 0], ex_stop[:, 1])
                             [ex_stop[:, 0] > -2].min(initial=1e9) < 30):
            hud_stop(cam, hx + 40, BH - 44)
        if len(ex_sig):
            d = float(np.hypot(ex_sig[:, 0], ex_sig[:, 1])
                      [ex_sig[:, 0] > -2].min(initial=1e9))
            if d < 60:
                cv2.putText(cam, f"signal {d:.0f}m", (12, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        frame = np.hstack([cam, bev])
        if vw is None:
            vw = cv2.VideoWriter(args.out.replace(".mp4", "_raw.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), 10,
                                 (frame.shape[1], frame.shape[0]))
        vw.write(frame)
    vw.release()
    subprocess.run(["ffmpeg", "-y", "-i", args.out.replace(".mp4", "_raw.mp4"),
                    "-c:v", "libx264", "-crf", "24", "-pix_fmt", "yuv420p",
                    args.out], check=True, capture_output=True)
    os.remove(args.out.replace(".mp4", "_raw.mp4"))
    print("->", args.out, flush=True)


if __name__ == "__main__":
    main()

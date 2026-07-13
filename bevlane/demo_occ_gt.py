#!/usr/bin/env python3
"""OCC GT visualisation video: FRONT RGB | top-down | isometric 3D voxels."""
import argparse
import json
import os
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bevlane.extract_occ import GX, GY, GZ, OCC_NAMES, OCC_PAL, VOX  # noqa: E402


def iso_render(occ, W=900, H=760):
    """Painter's-algorithm isometric voxel rendering (ego at centre)."""
    img = np.zeros((H, W, 3), np.uint8)
    zz, rr, cc = np.nonzero((occ > 0) & (occ != 255))
    if len(zz) == 0:
        return img
    cls = occ[zz, rr, cc]
    # iso axes: u along (col - row), v along (col + row)/2 - z
    su, sv, sz = 3.0, 1.5, 5.0
    u = ((cc.astype(np.int32) - rr) * su * 0.75 + W // 2).astype(np.int32)
    v = ((cc.astype(np.int32) + rr) * sv * 0.75 * 0.5
         - zz * sz + H * 0.28).astype(np.int32)
    order = np.argsort((rr + cc) * GZ + zz)      # far -> near, low -> high
    u, v, zz2, cls = u[order], v[order], zz[order], cls[order]
    shade = (0.55 + 0.45 * zz2 / (GZ - 1))
    col = (OCC_PAL[cls][:, ::-1] * shade[:, None]).astype(np.uint8)
    ok = (u >= 1) & (u < W - 2) & (v >= 1) & (v < H - 2)
    u, v, col = u[ok], v[ok], col[ok]
    for k in range(len(u)):                      # 2x3 px voxel tiles
        img[v[k]:v[k] + 2, u[k]:u[k] + 3] = col[k]
    # ego marker
    eu, ev = W // 2, int(GX * sv * 0.75 * 0.5 + H * 0.28 - sz)
    cv2.drawMarker(img, ((0 + GX // 2 - GX // 2) * 0 + eu,
                         int((GX // 2 + GY // 2) * sv * 0.75 * 0.5 + H * 0.28)),
                   (0, 255, 0), cv2.MARKER_TRIANGLE_UP, 16, 2)
    return img


def top_view(occ, size=760):
    top = np.zeros((GX, GY), np.uint8)
    free = np.zeros((GX, GY), bool)
    for z in range(GZ):
        lay = occ[z]
        free |= lay == 0
        m = (lay > 0) & (lay != 255)
        top[m] = lay[m]
    img = OCC_PAL[top][:, :, ::-1].copy()
    img[(top == 0) & free] = (40, 40, 40)        # observed-free = dark grey
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)
    cv2.drawMarker(img, (size // 2, size // 2), (0, 255, 0),
                   cv2.MARKER_TRIANGLE_UP, 18, 2)
    for m50 in (10, 20, 30):
        px = int(size / 2 * (1 - m50 / 40.0))
        cv2.line(img, (0, px), (size, px), (70, 70, 70), 1)
        cv2.putText(img, f"{m50}m", (4, px - 3), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (160, 160, 160), 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--out", default="out/demo_occ_gt.mp4")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--stride", type=int, default=2)
    args = ap.parse_args()
    VW, VH = 1920, 1080
    raw = args.out.replace(".mp4", "_raw.mp4")
    vw = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
                         (VW, VH))
    n = 0
    for scene in args.scenes:
        root = f"out/bevlane/{scene}"
        man = json.load(open(f"{root}/manifest.json"))
        for f in man["frames"][::args.stride]:
            if "occ" not in f:
                continue
            try:
                occ = np.load(f"{root}/" + f["occ"])["occ"]
            except Exception:
                continue
            frame = np.zeros((VH, VW, 3), np.uint8)
            p = f["imgs"].get("CAM_FRONT_WIDE")
            if p is not None:
                img = cv2.imread(f"{root}/" + p)
                if img is not None:
                    rw = 640
                    rh = int(rw * 432 / 768)
                    frame[70:70 + rh, 20:20 + rw] = cv2.resize(img, (rw, rh))
                    cv2.putText(frame, "CAM_FRONT_WIDE", (20, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                (220, 220, 220), 1, cv2.LINE_AA)
            # legend
            ly = 500
            for ci, nm in enumerate(OCC_NAMES):
                col = tuple(int(v) for v in OCC_PAL[ci][::-1]) \
                    if ci else (40, 40, 40)
                cv2.rectangle(frame, (24, ly), (44, ly + 16), col, -1)
                cv2.putText(frame, nm if ci else "free (observed)",
                            (52, ly + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (200, 200, 200), 1, cv2.LINE_AA)
                ly += 24
            cv2.putText(frame, "unknown = black (never observed)",
                        (24, ly + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (120, 120, 120), 1, cv2.LINE_AA)
            tv = top_view(occ)
            frame[70:70 + 760, 700:700 + 760] = tv
            cv2.putText(frame, "OCC GT top-down (+-40m, all z)", (700, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1,
                        cv2.LINE_AA)
            iso = iso_render(occ, W=430, H=760)
            frame[70:70 + 760, 1478:1478 + 430] = iso
            cv2.putText(frame, "isometric 3D voxels", (1478, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1,
                        cv2.LINE_AA)
            cv2.putText(frame, f"{scene.split('+0900_')[-1]}  f{f['frame']:03d}"
                        f"  |  OCC GROUND TRUTH  [16z x 200 x 200] @{VOX}m",
                        (20, VH - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                        (0, 255, 0), 1, cv2.LINE_AA)
            vw.write(frame)
            n += 1
        print(f"{scene}: total {n}", flush=True)
    vw.release()
    subprocess.run(["ffmpeg", "-y", "-i", raw, "-c:v", "libx264", "-crf", "24",
                    "-pix_fmt", "yuv420p", args.out], check=True,
                   capture_output=True)
    os.remove(raw)
    print("done", n, args.out, flush=True)


if __name__ == "__main__":
    main()

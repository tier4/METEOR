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


def iso_render(occ, W=900, H=760, rng_m=24.0, drop=(8,), zmax_m=3.0):
    """Painter's-algorithm isometric voxel rendering (ego at centre).

    rng_m: half-extent shown (crop around ego). drop: classes hidden
    (default 8 = building, which walls off the near field). zmax_m:
    voxels above this height are hidden (tree canopy over the ego)."""
    img = np.zeros((H, W, 3), np.uint8)
    occ = np.asarray(occ)
    n = min(int(rng_m / 0.4), occ.shape[1] // 2)
    r0 = occ.shape[1] // 2 - n
    zmax = min(int((zmax_m + 1.0) / 0.4), occ.shape[0])   # z0 = -1.0 m
    occ = occ[:zmax, r0:r0 + 2 * n, r0:r0 + 2 * n]
    keep = (occ > 0) & (occ != 255) & ~np.isin(occ, drop)
    zz, rr, cc = np.nonzero(keep)
    if len(zz) == 0:
        return img
    cls = occ[zz, rr, cc]
    # iso axes: u along (col - row), v along (col + row)/2 - z
    su = W / (3.0 * n)                            # width-filling
    sv, sz = su * 1.4, su * 0.8
    u = ((cc.astype(np.int32) - rr) * su * 0.75 + W // 2).astype(np.int32)
    v = ((cc.astype(np.int32) + rr) * sv * 0.75 * 0.5
         - zz * sz + H * 0.10).astype(np.int32)
    order = np.argsort((rr + cc) * GZ + zz)      # far -> near, low -> high
    u, v, zz2, cls = u[order], v[order], zz[order], cls[order]
    shade = (0.55 + 0.45 * zz2 / (GZ - 1))
    col = (OCC_PAL[cls][:, ::-1] * shade[:, None]).astype(np.uint8)
    # iso lattice: same-parity cells are 2*su*0.75 apart in u -> wide tiles
    tw, th = max(2, int(su * 1.5) + 1), max(2, int(sv * 0.375) + 1)
    ok = (u >= 1) & (u < W - tw) & (v >= 1) & (v < H - th)
    u, v, col = u[ok], v[ok], col[ok]
    for k in range(len(u)):                      # voxel tiles
        img[v[k]:v[k] + th, u[k]:u[k] + tw] = col[k]
    # ego marker
    cv2.drawMarker(img, (W // 2, int(2 * n * sv * 0.75 * 0.5 + H * 0.10)),
                   (0, 255, 0), cv2.MARKER_TRIANGLE_UP, 16, 2)
    return img


def cube_render(occ, W=900, H=760, rng_m=24.0, drop=(8,), zmax_m=3.0):
    """Voxel-CUBE isometric rendering: each occupied voxel is a small box
    (shaded top + two side faces) sitting on a metric ground grid — the
    classic occupancy-grid look. Painter's algorithm far->near, low->high.
    Same crop/hide conventions as iso_render."""
    img = np.zeros((H, W, 3), np.uint8)
    occ = np.asarray(occ)
    n = min(int(rng_m / 0.4), occ.shape[1] // 2)
    r0 = occ.shape[1] // 2 - n
    zmax = min(int((zmax_m + 1.0) / 0.4), occ.shape[0])
    occ = occ[:zmax, r0:r0 + 2 * n, r0:r0 + 2 * n]
    su = W / (3.0 * n)
    a, b = su * 0.75, su * 1.15 * 0.375
    sz = su * 0.9
    v0 = H * 0.14

    def pt(r, c, z):
        return (int((c - r) * a + W // 2), int((c + r) * b - z * sz + v0))

    # ---- metric ground grid (every 4 m = 10 cells) ----
    gcol = (60, 60, 60)
    for g in range(0, 2 * n + 1, 10):
        cv2.line(img, pt(g, 0, 0), pt(g, 2 * n, 0), gcol, 1, cv2.LINE_AA)
        cv2.line(img, pt(0, g, 0), pt(2 * n, g, 0), gcol, 1, cv2.LINE_AA)
    # ---- ground-class voxels as flat tiles (roads etc. stay flat) ----
    FLAT = (5, 6)                       # road / sidewalk render as carpet
    keep = (occ > 0) & (occ != 255) & ~np.isin(occ, drop)
    flat_m = keep & np.isin(occ, FLAT)
    cube_m = keep & ~np.isin(occ, FLAT)
    zz, rr, cc = np.nonzero(flat_m)
    order = np.argsort(rr + cc)
    for k in order:
        z, r, c = int(zz[k]), int(rr[k]), int(cc[k])
        col = (OCC_PAL[occ[z, r, c]][::-1] * 0.55).astype(np.uint8).tolist()
        p = np.array([pt(r, c, 0), pt(r + 1, c, 0),
                      pt(r + 1, c + 1, 0), pt(r, c + 1, 0)], np.int32)
        cv2.fillPoly(img, [p], col)
    # ---- solid voxels as cubes ----
    zz, rr, cc = np.nonzero(cube_m)
    if len(zz):
        order = np.argsort((rr + cc) * (occ.shape[0] + 1) + zz)
        for k in order:
            z, r, c = int(zz[k]), int(rr[k]), int(cc[k])
            base = OCC_PAL[occ[z, r, c]][::-1].astype(np.float32)
            shade = 0.6 + 0.4 * z / max(zmax - 1, 1)
            top = np.clip(base * shade, 0, 255).astype(np.uint8).tolist()
            left = np.clip(base * shade * 0.55, 0, 255).astype(np.uint8).tolist()
            right = np.clip(base * shade * 0.75, 0, 255).astype(np.uint8).tolist()
            t00, t10 = pt(r, c, z + 1), pt(r + 1, c, z + 1)
            t11, t01 = pt(r + 1, c + 1, z + 1), pt(r, c + 1, z + 1)
            b10, b11, b01 = pt(r + 1, c, z), pt(r + 1, c + 1, z), pt(r, c + 1, z)
            cv2.fillPoly(img, [np.array([t10, t11, b11, b10], np.int32)], left)
            cv2.fillPoly(img, [np.array([t01, t11, b11, b01], np.int32)], right)
            tp = np.array([t00, t10, t11, t01], np.int32)
            cv2.fillPoly(img, [tp], top)
            cv2.polylines(img, [tp], True,
                          tuple(int(v * 0.45) for v in top), 1)
    cv2.drawMarker(img, pt(n, n, 0), (0, 255, 0),
                   cv2.MARKER_TRIANGLE_UP, 16, 2)
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

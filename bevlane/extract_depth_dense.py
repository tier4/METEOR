#!/usr/bin/env python3
"""Dense per-camera depth GT (stride 4) using LiDAR + full panoptic masks.

- LiDAR sweep projected per camera, min-pooled onto a stride-4 grid (128x72).
- Panoptic segments (ALL categories) guide densification:
    sky         -> 49.5 m (far/last bin)
    ego_vehicle -> 2.0 m
    others      -> nearest valid LiDAR cell within the same segment
                   (EDT nearest + same-segment check, then segment median)
- Saved per frame as depth_gt4/<fi>.npz (float16 [6,72,128], 0 = invalid);
  manifest frames gain a "depth4" key.
"""
import argparse
import base64
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import cv2
import numpy as np
from pycocotools import mask as cocomask
from scipy import ndimage
from scipy.interpolate import griddata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from autolabel_bev import Transform  # noqa: E402
from bevlane.extract_gt import CAMS, ROOT, load_scene_light  # noqa: E402

cv2.setNumThreads(1)
OUT = os.environ.get("METEOR_BEVLANE_OUT", "out/bevlane")
DH, DW = 108, 192          # stride-4 grid for 768x432 input
# Finer depth targets: --grid H,W changes the grid (e.g. 180,320 = stride 4 at 720p, 360,640 = stride 2) and
# --out-dir/--key save under another name (defaults stay depth_gt4 / depth4). K is scaled from the manifest
# image size (img_hw, default 432x768) to the grid.
OUT_DIR, OUT_KEY = "depth_gt4", "depth4"
DENSIFY_MAX_SRC = int(os.environ.get("DEPTH_DENSIFY_MAX_SRC", "8000"))   # cap on LiDAR samples for road interpolation (108x192 stays below ~5k, i.e. unchanged)
MAX_DEPTH = 79.0
SKY_D, EGO_D = 79.5, 2.0


def load_all_anns(scene_dir):
    """sample_data_token -> list[(category_name, rle_mask_dict)] (surfaces first)."""
    ann = os.path.join(scene_dir, "annotation")
    cats = {c["token"]: c["name"] for c in
            json.load(open(os.path.join(ann, "category.json")))}
    by_sd = defaultdict(list)
    for name in ("surface_ann", "object_ann"):
        for a in json.load(open(os.path.join(ann, name + ".json"))):
            if a.get("mask"):
                by_sd[a["sample_data_token"]].append(
                    (cats.get(a["category_token"], ""), a["mask"]))
    return by_sd


# ground-plane classes: depth filled geometrically (ray ∩ ego z=0 plane) where
# LiDAR is absent. Fixes unfilled lane markings (separate panoptic segments
# with few LiDAR hits) and the far-biased segment-median fill on elongated
# lane-line segments.
GROUND = {"road", "parking_lot", "laneline_solid_white", "dashed_lane_marking",
          "stopline", "marking_arrow", "marking_character", "marking_other",
          "deceleration_line", "striped_road_marking", "sidewalk", "crosswalk",
          "freespace"}
# road + road-painting classes are MERGED into ONE segment for interpolation:
# markings lie on the road surface, so they borrow the surrounding road's
# LiDAR measurements directly (accurate on slopes, no per-segment bias).
ROADPAINT = {"road", "laneline_solid_white", "dashed_lane_marking", "stopline",
             "marking_arrow", "marking_character", "marking_other",
             "deceleration_line", "striped_road_marking", "crosswalk"}
ROAD_MERGED_ID = 10 ** 6


def ground_plane_depth(K4, T_ego_cam):
    """Camera-z depth of each stride-4 pixel's ray ∩ ground plane (ego z=0).

    [DH,DW] float32; 0 where the ray does not hit the ground (horizon/up).
    Capped at 79 m (far road saturates near the last depth bin).
    """
    vs, us = np.meshgrid(np.arange(DH), np.arange(DW), indexing="ij")
    pix = np.stack([us + 0.5, vs + 0.5, np.ones((DH, DW), np.float64)],
                   0).reshape(3, -1)
    r = np.linalg.inv(K4) @ pix                    # cam-frame rays, z=1
    R, t = T_ego_cam[:3, :3], T_ego_cam[:3, 3]
    dz = (R @ r)[2]
    down = dz < -0.02   # strictly downward rays only (no grazing far band)
    # fisheye guard: pinhole rays are wrong at the periphery -> only fill
    # within ~50 deg of the optical axis (radial tan <= 1.2)
    radial = np.sqrt(r[0] ** 2 + r[1] ** 2)
    down &= radial <= 1.2
    s = np.where(down, -t[2] / np.where(down, dz, -1.0), 0.0)
    g = np.where(s > 0.5, np.minimum(s, 79.0), 0.0)
    return g.reshape(DH, DW).astype(np.float32)


def _rle_sample(rle, hw_full, idx_cm):
    """COCO RLE (column-major, compressed string) -> bool at the sampled full-res pixels idx_cm (column-major linear index).
    Avoids decoding at full (4K) resolution: only the grid sample points are tested, by binary search over the
    run boundaries (about 100x faster per camera)."""
    cnt = np.asarray(_rle_counts(rle), dtype=np.int64)
    cum = np.cumsum(cnt)                       # run i covers [cum[i-1], cum[i]); even i = background, odd i = foreground
    k = np.searchsorted(cum, idx_cm, side="right")
    return (k % 2) == 1


def _rle_counts(rle):
    """Decode the compressed LEB128-style COCO counts string to a list of run lengths (same scheme as pycocotools rleFrString)."""
    s = rle["counts"]
    if isinstance(s, (list, tuple)):
        return list(s)
    if isinstance(s, str):
        s = s.encode()
    out, i, n = [], 0, len(s)
    while i < n:
        x, k, more = 0, 0, True
        while more:
            c = s[i] - 48; i += 1
            x |= (c & 0x1F) << (5 * k)
            more = bool(c & 0x20); k += 1
            if not more and (c & 0x10):
                x |= -1 << (5 * k)
        if len(out) > 2:
            x += out[-2]
        out.append(x)
    return out


def seg_map_small(anns, hw_full):
    """Segment-id map at (DH, DW); returns (seg int32, sky, ego, ground).

    Segments of road + road-painting categories are merged into one id so the
    interpolation treats them as a single surface. Masks are not expanded to full resolution: only the
    (DH, DW) nearest-neighbour sample points (the same points as cv2.resize INTER_NEAREST) are read from the RLE.
    """
    H, W = hw_full
    vs = np.minimum((np.arange(DH) * (H / DH)).astype(np.int64), H - 1)     # INTER_NEAREST: floor(y * H/DH)
    us = np.minimum((np.arange(DW) * (W / DW)).astype(np.int64), W - 1)
    V, U = np.meshgrid(vs, us, indexing="ij")
    idx_cm = (U * H + V).reshape(-1)                                          # column-major linear index
    small = np.zeros((DH, DW), np.int32)
    sky_ids, ego_ids, gnd_ids, rp_ids = set(), set(), set(), set()
    for i, (cname, m) in enumerate(anns, start=1):
        rle = {"size": m["size"], "counts": base64.b64decode(m["counts"])}
        inside = _rle_sample(rle, (H, W), idx_cm).reshape(DH, DW)
        small[inside] = i
        if cname == "sky":
            sky_ids.add(i)
        elif cname == "ego_vehicle":
            ego_ids.add(i)
        if cname in GROUND:
            gnd_ids.add(i)
        if cname in ROADPAINT:
            rp_ids.add(i)
    if rp_ids:
        small[np.isin(small, list(rp_ids))] = ROAD_MERGED_ID
    sky = np.isin(small, list(sky_ids)) if sky_ids else np.zeros_like(small, bool)
    ego = np.isin(small, list(ego_ids)) if ego_ids else np.zeros_like(small, bool)
    gnd = np.isin(small, list(gnd_ids)) if gnd_ids else np.zeros_like(small, bool)
    if rp_ids:
        gnd |= small == ROAD_MERGED_ID
    return small, sky, ego, gnd


def densify(sparse, seg, sky, ego, gnd=None, gpl=None):
    d = sparse.copy()
    invalid = d <= 0
    if invalid.any() and (d > 0).any():
        yy, xx = np.mgrid[0:d.shape[0], 0:d.shape[1]]
        # 1) smooth 2-D linear interpolation of LiDAR samples over the merged
        #    road(+paint) surface -> seamless around markings (no patchiness)
        if gnd is not None:
            src_m = (d > 0) & gnd
            tgt_m = invalid & gnd
            if src_m.sum() >= 16 and tgt_m.any():
                sy, sx, sv = yy[src_m], xx[src_m], d[src_m]
                if len(sv) > DENSIFY_MAX_SRC:          # speed: fine grids give tens of thousands of road samples -> subsample evenly to keep Delaunay cheap
                    k = int(np.ceil(len(sv) / DENSIFY_MAX_SRC)); sy, sx, sv = sy[::k], sx[::k], sv[::k]
                vals = griddata(
                    np.stack([sy, sx], 1), sv,
                    np.stack([yy[tgt_m], xx[tgt_m]], 1), method="linear")
                out = d[tgt_m]
                ok = np.isfinite(vals)
                out[ok] = vals[ok]
                d[tgt_m] = out
                invalid = d <= 0
        # 2) geometric ground-plane fill for what linear interp couldn't reach
        #    (outside the LiDAR hull, e.g. far road) — strictly downward rays
        if gnd is not None and gpl is not None and invalid.any():
            fill = invalid & gnd & (gpl > 0)
            d[fill] = gpl[fill]
            invalid = d <= 0
        # 3) same-segment nearest fill for the rest (object interiors)
        _, (iy, ix) = ndimage.distance_transform_edt(d <= 0, return_indices=True)
        near = d[iy, ix]
        same = (seg[iy, ix] == seg) & (seg > 0)
        fill = invalid & same & (near > 0)
        d[fill] = near[fill]
        still = d <= 0
        if still.any():
            # 4) segment median for whatever is left
            for sid in np.unique(seg[still]):
                if sid == 0:
                    continue
                cells = seg == sid
                vals = d[cells & (d > 0)]
                if len(vals) >= 3:
                    d[cells & (d <= 0)] = np.median(vals)
    elif gnd is not None and gpl is not None and invalid.any():
        fill = invalid & gnd & (gpl > 0)
        d[fill] = gpl[fill]
    # hard overrides LAST: sky = far bin, ego body = fixed near distance
    # (prevents far depths from leaking into the ego-vehicle region)
    d[sky] = SKY_D
    d[ego] = EGO_D
    return d


def process_scene(args):
    scene, stride = args
    try:
        out_dir = os.path.join(OUT, scene)
        man = json.load(open(os.path.join(out_dir, "manifest.json")))
        sdir = os.path.join(ROOT, scene)
        ordered, frames, calib, egop = load_scene_light(sdir)
        strided = ordered[::stride]
        anns_by_sd = load_all_anns(sdir)

        cams = []
        ih, iw = man.get("img_hw", (432, 768))
        S = np.diag([DW / float(iw), DH / float(ih), 1.0])      # image-K -> grid-K (x and y scaled separately)
        for ch in CAMS:
            K = np.array(man["cams"][ch]["K"])
            T_ego_cam = np.array(man["cams"][ch]["T_ego_cam"])
            T_cam_ego = np.linalg.inv(T_ego_cam)
            Kg = S @ K
            gpl = ground_plane_depth(Kg, T_ego_cam)
            cams.append((Kg, T_cam_ego, gpl))

        os.makedirs(os.path.join(out_dir, OUT_DIR), exist_ok=True)
        for fr in man["frames"]:
            fi = fr["frame"]
            path = os.path.join(out_dir, f"{OUT_DIR}/{fi:04d}.npz")
            if os.path.exists(path):
                fr[OUT_KEY] = f"{OUT_DIR}/{fi:04d}.npz"
                continue
            s = strided[fi]
            frame = frames[s["token"]]
            ld = frame["LIDAR_CONCAT"]
            pts = np.fromfile(os.path.join(sdir, ld["filename"]),
                              dtype=np.float32).reshape(-1, 5)[:, :3].astype(np.float64)
            cal_l = calib[ld["calibrated_sensor_token"]]
            pts_ego = Transform(cal_l["rotation"], cal_l["translation"]).apply(pts)

            depth = np.zeros((len(CAMS), DH, DW), np.float32)
            for ci, ch in enumerate(CAMS):
                K4, T_cam_ego, gpl = cams[ci]
                pc = pts_ego @ T_cam_ego[:3, :3].T + T_cam_ego[:3, 3]
                z = pc[:, 2]
                m = (z > 0.5) & (z < MAX_DEPTH)
                u = (K4[0, 0] * pc[m, 0] / z[m] + K4[0, 2]).astype(np.int32)
                v = (K4[1, 1] * pc[m, 1] / z[m] + K4[1, 2]).astype(np.int32)
                zm = z[m].astype(np.float32)
                ok = (u >= 0) & (u < DW) & (v >= 0) & (v < DH)
                # speed: instead of np.minimum.at (~1 s), sort and take the per-cell minimum z (identical result)
                idx = v[ok] * DW + u[ok]; zz = zm[ok]
                o = np.lexsort((zz, idx)); idx_s, z_s = idx[o], zz[o]
                first = np.ones(len(idx_s), bool); first[1:] = idx_s[1:] != idx_s[:-1]
                d = np.zeros(DH * DW, np.float32); d[idx_s[first]] = z_s[first]
                d = d.reshape(DH, DW)
                sd = frame.get(ch)
                anns = anns_by_sd.get(sd["token"], []) if sd else []
                if anns:
                    hw = tuple(anns[0][1]["size"])
                    seg, sky, ego, gnd = seg_map_small(anns, hw)
                    d = densify(d, seg, sky, ego, gnd, gpl)
                depth[ci] = d
            np.savez_compressed(path, depth=depth.astype(np.float16))
            fr[OUT_KEY] = f"{OUT_DIR}/{fi:04d}.npz"
        json.dump(man, open(os.path.join(out_dir, "manifest.json"), "w"))
        return f"[ok] {scene}"
    except Exception as e:
        return f"[fail] {scene}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--scenes", default=None)
    ap.add_argument("--grid", default="", help="grid H,W (e.g. 180,320 / 360,640); default 108,192")
    ap.add_argument("--out-dir", default="", help="output directory name (default depth_gt4; e.g. depth_gt_hr)")
    ap.add_argument("--key", default="", help="manifest key (default depth4; e.g. depth_hr)")
    args = ap.parse_args()
    global DH, DW, OUT_DIR, OUT_KEY
    if args.grid:
        DH, DW = [int(v) for v in args.grid.split(",")]
    if args.out_dir: OUT_DIR = args.out_dir
    if args.key: OUT_KEY = args.key
    print(f"grid {DH}x{DW} -> {OUT_DIR}/ (manifest key {OUT_KEY})", flush=True)
    if args.scenes:
        scenes = (open(args.scenes).read().split() if os.path.isfile(args.scenes)
                  else args.scenes.split(","))
    else:
        scenes = sorted(d for d in os.listdir(OUT)
                        if os.path.exists(os.path.join(OUT, d, "manifest.json")))
    print(f"{len(scenes)} scenes", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(process_scene,
                                     [(s, args.stride) for s in scenes])):
            if i % 25 == 0 or r.startswith("[fail"):
                print(f"{i + 1}/{len(scenes)} {r}", flush=True)


if __name__ == "__main__":
    main()

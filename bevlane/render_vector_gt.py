#!/usr/bin/env python3
"""Render hybrid ego-centric GT: area classes from the raster, line classes
re-drawn as fixed-width connected polylines from the vector map (v2).

Class ids are unchanged (autolabel 9-class scheme):
  0 unlabeled(ignore) 1 road 2 sidewalk 3 crosswalk 4 laneline 5 stopline
  6 road_edge 7 marking 8 parking
- raster thin-class pixels (4,5,6) are remapped to road, then lines are drawn
  from the smoothed vector polylines (connected across dash gaps).
- sidewalk / crosswalk polygons are re-filled from the vector map (smooth).
Reuses image cache & manifests from extract_gt.py; adds gt_vec/*.png.
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from autolabel_bev import quat_to_rot  # noqa: E402
from bevlane.extract_gt import ROOT, load_scene_light  # noqa: E402

cv2.setNumThreads(1)

PROD = "out/production"
OUT = "out/bevlane"
# fills from vector polygons: (vector class, target id)
FILLS = [("sidewalk", 2), ("crosswalk", 3)]
# lines from vector polylines: (vector class, target id, width [m])
# NOTE: road_edge is NOT taken from vectors; it is computed per crop as the
# pixel-accurate boundary between the drivable region and observed non-drivable.
LINES = [("stopline", 5, 0.45), ("laneline", 4, 0.30)]


def process_scene(args):
    scene, stride = args
    try:
        out_dir = os.path.join(OUT, scene)
        mf = os.path.join(out_dir, "manifest.json")
        man = json.load(open(mf))
        res = man["bev_res"]
        bh = man.get("bev_h", man["bev_size"])
        bw = man.get("bev_w", man["bev_size"])
        hx = man.get("bev_xh", bh * res / 2.0)
        hy = man.get("bev_yh", bw * res / 2.0)
        vec = json.load(open(os.path.join(PROD, scene, "vector_map.json")))
        meta = json.load(open(os.path.join(PROD, scene, "meta.json")))
        raster = np.load(os.path.join(PROD, scene, "bev_label_masked.npy"))
        # thin raster classes -> road (they are replaced by vector lines)
        raster = raster.copy()
        raster[np.isin(raster, (4, 5, 6))] = 1
        res0 = meta["resolution"]
        x0, y0 = meta["origin"]
        ordered, frames, _, egop = load_scene_light(os.path.join(ROOT, scene))
        strided = ordered[::stride]

        polys = {c: [np.asarray(p) for p in vec["classes"].get(c, [])]
                 for c in [f[0] for f in FILLS] + [l[0] for l in LINES]}
        os.makedirs(os.path.join(out_dir, "gt_vec"), exist_ok=True)

        for fr in man["frames"]:
            fi = fr["frame"]
            s = strided[fi]
            ld = frames[s["token"]]["LIDAR_CONCAT"]
            ep = egop[ld["ego_pose_token"]]
            tx, ty = ep["translation"][:2]
            R = quat_to_rot(ep["rotation"])
            yaw = np.arctan2(R[1, 0], R[0, 0])
            c, sn = np.cos(yaw), np.sin(yaw)

            A = np.array([[sn * res, -c * res], [-c * res, -sn * res]])
            b = np.array([tx + c * hx - sn * hy, ty + sn * hx + c * hy])
            M = np.zeros((2, 3))
            M[:, :2] = A / res0
            M[:, 2] = (b - [x0, y0]) / res0
            gt = cv2.warpAffine(raster, M, (bw, bh),
                                flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                                borderValue=0)

            def to_px(pw):
                dx, dy = pw[:, 0] - tx, pw[:, 1] - ty
                xe = c * dx + sn * dy
                ye = -sn * dx + c * dy
                return np.stack([(hy - ye) / res, (hx - xe) / res], 1)

            def in_roi(px):
                return not (px[:, 0].max() < -50 or px[:, 0].min() > bw + 50
                            or px[:, 1].max() < -50 or px[:, 1].min() > bh + 50)

            gt[gt == 6] = 1     # drop old raster road_edge -> road (recomputed)
            gt[gt == 7] = 1     # marking(incl deceleration) is noise -> road
            for cname, cid in FILLS:
                for pw in polys[cname]:
                    px = to_px(pw)
                    if in_roi(px):
                        cv2.fillPoly(gt, [np.round(px).astype(np.int32)
                                          .reshape(-1, 1, 2)], cid)
            for cname, cid, width in LINES:
                t = max(1, int(round(width / res)))
                for pw in polys[cname]:
                    px = to_px(pw)
                    if in_roi(px):
                        cv2.polylines(gt, [np.round(px).astype(np.int32)
                                           .reshape(-1, 1, 2)], False, cid, t)

            # keep only the ego-connected drivable region; drop physically
            # disconnected carriageways (e.g. opposing highway lanes across a
            # median). Erode first to break thin median bridges, seed from a
            # narrow ego-lane strip only, then dilate back within drivable.
            DRV = (1, 3, 4, 5)
            drv = np.isin(gt, DRV).astype(np.uint8)
            drv_e = cv2.erode(drv, np.ones((7, 7), np.uint8))
            ncc, lab = cv2.connectedComponents(drv_e, 8)
            seed = lab[:, bw // 2 - 6:bw // 2 + 6]        # ego's own lane column
            keep_ids = set(np.unique(seed[seed > 0]).tolist())
            if keep_ids:
                gt_before = gt.copy()
                before = int(drv.sum())
                km = cv2.dilate(np.isin(lab, list(keep_ids)).astype(np.uint8),
                                np.ones((9, 9), np.uint8))
                gt[(drv > 0) & (km == 0)] = 0     # remove opposing carriageway
                # safety: at wide intersections erosion fragments the road and
                # the seed may catch a sliver -> filter wipes the map. Revert.
                after = int(np.isin(gt, DRV).sum())
                if after < 0.3 * before:
                    gt = gt_before

            # fill small enclosed unlabeled holes INSIDE the drivable region
            # (LiDAR shadows / sparse cells): they are road, and leaving them
            # unlabeled teaches "background" speckle inside roads (--train-bg).
            drv2 = np.isin(gt, DRV).astype(np.uint8)
            inv = (drv2 == 0).astype(np.uint8)
            nh, hlab, hstats, _ = cv2.connectedComponentsWithStats(inv, 8)
            for hi in range(1, nh):
                x, y, w2, h2, area = hstats[hi]
                if area <= 400 and x > 0 and y > 0 \
                        and x + w2 < bw and y + h2 < bh:      # enclosed & small
                    hole = (hlab == hi) & (gt == 0)
                    gt[hole] = 1

            # road_edge := outer boundary of ego-connected road wherever a
            # SUBSTANTIAL non-road region (black/other) continues beyond it.
            # Captures curved guardrails, curbs and road ends in every
            # direction; excludes thin occlusion gaps and the +-80 m frontier.
            kd = np.isin(gt, DRV).astype(np.uint8)
            kd = cv2.morphologyEx(kd, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            cnts = cv2.findContours(kd, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)[-2]
            kdf = np.zeros_like(kd)
            cv2.drawContours(kdf, cnts, -1, 1, -1)
            outer = kdf & ~cv2.erode(kdf, np.ones((3, 3), np.uint8))
            nonroad = (kdf == 0).astype(np.uint8)
            sub = cv2.erode(nonroad, np.ones((7, 7), np.uint8))   # region continues
            sub = cv2.dilate(sub, np.ones((9, 9), np.uint8))
            edge = (outer & sub).astype(np.uint8)
            edge[:10, :] = 0                 # drop front/back ROI frontier
            edge[-10:, :] = 0
            edge = cv2.dilate(edge, np.ones((2, 2), np.uint8))
            gt[edge > 0] = 6
            cv2.imwrite(os.path.join(out_dir, f"gt_vec/{fi:04d}.png"), gt)
            fr["gt_vec"] = f"gt_vec/{fi:04d}.png"
        json.dump(man, open(mf, "w"))
        return f"[ok] {scene} {len(man['frames'])}"
    except Exception as e:
        return f"[fail] {scene}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, required=True,
                    help="stride used at extract time (frame index basis)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--scenes", default=None)
    args = ap.parse_args()
    if args.scenes:
        scenes = (open(args.scenes).read().split() if os.path.isfile(args.scenes)
                  else args.scenes.split(","))
    else:
        scenes = sorted(d for d in os.listdir(OUT)
                        if os.path.exists(os.path.join(OUT, d, "manifest.json")))
    print(f"{len(scenes)} scenes", flush=True)
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(process_scene,
                                     [(s, args.stride) for s in scenes])):
            done += 1
            if i % 50 == 0 or r.startswith("[fail"):
                print(f"{i + 1}/{len(scenes)} {r}", flush=True)
    print(f"DONE total={done}", flush=True)


if __name__ == "__main__":
    main()

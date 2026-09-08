#!/usr/bin/env python3
"""Create per-frame supervision-validity masks from LiDAR observation support.

The BEV label raster stores both true background and unobserved cells as zero.
With ``--train-bg`` those two meanings become indistinguishable and weak rear /
scene-boundary areas are learned as hard negatives.  This tool warps the
scene-level ``min_dist <= mask_dist`` support into each ego frame, writes a
compact 0/255 PNG, and registers it as ``gt_valid`` in the manifest.

Cosmos3 Transfer changes appearance but preserves geometry.  Condition aliases
therefore reuse the source scene's poses and observation support while writing
the mask into the transferred scene directory.
"""
import argparse
import json
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from autolabel_bev import quat_to_rot
from bevlane.extract_gt import load_scene_light

cv2.setNumThreads(1)

CONDITIONS = (
    "night_heavy_rain", "night_heavy_snow", "heavy_rain", "heavy_snow",
    "backlit", "night",
)


def source_name(scene):
    for condition in CONDITIONS:
        prefix = f"cosmos3_{condition}_"
        if scene.startswith(prefix):
            return scene[len(prefix):]
    return scene


def process_scene(job):
    scene, args = job
    try:
        src = source_name(scene)
        out_dir = os.path.join(args.out, scene)
        manifest_path = os.path.join(out_dir, "manifest.json")
        man = json.load(open(manifest_path))
        if args.register_existing:
            missing = []
            for fr in man["frames"]:
                rel = f"gt_valid/{int(fr['frame']):04d}.png"
                if not os.path.isfile(os.path.join(out_dir, rel)):
                    missing.append(rel)
                else:
                    fr["gt_valid"] = rel
            if missing:
                return f"[fail] {scene}: {len(missing)} existing masks missing"
            if not args.dry_run:
                man["gt_valid_v"] = 1
                man["gt_valid_mask_dist"] = args.mask_dist
                tmp = manifest_path + f".gtvalid.{os.getpid()}.tmp"
                with open(tmp, "w") as fp:
                    json.dump(man, fp)
                os.replace(tmp, manifest_path)
            return f"[register] {scene} n={len(man['frames'])}"
        # Appearance-only Cosmos variants share frame IDs and all geometry.
        # Reuse already generated source masks byte-for-byte rather than
        # loading/warping the same 170 MB observation archive six more times.
        if src != scene:
            base_dir = os.path.join(args.out, src)
            base_manifest_path = os.path.join(base_dir, "manifest.json")
            if os.path.isfile(base_manifest_path):
                base_man = json.load(open(base_manifest_path))
                by_frame = {int(f["frame"]): f for f in base_man["frames"]}
                reusable = all(int(f["frame"]) in by_frame and
                               by_frame[int(f["frame"])].get("gt_valid")
                               for f in man["frames"])
                if reusable:
                    ratios = []
                    if not args.dry_run:
                        os.makedirs(os.path.join(out_dir, "gt_valid"),
                                    exist_ok=True)
                    for fr in man["frames"]:
                        fi = int(fr["frame"])
                        rel_src = by_frame[fi]["gt_valid"]
                        source = os.path.join(base_dir, rel_src)
                        mask = cv2.imread(source, 0)
                        if mask is None:
                            raise IOError(f"unreadable source mask {source}")
                        ratios.append(float((mask > 0).mean()))
                        rel_dst = f"gt_valid/{fi:04d}.png"
                        if not args.dry_run:
                            target = os.path.join(out_dir, rel_dst)
                            if not os.path.exists(target):
                                try:
                                    os.link(source, target)
                                except OSError:
                                    shutil.copyfile(source, target)
                            fr["gt_valid"] = rel_dst
                    if not args.dry_run:
                        man["gt_valid_v"] = 1
                        man["gt_valid_mask_dist"] = base_man.get(
                            "gt_valid_mask_dist", args.mask_dist)
                        tmp = manifest_path + f".gtvalid.{os.getpid()}.tmp"
                        with open(tmp, "w") as fp:
                            json.dump(man, fp)
                        os.replace(tmp, manifest_path)
                    return (f"[reuse] {scene} <- {src} n={len(ratios)} "
                            f"valid={np.mean(ratios):.3f}")
        prod = os.path.join(args.prod, src)
        meta = json.load(open(os.path.join(prod, "meta.json")))
        min_dist = np.load(os.path.join(prod, "bev_counts.npz"))["min_dist"]
        support = (min_dist <= args.mask_dist).astype(np.uint8) * 255
        if args.dilate:
            k = np.ones((args.dilate, args.dilate), np.uint8)
            support = cv2.dilate(support, k)

        res = float(man["bev_res"])
        bh = int(man.get("bev_h", man["bev_size"]))
        bw = int(man.get("bev_w", man["bev_size"]))
        hx = float(man.get("bev_xh", bh * res / 2.0))
        hy = float(man.get("bev_yh", bw * res / 2.0))
        res0 = float(meta["resolution"])
        x0, y0 = meta["origin"]
        ordered, frames, _, egop = load_scene_light(os.path.join(args.root, src))
        stride = args.stride
        if stride <= 0:
            # The current JP cache has 147 manifest frames from 294 raw
            # samples (stride 2), while older extraction rounds used stride 5.
            # Infer the integer ratio instead of silently warping the wrong
            # pose sequence.
            nfi = max(int(fr["frame"]) for fr in man["frames"]) + 1
            stride = max(1, int(round(len(ordered) / nfi)))
        strided = ordered[::stride]

        if not args.dry_run:
            os.makedirs(os.path.join(out_dir, "gt_valid"), exist_ok=True)
        ratios = []
        for fr in man["frames"]:
            fi = int(fr["frame"])
            if fi >= len(strided):
                raise IndexError(f"frame {fi} outside strided source")
            sample = strided[fi]
            lidar = frames[sample["token"]]["LIDAR_CONCAT"]
            pose = egop[lidar["ego_pose_token"]]
            tx, ty = pose["translation"][:2]
            rot = quat_to_rot(pose["rotation"])
            yaw = np.arctan2(rot[1, 0], rot[0, 0])
            c, sn = np.cos(yaw), np.sin(yaw)
            A = np.array([[sn * res, -c * res],
                          [-c * res, -sn * res]])
            b = np.array([tx + c * hx - sn * hy,
                          ty + sn * hx + c * hy])
            M = np.zeros((2, 3))
            M[:, :2] = A / res0
            M[:, 2] = (b - [x0, y0]) / res0
            valid = cv2.warpAffine(
                support, M, (bw, bh),
                flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            ratios.append(float((valid > 0).mean()))
            rel = f"gt_valid/{fi:04d}.png"
            if not args.dry_run:
                if not cv2.imwrite(os.path.join(out_dir, rel), valid):
                    raise IOError(f"cannot write {rel}")
                fr["gt_valid"] = rel

        if not args.dry_run:
            man["gt_valid_v"] = 1
            man["gt_valid_mask_dist"] = args.mask_dist
            tmp = manifest_path + f".gtvalid.{os.getpid()}.tmp"
            with open(tmp, "w") as fp:
                json.dump(man, fp)
            os.replace(tmp, manifest_path)
        return (f"[ok] {scene} n={len(ratios)} valid="
                f"{np.mean(ratios):.3f} min={np.min(ratios):.3f} "
                f"max={np.max(ratios):.3f}")
    except Exception as exc:
        return f"[fail] {scene}: {type(exc).__name__}: {exc}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True,
                        help="raw converted dataset containing annotations")
    parser.add_argument("--prod", default="out/production")
    parser.add_argument("--out", default="out/bevlane")
    parser.add_argument("--stride", type=int, default=0,
                        help="raw sample stride; 0 infers it per scene")
    parser.add_argument("--mask-dist", type=float, default=20.0)
    parser.add_argument("--dilate", type=int, default=0,
                        help="optional support dilation kernel (pixels)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--scenes", default="",
                        help="scene-list path or comma-separated names")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-missing", action="store_true",
                        help="pre-filter scenes without source support")
    parser.add_argument("--register-existing", action="store_true",
                        help="only add manifest keys for existing mask PNGs")
    args = parser.parse_args()
    if args.scenes and os.path.isfile(args.scenes):
        scenes = open(args.scenes).read().split()
    elif args.scenes:
        scenes = [x for x in args.scenes.split(",") if x]
    else:
        scenes = sorted(d for d in os.listdir(args.out)
                        if os.path.isfile(os.path.join(args.out, d,
                                                       "manifest.json")))
    if args.skip_missing and not args.register_existing:
        before = len(scenes)
        keep = []
        for scene in scenes:
            src = source_name(scene)
            reusable = (scene != src and os.path.isfile(os.path.join(
                args.out, src, "manifest.json")))
            generatable = (os.path.isfile(os.path.join(
                                args.prod, src, "bev_counts.npz")) and
                           os.path.isdir(os.path.join(args.root, src,
                                                      "annotation")))
            if reusable or generatable:
                keep.append(scene)
        scenes = keep
        print(f"source-filter {before} -> {len(scenes)} scenes", flush=True)
    print(f"scenes={len(scenes)} workers={args.workers} dry={args.dry_run}",
          flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, result in enumerate(pool.map(process_scene,
                                             ((s, args) for s in scenes))):
            if result.startswith("[fail]") or i % 25 == 0:
                print(f"{i + 1}/{len(scenes)} {result}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""BEV 3D-box GT from LiDAR annotations, filtered by camera visibility.

For every manifest keyframe:
- take sample_annotation 3D boxes (map frame) of that sample,
- keep ONLY instances that have a 2D object_ann mask on at least one of the
  8 cameras at that frame (i.e. the 2D segmentation "sees" the object; fully
  occluded boxes are dropped, per GT policy),
- transform to ego frame via the LiDAR ego_pose and rasterise the oriented
  footprint onto the ego BEV grid (800x500 @0.2m, row0=+80m, col0=+50m).

Classes: 0 bg, 1 vehicle (car/truck/bus/motorcycle/construction/trailer),
2 VRU (pedestrian/bicycle/stroller/animal).
Saved as bev_box/<fi>.png; manifest frames gain "bev_box".
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from autolabel_bev import quat_to_rot  # noqa: E402
from bevlane.extract_gt import OUT, ROOT, load_scene_light  # noqa: E402

cv2.setNumThreads(1)
BEV_H, BEV_W = 800, 500
BEV_XH, BEV_YH, RES = 80.0, 50.0, 0.2
VEHICLE = {"car", "truck", "bus", "motorcycle", "construction", "trailer",
           "emergency_vehicle", "other_vehicle"}
VRU = {"pedestrian", "bicycle", "stroller", "animal", "wheelchair"}
CAMS = ["CAM_FRONT_WIDE", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
        "CAM_BACK_WIDE", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
        "CAM_FRONT_NARROW", "CAM_BACK_NARROW"]


def load_boxes(scene_dir):
    ann = os.path.join(scene_dir, "annotation")
    cats = {c["token"]: c["name"] for c in
            json.load(open(os.path.join(ann, "category.json")))}
    inst_cat = {i["token"]: cats.get(i["category_token"], "") for i in
                json.load(open(os.path.join(ann, "instance.json")))}
    boxes_by_sample = defaultdict(list)
    for a in json.load(open(os.path.join(ann, "sample_annotation.json"))):
        cname = inst_cat.get(a["instance_token"], "")
        cls = 1 if cname in VEHICLE else 2 if cname in VRU else 0
        if cls:
            boxes_by_sample[a["sample_token"]].append(
                (a["instance_token"], cls, a["translation"], a["size"],
                 a["rotation"]))
    # 2D annotation boxes per camera sample_data (class-grouped, image px)
    ann2d_by_sd = defaultdict(list)
    for a in json.load(open(os.path.join(ann, "object_ann.json"))):
        cname = cats.get(a["category_token"], "")
        cls = 1 if cname in VEHICLE else 2 if cname in VRU else 0
        if cls and a.get("bbox"):
            ann2d_by_sd[a["sample_data_token"]].append((cls, a["bbox"]))
    # image sizes per sample_data (to scale 2D boxes into the cached K space)
    sd_size = {}
    for sd in json.load(open(os.path.join(ann, "sample_data.json"))):
        if sd.get("width"):
            sd_size[sd["token"]] = (sd["width"], sd["height"])
    return boxes_by_sample, ann2d_by_sd, sd_size


def box_corners_ego(tr, size, rot, te, Re):
    """8 corners of a 3D box (map frame) -> ego frame [8,3]."""
    w, l, h = size
    Rb = quat_to_rot(rot)
    cs = []
    for sx in (l / 2, -l / 2):
        for sy in (w / 2, -w / 2):
            for sz in (h / 2, -h / 2):
                cs.append([sx, sy, sz])
    pm = np.asarray(cs) @ Rb.T + np.asarray(tr)
    return (pm - np.asarray(te)) @ Re          # Re passed as Re^T-ready


def visible_in_any_cam(corners_ego, cls, frame, cams, ann2d_by_sd, sd_size,
                       thresh=0.3):
    """Project the box into each camera; visible iff a same-class 2D ann box
    overlaps the projection (2D segmentation sees the object)."""
    for ch, (K, T_cam_ego) in cams.items():
        sd = frame.get(ch)
        if sd is None:
            continue
        anns = ann2d_by_sd.get(sd["token"], [])
        if not anns:
            continue
        pc = corners_ego @ T_cam_ego[:3, :3].T + T_cam_ego[:3, 3]
        z = pc[:, 2]
        if (z > 0.3).sum() < 2:
            continue
        m = z > 0.3
        u = K[0, 0] * pc[m, 0] / z[m] + K[0, 2]
        v = K[1, 1] * pc[m, 1] / z[m] + K[1, 2]
        x1, x2 = max(u.min(), 0), min(u.max(), 768)
        y1, y2 = max(v.min(), 0), min(v.max(), 432)
        if x2 - x1 < 3 or y2 - y1 < 3:
            continue
        pa = (x2 - x1) * (y2 - y1)
        ow, oh = sd_size.get(sd["token"], (2880, 1860))
        sx, sy = 768.0 / ow, 432.0 / oh
        for acls, bb in anns:
            if acls != cls:
                continue
            ax1, ay1, ax2, ay2 = bb[0] * sx, bb[1] * sy, bb[2] * sx, bb[3] * sy
            ix = max(0, min(x2, ax2) - max(x1, ax1))
            iy = max(0, min(y2, ay2) - max(y1, ay1))
            inter = ix * iy
            aa = max((ax2 - ax1) * (ay2 - ay1), 1e-3)
            if inter / min(pa, aa) > thresh:
                return True
    return False


def process_scene(args):
    scene, stride = args
    try:
        out_dir = os.path.join(OUT, scene)
        man = json.load(open(os.path.join(out_dir, "manifest.json")))
        sdir = os.path.join(ROOT, scene)
        ordered, frames, _, egop = load_scene_light(sdir)
        strided = ordered[::stride]
        boxes_by_sample, ann2d_by_sd, sd_size = load_boxes(sdir)
        cams = {}
        for ch in CAMS:
            if ch in man["cams"]:
                cams[ch] = (np.array(man["cams"][ch]["K"]),
                            np.linalg.inv(np.array(man["cams"][ch]["T_ego_cam"])))
        os.makedirs(os.path.join(out_dir, "bev_box"), exist_ok=True)
        kept = drop = 0
        for fr in man["frames"]:
            fi = fr["frame"]
            path = os.path.join(out_dir, f"bev_box/{fi:04d}.png")
            ppath = os.path.join(out_dir, f"bev_box/{fi:04d}.npz")
            if os.path.exists(path) and os.path.exists(ppath):
                fr["bev_box"] = f"bev_box/{fi:04d}.png"
                fr["bev_box_p"] = f"bev_box/{fi:04d}.npz"
                continue
            s = strided[fi]
            frame = frames[s["token"]]
            # ego pose (LiDAR)
            ep = egop[frame["LIDAR_CONCAT"]["ego_pose_token"]]
            te = np.array(ep["translation"])
            Re = quat_to_rot(ep["rotation"])
            yaw_e = np.arctan2(Re[1, 0], Re[0, 0])
            bev = np.zeros((BEV_H, BEV_W), np.uint8)
            params = []
            for inst, cls, tr, size, rot in boxes_by_sample.get(s["token"], []):
                corners = box_corners_ego(tr, size, rot, te, Re)
                if not visible_in_any_cam(corners, cls, frame, cams,
                                          ann2d_by_sd, sd_size):
                    drop += 1                  # not seen by 2D segmentation
                    continue
                # map -> ego (2D)
                dx, dy = tr[0] - te[0], tr[1] - te[1]
                c, sn = np.cos(-yaw_e), np.sin(-yaw_e)
                xe = c * dx - sn * dy
                ye = sn * dx + c * dy
                if not (-BEV_XH - 5 < xe < BEV_XH + 5
                        and -BEV_YH - 5 < ye < BEV_YH + 5):
                    continue
                Rb = quat_to_rot(rot)
                yaw_b = np.arctan2(Rb[1, 0], Rb[0, 0]) - yaw_e
                w, l = size[0], size[1]        # nuScenes: [w, l, h]
                # footprint corners in ego frame
                cb, sb = np.cos(yaw_b), np.sin(yaw_b)
                cors = []
                for lx, wy in ((l / 2, w / 2), (l / 2, -w / 2),
                               (-l / 2, -w / 2), (-l / 2, w / 2)):
                    px = xe + lx * cb - wy * sb
                    py = ye + lx * sb + wy * cb
                    row = (BEV_XH - px) / RES
                    col = (BEV_YH - py) / RES
                    cors.append([col, row])
                cv2.fillPoly(bev, [np.round(np.array(cors)).astype(np.int32)
                                   .reshape(-1, 1, 2)], int(cls))
                params.append([cls, xe, ye, l, w, yaw_b])
                kept += 1
            cv2.imwrite(path, bev)
            np.savez_compressed(ppath, boxes=np.array(params, np.float32)
                                if params else np.zeros((0, 6), np.float32))
            fr["bev_box"] = f"bev_box/{fi:04d}.png"
            fr["bev_box_p"] = f"bev_box/{fi:04d}.npz"
        json.dump(man, open(os.path.join(out_dir, "manifest.json"), "w"))
        return f"[ok] {scene} kept={kept} occluded-dropped={drop}"
    except Exception as e:
        return f"[fail] {scene}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=2)
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
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(process_scene,
                                     [(s, args.stride) for s in scenes])):
            if i % 40 == 0 or r.startswith("[fail"):
                print(f"{i + 1}/{len(scenes)} {r}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

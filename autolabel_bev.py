#!/usr/bin/env python3
"""BEV lane-topology autolabel from t4dataset (NuScenes-like).

Projects LiDAR points into all annotated cameras, samples the 2D panoptic
segmentation label at each projected pixel, accumulates labeled ground points
in the global frame across the scene, and rasterizes a BEV semantic map
(lane lines, stop lines, road edges, crosswalks, road, etc.).
"""
import argparse
import base64
import json
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import cv2
import numpy as np
from pycocotools import mask as cocomask

cv2.setNumThreads(0)

# ---------------------------------------------------------------- label space
UNLABELED, ROAD, SIDEWALK, CROSSWALK, LANELINE, STOPLINE, ROAD_EDGE, MARKING, PARKING = range(9)
N_CLASSES = 9

CLASS_NAMES = ["unlabeled", "road", "sidewalk", "crosswalk", "laneline",
               "stopline", "road_edge", "marking", "parking_lot"]

# dataset category name -> our class id
CATEGORY_TO_CLASS = {
    "road": ROAD,
    "sidewalk": SIDEWALK,
    "crosswalk": CROSSWALK,
    "laneline_solid_white": LANELINE,
    "dashed_lane_marking": LANELINE,
    "stopline": STOPLINE,
    "road_edge": ROAD_EDGE,
    "marking_arrow": MARKING,
    "marking_character": MARKING,
    "marking_other": MARKING,
    "deceleration_line": MARKING,
    "striped_road_marking": MARKING,
    "parking_lot": PARKING,
}

# paint order inside one image label map (later paints overwrite earlier)
PAINT_ORDER = [ROAD, PARKING, SIDEWALK, CROSSWALK, ROAD_EDGE, MARKING, STOPLINE, LANELINE]

# BEV override priority for thin classes (low -> high)
THIN_PRIORITY = [ROAD_EDGE, MARKING, STOPLINE, LANELINE]
THIN_CLASSES = np.array([ROAD_EDGE, MARKING, STOPLINE, LANELINE], dtype=np.uint8)
AREA_CLASSES = [ROAD, SIDEWALK, PARKING, CROSSWALK]
EGO_FILL_R = 1.75    # m: half a lane; road stamped under the ego path (blind zone)

PALETTE = np.array([
    [0, 0, 0],        # unlabeled
    [90, 90, 90],     # road
    [140, 90, 160],   # sidewalk
    [0, 200, 200],    # crosswalk
    [255, 255, 255],  # laneline
    [255, 40, 40],    # stopline
    [255, 140, 0],    # road_edge
    [240, 220, 60],   # marking
    [40, 60, 140],    # parking_lot
], dtype=np.uint8)


# ---------------------------------------------------------------- geometry
def quat_to_rot(q):
    """nuScenes quaternion [w, x, y, z] -> 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


class Transform:
    def __init__(self, rotation, translation):
        self.R = quat_to_rot(rotation)
        self.t = np.asarray(translation, dtype=np.float64)

    def apply(self, pts):        # sensor/child -> parent
        return pts @ self.R.T + self.t

    def inverse_apply(self, pts):  # parent -> sensor/child
        return (pts - self.t) @ self.R


# ---------------------------------------------------------------- t4 loader
class T4Scene:
    def __init__(self, root):
        self.root = root
        ann = os.path.join(root, "annotation")
        load = lambda n: json.load(open(os.path.join(ann, n + ".json")))
        self.samples = load("sample")
        self.sample_data = load("sample_data")
        self.calibrated_sensor = {c["token"]: c for c in load("calibrated_sensor")}
        self.ego_pose = {e["token"]: e for e in load("ego_pose")}
        self.sensor = {s["token"]: s for s in load("sensor")}
        cats = {c["token"]: c["name"] for c in load("category")}

        # sample_token -> {channel: sample_data}
        self.frames = defaultdict(dict)
        for d in self.sample_data:
            if not d["is_key_frame"]:
                continue
            ch = d["filename"].split("/")[1]
            self.frames[d["sample_token"]][ch] = d

        # sample_data_token -> [(class_id, mask_dict), ...]
        self.anns_by_sd = defaultdict(list)
        for name in ("surface_ann", "object_ann"):
            for a in load(name):
                cls = CATEGORY_TO_CLASS.get(cats.get(a["category_token"], ""))
                if cls is not None and a.get("mask"):
                    self.anns_by_sd[a["sample_data_token"]].append((cls, a["mask"]))

    def ordered_samples(self):
        by_tok = {s["token"]: s for s in self.samples}
        first = [s for s in self.samples if not s["prev"]][0]
        out, cur = [], first
        while True:
            out.append(cur)
            if not cur["next"]:
                break
            cur = by_tok[cur["next"]]
        return out


# ---------------------------------------------------------------- masks
def build_label_image(scene, sd_token, hw):
    """Compose panoptic masks of ground classes into one uint8 label image."""
    anns = scene.anns_by_sd.get(sd_token, [])
    if not anns:
        return None
    label = np.zeros(hw, dtype=np.uint8)
    order = {c: i for i, c in enumerate(PAINT_ORDER)}
    for cls, m in sorted(anns, key=lambda a: order.get(a[0], -1)):
        counts = base64.b64decode(m["counts"])
        rle = {"size": m["size"], "counts": counts}
        dec = cocomask.decode(rle)
        label[dec.astype(bool)] = cls
    return label


# ---------------------------------------------------------------- projection
def project_points(pts_cam, K, dist, fisheye, hw):
    """Camera-frame points -> pixel coords. Returns (uv int32 array, valid mask)."""
    h, w = hw
    x, y, z = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]
    if fisheye:
        chi = np.sqrt(x * x + y * y)
        theta = np.arctan2(chi, z)
        valid = theta < np.deg2rad(87.0)
        k1, k2, k3, k4 = dist[:4]
        t2 = theta * theta
        theta_d = theta * (1 + k1 * t2 + k2 * t2 ** 2 + k3 * t2 ** 3 + k4 * t2 ** 4)
        scale = np.where(chi > 1e-9, theta_d / np.maximum(chi, 1e-9), 0.0)
        u = K[0, 0] * x * scale + K[0, 2]
        v = K[1, 1] * y * scale + K[1, 2]
    else:
        valid = z > 0.5
        zz = np.where(valid, z, 1.0)
        u = K[0, 0] * x / zz + K[0, 2]
        v = K[1, 1] * y / zz + K[1, 2]
    valid &= (u >= 0) & (u < w) & (v >= 0) & (v < h)
    uv = np.stack([u, v], 1)
    return uv, valid


# ---------------------------------------------------------------- pipeline
_G = {}  # worker context (shared via fork)


def label_frame(args):
    """Label one keyframe's lidar points. Returns (labels uint8, xy float32)."""
    fi, sample_token = args
    scene, scene_dir = _G["scene"], _G["scene_dir"]
    cams, max_range, z_window = _G["cams"], _G["max_range"], _G["z_window"]
    debug_frames, dbg_dir = _G["debug_frames"], _G["dbg_dir"]

    frame = scene.frames[sample_token]
    ld = frame["LIDAR_CONCAT"]
    pts = np.fromfile(os.path.join(scene_dir, ld["filename"]),
                      dtype=np.float32).reshape(-1, 5)[:, :3].astype(np.float64)
    cal_l = scene.calibrated_sensor[ld["calibrated_sensor_token"]]
    T_ego_lidar = Transform(cal_l["rotation"], cal_l["translation"])
    ep_l = scene.ego_pose[ld["ego_pose_token"]]
    T_glob_ego_l = Transform(ep_l["rotation"], ep_l["translation"])

    pts_ego = T_ego_lidar.apply(pts)
    rng = np.linalg.norm(pts_ego[:, :2], axis=1)
    near = (rng < max_range) & (np.abs(pts_ego[:, 2]) < z_window)
    pts_ego = pts_ego[near]
    pts_glob = T_glob_ego_l.apply(pts_ego)

    # best (label, depth) per point across cameras
    point_label = np.zeros(len(pts_ego), dtype=np.uint8)
    point_depth = np.full(len(pts_ego), np.inf)

    for ch, sd in frame.items():
        if ch == "LIDAR_CONCAT" or (cams and ch not in cams):
            continue
        cal = scene.calibrated_sensor[sd["calibrated_sensor_token"]]
        K = np.array(cal["camera_intrinsic"], dtype=np.float64)
        dist = np.array(cal["camera_distortion"], dtype=np.float64)
        fisheye = "FISHEYE" in ch
        hw = (sd["height"], sd["width"])

        label_img = build_label_image(scene, sd["token"], hw)
        if label_img is None:
            continue

        ep_c = scene.ego_pose[sd["ego_pose_token"]]
        T_glob_ego_c = Transform(ep_c["rotation"], ep_c["translation"])
        T_ego_cam = Transform(cal["rotation"], cal["translation"])
        pts_cam = T_ego_cam.inverse_apply(T_glob_ego_c.inverse_apply(pts_glob))

        uv, valid = project_points(pts_cam, K, dist, fisheye, hw)
        idx = np.where(valid)[0]
        if not len(idx):
            continue
        ui = uv[idx].astype(np.int32)
        labels = label_img[ui[:, 1], ui[:, 0]]
        depth = np.linalg.norm(pts_cam[idx], axis=1)
        # thin structures are only reliable near the camera (pixel->ground error)
        thin = np.isin(labels, THIN_CLASSES)
        keep = (labels > 0) & (~thin | (depth < _G["thin_range"]))
        idx, labels, depth = idx[keep], labels[keep], depth[keep]
        better = depth < point_depth[idx]
        point_label[idx[better]] = labels[better]
        point_depth[idx[better]] = depth[better]

        if fi < debug_frames:
            save_debug_overlay(scene_dir, sd, label_img, uv[valid], None,
                               os.path.join(dbg_dir, f"f{fi:03d}_{ch}.jpg"))

    labeled = point_label > 0
    return (point_label[labeled], pts_glob[labeled, :2].astype(np.float32),
            point_depth[labeled].astype(np.float32))


def process_scene(scene_dir, out_dir, stride=10, max_frames=None, res=0.1,
                  max_range=35.0, thin_range=15.0, cams=None, z_window=1.0,
                  debug_frames=0, workers=1):
    os.makedirs(out_dir, exist_ok=True)
    scene = T4Scene(scene_dir)
    samples = scene.ordered_samples()[::stride]
    if max_frames:
        samples = samples[:max_frames]
    print(f"[scene] {os.path.basename(scene_dir)}: {len(samples)} frames (stride={stride})",
          flush=True)

    # ---- pass 1: trajectory extent (drop frames after localization jumps)
    traj, kept = [], []
    for s in samples:
        ld = scene.frames[s["token"]].get("LIDAR_CONCAT")
        p = scene.ego_pose[ld["ego_pose_token"]]["translation"][:2]
        if traj and np.hypot(p[0] - traj[-1][0], p[1] - traj[-1][1]) > 5.0 * stride:
            print(f"  [warn] pose jump at frame {len(traj)}, dropping rest", flush=True)
            break
        traj.append(p)
        kept.append(s)
    samples = kept
    traj = np.array(traj)
    margin = max_range + 5
    xmin, ymin = traj.min(0) - margin
    xmax, ymax = traj.max(0) + margin
    W = int(np.ceil((xmax - xmin) / res))
    H = int(np.ceil((ymax - ymin) / res))
    print(f"[grid] {W} x {H} @ {res} m  (x:[{xmin:.1f},{xmax:.1f}] y:[{ymin:.1f},{ymax:.1f}])",
          flush=True)
    counts = np.zeros((N_CLASSES, H, W), dtype=np.uint16)
    min_dist = np.full((H, W), np.inf, dtype=np.float32)

    dbg_dir = os.path.join(out_dir, "debug")
    if debug_frames:
        os.makedirs(dbg_dir, exist_ok=True)

    _G.update(scene=scene, scene_dir=scene_dir, cams=cams, max_range=max_range,
              thin_range=thin_range, z_window=z_window,
              debug_frames=debug_frames, dbg_dir=dbg_dir)

    jobs = [(fi, s["token"]) for fi, s in enumerate(samples)]
    t0 = time.time()

    def accumulate(fi, pl, xy, dep):
        ix = ((xy[:, 0] - xmin) / res).astype(np.int32)
        iy = ((xy[:, 1] - ymin) / res).astype(np.int32)
        ok = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
        np.add.at(counts, (pl[ok], iy[ok], ix[ok]), 1)
        np.minimum.at(min_dist, (iy[ok], ix[ok]), dep[ok])
        done = fi + 1
        if done % 10 == 0 or done == len(jobs):
            dt = time.time() - t0
            print(f"  {done}/{len(jobs)} frames  ({dt:.0f}s, {dt / done:.1f}s/frame)",
                  flush=True)

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for fi, r in enumerate(ex.map(label_frame, jobs, chunksize=1)):
                accumulate(fi, *r)
    else:
        for fi, job in enumerate(jobs):
            accumulate(fi, *label_frame(job))

    # ---- rasterize
    bev = rasterize(counts)

    # ego-path road fill: near the ego the LiDAR is blind (< ~1.4 m) and the
    # ground just beyond it projects onto the hood / into the nadir gap between
    # the outward cameras, so the driven strip gets no panoptic label and reads
    # as a hole. Normally motion backfills it from earlier distant views; where
    # the ego dwells (lights, congestion, parking) it stays empty. The ego is by
    # definition on drivable surface -> stamp ROAD into UNLABELED cells within
    # half a lane of the trajectory (never overwrites an observed class).
    ego_fill = np.zeros((H, W), np.uint8)
    tix_f = ((traj[:, 0] - xmin) / res).astype(np.int32)
    tiy_f = ((traj[:, 1] - ymin) / res).astype(np.int32)
    r_px = max(1, int(round(EGO_FILL_R / res)))
    for x, y in zip(tix_f, tiy_f):
        cv2.circle(ego_fill, (int(x), int(y)), r_px, 1, -1)
    fillm = (ego_fill > 0) & (bev == UNLABELED)
    bev[fillm] = ROAD
    min_dist[fillm] = np.minimum(min_dist[fillm], 0.0)  # keep in <=20 m mask

    np.save(os.path.join(out_dir, "bev_label.npy"), bev)
    np.savez_compressed(os.path.join(out_dir, "bev_counts.npz"), counts=counts,
                        min_dist=min_dist)
    meta = dict(origin=[float(xmin), float(ymin)], resolution=res,
                size=[H, W], classes=CLASS_NAMES,
                scene=os.path.basename(scene_dir), stride=stride,
                max_range=max_range, thin_range=thin_range, frames=len(samples))
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=2)

    vis = np.ascontiguousarray(PALETTE[bev][::-1])  # flip y so north-up
    cv2.imwrite(os.path.join(out_dir, "bev_label.png"), vis[:, :, ::-1])
    # confidence-masked version: keep only cells observed from <= 20 m
    bev_m = np.where(min_dist <= 20.0, bev, 0).astype(np.uint8)
    np.save(os.path.join(out_dir, "bev_label_masked.npy"), bev_m)
    vism = np.ascontiguousarray(PALETTE[bev_m][::-1])
    cv2.imwrite(os.path.join(out_dir, "bev_label_masked.png"), vism[:, :, ::-1])
    # trajectory overlay
    tix = ((traj[:, 0] - xmin) / res).astype(int)
    tiy = H - 1 - ((traj[:, 1] - ymin) / res).astype(int)
    for x, y in zip(tix, tiy):
        cv2.circle(vis, (int(x), int(y)), 3, (0, 255, 0), -1)
    cv2.imwrite(os.path.join(out_dir, "bev_label_traj.png"), vis[:, :, ::-1])
    print(f"[done] wrote {out_dir}/bev_label.png")
    return bev, counts, meta


def rasterize(counts):
    total = counts[1:].sum(0).astype(np.int32)
    bev = np.zeros(counts.shape[1:], dtype=np.uint8)
    # area classes: plain argmax
    area = np.stack([counts[c] for c in AREA_CLASSES])
    amax = area.argmax(0)
    has_area = area.sum(0) > 0
    bev[has_area] = np.array(AREA_CLASSES, dtype=np.uint8)[amax[has_area]]
    # thin classes override by priority; need meaningful support AND a
    # meaningful share of the cell's observations
    for c in THIN_PRIORITY:
        cc = counts[c].astype(np.int32)
        m = (cc >= 3) & (cc * 4 >= total)
        bev[m] = c
    return bev


def save_debug_overlay(scene_dir, sd, label_img, uv, depth, path):
    img = cv2.imread(os.path.join(scene_dir, sd["filename"]))
    if img is None:
        return
    overlay = PALETTE[label_img][:, :, ::-1]
    img = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)
    for u, v in uv[::7].astype(int):
        cv2.circle(img, (u, v), 2, (0, 255, 0), -1)
    cv2.imwrite(path, cv2.resize(img, None, fx=0.5, fy=0.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--res", type=float, default=0.1)
    ap.add_argument("--max-range", type=float, default=35.0)
    ap.add_argument("--thin-range", type=float, default=15.0)
    ap.add_argument("--z-window", type=float, default=1.0)
    ap.add_argument("--cams", default=None,
                    help="comma list; 'pinhole' = 8 pinhole cams; default all 12")
    ap.add_argument("--debug-frames", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    cams = None
    if args.cams == "pinhole":
        cams = {"CAM_FRONT_NARROW", "CAM_FRONT_WIDE", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
                "CAM_BACK_NARROW", "CAM_BACK_WIDE", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"}
    elif args.cams:
        cams = set(args.cams.split(","))

    process_scene(args.scene, args.out, stride=args.stride, max_frames=args.max_frames,
                  res=args.res, max_range=args.max_range,
                  thin_range=args.thin_range, cams=cams,
                  z_window=args.z_window, debug_frames=args.debug_frames,
                  workers=args.workers)


if __name__ == "__main__":
    main()

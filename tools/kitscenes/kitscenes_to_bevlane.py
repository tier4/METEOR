#!/usr/bin/env python3
"""Convert one KITScenes-Multimodal sequence (KIT-MRT) into the bevlane scene layout used by METEOR / METEOR2.

  usage: kitscenes_to_bevlane.py --tar data/val/<uuid>.tar --out OUT_ROOT --hr HR_ROOT [--stride 2] [--check]
         kitscenes_to_bevlane.py --scene-dir <extracted sequence> --out OUT_ROOT --hr HR_ROOT
  One sequence = 100 frames @10 Hz (10 s); --stride 2 writes 50 frames. Output scene name: KITS_<uuid>.

KITScenes side: calibration/calib.json (reference = base frame = lidar_top; T_to_reference is sensor -> reference),
  poses.txt (TUM format, map-local = UTM32N minus the origin in maps/origin.json; 1:1 with the reference timestamps),
  camera_* (6 ring cameras, 3504x2272, 87 deg; base_front_center 5856x3104, 88 deg), lidar_* (7 sensors, parquet,
  xyz discretised to 5 mm integers), maps/map.osm (Lanelet2, lat/lon). There are no 3D boxes: list the scenes in the
  no-box list of the trainer, or add auto-labels with kits_box_gt.py.
bevlane side (what training reads): manifest.json (cams K / T_ego_cam @768x432, frames), img/ (768x432),
  <hr_root>/<scene>/img (1152x648, HR_OK.json), gt/NNNN.png (BEV 800x500 @0.2 m, classes 1 road / 3 crosswalk /
  4 lane line / 5 stop line / 6 road edge / 7 marking / 8 parking, rendered from the HD map), ego_motion.npz
  (wp / v0 / acc / steer / brake / valid / pose), lidar_bev/NNNN.npz (4x400x250 f16), depth_gt4 (6x108x192) /
  depth_gt_hr (6x180x320) / depth_gt4n (2x108x192), agent_traj/NNNN.npz (empty), and for the route / map conditioned
  variant and bevlane/kitscenes_eval.py: poses10.txt (the raw 10 Hz poses) and maps/{map.osm, origin.json}.
Ego frame: the reference frame lowered by the ground height (mode of LiDAR z in the 3-12 m ring of frame 0), so ego z=0
  is the road surface.
Camera mapping: ring_front -> CAM_FRONT_WIDE, ring_front_left/right -> CAM_FRONT_LEFT/RIGHT, ring_rear -> CAM_BACK_WIDE,
  ring_rear_left/right -> CAM_BACK_LEFT/RIGHT, a centred 30 deg crop of base_front_center -> CAM_FRONT_NARROW (same field
  of view as the telephoto camera of the Co-MLOps rig). There is no CAM_BACK_NARROW (the dataset borrows a pose and
  feeds a zero image).
Requires: numpy, opencv-python, pyarrow, pyproj.
"""
import argparse, json, os, re, shutil, sys, tarfile, time
import numpy as np, cv2
import pyarrow.parquet as pq
import pyproj
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from bevlane.extract_lidar_bev import raster as lidar_raster   # noqa: E402

CAM_MAP = {"camera_ring_front": "CAM_FRONT_WIDE", "camera_ring_front_left": "CAM_FRONT_LEFT", "camera_ring_front_right": "CAM_FRONT_RIGHT",
           "camera_ring_rear": "CAM_BACK_WIDE", "camera_ring_rear_left": "CAM_BACK_LEFT", "camera_ring_rear_right": "CAM_BACK_RIGHT",
           "camera_base_front_center": "CAM_FRONT_NARROW"}
LIDARS = ["lidar_top", "lidar_front", "lidar_left", "lidar_right", "lidar_rear", "lidar_corner_left", "lidar_corner_right"]
WIDE6 = ["CAM_FRONT_WIDE", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK_WIDE", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
NARROW2 = ["CAM_FRONT_NARROW", "CAM_BACK_NARROW"]
IMG_W, IMG_H, HR_W, HR_H = 768, 432, 1152, 648
BEV_RES, BEV_XH, BEV_YH, BEV_H, BEV_W = 0.2, 80.0, 50.0, 800, 500
NARROW_HFOV_DEG = 30.0
HORIZON, WP_DT, WHEELBASE = 6, 0.5, 2.8
# Lanelet2 -> BEV classes. Ways are drawn as lines: (class, width m). bike_marking (bicycle-lane line) counts as a lane
# line (4), as in the Co-MLOps auto-labels; symbol and arrow are markings (7); road_border (edge without kerb) is a road edge (6).
WAY_CLASS = {"line_thin": (4, 0.30), "line_thick": (4, 0.45), "stop_line": (5, 0.45), "curbstone": (6, 0.30), "road_border": (6, 0.30),
             "pedestrian_marking": (3, 0.5), "arrow": (7, 0.30), "symbol": (7, 0.30), "bike_marking": (4, 0.30), "zebra_marking": (3, 0.5)}
LANELET_CLASS = {"crosswalk": 3, "parking": 8, "walkway": 2}       # every other lanelet (road, emergency_lane, bus_lane ...) is 1 road


def log(m):
    print(time.strftime("%H:%M:%S ") + m, flush=True)


def quat_yaw(qx, qy, qz, qw):
    return np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


def load_map(scene):
    """map.osm -> (node xy local [N,2], lanelets [(cls, node idx)], lines [(cls, width, node idx)]).
    The test split ships no map (poses only): an empty map is returned and the BEV GT is all zeros."""
    if not os.path.isfile(f"{scene}/maps/origin.json"):
        return np.zeros((0, 2)), [], []
    o = json.load(open(f"{scene}/maps/origin.json"))
    P = pyproj.Proj(proj="utm", zone=32, ellps="WGS84")
    ox, oy = P(o["longitude"], o["latitude"])
    txt = open(f"{scene}/maps/map.osm").read()
    nid, lat, lon = [], [], []
    for m in re.finditer(r'<node id="(\d+)"[^>]*lat="([-\d.]+)" lon="([-\d.]+)"', txt):
        nid.append(int(m.group(1))); lat.append(float(m.group(2))); lon.append(float(m.group(3)))
    xs, ys = P(np.array(lon), np.array(lat))
    xy = np.stack([xs - ox, ys - oy], 1)
    idx = {n: i for i, n in enumerate(nid)}
    ways = {}
    for m in re.finditer(r'<way id="(\d+)"[^>]*>(.*?)</way>', txt, re.S):
        body = m.group(2)
        refs = [idx[int(r)] for r in re.findall(r'<nd ref="(\d+)"', body) if int(r) in idx]
        tags = dict(re.findall(r'<tag k="([^"]+)" v="([^"]*)"', body))
        ways[int(m.group(1))] = (tags.get("type", ""), tags.get("subtype", ""), refs)
    lanelets = []
    for m in re.finditer(r'<relation id="(\d+)"[^>]*>(.*?)</relation>', txt, re.S):
        body = m.group(2); tags = dict(re.findall(r'<tag k="([^"]+)" v="([^"]*)"', body))
        if tags.get("type") != "lanelet":
            continue
        mem = dict((role, int(ref)) for ref, role in re.findall(r'<member type="way" ref="(\d+)" role="(left|right)"', body))
        if "left" not in mem or "right" not in mem or mem["left"] not in ways or mem["right"] not in ways:
            continue
        L, Rw = ways[mem["left"]][2], ways[mem["right"]][2]
        # OSM bounds may be stored against the driving direction: pick the orientation by end-point distance
        # (otherwise the polygon self-intersects and leaves triangular holes in the road)
        same = (np.linalg.norm(xy[Rw[0]] - xy[L[0]]) + np.linalg.norm(xy[Rw[-1]] - xy[L[-1]])
                <= np.linalg.norm(xy[Rw[0]] - xy[L[-1]]) + np.linalg.norm(xy[Rw[-1]] - xy[L[0]]))
        poly = L + (Rw[::-1] if same else Rw)
        if len(poly) >= 3:
            lanelets.append((LANELET_CLASS.get(tags.get("subtype", ""), 1), np.array(poly)))
    lines = [(WAY_CLASS[t][0], WAY_CLASS[t][1], np.array(r)) for (t, st, r) in ways.values() if t in WAY_CLASS and len(r) >= 2]
    return xy, lanelets, lines


def render_gt(xy_local, lanelets, lines, x, y, yaw):
    """map-local -> ego (x forward, y left) -> BEV pixels (row = (80 - x)/0.2, col = (50 - y)/0.2)."""
    c, s = np.cos(yaw), np.sin(yaw)
    dx, dy = xy_local[:, 0] - x, xy_local[:, 1] - y
    ex, ey = c * dx + s * dy, -s * dx + c * dy
    col, row = (BEV_YH - ey) / BEV_RES, (BEV_XH - ex) / BEV_RES
    pix = np.stack([col, row], 1)
    near = (np.abs(ex) < BEV_XH + 20) & (np.abs(ey) < BEV_YH + 20)
    gt = np.zeros((BEV_H, BEV_W), np.uint8)
    for cls, poly in lanelets:                                   # areas: road first, crosswalks etc. on top
        if cls == 1 and near[poly].any():
            cv2.fillPoly(gt, [np.round(pix[poly]).astype(np.int32)], 1)
    for cls, poly in lanelets:
        if cls != 1 and near[poly].any():
            cv2.fillPoly(gt, [np.round(pix[poly]).astype(np.int32)], int(cls))
    for cls, w, refs in lines:                                   # lines: width w [m] -> pixels
        if near[refs].any():
            cv2.polylines(gt, [np.round(pix[refs]).astype(np.int32)], False, int(cls), max(1, int(round(w / BEV_RES))))
    return gt


def load_lidar_ego(scene, fi, calib, T_ego_ref):
    pts = []
    for name in LIDARS:
        p = f"{scene}/{name}/{fi:010d}.parquet"
        if not os.path.isfile(p) or name not in calib:
            continue
        t = pq.read_table(p, columns=["x", "y", "z"]); r = float(t.schema.metadata[b"discretization_resolution"])
        xyz = np.stack([t["x"].to_numpy(), t["y"].to_numpy(), t["z"].to_numpy()], 1).astype(np.float64) * r
        T = T_ego_ref @ np.array(calib[name]["T_to_reference"], np.float64)
        pts.append(xyz @ T[:3, :3].T + T[:3, 3])
    return np.concatenate(pts).astype(np.float32) if pts else np.zeros((0, 3), np.float32)


def depth_grid(pe, K, T_ego_cam, dh, dw, img_w, img_h):
    """ego points -> camera projection -> per-cell minimum z on a dh x dw grid (sparse, 0 = unobserved). K is in img_w x img_h pixels."""
    Tc = np.linalg.inv(T_ego_cam)
    pc = pe @ Tc[:3, :3].T + Tc[:3, 3]
    z = pc[:, 2]; m = z > 0.5
    u = (K[0, 0] * pc[m, 0] / z[m] + K[0, 2]) * (dw / img_w); v = (K[1, 1] * pc[m, 1] / z[m] + K[1, 2]) * (dh / img_h)
    u = u.astype(np.int32); v = v.astype(np.int32); zz = z[m]
    ok = (u >= 0) & (u < dw) & (v >= 0) & (v < dh)
    d = np.zeros(dh * dw, np.float32)
    if ok.any():
        idx = v[ok] * dw + u[ok]; order = np.lexsort((zz[ok], idx)); idx_s, z_s = idx[order], zz[ok][order]
        first = np.r_[True, idx_s[1:] != idx_s[:-1]]; d[idx_s[first]] = z_s[first]
    return d.reshape(dh, dw)


def ego_motion(t, x, y, yaw, frame_idx):
    """Same definitions as bevlane/extract_ego.py: wp = future positions at +0.5..3.0 s (current ego frame), v0 = causal
    speed, valid when the whole future is available."""
    dt = np.gradient(t)
    vx, vy = np.gradient(x) / dt, np.gradient(y) / dt
    vs = np.hypot(vx, vy)
    bd = np.r_[vs[0], np.hypot(np.diff(x), np.diff(y)) / np.maximum(np.diff(t), 1e-3)]
    vs_causal = np.array([bd[max(0, i - 2):i + 1].mean() for i in range(len(bd))])
    acc = np.gradient(vs) / dt
    yr = np.gradient(np.unwrap(yaw)) / dt
    F = len(frame_idx)
    wp = np.zeros((F, HORIZON, 2), np.float32); v0 = np.zeros(F, np.float32); ac = np.zeros(F, np.float32)
    st = np.zeros(F, np.float32); br = np.zeros(F, np.float32); vd = np.zeros(F, np.float32); gp = np.zeros((F, 3), np.float32)
    v0c = np.zeros(F, np.float32)
    for fj, i in enumerate(frame_idx):
        v0[fj] = vs_causal[i]; ac[fj] = acc[i]; br[fj] = float(acc[i] < -0.5); v0c[fj] = vs[i]
        st[fj] = np.arctan(WHEELBASE * yr[i] / vs[i]) if vs[i] > 0.5 else 0.0
        gp[fj] = (x[i], y[i], yaw[i])
        tq = t[i] + WP_DT * np.arange(1, HORIZON + 1)
        if tq[-1] > t[-1] + 1e-6:
            continue
        xq, yq = np.interp(tq, t, x), np.interp(tq, t, y)
        c, s = np.cos(yaw[i]), np.sin(yaw[i]); dx, dy = xq - x[i], yq - y[i]
        wp[fj, :, 0] = c * dx + s * dy; wp[fj, :, 1] = -s * dx + c * dy; vd[fj] = 1.0
    return dict(wp=wp, v0=v0, acc=ac, steer=st, brake=br, valid=vd, pose=gp, v0_central=v0c)


def cam_crop(entry, narrow):
    """Crop (x0, y0, cw, ch) and K at 768x432 / 1152x648. Ring cameras: cut top / bottom to 16:9. Narrow: 30 deg around the principal point."""
    W, H = entry["resolution"]["width"], entry["resolution"]["height"]
    f, cx, cy = entry["intrinsics"]["focal_length"], entry["intrinsics"]["principal_point_u"], entry["intrinsics"]["principal_point_v"]
    if narrow:
        cw = int(round(2 * f * np.tan(np.radians(NARROW_HFOV_DEG / 2)))); ch = int(round(cw * 9 / 16))
        x0, y0 = int(round(cx - cw / 2)), int(round(cy - ch / 2))
    else:
        cw = W; ch = int(round(W * 9 / 16)); x0, y0 = 0, int(round((H - ch) / 2))

    def K_at(w, h):
        s = w / cw
        return [[f * s, 0.0, (cx - x0) * s], [0.0, f * s, (cy - y0) * s], [0.0, 0.0, 1.0]]
    return (x0, y0, cw, ch), K_at(IMG_W, IMG_H), K_at(HR_W, HR_H)


def copy_extras(scene_dir, od):
    """Raw 10 Hz poses and the Lanelet2 map, read by the route corridor (bevlane/ae2e_route.py) and kitscenes_eval.py."""
    shutil.copyfile(f"{scene_dir}/poses.txt", os.path.join(od, "poses10.txt"))
    if os.path.isfile(f"{scene_dir}/maps/map.osm") and os.path.isfile(f"{scene_dir}/maps/origin.json"):
        os.makedirs(os.path.join(od, "maps"), exist_ok=True)
        for fn in ("map.osm", "origin.json"):
            shutil.copyfile(f"{scene_dir}/maps/{fn}", os.path.join(od, "maps", fn))


def convert(scene_dir, out_root, hr_root, stride, check=False):
    uid = os.path.basename(scene_dir.rstrip("/")); name = f"KITS_{uid}"
    od = os.path.join(out_root, name); hd = os.path.join(hr_root, name)
    for d in ("img", "gt", "lidar_bev", "depth_gt4", "depth_gt_hr", "depth_gt4n", "agent_traj", "bev_box_p"):
        os.makedirs(os.path.join(od, d), exist_ok=True)
    os.makedirs(os.path.join(hd, "img"), exist_ok=True)
    calib = json.load(open(f"{scene_dir}/calibration/calib.json"))
    ts = np.loadtxt(f"{scene_dir}/timestamp.reference.txt", ndmin=1)
    pz = np.loadtxt(f"{scene_dir}/poses.txt", ndmin=2)
    if len(pz) != len(ts) or np.abs(pz[:, 0] - ts).max() > 0.02:
        raise RuntimeError(f"poses/timestamps mismatch {len(pz)} vs {len(ts)}")
    t, x, y = pz[:, 0], pz[:, 1], pz[:, 2]; yaw = quat_yaw(pz[:, 4], pz[:, 5], pz[:, 6], pz[:, 7])
    # ground height: mode of z in the 3-12 m ring of frame 0 -> ego = reference shifted by h (road at z=0)
    pe0 = load_lidar_ego(scene_dir, 0, calib, np.eye(4))
    rr = np.hypot(pe0[:, 0], pe0[:, 1]); zz = pe0[(rr > 3) & (rr < 12), 2]
    hcnt, edges = np.histogram(zz, bins=np.arange(-4, 1, 0.05)); h = -(edges[np.argmax(hcnt)] + 0.025)
    T_ego_ref = np.eye(4); T_ego_ref[2, 3] = h
    cams, crops, hrK = {}, {}, {}
    for kname, our in CAM_MAP.items():
        if kname not in calib:
            continue
        crop, K, Khr = cam_crop(calib[kname], narrow=(our == "CAM_FRONT_NARROW"))
        T = T_ego_ref @ np.array(calib[kname]["T_to_reference"], np.float64)
        cams[our] = {"K": K, "T_ego_cam": T.tolist()}; crops[our] = (kname, crop); hrK[our] = Khr
    xy_local, lanelets, lines = load_map(scene_dir)
    frame_idx = list(range(0, len(ts), stride))
    frames = []
    for fj, fi in enumerate(frame_idx):
        imgs = {}
        for our, (kname, (x0, y0, cw, ch)) in crops.items():
            im = cv2.imread(f"{scene_dir}/{kname}/{fi:010d}.jpg")
            if im is None:
                imgs = None; break
            im = im[y0:y0 + ch, x0:x0 + cw]
            fn = f"img/{fj:04d}_{our}.jpg"
            cv2.imwrite(os.path.join(od, fn), cv2.resize(im, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 92])
            cv2.imwrite(os.path.join(hd, fn), cv2.resize(im, (HR_W, HR_H), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 92])
            imgs[our] = fn
        if imgs is None:
            log(f"{name}: frame {fi} missing image -> skip"); continue
        gt = render_gt(xy_local, lanelets, lines, x[fi], y[fi], yaw[fi])
        cv2.imwrite(os.path.join(od, f"gt/{fj:04d}.png"), gt)
        pe = load_lidar_ego(scene_dir, fi, calib, T_ego_ref)
        np.savez_compressed(os.path.join(od, f"lidar_bev/{fj:04d}.npz"), lb=lidar_raster(pe).astype(np.float16))
        d4 = np.zeros((6, 108, 192), np.float32); dhr = np.zeros((6, 180, 320), np.float32); d4n = np.zeros((2, 108, 192), np.float32)
        for ci, c in enumerate(WIDE6):
            if c in cams:
                K = np.array(cams[c]["K"]); T = np.array(cams[c]["T_ego_cam"])
                d4[ci] = depth_grid(pe, K, T, 108, 192, IMG_W, IMG_H); dhr[ci] = depth_grid(pe, K, T, 180, 320, IMG_W, IMG_H)
        if "CAM_FRONT_NARROW" in cams:
            K = np.array(cams["CAM_FRONT_NARROW"]["K"]); T = np.array(cams["CAM_FRONT_NARROW"]["T_ego_cam"])
            d4n[0] = depth_grid(pe, K, T, 108, 192, IMG_W, IMG_H)
        np.savez_compressed(os.path.join(od, f"depth_gt4/{fj:04d}.npz"), depth=d4.astype(np.float16))
        np.savez_compressed(os.path.join(od, f"depth_gt_hr/{fj:04d}.npz"), depth=dhr.astype(np.float16))
        np.savez_compressed(os.path.join(od, f"depth_gt4n/{fj:04d}.npz"), depth=d4n.astype(np.float16))
        np.savez_compressed(os.path.join(od, f"agent_traj/{fj:04d}.npz"), boxes=np.zeros((64, 6), np.float32), count=0,
                            traj=np.zeros((64, 6, 2), np.float32), tvalid=np.zeros((64, 6), np.float32))
        np.savez_compressed(os.path.join(od, f"bev_box_p/{fj:04d}.npz"), boxes=np.zeros((0, 6), np.float32))   # empty box file expected by the evaluators
        frames.append({"frame": fj, "gt": f"gt/{fj:04d}.png", "imgs": imgs, "depth4": f"depth_gt4/{fj:04d}.npz",
                       "depth_hr": f"depth_gt_hr/{fj:04d}.npz", "depth4n": f"depth_gt4n/{fj:04d}.npz",
                       "lidar_bev": f"lidar_bev/{fj:04d}.npz", "agent_traj": f"agent_traj/{fj:04d}.npz", "bev_box_p": f"bev_box_p/{fj:04d}.npz"})
    em = ego_motion(t, x, y, yaw, frame_idx)
    np.savez_compressed(os.path.join(od, "ego_motion.npz"), **em)
    copy_extras(scene_dir, od)
    json.dump({"scene": name, "bev_res": BEV_RES, "bev_size": BEV_H, "bev_h": BEV_H, "bev_w": BEV_W, "bev_xh": BEV_XH, "bev_yh": BEV_YH,
               "img_hw": [IMG_H, IMG_W], "cams": cams, "frames": frames, "source": "KITScenes-Multimodal", "kitscenes_uuid": uid,
               "no_map": not lanelets,
               "ground_offset_m": float(h), "occ_shadow_filtered": 1, "occ_shadow_v2": 1}, open(os.path.join(od, "manifest.json"), "w"))
    json.dump({"scene": name, "hw": [HR_H, HR_W], "K": hrK, "source": "KITScenes 8 MPix crop"}, open(os.path.join(hd, "HR_OK.json"), "w"))
    if check and frames:
        check_overlay(od, frames[min(10, len(frames) - 1)], cams)
    return name, len(frames), float(h), float(em["valid"].mean())


def check_overlay(od, fr, cams):
    """Project the BEV GT (road points at z=0) into the front camera and write check_overlay.png to inspect the alignment."""
    gt = cv2.imread(os.path.join(od, fr["gt"]), 0)
    im = cv2.imread(os.path.join(od, fr["imgs"]["CAM_FRONT_WIDE"]))
    K = np.array(cams["CAM_FRONT_WIDE"]["K"]); Tc = np.linalg.inv(np.array(cams["CAM_FRONT_WIDE"]["T_ego_cam"]))
    rows, cols = np.nonzero(gt)
    ex, ey = BEV_XH - (rows + 0.5) * BEV_RES, BEV_YH - (cols + 0.5) * BEV_RES
    P = np.stack([ex, ey, np.zeros_like(ex), np.ones_like(ex)], 1) @ Tc.T
    z = P[:, 2]; m = z > 1.0
    u = (K[0, 0] * P[m, 0] / z[m] + K[0, 2]).astype(int); v = (K[1, 1] * P[m, 1] / z[m] + K[1, 2]).astype(int)
    ok = (u >= 0) & (u < IMG_W) & (v >= 0) & (v < IMG_H)
    col = {1: (80, 80, 80), 3: (255, 0, 255), 4: (0, 255, 255), 5: (0, 0, 255), 6: (0, 255, 0), 7: (255, 255, 0), 8: (255, 0, 0)}
    for uu, vv, cc in zip(u[ok], v[ok], gt[rows[m][ok], cols[m][ok]]):
        im[vv, uu] = col.get(int(cc), (255, 255, 255))
    bev = cv2.applyColorMap((gt * 28).astype(np.uint8), cv2.COLORMAP_JET); bev[gt == 0] = 0
    bev = cv2.resize(bev, (270, 432))
    cv2.imwrite(os.path.join(od, "check_overlay.png"), np.concatenate([im, bev], 1))


def regen_gt(scene_dir, out_root, stride, check=False):
    """Redraw only gt/*.png of an already converted scene from the map (images, LiDAR and depth are untouched)."""
    uid = os.path.basename(scene_dir.rstrip("/")); name = f"KITS_{uid}"; od = os.path.join(out_root, name)
    m = json.load(open(os.path.join(od, "manifest.json")))
    pz = np.loadtxt(f"{scene_dir}/poses.txt", ndmin=2); x, y = pz[:, 1], pz[:, 2]; yaw = quat_yaw(pz[:, 4], pz[:, 5], pz[:, 6], pz[:, 7])
    xy_local, lanelets, lines = load_map(scene_dir)
    for f in m["frames"]:
        fi = f["frame"] * stride
        cv2.imwrite(os.path.join(od, f["gt"]), render_gt(xy_local, lanelets, lines, x[fi], y[fi], yaw[fi]))
    copy_extras(scene_dir, od)
    if check and m["frames"]:
        check_overlay(od, m["frames"][min(10, len(m["frames"]) - 1)], {c: v for c, v in m["cams"].items()})
    return name, len(m["frames"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tar", help="one sequence archive (data/<split>/<uuid>.tar)")
    ap.add_argument("--scene-dir", help="an already extracted sequence directory")
    ap.add_argument("--out", required=True, help="bevlane data root (e.g. out/bevlane)")
    ap.add_argument("--hr", required=True, help="root of the 1152x648 images (passed to training as --hr-root)")
    ap.add_argument("--stride", type=int, default=2, help="frame stride over the 10 Hz sequence (2 = 5 Hz, as in training)")
    ap.add_argument("--check", action="store_true", help="write check_overlay.png (BEV GT projected into the front camera)")
    ap.add_argument("--keep", action="store_true", help="keep the extracted archive")
    ap.add_argument("--gt-only", action="store_true", help="only redraw gt/ (and copy poses10.txt / maps) of a converted scene")
    ap.add_argument("--extract-dir", default="")
    a = ap.parse_args()
    sd = a.scene_dir
    if a.tar:
        xd = a.extract_dir or os.path.join(a.out, "_extract"); os.makedirs(xd, exist_ok=True)
        with tarfile.open(a.tar) as tf:
            top = tf.getnames()[0].split("/")[0]; tf.extractall(xd, filter="data")
        sd = os.path.join(xd, top)
    if not sd:
        ap.error("pass --tar or --scene-dir")
    if a.gt_only:
        name, n = regen_gt(sd, a.out, a.stride, a.check); log(f"[gt-only] {name} frames={n}")
    else:
        name, n, h, vfrac = convert(sd, a.out, a.hr, a.stride, a.check)
        log(f"[ok] {name} frames={n} ground_offset={h:.2f} m wp_valid={vfrac:.2f}")
    if a.tar and not a.keep:
        shutil.rmtree(sd, ignore_errors=True)


if __name__ == "__main__":
    main()

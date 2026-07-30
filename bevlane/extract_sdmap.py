#!/usr/bin/env python3
"""FREE SD-map (OpenStreetMap) prior rasters for the BEV model (v46 idea).

Feasibility facts (checked 2026-07-28):
  * raw t4dataset ego_pose carries per-pose GNSS: geocoordinate=[lat,lon,alt]
  * the map-frame translation and the geocoordinate are BOTH per-pose, so the
    geo->map-frame similarity transform can be fitted from the scene's own
    trajectory -- no external calibration needed.

Per scene:
  1. fit 2D similarity (ENU around scene centre -> map frame) from poses
  2. fetch OSM ways (highway=*) for the trajectory bbox via Overpass
  3. render, per frame, ego-centred rasters [4,400,250] @0.4 m:
       ch0 road area (centerline buffered by class width)
       ch1 centerlines
       ch2 intersections (degree>=3 nodes)
       ch3 crossings + traffic signals
  saved as sdmap/<fi>.npz {"sd": uint8}; manifest key "sdmap".

--viz renders GT-BEV/SD-map overlays instead (alignment check).
"""
import argparse
import json
import os
import sys
import urllib.request

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GH, GW, RES = 400, 250, 0.4
WIDTH_M = {"motorway": 12, "trunk": 10, "primary": 9, "secondary": 8,
           "tertiary": 7, "residential": 5, "unclassified": 5,
           "service": 4, "living_street": 4, "motorway_link": 6,
           "trunk_link": 6, "primary_link": 6, "secondary_link": 6}


def fit_geo2map(ep):
    """similarity transform ENU(lat/lon around centre) -> map xy, fitted on
    the scene's own poses (Umeyama, 2D)."""
    geo = np.array([e["geocoordinate"][:2] for e in ep], np.float64)
    mxy = np.array([e["translation"][:2] for e in ep], np.float64)
    lat0, lon0 = geo.mean(0)
    # WGS84 meters-per-degree at lat0 (series expansion, <0.01% error);
    # fixed constants were off by ~0.4% N / ~0.1% E at Japanese latitudes.
    p = np.radians(lat0)
    m_lat = (111132.92 - 559.82 * np.cos(2 * p) + 1.175 * np.cos(4 * p)
             - 0.0023 * np.cos(6 * p))
    m_lon = (111412.84 * np.cos(p) - 93.5 * np.cos(3 * p)
             + 0.118 * np.cos(5 * p))
    ex = (geo[:, 1] - lon0) * m_lon
    ey = (geo[:, 0] - lat0) * m_lat
    src = np.stack([ex, ey], 1)
    ms, mm = src.mean(0), mxy.mean(0)
    s0, m0 = src - ms, mxy - mm
    U, S, Vt = np.linalg.svd(s0.T @ m0 / len(src))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, d])
    R = Vt.T @ D @ U.T
    # Both ENU and map are metric: scale is 1 by construction. Fitting it on
    # quantized GNSS (~90 m grid) inflates src variance and shrinks the map
    # by 8-23%, so pin it.
    scale = 1.0
    t = mm - scale * (R @ ms)
    res = (scale * (R @ src.T)).T + t - mxy
    return (lat0, lon0, m_lat, m_lon), scale, R, t, float(np.abs(res).mean())


TILE_DEG = 0.02          # ~2 km tiles; recordings cluster, so tiles are shared


def _fetch_tile(ti, tj, cache_dir):
    """Fetch (or load cached) one Overpass tile. Rate-limited across
    parallel workers by 3 flock slots with >=2 s spacing per slot."""
    cp = os.path.join(cache_dir, f"tile_{ti}_{tj}.json")
    if os.path.isfile(cp):
        try:
            return json.load(open(cp))
        except Exception:
            pass
    la0, lo0 = ti * TILE_DEG, tj * TILE_DEG
    q = f"""[out:json][timeout:60];
(way["highway"]({la0},{lo0},{la0 + TILE_DEG},{lo0 + TILE_DEG});
 node["highway"~"crossing|traffic_signals"]({la0},{lo0},{la0 + TILE_DEG},{lo0 + TILE_DEG}););
(._;>;);out body;"""
    endpoints = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
    ]
    import fcntl, time, random
    slots = list(range(3))
    random.shuffle(slots)
    lockf = None
    for s in slots:
        f = open(os.path.join(cache_dir, f".fetch{s}.lock"), "a+")
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            lockf, slot = f, s
            break
        except OSError:
            f.close()
    if lockf is None:
        slot = slots[0]
        lockf = open(os.path.join(cache_dir, f".fetch{slot}.lock"), "a+")
        fcntl.flock(lockf, fcntl.LOCK_EX)
    ts = os.path.join(cache_dir, f".fetch{slot}.ts")
    last = None
    try:
        # another worker may have fetched this tile while we waited
        if os.path.isfile(cp):
            try:
                return json.load(open(cp))
            except Exception:
                pass
        for rnd in range(4):
            for url in endpoints:
                try:
                    try:
                        dt = time.time() - os.path.getmtime(ts)
                    except OSError:
                        dt = 1e9
                    if dt < 2.0:
                        time.sleep(2.0 - dt)
                    req = urllib.request.Request(
                        url, data=q.encode(),
                        headers={"User-Agent": "METEOR-sdmap/1.0"})
                    with urllib.request.urlopen(req, timeout=45) as r:
                        data = json.loads(r.read())
                    open(ts, "w").close()
                    tmp = cp + f".tmp{os.getpid()}"
                    json.dump(data, open(tmp, "w"))
                    os.replace(tmp, cp)
                    return data
                except Exception as e:
                    last = e
                    print(f"[fetch_osm] {url} failed: {e}", flush=True)
                    open(ts, "w").close()
            time.sleep(10 * (rnd + 1))
    finally:
        fcntl.flock(lockf, fcntl.LOCK_UN)
    raise last


def fetch_osm(lat_min, lat_max, lon_min, lon_max):
    cache_dir = os.environ.get("METEOR_OSM_CACHE", "out/osm_cache")
    os.makedirs(cache_dir, exist_ok=True)
    els = {}
    for ti in range(int(np.floor(lat_min / TILE_DEG)),
                    int(np.floor(lat_max / TILE_DEG)) + 1):
        for tj in range(int(np.floor(lon_min / TILE_DEG)),
                        int(np.floor(lon_max / TILE_DEG)) + 1):
            d = _fetch_tile(ti, tj, cache_dir)
            for e in d.get("elements", []):
                els[(e["type"], e["id"])] = e
    return {"elements": list(els.values())}


def build_scene_map(raw_dir):
    ep = json.load(open(os.path.join(raw_dir, "annotation/ego_pose.json")))
    ep = [e for e in ep if e.get("geocoordinate")]
    ep.sort(key=lambda e: e["timestamp"])
    (lat0, lon0, m_lat, m_lon), scale, R, t, err = fit_geo2map(ep)
    geo = np.array([e["geocoordinate"][:2] for e in ep])
    pad = 0.004                                  # ~400 m
    osm = fetch_osm(geo[:, 0].min() - pad, geo[:, 0].max() + pad,
                    geo[:, 1].min() - pad, geo[:, 1].max() + pad)
    nodes = {el["id"]: (el["lat"], el["lon"]) for el in osm["elements"]
             if el["type"] == "node"}

    def to_map(lat, lon):
        ex = (lon - lon0) * m_lon
        ey = (lat - lat0) * m_lat
        return scale * (R @ np.array([ex, ey])) + t

    ways, widths = [], []
    node_use = {}
    for el in osm["elements"]:
        if el["type"] != "way" or "highway" not in el.get("tags", {}):
            continue
        hw = el["tags"]["highway"]
        if hw in ("footway", "path", "steps", "cycleway", "pedestrian",
                  "track", "bridleway", "corridor"):
            continue
        pts = np.array([to_map(*nodes[n]) for n in el["nodes"]
                        if n in nodes])
        if len(pts) < 2:
            continue
        w = el["tags"].get("width")
        try:
            w = float(w)
        except (TypeError, ValueError):
            lanes = el["tags"].get("lanes")
            w = (float(lanes) * 3.2 if lanes and lanes.isdigit()
                 else WIDTH_M.get(hw, 6))
        ways.append(pts); widths.append(float(w))
        for n in el["nodes"]:
            node_use[n] = node_use.get(n, 0) + 1
    inters = np.array([to_map(*nodes[n]) for n, c in node_use.items()
                       if c >= 3 and n in nodes]) if node_use else np.zeros((0, 2))
    cross = np.array([to_map(el["lat"], el["lon"]) for el in osm["elements"]
                      if el["type"] == "node"
                      and el.get("tags", {}).get("highway") in
                      ("crossing", "traffic_signals")])
    # sign metadata (map frame) for demo icons / future v47 input channels
    meta = {"signals": [], "stops": [], "speed_cameras": [], "maxspeed": []}
    for el in osm["elements"]:
        if el["type"] != "node":
            continue
        hw = el.get("tags", {}).get("highway")
        if hw == "traffic_signals":
            meta["signals"].append(to_map(el["lat"], el["lon"]).tolist())
        elif hw in ("stop", "give_way"):
            meta["stops"].append(to_map(el["lat"], el["lon"]).tolist())
        elif hw == "speed_camera":
            meta["speed_cameras"].append(to_map(el["lat"], el["lon"]).tolist())
    for el in osm["elements"]:
        if el["type"] == "way" and "maxspeed" in el.get("tags", {}):
            try:
                v = float(el["tags"]["maxspeed"])
            except ValueError:
                continue
            pts = [to_map(*nodes[n]).tolist() for n in el["nodes"]
                   if n in nodes]
            if len(pts) >= 2:
                meta["maxspeed"].append({"kmh": v, "pts": pts})
    return ways, widths, inters, cross, err, meta


def render_frame(ways, widths, inters, cross, pose):
    """pose = (x, y, yaw) map frame -> raster [4,GH,GW] uint8."""
    x0, y0, yaw = pose
    c, s = np.cos(-yaw), np.sin(-yaw)
    sd = np.zeros((4, GH, GW), np.uint8)

    def to_px(P):
        X = c * (P[:, 0] - x0) - s * (P[:, 1] - y0)
        Y = s * (P[:, 0] - x0) + c * (P[:, 1] - y0)
        r = (80.0 - X) / RES
        q = (50.0 - Y) / RES
        return np.stack([q, r], 1)               # (col,row) for cv2

    for P, w in zip(ways, widths):
        px = to_px(P).astype(np.int32)
        cv2.polylines(sd[0], [px.reshape(-1, 1, 2)], False, 1,
                      max(1, int(w / RES)))
        cv2.polylines(sd[1], [px.reshape(-1, 1, 2)], False, 1, 1)
    for arr, ch, rad in ((inters, 2, 10), (cross, 3, 5)):
        if len(arr):
            for p in to_px(arr).astype(np.int32):
                if -50 < p[0] < GW + 50 and -50 < p[1] < GH + 50:
                    cv2.circle(sd[ch], tuple(p), rad, 1, -1)
    return sd


def refine_alignment(ways, widths, eg_pose, root, scene, man, cross=None):
    """The exported geocoordinate is rounded to ~6 significant digits
    (lon 3 decimals = ~90 m quantisation!), so the geo fit is only a
    coarse seed. Refine an SE(2) map-frame correction per scene against
    the trusted GT road. Score = fraction of OSM *centerline* pixels
    inside GT road (sharp optimum even when rendered road width does not
    match the painted GT width) + 0.5 * road-area IoU (breaks lateral
    ties toward centered). The correction is time-varying: a global
    coarse fit, then local re-fits on early/mid/late anchor windows,
    linearly interpolated over frames — GNSS quantisation error is not
    constant along a 30 s scene. (At deployment the vehicle's own
    localisation provides the accurate pose; this rounding is a dataset
    export artifact.)"""
    n = len(man["frames"])
    step = max(1, (n - 15) // 8)
    frames = [f for f in man["frames"][10:n - 4:step] if f.get("gt")]
    gts = []
    cwc = {}                   # GT crosswalk centroids per anchor (ego m)
    for f in frames:
        g = cv2.imread(os.path.join(root, scene, f["gt"]), 0)
        if g is None:
            continue
        g = cv2.resize(g, (GW, GH), interpolation=cv2.INTER_NEAREST)
        # road area = road + crosswalk + markings (all painted ON the road)
        gts.append((f["frame"], (g == 1) | (g == 3) | (g == 7)))
        nlab, _, stats, cent = cv2.connectedComponentsWithStats(
            (g == 3).astype(np.uint8))   # class 3 = CROSSWALK (7 is MARKING:
                                         # lane arrows, painted BEFORE the
                                         # crossing -- matching those pulled
                                         # the map tens of meters backward)
        pts = [(80.0 - cent[k][1] * RES, 50.0 - cent[k][0] * RES)
               for k in range(1, nlab)
               if stats[k, cv2.CC_STAT_AREA] >= 6]
        if pts:
            cwc[f["frame"]] = np.array(pts)
    zero = np.zeros(len(eg_pose))
    if not gts:
        return zero, zero, zero, -1.0

    # Drop clip-boundary anchors: near the ends of a 30 s clip the GT only
    # covers where the ego has driven (e.g. a scene ending at a red light
    # has NO road beyond the stop line). Matching the complete OSM map to
    # such one-sided GT drags the whole map tens of meters along-track.
    def _balanced(gm):
        ahead = int(gm[:GH // 2 - 25].sum())     # x > +10 m
        behind = int(gm[GH // 2 + 25:].sum())    # x < -10 m
        return min(ahead, behind) > 2500
    bal_keys = {fi for fi, gm in gts if _balanced(gm)}

    # Restrict scoring to a corridor around the driven trajectory: on city
    # grids a parallel avenue rotated onto the GT road keeps the plain score
    # flat over tens of degrees (verified on a straight avenue scene).
    traj = eg_pose[::4, :2]
    gts_full = list(gts)      # unmasked: keeps the gate metric comparable
    side_gm = {}              # GT road OUTSIDE the corridor = side-street
    cors = {}
    for i, (fi, gm) in enumerate(gts):
        x0, y0, yaw = eg_pose[fi]
        cs, sn = np.cos(-yaw), np.sin(-yaw)
        X = cs * (traj[:, 0] - x0) - sn * (traj[:, 1] - y0)
        Y = sn * (traj[:, 0] - x0) + cs * (traj[:, 1] - y0)
        px = np.stack([(50.0 - Y) / RES, (80.0 - X) / RES],
                      1).astype(np.int32)
        cor = np.zeros((GH, GW), np.uint8)
        cv2.polylines(cor, [px.reshape(-1, 1, 2)], False, 1, int(16 / RES))
        # side-street mouths: LiDAR sees intersection openings even on the
        # not-yet-driven side, so this is the one along-track feature that
        # exists fore AND aft (GT crosswalks only complete behind the ego)
        side_gm[fi] = gm & (cor == 0)
        cors[fi] = cor > 0
        gts[i] = (fi, gm & (cor > 0))

    gts_all = list(gts)
    if len(bal_keys) >= 3:
        gts = [(fi, gm) for fi, gm in gts if fi in bal_keys]

    # analytic rotation seed: the way under the ego must parallel ego yaw
    diffs = []
    for fi, _ in gts:
        x0, y0, yaw = eg_pose[fi]
        bd, bang = 1e9, None
        for P in ways:
            d2 = (P[:, 0] - x0) ** 2 + (P[:, 1] - y0) ** 2
            j = int(np.argmin(d2))
            if d2[j] < bd and len(P) > 1:
                k = min(j, len(P) - 2)
                seg = P[k + 1] - P[k]
                bd, bang = d2[j], np.arctan2(seg[1], seg[0])
        if bang is not None and bd < 30 ** 2:
            # correction dth must take rendered way angle to ego heading
            d = (bang - yaw + np.pi / 2) % np.pi - np.pi / 2
            diffs.append(d)
    th0 = float(np.median(diffs)) if diffs else 0.0
    if abs(th0) > np.radians(25):
        th0 = 0.0

    have_cw = cross is not None and len(cross) and cwc

    def score(sub, dx, dy, dth, cw=None):  # cw = match tol in m (None=off)
        tot = 0.0
        for fi, gm in sub:
            x0, y0, yaw = eg_pose[fi]
            sd = render_frame(ways, widths, np.zeros((0, 2)),
                              np.zeros((0, 2)),
                              (x0 + dx, y0 + dy, yaw + dth))
            road, cl = sd[0].astype(bool), sd[1].astype(bool)
            inside = float((cl & gm).sum()) / max(float(cl.sum()), 1.0)
            inter = float((road & gm).sum())
            union = float((road | gm).sum())
            tot += inside + 0.5 * inter / max(union, 1.0)
            # intersection openings: OSM side-street area must land on GT
            # road outside the corridor (visible fore AND aft) -- the only
            # along-track feature not biased to behind-the-ego
            side_sd = road & ~cors[fi]
            n_sd = float(side_sd.sum())
            if n_sd > 50:
                tot += 0.6 * float((side_sd & side_gm[fi]).sum()) / n_sd
            # crosswalk anchor: GT crosswalk blobs vs OSM crossing nodes.
            # Point features break the parallel-avenue ambiguity of grids.
            if cw is not None and have_cw and len(cwc.get(fi, ())) >= 2:
                cs2 = np.cos(-(yaw + dth))
                sn2 = np.sin(-(yaw + dth))
                P = cross - [x0 + dx, y0 + dy]
                ex = cs2 * P[:, 0] - sn2 * P[:, 1]
                ey = sn2 * P[:, 0] + cs2 * P[:, 1]
                C = cwc[fi]
                dd = np.hypot(C[:, 0, None] - ex[None],
                              C[:, 1, None] - ey[None]).min(1)
                # support term only: too strong a weight (0.8, 6 m) let
                # dense-crosswalk grids support a wrongly rotated fit
                tot += 0.4 * float((dd < cw).mean())
        return tot / len(sub)

    def iou(sub, dx, dy, dth):
        tot = 0.0
        for fi, gm in sub:
            x0, y0, yaw = eg_pose[fi]
            sd = render_frame(ways, widths, np.zeros((0, 2)),
                              np.zeros((0, 2)),
                              (x0 + dx, y0 + dy, yaw + dth))
            road = sd[0].astype(bool)
            tot += float((road & gm).sum()) / max(float((road | gm).sum()), 1.0)
        return tot / len(sub)

    # --- global fit ---
    coarse = gts[::2] or gts   # half the anchors: coarse stage dominates cost
    best = (0.0, 0.0, th0, score(coarse, 0, 0, th0, cw=10.0))
    for dx in range(-60, 61, 6):
        for dy in range(-60, 61, 6):
            sc = score(coarse, dx, dy, th0, cw=10.0)                 - 0.0004 * float(np.hypot(dx, dy))
            if sc > best[3]:
                best = (float(dx), float(dy), th0, sc)
    # re-baseline on the full anchor set before comparing across stages
    # (keep the seeded rotation — resetting it to 0 here re-tilted grids)
    best = (best[0], best[1], best[2],
            score(gts, best[0], best[1], best[2]))
    # rotation and translation couple on city grids: alternate them.
    # Straight-line scenes constrain the geo-fit rotation poorly under the
    # ~90 m GNSS quantisation, so the first sweep must be wide (±16 deg).
    for rng, st in ((16, 4), (4, 2)):
        for dth in best[2] + np.radians(
                [v for v in range(-rng, rng + 1, st) if v]):
            sc = score(gts, best[0], best[1], dth)
            if sc > best[3]:
                best = (best[0], best[1], float(dth), sc)
        for dx in np.arange(best[0] - 18, best[0] + 18.1, 3.0):
            for dy in np.arange(best[1] - 18, best[1] + 18.1, 3.0):
                sc = score(gts, dx, dy, best[2], cw=10.0)                     - 0.0004 * float(np.hypot(dx, dy))
                if sc > best[3]:
                    best = (float(dx), float(dy), best[2], sc)
    bx, by, bth = best[0], best[1], best[2]

    # --- along-track stage: the corridor/area terms barely change when the
    # map slides along a straight road, so the longitudinal offset must be
    # pinned by the crosswalk point anchors over a wide range ---
    if have_cw:
        yaw_mid = float(np.median([eg_pose[fi][2] for fi, _ in gts]))
        ca, sa = np.cos(yaw_mid), np.sin(yaw_mid)
        sb = (bx, by, score(gts, bx, by, bth, cw=4.0))
        for s_ in np.arange(-36, 36.1, 3.0):
            sc = score(gts, bx + s_ * ca, by + s_ * sa, bth, cw=4.0)
            if sc > sb[2]:
                sb = (float(bx + s_ * ca), float(by + s_ * sa), sc)
        bx, by = sb[0], sb[1]

    best = (bx, by, bth, score(gts, bx, by, bth, cw=4.0))
    for dx in np.arange(bx - 3, bx + 3.1, 1.0):
        for dy in np.arange(by - 3, by + 3.1, 1.0):
            for dth in bth + np.radians([-1.5, -0.75, 0, 0.75, 1.5]):
                sc = score(gts, dx, dy, dth, cw=4.0)
                if sc > best[3]:
                    best = (float(dx), float(dy), float(dth), sc)
    bx, by, bth = best[0], best[1], best[2]

    # --- per-anchor drift tracking ---
    # The quantized GNSS makes the geo-fit correction vary by tens of
    # meters WITHIN one 30 s scene; a global fit + small window nudges
    # cannot follow it. Estimate an independent along-track / lateral
    # offset per anchor from local point/opening features, reject
    # low-confidence anchors, median-smooth, and interpolate.
    anchors_f, s_arr, t_arr = [], [], []
    for fi, gm in gts_all:
        yaw_i = eg_pose[fi][2]
        ca, sa = np.cos(yaw_i), np.sin(yaw_i)
        na, nb = -sa, ca                      # lateral (left) direction
        sub = [(fi, gm)]

        def lsc(s_, t_):
            return score(sub, bx + s_ * ca + t_ * na,
                         by + s_ * sa + t_ * nb, bth, cw=4.0)

        cand = np.arange(-50, 50.1, 2.0)
        vals = np.array([lsc(s_, 0.0) for s_ in cand])
        j = int(np.argmax(vals))
        conf = float(vals[j] - np.median(vals))
        if conf < 0.15:
            continue                           # ambiguous anchor: skip
        s_i = float(cand[j])
        tc = np.arange(-8, 8.1, 1.0)
        tv = np.array([lsc(s_i, t_) for t_ in tc])
        t_i = float(tc[int(np.argmax(tv))])
        anchors_f.append(float(fi))
        s_arr.append(s_i)
        t_arr.append(t_i)

    fi_all = np.arange(len(eg_pose), dtype=np.float64)

    def theil_sen(f, v, max_slope):
        # robust linear drift model: SLAM-vs-GNSS drift is smooth, so a
        # line beats interpolating noisy per-anchor picks (which wandered)
        f = np.asarray(f, np.float64); v = np.asarray(v, np.float64)
        sl = [(v[j] - v[i]) / (f[j] - f[i])
              for i in range(len(f)) for j in range(i + 1, len(f))
              if f[j] - f[i] >= 8]
        b = float(np.clip(np.median(sl) if sl else 0.0,
                          -max_slope, max_slope))
        a = float(np.median(v - b * f))
        return a, b

    if len(anchors_f) >= 4:
        a_s, b_s = theil_sen(anchors_f, s_arr, 0.30)
        a_t, b_t = theil_sen(anchors_f, t_arr, 0.10)
        s_f = np.clip(a_s + b_s * fi_all, -55, 55)
        t_f = np.clip(a_t + b_t * fi_all, -12, 12)
        yawf = eg_pose[:, 2]
        dxa = bx + s_f * np.cos(yawf) - t_f * np.sin(yawf)
        dya = by + s_f * np.sin(yawf) + t_f * np.cos(yawf)
        dtha = np.full(len(eg_pose), bth)
    else:
        dxa = np.full(len(eg_pose), bx)
        dya = np.full(len(eg_pose), by)
        dtha = np.full(len(eg_pose), bth)

    miou = np.mean([iou([(fi, gm)], dxa[fi], dya[fi], dtha[fi])
                    for fi, gm in gts_full])
    return dxa, dya, dtha, float(miou)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, help="converted scene name")
    ap.add_argument("--raw", required=True, help="raw t4dataset dir")
    ap.add_argument("--root", default="out/bevlane")
    ap.add_argument("--viz", default=None, help="write overlay mp4 and exit")
    args = ap.parse_args()

    ways, widths, inters, cross, err, meta = build_scene_map(args.raw)
    print(f"OSM: {len(ways)} ways, {len(inters)} intersections, "
          f"{len(cross)} crossings | geo-fit residual {err:.2f} m", flush=True)

    eg = np.load(os.path.join(args.root, args.scene, "ego_motion.npz"))
    pose = eg["pose"].astype(np.float64)
    man0 = json.load(open(os.path.join(args.root, args.scene,
                                       "manifest.json")))
    dx, dy, dth, iou = refine_alignment(ways, widths, pose, args.root,
                                        args.scene, man0, cross=cross)
    print(f"[refine] dx={dx.min():+.1f}..{dx.max():+.1f} "
          f"dy={dy.min():+.1f}..{dy.max():+.1f} "
          f"dth={np.degrees(dth.min()):+.1f}..{np.degrees(dth.max()):+.1f}deg "
          f"road-IoU={iou:.3f}", flush=True)
    pose = pose.copy()
    pose[:, 0] += dx
    pose[:, 1] += dy
    pose[:, 2] += dth
    if args.viz:
        from autolabel_bev import PALETTE
        PAL = np.zeros((256, 3), np.uint8); PAL[:len(PALETTE)] = PALETTE
        man = json.load(open(os.path.join(args.root, args.scene,
                                          "manifest.json")))
        vw = None
        for f in man["frames"][::2]:
            fi = f["frame"]
            g = cv2.imread(os.path.join(args.root, args.scene, f["gt"]), 0)
            img = PAL[np.where(g == 255, 0, g)][:, :, ::-1].copy()
            img = cv2.resize(img, (GW * 2, GH * 2),
                             interpolation=cv2.INTER_NEAREST)
            sd = render_frame(ways, widths, inters, cross, pose[fi])
            up = lambda a: cv2.resize(a * 255, (GW * 2, GH * 2),
                                      interpolation=cv2.INTER_NEAREST)
            img[up(sd[0]) > 0] = (img[up(sd[0]) > 0] * 0.55
                                  + np.array([80, 40, 0]) * 0.45)
            img[up(sd[1]) > 0] = (0, 200, 255)
            img[up(sd[2]) > 0] = (255, 120, 255)
            img[up(sd[3]) > 0] = (0, 255, 80)
            cv2.putText(img, "GT BEV + OSM SD-map overlay", (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            if vw is None:
                vw = cv2.VideoWriter(args.viz.replace(".mp4", "_raw.mp4"),
                                     cv2.VideoWriter_fourcc(*"mp4v"), 10,
                                     (GW * 2, GH * 2))
            vw.write(img)
        vw.release()
        import subprocess
        subprocess.run(["ffmpeg", "-y", "-i",
                        args.viz.replace(".mp4", "_raw.mp4"), "-c:v",
                        "libx264", "-crf", "24", "-pix_fmt", "yuv420p",
                        args.viz], check=True, capture_output=True)
        os.remove(args.viz.replace(".mp4", "_raw.mp4"))
        print(f"viz -> {args.viz}", flush=True)
        return

    if iou < 0.15:
        # Alignment gate: a wrong-basin fit would inject a confidently wrong
        # prior. Leave the scene raster-less — dataset falls back to zeros,
        # which is bit-equal to "no map" for the v46 stem.
        print(f"[gate] road-IoU {iou:.3f} < 0.15 — no rasters written",
              flush=True)
        return
    od = os.path.join(args.root, args.scene, "sdmap")
    os.makedirs(od, exist_ok=True)
    # per-scene refinement correction + sign metadata (map frame)
    meta["refine"] = {"dx": dx.tolist(), "dy": dy.tolist(),
                      "dth": dth.tolist(), "iou": iou}
    json.dump(meta, open(os.path.join(od, "meta.json"), "w"))
    man_p = os.path.join(args.root, args.scene, "manifest.json")
    man = json.load(open(man_p))
    for f in man["frames"]:
        fi = f["frame"]
        sd = render_frame(ways, widths, inters, cross, pose[fi])
        np.savez_compressed(os.path.join(od, f"{fi:04d}.npz"), sd=sd)
        f["sdmap"] = f"sdmap/{fi:04d}.npz"
    man["sdmap"] = 1
    json.dump(man, open(man_p, "w"))
    print("saved rasters + manifest", flush=True)


if __name__ == "__main__":
    main()

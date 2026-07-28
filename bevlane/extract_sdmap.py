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
    ex = (geo[:, 1] - lon0) * np.cos(np.radians(lat0)) * 111320.0
    ey = (geo[:, 0] - lat0) * 110540.0
    src = np.stack([ex, ey], 1)
    ms, mm = src.mean(0), mxy.mean(0)
    s0, m0 = src - ms, mxy - mm
    U, S, Vt = np.linalg.svd(s0.T @ m0 / len(src))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, d])
    R = Vt.T @ D @ U.T
    scale = np.trace(np.diag(S) @ D) / (s0 ** 2).sum() * len(src)
    t = mm - scale * (R @ ms)
    res = (scale * (R @ src.T)).T + t - mxy
    return (lat0, lon0), scale, R, t, float(np.abs(res).mean())


def fetch_osm(lat_min, lat_max, lon_min, lon_max):
    q = f"""[out:json][timeout:60];
(way["highway"]({lat_min},{lon_min},{lat_max},{lon_max});
 node["highway"~"crossing|traffic_signals"]({lat_min},{lon_min},{lat_max},{lon_max}););
(._;>;);out body;"""
    req = urllib.request.Request(
        "https://overpass-api.de/api/interpreter",
        data=q.encode(), headers={"User-Agent": "METEOR-sdmap/1.0"})
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.loads(r.read())


def build_scene_map(raw_dir):
    ep = json.load(open(os.path.join(raw_dir, "annotation/ego_pose.json")))
    ep = [e for e in ep if e.get("geocoordinate")]
    ep.sort(key=lambda e: e["timestamp"])
    (lat0, lon0), scale, R, t, err = fit_geo2map(ep)
    geo = np.array([e["geocoordinate"][:2] for e in ep])
    pad = 0.004                                  # ~400 m
    osm = fetch_osm(geo[:, 0].min() - pad, geo[:, 0].max() + pad,
                    geo[:, 1].min() - pad, geo[:, 1].max() + pad)
    nodes = {el["id"]: (el["lat"], el["lon"]) for el in osm["elements"]
             if el["type"] == "node"}

    def to_map(lat, lon):
        ex = (lon - lon0) * np.cos(np.radians(lat0)) * 111320.0
        ey = (lat - lat0) * 110540.0
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
    return ways, widths, inters, cross, err


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, help="converted scene name")
    ap.add_argument("--raw", required=True, help="raw t4dataset dir")
    ap.add_argument("--root", default="out/bevlane")
    ap.add_argument("--viz", default=None, help="write overlay mp4 and exit")
    args = ap.parse_args()

    ways, widths, inters, cross, err = build_scene_map(args.raw)
    print(f"OSM: {len(ways)} ways, {len(inters)} intersections, "
          f"{len(cross)} crossings | geo-fit residual {err:.2f} m", flush=True)

    eg = np.load(os.path.join(args.root, args.scene, "ego_motion.npz"))
    pose = eg["pose"]
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

    od = os.path.join(args.root, args.scene, "sdmap")
    os.makedirs(od, exist_ok=True)
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

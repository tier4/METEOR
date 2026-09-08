#!/usr/bin/env python3
"""Vectorize BEV label maps into connected polylines / polygons (map frame).

Line classes (laneline, road_edge, stopline): skeletonize -> trace segments ->
link across small gaps (dashed lines) -> simplify.
Area class (crosswalk): polygonize contours.

Output: vector_map.json with world-coordinate polylines, + visualization PNG.
"""
import argparse
import json
import os
from collections import defaultdict

import cv2
import numpy as np
from scipy.interpolate import splev, splprep
from skimage.morphology import skeletonize

from autolabel_bev import (CROSSWALK, LANELINE, PALETTE, ROAD_EDGE, SIDEWALK,
                           STOPLINE)

# per-class: (class id, close kernel, link gap [m], link angle [deg],
#             simplify eps [m], min polyline length [m], polyfit degree)
LINE_CLASSES = {
    "laneline":  (LANELINE, 5, 8.0, 20.0, 0.10, 2.0, 3),
    "road_edge": (ROAD_EDGE, 9, 6.0, 35.0, 0.20, 2.5, 3),
    "stopline":  (STOPLINE, 5, 2.5, 30.0, 0.10, 1.2, 2),
}
# polygon classes: (class id, close kernel, min area [m^2], approx eps [m],
#                   blur, rect_fit)
POLY_CLASSES = {
    "crosswalk": (CROSSWALK, 9, 3.0, 0.6, 9, True),
    "sidewalk":  (SIDEWALK, 15, 8.0, 0.40, 9, False),
}
NEI = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def prune_spurs(skel, max_len=12):
    """Iteratively remove endpoint branches shorter than max_len pixels."""
    skel = skel.copy()
    for _ in range(3):
        pix = set(zip(*(a.tolist() for a in np.nonzero(skel))))
        if not pix:
            break
        deg = {p: sum((p[0] + dy, p[1] + dx) in pix for dy, dx in NEI) for p in pix}
        removed = 0
        for p, d in list(deg.items()):
            if d != 1 or p not in pix:
                continue
            chain, prev, cur = [p], None, p
            while len(chain) <= max_len:
                nxt = None
                for dy, dx in NEI:
                    q = (cur[0] + dy, cur[1] + dx)
                    if q in pix and q != prev and q not in chain:
                        nxt = q
                        break
                if nxt is None:
                    break
                if sum((nxt[0] + dy, nxt[1] + dx) in pix for dy, dx in NEI) > 2:
                    for c in chain:  # branch reaches a junction: prune it
                        skel[c] = False
                        pix.discard(c)
                    removed += 1
                    break
                chain.append(nxt)
                prev, cur = cur, nxt
        if not removed:
            break
    return skel


# ------------------------------------------------------------- skeleton trace
def trace_skeleton(skel):
    """Trace a skeleton image into pixel-chain segments between endpoints/junctions."""
    ys, xs = np.nonzero(skel)
    pix = set(zip(ys.tolist(), xs.tolist()))
    deg = {}
    for p in pix:
        deg[p] = sum((p[0] + dy, p[1] + dx) in pix for dy, dx in NEI)
    nodes = {p for p, d in deg.items() if d != 2}  # endpoints & junctions
    segments = []
    visited_edges = set()

    def walk(start, first):
        chain = [start, first]
        prev, cur = start, first
        while cur not in nodes:
            nxt = None
            for dy, dx in NEI:
                q = (cur[0] + dy, cur[1] + dx)
                if q in pix and q != prev:
                    nxt = q
                    break
            if nxt is None:
                break
            chain.append(nxt)
            prev, cur = cur, nxt
            if len(chain) > 200000:
                break
        return chain

    for n in nodes:
        for dy, dx in NEI:
            q = (n[0] + dy, n[1] + dx)
            if q not in pix:
                continue
            ek = (n, q)
            if ek in visited_edges:
                continue
            chain = walk(n, q)
            visited_edges.add((chain[0], chain[1]))
            visited_edges.add((chain[-1], chain[-2]))
            if len(chain) >= 3:
                segments.append(chain)
    # isolated loops (all deg==2): pick arbitrary start
    covered = set()
    for s in segments:
        covered.update(s)
    rest = pix - covered - nodes
    while rest:
        start = next(iter(rest))
        for dy, dx in NEI:
            q = (start[0] + dy, start[1] + dx)
            if q in pix:
                chain = walk(start, q)
                if len(chain) >= 3:
                    segments.append(chain)
                rest -= set(chain)
                break
        else:
            rest.discard(start)
    return segments


def seg_to_xy(seg, res, x0, y0):
    a = np.array(seg, dtype=np.float64)
    return np.stack([x0 + (a[:, 1] + 0.5) * res, y0 + (a[:, 0] + 0.5) * res], 1)


def simplify(xy, eps):
    pts = xy.astype(np.float32).reshape(-1, 1, 2)
    out = cv2.approxPolyDP((pts / eps).astype(np.float32), 1.0, False)
    return out.reshape(-1, 2) * eps


def smooth_polyline(xy, deg=3, step=1.0, spline_len=40.0, tol=0.15):
    """Polynomial / spline smoothing of a polyline (arc-length parameterized).

    Short lines: single least-squares polyfit x(s), y(s) of degree <= deg.
    Long lines: smoothing cubic B-spline (piecewise 3rd-degree polynomial).
    """
    keep = np.ones(len(xy), bool)
    keep[1:] = np.linalg.norm(np.diff(xy, axis=0), axis=1) > 1e-6
    xy = xy[keep]
    if len(xy) < 3:
        return xy
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    L = s[-1]
    if L < 0.5:
        return xy
    n_out = max(int(np.ceil(L / step)) + 1, 2)
    if L <= spline_len or len(xy) < 8:
        d = int(min(deg, len(xy) - 1, 3))
        px = np.polyfit(s, xy[:, 0], d)
        py = np.polyfit(s, xy[:, 1], d)
        ss = np.linspace(0, L, n_out)
        return np.stack([np.polyval(px, ss), np.polyval(py, ss)], 1)
    try:
        tck, _ = splprep([xy[:, 0], xy[:, 1]], u=s / L, s=len(xy) * tol ** 2, k=3)
        ss = np.linspace(0, 1, n_out)
        x, y = splev(ss, tck)
        return np.stack([x, y], 1)
    except Exception:
        return xy


# ------------------------------------------------------------- gap linking
def link_segments(polys, max_gap=4.0, max_angle_deg=25.0):
    """Join polylines whose endpoints are close and aligned (dash gaps).

    Single greedy pass over KD-tree candidate endpoint pairs sorted by gap.
    """
    from scipy.spatial import cKDTree

    polys = [p for p in polys if len(p) >= 2]
    n = len(polys)
    if n < 2:
        return polys
    # endpoint table: 2i = head, 2i+1 = tail; outward = direction pointing away
    pts = np.empty((2 * n, 2))
    outward = np.empty((2 * n, 2))
    for i, p in enumerate(polys):
        pts[2 * i], pts[2 * i + 1] = p[0], p[-1]
        for k, d in ((2 * i, p[0] - p[1]), (2 * i + 1, p[-1] - p[-2])):
            nn = np.linalg.norm(d)
            outward[k] = d / nn if nn > 1e-9 else d

    coslim = np.cos(np.deg2rad(max_angle_deg))
    cands = []
    for ei, ej in cKDTree(pts).query_pairs(max_gap):
        if ei // 2 == ej // 2:
            continue
        # continuation: outward dirs opposed, bridge aligned with outward_i
        if np.dot(outward[ei], outward[ej]) > -coslim:
            continue
        bridge = pts[ej] - pts[ei]
        bn = np.linalg.norm(bridge)
        if bn > 0.5:
            bdir = bridge / bn
            if np.dot(outward[ei], bdir) < coslim or np.dot(-outward[ej], bdir) < coslim:
                continue
        cands.append((bn, ei, ej))
    cands.sort()

    # union polylines; comp_ends maps live original-endpoint id -> component id
    comp = {i: p for i, p in enumerate(polys)}
    comp_ends = {i: [2 * i, 2 * i + 1] for i in range(n)}  # [head_ep, tail_ep]
    where = {}
    for i in range(n):
        where[2 * i] = (i, 0)
        where[2 * i + 1] = (i, 1)

    for _, ei, ej in cands:
        if ei not in where or ej not in where:
            continue
        ci, si = where[ei]
        cj, sj = where[ej]
        if ci == cj:
            continue
        a = comp[ci] if si == 1 else comp[ci][::-1]   # join at a's tail
        b = comp[cj] if sj == 0 else comp[cj][::-1]   # join at b's head
        head_ep = comp_ends[ci][0] if si == 1 else comp_ends[ci][1]
        tail_ep = comp_ends[cj][1] if sj == 0 else comp_ends[cj][0]
        del where[ei], where[ej]
        del comp[cj], comp_ends[cj]
        comp[ci] = np.vstack([a, b])
        comp_ends[ci] = [head_ep, tail_ep]
        where[head_ep] = (ci, 0)
        where[tail_ep] = (ci, 1)
    return list(comp.values())


def fit_clean_shape(cnt, res, min_side=1.2, iou_accept=0.55):
    """Fit a clean convex quad/triangle to a contour; None if it is a sliver.

    Candidates: min-area rect and progressive convex-hull simplifications down
    to 4/3 vertices. The candidate with the best mask-IoU wins; if none reaches
    iou_accept, a 6-vertex simplified hull is returned.
    """
    area = cv2.contourArea(cnt)
    rect = cv2.minAreaRect(cnt)
    (rw, rh) = rect[1]
    if min(rw, rh) * res < min_side:      # sliver
        return None
    hull = cv2.convexHull(cnt)

    x, y, w, h = cv2.boundingRect(cnt)
    mask = np.zeros((h + 4, w + 4), np.uint8)
    cv2.drawContours(mask, [cnt - [x - 2, y - 2]], -1, 1, -1)

    def iou(poly):
        cand = np.zeros_like(mask)
        cv2.fillPoly(cand, [np.round(poly - [x - 2, y - 2]).astype(np.int32)], 1)
        inter = (cand & mask).sum()
        union = (cand | mask).sum()
        return inter / union if union else 0.0

    cands = [cv2.boxPoints(rect).astype(np.float64)]
    per = cv2.arcLength(hull, True)
    for frac in (0.02, 0.04, 0.06, 0.09, 0.13, 0.18):
        ap = cv2.approxPolyDP(hull, frac * per, True).reshape(-1, 2)
        if 3 <= len(ap) <= 4:
            cands.append(ap.astype(np.float64))
    best, best_iou = None, 0.0
    for cd in cands:
        i = iou(cd)
        if i > best_iou:
            best, best_iou = cd, i
    if best is not None and best_iou >= iou_accept:
        return best
    # fallback: gently simplified convex hull (max ~6 vertices)
    for frac in (0.01, 0.02, 0.03, 0.05, 0.08):
        ap = cv2.approxPolyDP(hull, frac * per, True).reshape(-1, 2)
        if len(ap) <= 6:
            return ap.astype(np.float64)
    return hull.reshape(-1, 2).astype(np.float64)


def extract_polygons(mask, res, x0, y0, close_k, min_area, eps_m, blur,
                     rect_fit=False):
    """Hole-filled, smoothed polygons from a class mask.

    rect_fit=True (crosswalk): merge nearby fragments, then fit clean convex
    quads/triangles per component (sliver fragments are dropped).
    """
    m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                         np.ones((close_k, close_k), np.uint8))
    if rect_fit:
        # merge fragments within ~0.6 m before shape fitting
        k = max(3, int(round(0.6 / res)) | 1)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    # fill holes: repaint external contours filled
    cnts = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    filled = np.zeros_like(m)
    cv2.drawContours(filled, cnts, -1, 1, -1)
    # boundary smoothing: gaussian blur + rethreshold
    if blur > 1:
        f = cv2.GaussianBlur(filled.astype(np.float32), (blur * 2 + 1, blur * 2 + 1), 0)
        filled = (f > 0.5).astype(np.uint8)
    cnts = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    polys = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area * res * res < min_area:
            continue
        if rect_fit:
            ap = fit_clean_shape(c, res)
            if ap is None:
                continue
        else:
            ap = cv2.approxPolyDP(c, eps_m / res, True).reshape(-1, 2).astype(np.float64)
        if len(ap) < 3:
            continue
        xy = np.stack([x0 + (ap[:, 0] + 0.5) * res, y0 + (ap[:, 1] + 0.5) * res], 1)
        polys.append(np.round(xy, 3))
    return polys


# ------------------------------------------------------------- main
def vectorize(scene_dir, min_len=1.0, gap=4.0):
    meta = json.load(open(os.path.join(scene_dir, "meta.json")))
    bev = np.load(os.path.join(scene_dir, "bev_label_masked.npy"))
    res = meta["resolution"]
    x0, y0 = meta["origin"]
    out = {"origin": [x0, y0], "resolution": res, "frame": "map",
           "scene": meta.get("scene", os.path.basename(scene_dir.rstrip("/"))),
           "classes": {}}

    # drivable region (carriageway incl. on-road markings) for BEV road edge
    drivable = np.isin(bev, (1, 3, 4, 5, 7)).astype(np.uint8)
    observed_nd = ((bev > 0) & (drivable == 0)).astype(np.uint8)

    for cname, (cid, ksz, cgap, cang, ceps, cminlen, cdeg) in LINE_CLASSES.items():
        if cname == "road_edge":
            # BEV road edge := drivable boundary adjacent to observed
            # non-drivable cells (image-annotation road_edge is NOT used;
            # boundaries against unobserved cells are skipped as unknown)
            dv = cv2.morphologyEx(drivable, cv2.MORPH_CLOSE,
                                  np.ones((7, 7), np.uint8))
            inner = dv & ~cv2.erode(dv, np.ones((3, 3), np.uint8))
            near_nd = cv2.dilate(observed_nd, np.ones((5, 5), np.uint8))
            m = (inner & near_nd).astype(np.uint8)
        else:
            m = (bev == cid).astype(np.uint8)
        if m.sum() < 20:
            out["classes"][cname] = []
            continue
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((ksz, ksz), np.uint8))
        # drop tiny specks before skeletonizing
        ncc, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        for i in range(1, ncc):
            if stats[i, cv2.CC_STAT_AREA] < 25:
                m[lab == i] = 0
        skel = prune_spurs(skeletonize(m.astype(bool)), max_len=int(1.2 / res))
        segs = trace_skeleton(skel)
        polys = [simplify(seg_to_xy(s, res, x0, y0), eps=ceps) for s in segs]
        polys = [p for p in polys if cv2.arcLength(
            p.astype(np.float32).reshape(-1, 1, 2), False) >= cminlen * 0.4]
        polys = link_segments(polys, max_gap=cgap, max_angle_deg=cang)
        polys = [smooth_polyline(p, deg=cdeg) for p in polys]
        polys = [p for p in polys if len(p) >= 2 and cv2.arcLength(
            p.astype(np.float32).reshape(-1, 1, 2), False) >= cminlen]
        out["classes"][cname] = [np.round(p, 3).tolist() for p in polys]

    # near-observation mask to purge far-range smear from shape-fitted classes
    min_dist = None
    cpath = os.path.join(scene_dir, "bev_counts.npz")
    if os.path.exists(cpath):
        try:
            min_dist = np.load(cpath)["min_dist"]
        except Exception:
            min_dist = None

    for cname, (cid, ck, marea, ceps, cblur, crect) in POLY_CLASSES.items():
        m = (bev == cid).astype(np.uint8)
        if crect and min_dist is not None and min_dist.shape == m.shape:
            m &= (min_dist <= 15.0).astype(np.uint8)
        polys = extract_polygons(m, res, x0, y0, ck, marea, ceps, cblur,
                                 rect_fit=crect) if m.sum() else []
        out["classes"][cname] = [p.tolist() for p in polys]
    return out


# ------------------------------------------------------------- nuScenes export
NUSC_LINE_LAYERS = {"laneline": "lane_divider", "road_edge": "road_edge",
                    "stopline": "stop_line"}
NUSC_POLY_LAYERS = {"crosswalk": "ped_crossing", "sidewalk": "walkway"}


def to_nuscenes(vec):
    """Convert to nuScenes map-expansion style tables (node/line/polygon + layers)."""
    scene = vec["scene"]
    nodes, lines, polygons = [], [], []
    layers = {v: [] for v in list(NUSC_LINE_LAYERS.values()) + list(NUSC_POLY_LAYERS.values())}

    def add_nodes(pts):
        toks = []
        for x, y in pts:
            t = f"{scene}.n{len(nodes):06d}"
            nodes.append({"token": t, "x": float(x), "y": float(y)})
            toks.append(t)
        return toks

    for cname, layer in NUSC_LINE_LAYERS.items():
        for p in vec["classes"].get(cname, []):
            lt = f"{scene}.l{len(lines):06d}"
            lines.append({"token": lt, "node_tokens": add_nodes(p)})
            layers[layer].append({"token": f"{scene}.{layer}{len(layers[layer]):05d}",
                                  "line_token": lt})
    for cname, layer in NUSC_POLY_LAYERS.items():
        for p in vec["classes"].get(cname, []):
            pt = f"{scene}.p{len(polygons):06d}"
            polygons.append({"token": pt, "exterior_node_tokens": add_nodes(p),
                             "holes": []})
            layers[layer].append({"token": f"{scene}.{layer}{len(layers[layer]):05d}",
                                  "polygon_token": pt})
    return {"version": "1.0", "scene": scene, "frame": "map",
            "origin": vec["origin"],
            "canvas_edge": None,
            "node": nodes, "line": lines, "polygon": polygons, **layers}


def visualize(scene_dir, vec, path):
    meta = json.load(open(os.path.join(scene_dir, "meta.json")))
    res = meta["resolution"]
    x0, y0 = meta["origin"]
    H, W = meta["size"]
    bev = np.load(os.path.join(scene_dir, "bev_label_masked.npy"))
    base = (PALETTE[bev][:, :, ::-1] * 0.35).astype(np.uint8)
    colors = {"laneline": (255, 255, 255), "road_edge": (0, 140, 255),
              "stopline": (40, 40, 255), "crosswalk": (200, 200, 0),
              "sidewalk": (200, 90, 190)}
    closed_cls = {"crosswalk", "sidewalk"}

    def to_px(p):
        return np.stack([(np.array(p)[:, 0] - x0) / res,
                         (np.array(p)[:, 1] - y0) / res], 1).astype(np.int32)

    for cname, polys in vec["classes"].items():
        closed = cname in closed_cls
        for p in polys:
            cv2.polylines(base, [to_px(p)], closed, colors[cname], 2)
            for q in to_px(p):
                cv2.circle(base, tuple(q), 3, colors[cname], -1)
    cv2.imwrite(path, np.ascontiguousarray(base[::-1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--gap", type=float, default=4.0, help="max dash gap to link [m]")
    ap.add_argument("--min-len", type=float, default=1.0)
    args = ap.parse_args()
    for d in args.dirs:
        try:
            vec = vectorize(d, min_len=args.min_len, gap=args.gap)
            json.dump(vec, open(os.path.join(d, "vector_map.json"), "w"))
            json.dump(to_nuscenes(vec),
                      open(os.path.join(d, "nuscenes_map.json"), "w"))
            visualize(d, vec, os.path.join(d, "vector_map.png"))
            stats = {k: len(v) for k, v in vec["classes"].items()}
            print(f"[ok] {d} {stats}", flush=True)
        except Exception as e:
            print(f"[fail] {d}: {e}", flush=True)


if __name__ == "__main__":
    main()

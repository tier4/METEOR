"""Lane-level route corridor from a Lanelet2 HD map (KITScenes), for the route / map conditioned METEOR2 variant.

A navigation route is a sequence of lanes, not the exact driven line, so the corridor is built from the lanelets
the scene's trajectory drives through:
  1. match the whole-scene 10 Hz poses (poses10.txt, map-local coordinates) to Lanelet2 lanelets with an HMM (Viterbi)
     emission   = 0.5 (d / 2 m)^2 + 0.5 (heading / 20 deg)^2 (within 8 m of the centreline, top 8 candidates,
                  sidewalks / crosswalks / parking etc. excluded)
     transition = same 0 / successor 0.25 / neighbour 1.0 / disconnected 25 (+ 0.5 per pose without candidates)
  2. the visited lanelet sequence (disconnected steps are bridged by the shortest path in the successor graph;
     otherwise counted as unresolved)
  3. quality gate: matched >= 0.8, distance p95 <= 5 m, heading error p95 <= 45 deg, unresolved = 0; otherwise the
     route is invalid (empty raster)
The corridor is each lanelet's centreline (mean of the left / right bounds) drawn 3.5 m wide, nothing more than 10 m
behind the ego; the destination is the last pose of the scene (radius 2 m); see dataset.ae2e_inputs.
The lanelet2 library is not needed: map.osm is parsed directly (same projection as tools/kitscenes/kitscenes_to_bevlane.py)."""
import os, re, json, math
from collections import deque
import numpy as np

BLOCKED = {"crosswalk", "walkway", "stairs", "bicycle_lane", "parking"}
R_CAND, K_CAND, D_SIG, H_SIG = 8.0, 8, 2.0, math.radians(20.0)
C_SAME, C_FOLLOW, C_ADJ, C_DISC, C_SKIP = 0.0, 0.25, 1.0, 25.0, 0.5


def _resample(p, n):
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    if s[-1] <= 0:
        return np.repeat(p[:1], n, 0)
    t = np.linspace(0, s[-1], n)
    return np.stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])], 1)


def load_lanelets(scene_dir):
    """-> list of dict(id, left, right (way ids), cl [N,2] centreline (in driving direction), l0/l1/r0/r1 end nodes)"""
    import pyproj
    o = json.load(open(f"{scene_dir}/maps/origin.json"))
    P = pyproj.Proj(proj="utm", zone=32, ellps="WGS84"); ox, oy = P(o["longitude"], o["latitude"])
    txt = open(f"{scene_dir}/maps/map.osm").read()
    nid, lat, lon = [], [], []
    for m in re.finditer(r'<node id="(-?\d+)"[^>]*lat="([-\d.]+)" lon="([-\d.]+)"', txt):
        nid.append(int(m.group(1))); lat.append(float(m.group(2))); lon.append(float(m.group(3)))
    xs, ys = P(np.array(lon), np.array(lat)); XY = dict(zip(nid, np.stack([xs - ox, ys - oy], 1)))
    ways = {}
    for m in re.finditer(r'<way id="(-?\d+)"[^>]*>(.*?)</way>', txt, re.S):
        ways[int(m.group(1))] = [int(r) for r in re.findall(r'<nd ref="(-?\d+)"', m.group(2)) if int(r) in XY]
    out = []
    for m in re.finditer(r'<relation id="(-?\d+)"[^>]*>(.*?)</relation>', txt, re.S):
        body = m.group(2); tags = dict(re.findall(r'<tag k="([^"]+)" v="([^"]*)"', body))
        if tags.get("type") != "lanelet" or tags.get("subtype", "road") in BLOCKED:
            continue
        mem = dict((role, int(ref)) for ref, role in re.findall(r'<member type="way" ref="(-?\d+)" role="(left|right)"', body))
        if "left" not in mem or "right" not in mem or len(ways.get(mem["left"], [])) < 2 or len(ways.get(mem["right"], [])) < 2:
            continue
        L, R = list(ways[mem["left"]]), list(ways[mem["right"]])
        # orient the left and right bounds the same way, as Lanelet2 does (the left bound defines the direction)
        pl, pr = np.array([XY[i] for i in L]), np.array([XY[i] for i in R])
        if np.linalg.norm(pr[0] - pl[0]) + np.linalg.norm(pr[-1] - pl[-1]) > np.linalg.norm(pr[0] - pl[-1]) + np.linalg.norm(pr[-1] - pl[0]):
            R = R[::-1]; pr = pr[::-1]
        n = max(len(pl), len(pr), 2)
        cl = 0.5 * (_resample(pl, n) + _resample(pr, n))
        # decide the direction geometrically: the left bound must lie left of the driving direction (independent of OSM way order)
        v = cl[-1] - cl[0]; lat = pl.mean(0) - pr.mean(0)
        if v[0] * lat[1] - v[1] * lat[0] < 0:
            cl = cl[::-1]; L, R = L[::-1], R[::-1]
        d = dict(id=int(m.group(1)), left=mem["left"], right=mem["right"], cl=cl, l0=L[0], l1=L[-1], r0=R[0], r1=R[-1])
        out.append(d)
        if tags.get("one_way", "yes") in ("no", "0"):                    # bidirectional lanelet: the reverse direction is a candidate too (Lanelet2 inverted lanelet)
            out.append(dict(d, id=-d["id"], cl=cl[::-1].copy(), l0=R[-1], l1=R[0], r0=L[-1], r1=L[0]))
    return out


def _nearest(p, cl):
    a, b = cl[:-1], cl[1:]; v = b - a; w = p - a
    t = np.clip(np.einsum("ij,ij->i", w, v) / np.maximum(np.einsum("ij,ij->i", v, v), 1e-9), 0, 1)
    q = a + t[:, None] * v; d = np.linalg.norm(q - p, axis=1); k = int(d.argmin())
    return float(d[k]), math.atan2(v[k, 1], v[k, 0])


def _parallel(a, b, lat_max=4.5, ang_max=math.radians(30)):
    """neighbouring lane (also parallel lanelets that share no bound): some point of a within 4.5 m of b and heading difference <= 30 deg"""
    for p in a[:: max(1, len(a) // 5)]:
        d, h = _nearest(p, b)
        if d <= lat_max:
            k = int(np.argmin(np.linalg.norm(a - p, axis=1))); k = min(k, len(a) - 2)
            ha = math.atan2(*(a[k + 1] - a[k])[::-1])
            if abs(_wrap(ha - h)) <= ang_max:
                return True
    return False


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def match_route(lls, pos, yaw):
    """→ dict(valid, segs [list of [N,2]], quality)"""
    n = len(lls)
    if n == 0:
        return dict(valid=False, segs=[], quality=dict(reason="no_map"))
    fol = [set() for _ in range(n)]; adj = [set() for _ in range(n)]
    for i, a in enumerate(lls):
        for j, b in enumerate(lls):
            if i == j: continue
            if abs(a["id"]) == abs(b["id"]): continue
            if (b["l0"] == a["l1"] and b["r0"] == a["r1"]) or np.linalg.norm(b["cl"][0] - a["cl"][-1]) < 1.0:
                fol[i].add(j)
            elif b["left"] == a["right"] or b["right"] == a["left"] or _parallel(a["cl"], b["cl"]):
                adj[i].add(j)
    ctr = np.array([c["cl"].mean(0) for c in lls]); rad = np.array([np.linalg.norm(c["cl"] - c["cl"].mean(0), axis=1).max() for c in lls])
    rows = []
    for t in range(len(pos)):
        near = np.where(np.linalg.norm(ctr - pos[t], axis=1) <= rad + R_CAND)[0]
        cs = []
        for j in near:
            d, h = _nearest(pos[t], lls[j]["cl"])
            if d <= R_CAND:
                he = abs(_wrap(yaw[t] - h)); cs.append((0.5 * (d / D_SIG) ** 2 + 0.5 * (he / H_SIG) ** 2, int(j), d, he))
        cs.sort()
        if cs: rows.append((t, cs[:K_CAND]))
    if not rows:
        return dict(valid=False, segs=[], quality=dict(reason="no_candidates"))

    def trans(i, j, skip):
        c = C_SKIP * skip
        return c + (C_SAME if i == j else C_FOLLOW if j in fol[i] else C_ADJ if j in adj[i] else C_DISC)
    cost = [c[0] for c in rows[0][1]]; back = []
    for r in range(1, len(rows)):
        skip = rows[r][0] - rows[r - 1][0] - 1; prev = rows[r - 1][1]; nc, nb = [], []
        for e, j, _, _ in rows[r][1]:
            k = min(range(len(prev)), key=lambda q: cost[q] + trans(prev[q][1], j, skip))
            nc.append(cost[k] + trans(prev[k][1], j, skip) + e); nb.append(k)
        cost = nc; back.append(nb)
    k = int(np.argmin(cost)); sel = [rows[-1][1][k]]
    for r in range(len(rows) - 1, 0, -1):
        k = back[r - 1][k]; sel.append(rows[r - 1][1][k])
    sel = sel[::-1]
    seq = [sel[0][1]]; unresolved = 0
    for c in sel[1:]:
        j = c[1]
        if j == seq[-1]: continue
        i = seq[-1]
        if j in fol[i] or j in adj[i]:
            seq.append(j); continue
        path = _bfs([f | g for f, g in zip(fol, adj)], i, j, 6)
        if path is None: unresolved += 1; seq.append(j)
        else: seq.extend(path[1:])
    d = np.array([c[2] for c in sel]); he = np.array([c[3] for c in sel])
    q = dict(matched=len(sel) / len(pos), p95_d=float(np.quantile(d, 0.95)), p95_h=float(np.quantile(he, 0.95)), unresolved=unresolved, n_lanes=len(seq))
    valid = q["matched"] >= 0.8 and q["p95_d"] <= 5.0 and q["p95_h"] <= math.radians(45) and unresolved == 0
    return dict(valid=valid, segs=[lls[j]["cl"] for j in seq], quality=q)


def _bfs(fol, a, b, depth):
    Q = deque([[a]]); seen = {a}
    while Q:
        p = Q.popleft()
        if len(p) > depth: continue
        for j in fol[p[-1]]:
            if j == b: return p + [j]
            if j not in seen: seen.add(j); Q.append(p + [j])
    return None


def scene_route(scene_dir, cache_name="route_lanelet.npz"):
    """Route of a scene, cached in scene_dir/route_lanelet.npz: pts [M,2] map-local, off [S+1] lanelet boundaries in pts, dest [2], valid"""
    f = os.path.join(scene_dir, cache_name)
    if os.path.exists(f):
        z = np.load(f); return dict(pts=z["pts"], off=z["off"], dest=z["dest"], valid=bool(z["valid"]))
    P = np.loadtxt(os.path.join(scene_dir, "poses10.txt"), ndmin=2)
    pos = P[:, 1:3]; qx, qy, qz, qw = P[:, 4], P[:, 5], P[:, 6], P[:, 7]
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy ** 2 + qz ** 2))
    try:
        r = match_route(load_lanelets(scene_dir), pos, yaw)
    except FileNotFoundError:
        r = dict(valid=False, segs=[], quality=dict(reason="no_map"))
    segs = r["segs"]; off = np.r_[0, np.cumsum([len(s) for s in segs])].astype(np.int32)
    pts = np.concatenate(segs, 0) if segs else np.zeros((0, 2))
    np.savez(f + ".tmp.npz", pts=pts, off=off, dest=pos[-1], valid=np.array(r["valid"]), quality=json.dumps(r["quality"]))
    os.replace(f + ".tmp.npz", f)
    return dict(pts=pts, off=off, dest=pos[-1], valid=r["valid"])

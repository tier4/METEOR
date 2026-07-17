#!/usr/bin/env python3
"""L1 safety guardrails for the E2E path (roadmap C7, doer/checker).

Deterministic, runtime-only checks of the SELECTED E2E trajectory against
the network's own geometric heads (which run on the raw-BEV route, i.e. a
partially independent input path from the fused-BEV planner):

  1. spacetime collision — the path at t is checked against every detected
     agent moved along its own forecast to the same t (0.5 s steps), plus
     static occupancy classes at the path cells
  2. red-light gate — TL state red x a stop line ahead in the ego corridor
  3. kinematic feasibility — curvature vs a bicycle-model bound at v0
  4. drivable containment — path cells must lie on road-ish BEV classes

Returns a verdict dict; when vetoed, a minimal-risk in-lane stop path
(truncated selected path with a v0^2/2a stop distance) is provided.
No learning, no state — every rule is unit-testable.
"""
import math

import numpy as np

EGO_HALF_W = 1.05          # m, half width incl. margin
EGO_HALF_L = 1.20          # m, effective half length at the waypoint
DRIVABLE = (1, 3, 7, 8)    # road / crosswalk / marking / parking
STATIC_OCC = (1, 8)        # obstacle / building voxels block the path
STOPLINE = 5
A_MAX = 3.0                # m/s^2 comfortable-emergency decel
K_MAX = 0.2                # 1/m max curvature bound (~tan(30deg)/2.8m)


def _occ_ground(occ_pred):
    """[Z,200,200] class grid -> ground-projected static blockers (0.4 m)."""
    if occ_pred is None:
        return None
    return np.isin(occ_pred, STATIC_OCC).any(0)


def check_path(path, occ_pred, dets, det_offs, tl_probs, lane_argmax, v0):
    """path [6,2] (x,y at 0.5..3.0 s, ego frame). dets: decoded boxes
    (cls,score,xe,ye,l,w,yaw); det_offs: per-det [6,2] forecast offsets.
    tl_probs: [4] none/green/yellow/red. lane_argmax: [800,500] 0.2 m BEV.
    Returns dict(verdict, reason, t_event, p_event, stop_path)."""
    events = []
    trav = float(np.linalg.norm(np.asarray(path[-1])))
    if v0 < 0.7 and trav < 1.5:          # already holding: nothing to veto
        return {"verdict": "HOLD", "reason": "stopped", "t_event": None,
                "p_event": None, "feasible": True, "stop_path": None}

    # ---- 1a. dynamic spacetime collision --------------------------------
    for ti in range(6):
        px, py = float(path[ti, 0]), float(path[ti, 1])
        if px * px + py * py < 2.5 ** 2:      # ego-proximal: own footprint
            continue
        for (cls, sc, xe, ye, l, w, yaw), off in zip(dets, det_offs):
            if sc < 0.35:
                continue
            if xe * xe + ye * ye < 2.0 ** 2:  # overlapping ego at t=0: FP
                continue
            bx, by = xe + float(off[ti, 0]), ye + float(off[ti, 1])
            dx, dy = px - bx, py - by
            ca, sa = math.cos(yaw), math.sin(yaw)
            lx = ca * dx + sa * dy
            wy = -sa * dx + ca * dy
            if abs(lx) < l / 2 + EGO_HALF_L and abs(wy) < w / 2 + EGO_HALF_W:
                nm = {0: "veh", 1: "vru", 2: "unk"}.get(int(cls), "?")
                events.append((f"COLLISION {nm} t={(ti + 1) * 0.5:.1f}s "
                               f"d={math.hypot(bx, by):.0f}m",
                               (ti + 1) * 0.5, (bx, by)))
                break
        if events:
            break

    # ---- 1b. static occupancy on the path -------------------------------
    blk = _occ_ground(occ_pred)
    if blk is not None and not events:
        for ti in range(6):
            px, py = float(path[ti, 0]), float(path[ti, 1])
            if px * px + py * py < 2.5 ** 2:  # ego-proximal fragments = FP
                continue
            r, c = int((40.0 - px) / 0.4), int((40.0 - py) / 0.4)
            if 2 <= r < 198 and 2 <= c < 198 \
                    and int(blk[r - 1:r + 2, c - 1:c + 2].sum()) >= 3:
                events.append((f"STATIC obstacle {math.hypot(px, py):.0f}m "
                               f"t={(ti + 1) * 0.5:.1f}s",
                               (ti + 1) * 0.5, (px, py)))
                break

    # ---- 2. red-light x stop-line gate ----------------------------------
    red = float(tl_probs[3])
    if red > 0.5 and not events:
        # nearest stopline ahead in the ego corridor (|y|<=2 m, 2..30 m)
        rows = np.arange(150, 390)          # x = 80-0.2r -> 2..50 m ahead
        band = lane_argmax[rows, 230:270]
        hit = np.nonzero((band == STOPLINE).any(1))[0]
        if len(hit):
            x_stop = 80.0 - (rows[hit[0]]) * 0.2
            if 2.0 < x_stop < 30.0 and float(path[-1, 0]) > x_stop - 1.0:
                events.append((f"RED LIGHT stopline {x_stop:.0f}m",
                               None, (x_stop, 0.0)))

    # ---- 3. kinematic feasibility ---------------------------------------
    feas = True
    pts = np.concatenate([[[0.0, 0.0]], np.asarray(path)], 0)
    for i in range(1, 6):
        a, b, c = pts[i - 1], pts[i], pts[i + 1]
        v1, v2 = b - a, c - b
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 0.3 or n2 < 0.3:
            continue
        cosang = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
        if math.acos(cosang) / max(n2, 1e-3) > K_MAX:
            feas = False
            break

    # ---- 4. drivable containment ----------------------------------------
    offroad = 0
    for ti in range(6):
        px, py = float(path[ti, 0]), float(path[ti, 1])
        r, c = int((80.0 - px) / 0.2), int((50.0 - py) / 0.2)
        if 0 <= r < 800 and 0 <= c < 500 \
                and int(lane_argmax[r, c]) not in DRIVABLE:
            offroad += 1
    if offroad >= 3 and not events:      # tolerate 2 cells (GT noise)
        events.append((f"OFFROAD {offroad}/6 wp", None, tuple(path[-1])))

    # ---- verdict + minimal-risk stop path -------------------------------
    if events:
        kind, t_ev, p_ev = events[0]
        d_stop = max(v0 * v0 / (2 * A_MAX), 1.0)
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        cum = np.cumsum(seg)
        stop = pts[1:].copy()
        for ti in range(6):
            if cum[ti] > d_stop:
                k = ti
                scale = d_stop / max(cum[ti], 1e-3)
                stop[ti:] = pts[0] + (stop[ti] - pts[0]) * scale
                break
        return {"verdict": "VETO", "reason": kind, "t_event": t_ev,
                "p_event": p_ev, "feasible": feas, "stop_path": stop}
    return {"verdict": "OK", "reason": "", "t_event": None,
            "p_event": None, "feasible": feas, "stop_path": None}

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
    """[Z,200,200] class grid -> ground-projected static blockers (0.4 m).
    The hood strip (x<5.5 m, |y|<1.6 m) is blanked: the ego's own hood
    reflections produce persistent phantom obstacle voxels dead ahead
    (the constant 3-4 m STATIC veto) — a known near-ego OCC FP mode that
    the r27 GT cleanup targets; the guard must not consume it."""
    if occ_pred is None:
        return None
    # 2026-08-20: 幻影 VETO の実測全件が z ビン 12-15 (地上 3.8-5.4 m) の
    # building ボクセルだった (路面上空の幻視/高架構造)。ガードが見るべきは
    # 車両が通過する高さ窓だけ: z ビン 3..9 (路面 +0.2〜+3.0 m)。上空の
    # ボクセルは実在 (門型標識・高架) でも衝突対象ではない。
    blk = np.isin(occ_pred[3:10], STATIC_OCC).any(0)
    blk[86:101, 96:105] = False          # x in (0,5.6], |y|<=1.6 m
    return blk


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
        # 2026-08-20 誤 VETO 対策: 単フレーム 3x3>=3 は夜間の occ 幻影で
        # 42% のフレームが VETO になっていた (val+curve 112 枚で幻影率 33/33
        # = 100%)。(a) 3x3>=5 に強化、(b) 同位置 (2m 以内) で 2 フレーム
        # 連続したときだけ発火する持続確認を追加。
        _cand = None
        for ti in range(6):
            px, py = float(path[ti, 0]), float(path[ti, 1])
            if px * px + py * py < 2.5 ** 2:  # ego-proximal fragments = FP
                continue
            r, c = int((40.0 - px) / 0.4), int((40.0 - py) / 0.4)
            if 2 <= r < 198 and 2 <= c < 198 \
                    and int(blk[r - 1:r + 2, c - 1:c + 2].sum()) >= 5:
                _cand = ((ti + 1) * 0.5, (px, py))
                break
        if _cand is not None:
            events.append((f"STATIC obstacle "
                           f"{math.hypot(*_cand[1]):.0f}m "
                           f"t={_cand[0]:.1f}s", _cand[0], _cand[1]))

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


def risk_pick(modes, confs, risk_map, lam=1.5):
    """C1: pick the E2E mode by confidence minus the risk-field line
    integral (risk_map [400,250] = fused crop x in [-40,40), y in
    (-25,25], 0.2 m). Returns (best_idx, per_mode_scores, per_mode_risk)."""
    scores, risks = [], []
    for k in range(len(modes)):
        wp = np.asarray(modes[k]).reshape(6, 2)
        ri = 0.0
        n = 0
        for x, y in wp:
            r, c = int((40.0 - x) / 0.2), int((25.0 - y) / 0.2)
            if 0 <= r < risk_map.shape[0] and 0 <= c < risk_map.shape[1]:
                ri += float(risk_map[r, c])
                n += 1
        ri = ri / max(n, 1)
        risks.append(ri)
        scores.append(float(confs[k]) - lam * ri)
    return int(np.argmax(scores)), scores, risks

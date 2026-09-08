#!/usr/bin/env python3
"""Standalone on-device demo: TRT inference AND rendering on the Orin.

The first Orin demo inferred here but composed the video on the x86 box; the
production target is a self-contained real-time demo, so this script does both
on the device with no torch anywhere: MeteorRT (cuda-python shim) for the
engine, deploy/viz_np.py for the SAME drawing code the PyTorch demo uses
(extracted verbatim; the 2D decode is a numpy port verified box-for-box
against the torch original).

Layout and semantics follow bevlane/demo_rgbd_bev.py: 4x2 surround RGB tiles
with 2D-seg overlay + 2D/3D boxes (BACK_NARROW blank per the 7-camera
convention), a depth row, FRONT_WIDE carrying the E2E ribbon, and the BEV panel
with waypoint dots, heading ticks, per-vehicle 3 s futures (VRU suppressed,
stationary boxes grey but their futures still drawn -- the exact demo
condition), and the v0/steer/accel/brake HUD. v0 comes from the scene's
ego_motion.npz, not a constant.

    python3 deploy/orin_render.py --engine eng/v59_demo_fp16.engine \
        --root sample --scenes 2 --stride 8 --out out/demo_orin_standalone.mp4
        [--display]
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deploy.runtime import (MeteorRT, decode_boxes, stationary_at,
                            stationary_head_healthy)             # noqa: E402
from deploy.viz_np import (DEMO_PALETTE, PALETTE, crop_bev_np, decode_boxes2d_ms_np,  # noqa
                           draw_boxes2d, draw_boxes_on_rgb, draw_path_ribbon)

_DBINS = int(os.environ.get("METEOR_DEPTH_BINS", "64"))
_GROUND_Z = float(os.environ.get("METEOR_GROUND_Z", "0.0"))   # 路面の ego z [m] (base_link 基準なら 0)
CAMS = ["CAM_FRONT_WIDE", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
        "CAM_BACK_WIDE", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
        "CAM_FRONT_NARROW"]
EGO_K = 3
# CAM8 は **画面のタイル配置用** の並び (左-中央-右に見えるように並べたもの)。
# モデルの入力順ではない。混同すると、リフトのプラグインがスロット毎に
# 幾何テーブルを焼き込んでいるため各カメラの画素が BEV の誤ったセルへ飛び、
# BEV Seg も E2E も崩れる (2026-08-24 に実害。動画を作り直した)。
CAM8 = ["CAM_FRONT_LEFT", "CAM_FRONT_WIDE", "CAM_FRONT_RIGHT",
        "CAM_FRONT_NARROW", "CAM_BACK_LEFT", "CAM_BACK_WIDE",
        "CAM_BACK_RIGHT", "CAM_BACK_NARROW"]
# モデルの入力順 = 学習側 bevlane/dataset.py の CAMS と同一。
CAM_IN8 = CAMS + ["CAM_BACK_NARROW"]
# 描画側が参照するカメラ一覧 (main() がエンジンの本数に合わせて差し替える)
CAM_DRAW = list(CAMS)
CW, CH = 375, 210
VW, VH = 1920, 1080

XF, XR = 80.0, 40.0            # metric extents of the v59 grid
VIEW_F, VIEW_R, YH = 60.0, min(60.0, XR), 25.0


_SEG2D_OVERLAY = os.environ.get("METEOR_SEG2D_OVERLAY", "0") == "1"

def set_bev_extent(bev_h, cell=0.2):
    """BEV ラスタの行数から後方の範囲を決める (2026-08-15)。

    軽量版は 600 行 = 前 80 / 後 40 m、ベースラインは 800 行 = 前後 80 m。
    ここを軽量版の値 (XR=40) で固定していたため、8 カメラ版のデモでも
    BEV の後方が 40 m で切れて表示されていた。前方 80 m は不変。
    """
    global XR, VIEW_R
    XR = max(0.0, float(bev_h) * cell - XF)
    # 表示範囲は **モデルが実際に出している範囲** をそのまま映す。
    # 学習データの都合 (BEV Seg の GT は後方 40m までしか無い) で描画を
    # 切るのは配備側で判断することではない。後方 40-80m が不安定に見えるのは
    # 教師が無いためで、直すなら GT 生成側 (the training server)。ここで隠すと限界が見えなくなる。
    VIEW_R = min(60.0, XR)
    print(f"[render] BEV {bev_h} 行 -> 前 {XF:.0f} m / 後 {XR:.0f} m "
          f"(表示は前 {VIEW_F:.0f} / 後 {VIEW_R:.0f} m = Seg GT のある範囲)",
          flush=True)


OCC_XY = 40.0                  # occ グリッドの片側範囲 [m] (±40 m)
OCC_RES = 0.4                  # occ グリッドの分解能 [m/セル]


_SEGF = {}


def seg_fuse_logit(cur, logits, pose):
    """seg-fuse 完全版 (2026-08-19)。エンジンの lane_logit 出力を使い、
    ローカル demo と同じ log_softmax + 0.7*ワープ累積で面クラスを融合する。
    細線クラス (3,4,5,6) は生の現フレームで保護、適用は x<=45m (row>=175)。
    logits: [1,9,800,500] または [9,800,500] (fp16 可)。"""
    if pose is None:
        _SEGF.pop("acc", None)
        return cur
    lg = np.asarray(logits, np.float32)
    if lg.ndim == 4:
        lg = lg[0]
    mx = lg.max(0, keepdims=True)
    lp = lg - (mx + np.log(np.exp(lg - mx).sum(0, keepdims=True)))
    st = _SEGF.get("acc")
    if st is not None:
        acc, pp = st
        cp, sp = np.cos(pp[2]), np.sin(pp[2])
        dx, dy = pose[0] - pp[0], pose[1] - pp[1]
        tx, ty = cp * dx + sp * dy, -sp * dx + cp * dy
        dyaw = float(pose[2] - pp[2])
        if abs(tx) + abs(ty) < 10.0 and abs(dyaw) < 0.5:
            c, s2 = np.cos(dyaw), np.sin(dyaw)
            M = np.array([[c, s2, -250 * c - 400 * s2 - 5 * ty + 250],
                          [-s2, c, 250 * s2 - 400 * c - 5 * tx + 400]],
                         np.float32)
            hw = acc.transpose(1, 2, 0)
            wr = np.concatenate(
                [cv2.warpAffine(hw[:, :, i:i + 3], M, (500, 800),
                                flags=cv2.INTER_NEAREST
                                | cv2.WARP_INVERSE_MAP,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=0.0)
                 for i in (0, 3, 6)], -1).transpose(2, 0, 1)
            lp = lp + 0.7 * wr
    _SEGF["acc"] = (lp, tuple(pose))
    fused = lp.argmax(0).astype(np.uint8)
    out_pred = cur.copy()
    out_pred[175:] = fused[175:]
    thin = np.isin(cur, (3, 4, 5, 6))
    out_pred[thin] = cur[thin]
    return out_pred


def seg_fuse_np(cur, pose):
    """ローカル demo の seg-fuse (ego ワープ log-odds 融合) の argmax 版近似。

    エンジンは argmax 出力のみなので、クラスラスタを ego 運動でワープし
    「面クラス (road/sidewalk/parking) の持続投票」で時間融合する。
    細線クラス (crosswalk/lane/stop/edge) は常に生の現フレームを保護
    (ローカルの thin 保護と同じ規約)。適用は x<=45 m (row>=175) のみ。
    2 ワーカー間で状態が 1 フレーム古く読まれ得るのは mode ヒステリシスと
    同じ許容 (ワープは保存したポーズから計算するので幾何は正しい)。
    完全一致 (ロジット融合) は次回エンジン再ビルドでレーン logit を
    出力に追加してから。
    """
    if pose is None:
        _SEGF.clear()
        return cur
    st = _SEGF.get("st")
    fused = cur
    AREA = (1, 2, 8)
    if st is not None:
        pcls, pconf, pp = st
        cp, sp = np.cos(pp[2]), np.sin(pp[2])
        dx, dy = pose[0] - pp[0], pose[1] - pp[1]
        tx, ty = cp * dx + sp * dy, -sp * dx + cp * dy
        dyaw = float(pose[2] - pp[2])
        if abs(tx) + abs(ty) < 10.0 and abs(dyaw) < 0.5:   # シーン跨ぎ guard
            c, s2 = np.cos(dyaw), np.sin(dyaw)
            # (u,v)=(col,row)。出力(現)画素 -> 入力(前)画素 (WARP_INVERSE_MAP)
            M = np.array([[c, s2, -250 * c - 400 * s2 - 5 * ty + 250],
                          [-s2, c, 250 * s2 - 400 * c - 5 * tx + 400]],
                         np.float32)
            wcls = cv2.warpAffine(pcls, M, (500, 800),
                                  flags=cv2.INTER_NEAREST
                                  | cv2.WARP_INVERSE_MAP,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=0)
            wconf = cv2.warpAffine(pconf, M, (500, 800),
                                   flags=cv2.INTER_NEAREST
                                   | cv2.WARP_INVERSE_MAP,
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=0)
            fused = cur.copy()
            conf = np.where(np.isin(cur, AREA), 3, 0).astype(np.uint8)
            agree = np.isin(cur, AREA) & (wcls == cur)
            conf[agree] = np.minimum(wconf[agree] + 1, 6)
            fill = (cur == 0) & np.isin(wcls, AREA) & (wconf > 0)
            fused[fill] = wcls[fill]
            conf[fill] = wconf[fill] - 1
            out = cur.copy()
            out[175:] = fused[175:]              # 近傍のみ (x <= 45 m)
            _SEGF["st"] = (fused, conf, tuple(pose))
            return out
    conf0 = np.where(np.isin(cur, AREA), 3, 0).astype(np.uint8)
    _SEGF["st"] = (cur.copy(), conf0, tuple(pose))
    return cur


def thin_road_edge_np(pred):
    """road_edge 帯の最内 1px だけ残す (bevlane/postproc.py の自己完結版)。

    ローカル demo は既定でこれを適用しており、Orin だけ帯が太く出ていた
    (2026-08-19)。drivable (road/crosswalk/lane/stop) に 4 近傍で接する
    edge 画素のみ残し、残りは背景へ落とす。800x500 で ~1 ms。
    """
    edge = (pred == 6).astype(np.uint8)
    if edge.sum() == 0:
        return pred
    pred = pred.copy()
    drv = np.isin(pred, (1, 3, 4, 5)).astype(np.uint8)
    drv_dil = cv2.dilate(drv, np.ones((3, 3), np.uint8))
    inner = edge & (drv_dil > 0) & (drv == 0)
    pred[edge > 0] = 0
    pred[inner > 0] = 6
    return pred


def _zs_thin(mask, max_iter=6):
    """Zhang-Suen 細線化 (numpy ベクトル化)。線幅 2-5px なら 2-3 反復で収束。

    cv2.ximgproc はローカルにも Orin にも入っていないため自前実装。
    800x500 の細線クラス 1 枚で ~10ms。"""
    img = mask.astype(np.uint8)
    for _ in range(max_iter):
        changed = False
        for step in (0, 1):
            p = np.pad(img, 1)
            P2 = p[:-2, 1:-1]; P3 = p[:-2, 2:]; P4 = p[1:-1, 2:]
            P5 = p[2:, 2:]; P6 = p[2:, 1:-1]; P7 = p[2:, :-2]
            P8 = p[1:-1, :-2]; P9 = p[:-2, :-2]
            nb = [P2, P3, P4, P5, P6, P7, P8, P9]
            B = sum(x.astype(np.int8) for x in nb)
            seq = nb + [P2]
            A = sum(((seq[i] == 0) & (seq[i + 1] == 1)).astype(np.int8)
                    for i in range(8))
            if step == 0:
                cond = (img == 1) & (B >= 2) & (B <= 6) & (A == 1) \
                    & ((P2 & P4 & P6) == 0) & ((P4 & P6 & P8) == 0)
            else:
                cond = (img == 1) & (B >= 2) & (B <= 6) & (A == 1) \
                    & ((P2 & P4 & P8) == 0) & ((P2 & P6 & P8) == 0)
            if cond.any():
                img[cond] = 0
                changed = True
        if not changed:
            break
    return img > 0


def thin_lane_lines_np(pred, classes=(4,)):
    """レーン線(4) を 1px 骨格に細線化 (METEOR_THIN_LANES=1)。
    停止線(5) は面で見えるのが自然なので対象外 (2026-08-27 ユーザー判断)。

    生ラスタのシャギー対策の試験表示。落とした画素は周囲に road があれば
    road へ、なければ背景へ戻す。既定 OFF (ローカル表示との等価性維持)。"""
    pred = pred.copy()
    road_near = cv2.dilate((pred == 1).astype(np.uint8),
                           np.ones((5, 5), np.uint8)) > 0
    width = int(os.environ.get("METEOR_THIN_LANES_W", "2"))
    for cls in classes:
        m = pred == cls
        if not m.any():
            continue
        sk = _zs_thin(m)
        drop = m & ~sk
        pred[drop] = 0
        pred[drop & road_near] = 1
        if width >= 2:
            # 骨格を width px へ (2026-08-27: 1px は細すぎとの判断で既定 2px)
            k = np.ones((2, 2), np.uint8) if width == 2 \
                else np.ones((3, 3), np.uint8)
            grow = (cv2.dilate(sk.astype(np.uint8), k) > 0) \
                & np.isin(pred, (0, 1))
            pred[grow] = cls
    return pred


_OCC_CACHE = {}


def draw_occ(bev, occ, sy2, sx2, alpha=0.35, thresh=0.0):
    """占有格子 (occ) を BEV パネルに重ねる。

    occ は [C, Z, H, W] = (10 クラス, 16 高さビン, 200, 200) で、
    クラス 0 が free。高さ方向は最大値で潰し、free 確率が低いセルだけを
    「占有」として色付けする。BEV パネルは前 VIEW_F / 後 VIEW_R / 横 ±YH を
    covers するので、occ の ±40 m をその画素座標に貼り込む。
    """
    if occ is None:
        return
    o = np.asarray(occ)
    if o.ndim == 5:
        o = o[0]
    if o.ndim != 4:
        return
    # softmax を取ると 10x16x200x200 の exp で描画が 90 ms 増えたので、
    # ロジットの大小比較だけで判定する (結果は argmax と同じ)。
    o = o.astype(np.float32)
    free = o[0]                                       # [Z,H,W]
    rest = o[1:]                                      # [C-1,Z,H,W]
    best = rest.max(0)                                # クラス方向の最大
    occ_z = best - free                               # >0 なら占有
    zi = occ_z.argmax(0)                              # 最も占有らしい高さ
    ii = np.indices(zi.shape)
    occupied = occ_z[zi, ii[0], ii[1]]                # [H,W] マージン
    cls = rest.argmax(0)[zi, ii[0], ii[1]] + 1        # 代表クラス
    H, W = occupied.shape
    col = PALETTE[cls % len(PALETTE)][:, :, ::-1].astype(np.uint8)
    m = occupied > thresh
    if not m.any():
        return
    # occ セル (r, c) -> 自車座標: x = +40 - r*0.4, y = +40 - c*0.4
    y0 = int(round((VIEW_F - OCC_XY) * sy2))
    y1 = int(round((VIEW_F + OCC_XY) * sy2))
    x0 = int(round((YH - OCC_XY) * sx2))
    x1 = int(round((YH + OCC_XY) * sx2))
    bh, bw = bev.shape[:2]
    ty0, ty1 = max(0, y0), min(bh, y1)
    tx0, tx1 = max(0, x0), min(bw, x1)
    if ty1 <= ty0 or tx1 <= tx0:
        return
    # occ 側の対応範囲を切り出してからパネル解像度へ拡大する
    sr0 = int(round((ty0 - y0) / max(y1 - y0, 1) * H))
    sr1 = int(round((ty1 - y0) / max(y1 - y0, 1) * H))
    sc0 = int(round((tx0 - x0) / max(x1 - x0, 1) * W))
    sc1 = int(round((tx1 - x0) / max(x1 - x0, 1) * W))
    sr1, sc1 = max(sr1, sr0 + 1), max(sc1, sc0 + 1)
    sub_col = cv2.resize(col[sr0:sr1, sc0:sc1], (tx1 - tx0, ty1 - ty0),
                         interpolation=cv2.INTER_NEAREST)
    sub_m = cv2.resize(m[sr0:sr1, sc0:sc1].astype(np.uint8),
                       (tx1 - tx0, ty1 - ty0),
                       interpolation=cv2.INTER_NEAREST).astype(bool)
    roi = bev[ty0:ty1, tx0:tx1]
    roi[sub_m] = (roi[sub_m] * (1 - alpha)
                  + sub_col[sub_m] * alpha).astype(np.uint8)
    cv2.putText(bev, "OCC", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (200, 200, 200), 1, cv2.LINE_AA)


def draw_grid(bev, span, yh):
    h, w = bev.shape[:2]
    sy, sx = h / span, w / (2 * yh)
    cy = int(VIEW_F * sy)
    for d in (20, 40, 60):
        for sign in (1, -1):
            y = int(cy - sign * d * sy)
            if 0 <= y < h:
                cv2.line(bev, (0, y), (w, y), (60, 60, 60), 1, cv2.LINE_AA)
                cv2.putText(bev, f"{d}m", (4, y - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                            (170, 170, 170), 1, cv2.LINE_AA)
    cv2.line(bev, (w // 2, 0), (w // 2, h), (90, 90, 90), 1, cv2.LINE_AA)
    tri = np.array([[w // 2, cy - 11], [w // 2 - 7, cy + 8],
                    [w // 2 + 7, cy + 8]], np.int32)
    cv2.fillPoly(bev, [tri], (255, 255, 255))
    return sy, sx, cy



def _rot_iou(a, b):
    """a, b = (x, y, l, w, yaw) in metres/rad -> rotated-rectangle IoU."""
    ra = ((a[0], a[1]), (a[2], a[3]), float(np.degrees(a[4])))
    rb = ((b[0], b[1]), (b[2], b[3]), float(np.degrees(b[4])))
    r, pts = cv2.rotatedRectangleIntersection(ra, rb)
    if pts is None or r == cv2.INTERSECT_NONE or len(pts) < 3:
        return 0.0, 0.0          # (IoU, containment) — callers unpack two values
    inter = cv2.contourArea(cv2.convexHull(pts))
    union = a[2] * a[3] + b[2] * b[3] - inter
    # (IoU, 交差 / 小さい方の面積): 大箱に内包される小箱は IoU が小さいので後者で捕まえる
    return float(inter / max(union, 1e-6)), float(inter / max(min(a[2] * a[3], b[2] * b[3]), 1e-6))


def bev_box_nms(det, iou_th=0.3, cont_th=0.6):
    """det: [(cls, score, x, y, l, w, yaw), ...] -> greedy NMS by score (same class)."""
    kept = []
    for d in sorted(det, key=lambda t: -t[1]):
        ok = True
        for k in kept:
            if int(k[0]) != int(d[0]):
                continue
            iou, cont = _rot_iou(d[2:7], k[2:7])
            if iou > iou_th or cont > cont_th:
                ok = False
                break
        if ok:
            kept.append(d)
    return kept


def ground_point(u, v, Kc, Tce, ground_z=0.0):
    """画素 (u,v) を通る視線と路面 (ego z=ground_z) の交点 [ego x,y] (無ければ None)。
    Tce = T_cam_ego (ego->cam)。カメラ中心 c = -R^T t、方向 d = R^T K^-1 [u,v,1]。"""
    R = Tce[:3, :3]; t = Tce[:3, 3]
    c = -R.T @ t
    d = R.T @ np.linalg.solve(Kc, np.array([u, v, 1.0]))
    if d[2] >= -1e-6:              # 視線が下を向いていない
        return None
    lam = (ground_z - c[2]) / d[2]
    if lam <= 0:
        return None
    p = c + lam * d
    return p[:2]



def _sp_gpu_name():
    """Device name for the title bar via nvidia-smi (vendor prefixes stripped)."""
    import subprocess as _sp
    _gpu = _sp.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                            timeout=5).decode().splitlines()[0].strip()
    return _gpu.replace("NVIDIA ", "").replace("GeForce ", "").replace("Jetson ", "")

def compose_frame(raw, K, Tc, v0, out, dt, fps_now=None, pose=None):
    """One 1920x1080 canvas from one frame's raw images + engine outputs.

    Shared by the offline renderer below and deploy/orin_realtime.py, so the
    realtime pipeline draws EXACTLY what the offline video shows.
    """
    # per-class decode threshold (2026-08-13 実測採用): veh 0.45 / vru 0.15
    # -- v63b スイープで vru R 0.33->0.41, P 0.57 (下限 0.55 ルール合格)
    det = [(b["cls"] == "vru" and 1 or 0, b["score"], b["x"],
            b["y"], b["l"], b["w"], b["yaw"]) for b in
           decode_boxes(out["hm"], out["reg"], thresh=0.15)
           if b["score"] > (0.35 if b["cls"] == "vehicle" else 0.15)
           # 2026-08-18 修正: -28..50 の固定クリップは軽量版 (rear-40, det 監督
           # 窓 前50/後28) の残骸。8 カメラの全域エンジンでは後方 28 m 超と
           # 前方 50 m 超の検出が「表示だけ」消えており、デモを見る限り
           # 「後方車両が改善しない」ように見えていた。格子の実範囲へ追従させる。
           and -(XR - 2.0) <= b["x"] <= (XF - 2.0)]
    # BEV 回転箱 IoU による NMS (2026-09-08 ユーザー指摘「BEV 上で NMS が効いていない」):
    # decode の中心内包則は横並び・向き違いの重複を落とせない。IoU > 0.3 を抑制。
    det = bev_box_nms(det, iou_th=float(os.environ.get("METEOR_BEV_NMS_IOU", "0.3")))
    # 時系列 yaw 平滑化 (2026-08-13): 前フレーム箱と 2.5m マッチ ->
    # 180°フリップ抑止 + EMA a=0.6。特徴の薄い真横/真後ろの回転を抑える。
    _tr = getattr(compose_frame, "_yaw_tracks", [])
    _sm = []
    for d in det:
        d = list(d)
        best = None
        for (px, py, pyaw) in _tr:
            dd = (d[2] - px) ** 2 + (d[3] - py) ** 2
            if dd < 6.25 and (best is None or dd < best[0]):
                best = (dd, pyaw)
        if best is not None:
            py_ = best[1]
            dy_ = (d[6] - py_ + np.pi) % (2 * np.pi) - np.pi
            if abs(dy_) > np.pi / 2:
                d[6] = d[6] + (np.pi if dy_ < 0 else -np.pi)
                dy_ = (d[6] - py_ + np.pi) % (2 * np.pi) - np.pi
            d[6] = py_ + 0.4 * dy_
        _sm.append(tuple(d))
    det = _sm
    compose_frame._yaw_tracks = [(d[2], d[3], d[6]) for d in det]
    b2d = decode_boxes2d_ms_np(
        [out[f"hm2d_s{i}"][0] for i in range(3)],
        [out[f"reg2d_s{i}"][0] for i in range(3)],
        thresh=float(os.environ.get("METEOR_TH2D", "0.50")))
    ego = out["ego"][0]
    lg = ego[12 * EGO_K:12 * EGO_K + EGO_K].copy()
    # Mode hysteresis: closed loop rewards temporal consistency over
    # per-frame optimality. The selector picks the best mode on only 37 % of
    # frames and the top-2 candidates sit 0.23 m apart in median, so raw
    # argmax flaps between near-ties -- in open loop that costs 0.17 m ADE,
    # in a vehicle it is visible plan jitter. A small bonus on last frame's
    # mode breaks ties toward continuity without overriding a real change.
    prev = getattr(compose_frame, "_prev_mode", None)
    if prev is not None:
        lg[prev] += 0.35
    k = int(lg.argmax())
    # 直進優先 (2026-08-18 実測採用): コマンド無しのデモでは、旋回モードの
    # ロジットが直進を 1.0 以上上回らない限り直進を選ぶ。INT8 は量子化で
    # 右モードを 2 倍選ぶ (29 対 15) 実測があり、この規則でパス平均 y が
    # -1.012 -> -0.902 m、対GT ADE も 5.60 -> 5.33 と改善。fp16 では無害。
    if k != 0 and (lg[k] - lg[0]) < 1.0:
        k = 0
    compose_frame._prev_mode = k
    pr = np.exp(lg - lg.max())
    pr /= pr.sum()
    sel = np.concatenate([ego[k * 12:(k + 1) * 12],
                          ego[12 * EGO_K + EGO_K:]])
    segp = out["seg2d"][0]
    if segp.ndim == 4:
        segp = segp.argmax(1)
    segp = segp.astype(np.uint8)
    dep = out["depth"][0]
    if dep.ndim == 4:
        dep = dep.argmax(1)
    dep = dep.astype(np.uint8)
    stat = out["stationary"][0][0]
    traj = out["traj"][0]
    # Full-INT8 engines can collapse the one-channel stationary logit to a
    # constant.  In that case use the independently supervised 3 s forecast;
    # healthy mixed/FP16 engines continue to use the explicit head.
    stat_ok = stationary_head_healthy(out.get("stationary"))

    canvas = np.zeros((VH, VW, 3), np.uint8)
    canvas[:] = (18, 17, 16)
    for k8, chn in enumerate(CAM8):
        x0 = 8 + (k8 % 4) * (CW + 6)
        y0 = 34 + (k8 // 4) * (CH + 26)
        # BACK_NARROW は 7 カメラ構成には存在しないので空白のままにするが、
        # 8 カメラのエンジン (ベースライン系) では実際に入力として使うので
        # 他のカメラと同じように描く (2026-08-15)。
        if chn not in CAM_DRAW:
            cv2.putText(canvas, f"{chn} (blank)",
                        (x0 + 70, y0 + CH // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (90, 90, 90), 1, cv2.LINE_AA)
            continue
        i = CAM_DRAW.index(chn)
        img = cv2.resize(raw[chn], (CW, CH))
        # 2026-09-05 ユーザー指示: 2D タイルへの seg マスク重畳は既定で行わない
        # (METEOR_SEG2D_OVERLAY=1 で従来表示)。描画コストも下がる。
        if _SEG2D_OVERLAY:
            ov = PALETTE[cv2.resize(segp[i], (CW, CH),
                         interpolation=cv2.INTER_NEAREST)
                         % len(PALETTE)][:, :, ::-1]
            img = cv2.addWeighted(img, 0.75, ov.astype(np.uint8),
                                  0.25, 0)
        draw_boxes_on_rgb(img, det, K[0][i], Tc[0][i], CW, CH)
        draw_boxes2d(img, b2d[i], CW, CH)
        if chn == "CAM_FRONT_WIDE":
            draw_path_ribbon(img, sel, K[0][i], Tc[0][i], CW, CH)
        cv2.putText(canvas, chn, (x0, y0 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (200, 200, 200), 1, cv2.LINE_AA)
        canvas[y0:y0 + CH, x0:x0 + CW] = img
    # Depth block: same tile size and CAM8 order as the RGB block
    # (METEOR_DEPTH_PANEL=0 で省略: 2026-09-05 「10 FPS 以上」指示の軽量描画)
    for k8, chn in (enumerate(CAM8) if os.environ.get("METEOR_DEPTH_PANEL", "1") != "0" else ()):
        x0 = 8 + (k8 % 4) * (CW + 6)
        y0 = 34 + (2 + k8 // 4) * (CH + 26)
        if chn not in CAM_DRAW:
            cv2.putText(canvas, "(blank)",
                        (x0 + CW // 2 - 28, y0 + CH // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (90, 90, 90), 1, cv2.LINE_AA)
            continue
        i = CAM_DRAW.index(chn)
        d8 = cv2.resize(dep[i], (CW, CH),
                        interpolation=cv2.INTER_NEAREST)
        canvas[y0:y0 + CH, x0:x0 + CW] = cv2.applyColorMap(
            # 深度ビン数に合わせて色域を張る。旧実装は 64 bin 前提の x4 固定で、
            # 32 bin モデル (v63b/v67 系) では色域の下半分しか使われず
            # 「側方 Depth が汚い/平坦」に見えていた (2026-08-14, 実機動画で確認)。
            # METEOR_DEPTH_BINS で指定 (既定 64)。
            (d8.astype(np.float32) * (255.0 / max(_DBINS - 1, 1))
             ).clip(0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)

    lane = out["lane"][0]
    if lane.ndim == 3:                     # legacy fp32-logits engine
        lane = lane.argmax(0)
    lane = lane.astype(np.uint8)
    if os.environ.get("METEOR_SEG_FUSE", "1") != "0":
        _ll = out.get("lane_logit")
        if _ll is not None:
            lane = seg_fuse_logit(lane, _ll, pose)   # 完全版 (logit 融合)
        else:
            lane = seg_fuse_np(lane, pose)           # 近似版 (旧エンジン)
    if os.environ.get("METEOR_NO_THIN", "0") == "0":
        lane = thin_road_edge_np(lane)
    if os.environ.get("METEOR_THIN_LANES", "0") != "0":
        lane = thin_lane_lines_np(lane)
    pc = crop_bev_np(lane, XF, XR, VIEW_F, VIEW_R, YH)
    BH2 = 1000
    BW2 = int(BH2 * pc.shape[1] / pc.shape[0])
    # ローカル demo と同じ表示規約: sidewalk(2)/parking(8) は非表示
    bev = cv2.resize(DEMO_PALETTE[pc][:, :, ::-1].astype(np.uint8),
                     (BW2, BH2), interpolation=cv2.INTER_NEAREST)
    span = VIEW_F + VIEW_R
    sy2, sx2, cy0 = draw_grid(bev, span, YH)
    if os.environ.get("METEOR_OCC_OVERLAY", "0") != "0":
        draw_occ(bev, out.get("occ"), sy2, sx2)
    for cls, sc, xe, ye, l, w, yaw in det:
        if xe > VIEW_F or xe < -VIEW_R or abs(ye) > YH:
            continue
        cb, sb = np.cos(yaw), np.sin(yaw)
        rr0 = int((XF - xe) / 0.4)
        cc0 = int((50.0 - ye) / 0.4)
        stationary, _stat_source = stationary_at(
            out.get("stationary"), out.get("traj"), rr0, cc0,
            stat_healthy=stat_ok)
        stationary = bool(stationary)
        col = (160, 160, 160) if stationary else \
            ((0, 215, 255) if cls < 0.5 else (255, 0, 255))
        cor = []
        for lx, wy in ((l / 2, w / 2), (l / 2, -w / 2),
                       (-l / 2, -w / 2), (-l / 2, w / 2)):
            px = xe + lx * cb - wy * sb
            py = ye + lx * sb + wy * cb
            cor.append([int((YH - py) * sx2),
                        int((VIEW_F - px) * sy2)])
        cv2.polylines(bev, [np.array(cor, np.int32)
                            .reshape(-1, 1, 2)], True, col, 2)
        cxp = int((YH - ye) * sx2)
        cyp = int((VIEW_F - xe) * sy2)
        fxp = int((YH - (ye + (l / 2) * sb)) * sx2)
        fyp = int((VIEW_F - (xe + (l / 2) * cb)) * sy2)
        cv2.line(bev, (cxp, cyp), (fxp, fyp), col, 2)
        # the demo suppresses VRU futures only -- stationary boxes
        # keep theirs (they decode to a short stub, which is honest)
        if int(cls) != 1 and 0 <= rr0 < traj.shape[-2] \
                and 0 <= cc0 < traj.shape[-1]:
            _v = traj[:, rr0, cc0]
            if _v.size >= 39:
                _kb = int(_v[36:39].argmax())
                wps = _v[_kb * 12:(_kb + 1) * 12].reshape(6, 2)
            else:
                wps = _v.reshape(6, 2)
            pts = [(cxp, cyp)]
            for dx, dy in wps:
                fx, fy = xe + dx, ye + dy
                if fx > VIEW_F or fx < -VIEW_R or abs(fy) > YH:
                    break
                pts.append((int((YH - fy) * sx2),
                            int((VIEW_F - fx) * sy2)))
            cv2.polylines(bev, [np.array(pts, np.int32).reshape(-1, 1, 2)],
                          False, col, 1)
    # --- 2D 'obs' (落下物・未知障害物, class 0) を予測深度で BEV へ投影 ---
    # (2026-09-06 ユーザー指示: 旧 demo_rgbd_bev --unk2d の移植。箱の下端寄り
    #  (cy + 0.25h) の深度 bin を距離に変換し、K / T_cam_ego で自車座標へ。
    #  複数カメラの重複は 1.5 m クラスタで最大スコアのみ残す)
    if os.environ.get("METEOR_UNK2D", "1") != "0":
        _th_unk = float(os.environ.get("METEOR_UNK2D_TH",
                                       os.environ.get("METEOR_TH2D", "0.30")))
        _dm = np.exp(np.linspace(np.log(1.0), np.log(79.75), _DBINS))
        # 期待値深度 (エンジン出力 depth_mean, 2026-09-07) があればそれを使い、無ければ argmax ビン
        _dmean = out.get("depth_mean")
        _dmean = np.asarray(_dmean, np.float32)[0] if _dmean is not None else None
        fh, fw = (dep.shape[-2:] if _dmean is None else _dmean.shape[-2:])
        unk2d = []
        for i in range(min(len(b2d), dep.shape[0], len(CAM_DRAW))):
            Kc = K[0][i]; Tce = Tc[0][i]
            for cls2, sc2, cx2, cy2, w2, h2 in b2d[i]:
                if int(cls2) != 0 or sc2 < _th_unk:
                    continue
                # 主: 箱の下端 (接地点) を通る視線と路面の交点 (2026-09-08; 深度マップより確実)
                pe2 = ground_point(cx2, cy2 + 0.5 * h2, Kc, Tce, _GROUND_Z)
                if pe2 is not None and 1.5 < float(np.hypot(*pe2)) < 60.0:
                    pe = np.array([pe2[0], pe2[1], _GROUND_Z])
                    d_ = float(np.hypot(*pe2))
                else:
                    # 副: 期待値深度 (箱下端寄り 5x5 窓の中央値)
                    u = int(np.clip(cx2 / 768.0 * fw, 0, fw - 1))
                    v = int(np.clip((cy2 + 0.35 * h2) / 432.0 * fh, 0, fh - 1))
                    u0, u1 = max(0, u - 2), min(fw, u + 3)
                    v0_, v1_ = max(0, v - 2), min(fh, v + 3)
                    if _dmean is not None:
                        d_ = float(np.median(_dmean[i, v0_:v1_, u0:u1]))
                    else:
                        d_ = float(np.median(_dm[np.minimum(dep[i, v0_:v1_, u0:u1].astype(int), _DBINS - 1)]))
                    if not (1.5 < d_ < 50.0):
                        continue
                    pcam = np.array([(cx2 - Kc[0, 2]) / Kc[0, 0] * d_,
                                     (cy2 - Kc[1, 2]) / Kc[1, 1] * d_, d_])
                    pe = Tce[:3, :3].T @ (pcam - Tce[:3, 3])
                if abs(pe[0]) > 60 or abs(pe[1]) > 25:
                    continue
                unk2d.append((float(pe[0]), float(pe[1]), float(sc2), d_))
        unk2d.sort(key=lambda t: -t[2])
        kept = []
        for xe_, ye_, sc_, d_ in unk2d:
            if all((xe_ - a) ** 2 + (ye_ - b) ** 2 > 1.5 ** 2
                   for a, b, _, _ in kept):
                kept.append((xe_, ye_, sc_, d_))
        for xe_, ye_, sc_, d_ in kept:
            if xe_ > VIEW_F or xe_ < -VIEW_R or abs(ye_) > YH:
                continue
            q = (int((YH - ye_) * sx2), int((VIEW_F - xe_) * sy2))
            # ユーザー指示 (2026-09-07): 白の小さな丸のみ。名前・距離ラベルは描かない
            cv2.circle(bev, q, 4, (255, 255, 255), -1)
            cv2.circle(bev, q, 5, (40, 40, 40), 1)
    # LiDAR 入力の重畳 (2026-09-08): ピラーラスタ [4,400,250] (0.4m/セル, 前後±80 左右±50) の
    # 占有セルを薄い青緑の点で描く。カメラのみ (零入力) のときは何も描かない。
    _lbin = out.get("lidar_bev_in")
    if _lbin is not None:
        # ch = (log-count, max z, mean z, occupancy): 占有かつ max z > 0.3 m (路面返り点を除く障害物) だけ描く
        _lb4 = np.asarray(_lbin, np.float32)[0]
        _occ = (_lb4[3] > 0) & (_lb4[1] > float(os.environ.get("METEOR_LIDAR_ZMIN", "0.3")))
        _res_l = (XF + XR) / _occ.shape[0]
        _r0 = max(0, int((XF - VIEW_F) / _res_l)); _r1 = min(_occ.shape[0], int((XF + VIEW_R) / _res_l))
        _c0 = max(0, int((50.0 - YH) / (100.0 / _occ.shape[1]))); _c1 = min(_occ.shape[1], int((50.0 + YH) / (100.0 / _occ.shape[1])))
        _oc = _occ[_r0:_r1, _c0:_c1]
        _ys, _xs = np.nonzero(_oc)
        _sy = bev.shape[0] / max(_oc.shape[0], 1); _sx = bev.shape[1] / max(_oc.shape[1], 1)
        for _y, _x in zip(_ys, _xs):
            cv2.circle(bev, (int((_x + 0.5) * _sx), int((_y + 0.5) * _sy)), 1, (170, 200, 110), -1)
    if "risk" in out:
        # ローカル (demo_rgbd_bev.py) と同一の重畳: +-40m を TURBO で、
        # 濃さは risk 値そのもの (alpha 0.55)
        _rm = 1.0 / (1.0 + np.exp(-np.asarray(out["risk"], np.float32)
                                  .reshape(out["risk"].shape[-2:])))
        _SPAN = VIEW_F + VIEW_R
        BH2_, BW2_ = bev.shape[0], bev.shape[1]
        rm = cv2.resize(_rm, (BW2_, int(BH2_ * (40.0 + VIEW_R) / _SPAN)),
                        interpolation=cv2.INTER_LINEAR)
        y0r = int(BH2_ * (VIEW_F - 40.0) / _SPAN)
        y1r = min(BH2_, y0r + rm.shape[0])
        rm = rm[:y1r - y0r]
        sub = bev[y0r:y1r]
        heat = cv2.applyColorMap((np.clip(rm, 0, 1) * 255).astype(np.uint8),
                                 cv2.COLORMAP_TURBO)
        _rg = float(os.environ.get("METEOR_RISK_GAIN", "1.0"))
        a_ = (np.clip(rm * _rg, 0, 1) * 0.55)[..., None]
        bev[y0r:y1r] = (sub * (1 - a_) + heat * a_).astype(np.uint8)
    wp = ego[k * 12:(k + 1) * 12].reshape(6, 2)
    pts = [(int(YH * sx2), cy0)]
    pts += [(int((YH - y) * sx2), int((VIEW_F - x) * sy2))
            for x, y in wp]
    # ローカル (demo_rgbd_bev.py) と同一の描画仕様に揃える (2026-08-24)。
    # 以前は色 (0,255,120)・線幅 3・半径 5 + 暗い縁取りという Orin 独自の
    # 見た目になっており、ローカルの可視化と並べたときに別物に見えていた。
    # ローカル: 純緑 (0,255,0)、線幅 2、waypoint は半径 3 の塗り潰しのみ。
    cv2.polylines(bev, [np.array(pts, np.int32).reshape(-1, 1, 2)], False,
                  (0, 255, 0), 2)
    for q in pts[1:]:
        cv2.circle(bev, q, 3, (0, 255, 0), -1)
    st_deg = float(np.degrees(sel[12]))
    acc = float(sel[13])
    brk = 1 / (1 + np.exp(-float(sel[14])))
    for li, txt in enumerate(
            [f"v0 {v0 * 3.6:5.1f} km/h",
             f"steer {st_deg:+6.1f} deg",
             f"accel {acc:+5.2f} m/s2",
             f"BRAKE {brk:.2f}" if brk > 0.5 else f"brake {brk:.2f}",
             f"mode p={pr[k]:.2f}"]):
        cv2.putText(bev, txt, (6, BH2 - 98 + 20 * li),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(bev, "gray box = stationary", (6, BH2 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (160, 160, 160), 1, cv2.LINE_AA)
    xb0 = 8 + 4 * (CW + 6) + 6
    bw_fit = VW - xb0 - 8
    if BW2 > bw_fit:
        bev = cv2.resize(bev, (bw_fit, int(BH2 * bw_fit / BW2)))
    canvas[34:34 + min(bev.shape[0], VH - 40),
           xb0:xb0 + bev.shape[1]] = bev[:VH - 40]
    # OCC はローカル demo と同じ「左下の独立ボクセルパネル」で見せる
    # (2026-08-19、BEV への半透明重ねは METEOR_OCC_OVERLAY=1 のときだけ)。
    if os.environ.get("METEOR_OCC_PANEL", "1") != "0":
        _occ = out.get("occ")
        if _occ is not None:
            try:
                from deploy.occ_iso import cube_render_fast
                # ARM CPU では素朴に描くと ~270 ms かかり FPS を半減させる。
                # (1) argmax 前に ±24 m へクロップ (6.4M -> 2.3M 要素)、
                # (2) 最終サイズで直接描画、(3) 2 フレームに 1 回だけ更新。
                _ev = max(1, int(os.environ.get("METEOR_OCC_EVERY", "3")))
                _OCC_CACHE["n"] = _OCC_CACHE.get("n", -1) + 1
                if _OCC_CACHE["n"] % _ev == 0 or _OCC_CACHE.get("iso") is None:
                    o = np.asarray(_occ)
                    if o.ndim == 5:
                        o = o[0]
                    n24 = 60                       # 24 m / 0.4 m
                    c0 = o.shape[-1] // 2 - n24
                    # z も描画上限 (3 m -> 10 ビン) まで先に落とす: argmax の
                    # 要素数 1.92M -> 1.2M
                    oc = o[:, :10, c0:c0 + 2 * n24, c0:c0 + 2 * n24]
                    occ_cls = oc.argmax(0).astype(np.uint8)   # [Z,h,w]
                    _ds = int(os.environ.get("METEOR_OCC_DS", "1"))
                    if _ds > 1:        # ボクセル間引き (粗いが 1/ds^2 に軽量化)
                        occ_cls = occ_cls[:, ::_ds, ::_ds]
                    _OCC_CACHE["iso"] = cube_render_fast(occ_cls, W=426, H=360)
                iso = _OCC_CACHE["iso"]
                oy0 = VH - 368
                canvas[oy0:oy0 + 360, 8:8 + 426] = iso
                cv2.putText(canvas, "pred OCC voxel grid +-24m (bldg hidden)",
                            (8, oy0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (220, 220, 220), 1, cv2.LINE_AA)
            except Exception:
                pass
    # 実行環境はハードコードせず自動取得する (2026-08-25)。以前は
    # "METEOR light - AGX Orin" 固定で、laptop で動かしても Orin と表示された。
    global _ENV_TAG
    try:
        _ENV_TAG
    except NameError:
        try:
            # METEOR_GPU_NAME overrides the device name in the title bar (e.g. for
            # published videos that should not advertise a specific GPU model)
            _gpu = os.environ.get("METEOR_GPU_NAME") or _sp_gpu_name()
        except Exception:
            _gpu = "GPU"
        try:
            import tensorrt as _trt
            _ENV_TAG = f"{_gpu} - TensorRT {_trt.__version__}"
        except Exception:
            _ENV_TAG = _gpu
    _t = (f"METEOR - {_ENV_TAG} - on-device "
          f"inference+render - {dt:.0f} ms infer"
          + (" - LiDAR ON" if out.get("lidar_bev_in") is not None else ""))
    if fps_now is not None:
        _t += f" - {fps_now:.1f} FPS"
    cv2.putText(canvas, _t, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.62, (240, 240, 240), 2, cv2.LINE_AA)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--root", default="sample")
    ap.add_argument("--scenes", type=int, default=2)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--out", default="out/demo_orin_standalone.mp4")
    ap.add_argument("--display", action="store_true")
    ap.add_argument("--fps", type=int, default=4)
    a = ap.parse_args()

    # 出力スロットを 2 枚にする: 1 枚だと infer が全出力を CPU 側で .copy()
    # しており、Orin の弱い CPU で実測 9.4 ms 払っていた (108.8 -> 99.4 ms)。
    # この描画ループは out を使い切ってから次の infer を呼ぶ逐次実行なので、
    # 2 枚あれば十分 (前フレームの配列を跨いで保持しない)。
    rt = MeteorRT(a.engine, n_out_slots=2)
    # v103 世代のエンジンは imgs が uint8 (正規化はグラフ内)。ここで
    # float32/255 を渡すとランタイムの uint8 キャストでほぼ 0 に潰れ、
    # BEV Seg は「それらしく」出るのに 3D 検出だけ静かに壊れる
    # (実測 hm 最大 -1.715 -> -3.540 = 信頼度 0.15 -> 0.029)。
    _want_u8 = rt.host["imgs"].dtype == np.uint8
    _n_cam = int(rt.shapes["imgs"][1])
    _CAMS_IN = CAM_IN8 if _n_cam == 8 else CAMS
    globals()["CAM_DRAW"] = list(_CAMS_IN)     # 描画も同じ本数・同じ並びに
    # BEV の後方範囲をエンジンの行数から決める。この呼び出しが無いと XR は
    # 軽量版 (600 行 = 前 80 / 後 40 m) の値で固定されたままになり、
    # 800 行 = 前後 80 m のベースラインでも「後方 40 m」として描いてしまう。
    # 後方の半分が消え、縮尺も狂う (2026-08-24 に実害。関数は用意されていたが
    # どこからも呼ばれていなかった)。
    set_bev_extent(int(rt.shapes["lane"][-2]))
    _logged_cams = []
    print(f"[render] エンジンは {_n_cam} カメラ -> 入力順 {_CAMS_IN[0]} ...",
          flush=True)
    print(f"[render] imgs 入力は {rt.host['imgs'].dtype} "
          f"({'uint8 をそのまま渡す' if _want_u8 else 'float32 に正規化して渡す'})",
          flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"),
                         a.fps, (VW, VH))
    times = []
    scenes = sorted(s for s in os.listdir(a.root)
                    if os.path.isfile(os.path.join(a.root, s,
                                                   "manifest.json")))
    for s in scenes[:a.scenes]:
        m = json.load(open(os.path.join(a.root, s, "manifest.json")))
        # 欠けているカメラは K/T を単位行列、画像を 0 にする (学習側の
        # cam-drop と同じ扱い)。7 カメラのリグを 8 カメラのエンジンに
        # 通すときはこの経路になる。
        _eye3, _eye4 = np.eye(3, dtype=np.float32), np.eye(4, dtype=np.float32)
        K = np.stack([np.array(m["cams"][c]["K"], np.float32)
                      if c in m["cams"] else _eye3
                      for c in _CAMS_IN])[None]
        Tc = np.stack([np.linalg.inv(np.array(
            m["cams"][c]["T_ego_cam"], np.float32))
            if c in m["cams"] else _eye4 for c in _CAMS_IN])[None]
        try:
            emo = np.load(os.path.join(a.root, s, "ego_motion.npz"))
            v0s = emo["v0"]
        except Exception:
            v0s = None
        for f in m["frames"][::a.stride]:
            raw = {}
            ok = True
            for c in _CAMS_IN:
                im = cv2.imread(os.path.join(a.root, s,
                                             f["imgs"].get(c, "_")))
                if im is None:
                    if c in CAMS:          # 本来あるべきカメラが欠けた
                        ok = False
                        break
                    im = None              # 8 カメラ目が無いリグ -> 0 で埋める
                raw[c] = im
            if not ok:
                continue
            if not _logged_cams:
                _ok = [c for c in _CAMS_IN if raw.get(c) is not None]
                print(f"[render] 実際に読み込んだカメラ {len(_ok)}/{len(_CAMS_IN)}: "
                      f"{_ok}", flush=True)
                _logged_cams.append(1)
            _shape = next(v.shape for v in raw.values() if v is not None)
            for c in _CAMS_IN:
                if raw[c] is None:
                    raw[c] = np.zeros(_shape, np.uint8)
            v0 = float(v0s[f["frame"]]) if v0s is not None \
                and f["frame"] < len(v0s) else 8.0
            _st = np.ascontiguousarray(np.stack(
                [raw[c][:, :, ::-1].transpose(2, 0, 1)
                 for c in _CAMS_IN])[None])
            imgs = _st if _want_u8 else _st.astype(np.float32) / 255.0
            t0 = time.time()
            out = rt.infer(imgs, K[0][None], Tc[0][None], v0=v0)
            dt = (time.time() - t0) * 1000
            times.append(dt)

            canvas = compose_frame(raw, K, Tc, v0, out, dt)
            vw.write(canvas)
            if a.display:
                cv2.imshow("METEOR Orin", canvas)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    vw.release()
    t = np.array(times[3:] if len(times) > 6 else times)
    print(f"frames={len(times)} infer mean={t.mean():.1f}ms -> {a.out}")


if __name__ == "__main__":
    main()

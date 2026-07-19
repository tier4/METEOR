#!/usr/bin/env python3
"""Deploy-side frame composer — pixel-identical to bevlane/demo_rgbd_bev.

Reuses the demo's palettes and drawing helpers directly (single source of
truth), consuming the 18 TensorRT engine outputs instead of the PyTorch
model. Layout (1920x1080): 8-cam RGB + 2D-seg overlay + 2D/3D boxes on
top, per-camera depth (top-mode expectation) below, occupancy voxel-cube
grid bottom-left, BEV column (lanes / boxes / K=3 E2E / TL / controls)
on the right.
"""
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from autolabel_bev import PALETTE  # noqa: E402
from bevlane.demo_occ_gt import cube_render  # noqa: E402
from bevlane.demo_rgbd_bev import (SURR, NARROW, draw_boxes2d,  # noqa: E402
                                   draw_boxes_on_rgb)
from bevlane.extract_seg2d import SEG21_PAL as _S21  # noqa: E402
SEG2D_PAL = np.zeros((256, 3), np.uint8)
SEG2D_PAL[:21] = _S21[:, ::-1]
from bevlane.model import DepthSegIPMNetV17  # noqa: E402
from bevlane.postproc import (crop_bev, draw_ego_and_grid,  # noqa: E402
                              thin_road_edge)

VW, VH = 1920, 1080
DEMO_PALETTE = np.zeros((12, 3), np.uint8)
DEMO_PALETTE[:len(PALETTE)] = PALETTE
DEMO_PALETTE[2] = 0
DEMO_PALETTE[8] = 0
CAM8 = SURR[:3] + [NARROW[0]] + SURR[3:] + [NARROW[1]]
D_MIN, D_STEP, ND = 1.0, 1.25, 64


def _depth_map(dlog):
    """[N,D,h,w] logits -> top-mode expected depth (demo-identical)."""
    p = torch.from_numpy(dlog).float().softmax(1)
    pk = p.argmax(1, keepdim=True)
    ar = torch.arange(p.shape[1]).view(1, -1, 1, 1)
    pw = p * ((ar - pk).abs() <= 2)
    bins = (torch.arange(ND).float() * D_STEP + D_MIN).view(1, -1, 1, 1)
    return ((pw * bins).sum(1) / pw.sum(1).clamp(min=1e-6)).numpy()


def compose_frame(imgs, K, T, out, v0, boxes, scene, fi, guard=None):
    """imgs: [8,432,768,3] BGR uint8 in CAMS order. out: engine dict.
    boxes: decoded (cls,score,xe,ye,l,w,yaw,stationary) list."""
    frame = np.zeros((VH, VW, 3), np.uint8)
    cw, ch = 375, 211
    seg = out["seg2d"]
    seg = seg[0] if seg.ndim == 5 else seg                # [8,21,h,w]
    dsh = out["depth"]
    dep = _depth_map(dsh[0] if dsh.ndim == 5 else dsh)
    # ---- 2D boxes via the shared torch decoder ----
    b2d = None
    try:
        hm2 = torch.from_numpy(out["hm2d"]).float()
        rg2 = torch.from_numpy(out["reg2d"]).float()
        b2d = DepthSegIPMNetV17.decode_boxes2d(hm2, rg2, thresh=0.25)
    except Exception:
        pass
    det7 = [b[:7] for b in boxes]
    # ---- top block: RGB + seg overlay + boxes ----
    for k, chn in enumerate(CAM8):
        ci = CAM8_TO_IDX[chn]
        r, c = divmod(k, 4)
        x, y = c * cw, 40 + r * (ch + 4)
        img = cv2.resize(imgs[ci], (cw, ch))
        sg = seg[ci].argmax(0).astype(np.uint8)
        ov = SEG2D_PAL[cv2.resize(sg, (cw, ch),
                                  interpolation=cv2.INTER_NEAREST)]
        img = cv2.addWeighted(img, 0.62, ov, 0.38, 0)
        draw_boxes_on_rgb(img, det7, K[ci], T[ci], cw, ch)
        if b2d is not None and ci < len(b2d):
            draw_boxes2d(img, b2d[ci], cw, ch)
        if chn in NARROW:
            cv2.putText(img, "NARROW", (6, 18), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 1, cv2.LINE_AA)
        frame[y:y + ch, x:x + cw] = img
    cv2.putText(frame, "RGB + predicted 2D Seg overlay", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1,
                cv2.LINE_AA)
    # ---- depth block ----
    dy0 = 40 + 2 * (ch + 4) + 36
    for k, chn in enumerate(CAM8):
        ci = CAM8_TO_IDX[chn]
        r, c = divmod(k, 4)
        if r == 1 and c == 0:
            continue                                  # cube panel slot
        x, y = c * cw, dy0 + r * (ch + 4)
        d = cv2.resize(dep[ci], (cw, ch))
        dv = np.clip(d / 80.0 * 255, 0, 255).astype(np.uint8)
        dm = cv2.applyColorMap(dv, cv2.COLORMAP_TURBO)
        cv2.putText(dm, chn.replace("CAM_", ""), (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)
        frame[y:y + ch, x:x + cw] = dm
    cv2.putText(frame, "predicted Depth (0-80m)", (10, dy0 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1,
                cv2.LINE_AA)
    # ---- occupancy cube panel (demo gating) ----
    op = torch.from_numpy(out["occ"][0]).float().softmax(0)
    conf = 1.0 - op[0]
    cls = (op[1:].argmax(0) + 1).to(torch.uint8)
    thr = torch.where((cls == 7) | (cls == 8),
                      torch.full_like(conf, 0.92),
                      torch.full_like(conf, 0.55))
    occ_pred = torch.where(conf > thr, cls,
                           torch.zeros_like(cls)).numpy().astype(np.uint8)
    oy0 = dy0 + ch + 4
    iso = cv2.resize(cube_render(occ_pred, W=900, H=760),
                     (cw + 45, ch))
    frame[oy0:oy0 + ch, 0:cw + 45] = iso
    cv2.putText(frame, "pred OCC voxel grid +-24m (bldg hidden)",
                (8, oy0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (220, 220, 220), 1, cv2.LINE_AA)
    # ---- BEV column ----
    lane = out["lane"][0].argmax(0).astype(np.uint8)
    lane = thin_road_edge(lane)
    pc = crop_bev(lane, xh_m=60.0, yh_m=25.0)
    BH2, BW2 = VH - 70, 420
    bev = np.ascontiguousarray(DEMO_PALETTE[pc][:, :, ::-1])
    bev = draw_ego_and_grid(bev, BH2, BW2, xh_m=60.0, yh_m=25.0)
    risk = 1.0 / (1.0 + np.exp(-out["risk"][0, 0]))
    rz = cv2.resize(risk, (BW2, BH2))
    glow = (np.stack([rz * 40, rz * 180, rz * 160], -1)).astype(np.uint8)
    bev = cv2.addWeighted(bev, 1.0, glow, 0.55, 0)
    sy2, sx2 = BH2 / 120.0, BW2 / 50.0
    xy2px = lambda x_, y_: (int((25.0 - y_) * sx2), int((60.0 - x_) * sy2))
    for b in boxes:
        cls_, sc, xe, ye, l, w, yaw = b[:7]
        st = len(b) > 7 and b[7]
        if not (-60 < xe < 60 and abs(ye) < 25):
            continue
        cb, sb = np.cos(yaw), np.sin(yaw)
        cor = [xy2px(xe + lx * cb - wy_ * sb, ye + lx * sb + wy_ * cb)
               for lx, wy_ in ((l / 2, w / 2), (l / 2, -w / 2),
                               (-l / 2, -w / 2), (-l / 2, w / 2))]
        col = (160, 160, 160) if st else \
            ((0, 215, 255) if cls_ == 0 else (255, 0, 255))
        cv2.polylines(bev, [np.array(cor, np.int32).reshape(-1, 1, 2)],
                      True, col, 2)
        # heading tick centre -> front edge (demo-identical)
        cxp, cyp = xy2px(xe, ye)
        fxp, fyp = xy2px(xe + (l / 2) * cb, ye + (l / 2) * sb)
        cv2.line(bev, (cxp, cyp), (fxp, fyp), col, 2)
        # predicted 3 s agent future from the traj head
        tm = out["traj"][0]
        rr = int((80.0 - xe) / 0.4)
        cc2 = int((50.0 - ye) / 0.4)
        if not st and 0 <= rr < tm.shape[-2] and 0 <= cc2 < tm.shape[-1]:
            v_ = tm[:, rr, cc2]
            if v_.shape[0] >= 39:
                kb_ = int(v_[36:39].argmax())
                wps = v_[kb_ * 12:(kb_ + 1) * 12].reshape(6, 2)
            else:
                wps = v_[:12].reshape(6, 2)
            pts_ = [(cxp, cyp)]
            for dx_, dy_ in wps:
                fx_, fy_ = xe + dx_, ye + dy_
                if abs(fx_) > 60 or abs(fy_) > 25:
                    break
                pts_.append(xy2px(fx_, fy_))
            if len(pts_) > 1:
                cv2.polylines(bev, [np.array(pts_, np.int32
                                             ).reshape(-1, 1, 2)],
                              False, col, 1, cv2.LINE_AA)
                cv2.circle(bev, pts_[-1], 3, col, -1)
    e = out["ego"][0]
    paths = e[:36].reshape(3, 6, 2)
    conf3 = np.exp(e[36:39]) / np.exp(e[36:39]).sum()
    kb = int(conf3.argmax())
    for k in range(3):
        pts = [xy2px(0, 0)]
        for x_, y_ in paths[k]:
            if abs(x_) > 60 or abs(y_) > 25:
                break
            pts.append(xy2px(float(x_), float(y_)))
        col = (0, 255, 0) if k == kb else (255, 200, 60)
        trav = float(np.hypot(*paths[k][-1]))
        if k == kb and trav < 1.0:
            cv2.circle(bev, pts[0], 9, (0, 255, 0), 2)
            cv2.putText(bev, "HOLD", (pts[0][0] + 12, pts[0][1] + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                        cv2.LINE_AA)
        elif len(pts) > 1:
            cv2.polylines(bev, [np.array(pts, np.int32).reshape(-1, 1, 2)],
                          False, col, 3 if k == kb else 1, cv2.LINE_AA)
            if k != kb:
                cv2.putText(bev, f"{conf3[k]:.2f}", pts[-1],
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1,
                            cv2.LINE_AA)
    tl_p = np.exp(out["tl"][0]) / np.exp(out["tl"][0]).sum()
    tli = int(tl_p.argmax())
    tname = ["none", "green", "yellow", "red"][tli]
    tcol = [(200, 200, 200), (0, 255, 0), (0, 255, 255), (0, 0, 255)][tli]
    cv2.circle(bev, (BW2 - 150, 22), 9, tcol, -1)
    cv2.putText(bev, f"TL {tname.upper()} {tl_p[tli]:.2f}",
                (BW2 - 135, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, tcol, 2,
                cv2.LINE_AA)
    st_deg = np.degrees(e[39])
    brk = 1.0 / (1.0 + np.exp(-e[41]))
    for li, txt in enumerate([f"v0 {v0 * 3.6:5.1f} km/h",
                              f"steer {st_deg:+6.1f} deg",
                              f"accel {e[40]:+.2f} m/s2",
                              f"brake {brk:.2f}"]):
        cv2.putText(bev, txt, (8, BH2 - 86 + 22 * li),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 100, 255) if (li == 3 and brk > 0.5)
                    else (0, 255, 0), 1, cv2.LINE_AA)
    if guard is not None:
        gv = guard["verdict"]
        if gv == "OK":
            cv2.putText(bev, "GUARD OK", (6, BH2 - 118),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2,
                        cv2.LINE_AA)
        elif gv == "HOLD":
            cv2.putText(bev, "GUARD OK (holding)", (6, BH2 - 118),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 220, 0), 1,
                        cv2.LINE_AA)
        else:
            cv2.putText(bev, "GUARD VETO", (6, BH2 - 138),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
                        cv2.LINE_AA)
            cv2.putText(bev, guard["reason"], (6, BH2 - 118),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 80, 255), 1,
                        cv2.LINE_AA)
            if guard.get("p_event") is not None:
                ex_, ey_ = guard["p_event"]
                if abs(ex_) < 60 and abs(ey_) < 25:
                    cv2.drawMarker(bev, xy2px(ex_, ey_), (0, 0, 255),
                                   cv2.MARKER_TILTED_CROSS, 18, 3)
            if guard.get("stop_path") is not None:
                sp_ = [xy2px(0, 0)]
                for gx_, gy_ in guard["stop_path"]:
                    if abs(gx_) > 60 or abs(gy_) > 25:
                        break
                    sp_.append(xy2px(gx_, gy_))
                cv2.polylines(bev, [np.array(sp_, np.int32
                                             ).reshape(-1, 1, 2)],
                              False, (0, 200, 255), 2, cv2.LINE_AA)
                cv2.putText(bev, "MRM stop", sp_[-1],
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (0, 200, 255), 1, cv2.LINE_AA)
    frame[40:40 + BH2, VW - BW2:] = bev
    cv2.putText(frame, "pred BEV+bbox+E2E +-25x+-60m", (VW - BW2, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2,
                cv2.LINE_AA)
    cv2.putText(frame, f"{scene.split('_')[-1]}  f{fi:03d}  |  "
                "6/8-cam -> Depth + BEV (+-25x+-60m)  |  no GT  |  "
                "TensorRT", (10, VH - 14), cv2.FONT_HERSHEY_SIMPLEX,
                0.62, (0, 255, 0), 2, cv2.LINE_AA)
    return frame


from bevlane.dataset import CAMS as _CAMS  # noqa: E402
CAM8_TO_IDX = {c: i for i, c in enumerate(_CAMS)}

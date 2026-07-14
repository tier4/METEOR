"""Area risk-field GT — shared definition (user-approved, 2026-07-15).

Smooth potential field on a +-40 m x +-25 m @ 0.2 m ego grid (400 x 250):
  - dynamic agents: anisotropic Gaussian lobes that grow and lead with the
    agent's GT speed (comet-shaped for vehicles, isotropic for VRUs)
  - stopped vehicles (|GT disp@3s| < 0.5 m or no future): tight lobes
  - static world (occupancy GT obstacle/wall/pole/vegetation): distance
    falloff exp(-d / 1.2 m)
  - saturating combination  risk = 1 - exp(-1.6 * sum(intensity))
  - ego-proximity emphasis  * exp(-d_ego / 30 m); ego footprint zeroed
"""
import cv2
import numpy as np

XH, YH, RES = 40.0, 25.0, 0.2
RH, RW = int(2 * XH / RES), int(2 * YH / RES)          # 400 x 250
SEV_VRU, SEV_VEH, SEV_PARK = 1.0, 0.75, 0.55
STATIC_RISK = {1: 0.5, 8: 0.5, 9: 0.5, 7: 0.3}         # occ class -> amp

_GX = XH - (np.arange(RH, dtype=np.float32) + 0.5) * RES
_GY = YH - (np.arange(RW, dtype=np.float32) + 0.5) * RES
GX = np.repeat(_GX[:, None], RW, 1)
GY = np.repeat(_GY[None, :], RH, 0)


def gauss_lobe(x, y, yaw, s_long, s_lat, amp, lead=0.0):
    cx = x + lead * np.cos(yaw)
    cy = y + lead * np.sin(yaw)
    dx, dy = GX - cx, GY - cy
    c, s = np.cos(yaw), np.sin(yaw)
    u = c * dx + s * dy
    v = -s * dx + c * dy
    return amp * np.exp(-0.5 * ((u / s_long) ** 2 + (v / s_lat) ** 2))


def risk_field(boxes, count, traj, tvalid, occ_arr):
    """boxes [K,6](cls,xe,ye,l,w,yaw), traj [K,6,2], tvalid [K,6],
    occ_arr [16,200,200] uint8 or None -> risk [400,250] float32 in [0,1]."""
    inten = np.zeros((RH, RW), np.float32)
    if occ_arr is not None:
        stat = np.zeros((200, 200), np.float32)
        for cls_id, rv in STATIC_RISK.items():
            stat = np.maximum(stat, rv * (occ_arr == cls_id).any(0))
        up = cv2.resize(stat, (400, 400), interpolation=cv2.INTER_NEAREST)
        c0 = int((40.0 - YH) / 0.2)
        stat = up[:, c0:c0 + RW]
        d = cv2.distanceTransform((stat < 0.05).astype(np.uint8),
                                  cv2.DIST_L2, 3) * RES
        amp = cv2.dilate(stat, np.ones((9, 9), np.uint8))
        inten += amp * np.exp(-d / 1.2)
    for k in range(int(count)):
        cls, xe, ye, l, w, yaw = boxes[k]
        if l <= 0 or abs(xe) > XH + 10 or abs(ye) > YH + 10:
            continue
        tj, tv = traj[k], tvalid[k]
        vru = cls >= 1.5
        v = 0.0
        hd = float(yaw)
        if tv[2] > 0.5:
            v = float(np.linalg.norm(tj[2])) / 1.5
            if np.linalg.norm(tj[2]) > 0.4:
                hd = float(np.arctan2(tj[2, 1], tj[2, 0]))
        if vru:
            sig = 1.3 + 0.8 * v
            inten += gauss_lobe(xe, ye, hd, sig + 0.6 * v, sig,
                                SEV_VRU, lead=0.5 * v)
        elif v < 0.35:
            inten += gauss_lobe(xe, ye, yaw, l * 0.55, w * 0.7, SEV_PARK)
        else:
            inten += gauss_lobe(xe, ye, hd, l * 0.6 + 1.2 * v, w * 0.7 + 0.3,
                                SEV_VEH, lead=0.75 * v)
    risk = 1.0 - np.exp(-1.6 * inten)
    risk *= np.exp(-np.sqrt(GX ** 2 + GY ** 2) / 30.0)
    r0 = int((XH - 3.8) / RES)
    r1 = int((XH + 1.2) / RES)
    c0 = int((YH - 1.1) / RES)
    c1 = int((YH + 1.1) / RES)
    risk[max(r0, 0):r1, max(c0, 0):c1] = 0.0
    return risk.astype(np.float32)

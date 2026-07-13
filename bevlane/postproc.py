"""Post-processing helpers for BEV predictions."""
import cv2
import numpy as np

# ego-frame geometry of the BEV raster (row 0 = +80 m front, col 0 = +50 m left)
BEV_H, BEV_W = 800, 500
BEV_XH, BEV_YH, BEV_RES = 80.0, 50.0, 0.2
ROAD_DRV = (1, 3, 4, 5)          # road / crosswalk / laneline / stopline
ROAD_EDGE = 6


def thin_road_edge(pred):
    """Keep only the innermost (road-facing) 1-px of the road_edge band.

    A road_edge pixel is kept iff it is 4-adjacent to the road/drivable
    region; the rest of the (thick) band is reassigned to background.
    """
    pred = pred.copy()
    edge = (pred == ROAD_EDGE).astype(np.uint8)
    if edge.sum() == 0:
        return pred
    drv = np.isin(pred, ROAD_DRV).astype(np.uint8)
    drv_dil = cv2.dilate(drv, np.ones((3, 3), np.uint8))    # 1-px ring into edge
    inner = edge & (drv_dil > 0) & (drv == 0)               # edge touching road
    pred[edge > 0] = 0                                       # drop whole band
    pred[inner > 0] = ROAD_EDGE                              # restore innermost
    return pred


def crop_bev(pred, xh_m=60.0, yh_m=25.0):
    """Crop the full 800x500 BEV to +-xh_m longitudinal x +-yh_m lateral,
    centred on ego. Reduces blank space when the road is narrow."""
    r0 = int((BEV_XH - xh_m) / BEV_RES)
    r1 = int((BEV_XH + xh_m) / BEV_RES)
    c0 = int((BEV_YH - yh_m) / BEV_RES)
    c1 = int((BEV_YH + yh_m) / BEV_RES)
    return pred[r0:r1, c0:c1]


def draw_ego_and_grid(bev_bgr, out_h, out_w, xh_m=80.0, yh_m=50.0,
                      ring_long=(20, 40, 60, 80), ring_lat=(25, 50)):
    """Resize a BEV colour image and overlay ego icon + distance grid/labels.

    bev_bgr is the palette image (row 0 = front). xh_m / yh_m are the metric
    half-extents that bev_bgr spans (longitudinal / lateral). Returns
    (out_h, out_w, 3) uint8.
    """
    img = cv2.resize(bev_bgr, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    cx, cy = out_w // 2, out_h // 2
    sx = out_w / (2 * yh_m)   # px per metre, lateral
    sy = out_h / (2 * xh_m)   # px per metre, longitudinal
    grid = (60, 60, 60)
    # longitudinal range lines (front/back)
    for d in ring_long:
        if d > xh_m:
            continue
        for sign in (1, -1):
            y = int(cy - sign * d * sy)
            if 0 <= y < out_h:
                cv2.line(img, (0, y), (out_w, y), grid, 1, cv2.LINE_AA)
                cv2.putText(img, f"{d}m", (4, y - 3), cv2.FONT_HERSHEY_SIMPLEX,
                            0.42, (170, 170, 170), 1, cv2.LINE_AA)
    # lateral range lines (left/right)
    for d in ring_lat:
        if d > yh_m:
            continue
        for sign in (1, -1):
            x = int(cx + sign * d * sx)
            if 0 <= x < out_w:
                cv2.line(img, (x, 0), (x, out_h), grid, 1, cv2.LINE_AA)
                cv2.putText(img, f"{d}m", (x + 3, out_h - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 170, 170), 1, cv2.LINE_AA)
    # centre axes
    cv2.line(img, (cx, 0), (cx, out_h), (90, 90, 90), 1, cv2.LINE_AA)
    cv2.line(img, (0, cy), (out_w, cy), (90, 90, 90), 1, cv2.LINE_AA)
    # ego icon (triangle pointing up = forward)
    tri = np.array([[cx, cy - 13], [cx - 8, cy + 9], [cx + 8, cy + 9]], np.int32)
    cv2.fillPoly(img, [tri], (0, 235, 0))
    cv2.polylines(img, [tri], True, (255, 255, 255), 1, cv2.LINE_AA)
    return img

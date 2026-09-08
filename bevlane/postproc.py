"""Post-processing helpers for BEV predictions."""
import cv2
import numpy as np

# ego-frame geometry of the BEV raster (row 0 = +80 m front, col 0 = +50 m left)
BEV_H, BEV_W = 800, 500
# Geometry comes from the model module so the METEOR_BEV_XF / _XR overrides
# reach the drawing code too; a display that assumes the symmetric default puts
# the ego icon 30 m off on a rear-truncated grid.
from bevlane.model import BEV_XF, BEV_XR, BEV_YH               # noqa: E402
BEV_XH, BEV_RES = BEV_XF, 0.2
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


def crop_bev(pred, xh_m=60.0, yh_m=25.0, xr_m=None):
    """Crop the BEV to xh_m ahead / xr_m behind x +-yh_m lateral.

    Row 0 of `pred` is BEV_XF metres AHEAD of the ego, and on a rear-truncated
    grid the ego is NOT at the tensor centre -- computing both row bounds from
    one symmetric extent put the display window 30 m ahead of the vehicle and
    drew the ego icon on the wrong row. Bounds are clamped, so asking for more
    rear than the grid carries just shows what exists.
    """
    xr_m = xh_m if xr_m is None else xr_m
    h = pred.shape[0]
    res = (BEV_XF + BEV_XR) / h                 # rows may be a coarser grid
    r0 = max(0, int((BEV_XF - xh_m) / res))
    r1 = min(h, int((BEV_XF + xr_m) / res))
    w = pred.shape[1]
    resw = 2 * BEV_YH / w
    c0 = max(0, int((BEV_YH - yh_m) / resw))
    c1 = min(w, int((BEV_YH + yh_m) / resw))
    return pred[r0:r1, c0:c1]


def draw_ego_and_grid(bev_bgr, out_h, out_w, xh_m=80.0, yh_m=50.0,
                      ring_long=(20, 40, 60, 80), ring_lat=(25, 50)):
    """Resize a BEV colour image and overlay ego icon + distance grid/labels.

    bev_bgr is the palette image (row 0 = front). xh_m / yh_m are the metric
    half-extents that bev_bgr spans (longitudinal / lateral). Returns
    (out_h, out_w, 3) uint8.
    """
    img = cv2.resize(bev_bgr, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    # xh_m is the FORWARD extent of the image; xr_m (attribute set by callers
    # on truncated grids, default symmetric) the rearward one. The ego row is
    # where x = 0, which is the centre only when the two are equal.
    xr_m = getattr(draw_ego_and_grid, "xr_m", None)
    xr_m = xh_m if xr_m is None else xr_m
    cx = out_w // 2
    sy = out_h / (xh_m + xr_m)   # px per metre, longitudinal
    sx = out_w / (2 * yh_m)      # px per metre, lateral
    cy = int(xh_m * sy)          # ego row: x = 0
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

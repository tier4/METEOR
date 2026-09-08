"""cube_render_fast の自己完結版 (Orin 用)。

bevlane パッケージは __init__ が torch を import するため、torch の無い
Orin システム python では読めない。OCC パレットごとここへ切り出した
(元: bevlane/extract_occ.py OCC_PAL / bevlane/demo_occ_gt.py cube_render_fast)。
"""
import cv2
import numpy as np

OCC_PAL = np.array([[0, 0, 0], [244, 244, 244], [0, 0, 255], [119, 11, 32],
                    [220, 20, 60], [128, 64, 128], [244, 35, 232],
                    [107, 142, 35], [70, 70, 70], [220, 220, 0]], np.uint8)


def cube_render_fast(occ, W=900, H=760, rng_m=24.0, drop=(8,), zmax_m=3.0):
    """Same picture as cube_render, drawn in batches instead of one voxel at a
    time.

    cube_render issues four cv2 calls per voxel inside a Python loop; on a real
    occupancy volume that is well over ten thousand calls and 385 ms, and being
    Python it holds the GIL, so in a threaded pipeline it starves every other
    stage. The painter's order it needs is only over DEPTH (far to near, low to
    high) -- voxels that share a diagonal r+c and a height never overlap each
    other, so they can be filled together. Grouping consecutive same-colour
    runs in that order keeps the image identical and cuts the call count by
    more than an order of magnitude.
    """
    img = np.zeros((H, W, 3), np.uint8)
    occ = np.asarray(occ)
    n = min(int(rng_m / 0.4), occ.shape[1] // 2)
    r0 = occ.shape[1] // 2 - n
    zmax = min(int((zmax_m + 1.0) / 0.4), occ.shape[0])
    occ = occ[:zmax, r0:r0 + 2 * n, r0:r0 + 2 * n]
    su = W / (3.0 * n)
    a, b = su * 0.75, su * 1.15 * 0.375
    sz = su * 0.9
    v0 = H * 0.14

    def pt(r, c, z):
        return (int((c - r) * a + W // 2), int((c + r) * b - z * sz + v0))

    def pts(r, c, z):                       # vectorised pt()
        return np.stack([((c - r) * a + W // 2).astype(np.int32),
                         ((c + r) * b - z * sz + v0).astype(np.int32)], -1)

    gcol = (60, 60, 60)
    for g in range(0, 2 * n + 1, 10):
        cv2.line(img, pt(g, 0, 0), pt(g, 2 * n, 0), gcol, 1, cv2.LINE_AA)
        cv2.line(img, pt(0, g, 0), pt(2 * n, g, 0), gcol, 1, cv2.LINE_AA)

    FLAT = (5, 6)
    keep = (occ > 0) & (occ != 255) & ~np.isin(occ, drop)
    flat_m = keep & np.isin(occ, FLAT)
    cube_m = keep & ~np.isin(occ, FLAT)

    def runs(order, key):
        """-> [(start, stop)] of consecutive equal keys, in painter order."""
        k = key[order]
        cut = np.nonzero(np.r_[True, k[1:] != k[:-1]])[0]
        return list(zip(cut, np.r_[cut[1:], len(order)]))

    zz, rr, cc = np.nonzero(flat_m)
    if len(zz):
        order = np.argsort(rr + cc, kind="stable")
        cls = occ[zz, rr, cc]
        quad = np.stack([pts(rr, cc, 0), pts(rr + 1, cc, 0),
                         pts(rr + 1, cc + 1, 0), pts(rr, cc + 1, 0)], 1)
        for s0, s1 in runs(order, cls):
            idx = order[s0:s1]
            col = (OCC_PAL[cls[idx[0]]][::-1] * 0.55).astype(np.uint8).tolist()
            cv2.fillPoly(img, list(quad[idx]), col)

    zz, rr, cc = np.nonzero(cube_m)
    if len(zz):
        cls = occ[zz, rr, cc]
        order = np.argsort((rr + cc) * (occ.shape[0] + 1) + zz, kind="stable")
        t00, t10 = pts(rr, cc, zz + 1), pts(rr + 1, cc, zz + 1)
        t11, t01 = pts(rr + 1, cc + 1, zz + 1), pts(rr, cc + 1, zz + 1)
        b10, b11 = pts(rr + 1, cc, zz), pts(rr + 1, cc + 1, zz)
        b01 = pts(rr, cc + 1, zz)
        left_q = np.stack([t10, t11, b11, b10], 1)
        right_q = np.stack([t01, t11, b11, b01], 1)
        top_q = np.stack([t00, t10, t11, t01], 1)
        key = cls.astype(np.int64) * (occ.shape[0] + 1) + zz   # colour + shade
        for s0, s1 in runs(order, key):
            idx = order[s0:s1]
            k0 = idx[0]
            base = OCC_PAL[cls[k0]][::-1].astype(np.float32)
            shade = 0.6 + 0.4 * zz[k0] / max(zmax - 1, 1)
            top = np.clip(base * shade, 0, 255).astype(np.uint8).tolist()
            left = np.clip(base * shade * 0.55, 0, 255
                           ).astype(np.uint8).tolist()
            right = np.clip(base * shade * 0.75, 0, 255
                            ).astype(np.uint8).tolist()
            cv2.fillPoly(img, list(left_q[idx]), left)
            cv2.fillPoly(img, list(right_q[idx]), right)
            cv2.fillPoly(img, list(top_q[idx]), top)
            cv2.polylines(img, list(top_q[idx]), True,
                          tuple(int(v * 0.45) for v in top), 1)
    cv2.drawMarker(img, pt(n, n, 0), (0, 255, 0),
                   cv2.MARKER_TRIANGLE_UP, 16, 2)
    return img


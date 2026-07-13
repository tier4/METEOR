#!/usr/bin/env python3
"""Merge per-scene BEV count grids (same global map frame) into one map."""
import argparse
import json
import os

import cv2
import numpy as np

from autolabel_bev import CLASS_NAMES, PALETTE, rasterize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="scene output dirs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mask-dist", type=float, default=20.0)
    args = ap.parse_args()

    metas, datas = [], []
    for d in args.inputs:
        meta = json.load(open(os.path.join(d, "meta.json")))
        z = np.load(os.path.join(d, "bev_counts.npz"))
        metas.append(meta)
        datas.append((z["counts"], z["min_dist"]))

    res = metas[0]["resolution"]
    assert all(abs(m["resolution"] - res) < 1e-9 for m in metas)
    x0 = min(m["origin"][0] for m in metas)
    y0 = min(m["origin"][1] for m in metas)
    x1 = max(m["origin"][0] + m["size"][1] * res for m in metas)
    y1 = max(m["origin"][1] + m["size"][0] * res for m in metas)
    W, H = int(round((x1 - x0) / res)), int(round((y1 - y0) / res))
    print(f"[merged grid] {W} x {H} @ {res} m")

    counts = np.zeros((datas[0][0].shape[0], H, W), dtype=np.uint32)
    min_dist = np.full((H, W), np.inf, dtype=np.float32)
    for meta, (c, md) in zip(metas, datas):
        ox = int(round((meta["origin"][0] - x0) / res))
        oy = int(round((meta["origin"][1] - y0) / res))
        h, w = meta["size"]
        counts[:, oy:oy + h, ox:ox + w] += c
        np.minimum(min_dist[oy:oy + h, ox:ox + w], md,
                   out=min_dist[oy:oy + h, ox:ox + w])

    os.makedirs(args.out, exist_ok=True)
    bev = rasterize(np.minimum(counts, 65535).astype(np.uint16))
    bev_m = np.where(min_dist <= args.mask_dist, bev, 0).astype(np.uint8)
    np.save(os.path.join(args.out, "bev_label.npy"), bev)
    np.save(os.path.join(args.out, "bev_label_masked.npy"), bev_m)
    json.dump(dict(origin=[x0, y0], resolution=res, size=[H, W],
                   classes=CLASS_NAMES, merged_from=[m["scene"] for m in metas]),
              open(os.path.join(args.out, "meta.json"), "w"), indent=2)
    for nm, arr in [("bev_label.png", bev), ("bev_label_masked.png", bev_m)]:
        vis = np.ascontiguousarray(PALETTE[arr][::-1])
        cv2.imwrite(os.path.join(args.out, nm), vis[:, :, ::-1])
    print(f"[done] {args.out}")


if __name__ == "__main__":
    main()

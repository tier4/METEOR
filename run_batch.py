#!/usr/bin/env python3
"""Batch-run BEV autolabel over multiple scenes."""
import argparse
import os
import traceback

import cv2

from autolabel_bev import process_scene

ROOT = os.environ.get("BEVLANE_ROOT",
                      "data/t4dataset")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", required=True, help="comma list or file with one scene per line")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--stride", type=int, default=1)
    args = ap.parse_args()

    if os.path.isfile(args.scenes):
        scenes = [l.split()[0] for l in open(args.scenes) if l.strip()]
    else:
        scenes = args.scenes.split(",")

    for name in scenes:
        out = os.path.join(args.out, name)
        if os.path.exists(os.path.join(out, "bev_label.png")):
            print(f"[skip] {name}")
            continue
        try:
            process_scene(os.path.join(ROOT, name), out,
                          stride=args.stride, workers=args.workers)
            img = cv2.imread(os.path.join(out, "bev_label_masked.png"))
            s = 1200.0 / max(img.shape)
            if s < 1:
                img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(os.path.join(out, "preview.png"), img)
        except Exception:
            print(f"[fail] {name}")
            traceback.print_exc()


if __name__ == "__main__":
    main()

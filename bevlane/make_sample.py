#!/usr/bin/env python3
"""Build a small, self-contained sample dataset for trying inference.

Inference needs only four things per scene:

    <scene>/manifest.json     camera intrinsics/extrinsics + the frame index
    <scene>/img/*.jpg         the camera images
    <scene>/ego_motion.npz    v0 (speed) and the ego poses the temporal
                              memory needs
    <scene>/gt_vec/*.png      only because the demo loads a GT layer for the
                              side-by-side panel; inference itself ignores it

Everything else in a converted scene (depth, occupancy, boxes, LiDAR rasters,
SD-map, ...) is TRAINING ground truth and is not copied.

The sample contains two 8-camera DRS scenes from the held-out validation
recording. For the seven-camera configuration, pass
`--zero-cams CAM_BACK_NARROW` to the demo: that is bit-identical to a recording
that does not carry the camera (bevlane/probe_cam_config.py, check C).

    python3 bevlane/make_sample.py --frames 40 --out sample
"""
import argparse
import json
import os
import shutil


def pick(root, lst, want, exclude=()):
    """First scene from `lst` that exists locally and has enough frames."""
    for line in open(lst):
        s = line.strip()
        if not s or s in exclude:
            continue
        mf = os.path.join(root, s, "manifest.json")
        if not os.path.exists(mf):
            continue
        try:
            m = json.load(open(mf))
        except Exception:
            continue
        if len(m.get("frames", [])) >= want and len(m.get("cams", {})) >= 7:
            return s, m
    return None, None


def copy_scene(root, out, scene, man, n_frames):
    dst = os.path.join(out, scene)
    os.makedirs(os.path.join(dst, "img"), exist_ok=True)
    os.makedirs(os.path.join(dst, "gt_vec"), exist_ok=True)
    frames = man["frames"][:n_frames]
    kept = []
    for f in frames:
        ok = True
        for c, rel in f.get("imgs", {}).items():
            src = os.path.join(root, scene, rel)
            if not os.path.exists(src):
                ok = False
                break
        if not ok:
            continue
        g = f.get("gt_vec")
        if g and os.path.exists(os.path.join(root, scene, g)):
            shutil.copy2(os.path.join(root, scene, g),
                         os.path.join(dst, g))
        else:
            continue                      # the demo needs the GT layer
        for c, rel in f["imgs"].items():
            shutil.copy2(os.path.join(root, scene, rel),
                         os.path.join(dst, rel))
        kept.append({k: v for k, v in f.items()
                     if k in ("frame", "imgs", "gt_vec")})
    ego = os.path.join(root, scene, "ego_motion.npz")
    if os.path.exists(ego):
        shutil.copy2(ego, os.path.join(dst, "ego_motion.npz"))
    small = {k: v for k, v in man.items() if k != "frames"}
    small["frames"] = kept
    small["ego_motion"] = "ego_motion.npz"
    json.dump(small, open(os.path.join(dst, "manifest.json"), "w"))
    mb = sum(os.path.getsize(os.path.join(dp, f_))
             for dp, _, fs in os.walk(dst) for f_ in fs) / 2**20
    print(f"  {scene[:56]:56s} {len(kept):3d} frames  "
          f"{len(man['cams'])} cams  {mb:6.1f} MiB")
    return len(kept), mb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="out/bevlane")
    ap.add_argument("--out", default="sample")
    ap.add_argument("--frames", type=int, default=40)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    total_f = total_mb = 0
    print("copying scenes (images + poses + one GT layer only):")

    # 8-camera Japanese validation scene
    s8, m8 = pick(a.root, "val.lst", a.frames)
    if s8:
        f, mb = copy_scene(a.root, a.out, s8, m8, a.frames)
        total_f += f
        total_mb += mb
    # A second 8-camera scene from the same held-out recording. The sample used
    # to include a 7-camera J6Gen2/x2gen2 scene to exercise the camera-count
    # path; that corpus is not for external distribution, so the sample is
    # DRS-only. Use --zero-cams CAM_BACK_NARROW on these scenes to try the
    # seven-camera configuration instead (measured bit-identical to a recording
    # that genuinely lacks the camera).
    s7 = m7 = None
    if s8:
        s7, m7 = pick(a.root, "val.lst", a.frames, exclude=(s8,))
    if s7:
        f, mb = copy_scene(a.root, a.out, s7, m7, a.frames)
        total_f += f
        total_mb += mb

    names = [x for x in (s8, s7) if x]
    open(os.path.join(a.out, "scenes.txt"), "w").write("\n".join(names) + "\n")
    print(f"\n{a.out}/: {len(names)} scenes, {total_f} frames, "
          f"{total_mb:.0f} MiB total")
    print(f"scene list -> {a.out}/scenes.txt")


if __name__ == "__main__":
    main()

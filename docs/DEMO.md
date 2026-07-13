# Demo Tooling

All demo renderers are GT-free at inference time (except the dedicated GT
visualisers) and write 1920×1080 H.264 mp4.

## Multi-task inference demo — `bevlane/demo_rgbd_bev.py`

```bash
python3 bevlane/demo_rgbd_bev.py --model v20 --n-seg2d 21 --thresh2d 0.25 \
  --ckpt out/ckpt/best.pt --show-seg2d \
  --scenes <SCENE1> <SCENE2> ... --out out/demo.mp4
```

Layout:

- **Top 2×4** — 8 camera RGB with: 21-class 2D-seg overlay (CSV palette),
  10-class 2D boxes (class abbreviation + score), projected 3D wireframes with
  distance labels (near-plane-clipped so nearby side-camera vehicles still
  draw), and the **E2E path ribbon** on CAM_FRONT_WIDE — a vehicle-width
  ground polygon that follows the predicted trajectory, retracts smoothly as
  the car stops (length tracks predicted travel; EMA-smoothed between frames).
- **Middle 2×4** — calibrated metric depth (0–80 m, turbo).
- **Bottom-left** — predicted 3D occupancy as isometric voxels (per-class
  confidence gates suppress unknown-region hallucination while the dense
  free-space GT converges).
- **Right column** — BEV lane map (±25 × ±60 m crop) with oriented boxes,
  the E2E trajectory (green waypoints; HOLD marker when stationary), and
  v0 / steering / accel / brake gauges.

Useful flags: `--thresh2d` (2D det score), `--no-thin` (disable road-edge
thinning), `--infer-hw` (override inference resolution).

## Ground-truth visualisers

```bash
# all 6 image/BEV supervision signals in one video
python3 bevlane/demo_gt_full.py --scenes <SCENE ...> --out out/demo_gt.mp4

# occupancy GT: front RGB | top-down | isometric 3D voxels
python3 bevlane/demo_occ_gt.py --scenes <SCENE ...> --out out/demo_occ_gt.mp4

# 200-scene BEV GT survey montage (~10 frames/scene)
python3 bevlane/montage_bev_gt.py --scenes list.txt --out out/montage.mp4
```

The GT visualiser annotates scene-end frames where the 3 s E2E future does not
exist ("E2E GT: none") so trajectory gaps read as by-design.

## Architecture slides

`build_pptx_v18arch.py` regenerates the architecture deck (overview, detailed
image/BEV branches with tensor shapes, losses, measured params/FLOPs tables).

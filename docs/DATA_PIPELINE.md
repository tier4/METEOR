# The Autolabel Factory

Every METEOR supervision signal is generated automatically from t4dataset-format
recordings (NuScenes-like: `sample_data`, `ego_pose`, `calibrated_sensor`,
LiDAR sweeps, 2D panoptic `surface_ann`/`object_ann` as base64 COCO RLE,
3D `sample_annotation`). **No human labels are used.**

`bevlane/convert_dtset.py` drives all stages per scene (scene-parallel,
every stage skips existing outputs → fully resumable):

| # | Stage | Output |
|---|---|---|
| 1 | `run_batch.py` (autolabel production) | map-frame BEV class counts |
| 2 | `vectorize_bev.py` | connected lane polylines (`vector_map.json`) |
| 3 | `extract_gt.py` | cached 768×432 images + per-frame BEV crops |
| 4 | `add_narrow_cams.py` | tele-camera cache (mixed resolutions) |
| 5 | `render_vector_gt.py` | hybrid raster+vector lane GT (`gt_vec`) |
| 6 | `extract_depth_dense.py` / `_narrow.py` | dense metric depth, 8 cams |
| 7 | `extract_bev_box.py` | camera-confirmed oriented 3D boxes |
| 8 | `annotate_gtcov.py` | per-frame GT-coverage stats (training filter) |
| 9 | `extract_seg2d.py` | 21-class 2D segmentation (`seg2d21`) |
| 10 | `extract_bbox2d.py` + `extract_ego.py` + `extract_occ.py` | 2D boxes, E2E, occupancy |

## BEV lanes (`gt_vec`)

1. **Point labeling** — every LiDAR keyframe is projected into all cameras
   (pinhole + equidistant fisheye), sampling road-surface classes from the
   panoptic masks. Thin classes (lane/stop/edge/marking) only within 15 m
   (pixel→ground error grows with range).
2. **Map-frame accumulation** — ego_pose transforms points to the world frame;
   class hit-counts accumulate on a 0.1 m grid across the whole scene.
3. **Rasterisation** — area classes by argmax; thin classes by hit-count
   priority override; a `min_dist ≤ 20 m` masked variant kills far smear.
4. **Vectorisation** — skeletonise → trace → spur-prune → Douglas-Peucker →
   dash-gap linking (8 m for lane lines) → `vector_map.json` polylines in
   map-frame metres.
5. **Hybrid re-render** — per-frame ego crops (±80 × ±50 m @ 0.2 m): area
   classes from the raster, lines re-drawn from vectors at fixed width;
   enclosed road holes ≤ 400 cells filled; opposing carriageways removed by an
   ego-connected component filter **with a 30 % revert guard** (wide
   intersections would otherwise be wiped); `road_edge` recomputed as the
   pixel-accurate drivable boundary.

## Metric depth

LiDAR splat (min-depth per pixel) → `griddata` linear interpolation over the
**merged road+paint panoptic segment** (treating paint separately leaves holes
around lane lines) → geometric ground-plane fill for downward rays within 50°
(fisheye-pinhole guard) → same-segment EDT fill → segment median → sky 79.5 m,
ego 2.0 m. Stride-4 (108×192), 64 bins × 1.25 m.

## 2D segmentation (21 classes)

Taxonomy is **CSV-driven** (`comlops-21cls-autolabel-2504.csv`: id, name, RGB —
Cityscapes-like palette). Key recipe points:

- Thin classes (lane, marking, pole, sign, light) survive the 4× downsample via
  **coverage-based rasterisation**: a cell becomes the thin class if >12 % of
  its footprint is covered (naive nearest resize deletes sub-pixel lane lines;
  this recovered +49 % lane pixels).
- `ego_vehicle` is supervised as background 0 (ignore left the hood
  unconstrained → noise).
- All 8 cameras are labeled; 255 = ignore only where annotations are absent.

## 2D boxes (10 classes)

Taxonomy from `fastlabel_2510_instance.csv`. `object_ann` boxes scaled into the
768×432 cache, area-sorted, **KMAX = 96 per camera** (a KMAX of 32 silently
dropped 29 % of the annotations in crowded scenes — exactly the small objects:
cones, traffic lights, distant unknowns).

## 3D boxes (camera-confirmed)

LiDAR `sample_annotation` boxes are kept only when **geometrically confirmed by
a camera**: project the 8 corners into each camera and require overlap with a
same-class 2D annotation (inter / min-area > 0.3). 2D and 3D instance tokens
are disjoint in this dataset, so the match must be geometric. Occluded objects
never enter the BEV GT.

## E2E driving GT (no CAN needed)

From `ego_pose` alone (`extract_ego.py`):

- **Trajectory** — future poses at +0.5…+3.0 s transformed into the current
  ego frame (6 waypoints).
- **Speed / accel** — smoothed (±0.5 s) finite differences. Written for
  *every* frame (instantaneous signals need no future — a scene-tail v0 of 0
  once made the model predict "hold" at 80 km/h).
- **Steering** — bicycle model `atan(wheelbase · yaw_rate / v)`, masked below
  0.5 m/s.
- **Brake** — accel < −0.5 m/s².
- **valid = 0** where < 3 s of future exists (scene tails) — those frames
  still train all other tasks; the E2E loss is masked.

## 3D occupancy

Ego-frame voxels `[16, 200, 200]` @ 0.4 m (±40 m, z ∈ [−1, 5.4) m), classes:
free / obstacle / vehicle / two-wheeler / pedestrian / road / sidewalk /
vegetation / building / pole+sign, 255 = never observed.

- Points are labeled by sampling the **cached seg2d21 maps** (no RLE decode —
  a panoptic-decode variant cost 690 s/scene; this one ~45 s).
- ±8 strided frames accumulate via ego_pose; **dynamic classes come from ±1
  frames only** (sub-voxel smear, 3× the points of a single sweep — full
  accumulation drags moving cars across the map).
- **Free space** is carved by ray-stepping the current sweep at 0.4 m steps
  (coarser steps leave most above-road voxels unknown, which lets the model
  hallucinate structure there unpunished).

## Quality gates (training-time)

- **Scene trimming** — first 3 and last 10 frames dropped (weak forward GT).
- **GT coverage filter** — per-frame labeled fractions (`gtcov`): core band
  ±30 m ≥ 3 %, forward band 30–80 m ≥ 0.5 %. Long-stationary spots (no
  accumulated GT ahead) are excluded automatically.
- **Scene selection** — indoor/underground scenes are detected by panoptic
  `sky` counts (NDT pose drift makes speed-based filtering unreliable);
  pose-jump frames are discarded.

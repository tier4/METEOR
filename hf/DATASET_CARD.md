---
license: cc-by-4.0
pretty_name: METEOR demo scenes
language:
  - en
tags:
  - autoware
  - autonomous-driving
  - camera
  - multi-view
  - surround-view
  - lidar
  - demo-data
  - meteor
task_categories:
  - image-segmentation
  - object-detection
  - depth-estimation
size_categories:
  - 1K<n<10K
---

# METEOR demo scenes for Autoware (`meteor-demo-scenes`)

Six short driving scenes, one per road type (147–148 frames each, 8 synchronised cameras, ego motion, LiDAR raster) to run
the released **METEOR** model ([`AutowareFoundation/meteor`](https://huggingface.co/AutowareFoundation/meteor)) and the demo renderers of
**https://github.com/tier4/METEOR** without access to the training corpus. These are the exact scene
roots the Orin demos and benchmarks in the repository refer to (`valday`, `valcurve`, `fast`).

The scenes come from the **validation split** (a held-out recording day) of the METEOR corpus —
TIER IV Co-MLOps Data Recording System (DRS) recordings on public roads in Japan, 2026-01-23. They are
not training scenes. No ground truth is included; the model runs GT-free.

| Root | Scenes | Frames | Content | Size |
|---|---|---|---|---|
| `highway_day/` | 1 | 147 | daytime elevated expressway, straight, ~90 km/h | 110 MB |
| `mountain_day/` | 1 | 147 | daytime winding mountain road, guard rails, forest | 137 MB |
| `arterial_day/` | 1 | 148 | daytime multi-lane urban arterial, straight-ahead | 132 MB |
| `valday/` | 1 | 147 | daytime urban, intersection and curves (the default demo root) | 121 MB |
| `valcurve/` | 1 | 148 | night, elevated expressway with tight curves | 84 MB |
| `fast/` | 1 | 147 | dusk expressway, used for `--bench` latency runs | 88 MB |

`arterial_day`, `valday`, `valcurve` and `fast` come from the **validation split** (a held-out
recording day, 2026-01-23); `highway_day` and `mountain_day` come from **training recordings**
(October 2025) because the held-out day has no daytime expressway or mountain driving. The model
has seen those two roots during training; treat them as illustration, not as a test set.

## Anonymisation

Every image was regenerated from the anonymised camera stream of the source dataset, in which
**faces and licence plates are blurred** before any use. `manifest.json` carries `"anonymized": 1`.
No un-anonymised pixel is part of this repository.

## Layout

```
<root>/
├── scenes.txt                          scene names, one per line
└── <scene>/
    ├── manifest.json                   camera intrinsics / extrinsics + per-frame file table
    ├── img/<frame>_<CAM_NAME>.jpg      768×432 JPEG, 8 cameras per frame (anonymised)
    ├── ego_motion.npz                  v0 (speed, m/s), pose (x, y, yaw), future waypoints, controls
    └── lidar_bev/<frame>.npz           key "lb": float16 pillar raster [4, 400, 250] @ 0.4 m (optional model input, cast to float32)
```

Camera names: `CAM_FRONT_WIDE, CAM_FRONT_LEFT, CAM_FRONT_RIGHT, CAM_BACK_WIDE, CAM_BACK_LEFT,
CAM_BACK_RIGHT, CAM_FRONT_NARROW, CAM_BACK_NARROW`.

`manifest.json` keys that matter for inference: `cams[<CAM>].K` (3×3 at 768×432),
`cams[<CAM>].T_ego_cam` (4×4, camera → ego; the model takes its inverse), `frames[i].imgs[<CAM>]`,
`frames[i].lidar_bev`. The manifests also list the ground-truth files of the full training format
(`gt/`, `occ/`, `seg2d/`, …); those directories are intentionally **not** shipped here and the
runtimes do not read them.

## Use

```bash
pip install -U huggingface_hub
hf download AutowareFoundation/meteor-demo-scenes --repo-type dataset --local-dir demo

# onnxruntime, one frame (repo: https://github.com/tier4/METEOR)
python3 hf/onnx_smoke_test.py --onnx meteor_v157c3Z.onnx --root demo/valday --frame 40

# Jetson AGX Orin, real-time demo (C++ runtime + the released INT8 engine)
meteor_realtime --engine eng/v157c3Zg_int8.engine --root demo/valday --out out/demo.mp4
meteor_realtime --engine eng/v157c3Zg_int8.engine --root demo/fast --bench 40
```

The Python and C++ runtimes in the repository (`deploy/orin_realtime.py`, `deploy/cpp`) take any of
these roots via `--root`.

## Notes

- Public-road imagery: faces and licence plates are **not** blurred. Use for evaluating and
  demonstrating the model only; do not redistribute frames out of this context.
- Ego speed comes from the vehicle's own odometry; the LiDAR raster is the same pillar raster used for
  the optional LiDAR input of the model (zeros when the input is not available).
- `SHA256SUMS` covers every file.

## License

Follows the license of the METEOR repository, which has not been finalised yet; until then this data
is provided for **research and demonstration use only**.

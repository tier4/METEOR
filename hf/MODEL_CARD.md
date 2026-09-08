---
license: apache-2.0
pipeline_tag: robotics
tags:
  - autoware
  - autonomous-driving
  - camera
  - multi-view
  - bird-eye-view
  - multi-task
  - e2e
  - planning
  - 3d-object-detection
  - semantic-segmentation
  - depth-estimation
  - jetson-orin
  - meteor
  - tensorrt
  - onnx
---

# METEOR for Autoware (`meteor`)

Surround-view **multi-task driving network**: 8 cameras of the TIER IV Co-MLOps Data Recording
System (DRS) plus calibration and ego speed in, twelve driving tasks out of one 54M-parameter
network, running on a Jetson AGX Orin in about 70 ms (INT8, 2:4 sparse trunk).

Code, training recipe, autolabel factory and the Python / C++ TensorRT runtimes:
**<https://github.com/tier4/METEOR>**. Demo scenes to run this model on:
[`AutowareFoundation/meteor-demo-scenes`](https://huggingface.co/datasets/AutowareFoundation/meteor-demo-scenes).

METEOR is published as a **reference model with its own runtimes**; an Autoware (ROS 2) node
does not exist yet. The model was trained entirely on auto-generated ground truth (zero human
labels), and the project — model, GT extractors, trainer, runtimes, CUDA plugin, this card — was
written by Claude (Anthropic) operating autonomously with humans setting goals and reviewing results.

## Model overview

| | |
| --- | --- |
| Task | 12 tasks from one forward pass: BEV lane segmentation (9 cls), metric depth, 3D oriented boxes (vehicle / VRU) with parked/stopped flag, unknown obstacles, 2D semantic segmentation (21 cls), 2D detection (10 cls), multimodal end-to-end trajectory (K=3, 3 s) + steer/accel/brake, 3D semantic occupancy, occupancy flow, agent forecasting, ego-relevant traffic-light state, area risk field |
| Architecture | ResNet-34 + FPN image backbone → depth-gated IPM lift → one 96-ch BEV feature (800×500 @ 0.2 m, ±80 m × ±50 m) → task heads + per-task residual refiners (`DepthSegIPMNetV52`) |
| Cameras | 8 views, fixed order: `CAM_FRONT_WIDE, CAM_FRONT_LEFT, CAM_FRONT_RIGHT, CAM_BACK_WIDE, CAM_BACK_LEFT, CAM_BACK_RIGHT, CAM_FRONT_NARROW, CAM_BACK_NARROW` |
| Input resolution | 768 × 432 per camera, RGB uint8 (normalisation inside the graph) |
| Optional inputs | LiDAR pillar raster and SD-map raster exist in the checkpoint (zero input is bit-equal to no input); this ONNX is the camera-only export |
| Temporal memory | Present in the checkpoint, **baked out** of the export: the deployed model is single-frame |
| Sparsity | 2:4 structured sparsity on the convolutional trunk (planner branches dense) |
| Runtime | TensorRT (INT8 on Orin, fp16 elsewhere) via `deploy/runtime.py`, `deploy/orin_realtime.py` or the C++ runtime in `deploy/cpp` |
| Latency | AGX Orin, INT8, CUDA Graph, zero-copy input: ~70 ms median per 8-camera frame; workstation fp16 (TensorRT 8.6, plugin-free, data-center GPU): ~30 ms |
| Format | PyTorch checkpoint + ONNX (opset 17, no custom ops). TensorRT engines are built locally |
| License | Apache-2.0 |

## Files

| File | Description |
| --- | --- |
| `meteor_v157c3Z.onnx` | Plain ONNX, camera-only, zero history baked out, uint8 image input, argmax outputs for lane / depth / seg2d. Runs in onnxruntime and any TensorRT ≥ 8.6 without a plugin |
| `meteor_v157.pt` | PyTorch checkpoint (`--model v52`, final-epoch raw weights, no optimizer state). Source of the ONNX; use it to fine-tune, probe or re-export (e.g. with the LiDAR input) |
| `meteor_v157.param.yaml` | Model-dependent parameters: camera order, input resolution, BEV grid, output tensors and their meaning, runtime defaults |
| `lift_plugin_tables_r64/` | Frustum tables for the optional CUDA lift plugin (`deploy/cpp/liftbench/plugin/make_plugin_onnx.py`), −4…5 ms on Orin |
| `deploy_metadata.yaml` | Deployment metadata recording the artifact version of this repository |
| `SHA256SUMS` | Checksums of every file in this repository |

> **TensorRT engines are not distributed here.** Engines are specific to the GPU architecture and
> TensorRT version they are built on. Build them from the ONNX: `deploy/build_engine_fp16.py` on a
> workstation (no plugin, no calibration), or `deploy/orin_build_int8.py` on a Jetson (real-frame INT8
> calibration, optional lift plugin) — see `deploy/README.md` in the repository.

## Inputs and outputs (ONNX)

**Inputs**

| Tensor | dtype / shape | Notes |
| --- | --- | --- |
| `imgs` | uint8 `[1, 8, 3, 432, 768]` | RGB 0..255, camera order as above |
| `K` | float32 `[1, 8, 3, 3]` | intrinsics at 768 × 432 |
| `T_cam_ego` | float32 `[1, 8, 4, 4]` | ego → camera transforms |
| `v0` | float32 `[1]` | ego speed, m/s |

**Outputs** (19 tensors; `meteor_v157.param.yaml` lists shape, dtype and meaning of each)

| Tensor | Notes |
| --- | --- |
| `lane`, `lane_logit` | 9-class BEV lane map `[1,800,500]` (uint8 argmax) and its logits |
| `depth`, `depth_mean` | 64-bin depth argmax per camera and the expected depth in metres |
| `seg2d` | 21-class 2D segmentation per camera |
| `hm`, `reg`, `stationary`, `traj` | 3D boxes (vehicle / VRU) at 0.4 m, parked flag, per-agent 3 s futures |
| `hm2d_s0..2`, `reg2d_s0..2` | 10-class 2D detection at three scales |
| `ego` | K=3 ego paths (6 × 0.5 s), confidences, steer / accel / brake |
| `occ`, `tl`, `risk` | 3D semantic occupancy, traffic-light state, area risk field |

Pre-processing (resize to 768 × 432, RGB) and post-processing (box decode, BEV rotated-box NMS,
2D thresholding, temporal fusion of the lane map) run in the runtime, not in the graph.
`hf/onnx_smoke_test.py` in the repository is the reference implementation of the input contract.

## Usage

```bash
pip install onnxruntime opencv-python numpy huggingface_hub
hf download AutowareFoundation/meteor meteor_v157c3Z.onnx --local-dir meteor
hf download AutowareFoundation/meteor-demo-scenes --repo-type dataset --include "valday/*" --local-dir demo
git clone https://github.com/tier4/METEOR && cd METEOR
python3 hf/onnx_smoke_test.py --onnx ../meteor/meteor_v157c3Z.onnx --root ../demo/valday --frame 40   # -> SMOKE PASS

# TensorRT on a workstation (verified: TensorRT 8.6, data-center GPU, ~10 min build, 194 MB plan)
python3 deploy/build_engine_fp16.py ../meteor/meteor_v157c3Z.onnx out/meteor_fp16.engine 8
METEOR_TH2D=0.30 PYTHONPATH=. python3 deploy/orin_realtime.py --engine out/meteor_fp16.engine \
    --root ../demo/valday --out out/demo.mp4
```

On a Jetson AGX Orin follow `deploy/README.md` §6–§7 (lift plugin, on-device INT8 calibration,
health checks); the C++ runtime `deploy/cpp` renders the same 12-task view at ~15 FPS.

## Training

Trained with the repository's `bevlane/train.py` on TIER IV Co-MLOps DRS recordings in Japan.
Every supervision signal comes from the autolabel factory in the repository (LiDAR, ego pose and
panoptic masks turned into BEV lanes, depth, boxes, occupancy, trajectories, traffic-light state,
risk); adverse conditions were added as NVIDIA Cosmos Transfer re-renderings of real scenes.
This checkpoint (v157) is a 2:4 sparse fine-tune of the dense baseline with gradual 1:4 → 2:4
pruning and matches the dense model on the closed-loop evaluation. Accuracy figures are measured
on an internal validation split and are not published (they would not be comparable to public
benchmarks); latency is reported above.

## Limitations

- **Sensor configuration.** Trained only on the 8-camera DRS rig at 768 × 432. Other rigs need
  re-training or at least the frustum tables re-derived from their calibration (`K` / `T_cam_ego`
  are inputs, so the plain ONNX runs, but accuracy is not guaranteed).
- **Single-frame.** The temporal memory was trained on a zero history and is baked out; the model
  does not use previous frames.
- **Domain.** Japanese roads (plus a US subset); the traffic-light head knows one lamp convention.
- **Research artefact.** Not a certified driving system; it must not control a vehicle on public
  roads. Outputs are meant for visualisation, evaluation and as a reference for integration.

## Provenance

| | |
| --- | --- |
| Original source | <https://github.com/tier4/METEOR> (`deploy/export_onnx.py` on this checkpoint, see `deploy/README.md` §1 for the exact flags) |
| This repository | `AutowareFoundation/meteor`, tag `v1.0` |

The ONNX was exported from `meteor_v157.pt` on 2026-09-07; its 420 weight tensors are byte-identical
to the checkpoint's. Re-exporting from the checkpoint reproduces it.

## Citation

```bibtex
@software{umeda2026meteor,
  author  = {Umeda, Dan},
  title   = {{METEOR}: Multi-task Estimation of Traffic Elements, Objects \& Roads},
  year    = {2026},
  url     = {https://github.com/tier4/METEOR},
  note    = {Surround-view multi-task driving network trained on auto-generated labels; NVIDIA GTC 2026 session S81897}
}
```

## References

- METEOR repository: <https://github.com/tier4/METEOR>
- Co-MLOps / CoMET autolabel platform: <https://co-mlops.tier4.jp/>
- CoMLOps dataset with NVIDIA Cosmos (TIER IV tech blog): <https://tier4.co.jp/en/updates/technology/20260807-comlops-dataset-foundation-for-autonomous-driving-with-nvidia-cosmos>
- NVIDIA GTC 2026 session S81897: <https://www.nvidia.com/ja-jp/gtc/session-catalog/sessions/gtc26-s81897/>
- Autoware: <https://github.com/autowarefoundation/autoware>

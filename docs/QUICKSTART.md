# Inference quickstart (from the published artefacts)

This page alone takes you from the published model and demo scenes to a 12-task inference video
on one GPU. Every step was run on a workstation (one data-center NVIDIA GPU, TensorRT 8.6);
the detailed record with timings is in [REPRODUCE.md §8](REPRODUCE.md#8-deployment--the-verified-release-path).

## 1. What is published

| | Where | Contents |
|---|---|---|
| Model | <https://huggingface.co/AutowareFoundation/meteor> | `meteor_v157c3Z.onnx` (plain ONNX, camera-only, no custom ops), `meteor_v157.pt` (PyTorch checkpoint), `meteor_v157.param.yaml` (camera order, resolution, BEV grid, output tensors), `lift_plugin_tables_r64/` (tables for the optional CUDA lift plugin on Orin) |
| Demo scenes | <https://huggingface.co/datasets/AutowareFoundation/meteor-demo-scenes> | six anonymised scenes (expressway, mountain road, arterial, city, night expressway, benchmark set). Per scene: `manifest.json`, `img/` (8 cameras × 147 frames, 768×432), `ego_motion.npz`, `lidar_bev/`. No ground truth |

Licenses: model Apache-2.0, demo scenes CC-BY-4.0. TensorRT engines are not distributed because they
depend on the GPU and the TensorRT version; build them from the ONNX.

## 2. Requirements

| Item | Requirement |
|---|---|
| GPU | NVIDIA; fp16 TensorRT inference needs about 4 GB of VRAM (the build asks for an 8 GB workspace) |
| Python | 3.10 |
| Packages | `huggingface_hub`, `onnxruntime`, `opencv-python`, `numpy`; for TensorRT inference `tensorrt` (8.6 or newer) and `pycuda`; for fine-tuning or re-export `torch` |
| Other | `ffmpeg` is handy for checking the video |

## 3. Steps

```bash
# 0. code and artefacts
git clone https://github.com/tier4/METEOR && cd METEOR
pip install -U huggingface_hub onnxruntime opencv-python numpy
hf download AutowareFoundation/meteor --local-dir models
hf download AutowareFoundation/meteor-demo-scenes --repo-type dataset --local-dir data
sha256sum -c models/SHA256SUMS --ignore-missing && (cd data && sha256sum -c SHA256SUMS --quiet)

# 1. does the ONNX run?  (CPU, about 4 s per frame) — prints every output tensor and ends with "SMOKE PASS"
python3 hf/onnx_smoke_test.py --onnx models/meteor_v157c3Z.onnx --root data/valday --frame 40

# 2. TensorRT fp16 engine (no plugin, no calibration; about 10 min, 194 MB)
pip install tensorrt pycuda
python3 deploy/build_engine_fp16.py models/meteor_v157c3Z.onnx out/meteor_v157_fp16.engine 8

# 3. demo video over the six scenes (same renderer as the Orin demo; ~30 ms inference, ~20 FPS with rendering)
mkdir -p out/demo6 && for r in highway_day mountain_day arterial_day valday valcurve fast; do \
  s=$(cat data/$r/scenes.txt); ln -sfn $PWD/data/$r/$s out/demo6/; echo $s >> out/demo6/scenes.txt; done
METEOR_TH2D=0.30 METEOR_SEG2D_OVERLAY=0 METEOR_OCC_PANEL=0 METEOR_2D_HIDE=7 PYTHONPATH=. \
python3 deploy/orin_realtime.py --engine out/meteor_v157_fp16.engine --root out/demo6 --out out/demo6.mp4
```

pycuda may print "context stack was not empty" when the process exits; it is harmless and the video
is complete. To try a single scene, point `--root` at it directly, e.g. `--root data/valday`.

## 4. Going further

- **Jetson AGX Orin (INT8, about 70 ms)**: insert the lift plugin with
  `make_plugin_onnx.py --tables models/lift_plugin_tables_r64`, then build on the device with
  `deploy/orin_build_int8.py` (real-frame calibration). Steps and pitfalls: [deploy/README.md](../deploy/README.md) §6–§7.
- **Re-export / fine-tune**: `deploy/export_onnx.py --ckpt models/meteor_v157.pt --model v52 --n-cams 8 …`
  (flags in deploy/README.md §1); `--with-lidar` produces the LiDAR-input variant. Training: [TRAINING.md](TRAINING.md).
- **Building the inputs yourself**: images are RGB uint8 `[1,8,3,432,768]` in the camera order given by
  `input.cameras` in `meteor_v157.param.yaml`; `K` are the intrinsics at 768×432, `T_cam_ego` maps ego → camera,
  `v0` is the ego speed in m/s. `hf/onnx_smoke_test.py` is the reference implementation.

## 5. Rendering environment variables (the main ones)

| Variable | Default | Meaning |
|---|---|---|
| `METEOR_TH2D` | 0.30 | score threshold for displayed 2D detections |
| `METEOR_SEG2D_OVERLAY` | 0 | overlay 2D segmentation on the camera tiles (1 = on) |
| `METEOR_OCC_PANEL` | 0 | 3D occupancy panel (1 = on; rendering gets heavier) |
| `METEOR_2D_HIDE` | 7 | 2D classes not drawn (7 = road paint) |
| `METEOR_CUDAGRAPH` | 0 | run the engine through a CUDA Graph (used on Orin) |
| `METEOR_LIDAR` | 0 | feed `lidar_bev/` to a LiDAR-input engine |
| `METEOR_GPU_NAME` | (auto) | device name shown in the title bar |
| `METEOR_VLA_JSONL` | (none) | overlay a per-frame JSONL of an external reasoner (trajectory `wp_reg`, `json.scene/hazards/rationale/command`) as a second path and a text strip (experimental) |

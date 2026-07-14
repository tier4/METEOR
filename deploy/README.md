# METEOR deployment (ONNX / TensorRT)

One static-graph engine runs all 7 tasks. The streaming temporal BEV is
deployed as a **host-side recurrence**: the engine takes `prev_bev` /
`warp_theta` as ordinary inputs and returns the current `raw_bev` as an
ordinary output, which the runtime feeds back on the next frame. No RNN,
no dynamic shapes, no plugins — `conv / grid_sample / gather / matmul`
only (`F.affine_grid` is replaced by a constant base grid + `matmul` at
export time).

```
imgs [1,8,3,432,768] ─┐
K, T_cam_ego          ├─► ┌─────────────┐ ─► lane / hm / reg / occ      (RAW BEV)
v0                    │   │  METEOR v26 │ ─► ego / traj / stationary    (FUSED BEV)
prev_bev  ◄─ feedback ┤   │   engine    │ ─► depth / seg2d / hm2d / reg2d (aux 2D)
warp_theta (from odom)┘   └─────────────┘ ─► raw_bev ──► next frame's prev_bev
```

## 1. Export ONNX

```bash
python3 deploy/export_onnx.py --ckpt out/bevlane_ckpt_r16/last.pt \
    --out out/meteor_v26.onnx --check --fp16
```

`--check` verifies every output against PyTorch with onnxruntime
(max |diff| ≈ 1e-5 across all 12 outputs). `--fp16` additionally writes a
weight-fp16 copy (~84 MB) — this is the artifact attached to release tags.

## 2. Build the TensorRT engine

```bash
trtexec --onnx=out/meteor_v26.onnx \
        --saveEngine=out/meteor_v26_fp16.engine \
        --fp16 --memPoolSize=workspace:8192
```

Verified on TensorRT 8.6 (single build pass, no plugins, INT64→INT32
weight cast warning is benign).

## 3. Run

```python
from deploy.runtime import (MeteorRT, preprocess_images,
                            decode_boxes, decode_agent_traj)

rt = MeteorRT("out/meteor_v26_fp16.engine")
rt.reset()                                   # at every scene boundary
for frame in scene:
    imgs = preprocess_images(frame.bgr_images)      # 8 cams, 768x432
    out = rt.infer(imgs, frame.K, frame.T_cam_ego,
                   v0=frame.speed_mps, pose=frame.xy_yaw)
    boxes = decode_boxes(out["hm"], out["reg"], out["stationary"])
    boxes = decode_agent_traj(out["traj"], boxes)
    lanes = out["lane"].argmax(1)                    # [1,800,500] BEV classes
    occ   = out["occ"].argmax(1)                     # [1,16,200,200] voxels
```

- `pose` is the global 2D ego pose `(x, y, yaw)`; the runtime derives
  `warp_theta` from consecutive poses (same math as training). Pass
  `pose=None` to run single-frame (temporal state is dropped).
- `decode_boxes` returns dicts with `x/y/l/w/yaw` (ego frame, metres),
  class, score, and the learned `stationary` flag (parked/stopped).
- Per-box futures (6 x 0.5 s) come from `decode_agent_traj` — constant
  cost regardless of agent count.

## Files

| file | role |
|---|---|
| `export_onnx.py` | training ckpt → ONNX (+parity check, +fp16 copy) |
| `runtime.py` | TensorRT streaming runtime + decode utilities |

ONNX artifacts are published as release assets on version tags
(`v26-r16` → `meteor_v26_fp16.onnx`), not tracked in the repo.

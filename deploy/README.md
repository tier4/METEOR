# METEOR deployment (ONNX / TensorRT)

One static-graph engine runs all 12 tasks. The streaming temporal memory is
deployed as a **host-side recurrence**: the engine takes `hist_bev[3]` /
`hist_theta[3]` as ordinary inputs and returns the current `raw_bev` as an
ordinary output, which the runtime rings back into the history on the next
frame. No RNN, no dynamic shapes, no plugins — `conv / grid_sample / gather
/ matmul` only (`F.affine_grid` is replaced by a constant base grid +
`matmul` at export time).

`infer_t4dataset.py` closes the loop: point it at a raw t4dataset scene and
it produces per-frame results and a video with no GT and no PyTorch.

```
imgs [1,8,3,432,768] ──┐
K, T_cam_ego           ├─► ┌─────────────┐ ─► lane / hm / reg / occ / flow   (RAW BEV)
v0                     │   │  METEOR v29 │ ─► lg_pts / lg_meta / lg_adj      (lane graph)
hist_bev[3] ◄─ ring ───┤   │   engine    │ ─► ego(K=3) / traj(K=3) / stationary / risk
hist_theta[3] (odom) ──┘   └─────────────┘ ─► depth / seg2d / hm2d / reg2d / tl (image)
                                           ─► raw_bev ──► next frame's history slot
```

## 1. Export ONNX

```bash
python3 deploy/export_onnx.py --ckpt out/bevlane_ckpt_r20/last.pt \
    --out out/meteor_v29.onnx --check --fp16
```

`--check` verifies every output against PyTorch with onnxruntime
(max |diff| ≈ 1e-5 across all 18 outputs). `--fp16` additionally writes a
weight-fp16 copy — this is the artifact attached to release tags.

## 2. Build the TensorRT engine

```bash
trtexec --onnx=out/meteor_v29.onnx \
        --saveEngine=out/meteor_v29_fp16.engine \
        --fp16 --memPoolSize=workspace:8192
```

Verified on TensorRT 8.6: single build pass, no plugins (INT64→INT32
weight cast warning is benign). Measured **14.4 qps ≈ 70 ms/frame** for the
full 8-camera / 8-task graph in fp16 — on a GPU concurrently running a
training job, so treat it as a lower bound.

## 3. Run on a raw t4dataset scene (no GT, no PyTorch)

```bash
python3 deploy/infer_t4dataset.py \
    --engine out/meteor_v29_fp16.engine \
    --scene /data1/dataset/DTSET/<batch>/<scene> \
    --out out/t4_infer --video out/t4_infer.mp4
```

It reads `annotation/*.json` + `data/CAM_*` directly, rescales the intrinsics
to 768x432, derives ego speed/pose from `ego_pose`, streams the memory, and
writes per frame: 3D boxes (class, score, pose, size, yaw, parked flag, 3 s
speed), the 9-class lane map, K=3 ego paths + confidences, controls, the
traffic-light state, the risk field and the occupancy voxels — plus an
overlay video.

## 4. Or drive the runtime yourself

```python
from deploy.runtime import (MeteorRT, preprocess_images,
                            decode_boxes, decode_agent_traj)

rt = MeteorRT("out/meteor_v29_fp16.engine")
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
| `export_onnx.py` | training ckpt → static 18-output ONNX (+parity check, +fp16) |
| `runtime.py` | TensorRT streaming runtime (3-slot memory ring) + decoders |
| `infer_t4dataset.py` | raw t4dataset scene → engine → per-frame npz + video |
| `expand_v29_init.py` | warm-start helper: single-mode ckpt → K=3 (diverse) |

ONNX artifacts are published as release assets on version tags
(`v26-r16` → `meteor_v26_fp16.onnx`), not tracked in the repo.

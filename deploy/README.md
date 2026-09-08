# METEOR deployment (ONNX / TensorRT / Jetson AGX Orin)

One static-graph engine runs all 12 tasks. There are no dynamic shapes, no
RNN cells and — apart from one optional CUDA lift plugin — no custom ops:
`conv / grid_sample / gather / matmul` only (`F.affine_grid` is replaced by a
constant base grid + `matmul` at export time).

The streaming temporal memory can be deployed two ways:

- **host-side recurrence** — the engine takes `hist_bev[3]` / `hist_theta[3]`
  as ordinary inputs and returns the current `raw_bev`, which the runtime
  rings back into the history on the next frame;
- **`--no-hist`** — when the shipped weights were trained with a zero history
  (the current production line), the history path is baked out of the graph
  entirely. Same outputs bit-for-bit, 107 → 83 ms on Orin (see §5).

`infer_t4dataset.py` closes the loop: point it at a raw t4dataset scene and
it produces per-frame results and a video with no GT and no PyTorch.
`orin_realtime.py` (Python) and `cpp/` (C++) are the pipelined real-time
runtimes with identical rendering.

```
imgs [1,8,3,432,768] uint8 ──┐
K, T_cam_ego, v0             ├─► ┌─────────────┐ ─► lane / hm / reg / occ / flow          (BEV)
lidar_bev [1,4,400,250]* ────┤   │   METEOR    │ ─► ego(K=3) / traj(K=3) / stationary / risk / tl
lidar_flag [1]*              │   │   engine    │ ─► depth / seg2d / hm2d / reg2d / depth_mean (image)
hist_bev[3] ◄─ ring ────────┘   └─────────────┘ ─► raw_bev ──► next frame's history slot**
 * optional (--with-lidar)   ** only without --no-hist
```

## 1. Export ONNX

```bash
# production export used for the Orin engines (camera-only, zero history baked out)
python3 deploy/export_onnx.py --ckpt out/ckpt_v151/best_e2e.pt --model v52 --n-cams 8 \
    --drop unk,pl,flow --uint8-in --argmax-out --lane-logits \
    --seg-bias "1:0.5,3:1.5,4:1.0,5:0.6,6:0.5" --no-hist --depth-mean \
    --out out/meteor_v151c3Z_prod.onnx
# then insert the CUDA lift plugin (optional, -4..5 ms on Orin)
python3 deploy/cpp/liftbench/plugin/make_plugin_onnx.py \
    --onnx out/meteor_v151c3Z_prod.onnx --tables deploy/cpp/liftbench/tables_r64 \
    --out out/meteor_v151c3Z_final.onnx
```

| flag | what it does | why |
|---|---|---|
| `--uint8-in` | images enter as uint8, `/255` and ImageNet normalisation happen in-graph | halves H2D traffic, lets the loader write straight into pinned slots |
| `--argmax-out` (+`--lane-logits`) | lane / depth / seg2d leave as uint8 argmax; lane logits kept for temporal fusion | D2H shrinks from ~150 MB to a few MB per frame |
| `--seg-bias` | bakes the per-class decision-boundary calibration of the BEV seg into the final bias | thin classes are systematically under-predicted; a bias shift is free at runtime and does not touch the ego path (verified) |
| `--drop` | leave unused heads (unknown, pseudo-LiDAR, flow) out of the graph | fewer kernel launches; the runtime decodes only what it displays |
| `--no-hist` | remove the 3-slot history path (tgate / tfuse3 weights cut to the first 96 ch) | −24 ms and −230 MB of I/O when the weights were trained on a zero history |
| `--depth-mean` | adds the expected depth (half resolution) as a 19th output | metric distance for 2D→BEV lifting of unknown obstacles, +0.8 ms |
| `--with-lidar` | optional `lidar_bev` input + a **host-side scalar `lidar_flag`** | the flag used to be derived in-graph as `lb.abs().sum() > 0`; TensorRT's Myelin ran that reduction over 800×500×96 in **14.8 ms**. Feeding a scalar from the host removed it |
| `METEOR_EXPORT_FAST=1` | simplifies the NaN-guard chains in the exported graph (function-preserving) | −3.6 ms |
| `--check` | onnxruntime vs PyTorch parity on every output | run it after every exporter change; the bake-outs above were all verified bit-equal this way |

## 2. Build the engine (workstation)

```bash
trtexec --onnx=out/meteor_v151c3Z_final.onnx --saveEngine=out/meteor_v151c3Z_fp16.engine \
        --fp16 --staticPlugins=deploy/cpp/liftbench/plugin/build/libmeteor_lift.so \
        --inputIOFormats=uint8:chw,fp32:chw,fp32:chw,fp32:chw --memPoolSize=workspace:8192
```

`build_and_bench.py` does the same from Python (engine build + latency + a
head-by-head parity check against the PyTorch model).

## 3. Run on a raw t4dataset scene (no GT, no PyTorch)

```bash
python3 deploy/infer_t4dataset.py --engine out/meteor_v151c3Z_fp16.engine \
    --scene /path/to/t4dataset/<scene> --out out/t4_infer --video out/t4_infer.mp4
```

## 4. Or drive the runtime yourself

```python
from deploy.runtime import MeteorRT, preprocess_images, decode_boxes, decode_agent_traj

rt = MeteorRT("out/meteor_v151c3Z_fp16.engine")    # detects no-hist / LiDAR inputs itself
rt.reset()                                           # at every scene boundary
for frame in scene:
    imgs = preprocess_images(frame.bgr_images)       # 8 cams, 768x432
    out = rt.infer(imgs, frame.K, frame.T_cam_ego, v0=frame.speed_mps, pose=frame.xy_yaw)
    boxes = decode_boxes(out["hm"], out["reg"], out["stationary"])
    boxes = decode_agent_traj(out["traj"], boxes)
```

Runtime options that matter on Orin: `METEOR_CUDAGRAPH=1` (capture
`enqueueV3` after two warm-up frames), pinned zero-copy input slots
(`rt.pinned_input_slots(n)`, the decoder writes CHW directly; −4.6 ms of
`np.copyto`), `METEOR_PLUGIN_SO` (lift plugin), `METEOR_ZERO_HIST=1` (feed a
zero history to an engine that still has the history inputs).

## 5. Graph-level levers, measured on Orin (INT8, 8 cameras, CUDA Graph)

Every lever below was accepted or rejected on a measured Δaccuracy / Δms
ledger, one variable at a time. The path from the first INT8 engine to the
current production engine:

| step | engine | ms | note |
|---|---|---|---|
| first sound INT8 (calibration fix, §6) | v125p | 114.3 | fp16 was 176 ms |
| NaN-guard simplification (`EXPORT_FAST`) | v125f | 110.7 | function-preserving |
| aux streams 0 + CUDA Graph | v125g | 108.0 | aux streams + graph capture together produced stale outputs; aux0 fixes it |
| depth head width ×0.75 (retrain) | v128g | 103.7 | −4.7 ms, ADEc kept |
| seg decision-boundary bias + risk map output | v128cRg | 103.4 | risk costs ≈0 |
| **history path baked out** (`--no-hist`) | v142c3Zg | 83.2 | −24 ms, bit-equal outputs |
| depth head width ×0.5 (retrain) | v147c3Zg | 77.8 | |
| dense production line | **v151c3Zg** | 78.7 | incl. half-res `depth_mean` |
| **2:4 structured sparsity** (§7) | **v157c3Zg** | **69.6** | closed-loop accuracy at dense parity |
| + optional LiDAR input | v157Lc3Zg | ≈70 (target) | 85 → 83.6 → rebuild after moving the LiDAR flag host-side |

Rejected with numbers (do not retry blindly): gather-form lift (89 vs 78 ms —
TRT10's ScatterND beat the padded gather), 48 depth bins (accuracy),
GridSample fp16 surgery (no effect), tactic re-rolls (±0.2 ms only),
7-camera export (the 8th camera costs 2.6 ms and the model was trained for 8).

## 6. Quantization (INT8) — what is done and what we learned

METEOR is quantized **post-training** with TensorRT's entropy calibrator, on
the device, without PyTorch (`orin_build_int8.py`). QAT was not needed once
calibration was done right; the training-side measures below are cheap
robustness aids, not quantization-aware training.

### 6.1 Training-side preparation
- **`--quant-noise 1.0`** (model v45+): forward hooks on the image-feature
  fuse and the temporal BEV fuse add per-channel uniform noise of one INT8
  step (`ch_absmax/127`). Zero parameters; makes the fused features tolerant
  to rounding.
- **fp16 overflow is a training problem first.** A conv feeding a BatchNorm
  has a free scale (BN divides it back out), nothing in the loss keeps it
  near 1, and it drifts until the fp16 activation passes 65504. One round
  silently discarded 69 % of its steps this way while validation still
  printed plausible numbers. `renorm_convbn.py` rescales conv/BN pairs
  function-preservingly (BEV mIoU unchanged to four decimals); `train.py --bn-guard`
  and the non-finite-loss snapshot/restore keep it from recurring.
  Well-scaled activations are also what makes INT8 ranges tight.
- **EMA and BN buffers**: EMA averages parameters only; BN running stats stay
  raw. In high-LR fine-tunes the averaged `seg_head` weights and the raw BN
  statistics drift apart and the 2D seg collapses to one class.
  `--ema-exclude seg_head.` keeps that head raw (its 2D-seg mIoU stays at the raw head's level
  instead of collapsing to a single class).
  This matters for deployment because PointPainting re-injects 2D seg into
  the BEV — a dead 2D head ships a weaker BEV.

### 6.2 Calibration
- **Real frames only, never random tensors.** A calibrator fed noise picks
  activation ranges that do not exist in the data. Default: 96 frames,
  stride 4, capped per scene, from converted sample scenes on the device.
- **Recurrent inputs are calibrated with real recurrence.** For engines that
  still carry `hist_bev`, a *companion* fp16 engine of the same graph runs
  the calibration stream first and maintains its device-resident BEV ring
  exactly as the runtime does; the calibrator hands TensorRT the companion's
  own device pointers. Feeding a zero history calibrates the temporal path on
  a tensor that only occurs at scene starts (it gave some layers a 0 scale
  and killed two builds).
- **The dtype trap (the bug that cost the most time).** The companion's
  `hist_bev` buffer is fp16 (I/O format), but the network under calibration
  reads it as the fp32 the ONNX declares. Reinterpreting fp16 bits as fp32
  injected ~1e9 garbage, the `hist`/`tfuse` scales in the cache came out
  **5×10⁷× too large**, and every INT8 engine from v95 to v120 had a frozen
  ego trajectory and a constant stationary flag while BEV seg and 3D boxes
  looked perfect. Fix: convert on the host (`dtoh → astype(fp32) → htod`).
  Rule: **feed the calibrator the dtype the network reads, not the dtype the
  pointer holds.** Fast diagnosis: diff the per-tensor scales in the `.calib`
  cache between a healthy and a broken build — the culprit stands out by
  orders of magnitude.
- **With `--no-hist` the companion trick is no longer needed for history**, but
  the same pipeline still feeds **real LiDAR** to `--with-lidar` engines: a
  zero-LiDAR calibration collapses the `lidar_stem` scales.
- Calibration caches are **per checkpoint** (not per architecture), versioned
  next to the engine (`eng/<tag>.engine.calib`), and do not cross TensorRT
  major versions. `trtexec` cannot read MinMax caches (header
  `MinMaxCalibration`) — a sweep produced 0-byte engines until this was
  noticed. Entropy / MinMax / legacy gave identical results on the frozen-ego
  bug (matching to 3 decimals): the calibrator *method* was never the cause.

### 6.3 Builder settings that made a difference
- `--max-aux-streams 0` + CUDA Graph (aux streams under graph capture
  produced stale outputs; aux0 removed the bug and lost nothing).
- `--sparsity=enable` only when the weights are actually 2:4 (§7).
- `builderOptimizationLevel 5` is available via `--builder-opt` but bought
  ±0.2 ms — tactic randomness, not a lever.
- `--fp16-keep <patterns>` pins selected layers to fp16
  (`PREFER_PRECISION_CONSTRAINTS`). It was the operational workaround for the
  frozen ego (65 layers, 118.6 ms) before the calibration fix; it is not used
  in production any more. Note that Myelin may ignore per-layer precision
  inside a fused region — check the profile, not the flag.
- I/O formats: `uint8:chw` in, `uint8:chw` argmax outputs, `fp16:chw` for
  the lane logits and `depth_mean`.

### 6.4 Acceptance: health checks, not just accuracy
An INT8 engine can be *plausible* and *dead* at the same time. Every build
passes three device-side checks before it can become the default:
1. **Frozen-output test** (`bev_frozen_test.py`): run real frames, zeros and
   noise; ego / occ / traj must change with the input. Frame-to-frame diff of
   ego vs the fp16 companion (broken INT8: 0.06–0.15, healthy: 0.7–2.3).
2. **Stationary-flag statistics** (`stat_probe.py`): logit std and stop rate
   must sit in the fp16 band (a collapsed head has std 4.8 vs 9.3 and marks
   every object "stopped"). A loose "N distinct values" rule once passed a
   broken engine — compare against fp16 numerically.
3. **SHIP-CHECK** (onnxruntime, before the ONNX leaves the workstation):
   `seg2d` non-zero fraction > 10 % on real frames — catches a dead 2D head
   that the EMA guard missed.
Then `bench_rt.py` (median of N frames, CUDA Graph on) for the ledger.

## 7. 2:4 structured sparsity — what is done and what we learned

Orin's Ampere tensor cores run 2:4 sparse kernels at up to 2× the dense
rate. On the production INT8 graph the measured gain is **−10.1 ms /
−13.9 %** (72.7 → 62.6 ms post-hoc; 69.6 ms for the trained, accuracy-recovered
engine incl. `depth_mean`) — the largest single lever after the history bake-out.

### 7.1 Training-time pruning (`train.py --sparse-24`)
- Magnitude pruning: for every 4-group along the input-channel axis of each
  4-D conv weight, the two smallest |w| are zeroed. The mask is derived once
  from the init weights (deterministic, so every DDP rank builds identical
  masks with no communication) and **re-applied after every optimizer step**,
  so the pattern survives fine-tuning and the exported weights are 2:4 by
  construction — no post-hoc zeroing at export.
- **Dense exceptions, learned the hard way.** Output layers are kept dense
  two ways: by name (`dec.out`, `seg_head.out`, `depth_head.4`, `occ_head`,
  `hm_head`, `reg_head`) and by shape (out-channels < 32, in-channels < 16 or
  not a multiple of 4). The first list missed the detection heads: pruning
  `reg_head`'s sin/cos regression tripled the vehicle yaw error
  and cut R50 by a quarter while seg and E2E recovered fine. Per-channel
  regression outputs are exactly what magnitude pruning butchers.
- **`--sparse-exclude ego_,traj_,tfuse3,tgate,sem_ego,delta_stat,refiner.e2e`**:
  keep the planning branch dense. Pruning everything cost the E2E head
  about +20 % ADEc and it did not recover in 3 epochs; the excluded branch
  is worth ≈1 ms on Orin, so the trade is one-sided. Sparsify what the
  profiler says is expensive (backbone, depth, BEV decoder, seg, det stem,
  refiner), not what is convenient.
- **`--sparse-ramp-steps N`**: train the first epoch with a 1:4 mask (only the
  smallest of each 4 zeroed), then switch to 2:4. Gradual pruning recovered
  better than one-shot: the closed-loop chain error matched the dense model to three decimals, where
  one-shot variants stayed 1–3 % behind.
- Tried and recorded as *not* helping: continued fine-tuning at lower LR
  (v153), widening the dense exclusions to the BEV decoder and refiner (v154:
  worse, and only −2.9 ms to gain), dense-teacher distillation of fused BEV +
  ego outputs (v156: distillation loss plateaued at 5.2; the teacher was not
  given the student's auxiliary inputs — parked, not refuted).
- Compare checkpoints at the **same position** (last epoch, or the chain-240
  verdict). `best_e2e` is the round minimum and in a fine-tune it is usually
  the epoch-0 EMA snapshot of the *initial* weights; comparing a sparse
  round's best against that number invented a "+10 % regression" that did not
  exist (the real gap was 2–3 %).

### 7.2 Deployment
- **TensorRT only uses sparse kernels if the weights really are 2:4.**
  `--sparsity=enable` / `--sparse` on dense weights changes nothing
  (72.76 vs 72.82 ms). Every earlier "sparsity does nothing on Orin"
  conclusion was measured on dense models after the flag had silently dropped
  out of the training recipe.
- **Compare like for like.** A first measurement showed sparse *slower* by
  +2.9 ms: the sparse ONNX had been sent before the lift plugin was inserted,
  so Myelin ran the raw lift (20 ms) where the dense engine had the plugin
  (0.4 ms). Per-layer profile diff (`prof_diff.py`) showed the shared 274
  layers at −9.3 ms (dec −2.2, depth −1.4, seg −0.9, det stem −0.9, layer3
  −0.8, refiner −0.7). Whenever a sparse number is quoted, state (a) whether
  the weights are 2:4, (b) whether the plugin is in, (c) whether accuracy has
  been recovered.
- fp16 gains less than INT8 from sparsity here (−4 % vs −14 %): the fp16
  engine is memory-bound in the lift, the INT8 one is compute-bound in the
  convolutions.
- Sparse engines go through the same calibration and health checks as dense
  ones (§6); the calibration cache is *not* shared between a dense and a
  sparse checkpoint.

## 8. Files

| file | role |
|---|---|
| `export_onnx.py` | ckpt → static ONNX (+parity check); `--no-hist`, `--with-lidar`, `--depth-mean`, `--seg-bias`, `--uint8-in`, `--argmax-out` |
| `cpp/liftbench/plugin/make_plugin_onnx.py` | inserts the CUDA lift plugin (`libmeteor_lift.so`) into the exported graph |
| `runtime.py` | Python TensorRT runtime: history ring, CUDA Graph, pinned zero-copy inputs, LiDAR, decoders |
| `orin_build_int8.py` | torch-free on-device INT8 calibration (real frames, companion engine for recurrent inputs, real LiDAR) |
| `orin_realtime.py`, `orin_render.py`, `viz_np.py` | pipelined real-time demo + renderer (reference look) |
| `infer_t4dataset.py` | raw t4dataset scene → engine → per-frame npz + video |
| `build_and_bench.py`, `bench_engine.py`, `profile_engine.py`, `profile_layers.py` | workstation build / latency / per-layer profile |
| `orin/` | Orin-side scripts: `demo.sh`, `demo_cpp.sh`, `bench_rt*.py`, `bev_frozen_test.py`, `stat_probe.py`, `prof_diff.py`, job template |
| `cpp/` | C++ TensorRT runtime (`meteor_rt.*`, `realtime_main.cpp`, `render.cpp` with Python parity), lift plugin sources |

ONNX files, calibration caches and lift-plugin tables are distributed on
Hugging Face, not tracked in this repository.

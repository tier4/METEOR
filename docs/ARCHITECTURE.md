# METEOR Architecture (v26)

One shared image backbone, one BEV representation, seven task heads.
Every operator is TensorRT-exportable (`conv / grid_sample / gather / maxpool /
avgpool / MLP` — no attention, no scatter).

**Inputs**: 8 cameras (WIDE / LEFT / RIGHT / NARROW × front / back) at 768×432,
camera intrinsics `K [8,3,3]`, extrinsics `T_cam_ego [8,4,4]`, current speed `v0`.

## Image branch (per camera, shared weights)

| Stage | Channels | Resolution | Stride | Params |
|---|---|---|---|---|
| ResNet-34 stem | 64 | 108×192 | s4 | — |
| layer1 / 2 / 3 / 4 | 64 / 128 / 256 / 512 | 108×192 … 14×24 | s4…s32 | 21.28 M |
| FPN (1×1 laterals → 160ch, upsample-add to s4, 3×3 fuse) | **160** | **108×192** | **s4** | 0.38 M |

The fused stride-4 feature `f [B·8, 160, 108, 192]` feeds four image-space heads:

| Head | Structure | Output | Params / GFLOPs |
|---|---|---|---|
| 2D segmentation | encoder–decoder: s8 192ch + s16 320ch ConvBlocks, 1×1 laterals, s4 skip (96ch) → 21 classes | `[B,8,21,108,192]` | 4.18 M / 219 G |
| 2D detection | shared 128ch stem at s4, downsampled 192ch (s8) and 256ch (s16) towers; per-scale CenterNet heads (10-class heatmap + 4ch box reg), focal init −2.19 | 3 scales; GT assigned by max(w,h): <40 px → s4, <120 px → s8, else s16 | 2.85 M / 219 G |
| Depth decoder | ConvBlocks 160→256→256→192→128 + 1×1→64 bins (1.25 m each) | `[B,8,64,108,192]` | 3.29 M / 1093 G |
| Context | 1×1 160→96 | `[B·8,96,108,192]` | 0.02 M / 5 G |

## Depth-gated IPM (parameter-free)

1. BEV ground grid (z = 0, 800×500 @ 0.2 m, ±80 m fwd × ±50 m lat) is projected
   into every camera with `K`, `T_cam_ego`.
2. Context features and depth probabilities are `grid_sample`d at the projections.
3. Each sample's **depth bin at its true range** is `gather`ed:
   weight `w = P(depth = range) + 0.05`. Depth is the *visibility valve* —
   features flow into BEV only where the predicted depth agrees with the
   geometric range, suppressing occlusion bleed-through.
4. Weighted average over the 8 cameras → BEV feature `[B, 96, 800, 500]`.

## BEV branch

| Head | Structure | Output | Params / GFLOPs |
|---|---|---|---|
| Lane segmentation | ConvBlocks 96→160→160→128 → 1×1→9, all at 800×500 | 9 classes @ 0.2 m | 1.16 M / 933 G |
| 3D oriented boxes | s2 stem 96→128 (400×250) + ConvBlock; 2-class heatmap + 6ch reg (Δr, Δc, log l, log w, sin yaw, cos yaw); decode = 3×3 maxpool NMS + top-K | vehicles / VRU | 0.41 M / 82 G |
| E2E driving | pyramid 96→128 (s4) →192 (s2) →256 (s2) →256 (s2) → AvgPool → concat `v0` → MLP 257→512→512→256→15 | 6 waypoints @ 0.5 s + steering + accel + brake logit | 4.04 M / 37 G |
| 3D occupancy | crop BEV to ±40 m (rows 200:600, cols 50:450) → s2 stem 96→128 + ConvBlock 192 → 1×1 → 160ch → view `[B,10,16,200,200]` | 10 classes, 0.4 m voxels, z ∈ [−1, 5.4) m | 0.70 M / 55 G |

## Cost summary (measured)

| Stage | Params | GFLOPs | Share |
|---|---|---|---|
| Backbone + FPN (8 cams) | 21.66 M | 475 | 15 % |
| Depth decoder | 3.29 M | 1093 | 35 % |
| BEV lane seg head | 1.16 M | 933 | 30 % |
| 2D seg head | 4.18 M | 219 | 7 % |
| 2D det head | 2.85 M | 219 | 7 % |
| 3D det head (s4 tower) | 2.06 M | 121 | 4 % |
| Occupancy head | 0.70 M | 55 | 2 % |
| E2E head | 4.04 M | 37 | 1 % |
| Temporal fuse (tfuse) | 0.18 M | 42 | 1 % |
| Agent traj + stationary | 0.41 M | 26 | <1 % |
| Context + IPM | 0.02 M | 6 | <1 % |
| **Total (v26)** | **43.50 M** | **~3100** | |

*(the BEV lane seg head is now the encoder–decoder `LaneDecED`: 4.10 M params at
~0.8× the FLOPs of the flat stack it replaced — same design rule below.)*

Two design rules fall out of this table:

- **Compute lives at high resolution, parameters live at low resolution.**
  The 2D heads run their heavy convolutions at s8/s16 (cheap pixels, big
  channels) and only touch s4 through 1×1 laterals — 18× more parameters for
  ~2× FLOPs versus a naive s4 head.
- **Adding a task is cheap.** The occupancy and E2E heads together are 3 % of
  total compute; the shared backbone/BEV representation carries them.

## Model zoo lineage

`v8` depth-gated IPM (TRT-verified) → `v13d/v14d` stride-4 big depth decoder →
`v16` +oriented 3D boxes → `v17` +2D detection → `v18` +E2E → `v19` capacity
re-balance (encoder–decoder seg, 3-scale det, big E2E) → `v20` +occupancy →
`v21` +one-shot agent forecasting → `v22` +streaming temporal BEV (prev-frame
BEV ego-motion-warped and residually fused; TRT-safe host-side recurrence) →
`v23` LaneDecED + s4 det tower + zero-init fusion → `v24/v25` task routing
(geometry heads on the RAW single-frame BEV, motion heads on the FUSED temporal
BEV — verified by prev-BEV perturbation) → **`v26` +learned parked/stopped flag
+ near-range-first detection supervision = METEOR**.

### Streaming temporal BEV (v22+)

`fused = bev + tfuse(concat(bev, warp(prev_bev, theta)))` where `theta` encodes
the relative ego pose between consecutive frames. At deployment `prev_bev` and
`theta` are ordinary engine inputs and the current raw BEV is an ordinary
output — the recurrence lives on the host (see `deploy/`), the graph stays
static. The last BN of `tfuse` is zero-initialised so fusion starts as an
identity. 3D detection regresses per-cell `(offset, log-size, sin/cos yaw)`
supervised on the full 3×3 neighbourhood of every GT centre (the decode reads
the heatmap peak, which is frequently 1 cell off the true centre).

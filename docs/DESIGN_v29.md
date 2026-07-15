# DESIGN — METEOR v29 (round 20): four upgrades, independently gated

Goal: land all four FSD-gap items in ONE model version, each behind its own
flag so any regression can be isolated without rebuilding the round.

| # | Feature | Flag | New params | New GT? |
|---|---------|------|-----------|---------|
| 1 | Multimodal E2E (K=3) + multimodal agent forecast | `--k-modes` / `--traj-modes` | +0.05 M | no |
| 2 | Temporal memory queue (3 frames, 0.4/1.2/2.8 s) | model=v29 | +0.6 M | no |
| 3 | Vector lane graph (slot decoder + adjacency) | `--lanegraph-w` | +1.4 M | extract_lanegraph.py |
| 4 | Occupancy flow (ground-plane velocity) | `--flow-w` | +0.05 M | no (on-the-fly) |

Total ≈ 46.9 M params (from 44.8 M).

---

## 1. Multimodal trajectories (winner-takes-all)

**Ego.** `ego_mlp` output 15 → 42: `K×12` waypoints + `K` mode logits +
steer/acc/brake (mode-independent). Loss per sample: compute the existing
curvature-weighted L1 for each mode against the single GT, backprop ONLY the
best mode; CE trains mode logits toward the argmin. Ambiguous futures stop
averaging: heads specialise (straight / turn / lane-change) per sample.

**Agents.** `traj_head` 12 → 39 ch (`K×12 + K` per cell); same WTA at GT
centres. The stationary head is unchanged (reads det stem).

**Metrics.** `[valE2E]` gains `minADE` (best-of-K) and `modeAcc`; headline
ADE/ADEc switch to top-1-mode. `[valTraj]` gains `agent-minADE`.

**Deploy.** Outputs widen; decode picks argmax-mode (or hands all K to the
risk-map line-integral scorer — see deploy/README).

## 2. Temporal memory queue

Replaces the single prev-frame fusion. History = raw BEV features at
**t−0.4 s, t−1.2 s, t−2.8 s** (log spacing: short horizon for velocity,
long for occlusion persistence), each ego-motion-warped to the current
frame with its own accumulated theta.

- `tfuse` → `tfuse3`: 1×1 `(4·96)→96` + ConvBlock, **last BN zero-init**
  (identity start — the v22/v23 lesson).
- Missing slots (scene start, unreadable frames): zero BEV + slot validity,
  matching the current `pvalid` convention.
- Training: history BEVs in their own `no_grad + autocast` region (the r12
  autocast-cache bug class). Dataset loads 3 extra frame image sets
  (+24 jpg decodes/sample). **Measured cost gate:** if it/s exceeds 2.0 s
  at batch 2×7GPU, drop the −2.8 s slot (N=2) before launch.
- Deploy: engine inputs `hist_bev[3]`, `hist_theta[3]`; the runtime keeps a
  device-side ring buffer of its own `raw_bev` outputs with poses and picks
  the nearest-age entries — no host copies, graph stays static.

## 3. Vector lane graph

**GT (`extract_lanegraph.py`, stage 14).** `out/production/<scene>/
vector_map.json` already stores connected polylines per class in the map
frame (0.1 m grid). Per frame: transform to ego via `ego_motion.pose`,
clip to ROI x∈[−10, 60] m, |y|≤25 m; split at ROI boundary; resample each
chain to **P=12 points**; keep **M=24** slots (nearest-first) over classes
{laneline, road_edge, stopline}; adjacency `A[M,M]=1` where two slots are
consecutive pieces of one source chain or endpoints meet within 1 m.
Saved per scene as `lanegraph.npz {pts[F,24,12,2], cls[F,24], n[F],
adj[F,24,24]}` (uint16-quantised points).

**Head (TRT-safe, no runtime Hungarian).** 24 anchors on a fixed 4×6 grid
over the ROI. A stride-4 conv tower on the RAW BEV crop produces a feature
map; each anchor bilinearly samples its cell (grid_sample) into a slot
embedding → shared MLP regresses 12×2 point offsets (w.r.t. anchor),
existence, 3-class; adjacency = MLP on concatenated slot-embedding pairs.

**Loss.** Train-time Hungarian matching (scipy on host) between predicted
and GT chains with point-wise L1 after direction-normalising each GT chain
(reverse if that halves the distance); exist BCE (unmatched slots → 0),
class CE, adjacency BCE on matched pairs. Weight 0.5.

**Metric `[valLane]`.** Chain precision/recall at mean-chamfer < 0.5 m,
plus adjacency accuracy on matched pairs.

## 4. Occupancy flow

**GT on the fly** (no extraction stage): the batch already carries agent
boxes + futures. Rasterise each box footprint into the occ ground grid
(200×200 @0.4 m, ±40 m) with value `traj[0]/0.5 s` (m/s, ego frame);
elsewhere masked. Stationary boxes get (0,0) — supervised, not masked.

**Head.** `flow_head = Conv2d(192, 2, 1)` on the existing `occ_stem`
feature → `[B,2,200,200]`. Masked L1. Metric `[valFlow]`: mean endpoint
error on box cells, split moving/stationary.

---

## Rollout & safety valves

1. r20 launches v29 with **all four enabled**, warm from r19 `last.pt`
   (shape-filtered load; tfuse3 and new heads initialise fresh).
2. Independent gates: any feature can be zero-weighted without touching
   the others (`--lanegraph-w 0` etc.). tfuse3 zero-init means feature 2
   starts as identity — the temporal change cannot perturb warm weights.
3. Regression rule: if BEV mIoU or veh Rn drops > 1 pt vs r19 at ep2, the
   biggest-unknown feature (lane graph) is zero-weighted first; the round
   is NOT restarted.
4. DDP mini-run (2 GPU, --limit-train 8) is mandatory before launch —
   every new output must appear in a loss (v26/v27 lesson).
5. Deploy export gains: `hist_bev/hist_theta` inputs, K-mode outputs, flow
   output, lane-graph slots — all static shapes; verify ORT parity before
   the round completes so the engine is ready with the checkpoint.

## Batch layout (append-only, temporal stays last)

`..., occ, tl, risk, lanegraph(pts,cls,n,adj), hist_imgs[3], hist_rel[3,3],
hist_valid[3]` — `vtmp` formulas gain `4*use_lanegraph`, temporal block
widens from (1 set, 1 rel, 1 valid) to 3-slot arrays.

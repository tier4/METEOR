# METEOR — candidate list for future rounds

Living list of what we could do next, why, and what it would cost. Nothing
here is committed work; each entry is sized so it can be picked up
independently. Ordered within each section by (expected value ÷ risk).

**Baseline to beat** (r20 = v29, clean data, held-out recording day):

| metric | value |
|---|---|
| BEV lane mIoU | 0.308 |
| 2D seg mIoU (21 cls) | 0.539 |
| 3D det veh P / Rn / yaw | 0.83 / 0.72 / 6.4° |
| 3D det VRU P / Rn | 0.75 / 0.45 |
| E2E ADE / ADEc | 0.71 / 0.78 m |
| agent ADE / stationary acc | 2.31 m / 0.68 |
| TL accuracy | 0.28 (regressed — see A3) |
| risk L1 (all / high) | 0.072 / 0.161 |
| occupancy flow EPE (mov / stat) | 1.47 / 0.24 m/s |
| lane graph P / R | 0.01 / 0.01 (not learning — see B1) |

**Rule of thumb**: every candidate must be checkable with `--val-every`
(BEV mIoU + 3D det every N steps, ~90 s). A change that cannot be measured
in a 10-minute probe run is not ready to be a round.

---

## A. Known defects (fix before adding anything)

| # | Item | Evidence | Cost |
|---|---|---|---|
| A1 | **Lane-graph decoder does not learn** — P/R pinned at 0.01 | 24 anchored slots predict independently; adjacency is a post-hoc pair MLP. Point init was fixed (±15 m noise → straight segments) but the ceiling looks structural, see B1 | — |
| A2 | **3D box flicker** across frames | user-visible in demos; scores fluctuate near the decode threshold | S: score EMA in the runtime (no retrain), or M: temporal consistency loss |
| A3 | **TL accuracy regressed 0.88 → 0.28** at r20 | the head survived the v29 task additions but its share of the loss did not; likely just `--tl-w` starvation | S: raise tl-w, verify with a probe |
| A4 | **VRU recall still low** (Rn 0.45 vs veh 0.72) | small 2D projections; camera-confirmation already relaxed to 0.15 | M: recall-oriented focal weighting, or a VRU-specific heatmap radius |
| A5 | **Lane-graph GT includes irrelevant edges** | road edges trace parking-lot outlines, not just the drivable corridor | S: filter chains to those touching the ego-connected drivable region |

## B. Transformer candidates (TRT-safe only)

TRT natively supports `MatMul / Softmax / LayerNorm / Add / Transpose`, so
**dense attention with a fixed token count exports and builds with stock
trtexec**. Deformable attention needs a plugin and is therefore out. The
entries below are ordered by (value ÷ risk).

### B1. Lane-graph: anchored MLP → DETR-style query decoder  ★ strongest case
- **Why**: whether slot *i* continues into slot *j* is a *relational*
  question. Today each slot regresses from its own anchor cell in isolation
  and adjacency is bolted on afterwards — the inductive bias is missing, and
  [valLane] has never left 0.01.
- **Shape**: 24 learned queries → self-attention (24×24) → cross-attention to
  a pooled BEV ROI (~22×16 = 352 tokens). Both counts are static and tiny.
- **TRT**: fully exportable, no plugin, negligible FLOPs at these sizes.
- **Cost**: ~1–2 M params. **Risk**: medium (new module), but it is the one
  head that is currently returning nothing, so the downside is bounded.

### B2. Temporal fusion: concat+1×1 → per-cell slot attention  ★ best value/cost
- **Why**: the fuse currently blends `[current, t−0.4, t−1.2, t−2.8]` with a
  fixed 1×1 conv. A per-cell softmax over the 4 slots would let it *ignore*
  a stale slot where an object has moved — which is exactly the
  moving-object-ghost problem that forced task routing in the first place.
  If fusion learns to suppress ghosts, **the geometry heads could take the
  temporal BEV too** instead of being routed away from it.
- **Shape**: 4 tokens per cell. This is a gate, not really an attention
  block: `conv → softmax over 4 → weighted sum`.
- **TRT**: trivially safe. **Cost**: <0.1 M params, ~0 FLOPs.
- **Risk**: low. Zero-init the gate to reproduce today's fusion exactly.

### B3. E2E head: global average pool → attention pooling
- **Why**: `AdaptiveAvgPool2d(1)` throws away all spatial structure before
  the planner MLP. K=3 queries cross-attending to the s16 BEV grid
  (50×32 = 1600 tokens) would let each hypothesis attend to what it needs
  (the lane ahead, the nearest hazard).
- **TRT**: 3 × 1600 attention — small and static.
- **Cost**: ~0.5 M. **Risk**: medium (E2E is a priority metric; ADEc is the
  thing to watch).

### B4. Agent interaction: per-agent self-attention
- **Why**: agents are predicted independently today — no interaction at all
  (a car yielding to a crossing pedestrian is unmodelled). Likely the ceiling
  for agentADE once the class-conditioning fix lands.
- **Blocker**: today's per-cell readout gathers at *detected* centres, which
  is dynamic. A static version needs a fixed K=64 learned-query decoder —
  i.e. re-architecting the detection→forecast interface.
- **Risk**: high (invasive). Park until B1 proves the query-decoder pattern.

### Explicitly rejected
- **Image→BEV cross-attention (BEVFormer-style)**: dense attention would be
  ~400 k BEV queries × 165 k image keys — not tractable. It only works with
  deformable attention, which is a TRT plugin. Our projection is
  geometrically exact anyway; the learned part (depth) is where the
  uncertainty actually lives.
- **Attention in the 2D seg/det heads**: CNNs are adequate; no evidence of a
  ceiling there.

## C. Capability candidates

| # | Item | Note |
|---|---|---|
| C1 | **Risk-map path scoring** | Line-integrate the K=3 ego hypotheses over the predicted risk field and pick by confidence×safety. Gives a Tesla-v11-style hybrid planner *from parts we already have*, plus an explainable safety argument. Runtime-only — no retraining. |
| C2 | **Failure auto-mining** | Auto-collect frames where val metrics spike (E2E error, det flicker, mode disagreement) and weight them into the next round's scene list. Our substitute for shadow mode. |
| C3 | **Per-lane TL association** | Today the TL state is one whole-image label. Associating lights to lane-graph branches (needs B1 working) is what makes it usable at multi-lane intersections. |
| C4 | **INT8 deployment** | trtexec `--int8` needs entropy calibration over real frames; `prev_bev` must be fed real BEVs or the fusion ranges calibrate wrong. Keep depth-softmax and the regression heads in fp16 (`--precisionConstraints`). ~80 % of FLOPs are backbone convs, so most of the win is available even with those exclusions. |
| C5 | **Longer horizon / more slots** | The memory queue is 2.8 s. Occlusion persistence beyond that (parked car hidden by a bus) would need more slots — cost is linear in backbone passes, so measure `it/s` first. |

## D. Data / infrastructure

| # | Item | Note |
|---|---|---|
| D1 | **Finish the GT catch-up** | seg2d21 missing on ~1.7 k scenes gates occ on the same scenes; running now. Once done the train list should grow well past 2,316. |
| D2 | **Val-only observation sets** | The turn-scene demo set mixed train and val scenes. Keep curated *val-only* sets per scenario (turns, stopped-vehicle overtakes, night, rain) so demos are always honest. |
| D3 | **Round bookkeeping** | `--seed-subset` per round already varies the epoch draw; consider recording the exact drawn indices per round for reproducibility. |

---

## Suggested order

1. **A3** (TL weight) — minutes, recovers a regressed metric.
2. **B2** (temporal gate) — cheapest structural win, and may retire task routing.
3. **A5 + B1** (lane-graph GT filter, then query decoder) — the only head
   currently producing nothing.
4. **C1** (risk path scoring) — no training, immediate demo/paper value.
5. **B3** (E2E attention pooling) — once ADEc is stable post-K=3 fix.
6. **B4 / C3 / C4** — after the above prove out.

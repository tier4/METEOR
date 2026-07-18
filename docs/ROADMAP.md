# METEOR — candidate list for future rounds

Living list of what we could do next, why, and what it would cost. Nothing
here is committed work; each entry is sized so it can be picked up
independently. Ordered within each section by (expected value ÷ risk).

**Baseline to beat** (r26 = v32 ep1 / best-so-far values, held-out recording day):

| metric | value |
|---|---|
| BEV lane mIoU | 0.314 (r25 ep1; r23 final 0.312) |
| 2D seg mIoU (21 cls) | 0.535 |
| 3D det veh P / Rn / yaw / dir-flips | 0.85 / 0.72 / 5.1° / 8% |
| 3D det VRU P / Rn | 0.75 / 0.47 |
| E2E ADE / ADEc | 0.69 / 0.47 m (r23) |
| agent ADE / vehHead / stationary acc | 1.87 m / 26° / 0.70 |
| TL accuracy | 0.86 (recovered, red class still weak ~0.5) |
| unknown obj P / R | 0.07 / 0.02 (first non-zero, v3 GT) |
| +LiDAR mIoU delta (same weights) | +0.004 and widening (C6b) |
| lane graph P / R | 0.01 / 0.01 (not learning — see B1) |

**Rule of thumb**: every candidate must be checkable with `--val-every`
(BEV mIoU + 3D det every N steps, ~90 s). A change that cannot be measured
in a 10-minute probe run is not ready to be a round.

---

## A. Known defects (fix before adding anything)

| # | Item | Evidence | Cost |
|---|---|---|---|
| A1 | **Lane-graph decoder does not learn** — P/R pinned at 0.01 | 24 anchored slots predict independently; adjacency is a post-hoc pair MLP. Point init was fixed (±15 m noise → straight segments) but the ceiling looks structural, see B1 | — |
| A2 | (partially addressed r27: crossing-yaw weight) **3D box flicker** across frames | user-visible in demos; scores fluctuate near the decode threshold | S: score EMA in the runtime (no retrain), or M: temporal consistency loss |
| A3 | ✅ FIXED r22 (0.86; red class ~0.5 remains) — **TL accuracy regressed 0.88 → 0.28** at r20 | the head survived the v29 task additions but its share of the loss did not; likely just `--tl-w` starvation | S: raise tl-w, verify with a probe |
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
| C6 | **Optional LiDAR input, single weights** | ✅ C6a IMPLEMENTED (v31, r23+); ✅ C6b IMPLEMENTED (v32, r25+): host-side pillar raster -> flag-gated 96ch residual, zeros = bit-equal camera-only; +lidar probe delta positive and widening.  Same checkpoint must run with AND without LiDAR (user requirement). Mechanism = modality dropout (drop LiDAR ~50% of train samples) + zero-fill-with-valid-gate at inference — the exact pattern already proven for missing memory slots. **C6a (do first)**: project points to per-camera sparse depth and *sharpen the predicted depth softmax* where measurements exist; depth is where our uncertainty lives, params ~0, Orin-free, single TRT engine (feed zeros + valid=0 when absent). **C6b (if C6a is not enough)**: light pillar branch → BEV 96ch fused as a gated residual like tfuse3; bigger 3D det/occ gains, real Orin cost, use GroupNorm in the branch (BN-pollution lesson from r20). |

| C7 | ✅ L1 IMPLEMENTED (bevlane/guardrail.py + demo --guard; HOLD state, fragment-tolerant, detailed reasons) — next: intervention-rate eval on val. **E2E guardrails (doer/checker safety channel)** | The 12 heads make a classic safety architecture nearly free: (L1) deterministic hard gates — spacetime collision check of the chosen path against predicted occupancy+flow+agent futures, red-light×stop-line gate, bicycle-model feasibility clamp, drivable/free containment; checker heads run on the RAW BEV route while E2E uses the fused route (partial input independence), and with LiDAR attached the raw-point near-field AEB is a **non-ML** last wall. (L2) = C1 risk-integral mode fallback (pick the safest of K=3). (L3) uncertainty monitors (mode spread, temporal path stability, depth/seg entropy OOD) trigger degraded mode. (L4) an independently generated in-lane-stop MRM path (centerline spline + decel profile) replaces vetoed plans. Measurable: correct-intervention vs false-intervention rate on val futures, per round. Runtime-only through L2; no retraining. |

## E. Next-generation candidates (2026-07-18 brainstorm)

Accuracy levers, grounded in measured weaknesses:

| # | Item | Why / evidence | Cost |
|---|---|---|---|
| E1 | **Adverse-domain rounds (snow/night/rain)** | The unseen-drive demo showed snow segmented as sidewalk; mine domain slices by image statistics (brightness, wiper motion, white fraction) and oversample like C2 | S |
| E2 | **Self-training on label-less recordings** | 342 scenes have no CoMET autolabels; the r28 model can pseudo-label them (2D seg / boxes / TL) and the factory can build BEV GT from its own predictions — closes the loop to "any recording is training data" | M |
| E3 | **Temporal-consistency losses** (finish A2) | Box flicker: EMA/matching loss across the 3 memory slots at train time; runtime score EMA already trivially available | S |
| E4 | **Per-agent velocity readout + TTC** | Occupancy flow exists but boxes carry no velocity output; a 2-ch reg head gives the guardrail true TTC instead of 0.5 s stepping | S |
| E5 | **3D box height from occupancy** | Boxes are drawn with fixed height; the occ column already knows it — free supervision, better camera wireframes and truck handling | S |
| E6 | **Route/intent conditioning for E2E** | K=3 covers geometry, but mode CHOICE at intersections is unobservable without intent; feed a 3-way route token (from future ego GT at train time, from navigation at runtime) — turns the planner into a commandable one | M |
| E7 | **Confidence calibration for all heads** | Guard thresholds are hand-set; temperature-calibrate on val per head so VETO margins mean probabilities | S |
| E8 | **Student distillation for Orin** | ResNet-18 + half-res student distilled from the v35 teacher (feature + output distillation); pairs with C4 INT8 and the IPM sector mask | L |
| E9 | **Per-round video regression CI** | Render the fixed holdout scenes every round, diff metrics + frames automatically; catches user-visible regressions (flicker, phantom OCC) that scalar metrics miss | S |
| E10 | **VLM-based demo triage** | Run a vision-language model over each round's demo video to auto-flag anomalies (wrong-way arrows, phantom boxes) — scales the "user watches the demo" loop | M |
| E11 | **US 5% fine-tune round** | Zero-shot already works; measure how little US data closes the seg/TL gaps (transfer-efficiency experiment for new-region rollout) | S |
| E12 | **Overtake/lane-change scenario metrics** | The overtake holdout demo showed parked-row stationary flags flickering in unseen domains; add scenario-sliced eval (overtake, cut-in, crossing) on the 150-scene holdout | S |

Sequencing after r29 (B1–B4) and the full-corpus round: E3/E4/E5/E7 are
one-round bundles (small, probe-measurable); E2 and E6 are the two big
capability unlocks; E8+C4 is the deployment endgame.

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

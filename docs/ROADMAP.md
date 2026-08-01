# METEOR — candidate list for future rounds

## Implementation ledger (updated 2026-08-01)

### Shipped — model/training rounds
(🔴 = measured, significant win; effect column = held-out val, before → after)

| Round | Date | What went in | Measured effect |
|---|---|---|---|
| r20 | 07-15 | 🔴 **BN-stats isolation for the memory queue** | step-50 probe mIoU **0.213→0.265**, vehP **0.48→0.83** (collapse eliminated) |
| r20 | 07-15 | 🔴 **Per-epoch subset sampler** | corpus utilisation **14%→100%** at zero compute cost |
| r21 | 07-15 | 🔴 **eps-WTA + diverse mode init** | K=3 collapse fixed: 1 mode won 80/80 → 3 distinct hypotheses at intersections |
| r21 | 07-15 | Class-aware forecasting (det-feat concat, VRU x2.5) | vruADE improved; heading root-caused (75.9° → later fixes) |
| r22 | 07-16 | 🔴 **tl-w 0.6 (task-starvation fix)** | TL accuracy **0.28→0.86** |
| r22 | 07-16 | occ dynamic-shadow GT filter | near-ego phantom-vehicle GT (18% of near voxels) removed from supervision |
| r23 | 07-17 | 🔴 **Forecast motion residual + oncoming x2.5** | vehHead **34°→23°** (by r28); oncoming-flip regime (21–51%) broken |
| r23 | 07-17 | C6a optional LiDAR (depth sharpening) | +lidar ≈ +0.001 mIoU (small; enabled the C6b path) — single-weights dual-mode proven bit-equal |
| r25 | 07-17 | 🔴 **C6b LiDAR pillar branch** | +lidar delta **+0.001 → +0.004..0.007** and widening |
| r27 | 07-18 | 🔴 **Crossing-yaw weight + heading loss** | yaw **5.8°→4.7–5.1°**, direction flips **10%→8%** |
| r27 | 07-18 | 🔴 **TL red recovery** | red class **0.38→0.75** |
| r27 | 07-18 | Edge-weighted depth CE + top-mode display | object boundaries visibly sharp (display verified; IPM-side effect unquantified) |
| r28 | 07-18 | 🔴 **VRU direction-gate fix (2.0→1.0 m)** | vruHead **74°→63°** |
| r28 | 07-18 | C2 failure mining x2 | agentADE best **1.72 m** in the mined round (attribution shared with data growth) |
| r28→r32 | 07-18.. | v34 unknown rework + v3 GT (occluded→ignore) | [valUnk] **0.00 → P 0.28..0.69** (first useful signal; recall still low) |
| r29/r30 | 07-19 | 🔴 **B3 attention pooling + ADE pack (time weights)** | ADEc **0.53→0.41 m** (record) |
| r31 | 07-19 | 🔴 **E6 intent tokens + ego-w 1.2** | probeE2E ADE **0.78→0.67** |
| r31 | 07-19 | 🔴 **Consensus GT v2 (+ ceiling measurement)** | GT-vs-GT ceiling quantified (mIoU 0.452, laneline 0.19 → plateau explained); laneline probe **0.14→0.162** on clean labels |
| r31 | 07-19 | BEV rotation aug ±10°; distributed ADE probe | probe coverage **x8** at same wall time; rotation effect judged at r32 end |
| r32 | 07-20 | C1 in-training risk selection; longitudinal 2x; v(t) aux head | ADEc gate passed |
| r33/r34 | 07-21..22 | 🔴 **v39 decoupled E2E head** (phi x v composition) | ADEc **0.53→0.43 m** |
| r35 | 07-22 | 🔴 **Method A refiner grafts** (5-head post-hoc residual refiners trained frozen, grafted back trainable) | mIoU 0.324→0.325→(compounding each round) |
| r36 | 07-23 | 🔴 **v41 dense unknown head + BOX_FAR_W far-vehicle boost** | veh R **0.47→0.51**, mIoU 0.331 |
| r37/r38 | 07-24..25 | v42 VRU 25-45 m band; 3014 corpus (+1.3k scenes); unknown_v3 **camera-visibility filter** (median 59% of accumulated GT is occluded → per-cell don't-care) | valUnkD recall first non-zero; pixel recall 0.83 |
| r39 | 07-26 | 🔴 **v43 + intent losses; unknown_v3 live** | mIoU **0.334**, ADEc **0.39 m** (both records) |
| r40 | 07-27 | v44 **command→mode binding** (+8 logit boost = structural switch) | ep2 mIoU 0.335; stopped early for J6 round |
| r41 | 07-27 | **J6 7-camera fine-tune** (--zero-cams CAM_BACK_NARROW) | measured 7-cam cost: mIoU −0.002, veh R −6 pt |
| r42 | 07-28 | clean corpus (148 corrupt-GT scenes removed) + turning batch | round42/44 lists (7,707 / 7,795) |
| r43 | 07-28..29 | 🔴 **v45 SE(2) lateral-recovery augmentation + INT8 quant-noise** | mIoU **0.336** probe / 0.333 final, ADEc **0.33 m** (records at the time) |
| r43 refiner | 07-29 | 6-head refiner (unknown x2) | ADE **0.860→0.797**, far stopline **0.017→0.040** |
| r44 | 07-29..30 | 🔴 **v46 optional SD-map (free OSM) BEV prior** | ep2 mIoU **0.345** — highest BEV mIoU to date |
| r45 | 07-30 | **v47 per-camera box-level traffic-light input** + US corpus (420 scenes) | mIoU 0.335, **ADEc 0.31 m** (record); refiner (fp32) ADE 0.860→0.797 |
| r46 | 07-31 | 🔴 **dense stationary supervision** (whole box footprint vs one centre cell) + **MAE-like BEV DropBlock** | stationary recall **0.61→0.85** at P 0.85 (new `[valStat]` metric) |
| r47 | 08-01 | **v48 pseudo-LiDAR** — predict the LiDAR BEV raster from cameras, feed it back through the optional-LiDAR stem (inference ON/OFF, bit-equal when off) | running: PL occupancy IoU **0.29→0.40**, height MAE **1.67→1.14 m** |
| r47 | 08-01 | 🔴 **E2E command binding fixed** (docs/FIX_COMMAND_BINDING.md phase 1): WTA winner routed by the command, selector CE de-boosted | baseline measured: K=3 spread 1.01 m, sign reversal 6% → acceptance is >=5 m / >60%, re-measure pending |

Unresolved despite attempts: lane graph P/R 0.01 (B1 transformer decoder pending verdict), BEV lane mIoU (GT-noise-limited — see consensus GT), ADE absolute ≤0.5 (in progress, 0.67 now).

### Shipped — infrastructure / deployment
| Item | Date | Detail |
|---|---|---|
| L1 guardrails (C7) | 07-17..18 | spacetime collision / red-light / feasibility / drivable + MRM stop path; HOLD state; hood-ghost masking; demo --guard |
| Occupancy voxel-cube renderer | 07-17 | metric grid + shaded cubes (both demos) |
| Sharp depth (display + loss) | 07-17 | top-mode expectation; edge-weighted CE (r27+) |
| v36 ONNX export + engines | 07-19 | direct-trace exporter (ORT parity 1.1e-4); Python-API engine build (no trtexec); version/HW-compatible engines (--compat, Ampere+); one runtime for v29/v36 engines |
| Demo-identical TRT visualization | 07-19 | deploy/visualize.py shares the PyTorch demo's palettes/helpers (ribbon, agent trajectories, waypoint dots, guard/MRM) |
| C++ TensorRT runtime | 07-19 | deploy/cpp: engine build, t4dataset parsing, temporal ring, overlay video |
| Watchdog auto-recovery | 07-19 | crash-detect + relaunch from newest ckpt + completion markers (round chains gated on markers) |
| Drive-level test holdout | 07-18 | 150 scenes permanently excluded; curve/overtake unseen-drive demos |
| v41 TRT engine, fp16-safe by construction | 07-24 | stable softplus + scale-then-sum pooling (two fp16 overflow modes fixed); device-resident temporal ring (420→108 ms/frame, 9.3 FPS L40S); NaN probe 0/147 |
| --t4d root-mode batch inference | 07-24 | one engine + ONE combined video over a dataset root; NaN guards for bad odometry |
| Per-module TRT profiler | 07-25 | DETAILED-verbosity engine + IProfiler: BEV projection/encoder = 51%, all sparse heads ≈ 1.5% |
| Refiner NaN hardening | 07-26..27 | tanh-bounded residuals + input clamps on every refiner head; consecutive-skip detector; periodic (1k-step) saves; poisoned-BN forensics |
| unk2d BEV lift | 07-27 | 2D 'obstacle' detections lifted through predicted depth to BEV (instance-separated markers, cross-camera dedupe) |
| Corrupt-GT audit | 07-27 | 148/7,846 train scenes with consensus road collapse (sparse high-speed accumulation → spaghetti vectors); cleaned round42 list (7,707) |
| SD-map pipeline (offline OSM) | 07-30 | Geofabrik japan pbf -> 1,923 local Overpass-JSON tiles (public endpoints rate-limited us out); per-scene SE(2) alignment against GT road: class-3 crosswalk anchors, corridor score, analytic rotation seed, side-street openings, Theil-Sen drift model; scale pinned to 1 (fitting it on ~90 m-quantised GNSS shrank the map 8-23%); alignment gate (road-IoU<0.15 -> zeros = prior off). 5,109 scenes rasterised at 1,644 scenes/h |
| Traffic-light label extraction | 07-31 | dataset lamp elements are separate `color_shape` categories (red_circle / green_arrow / ...) with row-level arrow `orientation` (0=up, +pi/2=right, verified on 512 real crops); 4,556 scenes, val coverage 273/273 |
| TL + sign GT demos | 07-31 | camera lamp boxes with orientation glyphs + BEV signal/speed-limit/stop icons; circle lamps set the state, arrows are drawn as separate limited permissions (a green arrow is not a green light) |
| US corpus ingest | 08-01 | idempotent `ingest_us.py`: symlink, consensus GT, quality gates (moth-eaten gt_cons, camera-incomplete frames), auto-repair when the provider re-converts and wipes our gt_cons; 734 scenes accepted (round48 = 8,528) |
| BN-poisoning guard | 07-31 | BN running stats update in FORWARD, so one inf batch poisons eval permanently while training looks fine (r47 lost its E2E metric this way, and the next save would have shipped the poison): snapshot/restore on non-finite loss, periodic `sanitize_bn`, init-time repair |
| OOM-tolerant diagnostics | 07-31 | probes and epoch-end val free the step's activations first and survive OOM instead of killing a multi-day round; val loaders keep batch>=2 independent of --batch (at batch 1 the capped evals halved coverage and ADEc went nan) |
| Dataloader robustness | 08-01 | camera-incomplete frames filtered at index build; unreadable-sample fallback is a bounded loop (recursion hit Python's 1000-frame limit and took down the round) |
| Throughput tuning | 08-01 | measured GPU utilisation 55.9% -> 62-79%: batch 2 (amortises the DDP allreduce, removes the need for SyncBN), workers 4 (validated against the historical shared-memory failure), probes every 1000 steps (they cost ~25% of wall time at 500) |
| Checkpoint provenance | 08-01 | ckpts record `args` + git hash (r45's had neither, which is why "were the intent flags on?" was unanswerable) |
| GPU sizing estimate | 07-31 | 1,000 h x 10 epochs: 8 GPUs 92 d / 32 GPUs 23 d / **64 GPUs ~12 d**; 48 GB-class cards required at batch 2 (out/GPU_estimate.pptx) |
| Recovery-augmentation GT recipe (v45) | 07-27 | departed-viewpoint synthesis + pursuit/record recovery targets (auto fallback, curvature-aware extrapolation, 82% pursuit adoption) — GT demos verified |

### Planned
| Item | Target | Detail |
|---|---|---|
| r48 = reinforcement stage | after r47 + refiner | `--rl-w`: rule-based rewards over the K=3 candidates (drivable area, time-resolved agent collision, comfort, speed-normalised progress, red-light compliance from the v47 TL input) + GRPO-style group-relative policy loss on the mode logits. No critic, no simulator, zero new params; the reward embeds the imitation error so a rule-compliant but absurd candidate cannot win. Scripts armed (chain_r48_start.sh) |
| Command-binding acceptance | during r47 | `eval_command_binding.py`: B >= 5.0 m spread and > 60% sign reversal, with ADE/ADEc not regressing. If phase 1 falls short, enable phase 2 (`--intent-wrong 0.15`, counterfactual commands with the waypoint target dropped) |
| Traffic-light input, measured | r48 | the A/B probe shows no seg/E2E effect yet, as expected: the TL input should move TL-state accuracy and red-light stopping behaviour. Needs a dedicated probe (TL accuracy ON vs OFF, E2E on signal-approach frames) and wider label coverage (currently 50% of training scenes) |
| Pseudo-LiDAR value, measured | after r47 | three-arm A/B on one checkpoint: camera-only / +pseudo raster / +real sweep. Report the geometry gain (PL occupancy IoU is 0.40 and still climbing) against the tasks it is supposed to lift |
| Perf ablation matrix | GPU-idle windows | docs/perf_analysis_plan.md: 7-cam engine (r41 weights), embedded head-set, INT8 PTQ (r43 quant-noise vs r42 baseline), asymmetric BEV grid (80 m fwd / 40 m back) |
| Chunked depth-gated projection | perf round | measured: the projection intermediates, not the backbone, dominate the 25 GB peak (grad-ckpt on the backbone only bought 1 GB). Per-camera chunking would free enough for larger batches / higher resolution — touches an exported core function, so it needs its own round |
| Unknown peak decode | no retrain | connected-components -> local-maxima decode; object-level P/R re-measure; fuse with the unk2d lift |
| GT re-render for the 148 corrupt scenes | factory idle | hole-filled accumulation at high speed, re-vectorize, re-consensus |
| J6 deployment package | after matrix | 7-cam export + INT8 engine + guardrail; Orin target |
| Source-data snapshotting | data ops | the US batch is rewritten in place while we train on symlinks: one re-conversion silently dropped 419 scenes from training and a mid-write read killed a round. Either snapshot on ingest or keep running the validating ingest before every round |

Living list of what we could do next, why, and what it would cost. Nothing
here is committed work; each entry is sized so it can be picked up
independently. Ordered within each section by (expected value ÷ risk).

**Baseline to beat** (best measured per metric, held-out recording day):

| metric | value | where |
|---|---|---|
| BEV lane mIoU | **0.345** | r44 ep2 (v46, SD-map) |
| 2D seg mIoU (21 cls) | 0.555 | r39 |
| 3D det veh P / R / Rn / yaw / dir-flips | 0.78 / 0.51 / 0.71 / 3.7° / 6% | r36-r39 |
| E2E ADE / **ADEc** / FDE | 0.89 m / **0.31 m** / 1.98 m | r45 (v47) |
| Stationary flag P / R | **0.85 / 0.85** | r46 (dense supervision; was 0.93/0.61) |
| TL state accuracy | 0.86 | r22+ |
| Pseudo-LiDAR occupancy IoU / z MAE | 0.40 / 1.14 m | r47, in progress |
| Refiner delta (E2E ADE, far stopline) | −4.4% / 0.017→0.040 | r43/r45 refiners |
| SD-map prior gain (road IoU, intersections >20 m) | +1.0 pt, OFF bit-equal | r45 A/B probe |
| Corpus | 8,528 scenes (7,795 JP + 734 US) | round48 list |

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

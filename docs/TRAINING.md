# Training METEOR

## Command

```bash
CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7 torchrun --nproc_per_node=7 bevlane/train.py \
  --model v20 --batch 2 --epochs 24 --workers 0 \
  --gt-key gt_vec --root out/bevlane --train-list scenes.txt \
  --limit-train 46000 --train-bg --aug --lr 1e-4 \
  --init-ckpt out/prev_round/best.pt \
  --seg-w 1.0 --dice-w .5 --lovasz-w .5 --boundary-w 3 --tversky-w .6 --far-w 1 \
  --depth-w 0.6 --box-w 0.8 --seg2d-w 0.4 --bbox2d-w 0.3 --ego-w 0.5 --occ-w 0.5 \
  --seg2d-key seg2d21 --n-seg2d 21 --out out/ckpt_r<N>
```

## Loss

```
L = 1.0 · BEV lane   (class-weighted CE(bg 0.5) + Lovász + boundary ×3
                      + Tversky(lines) + Dice(crosswalk) + far-row weighting)
  + 0.6 · depth      (smoothed bin-CE + 0.1 · expected-depth L1)
  + 0.8 · 3D boxes   (Gaussian focal + centre L1; VRU channel ×2.5,
                      positives within 20 m ×2)
  + 0.4 · 2D seg     (class-weighted CE: lane/marking ×4, pole/sign/light ×2,
                      VRU ×1.5, bg ×0.4; side cameras ×1.6; ignore-safe)
  + 0.3 · 2D boxes   (3-scale focal + L1; per-class positive weights:
                      obstacle ×3, two-wheelers/lights ×2, ped/sign ×1.5)
  + 0.5 · E2E        (waypoint L1 with lateral ×4 + curvature sample weight
                      1+|lat@3s|/1.5 + steering L1 ×2 + accel L1 + brake BCE)
  + 0.5 · occupancy  (class-weighted CE: free ×0.2, vehicle ×2,
                      obstacle ×3, ped/two-wheeler ×4; 255 ignored)
```

Why the unusual weights exist (each fixed a measured failure):

- **E2E curvature weighting** — 83 % of frames are near-straight and
  longitudinal targets are ~18× larger than lateral ones; plain L1 learned a
  go-straight prior (curve ADE 2.3 m vs 0.6 m straight). Lateral ×4 + curve
  sample weighting brought curve ADE to 0.72 m in one round.
- **Ignore-safe aux losses** — during rolling GT re-extraction whole batches
  can be all-ignore; mean-CE returns NaN and poisons training within 50 steps.
  Every aux loss returns a graph-preserving zero in that case.
- **Near-range 3D weighting / small-class 2D weighting** — targeted fixes for
  near VRUs and rare small objects (cones, lights).

## Rolling rounds

Training runs in ~24 h rounds. Each round: refresh the scene list with newly
converted recordings, `--init-ckpt` from the previous best, retrain all tasks
jointly. GT extractors run concurrently; the dataset tolerates missing
modalities (ignore fallback), so new supervision streams in mid-round.

The cadence is deliberately aggressive — **sixteen rounds (and six new task
heads) shipped in the first five days** — while the same loop is designed to
run for months: the data factory keeps converting recordings, every round
folds them in, and heads/losses evolve without ever restarting from scratch.

| Round | Date | Model | Change | BEV mIoU | Notes |
|---|---|---|---|---|---|
| r2 | 2026-07-11 | v16 | 4-task baseline (12-cls seg) | 0.283 | |
| r3 | 2026-07-12 | v16 | 21-class CSV seg taxonomy | 0.285 | |
| r4 | 2026-07-12 | v16 | +DTSET data | 0.286 | |
| r5 | 2026-07-12 | v17 | +10-class 2D detection | 0.286 | |
| r6 | 2026-07-12 | v16 | lane/ego GT fix | 0.286 | 2D-seg lane IoU 0.371 |
| r7 | 2026-07-12 | v18 | +E2E head | **0.287** | first ADE 0.95 m |
| r8 | 2026-07-12→13 | v19 | capacity re-balance, KMAX 96 | **0.292** | 2D-seg mIoU 0.451 |
| r9 | 2026-07-13 | v20 | +occupancy, curve-weighted E2E | 0.290 | curve ADE 2.31→0.72 m |
| r10 | 2026-07-13 | v20 | small-object / near-VRU / side-cam weights | 0.289 | 2D-seg mIoU 0.466 |
| r11 | 2026-07-13 | v21 | +one-shot agent forecasting | 0.288 | agentADE 2.17 m |
| r12 | 2026-07-13→14 | v22 | +streaming temporal BEV | 0.276 | ADEc 0.65 m, agentADE 1.76 m |
| r13 | 2026-07-14 | v23 | LaneDecED, det tower, tfuse zero-init, 8-ep fast-decay LR | 0.275 | E2E ADE 0.63 m (best) |
| r14 | 2026-07-14 | v25 | task routing: geometry on RAW BEV, motion on FUSED | 0.275 | val2d 0.466 |
| r15 | 2026-07-14 | v26 | +stationary flag; near-range-first 3D det (VRU GT relax, far damp) | 0.285 | VRU R 0.08→0.25, Rn veh 0.68 |
| r16 | 2026-07-14→15 | v26 | yaw pkg (3×3 reg targets, yaw-weighted loss); box/ego re-prioritised | (running) | yaw 8.3° baseline |

## Validation metrics (printed every epoch)

- `[val ep]` — BEV lane per-class IoU + road/edge precision-recall
- `[val2d ep]` — 2D seg IoU (bg / marking / road / sidewalk / lane / pole)
- `[valE2E ep]` — trajectory ADE, **ADEc** (curve subset |lat@3s| > 2 m), FDE,
  steering MAE, accel MAE, brake accuracy
- `[valOCC ep]` — occupancy IoU (free / vehicle / ped / road / veg / building)
- `[val3D ep]` — 3D det P/R per class + **R50** (< 50 m) + **Rn**
  (near corridor: < 30 m, |lat| < 12 m) + centre error + **yaw axis error /
  direction-flip rate** on matched boxes
- `[valTraj ep]` — agent-forecast ADE/FDE at GT centres + **statAcc**
  (learned parked/stopped flag vs GT |disp@3s| < 0.5 m)

## Operational notes (hard-won)

- **Never train with `--depth-w 0` for long** — the depth head degrades
  irreversibly within an epoch (3.1 → 6.5 m MAE); always train jointly.
- **DDP + depth GT requires `--workers 0`** — worker shared-memory collate
  crashes ("resize storage that is not resizable") with the extra per-sample
  tensors.
- **Never change GT tensor shapes while a run is reading them** — mixed-shape
  collate kills the run; the dataset now normalizes bbox2d to KMAX=96
  defensively.
- Fresh CenterNet-style heads start with a large focal loss (hundreds) that
  collapses within ~200 steps — expected, not divergence.
- Checkpoints: `last.pt` every epoch, `best.pt` by BEV mIoU. Resume with
  `--init-ckpt <ckpt>` (shape-mismatched heads are auto-dropped, so class-count
  changes retrain only the affected head).

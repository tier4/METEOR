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
  go-straight prior (curve error roughly 4× the straight-road error). Lateral ×4 + curve
  sample weighting closed most of that gap in one round.
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

| Round | Date | Model | Change |
|---|---|---|---|
| r2 | 2026-07-11 | v16 | 4-task baseline (12-cls seg) |
| r3 | 2026-07-12 | v16 | 21-class CSV seg taxonomy |
| r4 | 2026-07-12 | v16 | +DTSET data |
| r5 | 2026-07-12 | v17 | +10-class 2D detection |
| r6 | 2026-07-12 | v16 | lane/ego GT fix |
| r7 | 2026-07-12 | v18 | +E2E head |
| r8 | 2026-07-12→13 | v19 | capacity re-balance, KMAX 96 |
| r9 | 2026-07-13 | v20 | +occupancy, curve-weighted E2E |
| r10 | 2026-07-13 | v20 | small-object / near-VRU / side-cam weights |
| r11 | 2026-07-13 | v21 | +one-shot agent forecasting |
| r12 | 2026-07-13→14 | v22 | +streaming temporal BEV |
| r13 | 2026-07-14 | v23 | LaneDecED, det tower, tfuse zero-init, 8-ep fast-decay LR |
| r14 | 2026-07-14 | v25 | task routing: geometry on RAW BEV, motion on FUSED |
| r15 | 2026-07-14 | v26 | +stationary flag; near-range-first 3D det (VRU GT relax, far damp) |
| r16 | 2026-07-14→15 | v26 | yaw pkg (3×3 reg targets, yaw-weighted loss); box/ego re-prioritised |

## Validation metrics (printed every epoch)

- `[val ep]` — BEV lane per-class IoU + road/edge precision-recall
- `[val2d ep]` — 2D seg IoU (bg / marking / road / sidewalk / lane / pole)
- `[valE2E ep]` — trajectory ADE, **ADEc** (curve subset |lat@3s| > 2 m), FDE,
  steering MAE, accel MAE, brake accuracy
- `[valE2Ed ep]` — the same ADE **decomposed**: oracle (best of the K = 3
  hypotheses, so `ADE − oracle` is what mode *selection* costs), moving vs
  stopped, and a constant-velocity baseline. One ADE number cannot tell you
  which of those moved; the decomposition can. On one round about **a quarter of the error was lost to picking the wrong
  hypothesis**, and the model beat "keep going straight at this speed" by
  less than a quarter.
- `[valE2E-ema ep]` — the same metrics under the averaged weights, printed
  whenever `--ema` is on (see below)
- `[valOCC ep]` — occupancy IoU (free / vehicle / ped / road / veg / building)
- `[val3D ep]` — 3D det P/R per class + **R50** (< 50 m) + **Rn**
  (near corridor: < 30 m, |lat| < 12 m) + centre error + **yaw axis error /
  direction-flip rate** on matched boxes
- `[valTraj ep]` — agent-forecast ADE/FDE at GT centres + **statAcc**
  (learned parked/stopped flag vs GT |disp@3s| < 0.5 m)

## Weight averaging: EMA and soups

Both do the same thing -- average weights instead of trusting one point on the
trajectory -- and both were adopted because the weights here demonstrably do
not sit still.

### What the averaging is for

Consecutive epoch-end evaluations bounce while nothing about the data changes.
one round's ADE moved by ±3 % over its last four epochs; the VLA runs
on the other machine swing by ±40 % *inside 1000 steps*.
That is not learning and unlearning, it is the optimiser orbiting the bottom of
a basin: each SGD step is a noisy estimate of the gradient, so the weights
random-walk around the minimum instead of resting in it. The centre of that walk
is a better model than any point on it, and averaging is how you get the centre.

### `--ema <decay>` — the average along ONE run

Keeps a shadow copy of every trainable parameter and, after each optimiser step:

    shadow = decay * shadow + (1 - decay) * weights

At `--ema 0.999` that is a low-pass filter with an effective window of
`1/(1-0.999)` = **1000 steps**: the shadow is roughly the average of the last
thousand steps' weights, with older ones fading geometrically.

- **Cost.** One fp32 copy of the trainable set — 53.2 M parameters, 217 MB per
  rank for v52 — and one multiply-add per parameter per step. No extra forward
  or backward.
- **What it does NOT average.** Only `named_parameters()`, so the 330 BatchNorm
  `running_mean` / `running_var` buffers (0.06 M values) are left alone; the
  averaged weights are evaluated against the raw model's BN statistics. That is
  the standard recipe and it is what makes swapping cheap, but it is the part
  most likely to misbehave if the two ever drift far apart.
- **It is evaluated, not assumed.** Every epoch the raw weights are scored, then
  the shadow is swapped in, scored on the identical slice, and swapped back --
  `[valE2E ep]` and `[valE2E-ema ep]`. `best_e2e.pt` takes whichever won. This
  matters: early in one run the EMA scored 50 % worse than the raw weights, because early in a run the average is still dragged by weights the run
  has already improved on.
- **Measured here.** In mature rounds the EMA weights beat the raw weights by
  3–5 % on ADE, FDE and selection loss on the same slice, on both machines.

### `bevlane/make_soup.py` — the average ACROSS runs

Uniform mean of several checkpoints that share one architecture. Well-posed only
when the runs are warm-start descendants of each other, so they sit in the same
basin and the average is not interpolating between unrelated solutions -- r59,
r60 and r61 qualify (each started from the previous one's best).

Measured on val, every soup beat every one of its ingredients for free (ADE −2 to −3 %,
FDE and selection loss along with it).

Note what moved: not just ADE but the **selection loss (−15 %)**.
Averaging steadies the mode logits, which is part of what a training-side fix
for selection was meant to buy.

Unlike the EMA, the soup averages BN buffers too (integer counters such as
`num_batches_tracked` take the last value rather than a fraction). That is
defensible only because the runs saw identical data and augmentation.

The two are orthogonal -- EMA averages along a run, the soup averages across
runs -- so EMA checkpoints can themselves be souped.

## Operational notes (hard-won)

- **Never train with `--depth-w 0` for long** — the depth head degrades
  irreversibly within an epoch (depth MAE doubles); always train jointly.
- **DDP + depth GT requires `--workers 0`** — worker shared-memory collate
  crashes ("resize storage that is not resizable") with the extra per-sample
  tensors.
- **Never change GT tensor shapes while a run is reading them** — mixed-shape
  collate kills the run; the dataset now normalizes bbox2d to KMAX=96
  defensively.
- Fresh CenterNet-style heads start with a large focal loss (hundreds) that
  collapses within ~200 steps — expected, not divergence.
- **Uptime is not progress.** A round can run to completion while discarding
  every step: r59 threw away 23,533 of 34,280 (0 % up to step 8k, then 98-100 %
  for the rest) with the process alive and val still printing plausible numbers.
  The tell is a metric that sits too still — a mIoU that moves by less than 0.003
  across six epochs is the same model measured six times. Check
  `grep -c "SKIP non-finite"` over the WHOLE log, localise with
  `METEOR_MODPROBE=1`, repair with `bevlane/renorm_convbn.py` (a conv feeding a
  BatchNorm has a free scale, nothing keeps it near 1, and it drifts until the
  fp16 output passes 65504). `out/supervise_r*.sh` now trips automatically at a
  >50 % discard rate — it has already fired once on the training server and recovered.
- Checkpoints: `last.pt` every epoch, `best.pt` by BEV mIoU. Resume with
  `--init-ckpt <ckpt>` (shape-mismatched heads are auto-dropped, so class-count
  changes retrain only the affected head).

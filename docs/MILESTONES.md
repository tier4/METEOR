# METEOR — Milestones (2026-07-19)

## M1. Perception/planning quality on the current platform (now – ~1 week)
- **r31 (running)**: consensus GT v2 + intent tokens + rotation aug + 4.2k
  scenes. Gate: ADE trend with intent, road_edge on clean labels.
- **r32 full-corpus round**: all ~5.1k scenes after DTSET conversion
  completes; judge data-scaling on the clean-GT metric.
- **ADE <= 0.5 m** (then 0.3s): intent conditioning at runtime (navigation
  feed), selection-gap closure; ego-GT smoothing if the 0.3s wall appears.
- **Unknown detection usable**: decode-threshold calibration (P/R trade),
  target P>=0.5 / R>=0.3.
- **Small-bundle round (E3/E4/E5/E7)**: temporal-consistency loss, per-agent
  velocity+TTC, box height from occupancy, confidence calibration.

## M2. Trustworthy evaluation (parallel, ~1 week)
- Boundary-tolerant thin-class metrics (3b) on the consensus GT.
- Guardrail intervention-rate eval (correct vs false interventions).
- Scenario-sliced holdout evals (overtake / crossing / curves) on the
  150-scene drive-level holdout; per-round video regression CI (E9).

## M3. Lane graph & downstream capabilities (~2 weeks)
- B1 query decoder must leave P/R 0.01 (judge in r31; else iterate).
- C3 per-lane TL association once B1 works; C1 risk-scored planning is
  already runtime — integrate into the product path.
- A5 lane-graph GT cleanup (ego-connected drivable filter).

## M4. Deployment productisation (~2-4 weeks)
- INT8 (C4) + IPM sector masking: Orin-time budget (~60% is the IPM).
- Student distillation (E8): ResNet-18 / half-res for embedded targets.
- C++ runtime hardening: multi-scene loop, live display, pose from
  odometry when annotation JSONs are absent (true vehicle mode).
- Version/hardware-compatible engines already in place — validate on a
  real Orin.

## M5. Data engine (continuous)
- GT factory 3c/3d: sub-cell alignment, pose re-smoothing (the real
  ceiling lift for thin classes; factory re-run, days).
- E2 self-training: pseudo-label the 342 autolabel-less scenes; then any
  recording becomes training data.
- E1 adverse-domain mining (snow/night/rain) — snow seg failure observed.
- E11 US 5% fine-tune: transfer efficiency for new-region rollout.

## M6. Long-term research bets (1-3 months)
- Route-conditioned closed-loop evaluation (log-replay counterfactuals
  with the guardrail as safety net).
- World-model-lite planning: occupancy-flow rollout scoring beyond C1.
- B4-full agent interaction via a fixed-K query decoder (after B1).
- Continual rolling rounds as an operations product: watchdog + round
  chains + failure mining + video CI with zero human intervention.

**North star**: camera-first 12-task driving stack, trained by its own
label factory on any fleet recording, deployable to Orin from one
checkpoint (camera-only or +LiDAR), with deterministic guardrails and a
self-measuring evaluation loop — no human labels, no human code.

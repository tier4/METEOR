#!/usr/bin/env python3
"""Quasi-closed-loop evaluation: chained-rollout drift, recovery quality, and
guardrail intervention rates.

Open-loop ADE 0.5 does not certify closed-loop behaviour: the model's own
error moves it into states the training set never showed, and per-frame ADE is
blind to how fast that compounds. True closed loop needs re-rendered images
from the displaced pose, which log replay cannot produce -- these three probes
capture most of the failure surface without it:

  CHAIN   follow the model's own selected plan for one frame step (0.4 s at
          the 5 Hz stride-2 cadence, interpolated between the 0.5 s
          waypoints), accumulate the predicted displacement chain, and compare
          against the accumulated GT chain over N steps. Reported as drift
          per horizon: the COMPOUND rate a single-frame ADE hides.
  RECOV   the lat-aug recovery skill, finally measured: shift the DECISION
          point laterally (the model was trained with --lat-aug to return
          smoothly from offsets) and check how much of the offset the plan
          removes by 1.5 s / 3.0 s. No re-rendering: the offset is applied to
          the plan-consumer side, so this measures the TARGET the planner
          drives toward, which is exactly what its recovery training shaped.
  GUARD   bevlane/guardrail.py L1 gates over every valid frame: intervention
          rate, verdicts, reasons. The ROADMAP has carried "intervention-rate
          eval on val" as a next step since the guardrail landed.

    METEOR_BEV_XR=40.0 python3 bevlane/closed_loop_eval.py \
        --ckpt out/v59_remote_last.pt --model v55 --n-cams 7
"""
import argparse
import os
import sys
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bevlane.dataset import BevLaneDataset                      # noqa: E402
from bevlane.guardrail import check_path                        # noqa: E402
from bevlane.model import EGO_K, MODELS, enable_kinematic_anchor                         # noqa: E402

STEP_S = 0.4                    # one evaluation step = 2 stored frames @5 Hz


def sel_wp(ego):
    lg = ego[12 * EGO_K:12 * EGO_K + EGO_K]
    k = int(np.argmax(lg))
    return ego[k * 12:(k + 1) * 12].reshape(6, 2), k


def wp_at(wp, t):
    """Interpolate the 0.5 s-spaced waypoints at time t (ego frame)."""
    i = t / 0.5
    lo = int(np.floor(i)) - 1
    if lo < 0:
        return wp[0] * (t / 0.5)
    hi = min(lo + 1, 5)
    f = i - (lo + 1)
    return wp[lo] + (wp[hi] - wp[lo]) * f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", default="v55")
    ap.add_argument("--n-cams", type=int, default=7)
    ap.add_argument("--scenes", type=int, default=60)
    ap.add_argument("--chain", type=int, default=8,
                    help="chained steps (8 x 0.4 s = 3.2 s)")
    a = ap.parse_args()

    scenes = [l.strip() for l in open("val.lst") if l.strip()][:a.scenes]
    ds = BevLaneDataset(os.environ.get("METEOR_BEV_ROOT", "out/bevlane"),
                        scenes, gt_key="gt_cons",
                        with_ego=True,
                        n_cams=a.n_cams, max_per_scene=24,
                        trim_start=3, trim_end=10)
    from bevlane.probe_net import load_full
    m = load_full(a.ckpt, mv=a.model)   # 2026-09-05: 取りこぼしなし構築 (静かな故障 #10)

    # index frames per scene so chains walk stride-2 through real frames
    by_scene = {}
    for idx, (s, f) in enumerate(ds.items):
        by_scene.setdefault(s, []).append(idx)

    drift = np.zeros(a.chain)
    drift_n = np.zeros(a.chain)
    recov = Counter()
    guard = Counter()
    reasons = Counter()
    n_guard = 0

    cache = {}

    def infer(idx):
        if idx in cache:
            return cache[idx]
        b = ds[idx]
        if b is None:
            cache[idx] = None
            return None
        with torch.no_grad(), torch.autocast("cuda", torch.float16):
            out = m(b[0][None].cuda(), b[1][None].cuda(), b[2][None].cuda(),
                    b[4][12][None].cuda())      # v0 conditioning, as in eval
        # 出力は CPU に移してキャッシュする (2026-09-04)。GPU のまま持つと
        # フレーム数に比例して VRAM が増え、120 シーンで 66 GB、240 シーンで
        # OOM した (GUARD 段が 0 フレームだと pop されず全フレーム残る)。
        out = tuple(o.float().cpu() if torch.is_tensor(o) else o for o in out)
        cache[idx] = (b, out)
        return cache[idx]

    for s, idxs in by_scene.items():
        idxs = idxs[::2]                    # 0.4 s apart
        for st in range(0, max(len(idxs) - a.chain, 0), a.chain):
            pred_pos = np.zeros(2)
            gt_pos = np.zeros(2)
            ok = True
            for ci in range(a.chain):
                r = infer(idxs[st + ci])
                if r is None:
                    ok = False
                    break
                b, out = r
                e = b[4]
                if float(e[16]) <= 0:
                    ok = False
                    break
                ego = out[7].float()[0].cpu().numpy()
                wp, _ = sel_wp(ego)
                pred_pos += wp_at(wp, STEP_S)          # ego-frame step
                gtwp = e[:12].view(6, 2).numpy()
                gt_pos += wp_at(gtwp, STEP_S)
                if ok:
                    d = float(np.linalg.norm(pred_pos - gt_pos))
                    drift[ci] += d
                    drift_n[ci] += 1
            if not ok:
                continue

    # ---- recovery + guardrail on a strided subset ----
    flat = [i for v in by_scene.values() for i in v][::4][:400]
    for idx in flat:
        r = infer(idx)
        if r is None:
            continue
        b, out = r
        e = b[4]
        if float(e[16]) <= 0:
            continue
        ego = out[7].float()[0].cpu().numpy()
        wp, _ = sel_wp(ego)
        for off in (0.5, 1.0):
            # perception-side offset, exactly the training lat-aug transform:
            # inject dy into the extrinsics (Tc' = Tc @ R, R[1,3]=dy) so the
            # model SEES itself displaced; recovery = fraction of the offset
            # the selected plan removes by 1.5 / 3.0 s (target y = -off)
            Rm = torch.eye(4, dtype=b[2].dtype)
            Rm[1, 3] = off
            with torch.no_grad(), torch.autocast("cuda", torch.float16):
                o2 = m(b[0][None].cuda(), b[1][None].cuda(),
                       (b[2] @ Rm)[None].cuda(), b[4][12][None].cuda())
            wp2, _ = sel_wp(o2[7].float()[0].cpu().numpy())
            recov[f"off{off}_1.5s"] += 1 - min(abs(wp2[2, 1] + off) / off, 1.5)
            recov[f"off{off}_3.0s"] += 1 - min(abs(wp2[5, 1] + off) / off, 1.5)
        recov["n"] += 1
        # guardrail: build occ_pred the demo's way (uint8 class map on CPU)
        occ_pred = None
        if len(out) > 8:
            op = out[8][0].float().softmax(0).cpu()
            conf = 1.0 - op[0]
            cls = (op[1:].argmax(0) + 1).to(torch.uint8)
            thr = torch.where((cls == 7) | (cls == 8),
                              torch.full_like(conf, 0.92),
                              torch.full_like(conf, 0.55))
            occ_pred = torch.where(conf > thr, cls,
                                   torch.zeros_like(cls)).numpy().astype(np.uint8)
        dets = m.decode_boxes(out[3].float().cpu(), out[4].float().cpu(),
                              thresh=0.3)[0]
        dets = [[float(v) for v in d] for d in dets]
        # predicted futures for other agents, as the demo/guardrail does --
        # zeros here made every same-lane lead vehicle a phantom collision
        traj_map = out[9][0].float().cpu() if len(out) > 9 else None
        offs = []
        for d_ in dets:
            rr0 = int((80.0 - d_[2]) / 0.4)
            cc0 = int((50.0 - d_[3]) / 0.4)
            o_ = np.zeros((6, 2), np.float32)
            if traj_map is not None and 0 <= rr0 < traj_map.shape[-2] \
                    and 0 <= cc0 < traj_map.shape[-1]:
                v_ = traj_map[:, rr0, cc0]
                if v_.numel() >= 39:
                    kb_ = int(v_[36:39].argmax())
                    o_ = v_[kb_ * 12:(kb_ + 1) * 12].view(6, 2).numpy()
                elif v_.numel() >= 12:
                    o_ = v_[:12].view(6, 2).numpy()
            offs.append(o_)
        lane = out[0].float()[0].argmax(0).cpu().numpy()
        g = check_path(wp, occ_pred, dets, offs,
                       np.array([1.0, 0, 0, 0], np.float32), lane,
                       float(e[12]))
        guard[g["verdict"]] += 1
        if g["verdict"] not in ("OK", "HOLD"):
            reasons[g["reason"]] += 1
        n_guard += 1
        cache.pop(idx, None)

    print(f"=== CHAIN: 連鎖ロールアウトの累積乖離 (自計画追従 vs GT) ===")
    for ci in range(a.chain):
        if drift_n[ci]:
            t = STEP_S * (ci + 1)
            print(f"  +{t:.1f}s  {drift[ci] / drift_n[ci]:.3f} m "
                  f"(n={int(drift_n[ci])})")
    print(f"\n=== RECOV: 横オフセット復帰率 (1.0=完全復帰) ===")
    nn = max(recov["n"], 1)
    for k in sorted(recov):
        if k != "n":
            print(f"  {k}: {recov[k] / nn:.2f}")
    print(f"\n=== GUARD: ガードレール判定 ({n_guard} frames) ===")
    for k, v in guard.most_common():
        print(f"  {k}: {v} ({100 * v / max(n_guard, 1):.1f}%)")
    for k, v in reasons.most_common(5):
        print(f"    介入理由: {k} x{v}")


if __name__ == "__main__":
    main()

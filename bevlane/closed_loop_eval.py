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
    ap.add_argument("--list", default="val.lst",
                    help="scene list file (default val.lst; out/heldout549.lst = val+test, 549 scenes)")
    ap.add_argument("--intent", action="store_true",
                    help="feed the route intent (straight/left/right one-hot derived from the GT future exactly as "
                         "training does: any 1.5-3 s waypoint beyond +-2 m) to the planner. Without it the model runs "
                         "with no navigation, as the demo does")
    ap.add_argument("--chain", type=int, default=8,
                    help="chained steps (8 x 0.4 s = 3.2 s)")
    ap.add_argument("--h2-graft", default="",
                    help="v203: load the H2 head GRAFTED onto the model (enable_hist_traj) instead of a standalone head; value = h2 ckpt")
    ap.add_argument("--h2", default="",
                    help="HistoryTrajHead checkpoint (out/h2_*/best.pt): apply the history-conditioned bounded "
                         "trajectory residual to the selected waypoints, exactly as deployment would")
    ap.add_argument("--h2-cache", default="",
                    help="causal replay cache dir (bevlane/build_track_cache.py output) supplying the per-frame "
                         "object history for --h2. Frames missing from the cache run with history disabled (= baseline)")
    a = ap.parse_args()

    scenes = [l.strip() for l in open(a.list) if l.strip()][:a.scenes]
    ds = BevLaneDataset(os.environ.get("METEOR_BEV_ROOT", "out/bevlane"),
                        scenes, gt_key="gt_cons",
                        with_ego=True,
                        n_cams=a.n_cams, max_per_scene=24,
                        trim_start=3, trim_end=10)
    from bevlane.probe_net import load_full
    m = load_full(a.ckpt, mv=a.model)   # 2026-09-05: lossless build (silent failure #10)
    if os.environ.get("METEOR_F3_HEAD"):
        from bevlane.f3_refine import enable_f3_refine
        enable_f3_refine(m, os.environ["METEOR_F3_HEAD"])

    # ---- H2: history-conditioned trajectory residual (guide Stage C), fed from the causal replay cache ----
    h2_head = None
    h2_cache = {}
    h2_stat = {"hit": 0, "miss": 0}
    if a.h2_graft and not a.h2:
        a.h2 = a.h2_graft                     # same cache/eval path, head lives on the model
    if a.h2:
        from bevlane.history_encoder import HistoryTrajHead
        from bevlane.history_schema import INPUT_NAMES as _HKEYS
        if a.h2_graft:
            from bevlane.model import enable_hist_traj
            h2_head = enable_hist_traj(m, a.h2_graft)
        else:
            _ck = torch.load(a.h2, map_location="cpu", weights_only=False)
            h2_head = HistoryTrajHead(bound_m=float(_ck.get("args", {}).get("bound", 2.0)))
            h2_head.load_state_dict(_ck["head"]); h2_head.cuda().eval()
            print(f"[h2] head {a.h2} (ep {_ck.get('ep')}, val {_ck.get('val')})", flush=True)

        def _h2_rows(scene):
            if scene not in h2_cache:
                import numpy as _np
                fp = os.path.join(a.h2_cache, scene + ".npz")
                if os.path.isfile(fp):
                    z = _np.load(fp)
                    h2_cache[scene] = ({int(f): i for i, f in enumerate(z["frame"])},
                                       {k: z[k] for k in _HKEYS})
                else:
                    h2_cache[scene] = None
            return h2_cache[scene]

        def h2_apply(scene, frame, wp, v0, intent_vec):
            rows = _h2_rows(scene)
            if rows is None or int(frame) not in rows[0]:
                h2_stat["miss"] += 1
                return wp                                  # no history -> gated to baseline
            i = rows[0][int(frame)]
            hb = {k: torch.from_numpy(rows[1][k][i][None]).cuda() for k in _HKEYS}
            it = torch.zeros(1, 3).cuda()
            if intent_vec is not None:
                it = intent_vec.view(1, 3).float().cuda()
            with torch.no_grad():
                new12, _ = h2_head(torch.from_numpy(wp.reshape(1, 12).astype(np.float32)).cuda(), it,
                                   torch.tensor([float(v0)]).cuda(), hb["track_hist"], hb["track_mask"],
                                   hb["ego_hist"], hb["ego_mask"], hb["history_enable"])
            h2_stat["hit"] += 1
            return new12.float().cpu().numpy().reshape(6, 2)

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
        kw = {}
        if a.intent:                         # route intent from the GT future (training definition, v43+)
            wpy = b[4][:12].view(6, 2)[2:, 1]
            k_int = 1 if float(wpy.max()) > 2.0 else (2 if float(wpy.min()) < -2.0 else 0)
            kw["intent"] = torch.nn.functional.one_hot(torch.tensor([k_int]), 3).float().cuda()
        infer.last_intent = kw.get("intent")
        with torch.no_grad(), torch.autocast("cuda", torch.float16):
            out = m(b[0][None].cuda(), b[1][None].cuda(), b[2][None].cuda(),
                    b[4][12][None].cuda(), **kw)      # v0 conditioning, as in eval
        # Cache outputs on CPU (2026-09-04). Kept on GPU, VRAM grows with the frame
        # count: 66 GB at 120 scenes and OOM at 240 (when the GUARD stage has 0 frames
        # nothing is popped and every frame stays resident).
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
                if h2_head is not None:
                    wp = h2_apply(s, ds.items[idxs[st + ci]][1]["frame"], wp,
                                  float(e[12]), infer.last_intent)
                pred_pos += wp_at(wp, STEP_S)          # ego-frame step
                gtwp = e[:12].view(6, 2).numpy()
                gt_pos += wp_at(gtwp, STEP_S)
                if ok:
                    d = float(np.linalg.norm(pred_pos - gt_pos))
                    drift[ci] += d
                    drift_n[ci] += 1
            if not ok:
                continue
        # Drop this scene's cached frames once its chains are done; caching every frame of every scene
        # exhausts host memory. The guard stage below simply re-infers its 400 frames.
        for i in idxs:
            cache.pop(i, None)

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
        if h2_head is not None:
            wp = h2_apply(ds.items[idx][0], ds.items[idx][1]["frame"], wp,
                          float(e[12]), infer.last_intent)
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

    if h2_head is not None:
        print(f"[h2] applied on {h2_stat['hit']} frames, missing history on {h2_stat['miss']}", flush=True)
    print(f"=== CHAIN: cumulative divergence of chained rollout (following own plan vs GT) ===")
    for ci in range(a.chain):
        if drift_n[ci]:
            t = STEP_S * (ci + 1)
            print(f"  +{t:.1f}s  {drift[ci] / drift_n[ci]:.3f} m "
                  f"(n={int(drift_n[ci])})")
    print(f"\n=== RECOV: lateral offset recovery ratio (1.0=full recovery) ===")
    nn = max(recov["n"], 1)
    for k in sorted(recov):
        if k != "n":
            print(f"  {k}: {recov[k] / nn:.2f}")
    print(f"\n=== GUARD: guardrail decisions ({n_guard} frames) ===")
    for k, v in guard.most_common():
        print(f"  {k}: {v} ({100 * v / max(n_guard, 1):.1f}%)")
    for k, v in reasons.most_common(5):
        print(f"    intervention reason: {k} x{v}")


if __name__ == "__main__":
    main()

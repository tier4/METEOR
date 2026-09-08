#!/usr/bin/env python3
"""GT-backed stationary/moving evaluation for TensorRT engines on Orin.

Unlike ``stat_probe.py`` this measures labels at GT vehicle centres, reports
the explicit stationary head, the independently supervised trajectory head,
and the deployed health-gated fusion.  The GT dead-band is identical to r46:
<=0.35 m at 3 s is stationary, >=0.8 m is moving, and creep in between is
excluded.
"""
import argparse
import gc
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deploy.runtime import (DET_RES, MeteorRT, stationary_at,  # noqa: E402
                            stationary_head_healthy)

CAMS = ["CAM_FRONT_WIDE", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
        "CAM_BACK_WIDE", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
        "CAM_FRONT_NARROW", "CAM_BACK_NARROW"]


def metrics(pred, truth):
    pred = np.asarray(pred, bool)
    truth = np.asarray(truth, bool)
    tp = int(np.sum(pred & truth)); fp = int(np.sum(pred & ~truth))
    tn = int(np.sum(~pred & ~truth)); fn = int(np.sum(~pred & truth))
    div = lambda a, b: float(a / b) if b else float("nan")
    recall = div(tp, tp + fn)
    specificity = div(tn, tn + fp)
    return {
        "n": int(len(truth)), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": div(tp, tp + fp), "recall": recall,
        "moving_accuracy": specificity,
        "balanced_accuracy": (recall + specificity) / 2,
        "accuracy": div(tp + tn, len(truth)),
    }


def threshold_sweep(logits, truth, min_moving_accuracy=0.95):
    """Calibrate in logit space without changing the TensorRT graph."""
    logits = np.asarray(logits, np.float64)
    truth = np.asarray(truth, bool)
    if not len(logits):
        return {"balanced": None, "safe": None}
    vals = np.unique(logits[np.isfinite(logits)])
    if len(vals) > 2000:
        vals = np.quantile(vals, np.linspace(0., 1., 2000))
    thresholds = np.r_[vals[0] - 1., (vals[:-1] + vals[1:]) / 2.,
                       vals[-1] + 1.]
    trials = [(float(t), metrics(logits > t, truth)) for t in thresholds]
    balanced = max(trials, key=lambda x: (x[1]["balanced_accuracy"],
                                           x[1]["moving_accuracy"],
                                           x[1]["recall"]))
    safe_trials = [x for x in trials
                   if x[1]["moving_accuracy"] >= min_moving_accuracy]
    safe = (max(safe_trials, key=lambda x: (x[1]["recall"],
                                             x[1]["balanced_accuracy"]))
            if safe_trials else None)
    pack = lambda x: ({"logit_threshold": x[0], **x[1]} if x else None)
    return {"balanced": pack(balanced), "safe": pack(safe),
            "safe_min_moving_accuracy": min_moving_accuracy}


def fmt(name, value):
    m = value
    return (f"{name:<11} P={m['precision']:.3f} R={m['recall']:.3f} "
            f"movAcc={m['moving_accuracy']:.3f} "
            f"balAcc={m['balanced_accuracy']:.3f} acc={m['accuracy']:.3f} "
            f"TP/FP/TN/FN={m['tp']}/{m['fp']}/{m['tn']}/{m['fn']}")


def load_inputs(scene_dir, man, frame):
    K = np.stack([np.asarray(man["cams"][c]["K"], np.float32)
                  for c in CAMS])[None]
    T = np.stack([np.linalg.inv(np.asarray(
        man["cams"][c]["T_ego_cam"], np.float32)) for c in CAMS])[None]
    images = []
    for c in CAMS:
        image = cv2.imread(os.path.join(scene_dir, frame["imgs"][c]))
        if image is None:
            raise IOError(f"missing image {scene_dir}/{frame['imgs'][c]}")
        images.append(image[:, :, ::-1].transpose(2, 0, 1))
    return np.stack(images)[None].astype(np.uint8), K, T


def evaluate(engine, root, args):
    rt = MeteorRT(engine, n_out_slots=1)
    truth = []
    head_pred = []
    traj_pred = []
    fused_pred = []
    records = {}
    latencies = []
    health = []
    map_std = []
    map_range = []
    scenes = [d for d in sorted(os.listdir(root))
              if os.path.isfile(os.path.join(root, d, "manifest.json"))]
    if args.limit_scenes:
        scenes = scenes[:args.limit_scenes]
    for scene in scenes:
        scene_dir = os.path.join(root, scene)
        man = json.load(open(os.path.join(scene_dir, "manifest.json")))
        frames = man["frames"][args.start::args.stride]
        if args.limit_frames:
            frames = frames[:args.limit_frames]
        rt.reset()
        for frame in frames:
            gt_path = os.path.join(scene_dir, frame.get("agent_traj", "_"))
            if not os.path.isfile(gt_path):
                continue
            images, K, T = load_inputs(scene_dir, man, frame)
            t0 = time.perf_counter()
            out = rt.infer(images, K, T, args.speed, pose=(0., 0., 0.))
            latencies.append((time.perf_counter() - t0) * 1000.0)
            stat = np.asarray(out["stationary"], np.float32)
            traj_out = np.asarray(out["traj"], np.float32)
            good = stationary_head_healthy(stat)
            health.append(good)
            finite = stat[np.isfinite(stat)]
            map_std.append(float(finite.std()) if finite.size else 0.0)
            map_range.append(float(finite.max() - finite.min())
                             if finite.size else 0.0)
            z = np.load(gt_path)
            boxes = z["boxes"]
            nbox = int(z["count"])
            gt_traj = z["traj"]
            valid = z["tvalid"]
            for k in range(nbox):
                if boxes[k, 3] <= 0 or boxes[k, 0] >= 1.5 \
                        or valid[k, 5] < 0.5:
                    continue
                d3 = float(np.linalg.norm(gt_traj[k, 5]))
                if 0.35 < d3 < 0.8:
                    continue
                ri = int((80.0 - float(boxes[k, 1])) / DET_RES)
                ci = int((50.0 - float(boxes[k, 2])) / DET_RES)
                if not (0 <= ri < stat.shape[-2] and
                        0 <= ci < stat.shape[-1]):
                    continue
                label = d3 <= 0.35
                logit = float(stat[0, 0, ri, ci])
                by_traj, _ = stationary_at(None, traj_out, ri, ci,
                                            stat_healthy=False)
                fused, source = stationary_at(stat, traj_out, ri, ci,
                                               stat_healthy=good)
                if by_traj is None or fused is None:
                    continue
                key = f"{scene}:{int(frame['frame'])}:{k}"
                truth.append(label)
                head_pred.append(logit > 0.0)
                traj_pred.append(bool(by_traj))
                fused_pred.append(bool(fused))
                records[key] = {"label": bool(label), "logit": logit,
                                "head": bool(logit > 0),
                                "trajectory": bool(by_traj),
                                "fused": bool(fused), "source": source}
    result = {
        "engine": engine,
        "head": metrics(head_pred, truth),
        "trajectory": metrics(traj_pred, truth),
        "fused": metrics(fused_pred, truth),
        "head_calibration": threshold_sweep(
            [v["logit"] for v in records.values()], truth),
        "health_rate": float(np.mean(health)) if health else 0.0,
        "map_std": float(np.mean(map_std)) if map_std else 0.0,
        "map_range": float(np.mean(map_range)) if map_range else 0.0,
        "latency_mean_ms": float(np.mean(latencies[1:])) if len(latencies) > 1
                           else float(np.mean(latencies)),
        "latency_p95_ms": float(np.percentile(latencies[1:], 95))
                          if len(latencies) > 1 else float("nan"),
        "records": records,
    }
    del rt
    gc.collect()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("engines", nargs="+")
    parser.add_argument("--root", default="calib")
    parser.add_argument("--start", type=int, default=5)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--limit-scenes", type=int, default=0)
    parser.add_argument("--limit-frames", type=int, default=0)
    parser.add_argument("--speed", type=float, default=8.0)
    parser.add_argument("--json", default="")
    args = parser.parse_args()
    results = []
    for engine in args.engines:
        result = evaluate(engine, args.root, args)
        results.append(result)
        print(f"\n[{os.path.basename(engine)}] GT n={result['head']['n']} "
              f"health={result['health_rate']:.1%} "
              f"map std/range={result['map_std']:.3f}/"
              f"{result['map_range']:.3f} latency="
              f"{result['latency_mean_ms']:.1f}ms "
              f"p95={result['latency_p95_ms']:.1f}ms")
        print(fmt("head", result["head"]))
        cal = result["head_calibration"]
        for name in ("balanced", "safe"):
            if cal[name] is not None:
                threshold = cal[name]["logit_threshold"]
                print(fmt(f"head@{name}", cal[name]) +
                      f" logit_threshold={threshold:.4f}")
        print(fmt("trajectory", result["trajectory"]))
        print(fmt("deployed", result["fused"]))
    if len(results) > 1:
        ref = results[0]["records"]
        for result in results[1:]:
            cur = result["records"]
            keys = sorted(set(ref) & set(cur))
            a = np.asarray([ref[k]["logit"] for k in keys])
            b = np.asarray([cur[k]["logit"] for k in keys])
            corr = (float(np.corrcoef(a, b)[0, 1])
                    if len(keys) > 1 and a.std() > 0 and b.std() > 0
                    else float("nan"))
            sign = float(np.mean((a > 0) == (b > 0))) if len(keys) else 0.0
            print(f"[vs {os.path.basename(results[0]['engine'])}] "
                  f"{os.path.basename(result['engine'])}: n={len(keys)} "
                  f"logit_corr={corr:.3f} sign_agree={sign:.3f}")
    if args.json:
        with open(args.json, "w") as fp:
            json.dump(results, fp, indent=2, allow_nan=True)


if __name__ == "__main__":
    main()

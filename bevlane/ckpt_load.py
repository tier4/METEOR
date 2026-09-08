#!/usr/bin/env python3
"""Shared loader that reads a checkpoint without silently dropping anything.

Background (found 2026-08-22): many probes did
    m = MODELS["v52"](...); m.load_state_dict(..., strict=False)
so weights of branches that **only exist after enable_* is called** (paint-seg /
paint-det: paint_proj.*, paint_det_proj.*) were silently discarded.
With strict=False there was no warning and no hook registered, so a model with
that branch was being measured as a different, branch-less model. A silent failure.

This loader calls the needed enable_* based on the ckpt keys and always reports
keys it could not load. Adding a new branch is one more line in _ENABLERS.
"""
import torch

# ckpt key prefix -> (enable method name, default class count)
# Default class counts must match the training launch.
_ENABLERS = (
    ("paint_proj.", "enable_paint_seg", [2, 3, 4, 5, 6, 7, 8, 13]),
    ("paint_det_proj.", "enable_paint_det", [1, 2, 3, 4, 5, 6]),
)

# stat_head2.proj.* is the bounded stat head (QuantRobustStatHead). It needs the
# whole net, not a method, so it is handled separately (inside load_net).


def strip(sd):
    sd = sd.get("model", sd)
    return {k.replace("module.", ""): v for k, v in sd.items()}


def load_net(net, ckpt, verbose=True, quiet_ok=()):
    """Load sd into net, growing the required branches first.

    Returns: (ckpt keys that could not be loaded, keys dropped for shape mismatch)
    """
    if isinstance(ckpt, str):
        try:
            raw = torch.load(ckpt, map_location="cpu", weights_only=False)
        except TypeError:  # PyTorch < 2.0
            raw = torch.load(ckpt, map_location="cpu")
    else:
        raw = ckpt
    saved_args = raw.get("args") or {} if isinstance(raw, dict) else {}
    if not isinstance(saved_args, dict):
        saved_args = vars(saved_args)
    sd = strip(raw)
    if "depth_head.0.0.weight" in sd and hasattr(net, "depth_head"):
        _wck = tuple(sd[f"depth_head.{i}.0.weight"].shape[0]
                     for i in range(4)
                     if f"depth_head.{i}.0.weight" in sd)
        _wcur = tuple(m[0].out_channels for m in net.depth_head[:-1])
        if len(_wck) == 4 and _wck != _wcur:
            from bevlane.model import enable_depth_slim
            enable_depth_slim(net, widths=_wck)
            print(f"[load] depth-slim width {_wck} detected", flush=True)
    if any(k.startswith("sem_ego.") for k in sd):
        from bevlane.model import enable_semantic_ego
        enable_semantic_ego(net)
        if verbose:
            print("[load] semantic-ego residual enabled", flush=True)
    if saved_args.get("depth_log_bins"):
        from bevlane.model import enable_depth_logbins
        enable_depth_logbins(net)
        if verbose:
            print("[load] log depth bins enabled (from saved args)", flush=True)
    if any(k.startswith("mode_scorer.") for k in sd):
        from bevlane.model import enable_mode_scorer
        enable_mode_scorer(net)
        if verbose:
            print("[load] mode-selection scorer enabled", flush=True)
    if any(k.startswith("det_tmp.") for k in sd):
        from bevlane.model import enable_det_temporal
        enable_det_temporal(net)
        if verbose:
            print("[load] det temporal-feature residual enabled", flush=True)
    if any(k.startswith("traj_vel.") for k in sd):
        from bevlane.model import enable_traj_cv
        enable_traj_cv(net)
        if verbose:
            print("[load] traj CV reparameterization enabled", flush=True)
    if any(k.startswith("traj_flow.") for k in sd):
        from bevlane.model import enable_traj_flow
        enable_traj_flow(net)
        if verbose:
            print("[load] flow->traj residual enabled", flush=True)
    if any(k.startswith("delta_stat.") for k in sd):
        from bevlane.model import enable_delta_stat
        enable_delta_stat(net)
        if verbose:
            print("[load] temporal-delta stat head enabled", flush=True)
    if any(k.startswith("lane_sdf.") for k in sd):
        from bevlane.model import enable_lane_sdf
        enable_lane_sdf(net)
        if verbose:
            print("[load] lane_sdf aux head enabled", flush=True)
    if any(k.startswith("stat_head2.proj.") for k in sd):
        from bevlane.model import enable_quant_stat_head
        enable_quant_stat_head(net, 8.0)
        if verbose:
            print("[load] bounded stat head (trained) enabled", flush=True)
    for pre, meth, dflt in _ENABLERS:
        if any(k.startswith(pre) for k in sd) and hasattr(net, meth):
            # recover the injected class count from the weight's input channels (not the default)
            w = sd.get(pre + "weight")
            n = int(w.shape[1]) if w is not None and w.dim() == 4 else len(dflt)
            arg_name = "paint_seg" if meth == "enable_paint_seg" else "paint_det"
            recorded = saved_args.get(arg_name)
            if recorded:
                cls = ([int(x) for x in recorded.split(",")]
                       if isinstance(recorded, str) else list(recorded))
            else:
                cls = dflt if n == len(dflt) else list(range(n))
            if len(cls) != n:
                raise ValueError(f"{arg_name} metadata {cls} does not match "
                                 f"checkpoint input channels {n}")
            getattr(net, meth)(cls)
            if verbose:
                print(f"[load] {meth}({cls}) enabled", flush=True)
    if any(k.startswith("lane_branch.") for k in sd) \
            and getattr(net, "lane_branch", None) is None:
        from bevlane.model import enable_lane_branch
        enable_lane_branch(net)
        if verbose:
            print("[load] lane_branch enabled with checkpoint shape", flush=True)
    if any(k.startswith("stat_head2.proj.") for k in sd):
        from bevlane.model import enable_quant_stat_head
        enable_quant_stat_head(net, 8.0)
        if verbose:
            print("[load] quant-robust stationary head enabled", flush=True)
    cur = net.state_dict()
    ok = {k: v for k, v in sd.items() if k in cur and cur[k].shape == v.shape}
    bad_shape = [k for k, v in sd.items()
                 if k in cur and cur[k].shape != v.shape]
    unused = [k for k in sd if k not in cur and not k.startswith(quiet_ok)]
    net.load_state_dict(ok, strict=False)
    if verbose and (unused or bad_shape):
        print(f"[load] warning: unused {len(unused)} / shape mismatch {len(bad_shape)}"
              f"  e.g. {(unused + bad_shape)[:4]}", flush=True)
    elif verbose:
        print(f"[load] all {len(ok)} tensors loaded (nothing dropped)",
              flush=True)
    return unused, bad_shape

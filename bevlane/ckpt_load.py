#!/usr/bin/env python3
"""チェックポイントを「取りこぼしなく」読むための共通ローダ。

背景 (2026-08-22 発覚): probe 群の多くが
    m = MODELS["v52"](...); m.load_state_dict(..., strict=False)
と書いており、paint-seg / paint-det のように **enable_* を呼ばないと存在しない**
枝の重み (paint_proj.*, paint_det_proj.*) が黙って捨てられていた。
strict=False なので警告も出ず、フックも登録されないので、その枝を持つモデルを
「枝なしの別モデル」として測っていたことになる。静かな故障そのもの。

このローダは ckpt のキーから必要な enable_* を呼び、読めなかったキーを必ず
表示する。新しい枝を足したときは _ENABLERS に 1 行足すだけでよい。
"""
import torch

# ckpt キーの接頭辞 -> (有効化メソッド名, 既定クラス)
# 既定クラスは学習側の launch と揃えること。
_ENABLERS = (
    ("paint_proj.", "enable_paint_seg", [2, 3, 4, 5, 6, 7, 8, 13]),
    ("paint_det_proj.", "enable_paint_det", [1, 2, 3, 4, 5, 6]),
)

# stat_head2.proj.* は「有界 stat ヘッド」(QuantRobustStatHead)。関数ではなく
# net 全体を要するので _ENABLERS とは別扱い (load_net 内で処理)。


def strip(sd):
    sd = sd.get("model", sd)
    return {k.replace("module.", ""): v for k, v in sd.items()}


def load_net(net, ckpt, verbose=True, quiet_ok=()):
    """sd を net に読み込む。必要な枝を先に生やしてから読む。

    返り値: (読み込めなかった ckpt キー, 形が合わず捨てたキー)
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
            print(f"[load] depth-slim 幅 {_wck} を検出", flush=True)
    if any(k.startswith("sem_ego.") for k in sd):
        from bevlane.model import enable_semantic_ego
        enable_semantic_ego(net)
        if verbose:
            print("[load] semantic-ego 残差を有効化", flush=True)
    if saved_args.get("depth_log_bins"):
        from bevlane.model import enable_depth_logbins
        enable_depth_logbins(net)
        if verbose:
            print("[load] 深度対数ビンを有効化 (保存 args より)", flush=True)
    if any(k.startswith("mode_scorer.") for k in sd):
        from bevlane.model import enable_mode_scorer
        enable_mode_scorer(net)
        if verbose:
            print("[load] モード選択スコアラを有効化", flush=True)
    if any(k.startswith("det_tmp.") for k in sd):
        from bevlane.model import enable_det_temporal
        enable_det_temporal(net)
        if verbose:
            print("[load] det 時間特徴残差を有効化", flush=True)
    if any(k.startswith("traj_vel.") for k in sd):
        from bevlane.model import enable_traj_cv
        enable_traj_cv(net)
        if verbose:
            print("[load] traj CV 再パラメータ化を有効化", flush=True)
    if any(k.startswith("traj_flow.") for k in sd):
        from bevlane.model import enable_traj_flow
        enable_traj_flow(net)
        if verbose:
            print("[load] flow→traj 残差を有効化", flush=True)
    if any(k.startswith("delta_stat.") for k in sd):
        from bevlane.model import enable_delta_stat
        enable_delta_stat(net)
        if verbose:
            print("[load] 時間差分 stat ヘッドを有効化", flush=True)
    if any(k.startswith("lane_sdf.") for k in sd):
        from bevlane.model import enable_lane_sdf
        enable_lane_sdf(net)
        if verbose:
            print("[load] lane_sdf 補助ヘッドを有効化", flush=True)
    if any(k.startswith("stat_head2.proj.") for k in sd):
        from bevlane.model import enable_quant_stat_head
        enable_quant_stat_head(net, 8.0)
        if verbose:
            print("[load] 有界 stat ヘッド (学習済み) を有効化", flush=True)
    for pre, meth, dflt in _ENABLERS:
        if any(k.startswith(pre) for k in sd) and hasattr(net, meth):
            # 注入クラス数は重みの入力チャネル数から復元する (既定に頼らない)
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
                print(f"[load] {meth}({cls}) を有効化", flush=True)
    if any(k.startswith("lane_branch.") for k in sd) \
            and getattr(net, "lane_branch", None) is None:
        from bevlane.model import enable_lane_branch
        enable_lane_branch(net)
        if verbose:
            print("[load] lane_branch を checkpoint 形状で有効化", flush=True)
    if any(k.startswith("stat_head2.proj.") for k in sd):
        from bevlane.model import enable_quant_stat_head
        enable_quant_stat_head(net, 8.0)
        if verbose:
            print("[load] quant-robust stationary head を有効化", flush=True)
    cur = net.state_dict()
    ok = {k: v for k, v in sd.items() if k in cur and cur[k].shape == v.shape}
    bad_shape = [k for k, v in sd.items()
                 if k in cur and cur[k].shape != v.shape]
    unused = [k for k in sd if k not in cur and not k.startswith(quiet_ok)]
    net.load_state_dict(ok, strict=False)
    if verbose and (unused or bad_shape):
        print(f"[load] 警告: 未使用 {len(unused)} / 形不一致 {len(bad_shape)}"
              f"  例 {(unused + bad_shape)[:4]}", flush=True)
    elif verbose:
        print(f"[load] 全 {len(ok)} テンソルを読み込み (取りこぼしなし)",
              flush=True)
    return unused, bad_shape

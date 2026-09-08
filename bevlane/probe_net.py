#!/usr/bin/env python3
"""probe / 評価スクリプト共通の「取りこぼしなし」モデル構築 (2026-09-05, 静かな故障 #10)。

chain_decomp / closed_loop_eval / dump_* は素の MODELS["v52"] に strict=False で
重みを流していたため、sem_ego (E2E 入力路) 22 / lane_branch 14 / delta_stat 14 /
paint_proj 2 が捨てられ、depth_head 41 は幅不一致で乱数のままだった。
ここでは deploy/export_onnx.build (ckpt のキーから enable_* を呼ぶ) を使い、
読めなかったキー・形不一致が 0 であることを検証してから返す。"""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_full(ckpt, device="cuda", mv="v52", verbose=True):
    from deploy.export_onnx import build
    from bevlane.model import enable_kinematic_anchor
    net = build(ckpt, mv=mv)
    raw = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = {k.replace("module.", ""): v for k, v in raw["model"].items()}
    if "kin_gate" in sd and not hasattr(net, "kin_gate"):
        enable_kinematic_anchor(net)
    cur = net.state_dict()
    unexpected = [k for k in sd if k not in cur]
    mism = [k for k in sd if k in cur and tuple(cur[k].shape) != tuple(sd[k].shape)]
    # build() は読み込み済みだが、kin_gate 等を後付けした場合に備えて再ロード
    net.load_state_dict({k: v for k, v in sd.items() if k in cur and tuple(cur[k].shape) == tuple(sd[k].shape)}, strict=False)
    if verbose:
        print(f"[probe_net] {os.path.basename(os.path.dirname(ckpt))}/{os.path.basename(ckpt)}: "
              f"ckpt {len(sd)} keys, unexpected {len(unexpected)}, mismatch {len(mism)}"
              + (f"  !! {unexpected[:3]} {mism[:3]}" if unexpected or mism else "  (OK)"), flush=True)
    if unexpected or mism:
        raise RuntimeError(f"probe_net: 取りこぼし unexpected={len(unexpected)} mismatch={len(mism)} — enable_* が足りない")
    return net.to(device).eval()

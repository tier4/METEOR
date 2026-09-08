#!/usr/bin/env python3
"""Lossless model construction shared by probe / eval scripts (2026-09-05, silent failure #10).

chain_decomp / closed_loop_eval / dump_* poured weights into a bare MODELS["v52"] with
strict=False, dropping sem_ego (E2E input path) 22 / lane_branch 14 / delta_stat 14 /
paint_proj 2 tensors, and depth_head 41 stayed random due to a width mismatch.
Here we use deploy/export_onnx.build (calls enable_* from the ckpt keys) and verify
zero unloaded keys and zero shape mismatches before returning."""
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
    # build() has already loaded, but reload in case kin_gate etc. were attached afterwards
    net.load_state_dict({k: v for k, v in sd.items() if k in cur and tuple(cur[k].shape) == tuple(sd[k].shape)}, strict=False)
    if verbose:
        print(f"[probe_net] {os.path.basename(os.path.dirname(ckpt))}/{os.path.basename(ckpt)}: "
              f"ckpt {len(sd)} keys, unexpected {len(unexpected)}, mismatch {len(mism)}"
              + (f"  !! {unexpected[:3]} {mism[:3]}" if unexpected or mism else "  (OK)"), flush=True)
    if unexpected or mism:
        raise RuntimeError(f"probe_net: dropped weights unexpected={len(unexpected)} mismatch={len(mism)} — missing enable_*")
    return net.to(device).eval()

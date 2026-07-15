#!/usr/bin/env python3
"""Expand a v27/v28 checkpoint's single-mode heads into v29 K=3 heads.

Each hypothesis starts as a copy of the trained single trajectory (+ tiny
noise to break symmetry); mode logits start at zero. Everything else passes
through unchanged."""
import sys
import torch

src, dst = sys.argv[1], sys.argv[2]
ck = torch.load(src, map_location="cpu")
sd = ck["model"]
K = 3
g = torch.Generator().manual_seed(0)


def tile(w, reps, noise):
    out = torch.cat([w + noise * torch.randn(w.shape, generator=g)
                     for _ in range(reps)], 0)
    return out


if sd["ego_mlp.6.weight"].shape[0] == 15:
    w, b = sd["ego_mlp.6.weight"], sd["ego_mlp.6.bias"]
    sd["ego_mlp.6.weight"] = torch.cat(
        [tile(w[:12], K, 0.01), torch.zeros(K, w.shape[1]), w[12:]], 0)
    sd["ego_mlp.6.bias"] = torch.cat(
        [b[:12].repeat(K), torch.zeros(K), b[12:]], 0)
if sd["traj_head.weight"].shape[0] == 12:
    w, b = sd["traj_head.weight"], sd["traj_head.bias"]
    sd["traj_head.weight"] = torch.cat(
        [tile(w, K, 0.01), torch.zeros(K, *w.shape[1:])], 0)
    sd["traj_head.bias"] = torch.cat([b.repeat(K), torch.zeros(K)], 0)
torch.save(ck, dst)
print("expanded ->", dst)

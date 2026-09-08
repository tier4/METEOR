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
    # DIVERSE bias per mode: straight / left / right. Identical copies make
    # winner-takes-all collapse (one mode wins every sample and takes every
    # gradient) -- give each hypothesis a lateral prior to specialise from.
    # Lateral offset grows along the horizon: +-0, 0.35, 0.7 ... m at 3 s.
    lat = torch.zeros(K, 12)
    for k, sgn in enumerate((0.0, 1.0, -1.0)):
        for h in range(6):
            lat[k, 2 * h + 1] = sgn * 0.14 * (h + 1)      # y offsets
    sd["ego_mlp.6.bias"] = torch.cat(
        [(b[:12].repeat(K, 1) + lat).reshape(-1), torch.zeros(K), b[12:]], 0)
if sd["traj_head.weight"].shape[0] == 12:
    w, b = sd["traj_head.weight"], sd["traj_head.bias"]
    sd["traj_head.weight"] = torch.cat(
        [tile(w, K, 0.01), torch.zeros(K, *w.shape[1:])], 0)
    lat_t = torch.zeros(K, 12)
    for k, sgn in enumerate((0.0, 1.0, -1.0)):
        for h in range(6):
            lat_t[k, 2 * h + 1] = sgn * 0.14 * (h + 1)
    sd["traj_head.bias"] = torch.cat(
        [(b.repeat(K, 1) + lat_t).reshape(-1), torch.zeros(K)], 0)
# temporal fuse: graft the TRAINED v28 tfuse into the 3-slot tfuse3.
# Slot 0 keeps the same 0.4 s offset, so copying the [bev, warped] input
# channels and zeroing the two new slots reproduces the v28 fusion exactly
# at init -- without this, ego/traj heads see unfamiliar fused features and
# their (heavily weighted) losses wreck the shared backbone (r20 collapse).
if "tfuse.0.weight" in sd and "tfuse3.0.weight" not in sd:
    w = sd["tfuse.0.weight"]                     # [96, 192, 1, 1]
    w3 = torch.zeros(w.shape[0], 2 * w.shape[1], *w.shape[2:])
    w3[:, :w.shape[1]] = w
    sd["tfuse3.0.weight"] = w3
    for k in list(sd.keys()):
        if k.startswith("tfuse.") and not k.startswith("tfuse.0."):
            sd["tfuse3." + k[len("tfuse."):]] = sd[k]
    for k in [k for k in sd if k.startswith("tfuse.")]:
        del sd[k]
# Re-diversify an ALREADY-expanded checkpoint (42-dim ego head). r20's modes
# collapsed onto one hypothesis (mode 1 won 80/80 val samples, spread 0.74 m).
# Keep its trained BEV/detection weights, but re-spread the mode biases so
# eps-WTA has something to specialise from.
elif sd["ego_mlp.6.weight"].shape[0] == 12 * K + K + 3:
    b = sd["ego_mlp.6.bias"]
    lat = torch.zeros(K, 12)
    for k, sgn in enumerate((0.0, 1.0, -1.0)):
        for h in range(6):
            lat[k, 2 * h + 1] = sgn * 0.14 * (h + 1)
    wp = b[:12 * K].view(K, 12)
    base = wp.mean(0, keepdim=True)               # collapsed -> one prior
    sd["ego_mlp.6.bias"] = torch.cat(
        [(base + lat).reshape(-1), torch.zeros(K), b[12 * K + K:]], 0)
    tb = sd["traj_head.bias"]
    twp = tb[:12 * K].view(K, 12)
    tbase = twp.mean(0, keepdim=True)
    sd["traj_head.bias"] = torch.cat(
        [(tbase + lat).reshape(-1), torch.zeros(K)], 0)
    print("re-diversified collapsed modes")

torch.save(ck, dst)
print("expanded ->", dst)

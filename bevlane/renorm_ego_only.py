"""Rescale only the conv->BN pairs in ego_stem (2026-08-20).

v98 freezes its ego output in INT8 (activation std inflates 4x vs fp16 and input
sensitivity is lost). Cause: the BN running_var in ego_stem drifted to 1.6x that of
v95 (134->228). The conv->BN scale is a free degree of freedom, so normalize it
without changing the function and let INT8 use its quantization steps well.
renorm_convbn.py, which touches every layer, squashed lane_branch with s=21 and
amplified fp16 error, hence the restriction to ego_stem.
"""
import argparse
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--prefix", default="ego_stem")
ap.add_argument("--target-std", type=float, default=2.0)
ap.add_argument("--min-var", type=float, default=25.0)
a = ap.parse_args()

ck = torch.load(a.ckpt, map_location="cpu")
sd = ck.get("model", ck)
keys = list(sd)
n = 0
for k in keys:
    if not (k.endswith("running_var") and a.prefix in k):
        continue
    var = sd[k]
    if float(var.max()) < a.min_var:
        continue
    base = k[: -len("running_var")]
    # find the preceding conv (search backwards for a conv whose weight/bias shape
    # matches, rather than relying on naming like .{idx-1} in the same parent)
    ch = var.numel()
    cand = None
    for j in range(keys.index(k) - 1, -1, -1):
        kk = keys[j]
        if kk.endswith("weight") and sd[kk].dim() == 4 and sd[kk].shape[0] == ch:
            cand = kk
            break
    if cand is None:
        continue
    s = float((var.mean().sqrt() / a.target_std))
    if not (0.2 < s < 50):
        continue
    sd[cand] = sd[cand] / s
    cb = cand[:-6] + "bias"
    if cb in sd and sd[cb].shape == var.shape:
        sd[cb] = sd[cb] / s
    sd[base + "running_mean"] = sd[base + "running_mean"] / s
    sd[k] = var / (s * s)
    print(f"  {base[:-1]:38s} var {float(var.max()):8.1f} -> "
          f"{float(sd[k].max()):6.2f}  (s={s:.3f}, conv={cand})")
    n += 1
if "model" in ck:
    ck["model"] = sd
else:
    ck = sd
torch.save(ck, a.out)
print(f"{n} pairs rescaled -> {a.out}")

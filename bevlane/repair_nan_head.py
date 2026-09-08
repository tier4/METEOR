"""Repair a NaN-contaminated head by transplanting it from a healthy checkpoint.

2026-08-21: since local r74, seg_head.out.0.weight (final 2D seg conv, all
82944 elements) and the BN stats of seg_head.out.1 were NaN.
nan_to_num in model.py clamps the output to 0, so training completes, but
2D seg is always zero = PointPainting is meaningless (why r77 paint had zero effect).
Root cause of the score being stuck at 0.283 for 7 rounds r74..r80.
"""
import argparse
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True, help="checkpoint to repair (contains NaN)")
ap.add_argument("--donor", required=True, help="healthy donor checkpoint")
ap.add_argument("--out", required=True)
a = ap.parse_args()

ck = torch.load(a.ckpt, map_location="cpu")
sd = ck.get("model", ck)
dn = torch.load(a.donor, map_location="cpu")
dn = dn.get("model", dn)
sd = {k.replace("module.", ""): v for k, v in sd.items()}
dn = {k.replace("module.", ""): v for k, v in dn.items()}

bad = [k for k, v in sd.items()
       if torch.is_floating_point(v) and torch.isnan(v).any()]
print(f"repairing {len(bad)} NaN tensors:")
fixed = 0
for k in bad:
    if k in dn and dn[k].shape == sd[k].shape and not torch.isnan(dn[k]).any():
        sd[k] = dn[k].clone()
        print(f"  {k}: transplanted from donor")
        fixed += 1
    elif k.endswith("running_var"):
        sd[k] = torch.ones_like(sd[k])
        print(f"  {k}: reset to 1.0")
        fixed += 1
    elif k.endswith("running_mean"):
        sd[k] = torch.zeros_like(sd[k])
        print(f"  {k}: reset to 0.0")
        fixed += 1
    else:
        print(f"  {k}: not repaired (missing in donor)")
rest = sum(1 for k, v in sd.items()
           if torch.is_floating_point(v) and torch.isnan(v).any())
print(f"repaired {fixed}/{len(bad)}, remaining NaN {rest}")
if "model" in ck:
    ck["model"] = sd
else:
    ck = sd
torch.save(ck, a.out)
print(f"saved {a.out}")

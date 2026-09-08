"""NaN に汚染されたヘッドを健全なチェックポイントから移植して修復する。

2026-08-21: ローカル r74 以降、seg_head.out.0.weight (2D セグ最終 conv,
82944 要素すべて) と seg_head.out.1 の BN 統計が NaN になっていた。
model.py の nan_to_num が出力を 0 にクランプするため学習は完走するが、
2D セグは常にゼロ = PointPainting も無意味 (r77 の paint 効果ゼロの正体)。
score が r74..r80 の 7 ラウンド 0.283 に張り付いた根本原因。
"""
import argparse
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True, help="修復対象 (NaN あり)")
ap.add_argument("--donor", required=True, help="健全なチェックポイント")
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
print(f"NaN テンソル {len(bad)} 個を修復:")
fixed = 0
for k in bad:
    if k in dn and dn[k].shape == sd[k].shape and not torch.isnan(dn[k]).any():
        sd[k] = dn[k].clone()
        print(f"  {k}: donor から移植")
        fixed += 1
    elif k.endswith("running_var"):
        sd[k] = torch.ones_like(sd[k])
        print(f"  {k}: 1.0 でリセット")
        fixed += 1
    elif k.endswith("running_mean"):
        sd[k] = torch.zeros_like(sd[k])
        print(f"  {k}: 0.0 でリセット")
        fixed += 1
    else:
        print(f"  {k}: 修復できず (donor になし)")
rest = sum(1 for k, v in sd.items()
           if torch.is_floating_point(v) and torch.isnan(v).any())
print(f"修復 {fixed}/{len(bad)}、残存 NaN {rest}")
if "model" in ck:
    ck["model"] = sd
else:
    ck = sd
torch.save(ck, a.out)
print(f"saved {a.out}")

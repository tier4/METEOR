"""ego_stem の conv->BN 対だけを再スケールする (2026-08-20)。

v98 は INT8 で ego 出力が凍結する (活性 std が fp16 比 4 倍に膨張し
入力感度を喪失)。原因は ego_stem の BN running_var が v95 の 1.6 倍
(134->228) にドリフトしたこと。conv->BN のスケールは自由度なので、
関数を変えずに正規化して INT8 の量子化ステップを有効活用させる。
全層を触る renorm_convbn.py は lane_branch を s=21 で潰して fp16 誤差を
増幅したため、ego_stem に限定する。
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
    # 直前の conv を探す (同じ親モジュールの .{idx-1} など命名規則に依存
    # しないよう、weight/bias の形が合う conv を後方から探索)
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
print(f"{n} 対を再スケール -> {a.out}")

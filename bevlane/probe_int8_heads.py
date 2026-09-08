"""PyTorch INT8 シミュレーション時に、どのヘッドがどれだけ劣化するかを
GT 基準で測る (2026-08-22)。動画で risk map のノイズ増が見えたため、
ヘッド別に fp32 との差と GT 一致度を定量化する。
"""
import argparse, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bevlane.dataset import BevLaneDataset
from bevlane.model import MODELS

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--list", default="val.lst")
ap.add_argument("--root", default="out/bevlane")
ap.add_argument("--frames", type=int, default=24)
ap.add_argument("--pct", type=float, default=99.9)
ap.add_argument("--keep-fp", default="",
                help="量子化から除外する層名の部分文字列 (カンマ区切り)。"
                     "例 risk_head : risk map だけ fp32 のまま計算する")
a = ap.parse_args()
scenes = [l.strip() for l in open(a.list) if l.strip()][:6]
ds = BevLaneDataset(a.root, scenes, gt_key="gt_cons", with_ego=True,
                    with_risk=True, with_occ=True, max_per_scene=6,
                    n_cams=8, trim_start=3, trim_end=10)

def build(quant):
    m = MODELS["v52"](n_seg=21).cuda().eval()
    sd = torch.load(a.ckpt, map_location="cpu"); sd = sd.get("model", sd)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    cur = m.state_dict()
    m.load_state_dict({k: v for k, v in sd.items()
                       if k in cur and cur[k].shape == v.shape}, strict=False)
    if not quant:
        return m, None
    Q = 127
    SC, COL = {}, {"on": True}
    KEEP = [k for k in a.keep_fp.split(",") if k]
    def _skip(name):
        return any(k in name for k in KEEP)
    with torch.no_grad():
        for _n2, mod in m.named_modules():
            if isinstance(mod, (torch.nn.Conv2d, torch.nn.Linear)) \
                    and not _skip(_n2):
                w = mod.weight.data
                d = tuple(range(1, w.dim()))
                s = (w.abs().amax(dim=d, keepdim=True) / Q).clamp_min(1e-12)
                mod.weight.data = torch.round(w / s).clamp(-Q, Q) * s
    def fq(mod, inp, out):
        if not torch.is_tensor(out) or not out.is_floating_point():
            return out
        k = id(mod)
        if COL["on"]:
            v = out.detach().abs().flatten().float()
            kk = max(1, int(v.numel() * a.pct / 100.0))
            SC[k] = max(SC.get(k, 0.0), float(v.kthvalue(kk).values))
            return out
        mx = SC.get(k, 0.0)
        if mx <= 0: return out
        s = mx / Q
        return torch.round(out / s).clamp(-Q, Q) * s
    n_q = 0
    for _n3, mod in m.named_modules():
        if isinstance(mod, (torch.nn.Conv2d, torch.nn.Linear)) \
                and not _skip(_n3):
            mod.register_forward_hook(fq)
            n_q += 1
    if KEEP:
        print(f"[keep-fp] {KEEP} を量子化から除外 ({n_q} 層を量子化)")
    return m, COL

m32, _ = build(False)
m8, COL = build(True)
# 較正
for i in range(8):
    b = ds[i]
    if b is None: continue
    with torch.no_grad(), torch.autocast("cuda", torch.float16):
        m8(b[0][None].cuda(), b[1][None].cuda(), b[2][None].cuda())
COL["on"] = False

agg = {}
for i in range(a.frames):
    b = ds[i]
    if b is None: continue
    args_ = (b[0][None].cuda(), b[1][None].cuda(), b[2][None].cuda())
    with torch.no_grad(), torch.autocast("cuda", torch.float16):
        o32 = m32(*args_); o8 = m8(*args_)
    # risk (out[12]) は sigmoid 確率
    r32 = o32[12].float().sigmoid()
    r8 = o8[12].float().sigmoid()
    agg.setdefault("risk_mae", []).append(float((r32 - r8).abs().mean()))
    agg.setdefault("risk_mean32", []).append(float(r32.mean()))
    agg.setdefault("risk_mean8", []).append(float(r8.mean()))
    # GT risk と比較 (b の risk は [400,250] 0-1、-1 は未生成)
    for j, x in enumerate(b):
        if torch.is_tensor(x) and x.dim() == 2 and x.shape == (400, 250):
            g = x.cuda()
            if float(g.max()) >= 0:
                import torch.nn.functional as F
                for tag, r in (("32", r32), ("8", r8)):
                    rr = F.interpolate(r, size=(400, 250), mode="bilinear",
                                       align_corners=False)[0, 0]
                    agg.setdefault(f"risk_gt_mae{tag}", []).append(
                        float((rr - g).abs().mean()))
            break
    # 他ヘッドの相対差
    for idx, nm in ((0, "lane"), (3, "det_hm"), (8, "occ"), (7, "ego")):
        if idx < len(o32) and torch.is_tensor(o32[idx]):
            d = (o32[idx].float() - o8[idx].float()).abs().mean()
            sc = o32[idx].float().abs().mean().clamp_min(1e-6)
            agg.setdefault(f"rel_{nm}", []).append(float(d / sc))
print(f"\n=== PyTorch INT8 のヘッド別劣化 ({a.frames} 枚, pct={a.pct}) ===")
print(f"risk 確率の平均: fp32 {np.mean(agg['risk_mean32']):.4f} -> "
      f"INT8 {np.mean(agg['risk_mean8']):.4f}  (MAE {np.mean(agg['risk_mae']):.4f})")
if "risk_gt_mae32" in agg:
    print(f"risk の GT 誤差: fp32 {np.mean(agg['risk_gt_mae32']):.4f} -> "
          f"INT8 {np.mean(agg['risk_gt_mae8']):.4f}")
print("他ヘッドの相対差 (|fp32-INT8| / |fp32|):")
for nm in ("lane", "det_hm", "occ", "ego"):
    k = f"rel_{nm}"
    if k in agg:
        print(f"  {nm:7s} {np.mean(agg[k]):.4f}")
print("PROBE_INT8_HEADS_DONE")

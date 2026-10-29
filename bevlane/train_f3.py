#!/usr/bin/env python3
"""F3 probe: temporal BEV fusion vs single-frame, cache-trained (2026-10-08).

Loads the dump_f3_cache.py scene files and trains a small bounded-residual head
on the baseline-selected candidate (H2 contract), in one of three arms:
  --arm single : conv head on the current pooled BEV only (control; the arm used at inference, bevlane/f3_refine.py)
  --arm fuse   : conv head on [current, ego-motion-warped previous] BEV
  --arm stack  : conv head on [current, previous] BEV without ego-motion alignment (perception-only temporal input;
                 --gap N takes the frame N frames earlier)
Gate: fuse improves path ADE >= 10 % vs the baseline selection
AND p0 (0.5 s point) unchanged; `fuse` must also beat `single` (the temporal
part itself, not just BEV conditioning).

  train_f3.py --cache <f3_cache_dir> --arm fuse --out out/f3_fuse
"""
import argparse, glob, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bevlane.model import make_warp_theta            # noqa: E402

EGO_K = 3


def load(cache, val_every=10, gap=1):
    tr = {k: [] for k in ("bev", "prev", "rel", "cand", "gt", "v0", "acc", "brk")}
    va = {k: [] for k in tr}
    files = sorted(glob.glob(os.path.join(cache, "*.npz")))
    for i, f in enumerate(files):
        z = np.load(f)
        bev = z["bev"]                               # [T,96,64,40] fp16
        T = len(bev)
        if T < gap + 1:
            continue
        # --gap: prev = the frame gap frames earlier (only pairs with consecutive frame numbers; gaps are dropped). gap=1 = previous behaviour
        ok = (z["frame"][gap:] - z["frame"][:-gap]) == gap
        dst = va if i % val_every == 0 else tr
        dst["bev"].append(bev[gap:][ok])
        dst["prev"].append(bev[:-gap][ok])
        dst["rel"].append(z["rel"][gap:][ok])        # relative pose of adjacent frames (warp for fuse; only meaningful with gap=1)
        dst["cand"].append(z["cand"][gap:][ok])
        dst["gt"].append(z["gt"][gap:][ok])
        dst["v0"].append(z["v0"][gap:][ok])
        for k_ in ("acc", "brk"):                 # older caches lack these keys: fill with zeros
            dst[k_].append(z[k_][gap:][ok] if k_ in z.files else np.zeros(int(ok.sum()), np.float32))
    out = []
    for d in (tr, va):
        cat = {k: np.concatenate(v) for k, v in d.items()}
        for k in cat:                     # npz scalars arrive float64; the heads are fp32
            if k != "bev":
                cat[k] = cat[k].astype(np.float32)
        out.append({k: torch.from_numpy(v) for k, v in cat.items()})
    return out[0], out[1], len(files)


ACC_SEL = 0.0                                   # --acc-sel: re-select the candidate in training with the same rule as inference (METEOR_F3_ACC_SEL)


def selected(cand, acc=None):
    N = cand.shape[0]
    wp = cand[:, :12 * EGO_K].view(N, EGO_K, 12)
    k = cand[:, 12 * EGO_K:12 * EGO_K + EGO_K].argmax(1)
    if ACC_SEL > 0 and acc is not None:
        x3 = wp[:, :, 10]
        k = torch.where(acc < -ACC_SEL, x3.argmin(1), torch.where(acc > ACC_SEL, x3.argmax(1), k))
    return wp[torch.arange(N), k], wp


def ade(a, b):
    return (a.view(-1, 6, 2) - b.view(-1, 6, 2)).norm(dim=-1).mean(-1)


class F3Head(nn.Module):
    def __init__(self, cin, bound=2.0, extra=0):
        super().__init__()
        self.extra = extra                           # number of extra scalar inputs (acc / brk)
        self.conv = nn.Sequential(
            nn.Conv2d(cin, 128, 3, stride=2, padding=1), nn.GroupNorm(8, 128), nn.ReLU(True),
            nn.Conv2d(128, 192, 3, stride=2, padding=1), nn.GroupNorm(8, 192), nn.ReLU(True),
            nn.Conv2d(192, 256, 3, stride=2, padding=1), nn.GroupNorm(8, 256), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1))
        self.mlp = nn.Sequential(
            nn.Linear(256 + 12 + 1 + extra, 256), nn.ReLU(True),
            nn.Linear(256, 256), nn.ReLU(True),
            nn.Linear(256, 12))
        nn.init.zeros_(self.mlp[-1].weight); nn.init.zeros_(self.mlp[-1].bias)
        self.bound = bound

    def forward(self, feat, sel, v0, ext=None):
        g = self.conv(feat).flatten(1)
        parts = [g, sel, v0[:, None] / 10.0]
        if self.extra:
            parts.append(ext[:, :self.extra].float())
        r = self.mlp(torch.cat(parts, 1))
        res = self.bound * torch.tanh(r)
        res = res.clone()
        res[:, 0:2] = 0.0                            # p0 (0.5 s point) untouched by contract
        return sel + res


class F3HeadNPU(nn.Module):
    """NPU-friendly F3 head using standard ONNX operators only: GroupNorm -> BatchNorm (foldable
    into Conv), AdaptiveAvgPool -> GlobalAveragePool, Linear (Gemm) -> Conv1x1, and the p0 freeze is a
    Mul by a constant mask. The input feature is an integer AveragePool of the fused BEV
    (dump_f3_cache F3_POOLK=5: [96, 80, 50]). forward() arguments and outputs match F3Head."""

    def __init__(self, cin, bound=2.0, extra=0):
        super().__init__()
        self.extra = extra; self.bound = bound; self.npu = True
        self.conv = nn.Sequential(
            nn.Conv2d(cin, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 192, 3, stride=2, padding=1), nn.BatchNorm2d(192), nn.ReLU(True),
            nn.Conv2d(192, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(True))
        self.mlp = nn.Sequential(
            nn.Conv2d(256 + 12 + 1 + extra, 256, 1), nn.ReLU(True),
            nn.Conv2d(256, 256, 1), nn.ReLU(True),
            nn.Conv2d(256, 12, 1))
        nn.init.zeros_(self.mlp[-1].weight); nn.init.zeros_(self.mlp[-1].bias)
        mask = torch.ones(1, 12, 1, 1); mask[:, 0:2] = 0.0            # p0 (0.5 s point) unchanged
        self.register_buffer("mask", mask)

    def trunk(self, feat):
        return F.adaptive_avg_pool2d(self.conv(feat), 1)               # [B, 256, 1, 1] (GlobalAveragePool)

    def mlp_res(self, g, sel4, v4, ext4=None):
        """g [B,256,1,1], sel4 [B,12,1,1], v4 [B,1,1,1] (v0/10), ext4 [B,extra,1,1] -> refined candidate [B,12,1,1]"""
        parts = [g, sel4, v4] + ([ext4] if self.extra else [])
        r = self.mlp(torch.cat(parts, 1))
        return sel4 + self.bound * torch.tanh(r) * self.mask

    def forward(self, feat, sel, v0, ext=None):
        B = sel.shape[0]
        g = self.trunk(feat)
        ext4 = ext[:, :self.extra].float().view(B, self.extra, 1, 1) if self.extra else None
        return self.mlp_res(g, sel.view(B, 12, 1, 1), (v0.view(B, 1, 1, 1) / 10.0), ext4).view(B, 12)


def batch_feat(arm, b, dev):
    cur = b["bev"].to(dev).float()
    if arm == "single":
        return cur
    prev = b["prev"].to(dev).float()
    if arm == "stack":                               # perception-only temporal input: concatenate without ego-motion (odometry) alignment
        return torch.cat([cur, prev], 1)
    theta = make_warp_theta(b["rel"].to(dev).float())
    grid = F.affine_grid(theta, prev.shape, align_corners=False)
    warped = F.grid_sample(prev, grid, align_corners=False)
    return torch.cat([cur, warped], 1)


def ext_of(b, extra, dev):
    if not extra:
        return None
    cols = [b["acc"] / 3.0, b["brk"]][:extra]
    return torch.stack(cols, 1).to(dev)


def evaluate(head, arm, D, dev, bs=512):
    head.eval()
    s_base = s_new = p0b = p0n = n = 0
    with torch.no_grad():
        for i in range(0, len(D["cand"]), bs):
            b = {k: D[k][i:i + bs] for k in D}
            sel, _ = selected(b["cand"].to(dev), b["acc"].to(dev))
            g = b["gt"].to(dev)
            new = head(batch_feat(arm, b, dev), sel, b["v0"].to(dev), ext_of(b, head.extra, dev))
            s_base += ade(sel, g).sum().item(); s_new += ade(new, g).sum().item()
            p0b += (sel.view(-1, 6, 2)[:, 0] - g.view(-1, 6, 2)[:, 0]).norm(dim=-1).sum().item()
            p0n += (new.view(-1, 6, 2)[:, 0] - g.view(-1, 6, 2)[:, 0]).norm(dim=-1).sum().item()
            n += len(sel)
    return {"base": s_base / n, "new": s_new / n, "p0_base": p0b / n, "p0_new": p0n / n, "n": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--arm", choices=["single", "fuse", "stack"], required=True)
    ap.add_argument("--gap", type=int, default=1, help="distance of the previous frame for stack/fuse (in frames)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--bound", type=float, default=2.0)
    ap.add_argument("--acc-sel", type=float, default=0.0)
    ap.add_argument("--extra", default="", help="extra scalar inputs: '' | acc | acc,brk (acc is normalized by /3)")
    ap.add_argument("--npu", type=int, default=0, help="1: F3HeadNPU (BN / GlobalAveragePool / Conv1x1; for integer-pooled caches)")
    a = ap.parse_args()
    global ACC_SEL
    ACC_SEL = a.acc_sel
    dev = "cuda"
    tr, va, nf = load(a.cache, gap=a.gap)
    print(f"[f3:{a.arm}] scenes={nf} train={len(tr['cand'])} val={len(va['cand'])} feat={tuple(tr['bev'].shape[1:])}", flush=True)
    n_ext = len([x for x in a.extra.split(",") if x])
    cin = 96 if a.arm == "single" else 192
    head = (F3HeadNPU(cin, a.bound, n_ext) if a.npu else F3Head(cin, a.bound, n_ext)).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=1e-4)
    N = len(tr["cand"])
    os.makedirs(a.out, exist_ok=True)
    best = 9e9
    for ep in range(a.epochs):
        head.train()
        perm = torch.randperm(N)
        t0 = time.time(); tot = 0.0; nb = 0
        for i in range(0, N, a.bs):
            idx = perm[i:i + a.bs]
            b = {k: tr[k][idx] for k in tr}
            sel, _ = selected(b["cand"].to(dev), b["acc"].to(dev))
            g = b["gt"].to(dev)
            new = head(batch_feat(a.arm, b, dev), sel, b["v0"].to(dev), ext_of(b, head.extra, dev))
            loss = (new.view(-1, 6, 2) - g.view(-1, 6, 2)).norm(dim=-1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        ev = evaluate(head, a.arm, va, dev)
        print(f"[f3:{a.arm}] ep{ep} loss={tot/nb:.4f} val base={ev['base']:.4f} new={ev['new']:.4f} "
              f"({100*(ev['new']/ev['base']-1):+.1f}%) p0 {ev['p0_base']:.4f}->{ev['p0_new']:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        if ev["new"] < best:
            best = ev["new"]
            torch.save({"head": head.state_dict(), "arm": a.arm, "gap": a.gap, "val": ev, "extra": n_ext, "npu": a.npu,
                        "bound": a.bound, "feat": tuple(tr["bev"].shape[1:])}, os.path.join(a.out, "best.pt"))
            json.dump(ev, open(os.path.join(a.out, "best.json"), "w"))
    print(f"[f3:{a.arm}] done best={best:.4f}", flush=True)


if __name__ == "__main__":
    main()

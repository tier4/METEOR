#!/usr/bin/env python3
"""Train the post-hoc BEV-seg refiner (roadmap 3f, r33).

A small U-Net (bevlane.model.BEVSegRefiner) that sharpens / completes the
FROZEN main model's BEV-seg logits, targeting the >50 m dropout. The
backbone is loaded from an r32/r33 checkpoint and never updated, so this is
zero-risk to the running training and to near-range accuracy (the refiner is
residual + zero-init -> identity at start).

Single GPU:
    python bevlane/train_refiner.py --ckpt out/bevlane_ckpt_r32/last.pt \
        --model v38 --epochs 4 --batch 8 --out out/refiner_r33

8-GPU DDP:
    torchrun --nproc_per_node=8 bevlane/train_refiner.py \
        --ckpt out/bevlane_ckpt_r33/last.pt --model v39 --batch 6 \
        --out out/refiner_r33
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bevlane.dataset import BevLaneDataset          # noqa: E402
from bevlane.model import MODELS, BEVSegRefiner, N_CLASSES  # noqa: E402
from bevlane.train import (CLASS_NAMES, CLASS_W, lovasz_softmax,  # noqa: E402
                           split_scenes)

# forward-distance bands (rows): x = 80 - r*0.2
BANDS = [("40-80m", 0, 200), ("20-40m", 200, 300), ("0-20m", 300, 400)]


@torch.no_grad()
def band_iou(frozen, refiner, loader, device, ctx_ch, max_b=40):
    frozen.eval(); refiner.eval()
    iR = np.zeros((len(BANDS), N_CLASSES)); uR = iR.copy()
    iF = iR.copy(); uF = iR.copy()
    for bi, batch in enumerate(loader):
        if bi >= max_b:
            break
        imgs, K, Tc, gt = (t.to(device, non_blocking=True) for t in batch[:4])
        with torch.autocast("cuda", torch.float16):
            out = frozen(imgs, K, Tc)
            logits = (out[0] if isinstance(out, tuple) else out).float()
            ctx = frozen.lane_input().float() if ctx_ch else None
            ref = refiner(logits, ctx)
        pr, pf = logits.argmax(1), ref.argmax(1)
        m = gt > 0
        for bidx, (_, r0, r1) in enumerate(BANDS):
            gb, mb = gt[:, r0:r1], m[:, r0:r1]
            for c in range(1, N_CLASSES):
                gi = gb == c
                pi = (pr[:, r0:r1] == c) & mb
                iR[bidx, c] += (pi & gi).sum().item()
                uR[bidx, c] += (pi | gi).sum().item()
                pi = (pf[:, r0:r1] == c) & mb
                iF[bidx, c] += (pi & gi).sum().item()
                uF[bidx, c] += (pi | gi).sum().item()
    refiner.train()
    return iR, uR, iF, uF


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="out/bevlane")
    ap.add_argument("--ckpt", required=True, help="frozen main-model ckpt")
    ap.add_argument("--model", default="v38")
    ap.add_argument("--gt-key", default="gt_cons")
    ap.add_argument("--n-seg2d", type=int, default=21)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.0)
    ap.add_argument("--width", type=int, default=48)
    ap.add_argument("--ctx", type=int, default=0,
                    help="raw-BEV context channels into refiner (0=off, 96=on)")
    ap.add_argument("--far-w", type=float, default=3.0,
                    help="extra CE weight ramp toward the far rows")
    ap.add_argument("--lovasz-w", type=float, default=0.3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit-train", type=int, default=0)
    ap.add_argument("--out", default="out/refiner_r33")
    args = ap.parse_args()

    ddp = "RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl")
        rank = dist.get_rank(); world = dist.get_world_size()
        local = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local)
        device = torch.device(f"cuda:{local}")
    else:
        rank, world, device = 0, 1, torch.device("cuda:0")
    is_main = rank == 0
    lr = args.lr or 3e-4 * world ** 0.5
    if is_main:
        os.makedirs(args.out, exist_ok=True)

    train_s, val_s = split_scenes(args.root)

    # frozen backbone
    mkw = {"n_seg": args.n_seg2d}
    frozen = MODELS[args.model](**mkw).to(device)
    sd = torch.load(args.ckpt, map_location="cpu")
    sd = sd.get("model", sd)
    miss, unexp = frozen.load_state_dict(sd, strict=False)
    if is_main:
        print(f"[frozen] {args.ckpt} missing={len(miss)} unexpected={len(unexp)}",
              flush=True)
    frozen.eval()
    for p in frozen.parameters():
        p.requires_grad_(False)

    refiner = BEVSegRefiner(N_CLASSES, ctx_ch=args.ctx, width=args.width).to(device)
    n_par = sum(p.numel() for p in refiner.parameters()) / 1e6
    if is_main:
        print(f"[refiner] width={args.width} ctx={args.ctx} "
              f"params={n_par:.2f}M (residual, zero-init=identity)", flush=True)
    refiner = DDP(refiner, device_ids=[local]) if ddp else refiner
    net0 = refiner.module if ddp else refiner

    tr = BevLaneDataset(args.root, train_s, gt_key=args.gt_key, with_depth=False)
    va = BevLaneDataset(args.root, val_s, max_per_scene=4, gt_key=args.gt_key,
                        with_depth=False)
    sampler = DistributedSampler(tr) if ddp else None
    dl = DataLoader(tr, batch_size=args.batch, shuffle=sampler is None,
                    sampler=sampler, num_workers=args.workers, pin_memory=True,
                    drop_last=True, persistent_workers=args.workers > 0)
    dv = DataLoader(va, batch_size=args.batch, shuffle=False,
                    num_workers=2, pin_memory=True)
    if is_main:
        print(f"train {len(tr)} / {len(train_s)} scenes; "
              f"val {len(va)} / {len(val_s)} scenes; world={world} lr={lr:.1e}",
              flush=True)

    cw = CLASS_W.clone().to(device)
    opt = torch.optim.AdamW(net0.parameters(), lr=lr, weight_decay=1e-4)
    steps = args.epochs * (args.limit_train or len(dl))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    scaler = torch.cuda.amp.GradScaler()

    step = 0
    for ep in range(args.epochs):
        if ddp:
            sampler.set_epoch(ep)
        for bi, batch in enumerate(dl):
            if args.limit_train and bi >= args.limit_train:
                break
            imgs, K, Tc, gt = (t.to(device, non_blocking=True)
                               for t in batch[:4])
            with torch.no_grad(), torch.autocast("cuda", torch.float16):
                out = frozen(imgs, K, Tc)
                logits = (out[0] if isinstance(out, tuple) else out).float()
                ctx = frozen.lane_input().float() if args.ctx else None
            with torch.autocast("cuda", torch.float16):
                ref = refiner(logits, ctx)
                ce = F.cross_entropy(ref.float(), gt, weight=cw,
                                     ignore_index=0, reduction="none")
                H2 = ce.shape[-2]
                rows = torch.arange(H2, device=device, dtype=ce.dtype)
                # ramp toward the top rows (far forward = row 0)
                wrow = 1 + args.far_w * (1 - rows / (H2 - 1)).clamp(min=0)
                loss = (ce * wrow.view(1, -1, 1)).mean()
                if args.lovasz_w > 0:
                    loss = loss + args.lovasz_w * lovasz_softmax(
                        ref.float(), gt, ignore=0)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            step += 1
            if is_main and step % 100 == 0:
                print(f"ep{ep} step{step}/{steps} loss={loss.item():.4f} "
                      f"lr={sched.get_last_lr()[0]:.2e}", flush=True)
            if is_main and step % 1000 == 0:
                iR, uR, iF, uF = band_iou(frozen, net0, dv, device, args.ctx)
                _report(iR, uR, iF, uF, ep, step)
        if is_main:
            torch.save({"refiner": net0.state_dict(), "epoch": ep,
                        "args": vars(args)},
                       os.path.join(args.out, "last.pt"))
            print(f"[ckpt] saved epoch {ep}", flush=True)
    if is_main:
        iR, uR, iF, uF = band_iou(frozen, net0, dv, device, args.ctx, max_b=120)
        _report(iR, uR, iF, uF, args.epochs, step, tag="FINAL")
        print("REFINER DONE", flush=True)
    if ddp:
        dist.destroy_process_group()


def _report(iR, uR, iF, uF, ep, step, tag=""):
    print(f"[refBand ep{ep} step{step}] {tag} raw->refined IoU", flush=True)
    for c in [1, 3, 4, 5, 6]:
        row = "  " + CLASS_NAMES[c].ljust(10)
        for bidx, (nm, _, _) in enumerate(BANDS):
            r = iR[bidx, c] / uR[bidx, c] if uR[bidx, c] else float("nan")
            f_ = iF[bidx, c] / uF[bidx, c] if uF[bidx, c] else float("nan")
            row += f"  {nm} {r:.3f}->{f_:.3f}"
        print(row, flush=True)


if __name__ == "__main__":
    main()

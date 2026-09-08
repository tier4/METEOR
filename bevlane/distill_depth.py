#!/usr/bin/env python3
"""Train ONLY a regression depth head, by matching the frozen model's BEV.

Why not depth GT: the depth head's output is consumed by the BEV lift, and the
lift's output is what all twelve heads read. Two heads with the same depth error
can still produce different BEV features -- the 64-bin histogram carries a
SHAPE (multi-modal, wide where uncertain) that a regression plus a fixed kernel
does not. Training on depth GT alone would therefore move the BEV feature and
every frozen head downstream would be reading an input distribution it never saw.

So the objective is the BEV feature itself: teacher = the shipped v48 model,
student = v50 (same weights, 0.33M regression depth head, one-channel lift), and
the loss is |B_student - B_teacher| with the depth GT as a light regulariser.
Only the new head and its kernel width are trainable; everything else is frozen
and shared, so a match on B means the eleven other heads are undisturbed BY
CONSTRUCTION -- and it is checkable afterwards with eval_val_spread.py.

    torchrun --nproc_per_node=7 bevlane/distill_depth.py \
        --ckpt out/bevlane_ckpt_r49/last.pt --steps 6000 --out out/distill_v50
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bevlane.dataset import BevLaneDataset                      # noqa: E402
from bevlane.model import MODELS                                # noqa: E402
from bevlane.train import split_scenes                          # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--teacher", default="v48")
    ap.add_argument("--student", default="v50")
    ap.add_argument("--root", default="out/bevlane")
    ap.add_argument("--train-list", default="out/round49_scenes.txt")
    ap.add_argument("--out", default="out/distill_v50")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--depth-w", type=float, default=0.1,
                    help="weight of the depth-GT regulariser")
    ap.add_argument("--n-seg2d", type=int, default=21)
    ap.add_argument("--report", type=int, default=200)
    a = ap.parse_args()

    ddp = int(os.environ.get("WORLD_SIZE", 1)) > 1
    rank = int(os.environ.get("RANK", 0))
    local = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    if ddp:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local)
    dev = f"cuda:{local}"
    is_main = rank == 0
    if is_main:
        os.makedirs(a.out, exist_ok=True)

    sd = torch.load(a.ckpt, map_location="cpu")["model"]
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    teacher = MODELS[a.teacher](n_seg=a.n_seg2d).to(dev).eval()
    teacher.load_state_dict(sd, strict=False)
    for p in teacher.parameters():
        p.requires_grad_(False)
    student = MODELS[a.student](n_seg=a.n_seg2d).to(dev)
    # the new depth head is shape-mismatched by design; drop those tensors
    # rather than letting load_state_dict raise on them
    cur = student.state_dict()
    keep = {k: v for k, v in sd.items()
            if k in cur and cur[k].shape == v.shape}
    reinit = [k for k in cur if k not in keep]
    miss = student.load_state_dict(keep, strict=False)
    miss = type("M", (), {"missing_keys": reinit})()
    student.eval()                      # frozen BN everywhere; only the head trains
    train_names = {"depth_head", "log_sigma"}
    n_tr = 0
    for n, p in student.named_parameters():
        p.requires_grad_(any(n.startswith(t) for t in train_names))
        n_tr += p.numel() if p.requires_grad else 0
    if is_main:
        print(f"[distill] teacher {a.teacher} frozen | student {a.student}: "
              f"{n_tr / 1e6:.2f}M trainable of "
              f"{sum(p.numel() for p in student.parameters()) / 1e6:.2f}M "
              f"(re-init: {len(miss.missing_keys)} tensors)", flush=True)

    train_s, _ = split_scenes(a.root)
    if a.train_list and os.path.exists(a.train_list):
        keep = set(open(a.train_list).read().split())
        train_s = [s for s in train_s if s in keep]
    ds = BevLaneDataset(a.root, train_s, gt_key="gt_cons", with_depth=True,
                        depth_hw=(108, 192), augment=False, trim_start=3)
    if is_main:
        print(f"[distill] {len(ds)} samples / {len(train_s)} scenes", flush=True)
    sampler = DistributedSampler(ds) if ddp else None
    dl = DataLoader(ds, batch_size=a.batch, shuffle=sampler is None,
                    sampler=sampler, num_workers=a.workers, pin_memory=True,
                    drop_last=True, persistent_workers=a.workers > 0)

    model = DDP(student, device_ids=[local]) if ddp else student
    params = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr,
                                                total_steps=a.steps,
                                                pct_start=0.1)
    scaler = torch.cuda.amp.GradScaler()
    step = 0
    acc = np.zeros(4)
    t0 = time.time()
    while step < a.steps:
        if sampler is not None:
            sampler.set_epoch(step // max(len(dl), 1))
        for batch in dl:
            if step >= a.steps:
                break
            imgs = batch[0].to(dev, non_blocking=True)
            K = batch[1].to(dev, non_blocking=True)
            Tc = batch[2].to(dev, non_blocking=True)
            dgt = batch[4].to(dev, non_blocking=True)      # depth GT [B,8,h,w]
            with torch.no_grad(), torch.autocast("cuda", torch.float16):
                bt = teacher.compute_bev(imgs, K, Tc).float()
            with torch.autocast("cuda", torch.float16):
                net = model.module if ddp else model
                f = net.image_feats(imgs)
                dlog = net.depth_head(net.depth_up(f))
                dprob = dlog.softmax(1)
                B, N = imgs.shape[:2]
                bs = net.project_bev(dprob, net.ctx(f), K, Tc, B, N,
                                     imgs.shape[-2], imgs.shape[-1])
            bs = bs.float()
            scale = bt.abs().mean().clamp(min=1e-3)
            l_bev = (bs - bt).abs().mean() / scale
            # depth regulariser: metres, only where GT exists
            dm = net.depth_metres(dlog).float().view(B, N, *dlog.shape[-2:])
            g = dgt.float()
            m = (g > 0.5) & (g < 90.0)
            l_d = ((dm - g).abs() * m).sum() / m.sum().clamp(min=1) / 10.0 \
                if m.any() else dm.sum() * 0.0
            loss = l_bev + a.depth_w * l_d
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            acc += [float(l_bev), float(l_d), float(loss), 1]
            if is_main and step % a.report == 0:
                n = max(acc[3], 1)
                print(f"step {step}/{a.steps} bev-rel {acc[0] / n:.4f} "
                      f"depth-MAE {acc[1] / n * 10:.2f}m "
                      f"sigma {float(net.log_sigma.exp()):.2f}m "
                      f"({(time.time() - t0) / step:.2f}s/it)", flush=True)
                acc[:] = 0
                torch.save({"model": (model.module if ddp else model
                                      ).state_dict(),
                            "step": step, "args": vars(a)},
                           os.path.join(a.out, "last.pt"))
    if is_main:
        torch.save({"model": (model.module if ddp else model).state_dict(),
                    "step": step, "args": vars(a)},
                   os.path.join(a.out, "last.pt"))
        print("DISTILL DONE", flush=True)
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

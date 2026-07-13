#!/usr/bin/env python3
"""Train IPMSegNet on extracted bevlane samples.

Single GPU:  python3 bevlane/train.py
8-GPU DDP:   torchrun --nproc_per_node=8 bevlane/train.py --batch 14
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler

# avoid "resize storage that is not resizable" collate failures under many
# DDP workers (file-descriptor sharing exhausts FDs with extra sample tensors)
torch.multiprocessing.set_sharing_strategy("file_system")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bevlane.dataset import BevLaneDataset  # noqa: E402
from bevlane.model import MODELS, N_CLASSES  # noqa: E402

CLASS_NAMES = ["unlabeled", "road", "sidewalk", "crosswalk", "laneline",
               "stopline", "road_edge", "marking", "parking"]
CLASS_W = torch.tensor([0.0, 1.0, 1.0, 2.0, 5.0, 5.0, 1.5, 3.0, 1.0])
THIN = [3, 4, 5, 6, 7]
DICE_CLASSES = [3]        # area-like (crosswalk): want recall
LINE_CLASSES = [4, 5, 6]  # laneline / stopline / road_edge: precision/thin
# (marking/class 7 dropped from GT as noise -> not trained)


def dice_loss(logits, gt, classes=THIN, eps=1.0):
    """Soft dice over selected classes, masked to labeled cells."""
    prob = logits.softmax(1)
    m = (gt > 0).unsqueeze(1).float()
    loss = 0.0
    for c in classes:
        p = prob[:, c:c + 1] * m
        t = (gt == c).unsqueeze(1).float()
        inter = (p * t).sum((1, 2, 3))
        loss = loss + (1 - (2 * inter + eps)
                       / (p.sum((1, 2, 3)) + t.sum((1, 2, 3)) + eps)).mean()
    return loss / len(classes)


def tversky_loss(logits, gt, classes, alpha=0.2, beta=0.8, eps=1.0):
    """Tversky loss. beta>alpha penalizes false positives harder -> thinner,
    higher-precision predictions for line classes."""
    prob = logits.softmax(1)
    m = (gt > 0).unsqueeze(1).float()
    loss = 0.0
    for c in classes:
        p = prob[:, c:c + 1] * m
        t = (gt == c).unsqueeze(1).float()
        tp = (p * t).sum((1, 2, 3))
        fp = (p * (1 - t)).sum((1, 2, 3))
        fn = ((1 - p) * t).sum((1, 2, 3))
        ti = (tp + eps) / (tp + alpha * fn + beta * fp + eps)
        loss = loss + (1 - ti).mean()
    return loss / len(classes)


def _lovasz_grad(gt_sorted):
    gts = gt_sorted.sum()
    inter = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jacc = 1.0 - inter / union
    jacc[1:] = jacc[1:] - jacc[:-1].clone()
    return jacc


def lovasz_softmax(logits, gt, classes=None, ignore=0):
    """Multi-class Lovasz-Softmax (optimizes IoU -> sharp boundaries).

    Averaged over images and present classes. Heavy but O(P log P) per class.
    """
    prob = logits.softmax(1)
    B, C, H, W = prob.shape
    if classes is None:
        classes = list(range(1, C))
    total = 0.0
    n = 0
    for b in range(B):
        pr = prob[b].reshape(C, -1)          # [C, P]
        lb = gt[b].reshape(-1)               # [P]
        valid = lb != ignore
        if valid.sum() == 0:
            continue
        prv = pr[:, valid]
        lbv = lb[valid]
        for c in classes:
            fg = (lbv == c).float()
            if fg.sum() == 0:
                continue
            err = (fg - prv[c]).abs()
            err_s, perm = torch.sort(err, 0, descending=True)
            total = total + torch.dot(err_s, _lovasz_grad(fg[perm]))
            n += 1
    return total / max(n, 1)


def boundary_weight(gt, radius=2, w=4.0):
    """Per-pixel weight map: `w` on cells within `radius` of a class boundary."""
    g = gt.float().unsqueeze(1)
    mx = F.max_pool2d(g, 2 * radius + 1, 1, radius)
    mn = -F.max_pool2d(-g, 2 * radius + 1, 1, radius)
    bnd = (mx != mn).float().squeeze(1)      # neighborhood has >1 label
    return 1.0 + (w - 1.0) * bnd


def split_scenes(root):
    scenes = sorted(os.listdir(root))
    val = [s for s in scenes if "2026-01-23T15-26-01" in s]
    train = [s for s in scenes if s not in set(val)]
    return train, val


@torch.no_grad()
def evaluate(model, loader, device, max_batches=80):
    model.eval()
    inter = np.zeros(N_CLASSES)
    union = np.zeros(N_CLASSES)
    for bi, batch in enumerate(loader):
        if bi >= max_batches:
            break
        imgs, K, Tc, gt = (t.to(device, non_blocking=True) for t in batch[:4])
        with torch.autocast("cuda", torch.float16):
            logits = model(imgs, K, Tc)
            if isinstance(logits, tuple):
                logits = logits[0]
        pred = logits.argmax(1)
        m = gt > 0
        for c in range(1, N_CLASSES):
            pi, gi = (pred == c) & m, gt == c
            inter[c] += (pi & gi).sum().item()
            union[c] += (pi | gi).sum().item()
    ious = {CLASS_NAMES[c]: inter[c] / union[c] if union[c] else float("nan")
            for c in range(1, N_CLASSES)}
    model.train()
    return ious


@torch.no_grad()
def evaluate_seg2d(model, loader, device, n_cls, max_batches=40):
    """Per-class IoU of the 2D seg head (batch index 4 = seg2d GT, 255 ignore).

    Frames whose seg2d GT is not extracted yet are all-ignore and simply
    don't contribute. Returns {} if nothing labeled was seen.
    """
    model.eval()
    inter = np.zeros(n_cls)
    union = np.zeros(n_cls)
    for bi, batch in enumerate(loader):
        if bi >= max_batches:
            break
        imgs, K, Tc = (t.to(device, non_blocking=True) for t in batch[:3])
        sg = batch[4].to(device, non_blocking=True)
        with torch.autocast("cuda", torch.float16):
            out = model(imgs, K, Tc)
        if not (isinstance(out, tuple) and len(out) > 2):
            break
        pred = out[2].argmax(2)
        valid = sg != 255
        for c in range(n_cls):
            pi, gi = (pred == c) & valid, sg == c
            inter[c] += (pi & gi).sum().item()
            union[c] += (pi | gi).sum().item()
    model.train()
    return {c: inter[c] / union[c] for c in range(n_cls) if union[c]}


@torch.no_grad()
def evaluate_ego(model, loader, device, ego_idx, max_batches=40):
    """E2E head metrics: trajectory ADE/FDE [m], steer MAE [rad],
    accel MAE [m/s^2], brake accuracy. Valid frames only."""
    model.eval()
    n = ade = fde = smae = amae = bacc = 0.0
    nc = adec = 0.0
    for bi, batch in enumerate(loader):
        if bi >= max_batches:
            break
        imgs, K, Tc = (t.to(device, non_blocking=True) for t in batch[:3])
        eg = batch[ego_idx].to(device, non_blocking=True)
        with torch.autocast("cuda", torch.float16):
            out = model(imgs, K, Tc, eg[:, 12])
        if not (isinstance(out, tuple) and len(out) >= 8):
            break
        p = out[7].float()
        v = eg[:, 16]
        if v.sum() == 0:
            continue
        d = (p[:, :12].view(-1, 6, 2) - eg[:, :12].view(-1, 6, 2)).norm(dim=2)
        ade += (d.mean(1) * v).sum().item()
        fde += (d[:, -1] * v).sum().item()
        vc = v * (eg[:, 11].abs() > 2.0).float()   # curve subset |lat@3s|>2m
        adec += (d.mean(1) * vc).sum().item()
        nc += vc.sum().item()
        smae += ((p[:, 12] - eg[:, 14]).abs() * v).sum().item()
        amae += ((p[:, 13] - eg[:, 13]).abs() * v).sum().item()
        bacc += (((p[:, 14] > 0).float() == eg[:, 15]).float() * v).sum().item()
        n += v.sum().item()
    model.train()
    if n == 0:
        return None
    return {"ade": ade / n, "fde": fde / n, "steer": smae / n,
            "acc": amae / n, "brake": bacc / n,
            "ade_c": adec / nc if nc else float("nan")}


@torch.no_grad()
def evaluate_occ(model, loader, device, occ_idx, max_batches=25):
    """Occupancy IoU per class (0 free + 9 occupied; 255 unknown skipped)."""
    model.eval()
    inter = np.zeros(10)
    union = np.zeros(10)
    for bi, batch in enumerate(loader):
        if bi >= max_batches:
            break
        imgs, K, Tc = (t.to(device, non_blocking=True) for t in batch[:3])
        og = batch[occ_idx].to(device, non_blocking=True)
        with torch.autocast("cuda", torch.float16):
            out = model(imgs, K, Tc)
        if not (isinstance(out, tuple) and len(out) == 9):
            break
        pred = out[8].argmax(1)
        valid = og != 255
        if valid.sum() == 0:
            continue
        for c in range(10):
            pi, gi = (pred == c) & valid, og == c
            inter[c] += (pi & gi).sum().item()
            union[c] += (pi | gi).sum().item()
    model.train()
    return {c: inter[c] / union[c] for c in range(10) if union[c]}


@torch.no_grad()
def class_pr(model, loader, device, cls, max_batches=40):
    """Precision/recall + pred/GT area ratio for a class (labeled cells only)."""
    model.eval()
    tp = fp = fn = pcount = gcount = 0
    for bi, batch in enumerate(loader):
        if bi >= max_batches:
            break
        imgs, K, Tc, gt = (t.to(device, non_blocking=True) for t in batch[:4])
        with torch.autocast("cuda", torch.float16):
            logits = model(imgs, K, Tc)
            if isinstance(logits, tuple):
                logits = logits[0]
        pred = logits.argmax(1)
        pc, gc = pred == cls, gt == cls
        lab = gt > 0
        tp += (pc & gc).sum().item()
        fp += (pc & lab & ~gc).sum().item()
        fp += (pc & ~lab).sum().item()
        fn += (~pc & gc).sum().item()
        pcount += pc.sum().item()
        gcount += gc.sum().item()
    model.train()
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    ratio = pcount / max(gcount, 1)
    return p, r, ratio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="out/bevlane")
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--batch", type=int, default=14, help="per-GPU batch")
    ap.add_argument("--lr", type=float, default=None, help="default: 3e-4*worldsize^0.5")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="out/bevlane_ckpt")
    ap.add_argument("--limit-train", type=int, default=None)
    ap.add_argument("--model", default="v1", choices=["v1", "v2", "v3s", "lss", "v8", "v13", "v13d", "v14d", "v15", "v16", "v17", "v18", "v19", "v20"])
    ap.add_argument("--depth-w", type=float, default=0.3)
    ap.add_argument("--seg2d-w", type=float, default=0.5)
    ap.add_argument("--seg2d-key", default="seg2d",
                    help="manifest key: seg2d (12cls) or seg2d21 (csv 21cls)")
    ap.add_argument("--n-seg2d", type=int, default=12,
                    help="2D seg head classes (21 with --seg2d-key seg2d21)")
    ap.add_argument("--box-w", type=float, default=0.0,
                    help="BEV 3D-box occupancy multi-task loss weight (v15)")
    ap.add_argument("--bbox2d-w", type=float, default=0.0,
                    help="per-camera 10-class 2D bbox det loss weight (v17)")
    ap.add_argument("--ego-w", type=float, default=0.0,
                    help="E2E ego head loss weight: traj/steer/accel/brake (v18)")
    ap.add_argument("--occ-w", type=float, default=0.0,
                    help="3D semantic occupancy loss weight (v20)")
    ap.add_argument("--aug", action="store_true")
    ap.add_argument("--dice-w", type=float, default=0.0)
    ap.add_argument("--far-w", type=float, default=0.0,
                    help="extra loss weight at far rows (linear, max 1+far_w)")
    ap.add_argument("--boundary-w", type=float, default=0.0,
                    help="extra CE weight near class boundaries (sharpening)")
    ap.add_argument("--lovasz-w", type=float, default=0.0,
                    help="Lovasz-Softmax loss weight (IoU-direct, sharp edges)")
    ap.add_argument("--tversky-w", type=float, default=0.0,
                    help="Tversky loss weight on line classes (FP-heavy -> thin)")
    ap.add_argument("--seg-w", type=float, default=1.0)
    ap.add_argument("--init-ckpt", default="")
    ap.add_argument("--train-list", default="",
                    help="file of scene names to restrict training to")
    ap.add_argument("--min-cov-core", type=float, default=0.03,
                    help="min labeled frac +-30m band (0=off)")
    ap.add_argument("--min-cov-fwd", type=float, default=0.005,
                    help="min labeled frac +30..80m band (drops stationary)")
    ap.add_argument("--trim-end", type=int, default=10,
                    help="drop last K frames/scene (weak forward GT at scene end)")
    ap.add_argument("--train-bg", action="store_true",
                    help="supervise unlabeled(0) as background class")
    ap.add_argument("--dontcare-sidewalk", action="store_true")
    ap.add_argument("--gt-key", default="gt", choices=["gt", "gt_vec"],
                    help="gt = raster autolabel; gt_vec = hybrid vector-line GT")
    args = ap.parse_args()

    ddp = "RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world = dist.get_world_size()
        local = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local)
        device = torch.device(f"cuda:{local}")
    else:
        rank, world, device = 0, 1, torch.device("cuda:0")
    lr = args.lr or 3e-4 * world ** 0.5
    is_main = rank == 0
    if is_main:
        os.makedirs(args.out, exist_ok=True)

    train_s, val_s = split_scenes(args.root)
    use_depth = args.model in ("lss", "v8", "v13", "v13d", "v14d", "v15", "v16", "v17", "v18", "v19", "v20") and args.depth_w > 0
    use_seg2d = args.model in ("v13", "v13d", "v14d", "v15", "v16", "v17", "v18", "v19", "v20") and args.seg2d_w > 0
    use_box = args.model == "v15" and args.box_w > 0
    use_boxdet = args.model in ("v16", "v17", "v18", "v19", "v20") and args.box_w > 0
    use_bbox2d = args.model in ("v17", "v18", "v19", "v20") and args.bbox2d_w > 0
    use_ego = args.model in ("v18", "v19", "v20") and args.ego_w > 0
    use_occ = args.model == "v20" and args.occ_w > 0
    if args.train_list:                       # restrict train to a scene list
        keep = set(open(args.train_list).read().split())
        train_s = [s for s in train_s if s in keep]
    # v13d depth GT is stride-4 of 768 (108x192); resize any mixed-res depth
    depth_hw = (108, 192) if args.model in ("v13d", "v14d", "v15", "v16", "v17", "v18", "v19", "v20") else None
    tr = BevLaneDataset(args.root, train_s, gt_key=args.gt_key,
                        dontcare_sidewalk=args.dontcare_sidewalk,
                        with_depth=use_depth, augment=args.aug,
                        with_seg2d=use_seg2d, depth_hw=depth_hw,
                        with_box=use_box, with_boxdet=use_boxdet,
                        with_bbox2d=use_bbox2d, with_ego=use_ego,
                        with_occ=use_occ,
                        trim_start=3, trim_end=args.trim_end,
                        min_cov_core=args.min_cov_core,
                        min_cov_fwd=args.min_cov_fwd,
                        seg2d_key=args.seg2d_key)
    if args.limit_train:
        idx = np.random.RandomState(0).permutation(len(tr))[:args.limit_train]
        tr = torch.utils.data.Subset(tr, idx.tolist())
    if is_main:
        # val never needs depth GT (BEV mIoU eval only) -> with_depth=False
        va = BevLaneDataset(args.root, val_s, max_per_scene=8, gt_key=args.gt_key,
                            dontcare_sidewalk=args.dontcare_sidewalk,
                            with_depth=False, with_seg2d=use_seg2d,
                            seg2d_key=args.seg2d_key, with_ego=use_ego,
                            with_occ=use_occ,
                            trim_start=3, trim_end=args.trim_end)
        print(f"train {len(tr)} samples / {len(train_s)} scenes; "
              f"val {len(va)} samples / {len(val_s)} scenes; "
              f"world={world} lr={lr:.1e}", flush=True)
        dv = DataLoader(va, batch_size=args.batch, shuffle=False,
                        num_workers=4, pin_memory=True)

    sampler = DistributedSampler(tr) if ddp else None
    # with depth the extra per-sample tensor + many DDP workers exhaust the
    # shared-memory collate ("resize storage not resizable"); lighten the loader.
    pin = not use_depth
    nw = args.workers
    # num_workers=0 bypasses the worker->main shared-memory collate entirely
    # (the "resize storage not resizable" failure under DDP + depth tensor)
    dl_kw = dict(persistent_workers=True, prefetch_factor=2 if use_depth else 4) \
        if nw > 0 else {}
    dl = DataLoader(tr, batch_size=args.batch, shuffle=sampler is None,
                    sampler=sampler, num_workers=nw, pin_memory=pin,
                    drop_last=True, **dl_kw)

    mkw = {"n_seg": args.n_seg2d} \
        if args.model in ("v13", "v13d", "v14d", "v15", "v16", "v17", "v18", "v19", "v20") else {}
    model = MODELS[args.model](**mkw).to(device)
    if args.init_ckpt:
        sd = torch.load(args.init_ckpt, map_location="cpu")["model"]
        cur = model.state_dict()   # drop shape-mismatched heads (12->21cls seg)
        sd = {k: v for k, v in sd.items()
              if k in cur and cur[k].shape == v.shape}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if is_main:
            print(f"[init] {args.init_ckpt} missing={len(missing)} "
                  f"unexpected={len(unexpected)}", flush=True)
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local],
            find_unused_parameters=(args.seg_w == 0 or
                                    (args.model in ("v13", "v13d", "v14d", "v15", "v16", "v17", "v18", "v19", "v20") and not use_seg2d)))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    total_steps = len(dl) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr,
                                                total_steps=total_steps)
    scaler = torch.cuda.amp.GradScaler()
    cw = CLASS_W.clone()
    ignore = 0
    if args.train_bg:
        cw[0] = 0.5
        ignore = -100
    cw = cw.to(device)

    step = 0
    t0 = time.time()
    best = 0.0
    for ep in range(args.epochs):
        if sampler:
            sampler.set_epoch(ep)
        for batch in dl:
            batch = [t.to(device, non_blocking=True) for t in batch]
            imgs, K, Tc, gt = batch[:4]
            # extra batch tensors follow (imgs,K,T,gt) in fixed order
            bi = 4
            depth_gt = batch[bi] if use_depth else None
            bi += 1 if use_depth else 0
            seg2d_gt = batch[bi] if use_seg2d else None
            bi += 1 if use_seg2d else 0
            box_gt = batch[bi] if use_box else None
            bi += 1 if use_box else 0
            det_boxes = batch[bi] if use_boxdet else None
            det_n = batch[bi + 1] if use_boxdet else None
            bi += 2 if use_boxdet else 0
            bb2d = batch[bi] if use_bbox2d else None
            nb2d = batch[bi + 1] if use_bbox2d else None
            bi += 2 if use_bbox2d else 0
            ego_gt = batch[bi] if use_ego else None
            bi += 1 if use_ego else 0
            occ_gt = batch[bi] if use_occ else None
            with torch.autocast("cuda", torch.float16):
                # v18+ is conditioned on the current speed (ego_gt col 12)
                out = model(imgs, K, Tc, ego_gt[:, 12]) if use_ego \
                    else model(imgs, K, Tc)
                net0 = model.module if ddp else model
                # robustly unpack: v15 -> 4-tuple, v13* -> 3, lss/v8 -> 2
                logits, dlog, seg2d, boxl, hm, rg = out, None, None, None, None, None
                hm2d, rg2d, ego_pred, occ_pred = None, None, None, None
                if isinstance(out, tuple):
                    logits = out[0]
                    dlog = out[1] if len(out) > 1 else None
                    seg2d = out[2] if len(out) > 2 else None
                    if len(out) >= 5:            # v16/v17: hm + reg det head
                        hm, rg = out[3], out[4]
                        if len(out) >= 7:        # v17: + per-cam 2D det head
                            hm2d, rg2d = out[5], out[6]
                        if len(out) >= 8:        # v18+: E2E ego head
                            ego_pred = out[7]
                        occ_pred = out[8] if len(out) == 9 else None
                    elif len(out) > 3:
                        boxl = out[3]
                loss = 0.0
                if occ_pred is not None and use_occ:
                    loss = loss + args.occ_w * net0.occ_loss(occ_pred.float(),
                                                             occ_gt)
                if ego_pred is not None and use_ego:
                    loss = loss + args.ego_w * net0.ego_loss(ego_pred.float(),
                                                             ego_gt)
                if dlog is not None and use_depth:
                    loss = loss + args.depth_w * net0.depth_loss(dlog.float(), depth_gt)
                if seg2d is not None and use_seg2d:
                    loss = loss + args.seg2d_w * net0.seg2d_loss(seg2d.float(), seg2d_gt)
                if boxl is not None and use_box:
                    loss = loss + args.box_w * net0.box_loss(boxl.float(), box_gt)
                if hm is not None and use_boxdet:
                    loss = loss + args.box_w * net0.boxdet_loss(hm, rg, det_boxes,
                                                                det_n)
                if hm2d is not None and use_bbox2d:
                    loss = loss + args.bbox2d_w * net0.bbox2d_loss(
                        hm2d, rg2d, bb2d, nb2d)
                if args.seg_w > 0:
                    ce = F.cross_entropy(logits, gt, weight=cw,
                                         ignore_index=ignore, reduction="none")
                    H2 = ce.shape[-2]
                    wmap = torch.ones_like(ce)
                    if args.far_w > 0:            # upweight far (top/bottom) rows
                        rows = torch.arange(H2, device=ce.device, dtype=ce.dtype)
                        wrow = 1 + args.far_w * (rows - (H2 - 1) / 2).abs() \
                            / ((H2 - 1) / 2)
                        wmap = wmap * wrow.view(1, -1, 1)
                    if args.boundary_w > 0:      # sharpen class boundaries
                        wmap = wmap * boundary_weight(gt, radius=2,
                                                      w=1 + args.boundary_w)
                    loss = loss + args.seg_w * (ce * wmap).mean()
                if args.dice_w > 0:
                    loss = loss + args.dice_w * dice_loss(
                        logits.float(), gt, classes=DICE_CLASSES)
                if args.lovasz_w > 0:
                    loss = loss + args.lovasz_w * lovasz_softmax(
                        logits.float(), gt, ignore=ignore)
                if args.tversky_w > 0:
                    loss = loss + args.tversky_w * tversky_loss(
                        logits.float(), gt, classes=LINE_CLASSES,
                        alpha=0.2, beta=0.8)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            if is_main and step % 50 == 0:
                print(f"ep{ep} step{step}/{total_steps} loss={loss.item():.4f} "
                      f"lr={sched.get_last_lr()[0]:.2e} "
                      f"({(time.time() - t0) / step:.2f}s/it)", flush=True)
        if is_main:
            net = model.module if ddp else model
            ious = evaluate(net, dv, device)
            miou = float(np.nanmean(list(ious.values())))
            rp, rr, _ = class_pr(net, dv, device, 1)
            ep_, er_, eratio = class_pr(net, dv, device, 6)
            print(f"[val ep{ep}] mIoU={miou:.3f} " +
                  " ".join(f"{k}={v:.3f}" for k, v in ious.items()) +
                  f" | road P={rp:.3f} R={rr:.3f}" +
                  f" | redge P={ep_:.3f} R={er_:.3f} x{eratio:.2f}", flush=True)
            if use_seg2d:
                s2 = evaluate_seg2d(net, dv, device, args.n_seg2d)
                if s2:
                    key = {0: "bg", 8: "mark", 11: "road", 12: "swalk",
                           13: "lane", 20: "pole"}
                    print(f"[val2d ep{ep}] mIoU={np.mean(list(s2.values())):.3f} "
                          + " ".join(f"{key[c]}={s2[c]:.3f}"
                                     for c in key if c in s2), flush=True)
            if use_occ:
                oc = evaluate_occ(net, dv, device,
                                  4 + int(use_seg2d) + int(use_ego))
                if oc:
                    onm = {0: "free", 2: "veh", 4: "ped", 5: "road",
                           7: "veg", 8: "bldg", 9: "pole"}
                    print(f"[valOCC ep{ep}] mIoU={np.mean(list(oc.values())):.3f} "
                          + " ".join(f"{onm[c]}={oc[c]:.3f}"
                                     for c in onm if c in oc), flush=True)
            if use_ego:
                eg = evaluate_ego(net, dv, device, 4 + int(use_seg2d))
                if eg:
                    print(f"[valE2E ep{ep}] ADE={eg['ade']:.2f}m "
                          f"ADEc={eg['ade_c']:.2f}m "
                          f"FDE={eg['fde']:.2f}m steer={eg['steer']:.3f}rad "
                          f"acc={eg['acc']:.2f}m/s2 brakeAcc={eg['brake']:.2f}",
                          flush=True)
            torch.save({"model": net.state_dict(), "epoch": ep, "ious": ious},
                       os.path.join(args.out, "last.pt"))
            if miou > best:
                best = miou
                torch.save({"model": net.state_dict(), "epoch": ep, "ious": ious},
                           os.path.join(args.out, "best.pt"))
        if ddp:
            dist.barrier()
    if is_main:
        print(f"[done] best mIoU={best:.3f}", flush=True)
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

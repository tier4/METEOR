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
import math
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler

# avoid "resize storage that is not resizable" collate failures under many
# DDP workers (file-descriptor sharing exhausts FDs with extra sample tensors)
torch.multiprocessing.set_sharing_strategy("file_system")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bevlane.dataset import BevLaneDataset  # noqa: E402
from bevlane.model import MODELS, N_CLASSES, make_warp_theta  # noqa: E402

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


class EpochSubsetSampler(torch.utils.data.Sampler):
    """Draw a FRESH random subset of `n` samples every epoch, sharded across
    DDP ranks.

    It replaces `Subset(RandomState(0).permutation(len)[:n])`, which pinned
    every epoch of every round to the same 46k of 340k frames -- 86% of the
    extracted corpus was never trained on. All ranks share the same seed so
    they draw the same subset, then take disjoint slices of it.
    """
    def __init__(self, n_total, n_draw, rank=0, world=1, seed=0,
                 weights=None):
        self.n_total, self.n_draw = n_total, min(n_draw, n_total)
        self.rank, self.world, self.seed = rank, world, seed
        self.weights = weights            # optional per-sample draw weights
        self.epoch = 0

    def set_epoch(self, ep):
        self.epoch = ep

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + 9973 * self.epoch)
        if self.weights is not None:
            idx = torch.multinomial(self.weights, self.n_draw,
                                    replacement=False, generator=g)
        else:
            idx = torch.randperm(self.n_total, generator=g)[:self.n_draw]
        per = self.n_draw // self.world
        return iter(idx[self.rank * per:(self.rank + 1) * per].tolist())

    def __len__(self):
        return self.n_draw // self.world


def split_scenes(root):
    scenes = sorted(os.listdir(root))
    # indoor / GNSS-dead scenes (annotate_indoor.py): ego pose is a smooth
    # fiction there -> E2E / temporal / trajectory GT silently corrupted
    bad = set()
    p = os.path.join(os.path.dirname(root), "indoor_scenes.txt")
    if os.path.exists(p):
        bad = set(open(p).read().split())
    scenes = [s for s in scenes if s not in bad]
    val = [s for s in scenes if "2026-01-23T15-26-01" in s]
    train = [s for s in scenes if s not in set(val)]
    return train, val


@torch.no_grad()
def evaluate(model, loader, device, max_batches=80, use_lidar=False):
    """use_lidar: loader must carry depth4 at batch[4]; measures the v31
    LiDAR-assisted mode (camera-only is the plain call)."""
    model.eval()
    inter = np.zeros(N_CLASSES)
    union = np.zeros(N_CLASSES)
    for bi, batch in enumerate(loader):
        if bi >= max_batches:
            break
        imgs, K, Tc, gt = (t.to(device, non_blocking=True) for t in batch[:4])
        with torch.autocast("cuda", torch.float16):
            if use_lidar:
                kw = {"lidar": batch[4].to(device)}
                if len(batch) > 5:      # v32 loader also carries the raster
                    kw["lidar_bev"] = batch[5].to(device)
                logits = model(imgs, K, Tc, **kw)
            else:
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

def bev_rotation_aug(theta_max_deg, Tc, gt, det_boxes, det_n, traj_gt,
                     ego_gt, occ_gt, risk_gt, lg_pts, unk_c, rel_pose,
                     p_apply=0.5):
    """BEV-space rotation augmentation: rotate the EGO FRAME, not pixels.

    T_cam_ego absorbs the rotation, so the projected BEV features are
    rebuilt exactly by the geometry (no feature interpolation, image-space
    heads untouched); every BEV-space GT is rotated by the same angle.
    Physically = the same scene recorded with the rig yawed by theta."""
    B = Tc.shape[0]
    dev = Tc.device
    th = (torch.rand(B, device=dev) * 2 - 1) * math.radians(theta_max_deg)
    th = th * (torch.rand(B, device=dev) < p_apply).float()
    c, s = th.cos(), th.sin()
    # 1. extrinsics: p_old = Rz(th) p_new -> Tc' = Tc @ Rz(th)
    R = torch.zeros(B, 4, 4, device=dev, dtype=Tc.dtype)
    R[:, 0, 0] = c; R[:, 0, 1] = -s
    R[:, 1, 0] = s; R[:, 1, 1] = c
    R[:, 2, 2] = 1; R[:, 3, 3] = 1
    Tc = Tc @ R[:, None]
    # 2D rotation for points expressed in NEW frame: p_new = R(-th) p_old
    def rot_pts(xy):                     # [...,2] (x,y)
        shp = [B] + [1] * (xy.dim() - 2)
        cc, ss = c.view(shp), s.view(shp)
        x, y = xy[..., 0], xy[..., 1]
        return torch.stack([cc * x + ss * y, -ss * x + cc * y], -1)
    # 2. label rasters via inverse-rotated sampling grid
    def rot_raster(r, fill, nearest=True):
        if r is None:
            return None
        r4 = r.float().unsqueeze(1) if r.dim() == 3 else r.float()
        A = torch.zeros(B, 2, 3, device=dev, dtype=torch.float32)
        A[:, 0, 0] = c; A[:, 0, 1] = -s * (r4.shape[2] / r4.shape[3])
        A[:, 1, 0] = s * (r4.shape[3] / r4.shape[2]); A[:, 1, 1] = c
        g = F.affine_grid(A, list(r4.shape), align_corners=False)
        out = F.grid_sample(r4 + 1.0, g, mode="nearest" if nearest
                            else "bilinear", padding_mode="zeros",
                            align_corners=False)
        res = torch.where(out < 0.5, torch.full_like(out, fill + 1.0),
                          out) - 1.0
        return res.squeeze(1).to(r.dtype) if r.dim() == 3 else res.to(r.dtype)
    gt = rot_raster(gt, 0)
    risk_gt = rot_raster(risk_gt, 0, nearest=False)         if risk_gt is not None else None
    if occ_gt is not None:
        occ_gt = rot_raster(occ_gt, 255)
    # 3. boxes / futures / ego path / graph points / unknown centres
    if det_boxes is not None:
        det_boxes = det_boxes.clone()
        det_boxes[..., 1:3] = rot_pts(det_boxes[..., 1:3])
        det_boxes[..., 5] = det_boxes[..., 5] - th[:, None]
        if traj_gt is not None:
            traj_gt = rot_pts(traj_gt)
    if ego_gt is not None:
        ego_gt = ego_gt.clone()
        ego_gt[:, :12] = rot_pts(ego_gt[:, :12].view(B, 6, 2)).reshape(B, 12)
    if lg_pts is not None:
        lg_pts = rot_pts(lg_pts)
    if unk_c is not None:
        unk_c = rot_pts(unk_c)
    if rel_pose is not None:
        rel_pose = rel_pose.clone()
        rel_pose[..., :2] = rot_pts(rel_pose[..., :2])
    return (Tc, gt, det_boxes, traj_gt, ego_gt, occ_gt, risk_gt, lg_pts,
            unk_c, rel_pose)


def _temporal_inputs(model, batch, device, tmp_idx):
    if tmp_idx is None:
        return None, None
    pi = batch[tmp_idx].to(device, non_blocking=True)
    rel = batch[tmp_idx + 1].to(device)
    pv = batch[tmp_idx + 2].to(device)
    K = batch[1].to(device)
    Tc = batch[2].to(device)
    if hasattr(model, "tfuse3"):               # v29 memory queue
        pbs, ths = [], []
        with torch.no_grad():
            for hi in range(pi.shape[1]):
                pbs.append(model.compute_bev(pi[:, hi], K, Tc)
                           * pv[:, hi].view(-1, 1, 1, 1))
                ths.append(make_warp_theta(rel[:, hi]))
        return torch.stack(pbs, 1), torch.stack(ths, 1)
    with torch.no_grad():
        pb = model.compute_bev(pi, K, Tc) * pv.view(-1, 1, 1, 1)
    return pb, make_warp_theta(rel)


def evaluate_ego(model, loader, device, ego_idx, max_batches=40,
                 batch_stride=1, shard=(0, 1), raw=False,
                 tmp_idx=None):
    """E2E head metrics: trajectory ADE/FDE [m], steer MAE [rad],
    accel MAE [m/s^2], brake accuracy. Valid frames only."""
    model.eval()
    n = ade = fde = smae = amae = bacc = 0.0
    nc = adec = 0.0
    done = 0
    for bi, batch in enumerate(loader):
        if bi % batch_stride:
            continue
        done += 1
        if (done - 1) % shard[1] != shard[0]:
            continue
        if done > max_batches * shard[1]:
            break
        imgs, K, Tc = (t.to(device, non_blocking=True) for t in batch[:3])
        eg = batch[ego_idx].to(device, non_blocking=True)
        pb, th = _temporal_inputs(model, batch, device, tmp_idx)
        with torch.autocast("cuda", torch.float16):
            out = model(imgs, K, Tc, eg[:, 12], pb, th) if th is not None \
                else model(imgs, K, Tc, eg[:, 12])
        if not (isinstance(out, tuple) and len(out) >= 8):
            break
        p = out[7].float()
        if p.shape[1] > 15:                    # v29 multimodal (K=3)
            Kn = 3
            wps = p[:, :12 * Kn].view(-1, Kn, 6, 2)
            mode = p[:, 12 * Kn:12 * Kn + Kn].argmax(1)
            wp1 = wps[torch.arange(len(p)), mode].reshape(-1, 12)
            p = torch.cat([wp1, p[:, 12 * Kn + Kn:]], 1)
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
    if raw:
        return torch.tensor([ade, fde, smae, amae, bacc, n,
                             adec, nc], dtype=torch.float64, device=device)
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
@torch.no_grad()
def evaluate_tl(model, loader, device, tl_idx, max_batches=40, tmp_idx=None):
    """Ego-relevant traffic-light state: overall accuracy + per-class recall."""
    model.eval()
    hit = [0, 0, 0, 0]
    tot = [0, 0, 0, 0]
    for bi, batch in enumerate(loader):
        if bi >= max_batches:
            break
        imgs, K, Tc = (t.to(device, non_blocking=True) for t in batch[:3])
        tg = batch[tl_idx]
        pb, th = _temporal_inputs(model, batch, device, tmp_idx)
        with torch.autocast("cuda", torch.float16):
            out = model(imgs, K, Tc, None, pb, th) if th is not None                 else model(imgs, K, Tc)
        if not (isinstance(out, tuple) and len(out) >= 12):
            break
        pred = out[11].float().argmax(1).cpu()
        for p, g in zip(pred.tolist(), tg.tolist()):
            if g == 255:
                continue
            tot[g] += 1
            hit[g] += int(p == g)
    model.train()
    if sum(tot) == 0:
        return None
    r = {"acc": sum(hit) / sum(tot)}
    for c, nm in enumerate(("none", "green", "yellow", "red")):
        if tot[c]:
            r[nm] = hit[c] / tot[c]
    return r


@torch.no_grad()
def evaluate_risk(model, loader, device, rk_idx, max_batches=30, tmp_idx=None):
    """Risk-map regression quality: overall L1 and L1 on high-risk cells."""
    model.eval()
    n = l1 = l1h = nh = 0.0
    for bi, batch in enumerate(loader):
        if bi >= max_batches:
            break
        imgs, K, Tc = (t.to(device, non_blocking=True) for t in batch[:3])
        gt = batch[rk_idx].to(device)
        pb, th = _temporal_inputs(model, batch, device, tmp_idx)
        with torch.autocast("cuda", torch.float16):
            out = model(imgs, K, Tc, None, pb, th) if th is not None                 else model(imgs, K, Tc)
        if not (isinstance(out, tuple) and len(out) >= 13):
            break
        p = out[12][:, 0].float().sigmoid()
        m = gt >= 0
        if m.sum() == 0:
            continue
        d = (p - gt.clamp(min=0)).abs()
        l1 += float(d[m].sum()); n += float(m.sum())
        hi = m & (gt > 0.4)
        if hi.any():
            l1h += float(d[hi].sum()); nh += float(hi.sum())
    model.train()
    if n == 0:
        return None
    return {"l1": l1 / n, "l1_hi": l1h / max(nh, 1)}


@torch.no_grad()
def evaluate_lanegraph(model, loader, device, lg_idx, max_batches=20,
                       tmp_idx=None):
    """chain P/R at mean-chamfer < 0.5 m + adjacency accuracy."""
    tp = fp = fn = 0
    a_hit = a_tot = 0
    model.eval()
    for bi, batch in enumerate(loader):
        if bi >= max_batches:
            break
        imgs, K, Tc = (t.to(device, non_blocking=True) for t in batch[:3])
        gp = batch[lg_idx].to(device)
        gn = batch[lg_idx + 2]
        ga = batch[lg_idx + 3].to(device)
        pb, th = _temporal_inputs(model, batch, device, tmp_idx)
        with torch.autocast("cuda", torch.float16):
            out = model(imgs, K, Tc, None, pb, th)
        if len(out) < 17:
            break
        pts, meta, adj = out[14].float(), out[15].float(), out[16].float()
        for b in range(gp.shape[0]):
            keep = meta[b, :, 0].sigmoid() > 0.5
            pk = pts[b][keep]
            n = int(gn[b])
            g = gp[b, :n].float()
            used = torch.zeros(n, dtype=torch.bool)
            pi_match = {}
            for i in range(pk.shape[0]):
                if n == 0:
                    fp += 1
                    continue
                d1 = (pk[i:i + 1] - g).abs().mean((1, 2))
                d2 = (pk[i:i + 1] - g.flip(1)).abs().mean((1, 2))
                d = torch.minimum(d1, d2)
                j = int(d.argmin())
                if d[j] < 0.5 and not used[j]:
                    used[j] = True
                    tp += 1
                    pi_match[i] = j
                else:
                    fp += 1
            fn += int(n - used.sum())
            kidx = keep.nonzero()[:, 0]
            for i, j in pi_match.items():
                for i2, j2 in pi_match.items():
                    if i2 <= i:
                        continue
                    a_p = adj[b, kidx[i], kidx[i2]] > 0
                    a_g = ga[b, j, j2] > 0.5
                    a_hit += int(a_p == a_g)
                    a_tot += 1
    model.train()
    if tp + fn == 0:
        return None
    return {"p": tp / max(tp + fp, 1), "r": tp / max(tp + fn, 1),
            "adj": a_hit / max(a_tot, 1)}


@torch.no_grad()
def evaluate_flow(model, loader, device, bx_idx, max_batches=20,
                  tmp_idx=None):
    """mean endpoint error [m/s] on agent-box cells, moving/stationary."""
    import cv2 as _cv
    em = es = nm = ns = 0.0
    model.eval()
    for bi, batch in enumerate(loader):
        if bi >= max_batches:
            break
        imgs, K, Tc = (t.to(device, non_blocking=True) for t in batch[:3])
        bx, nb = batch[bx_idx], batch[bx_idx + 1]
        tj, tv = batch[bx_idx + 2], batch[bx_idx + 3]
        pb, th = _temporal_inputs(model, batch, device, tmp_idx)
        with torch.autocast("cuda", torch.float16):
            out = model(imgs, K, Tc, None, pb, th)
        if len(out) < 14:
            break
        fl = out[13].float().cpu().numpy()
        for b in range(bx.shape[0]):
            for k in range(int(nb[b])):
                cls, xe, ye, l, w, yaw = bx[b, k].tolist()
                if l <= 0 or abs(xe) > 38 or abs(ye) > 38:
                    continue
                ri = int((40 - xe) / 0.4)
                ci = int((40 - ye) / 0.4)
                vx = vy = 0.0
                if tv[b, k, 0] > 0.5:
                    vx, vy = (float(tj[b, k, 0, 0]) / 0.5,
                              float(tj[b, k, 0, 1]) / 0.5)
                pv_ = fl[b, :, ri, ci]
                e = ((pv_[0] - vx) ** 2 + (pv_[1] - vy) ** 2) ** 0.5
                if (vx * vx + vy * vy) ** 0.5 > 0.5:
                    em += e; nm += 1
                else:
                    es += e; ns += 1
    model.train()
    if nm + ns == 0:
        return None
    return {"epe_mov": em / max(nm, 1), "epe_stat": es / max(ns, 1)}


@torch.no_grad()
def evaluate_unknown(model, loader, device, uk_idx, max_batches=20,
                     tmp_idx=None):
    """unknown-object P/R at 1 m centre match (fixed-size class)."""
    tp = fp = fn = 0
    model.eval()
    for bi, batch in enumerate(loader):
        if bi >= max_batches:
            break
        imgs, K, Tc = (t.to(device, non_blocking=True) for t in batch[:3])
        uc, un = batch[uk_idx], batch[uk_idx + 1]
        pb, th = _temporal_inputs(model, batch, device, tmp_idx)
        with torch.autocast("cuda", torch.float16):
            out = model(imgs, K, Tc, None, pb, th)
        if len(out) < 18:
            break
        dets = model.decode_unknown(out[17].float())
        for b in range(uc.shape[0]):
            gt = [(float(uc[b, k, 0]), float(uc[b, k, 1]))
                  for k in range(int(un[b]) & 0xFF)]   # low byte = positives
            used = [False] * len(gt)
            for _, sc, xe, ye, *_ in sorted(dets[b], key=lambda d: -d[1]):
                best, bd = -1, 1.0
                for gi, (gx, gy) in enumerate(gt):
                    if used[gi]:
                        continue
                    d = ((gx - xe) ** 2 + (gy - ye) ** 2) ** 0.5
                    if d < bd:
                        best, bd = gi, d
                if best >= 0:
                    used[best] = True
                    tp += 1
                else:
                    fp += 1
            fn += used.count(False)
    model.train()
    if tp + fn == 0:
        return None
    return {"p": tp / max(tp + fp, 1), "r": tp / max(tp + fn, 1)}


def evaluate_traj(model, loader, device, tj_idx, max_batches=40,
                  tmp_idx=None):
    _STAT = [0, 0]
    _pc = [0.0, 0.0]; _pn = [0, 0]        # per-class ADE (veh, vru)
    _hd = [0.0, 0.0]; _hn = [0, 0]        # per-class heading error @3s
    """Agent-forecast ADE/FDE [m] sampled at GT box centres (valid wps)."""
    model.eval()
    n = ade = fde = nf = 0.0
    for bi, batch in enumerate(loader):
        if bi >= max_batches:
            break
        imgs, K, Tc = (t.to(device, non_blocking=True) for t in batch[:3])
        bx = batch[tj_idx].to(device)
        nb = batch[tj_idx + 1]
        tj = batch[tj_idx + 2].to(device)
        tv = batch[tj_idx + 3].to(device)
        pb, th = _temporal_inputs(model, batch, device, tmp_idx)
        with torch.autocast("cuda", torch.float16):
            out = model(imgs, K, Tc, None, pb, th) if th is not None \
                else model(imgs, K, Tc)
        if not (isinstance(out, tuple) and len(out) >= 10):
            break
        tp = out[9].float()
        stat = out[10].float() if len(out) >= 11 else None
        for b in range(bx.shape[0]):
            for k in range(int(nb[b])):
                ri = int((80.0 - float(bx[b, k, 1])) / 0.4)
                ci = int((50.0 - float(bx[b, k, 2])) / 0.4)
                if not (0 <= ri < tp.shape[-2] and 0 <= ci < tp.shape[-1]):
                    continue
                vec = tp[b, :, ri, ci]
                if vec.numel() >= 39:          # v29 multimodal per cell
                    kbest = int(vec[36:39].argmax())
                    p = vec[kbest * 12:(kbest + 1) * 12].view(6, 2)
                else:
                    p = vec.view(6, 2)
                d = (p - tj[b, k]).norm(dim=1)
                v = tv[b, k]
                if v.sum() == 0:
                    continue
                ade += float((d * v).sum() / v.sum()); n += 1
                _c = 0 if float(bx[b, k, 0]) < 1.5 else 1
                _pc[_c] += float((d * v).sum() / v.sum()); _pn[_c] += 1
                if v[5] > 0:
                    fde += float(d[5]); nf += 1
                    # heading error at 3 s (only when both actually move)
                    _g = tj[b, k, 5]; _p = p[5]
                    if float(_g.norm()) > 1.0 and float(_p.norm()) > 0.3:
                        import math as _m
                        _ga = _m.atan2(float(_g[1]), float(_g[0]))
                        _pa = _m.atan2(float(_p[1]), float(_p[0]))
                        _d = abs((_pa - _ga + _m.pi) % (2 * _m.pi) - _m.pi)
                        _hd[_c] += _m.degrees(_d); _hn[_c] += 1
                    if stat is not None:
                        pred_s = float(stat[b, 0, ri, ci]) > 0
                        gt_s = float(tj[b, k, 5].norm()) < 0.5
                        _STAT[0] += int(pred_s == gt_s); _STAT[1] += 1
    model.train()
    if n == 0:
        return None
    r = {"ade": ade / n, "fde": fde / max(nf, 1)}
    if _STAT[1]:
        r["stat_acc"] = _STAT[0] / _STAT[1]
    for c, nm in ((0, "veh"), (1, "vru")):
        if _pn[c]:
            r[nm + "_ade"] = _pc[c] / _pn[c]
        if _hn[c]:
            r[nm + "_head"] = _hd[c] / _hn[c]
    return r


@torch.no_grad()
def evaluate_det3d(model, loader, device, bx_idx, max_batches=30,
                   tmp_idx=None, thresh=0.3, match_m=2.0):
    """BEV 3D detection: per-class precision/recall (centre match <2 m) and
    mean centre error on matched pairs."""
    model.eval()
    tp = [0, 0]
    fp = [0, 0]
    fn = [0, 0]
    cerr = [0.0, 0.0]
    yerr = [0.0, 0.0]
    yflip = [0, 0]
    tp50 = [0, 0]
    fn50 = [0, 0]
    tpn = [0, 0]
    fnn = [0, 0]
    for bi, batch in enumerate(loader):
        if bi >= max_batches:
            break
        imgs, K, Tc = (t.to(device, non_blocking=True) for t in batch[:3])
        bx = batch[bx_idx]
        nb = batch[bx_idx + 1]
        pb, th = _temporal_inputs(model, batch, device, tmp_idx)
        with torch.autocast("cuda", torch.float16):
            out = model(imgs, K, Tc, None, pb, th) if th is not None \
                else model(imgs, K, Tc)
        dets = model.decode_boxes(out[3].float(), out[4].float(),
                                  thresh=thresh, topk=64)
        for b in range(bx.shape[0]):
            gt = [(0 if bx[b, k, 0] < 1.5 else 1,
                   float(bx[b, k, 1]), float(bx[b, k, 2]),
                   float(bx[b, k, 5]))
                  for k in range(int(nb[b])) if bx[b, k, 3] > 0]
            used = [False] * len(gt)
            for cls, sc, xe, ye, l, w, yaw in sorted(dets[b],
                                                     key=lambda d: -d[1]):
                best, bd = -1, match_m
                for gi, (gc, gx, gy, gyaw) in enumerate(gt):
                    if used[gi] or gc != cls:
                        continue
                    d = ((gx - xe) ** 2 + (gy - ye) ** 2) ** 0.5
                    if d < bd:
                        best, bd = gi, d
                if best >= 0:
                    used[best] = True
                    tp[cls] += 1
                    cerr[cls] += bd
                    dy = abs((yaw - gt[best][3] + np.pi) % (2 * np.pi) - np.pi)
                    yflip[cls] += dy > np.pi / 2
                    yerr[cls] += min(dy, np.pi - dy)   # axis error
                    gx, gy = gt[best][1], gt[best][2]
                    if gx * gx + gy * gy < 50.0 ** 2:
                        tp50[cls] += 1
                    if gx * gx + gy * gy < 30.0 ** 2 and abs(gy) < 12.0:
                        tpn[cls] += 1
                else:
                    fp[cls] += 1
            for gi, (gc, gx, gy, _) in enumerate(gt):
                if not used[gi]:
                    fn[gc] += 1
                    if gx * gx + gy * gy < 50.0 ** 2:
                        fn50[gc] += 1
                    if gx * gx + gy * gy < 30.0 ** 2 and abs(gy) < 12.0:
                        fnn[gc] += 1
    model.train()
    r = {}
    for c, nm in ((0, "veh"), (1, "vru")):
        p_ = tp[c] / max(tp[c] + fp[c], 1)
        rc = tp[c] / max(tp[c] + fn[c], 1)
        r[nm] = (p_, rc, cerr[c] / max(tp[c], 1))
        r[nm + "_yaw"] = np.degrees(yerr[c] / max(tp[c], 1))
        r[nm + "_flip"] = yflip[c] / max(tp[c], 1)
        r[nm + "50"] = tp50[c] / max(tp50[c] + fn50[c], 1)
        r[nm + "n"] = tpn[c] / max(tpn[c] + fnn[c], 1)
    return r


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
    ap.add_argument("--model", default="v1", choices=["v1", "v2", "v3s", "lss", "v8", "v13", "v13d", "v14d", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36"])
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
    ap.add_argument("--traj-w", type=float, default=0.0,
                    help="agent trajectory forecast loss weight (v21)")
    ap.add_argument("--tl-w", type=float, default=0.0)
    ap.add_argument("--risk-w", type=float, default=0.0)
    ap.add_argument("--lanegraph-w", type=float, default=0.0)
    ap.add_argument("--flow-w", type=float, default=0.0)
    ap.add_argument("--val-every", type=int, default=0,
                    help="probe BEV mIoU + 3D det every N steps (0=off)")
    ap.add_argument("--seed-subset", type=int, default=0,
                    help="base seed for the per-epoch training subset")
    ap.add_argument("--turn-oversample", type=float, default=1.0,
                    help="draw weight for turn frames (|lat@3s|>4m)")
    ap.add_argument("--unk-w", type=float, default=0.0)
    ap.add_argument("--bev-rot-aug", type=float, default=0.0,
                    help="BEV-frame rotation augmentation: max |yaw| in "
                    "degrees rotated into the extrinsics + all BEV GT "
                    "(images and camera-space heads untouched)")
    ap.add_argument("--mined-oversample", type=float, default=1.0,
                    help="C2: weight boost for scenes in out/mined_scenes.txt")
    ap.add_argument("--lidar-drop", type=float, default=0.5,
                    help="v31: per-sample probability of hiding the LiDAR "
                    "input during training (modality dropout keeps one set "
                    "of weights valid for camera-only AND LiDAR inference)")
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
    use_depth = args.model in ("lss", "v8", "v13", "v13d", "v14d", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36") and args.depth_w > 0
    use_seg2d = args.model in ("v13", "v13d", "v14d", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36") and args.seg2d_w > 0
    use_box = args.model == "v15" and args.box_w > 0
    use_boxdet = args.model in ("v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36") and args.box_w > 0
    use_bbox2d = args.model in ("v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36") and args.bbox2d_w > 0
    use_ego = args.model in ("v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36") and args.ego_w > 0
    use_occ = args.model in ("v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36") and args.occ_w > 0
    use_traj = args.model in ("v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36") and args.traj_w > 0
    use_temporal = args.model in ("v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36")
    use_tl = args.model in ("v27", "v28", "v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36") and args.tl_w > 0
    use_risk = args.model in ("v28", "v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36") and args.risk_w > 0
    use_lg = args.model in ("v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36") and args.lanegraph_w > 0
    use_unk = args.model in ("v30", "v31", "v32", "v33", "v34", "v35", "v36") and args.unk_w > 0
    # v31 reuses the depth4 GT tensor as the (train-time) LiDAR input
    use_lidar = args.model in ("v31", "v32", "v33", "v34", "v35", "v36")
    # v32 additionally takes the pillar BEV raster (extract_lidar_bev.py)
    use_lidarbev = args.model in ("v32", "v33", "v34", "v35", "v36")
    use_flow = args.model in ("v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36") and args.flow_w > 0
    hist_n = 3 if args.model in ("v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36") else 0
    if args.train_list:                       # restrict train to a scene list
        keep = set(open(args.train_list).read().split())
        train_s = [s for s in train_s if s in keep]
    # v13d depth GT is stride-4 of 768 (108x192); resize any mixed-res depth
    depth_hw = (108, 192) if args.model in ("v13d", "v14d", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36") else None
    tr = BevLaneDataset(args.root, train_s, gt_key=args.gt_key,
                        dontcare_sidewalk=args.dontcare_sidewalk,
                        with_depth=use_depth, augment=args.aug,
                        with_seg2d=use_seg2d, depth_hw=depth_hw,
                        with_box=use_box,
                        with_boxdet=use_boxdet and not use_traj,
                        with_agenttraj=use_traj,
                        with_bbox2d=use_bbox2d, with_ego=use_ego,
                        with_occ=use_occ, with_temporal=use_temporal,
                        with_tl=use_tl, with_risk=use_risk,
                        with_lanegraph=use_lg, temporal_hist=hist_n,
                        with_unknown=use_unk, with_lidarbev=use_lidarbev,
                        trim_start=3, trim_end=args.trim_end,
                        min_cov_core=args.min_cov_core,
                        min_cov_fwd=args.min_cov_fwd,
                        seg2d_key=args.seg2d_key)
    # NOTE: --limit-train no longer slices the dataset here; it is applied
    # per epoch by EpochSubsetSampler so each epoch sees fresh frames.
    if True:   # all ranks: the distributed ADE probe shards the val set
        # val never needs depth GT (BEV mIoU eval only) -> with_depth=False
        va = BevLaneDataset(args.root, val_s, max_per_scene=8, gt_key=args.gt_key,
                            dontcare_sidewalk=args.dontcare_sidewalk,
                            with_depth=False, with_seg2d=use_seg2d,
                            seg2d_key=args.seg2d_key, with_ego=use_ego,
                            with_occ=use_occ, with_agenttraj=use_traj,
                            with_temporal=use_temporal, with_tl=use_tl,
                            with_risk=use_risk, with_lanegraph=use_lg,
                            temporal_hist=hist_n, with_unknown=use_unk,
                            with_lidarbev=use_lidarbev,
                            trim_start=3, trim_end=args.trim_end)
        seen = min(args.limit_train or len(tr), len(tr)) * args.epochs
        print(f"train {len(tr)} samples / {len(train_s)} scenes; "
              f"val {len(va)} samples / {len(val_s)} scenes; "
              f"world={world} lr={lr:.1e}", flush=True)
        print(f"[data] {args.limit_train or len(tr)} fresh samples/epoch x "
              f"{args.epochs} ep = {seen} draws over {len(tr)} frames "
              f"({100 * min(seen, len(tr)) / max(len(tr), 1):.0f}% expected "
              f"coverage)", flush=True)
        dv = DataLoader(va, batch_size=args.batch, shuffle=False,
                        num_workers=4 if is_main else 1, pin_memory=is_main)
        dv_lid = None
        if use_lidar:
            # separate minimal loader (imgs,K,T,gt,depth4) so the main val
            # batch layout (indexed positionally everywhere) is untouched
            va_lid = BevLaneDataset(args.root, val_s, max_per_scene=8,
                                    gt_key=args.gt_key, with_depth=True,
                                    with_lidarbev=use_lidarbev,
                                    dontcare_sidewalk=args.dontcare_sidewalk,
                                    trim_start=3, trim_end=args.trim_end)
            dv_lid = DataLoader(va_lid, batch_size=args.batch, shuffle=False,
                                num_workers=2, pin_memory=True)

    if args.limit_train and args.limit_train < len(tr):
        weights = None
        if args.turn_oversample > 1.0:
            # WTA multimodality only differentiates on samples where the
            # future turns; those are 7.5% of moving frames (|lat@3s|>4 m),
            # so straight frames win the modes 12:1. Upweight turn frames.
            import numpy as _np
            weights = torch.ones(len(tr))
            cache = {}
            n_turn = 0
            for ii, (s_, f_) in enumerate(tr.items):
                if s_ not in cache:
                    try:
                        z_ = _np.load(os.path.join(args.root, s_,
                                                   "ego_motion.npz"))
                        cache[s_] = (z_["wp"], z_["v0"], z_["valid"])
                    except Exception:
                        cache[s_] = None
                z_ = cache[s_]
                if z_ is None:
                    continue
                fi_ = f_["frame"]
                if fi_ < len(z_[1]) and z_[2][fi_] > 0 and z_[1][fi_] > 2.0 \
                        and abs(z_[0][fi_, 5, 1]) > 4.0:
                    weights[ii] = args.turn_oversample
                    n_turn += 1
            if is_main:
                print(f"[data] turn oversample x{args.turn_oversample}: "
                      f"{n_turn}/{len(tr)} frames boosted", flush=True)
        if args.mined_oversample > 1.0 and os.path.exists("out/mined_scenes.txt"):
            mined = set(open("out/mined_scenes.txt").read().split())
            if weights is None:
                weights = torch.ones(len(tr))
            n_m = 0
            for ii, (s_, _f) in enumerate(tr.items):
                if s_ in mined:
                    weights[ii] = max(float(weights[ii]),
                                      args.mined_oversample)
                    n_m += 1
            if is_main:
                print(f"[data] failure-mined oversample x"
                      f"{args.mined_oversample}: {n_m}/{len(tr)} frames "
                      f"({len(mined)} scenes)", flush=True)
        sampler = EpochSubsetSampler(len(tr), args.limit_train,
                                     rank=rank if ddp else 0,
                                     world=world, seed=args.seed_subset,
                                     weights=weights)
    else:
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
        if args.model in ("v13", "v13d", "v14d", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36") else {}
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
                                    (args.model in ("v13", "v13d", "v14d", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36") and not use_seg2d)))
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
            if use_traj:
                det_boxes, det_n = batch[bi], batch[bi + 1]
                traj_gt, tvalid_gt = batch[bi + 2], batch[bi + 3]
                bi += 4
            else:
                det_boxes = batch[bi] if use_boxdet else None
                det_n = batch[bi + 1] if use_boxdet else None
                bi += 2 if use_boxdet else 0
                traj_gt = tvalid_gt = None
            bb2d = batch[bi] if use_bbox2d else None
            nb2d = batch[bi + 1] if use_bbox2d else None
            bi += 2 if use_bbox2d else 0
            ego_gt = batch[bi] if use_ego else None
            bi += 1 if use_ego else 0
            occ_gt = batch[bi] if use_occ else None
            bi += 1 if use_occ else 0
            tl_gt = batch[bi] if use_tl else None
            bi += 1 if use_tl else 0
            risk_gt = batch[bi] if use_risk else None
            bi += 1 if use_risk else 0
            if use_lg:
                lg_pts_gt, lg_cls_gt = batch[bi], batch[bi + 1]
                lg_n_gt, lg_adj_gt = batch[bi + 2], batch[bi + 3]
                bi += 4
            else:
                lg_pts_gt = lg_cls_gt = lg_n_gt = lg_adj_gt = None
            unk_c = batch[bi] if use_unk else None
            unk_n = batch[bi + 1] if use_unk else None
            bi += 2 if use_unk else 0
            lidbev = batch[bi] if use_lidarbev else None
            bi += 1 if use_lidarbev else 0
            if use_temporal:
                prev_imgs, rel_pose, prev_valid = (batch[bi], batch[bi + 1],
                                                   batch[bi + 2])
            else:
                prev_imgs = rel_pose = prev_valid = None
            if args.bev_rot_aug > 0:
                (Tc, gt, det_boxes, traj_gt, ego_gt, occ_gt, risk_gt,
                 lg_pts_gt, unk_c, rel_pose) = bev_rotation_aug(
                    args.bev_rot_aug, Tc, gt, det_boxes, det_n, traj_gt,
                    ego_gt, occ_gt, risk_gt, lg_pts_gt, unk_c, rel_pose)
            if use_temporal and hist_n > 0:
                # v29 memory queue: N history BEVs, each in its own no_grad
                # + autocast region (r12 autocast-cache lesson).
                # eval() around the history pass: these are auxiliary feature
                # extractions and must NOT drive BatchNorm running stats.
                # With 3 slots they outnumber the current frame 3:1, and
                # ~16% of them are all-zero images (missing history at scene
                # starts); that mixture corrupts the running stats within
                # ~50 steps and collapses every eval-mode metric while the
                # training loss still looks healthy.
                net00 = model.module if ddp else model
                model.eval()
                pbs, ths = [], []
                with torch.no_grad(), torch.autocast("cuda", torch.float16):
                    for hi in range(hist_n):
                        pbs.append(net00.compute_bev(prev_imgs[:, hi], K, Tc)
                                   * prev_valid[:, hi].view(-1, 1, 1, 1))
                        ths.append(make_warp_theta(rel_pose[:, hi]))
                model.train()
                pb = torch.stack(pbs, 1).float()
                theta = torch.stack(ths, 1)
            elif use_temporal:
                # prev-frame BEV in its OWN autocast region: computing it
                # inside the main region caches detached fp16 weight casts
                # and silently cuts gradients to the whole backbone
                net00 = model.module if ddp else model
                model.eval()                 # same BN-stat rule as above
                with torch.no_grad(), torch.autocast("cuda", torch.float16):
                    pb = net00.compute_bev(prev_imgs, K, Tc) \
                        * prev_valid.view(-1, 1, 1, 1)
                model.train()
                pb = pb.float()
                theta = make_warp_theta(rel_pose)
            lid = None
            if use_lidar and depth_gt is not None:
                # modality dropout: whole-sample, so BN sees both modes
                keep = (torch.rand(imgs.shape[0], 1, 1, 1, device=device)
                        >= args.lidar_drop).to(depth_gt.dtype)
                lid = depth_gt * keep
                if use_lidarbev and lidbev is not None:
                    lidbev = lidbev * keep
            with torch.autocast("cuda", torch.float16):
                # v18+ is conditioned on the current speed (ego_gt col 12)
                if use_temporal:
                    out = model(imgs, K, Tc,
                                ego_gt[:, 12] if use_ego else None, pb, theta,
                                **({"lidar": lid} if use_lidar else {}),
                                **({"lidar_bev": lidbev}
                                   if use_lidarbev else {}),
                                **({"kin": rel_pose}
                                   if args.model == "v36" else {}))
                elif use_ego:
                    out = model(imgs, K, Tc, ego_gt[:, 12])
                else:
                    out = model(imgs, K, Tc)
                net0 = model.module if ddp else model
                # robustly unpack: v15 -> 4-tuple, v13* -> 3, lss/v8 -> 2
                logits, dlog, seg2d, boxl, hm, rg = out, None, None, None, None, None
                hm2d, rg2d, ego_pred, occ_pred = None, None, None, None
                traj_pred = None
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
                        occ_pred = out[8] if len(out) >= 9 else None
                        traj_pred = out[9] if len(out) >= 10 else None
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
                if traj_pred is not None and use_traj:
                    loss = loss + args.traj_w * net0.traj_loss(
                        traj_pred, det_boxes, det_n, traj_gt, tvalid_gt)
                    if len(out) >= 11:      # v26 stationary-flag head
                        loss = loss + args.traj_w * net0.stat_loss(
                            out[10], det_boxes, det_n, traj_gt, tvalid_gt)
                if use_tl and len(out) >= 12:   # v27 traffic-light state
                    loss = loss + args.tl_w * net0.tl_loss(out[11], tl_gt)
                if use_risk and len(out) >= 13:  # v28 area risk map
                    loss = loss + args.risk_w * net0.risk_loss(out[12],
                                                               risk_gt)
                if use_flow and len(out) >= 14 and traj_gt is not None:
                    loss = loss + args.flow_w * net0.flow_loss(
                        out[13], det_boxes, det_n, traj_gt, tvalid_gt)
                if use_lg and len(out) >= 17:
                    loss = loss + args.lanegraph_w * net0.lanegraph_loss(
                        out[14], out[15], out[16],
                        lg_pts_gt, lg_cls_gt, lg_n_gt, lg_adj_gt)
                if use_unk and len(out) >= 18:
                    loss = loss + args.unk_w * net0.unk_loss(
                        out[17], unk_c, unk_n)
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
            # ---- step-level probe of the two priority metrics -------------
            # An epoch is ~75 min; a regression (or a fix) must be visible in
            # minutes, not hours. Rank 0 runs a small val slice while the
            # other ranks block on the next all-reduce, then a barrier
            # re-syncs everyone.
            if args.val_every and step % args.val_every == 0:
                if is_main:
                    torch.cuda.empty_cache()
                    netq = model.module if ddp else model
                    iq = evaluate(netq, dv, device, max_batches=10)
                    mq = float(np.nanmean(list(iq.values())))
                    msg = (f"[probe ep{ep} step{step}] mIoU={mq:.3f} "
                           f"road={iq.get('road', float('nan')):.3f} "
                           f"lane={iq.get('laneline', float('nan')):.3f}")
                    if use_lidar and dv_lid is not None:
                        il = evaluate(netq, dv_lid, device, max_batches=10,
                                      use_lidar=True)
                        msg += (f" | +lidar mIoU="
                                f"{float(np.nanmean(list(il.values()))):.3f}")
                    if use_boxdet:
                        dq = evaluate_det3d(netq, dv, device,
                                            4 + int(use_seg2d), max_batches=8,
                                            tmp_idx=(4 + int(use_seg2d)
                                                     + 4 * int(use_traj)
                                                     + int(use_ego)
                                                     + int(use_occ)
                                                     + int(use_tl)
                                                     + int(use_risk)
                                                     + 4 * int(use_lg)
                                                     + 2 * int(use_unk)
                                                     + int(use_lidarbev))
                                            if use_temporal else None)
                        msg += (f" | vehRn={dq['vehn']:.2f} P={dq['veh'][0]:.2f}"
                                f" vruRn={dq['vrun']:.2f}"
                                f" yaw={dq['veh_yaw']:.1f}deg")
                    print(msg, flush=True)
                # distributed ADE/ADEc probe: every rank evaluates its own
                # shard of the val set (the other 7 GPUs used to idle here),
                # sums are all-reduced, rank 0 prints -> 8x coverage at the
                # same wall time
                if use_ego and use_temporal:
                    netq2 = model.module if ddp else model
                    sums = evaluate_ego(
                        netq2, dv, device,
                        4 + int(use_seg2d) + 4 * int(use_traj),
                        max_batches=12, batch_stride=3,
                        shard=(rank if ddp else 0, world if ddp else 1),
                        raw=True,
                        tmp_idx=(4 + int(use_seg2d) + 4 * int(use_traj)
                                 + int(use_ego) + int(use_occ)
                                 + int(use_tl) + int(use_risk)
                                 + 4 * int(use_lg) + 2 * int(use_unk)
                                 + int(use_lidarbev)))
                    torch.cuda.empty_cache()
                    if ddp:
                        dist.all_reduce(sums)
                    if is_main:
                        s_ = sums.cpu().numpy()
                        if s_[5] > 0:
                            ac = s_[6] / s_[7] if s_[7] > 0 else float("nan")
                            print(f"[probeE2E ep{ep} step{step}] "
                                  f"ADE={s_[0] / s_[5]:.3f} ADEc={ac:.3f} "
                                  f"(n={int(s_[5])} nc={int(s_[7])})",
                                  flush=True)
                if ddp:
                    dist.barrier()
        if is_main:
            torch.cuda.empty_cache()
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
                                  4 + int(use_seg2d) + 4 * int(use_traj)
                                  + int(use_ego))
                if oc:
                    onm = {0: "free", 2: "veh", 4: "ped", 5: "road",
                           7: "veg", 8: "bldg", 9: "pole"}
                    print(f"[valOCC ep{ep}] mIoU={np.mean(list(oc.values())):.3f} "
                          + " ".join(f"{onm[c]}={oc[c]:.3f}"
                                     for c in onm if c in oc), flush=True)
            vtmp = (4 + int(use_seg2d) + 4 * int(use_traj) + int(use_ego)
                    + int(use_occ) + int(use_tl) + int(use_risk)
                    + 4 * int(use_lg) + 2 * int(use_unk)
                    + int(use_lidarbev)) \
                if use_temporal else None
            if use_traj and use_boxdet:
                d3 = evaluate_det3d(net, dv, device, 4 + int(use_seg2d),
                                    tmp_idx=vtmp)
                print(f"[val3D ep{ep}] "
                      f"veh P={d3['veh'][0]:.2f} R={d3['veh'][1]:.2f} "
                      f"R50={d3['veh50']:.2f} Rn={d3['vehn']:.2f} "
                      f"err={d3['veh'][2]:.2f}m "
                      f"yaw={d3['veh_yaw']:.1f}deg flip={d3['veh_flip']:.2f} | "
                      f"vru P={d3['vru'][0]:.2f} R={d3['vru'][1]:.2f} "
                      f"R50={d3['vru50']:.2f} Rn={d3['vrun']:.2f} "
                      f"err={d3['vru'][2]:.2f}m", flush=True)
            if use_traj:
                tj = evaluate_traj(net, dv, device, 4 + int(use_seg2d),
                                   tmp_idx=vtmp)
                if tj:
                    ss = (f" statAcc={tj['stat_acc']:.2f}"
                          if "stat_acc" in tj else "")
                    pc = "".join(
                        f" {k}ADE={tj[k + '_ade']:.2f}" for k in ("veh", "vru")
                        if k + "_ade" in tj)
                    hd = "".join(
                        f" {k}Head={tj[k + '_head']:.0f}deg"
                        for k in ("veh", "vru") if k + "_head" in tj)
                    print(f"[valTraj ep{ep}] agentADE={tj['ade']:.2f}m "
                          f"agentFDE={tj['fde']:.2f}m" + ss + pc + hd,
                          flush=True)
            if use_tl:
                tl_idx = (4 + int(use_seg2d) + 4 * int(use_traj)
                          + int(use_ego) + int(use_occ))
                tr_ = evaluate_tl(net, dv, device, tl_idx, tmp_idx=vtmp)
                if tr_:
                    print(f"[valTL ep{ep}] acc={tr_['acc']:.2f} "
                          + " ".join(f"{k}={tr_[k]:.2f}" for k in
                                     ("none", "green", "yellow", "red")
                                     if k in tr_), flush=True)
            if use_unk:
                uk_idx = (4 + int(use_seg2d) + 4 * int(use_traj)
                          + int(use_ego) + int(use_occ) + int(use_tl)
                          + int(use_risk) + 4 * int(use_lg))
                uk = evaluate_unknown(net, dv, device, uk_idx, tmp_idx=vtmp)
                if uk:
                    print(f"[valUnk ep{ep}] P={uk['p']:.2f} R={uk['r']:.2f}",
                          flush=True)
            if use_risk:
                rk_idx = (4 + int(use_seg2d) + 4 * int(use_traj)
                          + int(use_ego) + int(use_occ) + int(use_tl))
                rk = evaluate_risk(net, dv, device, rk_idx, tmp_idx=vtmp)
                if rk:
                    print(f"[valRisk ep{ep}] L1={rk['l1']:.3f} "
                          f"L1(hi)={rk['l1_hi']:.3f}", flush=True)
            if use_flow:
                fl_ = evaluate_flow(net, dv, device, 4 + int(use_seg2d),
                                    tmp_idx=vtmp)
                if fl_:
                    print(f"[valFlow ep{ep}] EPE(mov)={fl_['epe_mov']:.2f} "
                          f"EPE(stat)={fl_['epe_stat']:.2f} m/s", flush=True)
            if use_lg:
                lg_idx = (4 + int(use_seg2d) + 4 * int(use_traj)
                          + int(use_ego) + int(use_occ) + int(use_tl)
                          + int(use_risk))
                lg_ = evaluate_lanegraph(net, dv, device, lg_idx,
                                         tmp_idx=vtmp)
                if lg_:
                    print(f"[valLane ep{ep}] P={lg_['p']:.2f} "
                          f"R={lg_['r']:.2f} adjAcc={lg_['adj']:.2f}",
                          flush=True)
            if use_ego:
                eg = evaluate_ego(net, dv, device,
                                  4 + int(use_seg2d) + 4 * int(use_traj),
                                  tmp_idx=vtmp)
                if eg:
                    print(f"[valE2E ep{ep}] ADE={eg['ade']:.2f}m "
                          f"ADEc={eg['ade_c']:.2f}m "
                          f"FDE={eg['fde']:.2f}m steer={eg['steer']:.3f}rad "
                          f"acc={eg['acc']:.2f}m/s2 brakeAcc={eg['brake']:.2f}",
                          flush=True)
            torch.save({"model": net.state_dict(), "epoch": ep, "ious": ious},
                       os.path.join(args.out, "last.pt"))
            # composite best: BEV mIoU minus a small penalty for E2E curve
            # error, so "best" never selects a pre-curve-convergence epoch
            score = miou
            if use_ego and eg and eg.get("ade_c") == eg.get("ade_c"):
                score = miou - 0.01 * min(eg["ade_c"], 5.0)
            if score > best:
                best = score
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

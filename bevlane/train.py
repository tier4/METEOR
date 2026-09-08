#!/usr/bin/env python3
"""Train IPMSegNet on extracted bevlane samples.

Single GPU:  python3 bevlane/train.py
8-GPU DDP:   torchrun --nproc_per_node=8 bevlane/train.py --batch 14
"""
import argparse
import os
import subprocess
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
from bevlane.model import (DET_RES, MODELS, N_CLASSES, EGO_K,  # noqa: E402
                           make_warp_theta)

CLASS_NAMES = ["unlabeled", "road", "sidewalk", "crosswalk", "laneline",
               "stopline", "road_edge", "marking", "parking"]
CLASS_W = torch.tensor([0.0, 1.0, 1.5, 3.0, 5.0, 6.0, 1.5, 3.0, 1.0])
THIN = [3, 4, 5, 6, 7]
DICE_CLASSES = [3]        # area-like (crosswalk): want recall
LINE_CLASSES = [4, 5, 6]  # laneline / stopline / road_edge: precision/thin
# marking/class 7 carried no supervision until r49: Japanese gt_cons has
# 0.000 % marking pixels (hence `marking=nan` in every val line) while x2gen2
# GT carries 0.403 %. CLASS_W[7]=3.0 was already set, so simply having x2gen2
# in the round starts training it -- measure it on an x2gen2 val slice, the
# Japanese val set cannot see the class at all.


# ---- distance field from the drivable surface + off-road penalty (2026-08-16, ported from the VLA variant) ----
# The lateral-command loss (intent_loss) has no notion of "does the road allow it";
# the reference implementation recorded 1.15 of 6 points leaving the road on left
# commands and hitting the safety-layer VETO threshold in 14% of frames. Our
# closed loop shows VETO 7.5%, presumably the same pathology.
DRIVABLE = (1, 3, 4, 5, 7, 8)   # road/crosswalk/laneline/stopline/marking/parking
DIST_CELL_M = 1.6               # 0.2 m raster downsampled 8x
DIST_MAX_CELL = 12              # saturates at ~19 m


def drivable_dist(seg_gt):
    """Distance [m] from each cell to the nearest drivable cell (capped).

    A binary mask is flat beyond the wall so the gradient vanishes (measured in the
    reference impl: the avg-pool variant did nothing). A distance field pushes back
    toward the road even from 15 m inside. Iterated dilation, not cv2.distanceTransform: stays on the GPU.
    """
    dr = torch.zeros(seg_gt.shape, device=seg_gt.device, dtype=torch.float32)
    for c in DRIVABLE:
        dr = dr + (seg_gt == c).float()
    cur = (F.avg_pool2d(dr.clamp(max=1.0)[:, None], 8) >= 0.5).float()
    dist = torch.zeros_like(cur)
    for _ in range(DIST_MAX_CELL):
        cur = F.max_pool2d(cur, 3, 1, 1)
        dist = dist + (1.0 - cur)
    return dist * DIST_CELL_M


def offroad_loss(seg_gt, ego_pred, rows=None, topk=2, k_modes=None):
    """Penalize the commanded path leaving the drivable surface.

    Design points (all settled empirically in the reference implementation):
      * use a distance field (gradient survives beyond the wall)
      * worst 2 of the 6 points, not the mean (5 good points must not hide 1 breach)
      * only the selected mode, not a blend of the 3 (keeps the pressure from being diluted 3x)
      * only rows that carry a command (on command-less rows the GT itself grazes the
        curb, so the penalty would become a second, conflicting objective)
    """
    from bevlane.model import EGO_K as _K
    K = k_modes or _K
    if rows is not None:
        if rows.sum() == 0:
            return ego_pred.sum() * 0.0
        seg_gt, ego_pred = seg_gt[rows], ego_pred[rows]
    B = seg_gt.shape[0]
    cost = (drivable_dist(seg_gt) / 5.0).clamp(max=1.0)     # 0..1 (0..5 m)
    e = ego_pred.float()
    wp = e[:, :12 * K].view(B, K, 6, 2)
    md = e[:, 12 * K:12 * K + K].argmax(1)
    sel = wp[torch.arange(B, device=wp.device), md]         # [B,6,2]
    H, W = seg_gt.shape[-2], seg_gt.shape[-1]
    xf = 80.0                       # row 0 = 80 m ahead (rear extent follows from H)
    gx = ((50.0 - sel[..., 1]) / 0.2) / W * 2.0 - 1.0
    gy = ((xf - sel[..., 0]) / 0.2) / H * 2.0 - 1.0
    grid = torch.stack([gx, gy], -1).view(B, 6, 1, 2)
    s = F.grid_sample(cost, grid.float(), align_corners=False,
                      padding_mode="border").view(B, 6)
    return s.topk(min(topk, 6), dim=1).values.mean()


def dice_loss(logits, gt, classes=THIN, eps=1.0):
    """Soft dice over selected classes, masked to labeled cells."""
    prob = logits.softmax(1)
    m = ((gt > 0) & (gt != 255)).unsqueeze(1).float()
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
    m = ((gt > 0) & (gt != 255)).unsqueeze(1).float()
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


def lane_cldice_loss(logits, gt, iters=5, eps=1e-6):
    """Topology-aware soft-clDice for the laneline class (4).

    It penalises broken centreline connectivity, which pixel IoU/Tversky do
    not distinguish from an equal-area set of disconnected fragments. The
    map is pooled once to keep the training cost modest; max pooling preserves
    thin evidence. Consensus 255 cells remain completely outside the loss.
    """
    p = logits.softmax(1)[:, 4:5]
    t = (gt == 4).float().unsqueeze(1)
    valid = (gt != 255).float().unsqueeze(1)
    p = F.max_pool2d(p * valid, 2, 2)
    t = F.max_pool2d(t, 2, 2)
    valid = -F.max_pool2d(-valid, 2, 2)  # valid only when all 2x2 cells valid
    p, t = p * valid, t * valid

    def erode(x):
        return -F.max_pool2d(-x, 3, 1, 1)

    def skel(x):
        opened = F.max_pool2d(erode(x), 3, 1, 1)
        s = F.relu(x - opened)
        cur = x
        for _ in range(iters):
            cur = erode(cur)
            opened = F.max_pool2d(erode(cur), 3, 1, 1)
            delta = F.relu(cur - opened)
            s = s + F.relu(delta - s * delta)
        return s

    sp, st = skel(p), skel(t)
    tprec = (sp * t).sum() / (sp.sum() + eps)
    tsens = (st * p).sum() / (st.sum() + eps)
    return 1.0 - (2.0 * tprec * tsens + eps) / (tprec + tsens + eps)


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


def boundary_weight(gt, radius=2, w=4.0, ignore_unlabeled=False):
    """Per-pixel weight map: `w` on cells within `radius` of a class boundary.

    ignore_unlabeled: exclude boundaries against class 0 (unlabeled / black).
    That edge is just the moving rim of the observed BEV area, not a real
    semantic boundary; weighting it trains the model to draw a sharp,
    frame-unstable line against the don't-care region. With this on, a cell
    is a boundary only when its neighborhood holds >=2 LABELED classes."""
    g = gt.float().unsqueeze(1)
    if ignore_unlabeled:
        # min over labeled cells only: push class 0 to a large value so it is
        # never the neighborhood minimum (black cannot create an edge)
        gpos = torch.where(g == 0, torch.full_like(g, 1e4), g)
        mx = F.max_pool2d(g, 2 * radius + 1, 1, radius)          # black=0 low
        mn = -F.max_pool2d(-gpos, 2 * radius + 1, 1, radius)     # ignores black
        bnd = ((mx != mn) & (mx > 0) & (mn < 1e3)).float().squeeze(1)
    else:
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



_GIT = subprocess.run(["git", "rev-parse", "HEAD"],
                      capture_output=True, text=True).stdout.strip()


def sanitize_bn(model):
    """Repair non-finite BN running statistics; returns the names fixed.

    BN updates its running stats during the FORWARD pass, so one inf/nan
    activation poisons them for good: training keeps working (it uses batch
    statistics) while every eval-mode forward returns nan. r47 lost its E2E
    metric this way. DDP broadcasts buffers from rank 0, so repairing them
    here propagates to all ranks."""
    bad = []
    with torch.no_grad():
        for n, b in model.named_buffers():
            if b.dtype.is_floating_point and not torch.isfinite(b).all():
                bad.append(n)
                if "running_var" in n:
                    b[~torch.isfinite(b)] = 1.0
                else:
                    b[~torch.isfinite(b)] = 0.0
        for n, b in model.named_buffers():
            if "running_var" in n:
                b.clamp_(min=1e-5)
    return bad


HOLDOUT_FILES = ("test.lst", "out/x2gen2_test.txt", "out/newdata_test.txt",
                 "out/okinawa_test.txt")


def holdout_scenes(root):
    """Scenes that must NEVER be trained on, from every holdout list we keep.

    Enforced in code, not in launcher flags: the refiner takes its train set
    from split_scenes(root) -- i.e. everything symlinked under out/bevlane --
    so r47's refiner silently trained on all 279 held-out test scenes. A flag
    that has to be remembered is not a guarantee.
    """
    base = os.path.dirname(os.path.abspath(root.rstrip("/")))
    out = set()
    for f in HOLDOUT_FILES:
        for p in (f, os.path.join(base, f), os.path.join(base, os.path.basename(f))):
            if os.path.exists(p):
                out |= {l.strip() for l in open(p) if l.strip()}
                break
    return out


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
    ho = holdout_scenes(root)
    train = [s for s in scenes if s not in set(val) and s not in ho]
    leak = [s for s in val if s in ho]
    assert not leak, f"{len(leak)} val scenes are also in a holdout list"
    if ho:
        print(f"[holdout] {len(ho)} scenes excluded from training "
              f"({', '.join(HOLDOUT_FILES)})", flush=True)
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
        gt = _fit(gt, pred)
        m = (gt > 0) & (gt != 255)
        for c in range(1, N_CLASSES):
            pi, gi = (pred == c) & m, gt == c
            inter[c] += (pi & gi).sum().item()
            union[c] += (pi | gi).sum().item()
    ious = {CLASS_NAMES[c]: inter[c] / union[c] if union[c] else float("nan")
            for c in range(1, N_CLASSES)}
    model.train()
    _reeval_frozen(model)
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
    _reeval_frozen(model)
    return {c: inter[c] / union[c] for c in range(n_cls) if union[c]}


@torch.no_grad()

def bev_rotation_aug(theta_max_deg, Tc, gt, det_boxes, det_n, traj_gt,
                     ego_gt, occ_gt, risk_gt, lg_pts, unk_c, rel_pose,
                     unk_v2=None, p_apply=0.5, lat_max=0.0, lat_p=0.0,
                     lat_min=0.3):
    """BEV-space SE(2) augmentation: rotate/TRANSLATE the EGO FRAME, not
    pixels.

    T_cam_ego absorbs the transform; because features are lifted by
    per-pixel DEPTH before projection, a rigid SE(2) of the rig is exact
    for ALL structures (not just the ground plane). Rotation = the rig
    yawed by theta (r31+). NEW (v45): lateral offset dy simulates a
    LANE-DEPARTED ego; the recorded future, re-expressed in the offset
    frame and hermite-smoothed from the new origin, becomes a
    recovery-to-lane target (ChauffeurNet-style)."""
    B = Tc.shape[0]
    dev = Tc.device
    th = (torch.rand(B, device=dev) * 2 - 1) * math.radians(theta_max_deg)
    th = th * (torch.rand(B, device=dev) < p_apply).float()
    from bevlane.model import (BEV_XF as _BXF, BEV_XR as _BXR,
                               BEV_YH as _BYH)
    dy = torch.zeros(B, device=dev)
    if lat_max > 0 and lat_p > 0:
        mag = lat_min + (lat_max - lat_min) * torch.rand(B, device=dev)
        sgn = torch.where(torch.rand(B, device=dev) < 0.5, -1.0, 1.0)
        dy = mag * sgn * (torch.rand(B, device=dev) < lat_p).float()
    c, s = th.cos(), th.sin()
    # 1. extrinsics: p_old = Rz(th) p_new -> Tc' = Tc @ Rz(th)
    R = torch.zeros(B, 4, 4, device=dev, dtype=Tc.dtype)
    R[:, 0, 0] = c; R[:, 0, 1] = -s
    R[:, 1, 0] = s; R[:, 1, 1] = c
    R[:, 2, 2] = 1; R[:, 3, 3] = 1
    R[:, 1, 3] = dy.to(Tc.dtype)         # SE(2): virtual lateral offset
    Tc = Tc @ R[:, None]
    # SE(2) for points expressed in NEW frame: p_new = R(-th)(p_old - t)
    def rot_pts(xy):                     # [...,2] (x,y)
        shp = [B] + [1] * (xy.dim() - 2)
        cc, ss = c.view(shp), s.view(shp)
        x, y = xy[..., 0], xy[..., 1] - dy.view(shp)
        return torch.stack([cc * x + ss * y, -ss * x + cc * y], -1)
    # 2. label rasters via inverse-rotated sampling grid
    def rot_raster(r, fill, nearest=True, ego_v=None):
        if r is None:
            return None
        r4 = r.float().unsqueeze(1) if r.dim() == 3 else r.float()
        A = torch.zeros(B, 2, 3, device=dev, dtype=torch.float32)
        # Rotation direction (fixed 2026-08-14): affine_grid's (u,v)=(col,row)
        # coordinates are an axis swap = mirror of BEV (x,y), so rotating the
        # same way as the point side (rot_pts / camera Tc@R) requires flipping
        # the sign of s. The old code rotated only the rasters backwards, so every
        # rot-aug sample had seg/occ GT off from images/boxes by 2θ (up to 20 deg).
        A[:, 0, 0] = c; A[:, 0, 1] = s * (r4.shape[2] / r4.shape[3])
        A[:, 1, 0] = -s * (r4.shape[3] / r4.shape[2]); A[:, 1, 1] = c
        # Rotation-center fix (2026-08-13): affine_grid rotates about the raster
        # center, but cameras/boxes/ego (rot_pts) rotate about the ego. On the
        # symmetric ±80 grid the two coincide (harmless); on rear-40 (+80/−40) the
        # raster center is +20m and GT rasters alone shift up to ~20*sinθ≈3.5m
        # (every round since v58 was hit). Correct to rotate about the ego row's
        # normalized coordinate y_e; on ±80, y_e=0 reproduces the old behaviour exactly.
        # The rotation center differs per raster. Lane/det/unk span the whole grid
        # (+XF..−XR) with the ego off-center, so they need the ego-row correction;
        # occ/risk (symmetric ±40 m window) have center = ego and must NOT be corrected.
        # Inferring from shape is unsafe (on the symmetric grid lane 800x500 and
        # risk 400x250 share aspect 1.6), so the caller states it explicitly.
        _ye = (2.0 * _BXF / (_BXF + _BXR) - 1.0) if ego_v is None else ego_v
        A[:, 0, 2] = -dy / _BYH - s * (r4.shape[2] / r4.shape[3]) * _ye
        A[:, 1, 2] = _ye * (1.0 - c)
        g = F.affine_grid(A, list(r4.shape), align_corners=False)
        out = F.grid_sample(r4 + 1.0, g, mode="nearest" if nearest
                            else "bilinear", padding_mode="zeros",
                            align_corners=False)
        res = torch.where(out < 0.5, torch.full_like(out, fill + 1.0),
                          out) - 1.0
        return res.squeeze(1).to(r.dtype) if r.dim() == 3 else res.to(r.dtype)
    gt = rot_raster(gt, 0)
    risk_gt = rot_raster(risk_gt, 0, nearest=False, ego_v=0.0)         if risk_gt is not None else None
    if occ_gt is not None:
        occ_gt = rot_raster(occ_gt, 255, ego_v=0.0)
    if unk_v2 is not None:                    # -1 fill = no-GT sentinel
        unk_v2 = rot_raster(unk_v2, -1)
    # 3. boxes / futures / ego path / graph points / unknown centres
    if det_boxes is not None:
        det_boxes = det_boxes.clone()
        det_boxes[..., 1:3] = rot_pts(det_boxes[..., 1:3])
        det_boxes[..., 5] = det_boxes[..., 5] - th[:, None]
        if traj_gt is not None:
            traj_gt = rot_pts(traj_gt)
    if ego_gt is not None:
        ego_gt = ego_gt.clone()
        wp = rot_pts(ego_gt[:, :12].view(B, 6, 2))
        # v45 recovery target: for laterally offset samples, smooth-connect
        # from the (departed) origin so the target is a kink-free
        # return-to-lane path, not a teleport
        for b in range(B):
            if abs(float(dy[b])) < 1e-6:
                continue
            v0b = float(ego_gt[b, 12])
            k = 4 if v0b > 15.0 else 3
            P0 = wp.new_zeros(2)
            T0 = wp.new_tensor([max(v0b * 0.5, 2.0), 0.0])
            P1 = wp[b, k]
            T1 = (wp[b, k + 1] - wp[b, k - 1]) if k + 1 < 6 \
                else (wp[b, k] - wp[b, k - 1])
            for i in range(k):
                u = (i + 1) / (k + 1)
                h00 = 2*u**3 - 3*u**2 + 1; h10 = u**3 - 2*u**2 + u
                h01 = -2*u**3 + 3*u**2;    h11 = u**3 - u**2
                wp[b, i] = h00*P0 + h10*T0 + h01*P1 + h11*T1
        ego_gt[:, :12] = wp.reshape(B, 12)
    if lg_pts is not None:
        lg_pts = rot_pts(lg_pts)
    if unk_c is not None:
        unk_c = rot_pts(unk_c)
    if rel_pose is not None:
        rel_pose = rel_pose.clone()
        rel_pose[..., :2] = rot_pts(rel_pose[..., :2])
    return (Tc, gt, det_boxes, traj_gt, ego_gt, occ_gt, risk_gt, lg_pts,
            unk_c, rel_pose, unk_v2)


class EMA:
    """Exponential moving average of the trainable weights.

    Consecutive epoch-end evaluations bounce while nothing about the data
    changes -- r61 went ADE 0.63 / 0.66 / 0.62 / 0.63 over its last four -- which
    is weights orbiting the basin rather than sitting in it. The independent
    evidence that averaging helps HERE is that the weight soup over r59/r60/r61
    beat every one of them (0.6199 -> 0.6056) for free, and narrowed the mode
    selection loss from 0.171 to 0.143 along with it.

    Costs one fp32 copy of the trainable set (54.2 M = 217 MB per rank). The
    averaged weights are EVALUATED ALONGSIDE the raw ones every epoch rather
    than assumed better, so the choice stays measured.
    """

    def __init__(self, module, decay=0.999, exclude=()):
        self.decay = decay
        # exclude: prefixes kept out of the EMA (2026-09-08, v158): averaging seg_head.* makes the
        # 2D seg collapse to a single class in lr 1e-4 runs (reproduced every epoch in v151/v152/v155/v157).
        # Excluded heads keep their raw weights on swap_in (same as making the guard's "graft raw seg_head" permanent).
        self.shadow = {n: p.detach().clone().float()
                       for n, p in module.named_parameters()
                       if p.requires_grad and not n.startswith(tuple(exclude))}

    @torch.no_grad()
    def update(self, module):
        d = self.decay
        for n, p in module.named_parameters():
            if n in self.shadow:
                self.shadow[n].mul_(d).add_(p.detach().float(), alpha=1 - d)

    @torch.no_grad()
    def swap_in(self, module):
        """Install the averaged weights; returns the raw ones for swap_out."""
        backup = {}
        for n, p in module.named_parameters():
            if n in self.shadow:
                backup[n] = p.detach().clone()
                p.copy_(self.shadow[n].to(p.dtype))
        return backup

    @torch.no_grad()
    def swap_out(self, module, backup):
        for n, p in module.named_parameters():
            if n in backup:
                p.copy_(backup[n])



EGO_FREEZE_PREFIX = ("ego_stem", "ego_mlp", "ego_q", "ego_attn", "ego_delta",
                     "sem_ego", "kin_delta", "intent_delta", "intent_mlp",
                     "vprof_head", "dec_head", "dec_gate", "risk_gate",
                     "mode_scorer", "kin_gate", "refiner.e2e")


def _apply_freeze_ego(mdl):
    """--freeze-ego (v142): freeze the E2E head. Clears requires_grad and fixes BN stats.
    **Call before the DDP wrap** (freezing afterwards makes DDP wait for gradients
    and crash with "Expected to have finished reduction" — seen on the first v142 run).
    _reeval_frozen re-applies the BN eval pin on every model.train()."""
    base = mdl.module if hasattr(mdl, "module") else mdl
    base._freeze_ego_prefix = EGO_FREEZE_PREFIX
    n = 0
    for nm, p in base.named_parameters():
        if nm.startswith(EGO_FREEZE_PREFIX):
            p.requires_grad_(False); n += p.numel()
    for nm, m in base.named_modules():
        if nm.startswith(EGO_FREEZE_PREFIX) and isinstance(
                m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.SyncBatchNorm)):
            m.eval()
    return n


def _reeval_frozen(model):
    """--det-head-only: keep the frozen part's BN from waking up on model.train()."""
    n0 = model.module if hasattr(model, "module") else model
    # --freeze-ego (v142): pin the BN of the ego modules to eval
    _fe = getattr(n0, "_freeze_ego_prefix", None)
    if _fe:
        for nm, m in n0.named_modules():
            if nm.startswith(_fe) and isinstance(
                    m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.SyncBatchNorm)):
                m.eval()
    keep = getattr(n0, "_freeze_eval_keep", None)
    if not keep:
        return
    for nm, m in n0.named_modules():
        if nm and not (nm.split(".")[0] in keep
                       or any(nm.startswith(k) for k in keep if "." in k)):
            m.eval()


def _fit(gt, pred):
    """Crop a BEV-space GT tensor's ROW dimension to the prediction's.

    Every BEV product in this pipeline puts row 0 at the far FRONT and shares
    its resolution with the head that predicts it, so truncating the rear of the
    grid means keeping exactly the first N rows of the label -- whatever N the
    head now emits. Deriving N from the prediction rather than from a per-key
    table is what makes this safe: two products can have the SAME shape and
    different geometry (lidar_bev is 400x250 at 0.4 m over +-80 m, risk is
    400x250 at 0.2 m over +-40 m), so a table keyed on shape would silently
    mis-crop one of them.

    A no-op when the grid is full length, which is the default.
    """
    if gt is None or not torch.is_tensor(gt) or not torch.is_tensor(pred):
        return gt
    h = pred.shape[-2]
    return gt if gt.shape[-2] <= h else gt[..., :h, :]


def _temporal_inputs(model, batch, device, tmp_idx):
    if tmp_idx is None:
        return None, None
    # History images are uniquely [B,H,N,3,432,768] = 6-D. A fixed index formula
    # breaks when the flag set changes, so fall back to a shape lookup if it does not match.
    if not (tmp_idx < len(batch) and torch.is_tensor(batch[tmp_idx])
            and batch[tmp_idx].dim() == 6):
        _i = next((i for i, t in enumerate(batch)
                   if torch.is_tensor(t) and t.dim() == 6), None)
        if _i is None:
            return None, None
        tmp_idx = _i
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
    ado = adcv = admv = adst = 0.0
    nmv = nst = 0.0
    # Chain-divergence proxy (2026-09-04): wp0 longitudinal bias in the high-speed band (v0 8-15 m/s).
    # chain_decomp.py showed the chain divergence at +3.2s is nearly proportional to this
    # quantity (v132 −0.50, v140 last −0.87). Reading it at epoch end allows ckpt selection.
    hsb = hsn = 0.0
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
        v_ = eg[:, 16]
        if v_.sum() == 0:
            continue
        pb, th = _temporal_inputs(model, batch, device, tmp_idx)
        with torch.autocast("cuda", torch.float16):
            out = model(imgs, K, Tc, eg[:, 12], pb, th) if th is not None \
                else model(imgs, K, Tc, eg[:, 12])
        if not (isinstance(out, tuple) and len(out) >= 8):
            break
        p = out[7].float()
        # Decomposition, so an ADE change can be attributed instead of guessed.
        # Measured on r60 best_e2e over 720 val samples: ADE 0.622, oracle
        # 0.491 (0.131 lost to mode SELECTION), constant-velocity 0.808 (the
        # model beats "keep going straight at this speed" by only 23 %), and
        # the stationary quarter of the frames is the WORST group at 0.688.
        # None of that is visible from a single ADE number.
        gwp = eg[:, :12].view(-1, 6, 2)
        if p.shape[1] > 15:                    # v29 multimodal (K=3)
            Kn = 3
            wps = p[:, :12 * Kn].view(-1, Kn, 6, 2)
            dk = (wps - gwp[:, None]).norm(dim=3).mean(2)      # [B,K]
            ado += (dk.min(1).values * v_).sum().item()
            mode = p[:, 12 * Kn:12 * Kn + Kn].argmax(1)
            wp1 = wps[torch.arange(len(p)), mode].reshape(-1, 12)
            p = torch.cat([wp1, p[:, 12 * Kn + Kn:]], 1)
        else:
            ado += 0.0
        _t = (torch.arange(6, device=p.device, dtype=p.dtype) + 1) * 0.5
        cvp = torch.zeros_like(gwp)
        cvp[..., 0] = eg[:, 12:13] * _t[None]
        adcv += ((cvp - gwp).norm(dim=2).mean(1) * v_).sum().item()
        v = v_
        d = (p[:, :12].view(-1, 6, 2) - eg[:, :12].view(-1, 6, 2)).norm(dim=2)
        ade += (d.mean(1) * v).sum().item()
        fde += (d[:, -1] * v).sum().item()
        vc = v * (eg[:, 11].abs() > 2.0).float()   # curve subset |lat@3s|>2m
        adec += (d.mean(1) * vc).sum().item()
        nc += vc.sum().item()
        vhs = v * ((eg[:, 12] >= 8.0) & (eg[:, 12] < 15.0)).float()
        hsb += ((p[:, 0] - eg[:, 0]) * vhs).sum().item(); hsn += vhs.sum().item()
        vmv = v * (eg[:, 12] > 2.0).float()          # moving
        vst = v * (eg[:, 12] <= 2.0).float()         # stopped / crawling
        admv += (d.mean(1) * vmv).sum().item(); nmv += vmv.sum().item()
        adst += (d.mean(1) * vst).sum().item(); nst += vst.sum().item()
        smae += ((p[:, 12] - eg[:, 14]).abs() * v).sum().item()
        amae += ((p[:, 13] - eg[:, 13]).abs() * v).sum().item()
        bacc += (((p[:, 14] > 0).float() == eg[:, 15]).float() * v).sum().item()
        n += v.sum().item()
    model.train()
    _reeval_frozen(model)
    if raw:
        return torch.tensor([ade, fde, smae, amae, bacc, n,
                             adec, nc], dtype=torch.float64, device=device)
    if n == 0:
        return None
    return {"ade": ade / n, "fde": fde / n, "steer": smae / n,
            "acc": amae / n, "brake": bacc / n,
            "ade_c": adec / nc if nc else float("nan"),
            "ade_o": ado / n, "ade_cv": adcv / n,
            "ade_mv": admv / nmv if nmv else float("nan"),
            "ade_st": adst / nst if nst else float("nan"),
            "hs_bias": hsb / hsn if hsn else float("nan"), "hs_n": hsn}


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
        og = _fit(og, pred)
        valid = og != 255
        if valid.sum() == 0:
            continue
        for c in range(10):
            pi, gi = (pred == c) & valid, og == c
            inter[c] += (pi & gi).sum().item()
            union[c] += (pi | gi).sum().item()
    model.train()
    _reeval_frozen(model)
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
    _reeval_frozen(model)
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
        gt = _fit(gt, p)
        m = gt >= 0
        if m.sum() == 0:
            continue
        d = (p - gt.clamp(min=0)).abs()
        l1 += float(d[m].sum()); n += float(m.sum())
        hi = m & (gt > 0.4)
        if hi.any():
            l1h += float(d[hi].sum()); nh += float(hi.sum())
    model.train()
    _reeval_frozen(model)
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
    _reeval_frozen(model)
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
        bx, nb = batch[bx_idx], batch[bx_idx + 1].clamp(min=0)
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
                # a rear-truncated grid gives the flow head fewer rows than the
                # +-40 m window; agents behind the new limit have no cell
                if not (0 <= ri < fl.shape[-2] and 0 <= ci < fl.shape[-1]):
                    continue
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
    _reeval_frozen(model)
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
    _reeval_frozen(model)
    if tp + fn == 0:
        return None
    return {"p": tp / max(tp + fp, 1), "r": tp / max(tp + fn, 1)}


def evaluate_unknown_dense(model, loader, device, um_idx, max_batches=20,
                           tmp_idx=None, thresh=0.3):
    """dense small-obstacle occupancy: object-level P/R (2 m centroid match
    via connected components) + far-range (>30 m) recall. Grid is the DET
    grid (400x250 @ DET_RES m/cell); row r -> xe = BEV_XH - r*DET_RES."""
    from scipy import ndimage
    from bevlane.model import DET_RES, BEV_XH, BEV_YH
    tp = fp = fn = tp_f = fn_f = 0
    model.eval()
    for bi, batch in enumerate(loader):
        if bi >= max_batches:
            break
        imgs, K, Tc = (t.to(device, non_blocking=True) for t in batch[:3])
        um = batch[um_idx]                       # [B,400,250] float, -1=no GT
        pb, th = _temporal_inputs(model, batch, device, tmp_idx)
        with torch.no_grad(), torch.autocast("cuda", torch.float16):
            out = model(imgs, K, Tc, None, pb, th)
        if len(out) < 18:
            break
        prob = out[17].float().sigmoid()[:, 0].detach().cpu().numpy()
        um = _fit(um, out[17])
        for b in range(prob.shape[0]):
            g = um[b].numpy()
            if (g >= 0).sum() == 0:              # all -1 = no GT frame
                continue
            dc = g < -0.5                        # per-cell don't-care (v3)
            gl, gn = ndimage.label(g > 0.5)
            pl, pn = ndimage.label(prob[b] > thresh)
            # drop tiny predicted blobs (<3 cells) as noise
            sz = ndimage.sum(prob[b] > thresh, pl, range(1, pn + 1))
            keep = [j + 1 for j in range(pn) if sz[j] >= 3]
            gt_c = ndimage.center_of_mass(g > 0.5, gl, range(1, gn + 1))
            pr_c = ndimage.center_of_mass(prob[b] > thresh, pl, keep)
            # a prediction on a don't-care (occluded) cell is neither
            # TP nor FP
            pr_c = [(r_, c_) for (r_, c_) in pr_c
                    if not dc[int(round(r_)), int(round(c_))]]
            used = [False] * len(gt_c)
            for (pr_, pc_) in pr_c:
                xe = BEV_XH - pr_ * DET_RES
                ye = BEV_YH - pc_ * DET_RES
                best, bd = -1, 2.0               # 2 m centroid match
                for gi, (gr, gc) in enumerate(gt_c):
                    if used[gi]:
                        continue
                    d = ((BEV_XH - gr * DET_RES - xe) ** 2
                         + (BEV_YH - gc * DET_RES - ye) ** 2) ** 0.5
                    if d < bd:
                        best, bd = gi, d
                if best >= 0:
                    used[best] = True
                    tp += 1
                else:
                    fp += 1
            for gi, (gr, gc) in enumerate(gt_c):
                far = (BEV_XH - gr * DET_RES) > 30.0
                if not used[gi]:
                    fn += 1
                    fn_f += int(far)
                elif far:
                    tp_f += 1
    model.train()
    _reeval_frozen(model)
    if tp + fn == 0:
        return None
    return {"p": tp / max(tp + fp, 1), "r": tp / max(tp + fn, 1),
            "r_far": tp_f / max(tp_f + fn_f, 1)}


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
        nb = batch[tj_idx + 1].clamp(min=0)
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
    _reeval_frozen(model)
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
def thickness_ratio(model, loader, device, classes=(4, 5, 6), max_batches=10):
    """predicted area / GT area per class. 1.0 is correct, >1 is too fat.

    mIoU cannot see this: widening a 1-cell line to 3 cells grows the union
    about as fast as the intersection, so r54 set a best-ever mIoU (0.3359 on
    the 7-class slice) while laneline went from 2.71x to 2.96x the GT area and
    the lines visibly bled. Anything the metric does not show, the round will
    happily make worse.
    """
    model.eval()
    P = {c: 0 for c in classes}
    G = {c: 0 for c in classes}
    for bi, batch in enumerate(loader):
        if bi >= max_batches:
            break
        imgs, K, Tc = (t.to(device, non_blocking=True) for t in batch[:3])
        with torch.autocast("cuda", torch.float16):
            out = model(imgs, K, Tc)
        pr = (out[0] if isinstance(out, tuple) else out).float().argmax(1)
        gt = _fit(batch[3].to(device), pr)
        for c in classes:
            P[c] += int((pr == c).sum())
            G[c] += int((gt == c).sum())
    return {c: (P[c] / G[c] if G[c] else float("nan")) for c in classes}


@torch.no_grad()
def evaluate_det3d(model, loader, device, bx_idx, max_batches=30,
                   tmp_idx=None, thresh=0.3, match_m=2.0):
    """BEV 3D detection: per-class precision/recall (centre match <2 m) and
    centre, size, yaw and four-corner errors on matched pairs."""
    model.eval()
    tp = [0, 0]
    fp = [0, 0]
    fn = [0, 0]
    cerr = [0.0, 0.0]
    lerr = [0.0, 0.0]
    werr = [0.0, 0.0]
    corner_err = [0.0, 0.0]
    yerr = [0.0, 0.0]
    yflip = [0, 0]
    tp50 = [0, 0]
    fn50 = [0, 0]
    tpn = [0, 0]
    fnn = [0, 0]
    # Vehicle localisation split around ego. Aggregate metrics hid the exact
    # rear/far failure this round is targeting.
    z_gt = [0, 0, 0, 0]       # front 0-40, front 40-80, rear 0-40, rear 40-80
    z_tp = [0, 0, 0, 0]
    z_err = [0.0, 0.0, 0.0, 0.0]
    z_lerr = [0.0, 0.0, 0.0, 0.0]
    z_werr = [0.0, 0.0, 0.0, 0.0]
    z_corner = [0.0, 0.0, 0.0, 0.0]
    z_yerr = [0.0, 0.0, 0.0, 0.0]
    def _zone(x):
        if x >= 0:
            return 0 if x < 40 else (1 if x <= 80 else None)
        return 2 if x > -40 else (3 if x >= -80 else None)
    def _corners(x, y, l, w, yaw):
        """Clockwise BEV corners; cyclic matching makes yaw+pi equivalent."""
        local = np.asarray(((l / 2, w / 2), (l / 2, -w / 2),
                            (-l / 2, -w / 2), (-l / 2, w / 2)),
                           dtype=np.float64)
        cs, sn = np.cos(yaw), np.sin(yaw)
        rot = np.asarray(((cs, -sn), (sn, cs)), dtype=np.float64)
        return local @ rot.T + np.asarray((x, y), dtype=np.float64)

    def _corner_mae(pred, truth):
        pc = _corners(*pred)
        gc = _corners(*truth)
        return min(float(np.linalg.norm(pc - np.roll(gc, k, axis=0), axis=1).mean())
                   for k in range(4))
    s_tp = s_fp = s_tn = s_fn = 0        # stationary-flag confusion
    s_scores, s_truth = [], []             # threshold/calibration audit
    s_zone = [[0, 0, 0, 0] for _ in range(4)]  # TP, FP, TN, FN by x zone
    for bi, batch in enumerate(loader):
        if bi >= max_batches:
            break
        imgs, K, Tc = (t.to(device, non_blocking=True) for t in batch[:3])
        # Locate the box tensor by shape (2026-08-14): zeroing a loss weight drops
        # use_depth/use_seg2d etc., the tuple length changes, and the caller's
        # fixed index formula (4 + use_seg2d ...) points at a different tensor.
        # Boxes are uniquely [B,K,6]; counts are an integer [B].
        if not (torch.is_tensor(batch[bx_idx]) and batch[bx_idx].dim() == 3
                and batch[bx_idx].shape[-1] == 6):
            bx_idx = next((i for i, t in enumerate(batch)
                           if torch.is_tensor(t) and t.dim() == 3
                           and t.shape[-1] == 6), bx_idx)
        if os.environ.get("METEOR_DET_DEBUG") and bi == 0:
            print("[det3d-dbg] bx_idx=", bx_idx, "tmp_idx=", tmp_idx,
                  "shapes=", [tuple(t.shape) if torch.is_tensor(t) else "?"
                              for t in batch][:8], flush=True)
        bx = batch[bx_idx]
        nb = batch[bx_idx + 1]
        # The count tensor may be [B] or [B,1]. This used to treat dim()!=1 as
        # "wrong layout" and skip every batch, so detections were always 0 =
        # vehRn/P/yaw all displayed as 0 (2026-08-14, user report).
        # Flatten the shape and accept it.
        if torch.is_tensor(nb) and nb.dim() > 1:
            nb = nb.reshape(nb.shape[0], -1)[:, 0]
        nb = nb.clamp(min=0)
        pb, th = _temporal_inputs(model, batch, device, tmp_idx)
        with torch.autocast("cuda", torch.float16):
            out = model(imgs, K, Tc, None, pb, th) if th is not None \
                else model(imgs, K, Tc)
        dets = model.decode_boxes(out[3].float(), out[4].float(),
                                  thresh=thresh, topk=64)
        # stationary-flag accuracy at GT vehicle centres (r46 metric):
        # GT static = 3 s displacement <=0.35 m; 0.35-0.8 m dead-band
        traj_g = batch[bx_idx + 2] if len(batch) > bx_idx + 2 else None
        tval_g = batch[bx_idx + 3] if len(batch) > bx_idx + 3 else None
        if (len(out) >= 11 and torch.is_tensor(traj_g) and traj_g.dim() == 4
                and torch.is_tensor(tval_g) and tval_g.dim() == 3
                and tval_g.shape[1] == bx.shape[1]):
            stat_p = out[10][:, 0].float().sigmoid().cpu()
            from bevlane.model import DET_RES as _DR
            for b in range(bx.shape[0]):
                for k in range(int(nb[b])):
                    if bx[b, k, 3] <= 0 or tval_g[b, k, 5] < 0.5                             or bx[b, k, 0] >= 1.5:
                        continue
                    d3 = float(traj_g[b, k, 5].norm())
                    if 0.35 < d3 < 0.8:
                        continue
                    ri = int((80.0 - float(bx[b, k, 1])) / _DR)
                    ci = int((50.0 - float(bx[b, k, 2])) / _DR)
                    if not (0 <= ri < stat_p.shape[-2]
                            and 0 <= ci < stat_p.shape[-1]):
                        continue
                    score_s = float(stat_p[b, ri, ci])
                    pred_s = score_s > 0.5
                    gt_s = d3 <= 0.35
                    s_scores.append(score_s)
                    s_truth.append(gt_s)
                    _szi = _zone(float(bx[b, k, 1]))
                    if pred_s and gt_s:
                        s_tp += 1
                        if _szi is not None: s_zone[_szi][0] += 1
                    elif pred_s and not gt_s:
                        s_fp += 1
                        if _szi is not None: s_zone[_szi][1] += 1
                    elif not pred_s and gt_s:
                        s_fn += 1
                        if _szi is not None: s_zone[_szi][3] += 1
                    else:
                        s_tn += 1
                        if _szi is not None: s_zone[_szi][2] += 1
        for b in range(bx.shape[0]):
            gt = [(0 if bx[b, k, 0] < 1.5 else 1,
                   float(bx[b, k, 1]), float(bx[b, k, 2]),
                   float(bx[b, k, 3]), float(bx[b, k, 4]),
                   float(bx[b, k, 5]))
                  for k in range(int(nb[b])) if bx[b, k, 3] > 0]
            for gc, gx, _gy, _gl, _gw, _ga in gt:
                zi = _zone(gx)
                if gc == 0 and zi is not None:
                    z_gt[zi] += 1
            used = [False] * len(gt)
            for cls, sc, xe, ye, l, w, yaw in sorted(dets[b],
                                                     key=lambda d: -d[1]):
                best, bd = -1, match_m
                for gi, (gc, gx, gy, _gl, _gw, _gyaw) in enumerate(gt):
                    if used[gi] or gc != cls:
                        continue
                    d = ((gx - xe) ** 2 + (gy - ye) ** 2) ** 0.5
                    if d < bd:
                        best, bd = gi, d
                if best >= 0:
                    used[best] = True
                    tp[cls] += 1
                    cerr[cls] += bd
                    _, gx, gy, gl, gw, gyaw = gt[best]
                    lerr[cls] += abs(l - gl)
                    werr[cls] += abs(w - gw)
                    ce = _corner_mae((xe, ye, l, w, yaw),
                                     (gx, gy, gl, gw, gyaw))
                    corner_err[cls] += ce
                    dy = abs((yaw - gyaw + np.pi) % (2 * np.pi) - np.pi)
                    yflip[cls] += dy > np.pi / 2
                    yerr[cls] += min(dy, np.pi - dy)   # axis error
                    zi = _zone(gx)
                    if cls == 0 and zi is not None:
                        z_tp[zi] += 1
                        z_err[zi] += bd
                        z_lerr[zi] += abs(l - gl)
                        z_werr[zi] += abs(w - gw)
                        z_corner[zi] += ce
                        z_yerr[zi] += min(dy, np.pi - dy)
                    if gx * gx + gy * gy < 50.0 ** 2:
                        tp50[cls] += 1
                    if gx * gx + gy * gy < 30.0 ** 2 and abs(gy) < 12.0:
                        tpn[cls] += 1
                else:
                    fp[cls] += 1
            for gi, (gc, gx, gy, _gl, _gw, _gyaw) in enumerate(gt):
                if not used[gi]:
                    fn[gc] += 1
                    if gx * gx + gy * gy < 50.0 ** 2:
                        fn50[gc] += 1
                    if gx * gx + gy * gy < 30.0 ** 2 and abs(gy) < 12.0:
                        fnn[gc] += 1
    model.train()
    _reeval_frozen(model)
    r = {}
    for c, nm in ((0, "veh"), (1, "vru")):
        p_ = tp[c] / max(tp[c] + fp[c], 1)
        rc = tp[c] / max(tp[c] + fn[c], 1)
        r[nm] = (p_, rc, cerr[c] / max(tp[c], 1))
        r[nm + "_shape"] = (lerr[c] / max(tp[c], 1),
                             werr[c] / max(tp[c], 1),
                             corner_err[c] / max(tp[c], 1))
        r[nm + "_yaw"] = np.degrees(yerr[c] / max(tp[c], 1))
        r[nm + "_flip"] = yflip[c] / max(tp[c], 1)
        r[nm + "50"] = tp50[c] / max(tp50[c] + fn50[c], 1)
        r[nm + "n"] = tpn[c] / max(tpn[c] + fnn[c], 1)
    tot = s_tp + s_fp + s_tn + s_fn
    r["stat"] = (s_tp / max(s_tp + s_fp, 1), s_tp / max(s_tp + s_fn, 1),
                 (s_tp + s_tn) / max(tot, 1),
                 (s_tn) / max(s_tn + s_fp, 1), tot)
    r["stat_zones"] = tuple(
        (v[0] / max(v[0] + v[1], 1),
         v[0] / max(v[0] + v[3], 1),
         v[2] / max(v[2] + v[1], 1), sum(v),
         (v[0] + v[3]) / max(sum(v), 1))
        for v in s_zone)
    if s_scores and any(s_truth) and not all(s_truth):
        ss = np.asarray(s_scores, np.float64)
        st = np.asarray(s_truth, bool)
        vals = np.unique(ss)
        cuts = np.r_[vals[0] - 1e-6, (vals[:-1] + vals[1:]) / 2,
                     vals[-1] + 1e-6]
        def _sm(t):
            pred = ss > t
            _tp = int(np.sum(pred & st)); _fp = int(np.sum(pred & ~st))
            _tn = int(np.sum(~pred & ~st)); _fn = int(np.sum(~pred & st))
            _p = _tp / max(_tp + _fp, 1)
            _r = _tp / max(_tp + _fn, 1)
            _sp = _tn / max(_tn + _fp, 1)
            return (float(t), _p, _r, (_tp + _tn) / len(st), _sp,
                    (_r + _sp) / 2)
        trials = [_sm(t) for t in cuts]
        bal = max(trials, key=lambda x: (x[5], x[4], x[2]))
        safe_set = [x for x in trials if x[4] >= 0.95]
        safe = max(safe_set, key=lambda x: (x[2], x[5])) \
            if safe_set else None
        r["stat_cal"] = {"balanced": bal, "safe95": safe}
    r["veh_zones"] = tuple(
        (z_tp[i] / max(z_gt[i], 1), z_err[i] / max(z_tp[i], 1), z_gt[i],
         z_corner[i] / max(z_tp[i], 1), z_lerr[i] / max(z_tp[i], 1),
         z_werr[i] / max(z_tp[i], 1),
         np.degrees(z_yerr[i] / max(z_tp[i], 1)))
        for i in range(4))
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
        gt = _fit(gt, pred)
        pc, gc = pred == cls, gt == cls
        lab = gt > 0
        tp += (pc & gc).sum().item()
        fp += (pc & lab & ~gc).sum().item()
        fp += (pc & ~lab).sum().item()
        fn += (~pc & gc).sum().item()
        pcount += pc.sum().item()
        gcount += gc.sum().item()
    model.train()
    _reeval_frozen(model)
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
    ap.add_argument("--model", default="v1", choices=["v1", "v2", "v3s", "lss", "v8", "v13", "v13d", "v14d", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36", "v37", "v38", "v39", "v40", "v41", "v42", "v43", "v44", "v45", "v46", "v47", "v48", "v49", "v50", "v51", "v52", "v53", "v54", "v55", "v56", "v63b", "v64r50", "v52r50", "v52r50s8", "v52rvgg", "v55rvgg", "v52s8"])
    ap.add_argument("--freeze-depth", action="store_true",
                    help="train no depth-head parameter. The head is pulled by "
                         "25 downstream losses through the lift, and shifting "
                         "every depth by the same amount is very nearly a FLAT "
                         "direction of that objective -- measured: a +30 m bias "
                         "left the lift weight sharpness at 7.22 against 7.31 "
                         "and mIoU unchanged, because the lift normalises "
                         "across cameras, not along the ray. The only force "
                         "holding the metric scale is the depth L1 at an "
                         "effective weight of 0.06, and it loses: r51 and r52 "
                         "both drifted from 3.10 m to ~30 m MAE while every "
                         "other metric stayed healthy. Freeze it here and "
                         "re-distil the head on its own instead.")
    ap.add_argument("--depth-w", type=float, default=0.3)
    ap.add_argument("--seg2d-w", type=float, default=0.5)
    ap.add_argument("--seg2d-key", default="seg2d",
                    help="manifest key: seg2d (12cls) or seg2d21 (csv 21cls)")
    ap.add_argument("--n-seg2d", type=int, default=12,
                    help="2D seg head classes (21 with --seg2d-key seg2d21)")
    ap.add_argument("--gt-lidar-w", type=float, default=40.0,
                    help="LiDAR points a GT box needs for full weight. "
                         "0 disables. A box with no points is dropped "
                         "from the target entirely: measured on val, "
                         "13 %% of 60-80 m GT boxes have zero LiDAR "
                         "support, and evaluating only against boxes "
                         "with >=40 points lifts 40-60 m recall from "
                         "0.34 to 0.44. Training on unsupported boxes "
                         "teaches the detector to fire without "
                         "evidence.")
    ap.add_argument("--box-w", type=float, default=0.0,
                    help="BEV 3D-box occupancy multi-task loss weight (v15)")
    ap.add_argument("--bbox2d-w", type=float, default=0.0,
                    help="per-camera 10-class 2D bbox det loss weight (v17)")
    ap.add_argument("--n-cams", type=int, default=8,
                    help="cameras fed to the network. CAMS ends with "
                         "CAM_BACK_NARROW, so 7 drops exactly that one and "
                         "runs the backbone and depth tower 7/8 as often. "
                         "Different from --cam-drop, which ZEROES it and keeps "
                         "the compute.")
    ap.add_argument("--sparse-24", action="store_true",
                    help="2:4 structured-sparse TRAINING: prune once by "
                         "magnitude (groups of 4 along input channels, the "
                         "pattern TensorRT SPARSE_WEIGHTS accepted and timed "
                         "at -11%% workstation GPU / -9.5%% Orin), then keep the mask "
                         "fixed and re-apply after every optimiser step so "
                         "the surviving weights retrain. First conv is dense "
                         "automatically (in=3); named output layers stay "
                         "dense too.")
    ap.add_argument("--freeze-trunk", action="store_true",
                    help="freeze the shared part (backbone/FPN/depth/ctx/lift/temporal "
                         "fusion) and train only the heads. With no shared trainable "
                         "parameters, gradient interference between heads is "
                         "structurally zero (BN stats frozen too)")
    ap.add_argument("--det-head-only", action="store_true",
                    help="fine-tune only the 3D BBox head (backbone/lift/temporal fusion/"
                         "other heads all frozen, BN stats stopped). Shared features "
                         "do not move, so seg/E2E and other outputs are numerically unchanged = "
                         "a zero-risk way to test/ship a pose-only fix")
    ap.add_argument("--det-only", action="store_true",
                    help="3D BBox only (for isolation): zero all non-det losses "
                         "and freeze heads off the det path. DDP crashes on parameters "
                         "that receive no gradient, so zero weights and freezing go together. "
                         "The refiner emits hm/reg/seg/ego jointly, so "
                         "combine with METEOR_NOREF=1")
    ap.add_argument("--depth-ent-w", type=float, default=0.0,
                    help="entropy penalty on the depth distribution (sharpening). Measured: "
                         "the 30-60m band is nearly uniform (max prob 0.07), which is "
                         "the direct cause of the far-range 3D BBox recall ceiling")
    ap.add_argument("--depth-far-w", type=float, default=0.0,
                    help="depth CE weight for far pixels (0 = unchanged)")
    ap.add_argument("--lidar-distill-w", type=float, default=0.0,
                    help="LiDAR->camera modality distillation weight (0=off). "
                         "Teacher = no-grad pass with LiDAR forced ON, student = the "
                         "LiDAR-dropped rows of the main pass. L2 on fused BEV and det hm")
    ap.add_argument("--lidar-distill-every", type=int, default=4,
                    help="step interval for distillation (step-based so all ranks stay in sync)")
    ap.add_argument("--yaw-fix-deg", type=float, default=0.0,
                    help="correction angle [deg] for the inter-layer GT rotation. Rotates pose-derived "
                         "GT (BEV rasters, wp) about the ego origin (out/yawfix_plan.md)")
    ap.add_argument("--bn-guard", type=float, default=0.0,
                    help="in-loop conv->BN renorm when running_var exceeds "
                         "this (0=off). Function-preserving; 1e4 recommended")
    ap.add_argument("--det-sup-front", type=float, default=None,
                    help="det loss supervision window forward limit [m]")
    ap.add_argument("--det-sup-rear", type=float, default=None,
                    help="det loss supervision window rear limit [m]")
    ap.add_argument("--lat-min", type=float, default=0.3,
                    help="lat-aug magnitude lower bound [m]. C2 measured the "
                         "recovery skill weakest exactly below the training "
                         "range floor (0.5 m offsets recover 19-25 %% by 3 s "
                         "vs 51 %% at 1.0 m); 0.1 widens the taught range")
    ap.add_argument("--img-scale", type=int, default=1,
                    help="upsample input images (and K/bbox2d) by this "
                         "factor at load time -- R7 resolution axis")
    ap.add_argument("--seg-deep", type=int, default=0,
                    help="attach an N-block zero-init residual tower on the "
                         "BEV seg head (heads are cheap on Orin)")
    ap.add_argument("--det-deep", type=int, default=0,
                    help="attach an N-block zero-init residual tower on the "
                         "3D det trunk")
    ap.add_argument("--lane-branch", action="store_true",
                    help="dedicated residual decoder for the thin classes "
                         "(laneline/stopline/road_edge). Zero-init: attaches "
                         "to a trained checkpoint function-preservingly. The "
                         "Orin profile priced heads at ~18 ms total, so the "
                         "1-2 ms cost objection to a separate lane head is "
                         "gone; this tests the accuracy side.")
    ap.add_argument("--lane-sdf-w", type=float, default=0.0,
                    help="auxiliary lane signed-distance regression (probe). "
                         "L1 to the metre distance-to-nearest-laneline, "
                         "clipped at 2 m, supervised within a 3 m band. "
                         "Attacks the sub-cell label-jitter ceiling that CE "
                         "cannot (M5); measured levers on the loss side are "
                         "exhausted.")
    ap.add_argument("--ema", type=float, default=0.0,
                    help="EMA decay for the trainable weights (0 = off). The "
                         "averaged weights are evaluated next to the raw ones "
                         "each epoch, never assumed better.")
    ap.add_argument("--ego-ce-tau", type=float, default=0.0,
                    help="soft selector target: CE against "
                         "softmax(-candidate_error/tau) instead of the hard "
                         "argmin. 0 keeps the hard target. The argmin is a "
                         "coin flip on 44%% of frames (median best-vs-second "
                         "gap 0.231 m), which is why the selector picks the "
                         "best mode on only 37%% against a 33%% baseline.")
    ap.add_argument("--ego-fde-w", type=float, default=0.3,
                    help="extra weight on the +3.0 s waypoint of the "
                         "SELECTED candidate, added on top of the "
                         "per-step loss. Re-weighting the profile "
                         "instead (r56, flat EGO_TW) cost 18 % of ADE "
                         "and ADEc; this adds supervision without "
                         "removing any.")
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
    ap.add_argument("--unk-dense-w", type=float, default=0.0)
    ap.add_argument("--intent-drop", type=float, default=0.3,
                    help="fraction of samples whose driving command is zeroed "
                         "(modality dropout). The selector is evaluated with "
                         "NO command, so this is what trains it to infer the "
                         "manoeuvre instead of copying the command.")
    ap.add_argument("--intent-w", type=float, default=0.0,
                    help="v43 command-consistency hinge weight")
    ap.add_argument("--paint-seg", default="",
                    help="PointPainting: add the given seg2d class probabilities to the pre-lift "
                         "ctx via a zero-initialized 1x1 projection (e.g. 2,3,4,5,6,7)")
    ap.add_argument("--ego-conv-pool", default="",
                    help="replace the ego global average pooling with an INT8-tolerant "
                         "convolution. Value is the JSON written by make_ego_pool_stats.py. "
                         "The normalization is folded in with the stats and cancelled in ego_mlp, "
                         "so the output right after conversion is unchanged (function-preserving)")
    ap.add_argument("--pact", default="",
                    help="prefixes of layers to replace with clipped ReLU (PACT) "
                         "(e.g. tfuse,ego_stem). alpha is initialized from the measured max, "
                         "so it is function-preserving at insertion")
    ap.add_argument("--pact-w", type=float, default=0.0,
                    help="L1 on the PACT alphas. Pushes the clip down to prune outliers")
    ap.add_argument("--pact-lr", type=float, default=0.02,
                    help="dedicated lr for the PACT alphas. alpha must come down from the measured "
                         "max (30-130) to a few times p99.9 (10-15), which the base lr "
                         "(1e-4) cannot reach even with Adam, hence a separate group")
    ap.add_argument("--pact-alpha-init", default="out/pact_alpha_init.json",
                    help="JSON of layer name -> initial alpha (measured max x1.10)")
    ap.add_argument("--dense-teacher", default="",
                    help="dense teacher ckpt (e.g. out/ckpt_v151/best_e2e.pt). During sparse fine-tuning, match fused BEV and E2E outputs to the teacher (2026-09-07)")
    ap.add_argument("--dense-distill-w", type=float, default=0.0, help="dense-teacher distillation weight (0=off)")
    ap.add_argument("--dense-distill-ego-w", type=float, default=1.0, help="relative weight of the E2E output (ego) within the distillation")
    ap.add_argument("--dense-distill-every", type=int, default=1, help="run the teacher pass every N steps")
    ap.add_argument("--ema-exclude", default="",
                    help="parameter prefixes excluded from the EMA (comma-separated, e.g. seg_head.)")
    ap.add_argument("--sparse-ramp-steps", type=int, default=0,
                    help="ramp into 2:4: train the first N steps at 1:4 (only the smallest of each 4 zeroed), "
                         "then switch to 2:4 after N steps (recovers better than one-shot pruning; v157, 2026-09-08)")
    ap.add_argument("--sparse-exclude", default="",
                    help="module prefixes kept dense under --sparse-24 (comma-separated, e.g. ego_,traj_,tfuse3,tgate,sem_ego,delta_stat,refiner.e2e)")
    ap.add_argument("--hist-lr-mult", type=float, default=1.0,
                    help="lr multiplier for the history modules (tfuse3/tgate/traj_stem/delta_stat/ego family)")
    ap.add_argument("--val-hs", type=int, default=240,
                    help="at epoch end, separately evaluate up to N high-speed (v0>=8) val frames and "
                         "use the wp0 longitudinal bias for best_chain selection (0=off)")
    ap.add_argument("--depth-slim-force", action="store_true",
                    help="keep the requested --depth-slim width even if it differs from the init's "
                         "depth head width, initializing from the init's leading channels")
    ap.add_argument("--depth-band-balance", type=float, default=0.0,
                    help="weight the depth CE by inverse frequency of 10m bands (1.0 = fully balanced). "
                         "Valid depth GT pixels are 58.9%% at 0-10m but only 4.6%% at 40-60m, "
                         "so --depth-far-w gives at most 2x")
    ap.add_argument("--paint-det", default="",
                    help="add the given 2D detection heatmap (hm2d) class probabilities "
                         "to the pre-lift ctx via a zero-initialized 1x1 projection "
                         "(e.g. 0,1,2). Delivers far objects, where depth sharpening has saturated, "
                         "to the BEV as evidence seen in 2D")
    ap.add_argument("--offroad-w", type=float, default=0.0,
                    help="penalty for the commanded path leaving the drivable surface "
                         "(distance field x worst 2 points x selected mode only)")
    ap.add_argument("--intent-mode-w", type=float, default=0.0,
                    help="v44 command->mode CE weight (raw logits)")
    ap.add_argument("--lat-aug", type=float, default=0.0,
                    help="v45 recovery aug: max lateral ego offset [m]")
    ap.add_argument("--lat-p", type=float, default=0.25,
                    help="fraction of samples given a lateral offset")
    ap.add_argument("--quant-noise", type=float, default=0.0,
                    help="v45 INT8-robust feature noise (1.0 = 1 LSB)")
    ap.add_argument("--use-sdmap", action="store_true",
                    help="v46: feed the OSM SD-map raster (optional input)")
    ap.add_argument("--val-scenes-file", default=None,
                    help="override the val scenes with a fixed list (for holdout evaluation)")
    ap.add_argument("--val-batch", type=int, default=0,
                    help="batch for the val loaders (default: max(train batch, 2)); evaluation is no-grad, so it should not shrink with the training batch -- at batch 1 the capped evals covered half the samples and the rare curve subset (ADEc) hit zero -> nan")
    ap.add_argument("--sync-bn", action="store_true",
                    help="SyncBatchNorm: needed when the batch per GPU drops to 1")
    ap.add_argument("--grad-ckpt", action="store_true",
                    help="checkpoint the image backbone: ~15%% slower, frees several GB of activations")
    ap.add_argument("--intent-margin", type=float, default=1.5,
                    help="metres the commanded mode must reach in the "
                         "commanded direction before the direction hinge goes "
                         "silent; set from the acceptance bar (>= half the "
                         "required command spread), not left at the default")
    ap.add_argument("--intent-wrong", type=float, default=0.0,
                    help="fraction of commanded rows given a "
                         "DELIBERATELY wrong command; their "
                         "waypoint supervision is dropped so only "
                         "intent_loss shapes them (phase 2 of "
                         "docs/FIX_COMMAND_BINDING.md, off by "
                         "default)")
    ap.add_argument("--rl-w", type=float, default=0.0,
                    help="r48: weight of the GRPO-style reward "
                         "loss on the E2E mode logits (0 = off)")
    ap.add_argument("--rl-ent", type=float, default=0.01,
                    help="entropy bonus in the RL mode loss")
    ap.add_argument("--rl-imit-w", type=float, default=0.5,
                    help="imitation-error term inside the reward")
    ap.add_argument("--rl-tl-w", type=float, default=1.0,
                    help="red-light compliance term in the reward")
    ap.add_argument("--pseudo-lidar-w", type=float, default=0.0,
                    help="v48: weight of the pseudo-LiDAR "
                         "distillation loss (0 = head off)")
    ap.add_argument("--pl-feed-p", type=float, default=0.5,
                    help="v48: fraction of LiDAR-less samples "
                         "that get the predicted raster fed back")
    ap.add_argument("--stat-w", type=float, default=None,
                    help="stationary-flag loss weight (default: traj-w)")
    ap.add_argument("--stat-margin", type=float, default=0.0,
                    help="extra signed-logit margin for stationary/moving "
                         "cells. A positive value keeps logits away from the "
                         "zero decision boundary and improves INT8 robustness")
    ap.add_argument("--stat-quant-head", type=float, default=0.0,
                    help="replace stat_head2 by an equivalent +/- branch "
                         "with this bounded logit cap (e.g. 8); creates a "
                         "stable head-specific INT8 activation range")
    ap.add_argument("--bev-dropblock", type=float, default=0.0,
                    help="MAE-like BEV block-mask prob per sample")
    ap.add_argument("--bev-wedgedrop", type=float, default=0.0,
                    help="per-sample probability of zeroing a BEV angular sector (wedge). "
                         "Simulates losing one camera's field of view at feature level")
    ap.add_argument("--bev-ringdrop", type=float, default=0.0,
                    help="per-sample probability of zeroing a BEV range ring")
    ap.add_argument("--bev-chandrop", type=float, default=0.0,
                    help="BEV channel drop rate (SpatialDropout-style)")
    ap.add_argument("--use-tl", action="store_true",
                    help="v47: per-camera box-level traffic-light input")
    ap.add_argument("--tl-drop", type=float, default=0.5,
                    help="prob of dropping the TL input per sample")
    ap.add_argument("--sdmap-drop", type=float, default=0.5,
                    help="whole-sample SD-map dropout (modality dropout)")
    ap.add_argument("--cam-drop", type=float, default=0.0,
                    help="probability of feeding CAM_BACK_NARROW as zeros on "
                         "an 8-camera sample, so one set of weights stays "
                         "calibrated for both the 8- and the 7-camera rig "
                         "(x2gen2 has no CAM_BACK_NARROW)")
    ap.add_argument("--zero-cams", default="",
                    help="comma-separated camera names to hard-zero "
                         "(J6 7-cam fine-tune: CAM_BACK_NARROW)")
    ap.add_argument("--unk-key", default="unknown_v2",
                    help="dense unknown GT key: unknown_v2 or unknown_v3 "
                         "(camera-visibility-filtered, occluded=don't-care)")
    ap.add_argument("--bev-rot-aug", type=float, default=0.0,
                    help="BEV-frame rotation augmentation: max |yaw| in "
                    "degrees rotated into the extrinsics + all BEV GT "
                    "(images and camera-space heads untouched)")
    ap.add_argument("--x2-oversample", type=float, default=1.0,
                    help="sampling weight for the x2gen2 (7-camera) scenes")
    ap.add_argument("--farveh-oversample", type=float, default=1.0,
                    help="D2': sampling weight for far-vehicle-rich scenes (out/farveh_scenes.txt). "
                         "Targets 40-80m recall")
    ap.add_argument("--tversky-area-w", type=float, default=0.0,
                    help="S2: tversky FP penalty on road(1)/crosswalk(3)")
    ap.add_argument("--vru-cw", type=float, default=5.0,
                    help="V lever: VRU class weight in the det heatmap (default 5.0)")
    ap.add_argument("--okinawa-oversample", type=float, default=1.0,
                    help="weight for Okinawa scenes (out/okinawa_train_scenes.txt)")
    ap.add_argument("--vru-oversample", type=float, default=1.0,
                    help="V lever: weight for VRU-rich scenes (out/vru_scenes.txt)")
    ap.add_argument("--cosmos-no-ego", action="store_true",
                    help="v131 design: exclude cosmos3_* frames from the ego (E2E) loss "
                         "(perception-only training). Counter to the v126 ADEc +0.027")
    ap.add_argument("--cosmos-oversample", type=float, default=1.0,
                    help="sampling weight for registered cosmos3_* weather/"
                         "lighting transfer scenes")
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
    ap.add_argument("--seg-ignore-unlabeled", action="store_true",
                    help="treat BEV-seg class 0 (black/unlabeled) as strict "
                         "don't-care: exclude it from boundary weighting so "
                         "the moving rim of the observed area is not trained "
                         "(black is already ignored by CE/dice/tversky/lovasz)")
    ap.add_argument("--lovasz-w", type=float, default=0.0,
                    help="Lovasz-Softmax loss weight (IoU-direct, sharp edges)")
    ap.add_argument("--tversky-w", type=float, default=0.0,
                    help="Tversky loss weight on line classes (FP-heavy -> thin)")
    ap.add_argument("--cldice-w", type=float, default=0.0,
                    help="soft-clDice weight for laneline connectivity")
    ap.add_argument("--seg-w", type=float, default=1.0)
    ap.add_argument("--init-ckpt", default="")
    ap.add_argument("--depth-bins", type=int, default=64,
                    help="number of depth head bins (Orin 48bin lever. Anything but 64 "
                         "drops the init's depth head and retrains it)")
    ap.add_argument("--depth-slim", type=float, default=0.0,
                    help="depth head width shrink factor (0=off, e.g. 0.75). "
                         "Replaced by a freshly initialized head after loading init-ckpt")
    ap.add_argument("--train-list", default="",
                    help="file of scene names to restrict training to")
    ap.add_argument("--min-cov-core", type=float, default=0.03,
                    help="min labeled frac +-30m band (0=off)")
    ap.add_argument("--min-cov-fwd", type=float, default=0.005,
                    help="min labeled frac +30..80m band (drops stationary)")
    ap.add_argument("--trim-start", type=int, default=3,
                    help="drop first K frames/scene (weak rear GT at scene start)")
    ap.add_argument("--trim-end", type=int, default=10,
                    help="drop last K frames/scene (weak forward GT at scene end)")
    ap.add_argument("--train-bg", action="store_true",
                    help="supervise unlabeled(0) as background class")
    ap.add_argument("--ego-speed-w", type=float, default=0.0,
                    help="ego loss speed weight 1+v0/this (cap 4). Keeps the few high-speed "
                         "straight frames from being ignored by the L1 median (v139b)")
    ap.add_argument("--freeze-ego", action="store_true",
                    help="freeze the E2E (ego) head: clear requires_grad on ego_stem/ego_mlp/ego_attn/ego_delta/"
                         "sem_ego/kin_delta/dec_head/refiner.e2e etc. and pin BN to eval. "
                         "Protects chain divergence in perception-lever rounds (v142)")
    ap.add_argument("--kin-anchor", action="store_true",
                    help="v139: add g_t*[v0*t,0] to the ego waypoints (zero-initialized)")
    ap.add_argument("--semantic-ego", action="store_true",
                    help="add a zero-initialized residual to ego that reads the semantic outputs (seg/det). "
                         "Learns a path that reads only INT8-healthy tensors, aiming to drop "
                         "tfuse/ego from fp16-keep for plain INT8 (99ms)")
    ap.add_argument("--traj-flow", action="store_true",
                    help="A1: connect the flow field to the traj head input via a zero-initialized residual"
                         " (fed detached, stat input unchanged)")
    ap.add_argument("--depth-log-bins", action="store_true",
                    help="D5: log-spaced depth bins (preserves relative resolution at range)")
    ap.add_argument("--mode-scorer", action="store_true",
                    help="E3: route-conditioned mode selection scorer (zero-initialized)")
    ap.add_argument("--det-temporal", action="store_true",
                    help="D7: zero-initialized temporal-feature residual into the det hm")
    ap.add_argument("--traj-cv", action="store_true",
                    help="A6: reparameterize other-agent trajectories as CV(v̂)+residual")
    ap.add_argument("--graft-lr-mult", type=float, default=1.0,
                    help="dedicated lr multiplier for the det_tmp/traj_vel residuals (for probes)")
    ap.add_argument("--traj-flow-lr-mult", type=float, default=1.0,
                    help="A1b: dedicated lr multiplier for the traj_flow residual (to grow the "
                         "zero init within a short probe)")
    ap.add_argument("--delta-stat", action="store_true",
                    help="replace the stop decision with a new head driven by the temporal difference "
                         "|bev - warp(prev)| (root fix for stat collapsing under INT8)")
    ap.add_argument("--gt-valid", action="store_true",
                    help="ignore cells outside per-frame LiDAR-observed mask")
    ap.add_argument("--box-corner-w", type=float, default=0.0,
                    help="metric 4-corner geometry loss (train-only)")
    ap.add_argument("--dontcare-sidewalk", action="store_true")
    ap.add_argument("--gt-key", default="gt", choices=["gt", "gt_vec", "gt_cons"],
                    help="gt = raster autolabel; gt_vec = hybrid vector-line GT")
    args = ap.parse_args()
    if getattr(args, "det_head_only", False):
        args.det_only = True          # losses are dropped the same way as det-only
    if getattr(args, "det_only", False):
        for _w in ("seg_w", "dice_w", "lovasz_w", "boundary_w", "tversky_w",
                   "depth_w", "seg2d_w", "bbox2d_w", "ego_w", "occ_w",
                   "traj_w", "tl_w", "risk_w", "lanegraph_w", "flow_w",
                   "unk_w", "unk_dense_w", "pseudo_lidar_w", "stat_w",
                   "intent_w", "intent_mode_w", "lane_sdf_w", "rl_w",
                   "ego_fde_w"):
            if hasattr(args, _w):
                setattr(args, _w, 0.0)
        args.pl_feed_p = 0.0

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
    if args.val_scenes_file:
        # Holdout evaluation: override val with a fixed list (existing scenes only).
        # Also excluded from train (harmless if it overlaps HOLDOUT_FILES).
        want = set(open(args.val_scenes_file).read().split())
        val_s = sorted(want & set(os.listdir(args.root)))
        train_s = [s for s in train_s if s not in want]
        print(f"[val-override] {args.val_scenes_file}: "
              f"{len(val_s)} scenes", flush=True)
    use_depth = args.model in ("lss", "v8", "v13", "v13d", "v14d", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36", "v37", "v38", "v39", "v40", "v41", "v42", "v43", "v44", "v45", "v46", "v47", "v48", "v49", "v51", "v52", "v53", "v54", "v55", "v56", "v63b", "v64r50", "v52r50", "v52r50s8", "v52rvgg", "v55rvgg", "v52s8") and args.depth_w > 0
    use_seg2d = args.model in ("v13", "v13d", "v14d", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36", "v37", "v38", "v39", "v40", "v41", "v42", "v43", "v44", "v45", "v46", "v47", "v48", "v49", "v51", "v52", "v53", "v54", "v55", "v56", "v63b", "v64r50", "v52r50", "v52r50s8", "v52rvgg", "v55rvgg", "v52s8") and args.seg2d_w > 0
    use_box = args.model == "v15" and args.box_w > 0
    use_boxdet = args.model in ("v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36", "v37", "v38", "v39", "v40", "v41", "v42", "v43", "v44", "v45", "v46", "v47", "v48", "v49", "v51", "v52", "v53", "v54", "v55", "v56", "v63b", "v64r50", "v52r50", "v52r50s8", "v52rvgg", "v55rvgg", "v52s8") and args.box_w > 0
    use_bbox2d = args.model in ("v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36", "v37", "v38", "v39", "v40", "v41", "v42", "v43", "v44", "v45", "v46", "v47", "v48", "v49", "v51", "v52", "v53", "v54", "v55", "v56", "v63b", "v64r50", "v52r50", "v52r50s8", "v52rvgg", "v55rvgg", "v52s8") and args.bbox2d_w > 0
    use_ego = args.model in ("v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36", "v37", "v38", "v39", "v40", "v41", "v42", "v43", "v44", "v45", "v46", "v47", "v48", "v49", "v51", "v52", "v53", "v54", "v55", "v56", "v63b", "v64r50", "v52r50", "v52r50s8", "v52rvgg", "v55rvgg", "v52s8") and args.ego_w > 0
    use_occ = args.model in ("v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36", "v37", "v38", "v39", "v40", "v41", "v42", "v43", "v44", "v45", "v46", "v47", "v48", "v49", "v51", "v52", "v53", "v54", "v55", "v56", "v63b", "v64r50", "v52r50", "v52r50s8", "v52rvgg", "v55rvgg", "v52s8") and args.occ_w > 0
    use_traj = args.model in ("v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36", "v37", "v38", "v39", "v40", "v41", "v42", "v43", "v44", "v45", "v46", "v47", "v48", "v49", "v51", "v52", "v53", "v54", "v55", "v56", "v63b", "v64r50", "v52r50", "v52r50s8", "v52rvgg", "v55rvgg", "v52s8") and args.traj_w > 0
    use_temporal = args.model in ("v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36", "v37", "v38", "v39", "v40", "v41", "v42", "v43", "v44", "v45", "v46", "v47", "v48", "v49", "v51", "v52", "v53", "v54", "v55", "v56", "v63b", "v64r50", "v52r50", "v52r50s8", "v52rvgg", "v55rvgg", "v52s8")
    use_tl = args.model in ("v27", "v28", "v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36", "v37", "v38", "v39", "v40", "v41", "v42", "v43", "v44", "v45", "v46", "v47", "v48", "v49", "v51", "v52", "v53", "v54", "v55", "v56", "v63b", "v64r50", "v52r50", "v52r50s8", "v52rvgg", "v55rvgg", "v52s8") and args.tl_w > 0
    use_risk = args.model in ("v28", "v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36", "v37", "v38", "v39", "v40", "v41", "v42", "v43", "v44", "v45", "v46", "v47", "v48", "v49", "v51", "v52", "v53", "v54", "v55", "v56", "v63b", "v64r50", "v52r50", "v52r50s8", "v52rvgg", "v55rvgg", "v52s8") and args.risk_w > 0
    use_lg = args.model in ("v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36", "v37", "v38", "v39", "v40", "v41", "v42", "v43", "v44", "v45", "v46", "v47", "v48", "v49", "v51", "v52", "v53", "v54", "v55", "v56", "v63b", "v64r50", "v52r50", "v52r50s8", "v52rvgg", "v55rvgg", "v52s8") and args.lanegraph_w > 0
    use_unk = args.model in ("v30", "v31", "v32", "v33", "v34", "v35", "v36", "v37", "v38", "v39", "v40", "v41", "v42", "v43", "v44", "v45", "v46", "v47", "v48", "v49", "v51", "v52", "v53", "v54", "v55", "v56", "v63b", "v64r50", "v52r50", "v52r50s8", "v52rvgg", "v55rvgg", "v52s8") and args.unk_w > 0
    use_unk_v2 = args.model in ("v41", "v42", "v43", "v44", "v45", "v46", "v47", "v48", "v49", "v51", "v52", "v53", "v54", "v55", "v56", "v63b", "v64r50", "v52r50", "v52r50s8", "v52rvgg", "v55rvgg", "v52s8") and args.unk_dense_w > 0
    use_sdmap = args.model in ("v46", "v47", "v48", "v49", "v51", "v52", "v53", "v54", "v55", "v56", "v63b", "v64r50", "v52r50", "v52r50s8", "v52rvgg", "v55rvgg", "v52s8") and args.use_sdmap
    use_tlin = args.model in ("v47", "v48", "v49", "v51", "v52", "v53", "v54", "v55", "v56", "v63b", "v64r50", "v52r50", "v52r50s8", "v52rvgg", "v55rvgg", "v52s8") and args.use_tl
    # every version FROM v48 on has the pseudo-LiDAR head. An equality test
    # here silently disabled the loss for v49, and pl_head then received no
    # gradient at all -- which DDP reports as "parameters that were not used in
    # producing loss" and kills the round before step 1.
    use_pl = args.model in ("v48", "v49", "v51", "v52", "v53", "v54", "v55", "v56", "v63b", "v64r50", "v52r50", "v52r50s8", "v52rvgg", "v55rvgg", "v52s8") and args.pseudo_lidar_w > 0
    use_rl = args.rl_w > 0 and use_ego
    # v31 reuses the depth4 GT tensor as the (train-time) LiDAR input
    use_lidar = args.model in ("v31", "v32", "v33", "v34", "v35", "v36", "v37", "v38", "v39", "v40", "v41", "v42", "v43", "v44", "v45", "v46", "v47", "v48", "v49", "v51", "v52", "v53", "v54", "v55", "v56", "v63b", "v64r50", "v52r50", "v52r50s8", "v52rvgg", "v55rvgg", "v52s8")
    # v32 additionally takes the pillar BEV raster (extract_lidar_bev.py)
    use_lidarbev = args.model in ("v32", "v33", "v34", "v35", "v36", "v37", "v38", "v39", "v40", "v41", "v42", "v43", "v44", "v45", "v46", "v47", "v48", "v49", "v51", "v52", "v53", "v54", "v55", "v56", "v63b", "v64r50", "v52r50", "v52r50s8", "v52rvgg", "v55rvgg", "v52s8")
    use_flow = args.model in ("v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36", "v37", "v38", "v39", "v40", "v41", "v42", "v43", "v44", "v45", "v46", "v47", "v48", "v49", "v51", "v52", "v53", "v54", "v55", "v56", "v63b", "v64r50", "v52r50", "v52r50s8", "v52rvgg", "v55rvgg", "v52s8") and args.flow_w > 0
    hist_n = 3 if args.model in ("v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36", "v37", "v38", "v39", "v40", "v41", "v42", "v43", "v44", "v45", "v46", "v47", "v48", "v49", "v51", "v52", "v53", "v54", "v55", "v56", "v63b", "v64r50", "v52r50", "v52r50s8", "v52rvgg", "v55rvgg", "v52s8") else 0
    if args.train_list:                       # restrict train to a scene list
        keep = set(open(args.train_list).read().split())
        # holdouts are already gone from train_s, but a --train-list could name
        # them explicitly: intersect, never union, and say so out loud
        _ho = holdout_scenes(args.root) & keep
        if _ho:
            print(f"[holdout] --train-list names {len(_ho)} held-out scenes; "
                  f"they stay excluded", flush=True)
        train_s = [s for s in train_s if s in keep]
    assert not (set(train_s) & holdout_scenes(args.root)), \
        "held-out scenes reached the training set"
    # v13d depth GT is stride-4 of 768 (108x192); resize any mixed-res depth
    depth_hw = (108, 192) if args.model in ("v13d", "v14d", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36", "v37", "v38", "v39", "v40", "v41", "v42", "v43", "v44", "v45", "v46", "v47", "v48", "v49", "v51", "v52", "v53", "v54", "v55", "v56", "v63b", "v64r50", "v52r50", "v52r50s8", "v52rvgg", "v55rvgg", "v52s8") else None
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
                        with_unknown_v2=use_unk_v2, unk2_key=args.unk_key,
                        with_sdmap=use_sdmap, with_tlin=use_tlin,
                        cam_drop=args.cam_drop, img_scale=args.img_scale,
                        trim_start=args.trim_start, trim_end=args.trim_end,
                        n_cams=args.n_cams,
                        min_cov_core=args.min_cov_core,
                        min_cov_fwd=args.min_cov_fwd,
                        seg2d_key=args.seg2d_key,
                        yaw_fix_deg=args.yaw_fix_deg,
                        use_gt_valid=args.gt_valid,
                        ego_mask_prefix=("cosmos3_" if args.cosmos_no_ego
                                         else None))
    # NOTE: --limit-train no longer slices the dataset here; it is applied
    # per epoch by EpochSubsetSampler so each epoch sees fresh frames.
    if True:   # all ranks: the distributed ADE probe shards the val set
        # val never needs depth GT (BEV mIoU eval only) -> with_depth=False
        va = BevLaneDataset(args.root, val_s, max_per_scene=8, gt_key=args.gt_key,
                            img_scale=args.img_scale,
                            dontcare_sidewalk=args.dontcare_sidewalk,
                            with_depth=False, with_seg2d=use_seg2d,
                            seg2d_key=args.seg2d_key, with_ego=use_ego,
                            with_occ=use_occ, with_agenttraj=use_traj,
                            with_temporal=use_temporal, with_tl=use_tl,
                            # Boxes normally arrive via agenttraj, so configs with
                            # the traj loss off (--det-only etc.) lost the boxes
                            # from val and every 3D metric read 0
                            # (2026-08-14, user report). Request them explicitly.
                            with_boxdet=(use_boxdet and not use_traj),
                            with_risk=use_risk, with_lanegraph=use_lg,
                            temporal_hist=hist_n, with_unknown=use_unk,
                            with_lidarbev=use_lidarbev,
                            with_unknown_v2=use_unk_v2,
                            unk2_key=args.unk_key,
                            with_sdmap=use_sdmap, with_tlin=use_tlin,
                            trim_start=3, trim_end=args.trim_end,
                            n_cams=args.n_cams,
                            yaw_fix_deg=args.yaw_fix_deg,
                            use_gt_valid=args.gt_valid)
        seen = min(args.limit_train or len(tr), len(tr)) * args.epochs
        print(f"train {len(tr)} samples / {len(train_s)} scenes; "
              f"val {len(va)} samples / {len(val_s)} scenes; "
              f"world={world} lr={lr:.1e}", flush=True)
        print(f"[data] {args.limit_train or len(tr)} fresh samples/epoch x "
              f"{args.epochs} ep = {seen} draws over {len(tr)} frames "
              f"({100 * min(seen, len(tr)) / max(len(tr), 1):.0f}% expected "
              f"coverage)", flush=True)
        vb = args.val_batch or max(args.batch, 2)
        # Every evaluator caps BATCHES, not samples, so halving the val batch
        # silently halved the val coverage: [val ep*] went from 160 samples
        # (20 scenes) to 80 (10 scenes) when --val-batch 1 was introduced for
        # batch 2, and rare classes started reading 0.000 -- parking looked
        # like a regression (0.318 -> 0.000) while a direct measurement on the
        # same weights showed it IMPROVING (IoU 0.101 -> 0.168). Scale the caps
        # so the sample count, and therefore comparability with r47, is fixed.
        vcap = lambda n: max(1, int(round(n * 2.0 / vb)))   # noqa: E731
        dv = DataLoader(va, batch_size=vb, shuffle=False,
                        num_workers=4 if is_main else 1, pin_memory=is_main)
        # Epoch-end val used to read the FIRST `max_batches` batches of `dv`,
        # i.e. the head of the list: 10 of 270 scenes at val_batch 2, and only
        # 5 (with ZERO turn frames, hence ADEc=nan) at val_batch 1. Every
        # epoch-end number therefore described a handful of scenes. dv_ep walks
        # the same samples with a STRIDE so the identical evaluation budget
        # spans all 270 val scenes; deterministic, so epochs stay comparable.
        # dv itself is left alone: the in-epoch probe does its own striding and
        # shards across ranks.
        _st = max(1, len(va) // max(80 * 2 // max(vb, 1), 1))
        va_ep = torch.utils.data.Subset(va, list(range(0, len(va), _st)))
        dv_ep = DataLoader(va_ep, batch_size=vb, shuffle=False,
                           num_workers=4 if is_main else 1, pin_memory=is_main)
        if is_main:
            _sc = {va.items[i][0] for i in va_ep.indices}
            print(f"[val] epoch-end slice: {len(va_ep)} samples over "
                  f"{len(_sc)} of {len(val_s)} scenes (stride {_st})",
                  flush=True)
        # High-speed val (2026-09-05, lesson from the v145 call): the epoch-end
        # slice has only ~19 frames with v0>=8 m/s, so the high-speed longitudinal
        # bias that dominates chain divergence (v145: −1.13 m) showed as −0.02 in val.
        # Gather up to --val-hs frames with v0>=8 evenly from the whole val set,
        # measure them as a separate slice, and use this bias for best_chain selection.
        dv_hs = None
        if use_ego and args.val_hs > 0:
            _v0c = {}
            _hs_idx = []
            for _i, (_s, _f) in enumerate(va.items):
                if _s not in _v0c:
                    try:
                        _v0c[_s] = np.load(os.path.join(args.root, _s,
                                                        "ego_motion.npz"))["v0"]
                    except Exception:
                        _v0c[_s] = None
                _v = _v0c[_s]
                if _v is not None and _f["frame"] < len(_v) \
                        and float(_v[_f["frame"]]) >= 8.0:
                    _hs_idx.append(_i)
            if len(_hs_idx) > args.val_hs:
                _hs_idx = [_hs_idx[int(k)] for k in
                           np.linspace(0, len(_hs_idx) - 1, args.val_hs)]
            if _hs_idx:
                dv_hs = DataLoader(torch.utils.data.Subset(va, _hs_idx),
                                   batch_size=vb, shuffle=False,
                                   num_workers=2 if is_main else 1,
                                   pin_memory=is_main)
            if is_main:
                print(f"[val] high-speed slice: {len(_hs_idx)} frames "
                      f"(v0>=8 m/s) for wp0 bias", flush=True)
        dv_lid = None
        if use_lidar:
            # separate minimal loader (imgs,K,T,gt,depth4) so the main val
            # batch layout (indexed positionally everywhere) is untouched
            va_lid = BevLaneDataset(args.root, val_s, max_per_scene=8,
                                    gt_key=args.gt_key, with_depth=True,
                                    img_scale=args.img_scale,
                                    with_lidarbev=use_lidarbev,
                                    dontcare_sidewalk=args.dontcare_sidewalk,
                                    trim_start=3, trim_end=args.trim_end,
                                    n_cams=args.n_cams,
                                    use_gt_valid=args.gt_valid)
            dv_lid = DataLoader(va_lid, batch_size=vb, shuffle=False,
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
        if args.x2_oversample > 1.0 and os.path.exists("out/x2gen2_train.txt"):
            # x2gen2 is 2.7 % of the frames (26,540 / 977,420) and scores 0.263
            # against 0.334 on the Japanese rig: too little of it on its own to
            # close the domain gap.
            x2 = set(open("out/x2gen2_train.txt").read().split())
            if weights is None:
                weights = torch.ones(len(tr))
            n_x = 0
            for ii, (s_, _f) in enumerate(tr.items):
                if s_ in x2:
                    weights[ii] = max(float(weights[ii]), args.x2_oversample)
                    n_x += 1
            if is_main:
                print(f"[data] x2gen2 oversample x{args.x2_oversample}: "
                      f"{n_x}/{len(tr)} frames ({len(x2)} scenes)", flush=True)
        if args.okinawa_oversample > 1.0 and \
                os.path.exists("out/okinawa_train_scenes.txt"):
            # Okinawa domain (2026-09-01): holdout ADEc 1.18 vs mainland 0.49.
            # 496 scenes / 11.6k = 4.3% exposure is not enough -> boost the weight.
            oki = set(open("out/okinawa_train_scenes.txt").read().split())
            if weights is None:
                weights = torch.ones(len(tr))
            n_o = 0
            for ii, (s_, _f) in enumerate(tr.items):
                if s_ in oki:
                    weights[ii] = max(float(weights[ii]),
                                      args.okinawa_oversample)
                    n_o += 1
            if is_main:
                print(f"[data] okinawa oversample x{args.okinawa_oversample}: "
                      f"{n_o}/{len(tr)} frames ({len(oki)} scenes)", flush=True)
        if args.vru_oversample > 1.0 and \
                os.path.exists("out/vru_scenes.txt"):
            vs = set(open("out/vru_scenes.txt").read().split())
            if weights is None:
                weights = torch.ones(len(tr))
            n_vs = 0
            for ii, (s_, _f) in enumerate(tr.items):
                if s_ in vs:
                    weights[ii] = max(float(weights[ii]),
                                      args.vru_oversample)
                    n_vs += 1
            if is_main:
                print(f"[data] vru oversample x{args.vru_oversample}: "
                      f"{n_vs}/{len(tr)} frames ({len(vs)} scenes)",
                      flush=True)
        if args.farveh_oversample > 1.0 and \
                os.path.exists("out/farveh_scenes.txt"):
            # D2' (2026-08-27): 40-80m recall is half of near range (0.19-0.28 vs
            # 0.46-0.53). Upweight far-vehicle-rich scenes to raise exposure.
            fv = set(open("out/farveh_scenes.txt").read().split())
            if weights is None:
                weights = torch.ones(len(tr))
            n_fv = 0
            for ii, (s_, _f) in enumerate(tr.items):
                if s_ in fv:
                    weights[ii] = max(float(weights[ii]),
                                      args.farveh_oversample)
                    n_fv += 1
            if is_main:
                print(f"[data] far-veh oversample x{args.farveh_oversample}: "
                      f"{n_fv}/{len(tr)} frames ({len(fv)} scenes)",
                      flush=True)
        if args.cosmos_oversample > 1.0:
            if weights is None:
                weights = torch.ones(len(tr))
            n_cos = 0
            for ii, (s_, _f) in enumerate(tr.items):
                if s_.startswith("cosmos3_"):
                    weights[ii] = max(float(weights[ii]),
                                      args.cosmos_oversample)
                    n_cos += 1
            if is_main:
                print(f"[data] cosmos3 oversample x"
                      f"{args.cosmos_oversample}: {n_cos}/{len(tr)} frames",
                      flush=True)
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
        if args.model in ("v13", "v13d", "v14d", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36", "v37", "v38", "v39", "v40", "v41", "v42", "v43", "v44", "v45", "v46", "v47", "v48", "v49", "v51", "v52", "v53", "v54", "v55", "v56", "v63b", "v64r50", "v52r50", "v52r50s8", "v52rvgg", "v55rvgg", "v52s8") else {}
    if args.depth_bins != 64:
        # Orin 48bin lever (2026-08-31): D is a class attribute, so shadow it on
        # the concrete class before construction. Changing the bin count keeps
        # the range (1..79.75m) = coarser step. The init's 64bin head is dropped
        # on shape mismatch and retrained.
        _M = MODELS[args.model]
        _span = (_M.D - 1) * _M.D_STEP
        _M.D = args.depth_bins
        _M.D_STEP = _span / (args.depth_bins - 1)
        if is_main:
            print(f"[depth-bins] D={_M.D} step={_M.D_STEP:.4f} (range kept)",
                  flush=True)
    model = MODELS[args.model](**mkw).to(device)
    # METEOR_MODPROBE=1: name the FIRST module whose output goes non-finite.
    # The existing report names the bad OUTPUT (always out[2], the 2D seg), and
    # the clamp on those logits did not stop it -- which means the NaN is
    # already present upstream and clamp(NaN) is NaN. r59 discarded 69 % of its
    # 34,280 steps (0 % up to step 8k, then 98-100 % for the rest of the round)
    # and the output-level report could not localise it any further. One
    # isfinite() per module is a device sync per module, so this is opt-in.
    _MODBAD = []
    if os.environ.get("METEOR_MODPROBE"):
        def _mk(nm):
            def _h(mod, inp, outp):
                if _MODBAD or not torch.is_tensor(outp):
                    return
                if bool(torch.isfinite(outp.detach()).all()):
                    return
                fin_in = all(bool(torch.isfinite(t.detach()).all())
                             for t in inp if torch.is_tensor(t))
                _MODBAD.append(f"{nm} ({type(mod).__name__}) "
                               f"out{tuple(outp.shape)} "
                               f"input {'finite' if fin_in else 'already non-finite'}")
            return _h
        n_h = 0
        for _nm, _m in model.named_modules():
            if len(list(_m.children())) == 0:
                _m.register_forward_hook(_mk(_nm))
                n_h += 1
        if is_main:
            print(f"[modprobe] hooked {n_h} modules", flush=True)
    if args.grad_ckpt:
        model.grad_ckpt = True
        if is_main:
            print("[grad-ckpt] backbone activations recomputed in backward",
                  flush=True)
    if use_rl and is_main:
        print(f"[rl] GRPO mode loss w={args.rl_w} ent={args.rl_ent} "
              f"imit={args.rl_imit_w} tl={args.rl_tl_w}", flush=True)
    if use_pl:
        model.pl_feed = True
        model.pl_feed_p = args.pl_feed_p
        if is_main:
            print(f"[pseudo-lidar] w={args.pseudo_lidar_w} "
                  f"feed_p={args.pl_feed_p}", flush=True)
    if args.bev_dropblock > 0:
        model.bev_dropblock = args.bev_dropblock
        if is_main:
            print(f"[bev-dropblock] p={args.bev_dropblock}", flush=True)
    if args.bev_wedgedrop > 0:
        model.bev_wedgedrop = args.bev_wedgedrop
        if is_main:
            print(f"[bev-wedgedrop] p={args.bev_wedgedrop}", flush=True)
    if args.bev_ringdrop > 0:
        model.bev_ringdrop = args.bev_ringdrop
        if is_main:
            print(f"[bev-ringdrop] p={args.bev_ringdrop}", flush=True)
    if args.bev_chandrop > 0:
        model.bev_chandrop = args.bev_chandrop
        if is_main:
            print(f"[bev-chandrop] p={args.bev_chandrop}", flush=True)
    if args.quant_noise > 0:
        model.quant_noise = args.quant_noise
        if rank == 0:
            print(f"[quant-noise] {args.quant_noise} LSB", flush=True)
    if args.zero_cams:
        from bevlane.dataset import CAMS as _CAMS
        model.zero_cams = tuple(_CAMS.index(c)
                                for c in args.zero_cams.split(","))
        if rank == 0:
            print(f"[zero-cams] {args.zero_cams} -> idx {model.zero_cams}",
                  flush=True)
    if args.init_ckpt:
        # Trusted in-house checkpoints include argparse metadata. PyTorch 2.6
        # otherwise defaults to weights_only=True and can reject that metadata.
        try:
            _init_raw = torch.load(args.init_ckpt, map_location="cpu",
                                   weights_only=False)
        except TypeError:  # PyTorch < 2.0
            _init_raw = torch.load(args.init_ckpt, map_location="cpu")
        sd = _init_raw["model"]
        # If the init already has a slimmed depth head, rebuild at the same width
        # before loading (avoids dropping trained weights on shape mismatch)
        _npre0 = model.module if hasattr(model, "module") else model
        if "depth_head.0.0.weight" in sd and hasattr(_npre0, "depth_head"):
            _w_ck = tuple(sd[f"depth_head.{i}.0.weight"].shape[0]
                          for i in range(4)
                          if f"depth_head.{i}.0.weight" in sd)
            _w_cur = tuple(m[0].out_channels for m in _npre0.depth_head[:-1])
            if len(_w_ck) == 4 and args.depth_slim_force and args.depth_slim > 0:
                # --depth-slim-force (2026-09-05, v146): rebuild at the requested width
                # first, then warm-start by slicing the init's depth head tensors to
                # their leading channels (keeps the init-matching rebuild from disabling
                # the lever, and converges faster than fresh init). The later depth_slim
                # block does not apply twice because the width is no longer the default (256).
                from bevlane.model import enable_depth_slim
                enable_depth_slim(_npre0, scale=args.depth_slim)
                _w_cur = tuple(m[0].out_channels for m in _npre0.depth_head[:-1])
                _msd = _npre0.state_dict(); _nsl = 0
                for _k in list(sd.keys()):
                    if _k.startswith("depth_head.") and _k in _msd \
                            and sd[_k].dim() == _msd[_k].dim() \
                            and tuple(sd[_k].shape) != tuple(_msd[_k].shape):
                        sd[_k] = sd[_k][tuple(slice(0, d) for d in _msd[_k].shape)].clone()
                        _nsl += 1
                print(f"[depth-slim] kept requested width {_w_cur}; initialized by "
                      f"slicing init {_w_ck} ({_nsl} tensor)", flush=True)
            elif len(_w_ck) == 4 and _w_ck != _w_cur:
                from bevlane.model import enable_depth_slim
                enable_depth_slim(_npre0, widths=_w_ck)
                print(f"[depth-slim] rebuilt at width {_w_ck} to match init",
                      flush=True)
        # Dynamic branches MUST exist before state_dict filtering/loading.
        # They used to be attached below, after this block, so every warm
        # start silently discarded their learned tensors and recreated a
        # zero-initialised lane/paint/deep head. The shared trunk continued,
        # which hid the reset behind missing=0/unexpected=0 in the log.
        _npre = model.module if hasattr(model, "module") else model
        if args.paint_seg and not hasattr(_npre, "paint_proj"):
            _npre.enable_paint_seg([int(x) for x in args.paint_seg.split(",")])
        if args.paint_det and not hasattr(_npre, "paint_det_proj"):
            _npre.enable_paint_det([int(x) for x in args.paint_det.split(",")])
        if args.lane_branch and getattr(_npre, "lane_branch", None) is None:
            from bevlane.model import enable_lane_branch
            enable_lane_branch(_npre)
        if args.seg_deep and getattr(_npre, "seg_deep", None) is None:
            from bevlane.model import enable_seg_deep
            enable_seg_deep(_npre, n=args.seg_deep)
        if args.det_deep and getattr(_npre, "det_deep", None) is None:
            from bevlane.model import enable_det_deep
            enable_det_deep(_npre, n=args.det_deep)
        if args.lane_sdf_w > 0 and getattr(_npre, "lane_sdf", None) is None:
            from bevlane.model import enable_lane_sdf
            enable_lane_sdf(_npre)
        if args.pact and not getattr(_npre, "_pact_names", None):
            import json as _jspre
            _aipre = {}
            if args.pact_alpha_init and os.path.exists(args.pact_alpha_init):
                _aipre = _jspre.load(open(args.pact_alpha_init))
            _npre.enable_pact(args.pact, alpha_init=_aipre, verbose=is_main)
        # A quant-robust stationary checkpoint has a reparameterised module.
        # Attach that shape before filtering keys, otherwise continuation
        # silently drops the trained head and recreates it from random weights.
        if any(k.replace("module.", "").startswith("stat_head2.proj.")
               for k in sd):
            from bevlane.model import enable_quant_stat_head
            enable_quant_stat_head(model, args.stat_quant_head or 8.0)
        cur = model.state_dict()   # drop shape-mismatched heads (12->21cls seg)
        _dropped = [k for k, v in sd.items()
                    if k not in cur or cur[k].shape != v.shape]
        sd = {k: v for k, v in sd.items()
              if k in cur and cur[k].shape == v.shape}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        _b0 = sanitize_bn(model)
        if _b0 and is_main:
            print(f"[init] sanitized {len(_b0)} non-finite BN buffers "
                  "from the checkpoint", flush=True)
        if is_main:
            print(f"[init] {args.init_ckpt} missing={len(missing)} "
                  f"unexpected={len(unexpected)} dropped={len(_dropped)}"
                  + (f" first={_dropped[:4]}" if _dropped else ""),
                  flush=True)
    if args.stat_quant_head > 0:
        from bevlane.model import enable_quant_stat_head
        _n0s = model.module if hasattr(model, "module") else model
        enable_quant_stat_head(_n0s, args.stat_quant_head)
        if is_main:
            print(f"[stat-quant-head] +/- bounded reparameterisation "
                  f"cap={args.stat_quant_head:g}", flush=True)
    if args.freeze_depth:
        _nfz = 0
        for _n, _p in model.named_parameters():
            if _n.startswith("depth_head") or _n == "log_sigma":
                _p.requires_grad_(False)
                _nfz += _p.numel()
        if is_main:
            print(f"[freeze-depth] {_nfz / 1e6:.2f}M depth parameters frozen",
                  flush=True)
    if args.paint_seg:
        _n0p = model.module if hasattr(model, "module") else model
        if not hasattr(_n0p, "paint_proj"):
            _n0p.enable_paint_seg([int(x) for x in args.paint_seg.split(",")])
        if is_main:
            print(f"[paint-seg] injecting classes {args.paint_seg} before the lift "
                  f"(zero-initialized = function-preserving)", flush=True)
    if args.ego_speed_w > 0:
        (model.module if hasattr(model, "module") else model).EGO_SPEED_W = args.ego_speed_w
        if is_main:
            print(f"[ego-speed-w] speed weight 1+v0/{args.ego_speed_w} (cap 4)", flush=True)
    if args.kin_anchor:
        from bevlane.model import enable_kinematic_anchor
        enable_kinematic_anchor(model.module if hasattr(model, "module") else model)
        if is_main:
            print("[kin-anchor] kinematic anchor g_t*[v0*t,0] enabled (zero-initialized)", flush=True)
    if args.freeze_ego:
        _nf = _apply_freeze_ego(model)
        if is_main:
            print(f"[freeze-ego] E2E head frozen: {_nf/1e6:.2f}M params (requires_grad=False, BN eval; before DDP)", flush=True)
    if args.semantic_ego:
        from bevlane.model import enable_semantic_ego
        _n0se = model.module if hasattr(model, "module") else model
        enable_semantic_ego(_n0se)
        if is_main:
            print("[semantic-ego] semantic-output-reading ego residual enabled", flush=True)
    if args.vru_cw != 5.0:
        _n0v = model.module if hasattr(model, "module") else model
        _n0v.VRU_CW = args.vru_cw
        if is_main:
            print(f"[vru-cw] VRU class weight {args.vru_cw}", flush=True)
    if args.depth_log_bins:
        from bevlane.model import enable_depth_logbins
        _n0lb = model.module if hasattr(model, "module") else model
        enable_depth_logbins(_n0lb)
        if is_main:
            print("[depth-log-bins] log bins enabled", flush=True)
    if args.mode_scorer:
        from bevlane.model import enable_mode_scorer
        _n0m = model.module if hasattr(model, "module") else model
        enable_mode_scorer(_n0m)
        if is_main:
            print("[mode-scorer] route-conditioned selection scorer enabled", flush=True)
    if args.det_temporal:
        from bevlane.model import enable_det_temporal
        _n0d = model.module if hasattr(model, "module") else model
        enable_det_temporal(_n0d)
        if is_main:
            print("[det-temporal] temporal-feature residual into hm enabled", flush=True)
    if args.traj_cv:
        from bevlane.model import enable_traj_cv
        _n0c = model.module if hasattr(model, "module") else model
        enable_traj_cv(_n0c)
        if is_main:
            print("[traj-cv] CV reparameterization enabled", flush=True)
    if args.traj_flow:
        from bevlane.model import enable_traj_flow
        _n0tf = model.module if hasattr(model, "module") else model
        enable_traj_flow(_n0tf)
        if is_main:
            print("[traj-flow] flow->traj zero-initialized residual enabled", flush=True)

    if args.delta_stat:
        from bevlane.model import enable_delta_stat
        _n0m = model.module if hasattr(model, "module") else model
        enable_delta_stat(_n0m)
        if is_main:
            print("[delta-stat] stop decision switched to the temporal-difference head", flush=True)

    if args.depth_slim > 0:
        # If init-ckpt was already slimmed, the width was matched at load time,
        # so replace here only when the width is still the default (starts at 256).
        _n0d = model.module if hasattr(model, "module") else model
        if _n0d.depth_head[0][0].out_channels == 256:
            from bevlane.model import enable_depth_slim
            enable_depth_slim(_n0d, scale=args.depth_slim)
            if is_main:
                _w = tuple(m[0].out_channels for m in _n0d.depth_head[:-1])
                print(f"[depth-slim] depth head slimmed to width {_w} (fresh init)",
                      flush=True)

    if args.ego_conv_pool:
        import json as _js2
        _st = _js2.load(open(args.ego_conv_pool))
        _n0e = model.module if hasattr(model, "module") else model
        _n0e.convert_ego_pool(_st["hw"], _st.get("mean"), _st.get("var"),
                              verbose=is_main)

    if args.pact:
        import json as _js
        _ai = {}
        if args.pact_alpha_init and os.path.exists(args.pact_alpha_init):
            _ai = _js.load(open(args.pact_alpha_init))
        _n0p = model.module if hasattr(model, "module") else model
        if not getattr(_n0p, "_pact_names", None):
            _n0p.enable_pact(args.pact, alpha_init=_ai, verbose=is_main)

    if args.paint_det:
        _n0d = model.module if hasattr(model, "module") else model
        if not hasattr(_n0d, "paint_det_proj"):
            _n0d.enable_paint_det([int(x) for x in args.paint_det.split(",")])
        if is_main:
            print(f"[paint-det] injecting 2D detection classes {args.paint_det} before "
                  f"the lift (zero-initialized = function-preserving)", flush=True)
    # Heads with loss weight 0 receive no gradient, so the DDP reducer crashes
    # with "reduction from the previous iteration has not finished".
    # 2026-08-15: this freeze lived only inside --freeze-trunk, so trunk-on
    # (phase A) training crashed right after start on the pl_head / lanegraph
    # set. Zero-weight heads were never trained anyway, so freezing them always
    # changes nothing (function-preserving). Modules like lidar_stem/sdmap_stem that
    # count as trunk only under freeze-trunk do not belong here.
    _ZERO_W = {
        "pl_head": args.pseudo_lidar_w,
        "lg_tower": args.lanegraph_w, "lg_in": args.lanegraph_w,
        "lgdec": args.lanegraph_w, "lgq": args.lanegraph_w,
        "lg_mlp": args.lanegraph_w, "lg_pts": args.lanegraph_w,
        "lg_pts2": args.lanegraph_w, "lg_meta": args.lanegraph_w,
        "lg_meta2": args.lanegraph_w, "lg_adj": args.lanegraph_w,
        "risk_head": args.risk_w, "risk_gate": args.risk_w,
        "flow_head": args.flow_w,
        "unk_head": args.unk_w, "unk_stem": args.unk_w,
        "unk_head2": args.unk_w,
        "occ_stem": args.occ_w, "occ_head": args.occ_w,
        "traj_stem": args.traj_w, "traj_head": args.traj_w,
        "stat_head": args.stat_w, "stat_head2": args.stat_w,
        "tl_stem": args.tl_w, "tl_head": args.tl_w, "tl_fc": args.tl_w,
        "lane_sdf": args.lane_sdf_w,
    }
    _n0z = model.module if hasattr(model, "module") else model
    _offz = []
    for _nm, _p in _n0z.named_parameters():
        _top = _nm.split(".")[0]
        if _ZERO_W.get(_top, 1.0) <= 0:
            _p.requires_grad_(False)
            if _top not in _offz:
                _offz.append(_top)
    for _nm, _m in _n0z.named_modules():
        if _nm and _nm.split(".")[0] in _offz:
            _m.eval()
    if is_main and _offz:
        print(f"[freeze-zero] frozen (loss 0): {', '.join(_offz)}",
              flush=True)
    if args.freeze_trunk and not args.det_only:
        # Shared trunk = image features -> depth/ctx -> lift -> temporal fusion.
        # Freezing it leaves no trainable parameter shared between heads, so
        # interference vanishes structurally (each head's gradient reaches only its own parameters).
        _TRUNK = ("stem", "layer1", "layer2", "layer3", "layer4",
                  "lat1", "lat2", "lat3", "lat4", "fuse",
                  "depth_head", "depth_up", "ctx", "tgate", "tfuse3", "tfuse")
        _n0 = model.module if hasattr(model, "module") else model
        _tr = _fr = 0
        for _nm, _p in _n0.named_parameters():
            if _nm.split(".")[0] in _TRUNK:
                _p.requires_grad_(False); _fr += _p.numel()
            else:
                _tr += _p.numel()
        for _nm, _m in _n0.named_modules():
            if _nm and _nm.split(".")[0] in _TRUNK:
                _m.eval()
        # Heads with loss 0 receive no gradient and crash DDP -> freeze them too.
        _HEAD_W = {
            "dec": args.seg_w, "lane_branch": args.seg_w,
            "lane_sdf": args.lane_sdf_w, "seg_deep": args.seg_w,
            "seg_head": args.seg2d_w,
            "det2d": args.bbox2d_w, "det2d_d8": args.bbox2d_w,
            "det2d_d16": args.bbox2d_w, "hm2d_head": args.bbox2d_w,
            "reg2d_head": args.bbox2d_w, "hm2d_head8": args.bbox2d_w,
            "reg2d_head8": args.bbox2d_w, "hm2d_head16": args.bbox2d_w,
            "reg2d_head16": args.bbox2d_w, "det2d_stem": args.bbox2d_w,
            "det_stem": args.box_w, "hm_head": args.box_w,
            "reg_head": args.box_w, "det_deep": args.box_w,
            "occ_stem": args.occ_w, "occ_head": args.occ_w,
            "traj_stem": args.traj_w, "traj_head": args.traj_w,
            "agent_q": args.traj_w, "agent_attn": args.traj_w,
            "agent_delta": args.traj_w,
            "stat_head": args.stat_w, "stat_head2": args.stat_w,
            "tl_stem": args.tl_w, "tl_head": args.tl_w, "tl_fc": args.tl_w,
            "risk_head": args.risk_w, "risk_gate": args.risk_w,
            "flow_head": args.flow_w,
            "unk_head": args.unk_w, "unk_stem": args.unk_w,
            "unk_head2": args.unk_w, "unk_dense": args.unk_dense_w,
            "pl_head": args.pseudo_lidar_w,
            "lg_tower": args.lanegraph_w, "lg_in": args.lanegraph_w,
            "lgdec": args.lanegraph_w, "lgq": args.lanegraph_w,
            "lg_mlp": args.lanegraph_w, "lg_pts": args.lanegraph_w,
            "lg_pts2": args.lanegraph_w, "lg_meta": args.lanegraph_w,
            "lg_meta2": args.lanegraph_w, "lg_adj": args.lanegraph_w,
            "ego_stem": args.ego_w, "ego_mlp": args.ego_w,
            "ego_q": args.ego_w, "ego_attn": args.ego_w,
            "ego_delta": args.ego_w, "intent_mlp": args.ego_w,
            "intent_delta": args.ego_w, "kin_delta": args.ego_w,
            "vprof_head": args.ego_w, "dec_head": args.ego_w,
            "dec_gate": args.ego_w,
            "lidar_stem": 0.0, "sdmap_stem": 0.0, "lid_alpha": 0.0,
        }
        _off = []
        for _nm, _p in _n0.named_parameters():
            _top = _nm.split(".")[0]
            if _top in _TRUNK:
                continue
            if _HEAD_W.get(_top, 1.0) <= 0:
                _p.requires_grad_(False); _tr -= _p.numel(); _fr += _p.numel()
                if _top not in _off:
                    _off.append(_top)
        for _nm, _m in _n0.named_modules():
            if _nm and _nm.split(".")[0] in _off:
                _m.eval()
        if is_main and _off:
            print(f"[freeze-trunk] frozen (loss 0): {', '.join(_off)}",
                  flush=True)
        _n0._freeze_eval_keep = tuple(
            n for n, _ in _n0.named_children()
            if n not in _TRUNK and n not in _off)
        if is_main:
            print(f"[freeze-trunk] training heads {_tr / 1e6:.1f}M / shared trunk "
                  f"{_fr / 1e6:.1f}M frozen (incl. BN stats)", flush=True)
    if args.det_only:
        # det path = backbone -> FPN -> depth/ctx -> lift -> temporal fusion ->
        # det_stem -> hm/reg. Everything else is frozen.
        # --det-head-only additionally freezes the trunk and moves only the det head.
        # --det-head-only also trains the refiner's box branch (the refiner
        # ultimately overwrites hm/reg, so moving only the raw head with it frozen
        # makes the producer and the corrector disagree). seg/ego branches stay
        # frozen, so other outputs are unchanged and DDP's unused-parameter issue does not arise.
        _KEEP = (("det_stem", "hm_head", "reg_head", "refiner.box")
                 if args.det_head_only else
                 ("stem", "layer1", "layer2", "layer3", "layer4",
                  "lat1", "lat2", "lat3", "lat4", "fuse",
                  "depth_head", "depth_up", "ctx",
                  "tgate", "tfuse3", "tfuse",
                  "det_stem", "hm_head", "reg_head"))
        _n0 = model.module if hasattr(model, "module") else model
        _tr = _fr = 0
        def _keep(nm):
            return (nm.split(".")[0] in _KEEP
                    or any(nm.startswith(k) for k in _KEEP if "." in k))
        for _nm, _p in _n0.named_parameters():
            if _keep(_nm):
                _tr += _p.numel()
            else:
                _p.requires_grad_(False)
                _fr += _p.numel()
        if args.det_head_only:
            # BatchNorm in the frozen part left in train() moves its running stats,
            # breaking "other outputs unchanged". Pin those modules to eval().
            for _nm, _m in _n0.named_modules():
                if _nm and not _keep(_nm):
                    _m.eval()
            _n0._freeze_eval_keep = _KEEP     # marker to preserve on re-entering train()
        if is_main:
            print(f"[det-only{'/head' if args.det_head_only else ''}] "
                  f"training {_tr / 1e6:.1f}M / frozen {_fr / 1e6:.1f}M "
                  f"params; non-det losses are 0", flush=True)
    if args.det_sup_front is not None or args.det_sup_rear is not None:
        _n = model.module if hasattr(model, "module") else model
        _n.DET_SUP_XF = args.det_sup_front
        _n.DET_SUP_XR = args.det_sup_rear
        if is_main:
            print(f"[det-sup] loss window front<={args.det_sup_front}m "
                  f"rear<={args.det_sup_rear}m", flush=True)
    if args.ego_ce_tau:
        (model.module if hasattr(model, "module") else model
         ).EGO_CE_TAU = args.ego_ce_tau
        if is_main:
            print(f"[e2e] soft selector target tau={args.ego_ce_tau}",
                  flush=True)
    if args.ego_fde_w:
        (model.module if hasattr(model, 'module') else model
         ).EGO_FDE_W = args.ego_fde_w
        if is_main:
            print(f"[e2e] endpoint term w={args.ego_fde_w}",
                  flush=True)
    if args.lane_branch:
        # MUST attach before SyncBN/DDP: attached after, the branch lands on
        # the DDP WRAPPER -- absent from module.state_dict() (never saved) and
        # bypassed by DDP's forward (never trained). v61/v62/r66 ran this flag
        # as a no-op before the 2026-08-12 fix.
        from bevlane.model import enable_lane_branch
        if getattr(model, "lane_branch", None) is None:
            enable_lane_branch(model)
        if is_main:
            print("[lane-branch] thin-class residual decoder on", flush=True)
    if args.seg_deep:
        from bevlane.model import enable_seg_deep
        if getattr(model, "seg_deep", None) is None:
            enable_seg_deep(model, n=args.seg_deep)
        if is_main:
            print(f"[seg-deep] {args.seg_deep}-block BEV-seg residual tower on",
                  flush=True)
    if args.det_deep:
        from bevlane.model import enable_det_deep
        if getattr(model, "det_deep", None) is None:
            enable_det_deep(model, n=args.det_deep)
        if is_main:
            print(f"[det-deep] {args.det_deep}-block det residual tower on",
                  flush=True)
    if args.lane_sdf_w > 0:
        from bevlane.model import enable_lane_sdf
        if getattr(model, "lane_sdf", None) is None:
            enable_lane_sdf(model)
        if is_main:
            print(f"[lane-sdf] auxiliary head on, w={args.lane_sdf_w}",
                  flush=True)
    if args.sync_bn and ddp:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if is_main:
            print("[sync-bn] BN statistics pooled across ranks", flush=True)
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local],
            find_unused_parameters=(args.seg_w == 0 or
                                    (args.model in ("v13", "v13d", "v14d", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36", "v37", "v38", "v39", "v40", "v41", "v42", "v43", "v44", "v45", "v46", "v47", "v48", "v49", "v51", "v52", "v53", "v54", "v55", "v56", "v63b", "v64r50", "v52r50", "v52r50s8", "v52rvgg", "v55rvgg", "v52s8") and not use_seg2d)))
    _sp_pairs = []
    _sp_final = []          # for the ramp: post-switch 2:4 masks (same order as _sp_pairs)
    if args.sparse_24:
        # Dense exceptions: the network outputs (their per-channel scale is the
        # calibration surface) and anything whose input-channel count is not a
        # multiple of 4. Deterministic from the init weights, so every DDP rank
        # builds identical masks without communication.
        # "Last dense" means EVERY output layer. The first list missed the
        # detection heads and v60 paid for it: veh yaw error went 5.5->16.7
        # deg and R50 0.46->0.35 while seg/E2E recovered fine -- the 2:4 mask
        # on reg_head's sin/cos regression is exactly the kind of layer
        # magnitude pruning butchers. Output layers are now excluded two ways:
        # by name, and by out-channel count (<32 catches hm 2 / reg 6 / risk 1
        # / flow 2 / stat 1 / lane_branch 3 / seg finals).
        _DENSE = ("dec.out.3", "seg_head.out.3", "depth_head.4", "occ_head",
                  "hm_head", "reg_head")
        _net0 = model.module if ddp else model
        _n_sp = _n_dense = 0
        with torch.no_grad():
            for _nm, _p in _net0.named_parameters():
                # --sparse-exclude (2026-09-06, v152 candidate): keep accuracy-sensitive branches
                # that add no Orin ms (E2E/planning etc.) dense (prefix list).
                _excl = tuple(x for x in args.sparse_exclude.split(",") if x)
                if _p.dim() != 4 or _p.shape[1] % 4 != 0 or _p.shape[1] < 16 \
                        or _p.shape[0] < 32 \
                        or any(k in _nm for k in _DENSE) \
                        or (_excl and _nm.startswith(_excl)):
                    _n_dense += 1
                    continue
                _o, _c, _kh, _kw = _p.shape
                _f = _p.detach().abs().reshape(_o, _c // 4, 4, _kh * _kw)
                _srt = _f.argsort(2)
                _idx = _srt[:, :, :2]
                _m = torch.ones_like(_f)
                _m.scatter_(2, _idx, 0.0)
                _m = _m.reshape(_o, _c, _kh, _kw)
                if args.sparse_ramp_steps > 0:
                    # start with a 1:4 mask (only the smallest zeroed), switch to 2:4 (_m) after the ramp
                    _m1 = torch.ones_like(_f)
                    _m1.scatter_(2, _srt[:, :, :1], 0.0)
                    _m1 = _m1.reshape(_o, _c, _kh, _kw)
                    _p.mul_(_m1)
                    _sp_pairs.append((_p, _m1))
                    _sp_final.append(_m)
                else:
                    _p.mul_(_m)                     # prune once
                    _sp_pairs.append((_p, _m))
                _n_sp += 1
        if is_main:
            _kept = sum(int(m.sum()) for _, m in _sp_pairs)
            _tot = sum(m.numel() for _, m in _sp_pairs)
            print(f"[sparse-24] {_n_sp} tensors masked "
                  f"({_kept / max(_tot, 1):.0%} kept), {_n_dense} left dense"
                  + (f"; ramp: 1:4 for the first {args.sparse_ramp_steps} steps"
                     if args.sparse_ramp_steps > 0 else ""),
                  flush=True)
    # ---- dense-teacher distillation (2026-09-07, v156): recover the planning accuracy lost
    # in 2:4 sparse fine-tuning by matching the dense model's fused BEV and E2E outputs on the same input.
    # The teacher is built with probe_net.load_full (same correct construction as export, verified unexpected/mismatch=0).
    _teacher = None
    if args.dense_teacher and args.dense_distill_w > 0:
        from bevlane.probe_net import load_full as _load_full
        _teacher = _load_full(args.dense_teacher, device=next(model.parameters()).device,
                              verbose=is_main).half().eval()
        for _p in _teacher.parameters():
            _p.requires_grad_(False)
        if is_main:
            print(f"[dense-teacher] {args.dense_teacher} w={args.dense_distill_w} "
                  f"ego_w={args.dense_distill_ego_w} every={args.dense_distill_every}", flush=True)
    ema = EMA(model.module if ddp else model, args.ema,
              exclude=tuple(x for x in args.ema_exclude.split(",") if x)) if args.ema > 0 else None
    if ema is not None and args.ema_exclude and is_main:
        print(f"[ema] excluded prefixes: {args.ema_exclude}", flush=True)
    if ema is not None and is_main:
        print(f"[ema] decay {args.ema} over "
              f"{sum(v.numel() for v in ema.shadow.values()) / 1e6:.1f}M "
              "parameters; evaluated alongside the raw weights", flush=True)
    # PACT alphas must move by orders of magnitude more than the body (30-130 -> 10-15).
    # At the body lr 1e-4, Adam moves 1e-4 per step x 240k steps = 24 at most,
    # short of tfuse's 127 -> 10. Give them a separate param group with their own lr.
    _pact_ps = [p for n, p in model.named_parameters()
                if p.requires_grad and n.endswith(".alpha")]
    _pact_id = {id(p) for p in _pact_ps}
    # A1b (2026-08-27): the zero-initialized traj_flow residual does not grow at the
    # body lr within a ~4000-step probe (measured |w| is 1/70 of traj_head). Same
    # rule as PACT: "what must move by orders of magnitude gets its own group", with its own multiplier.
    _tfl_ps = [p for n, p in model.named_parameters()
               if p.requires_grad and n.startswith(
                   ("traj_flow.", "module.traj_flow."))] \
        if args.traj_flow_lr_mult != 1.0 else []
    _tfl_id = {id(p) for p in _tfl_ps}
    _gr_ps = [p for n, p in model.named_parameters()
              if p.requires_grad and n.startswith(
                  ("det_tmp.", "module.det_tmp.", "mode_scorer.",
                   "module.mode_scorer.", "traj_vel.", "module.traj_vel."))] \
        if args.graft_lr_mult != 1.0 else []
    _gr_id = {id(p) for p in _gr_ps}
    # v139b: the kinematic anchor gates (6) get their own group with no decay and lr x20.
    # In v139 they sat in the base group (wd 1e-4, sign-cancelling gradients) and were
    # still at 0.001 after 30k steps, so the anchor was effectively absent.
    _kin_ps = [p for n, p in model.named_parameters()
               if p.requires_grad and n.endswith("kin_gate")]
    _kin_id = {id(p) for p in _kin_ps}
    # --hist-lr-mult (2026-09-05, v149): split the history modules (temporal fusion, motion
    # residual, temporal-difference stop head, ego head) into their own lr-scaled group. History
    # was zero in every E2E round, so these are nearly untrained on real history (v144 lagged with a uniform lr).
    _HIST_PREFIX = ("tfuse3.", "tgate.", "traj_stem.", "delta_stat.", "ego_stem.",
                    "ego_mlp.", "ego_q.", "ego_attn.", "ego_delta.", "sem_ego.")
    _hl_ps = [p for n, p in model.named_parameters()
              if p.requires_grad and (n.startswith(_HIST_PREFIX)
                                      or n.startswith(tuple("module." + x for x in _HIST_PREFIX)))
              and id(p) not in _kin_id] if args.hist_lr_mult != 1.0 else []
    _hl_id = {id(p) for p in _hl_ps}
    _base_ps = [p for p in model.parameters()
                if p.requires_grad and id(p) not in _pact_id
                and id(p) not in _tfl_id and id(p) not in _gr_id
                and id(p) not in _kin_id and id(p) not in _hl_id]
    _groups = [{"params": _base_ps, "lr": lr, "weight_decay": 1e-4}]
    _maxlr = [lr]
    if _kin_ps:
        _groups.append({"params": _kin_ps, "lr": lr * 20.0, "weight_decay": 0.0})
        _maxlr.append(lr * 20.0)
        if is_main:
            print(f"[kin-anchor] gate {len(_kin_ps)} tensor split into a lr x20 / wd 0 "
                  f"param group", flush=True)
    if _hl_ps:
        _groups.append({"params": _hl_ps, "lr": lr * args.hist_lr_mult,
                        "weight_decay": 1e-4})
        _maxlr.append(lr * args.hist_lr_mult)
        if is_main:
            print(f"[hist-lr] history {len(_hl_ps)} tensors split into a lr x{args.hist_lr_mult} "
                  f"param group", flush=True)
    if _gr_ps:
        _groups.append({"params": _gr_ps,
                        "lr": lr * args.graft_lr_mult,
                        "weight_decay": 0.0})
        _maxlr.append(lr * args.graft_lr_mult)
        if is_main:
            print(f"[graft] {len(_gr_ps)} tensors split into a lr x"
                  f"{args.graft_lr_mult} param group", flush=True)
    if _tfl_ps:
        _groups.append({"params": _tfl_ps,
                        "lr": lr * args.traj_flow_lr_mult,
                        "weight_decay": 0.0})
        _maxlr.append(lr * args.traj_flow_lr_mult)
        if is_main:
            print(f"[traj-flow] {len(_tfl_ps)} tensors split into a lr x"
                  f"{args.traj_flow_lr_mult} param group",
                  flush=True)
    if _pact_ps:
        _groups.append({"params": _pact_ps, "lr": args.pact_lr,
                        "weight_decay": 0.0})
        _maxlr.append(args.pact_lr)
        if is_main:
            print(f"[pact] {len(_pact_ps)} alphas split into a dedicated lr {args.pact_lr} "
                  f"param group (no weight_decay)", flush=True)
    opt = torch.optim.AdamW(_groups, lr=lr, weight_decay=1e-4)
    total_steps = len(dl) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=(_maxlr if len(_maxlr) > 1 else lr),
        total_steps=total_steps)
    scaler = torch.cuda.amp.GradScaler()
    pl_acc = [0.0, 0.0, 0.0, 0.0]   # occ inter/union, z abs-err, n
    rl_acc = [0.0, 0.0, 0.0, 0.0]   # rew sel/best, pick acc, n
    cw = CLASS_W.clone()
    # 255 is the explicit consensus don't-care value. Class 0 is controlled
    # independently by cw[0] (zero by default, 0.5 with --train-bg).
    ignore = 255
    if args.train_bg:
        cw[0] = 0.5
    cw = cw.to(device)

    step = 0
    t0 = time.time()
    best = 0.0
    best_e2e = 1e9
    val2d_miou = None   # seg2d survival guard (2026-08-28)
    best_chain = 1e9    # EMA ckpt with the lowest chain proxy (ADEc + |high-speed wp0 bias|)

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
            unk_v2 = batch[bi] if use_unk_v2 else None
            bi += 1 if use_unk_v2 else 0
            lidbev = batch[bi] if use_lidarbev else None
            bi += 1 if use_lidarbev else 0
            sdmap_t = batch[bi] if use_sdmap else None
            bi += 1 if use_sdmap else 0
            tl_t = batch[bi] if use_tlin else None
            bi += 1 if use_tlin else 0
            if use_temporal:
                prev_imgs, rel_pose, prev_valid = (batch[bi], batch[bi + 1],
                                                   batch[bi + 2])
            else:
                prev_imgs = rel_pose = prev_valid = None
            if args.bev_rot_aug > 0:
                (Tc, gt, det_boxes, traj_gt, ego_gt, occ_gt, risk_gt,
                 lg_pts_gt, unk_c, rel_pose, unk_v2) = bev_rotation_aug(
                    args.bev_rot_aug, Tc, gt, det_boxes, det_n, traj_gt,
                    ego_gt, occ_gt, risk_gt, lg_pts_gt, unk_c, rel_pose,
                    unk_v2=unk_v2, lat_max=args.lat_aug, lat_p=args.lat_p,
                    lat_min=args.lat_min)
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
                _reeval_frozen(model)
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
                _reeval_frozen(model)
                pb = pb.float()
                theta = make_warp_theta(rel_pose)
            intent_oh = None
            if args.model in ("v37", "v38", "v39", "v40", "v41", "v42", "v43", "v44", "v45", "v46", "v47", "v48", "v49", "v51", "v52", "v53", "v54", "v55", "v56", "v63b", "v64r50", "v52r50", "v52r50s8", "v52rvgg", "v55rvgg", "v52s8") and ego_gt is not None:
                lat = ego_gt[:, 11]
                if args.model in ("v43", "v44", "v45", "v46", "v47", "v48", "v49", "v51", "v52", "v53", "v54", "v55", "v56", "v63b", "v64r50", "v52r50", "v52r50s8", "v52rvgg", "v55rvgg", "v52s8"):
                    # v43: earlier-firing command -- ANY waypoint (1.5-3 s)
                    # crossing +-2.0 m counts, so the command is active on
                    # the approach, not only mid-turn (matches the pseudo-nav
                    # definition used at inference)
                    wpy = ego_gt[:, :12].view(-1, 6, 2)[:, 2:, 1]
                    lmax = wpy.max(1).values
                    lmin = wpy.min(1).values
                    idx = torch.where(lmax > 2.0, 1,
                                      torch.where(lmin < -2.0, 2, 0))
                else:
                    idx = torch.where(lat > 2.5, 1,
                                      torch.where(lat < -2.5, 2, 0))
                intent_oh = F.one_hot(idx.long(), 3).float()
                # COUNTERFACTUAL COMMANDS (phase 2). The command is derived
                # from the GT future, so "turn left" is only ever seen on
                # frames that do turn left: mode 1 learns "the left turn this
                # scene affords", not "go left". Feed a deliberately WRONG
                # command on a fraction of rows and drop their waypoint
                # supervision -- the GT no longer describes the commanded
                # manoeuvre, so only the direction hinge may speak there.
                if args.intent_wrong > 0:
                    _B = intent_oh.shape[0]
                    _has = intent_oh.sum(1) > 0.5
                    _pick = (torch.rand(_B, device=intent_oh.device)
                             < args.intent_wrong) & _has
                    if _pick.any():
                        # Flip to the OPPOSITE TURN, not to a random other
                        # command. Measured on r48: reversal is asymmetric --
                        # 46 % on GT-left frames vs 26 % on GT-right ones --
                        # and the asymmetry tracks the data (96 right vs 46
                        # left turn frames), so a uniform flip spends most of
                        # its budget on the case that already works.
                        # 1<->2 is left<->right; a straight row gets a random
                        # turn as before.
                        _cur = intent_oh.argmax(1)
                        _rnd = 1 + torch.randint(0, 2, _cur.shape,
                                                 device=intent_oh.device)
                        _new = torch.where(
                            _cur == 1, torch.full_like(_cur, 2),
                            torch.where(_cur == 2, torch.full_like(_cur, 1),
                                        _rnd))
                        intent_oh = torch.where(
                            _pick[:, None], F.one_hot(_new, 3).float(),
                            intent_oh)
                        ego_gt = ego_gt.clone()
                        ego_gt[_pick, 16] = 0.0     # no waypoint target here
                # Modality dropout on the driving command, same contract as
                # lidar_bev / sdmap / tl (all 0.5). It was hard-coded at 0.3,
                # which leaves a 70/30 train-eval mismatch on exactly the thing
                # the selector is scored on: ego_loss overrides `best` with the
                # command whenever one is present, so 70 % of samples teach the
                # selector to COPY a command, while evaluate_ego never passes
                # one and asks it to INFER the manoeuvre. Measured selection gap
                # on r60 best_e2e over 720 val samples: ADE 0.622 vs oracle
                # 0.491, i.e. 0.131 m (21 %) lost to picking the wrong mode.
                intent_oh = intent_oh * (torch.rand(
                    imgs.shape[0], 1, device=device)
                    >= args.intent_drop).float()
            pl_gt = lidbev            # before modality dropout
            lid = None
            if use_lidar and depth_gt is not None:
                # modality dropout: whole-sample, so BN sees both modes
                keep = (torch.rand(imgs.shape[0], 1, 1, 1, device=device)
                        >= args.lidar_drop).to(depth_gt.dtype)
                lid = depth_gt * keep
                if use_lidarbev and lidbev is not None:
                    lidbev = lidbev * keep
            sd_in = None
            tl_in = None
            if use_tlin and tl_t is not None:
                tl_t = tl_t.to(device, non_blocking=True)
                keep_tl = (torch.rand(tl_t.shape[0], 1, 1, 1, 1,
                                      device=device)
                           >= args.tl_drop).to(tl_t.dtype)
                tl_in = tl_t * keep_tl
            if use_sdmap and sdmap_t is not None:
                keep_sd = (torch.rand(imgs.shape[0], 1, 1, 1, device=device)
                           >= args.sdmap_drop).to(sdmap_t.dtype)
                sd_in = sdmap_t * keep_sd
            # ---- teacher pass for LiDAR->camera modality distillation (registered 2026-08-20) ----
            # Teacher = no-grad/eval pass with LiDAR forced ON (no drop). Student =
            # the LiDAR-dropped rows of the main pass that follows. Matches fused BEV
            # and det hm. Runs intermittently on a step basis (all ranks in sync).
            _dist_t = None
            _dist_keep = keep if (use_lidar and depth_gt is not None) else None
            if (args.lidar_distill_w > 0 and use_lidarbev
                    and pl_gt is not None and use_temporal
                    and step % max(args.lidar_distill_every, 1) == 0):
                _net00 = model.module if ddp else model
                model.eval()
                with torch.no_grad(), torch.autocast("cuda", torch.float16):
                    _out_t = model(imgs, K, Tc,
                                   ego_gt[:, 12] if use_ego else None,
                                   pb, theta,
                                   **({"lidar": depth_gt} if use_lidar
                                      and depth_gt is not None else {}),
                                   lidar_bev=pl_gt,
                                   **({"sdmap": sd_in} if use_sdmap else {}),
                                   **({"tl": tl_in} if use_tlin else {}),
                                   **({"kin": rel_pose} if args.model in ("v36", "v37", "v38", "v39", "v40", "v41", "v42", "v43", "v44", "v45", "v46", "v47", "v48", "v49", "v51", "v52", "v53", "v54", "v55", "v56", "v63b", "v64r50", "v52r50", "v52r50s8", "v52rvgg", "v55rvgg", "v52s8") else {}),
                                   **({"intent": intent_oh} if args.model in ("v37", "v38", "v39", "v40", "v41", "v42", "v43", "v44", "v45", "v46", "v47", "v48", "v49", "v51", "v52", "v53", "v54", "v55", "v56", "v63b", "v64r50", "v52r50", "v52r50s8", "v52rvgg", "v55rvgg", "v52s8") else {}))
                    _dist_t = (_net00._fused_bev.detach().float(),
                               (_out_t[3].detach().float()
                                if isinstance(_out_t, tuple)
                                and len(_out_t) >= 5 else None))
                model.train()
                _reeval_frozen(model)
            _bn_bak = {n: b.detach().clone()
                       for n, b in (model.module if ddp else model
                                    ).named_buffers()
                       if "running_" in n}
            # ---- dense-teacher pass (same input, no-grad) ----
            _dt_t = None
            if _teacher is not None and use_temporal \
                    and step % max(args.dense_distill_every, 1) == 0:
                with torch.no_grad(), torch.autocast("cuda", torch.float16):
                    _out_dt = _teacher(imgs, K, Tc,
                                       ego_gt[:, 12] if use_ego else None, pb, theta,
                                       **({"lidar": lid} if use_lidar else {}),
                                       **({"lidar_bev": lidbev} if use_lidarbev else {}),
                                       **({"sdmap": sd_in} if use_sdmap else {}),
                                       **({"tl": tl_in} if use_tlin else {}))
                    _dt_t = (_teacher._fused_bev.detach().float(),
                             (_out_dt[3].detach().float() if isinstance(_out_dt, tuple)
                              and len(_out_dt) >= 5 else None),
                             (_out_dt[7].detach().float() if isinstance(_out_dt, tuple)
                              and len(_out_dt) >= 8 else None))
            with torch.autocast("cuda", torch.float16):
                # v18+ is conditioned on the current speed (ego_gt col 12)
                if use_temporal:
                    out = model(imgs, K, Tc,
                                ego_gt[:, 12] if use_ego else None, pb, theta,
                                **({"lidar": lid} if use_lidar else {}),
                                **({"lidar_bev": lidbev}
                                   if use_lidarbev else {}),
                                **({"sdmap": sd_in} if use_sdmap else {}),
                                **({"tl": tl_in} if use_tlin else {}),
                                **({"kin": rel_pose}
                                   if args.model in ("v36", "v37", "v38", "v39", "v40", "v41", "v42", "v43", "v44", "v45", "v46", "v47", "v48", "v49", "v51", "v52", "v53", "v54", "v55", "v56", "v63b", "v64r50", "v52r50", "v52r50s8", "v52rvgg", "v55rvgg", "v52s8") else {}),
                                **({"intent": intent_oh}
                                   if args.model in ("v37", "v38", "v39", "v40", "v41", "v42", "v43", "v44", "v45", "v46", "v47", "v48", "v49", "v51", "v52", "v53", "v54", "v55", "v56", "v63b", "v64r50", "v52r50", "v52r50s8", "v52rvgg", "v55rvgg", "v52s8") else {}))
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
                # ---- modality distillation loss (LiDAR-dropped rows only, 2026-08-20) ----
                if _dist_t is not None and _dist_keep is not None:
                    _m = (_dist_keep.view(-1) < 0.5)
                    if bool(_m.any()):
                        # Foreground-focused weighting (fixed 2026-08-20). Plain MSE is
                        # area-dominated: large regions like the road surface pull it and
                        # far vehicles (a few cells = 0.01% of the total) were ignored
                        # (why cam recall did not budge from 0.486 even though w=0.5
                        # contributed a sizeable 1.38). Use the teacher hm peak
                        # intensity as a spatial weight, 10x emphasis on vehicle cells.
                        _fs = net0._fused_bev.float()
                        _th = _dist_t[1]
                        if _th is not None:
                            _wfg = _th.sigmoid().amax(1, keepdim=True)
                            _wfg = F.interpolate(_wfg,
                                                 size=_fs.shape[-2:],
                                                 mode="bilinear",
                                                 align_corners=False)
                            _w = 0.1 + 0.9 * _wfg.clamp(0, 1)
                            _se = ((_fs[_m] - _dist_t[0][_m]) ** 2
                                   ).mean(1, keepdim=True)
                            _dl = (_se * _w[_m]).sum() / _w[_m].sum().clamp(
                                min=1e-6)
                        else:
                            _dl = F.mse_loss(_fs[_m], _dist_t[0][_m])
                        if hm is not None and _th is not None:
                            # hm is foreground-weighted too (keeps background-zero matching from dominating)
                            _hw = 0.1 + 0.9 * _th.sigmoid().amax(
                                1, keepdim=True).clamp(0, 1)
                            _hse = ((hm.float()[_m] - _th[_m]) ** 2
                                    ).mean(1, keepdim=True)
                            _dl = _dl + 2.0 * (
                                (_hse * _hw[_m]).sum()
                                / _hw[_m].sum().clamp(min=1e-6))
                        loss = loss + args.lidar_distill_w * _dl
                        if step % 100 == 0 and rank == 0:
                            print(f"[distill step{step}] loss={float(_dl):.4f}"
                                  f" x w={args.lidar_distill_w} -> "
                                  f"{float(args.lidar_distill_w * _dl):.4f}"
                                  f" (n={int(_m.sum())}/{_m.numel()})",
                                  flush=True)
                # ---- dense-teacher distillation loss (2026-09-07): fused BEV (teacher hm peak weight) + E2E output ----
                if _dt_t is not None:
                    _fs2 = net0._fused_bev.float()
                    if _dt_t[1] is not None:
                        _wfg2 = F.interpolate(_dt_t[1].sigmoid().amax(1, keepdim=True),
                                              size=_fs2.shape[-2:], mode="bilinear",
                                              align_corners=False)
                        _w2 = 0.1 + 0.9 * _wfg2.clamp(0, 1)
                        _se2 = ((_fs2 - _dt_t[0]) ** 2).mean(1, keepdim=True)
                        _dl2 = (_se2 * _w2).sum() / _w2.sum().clamp(min=1e-6)
                    else:
                        _dl2 = F.mse_loss(_fs2, _dt_t[0])
                    if ego_pred is not None and _dt_t[2] is not None \
                            and ego_pred.shape == _dt_t[2].shape:
                        _dl2 = _dl2 + args.dense_distill_ego_w * F.mse_loss(
                            ego_pred.float(), _dt_t[2])
                    loss = loss + args.dense_distill_w * _dl2
                    if step % 100 == 0 and rank == 0:
                        print(f"[dense-distill step{step}] loss={float(_dl2):.4f}"
                              f" x w={args.dense_distill_w} -> "
                              f"{float(args.dense_distill_w * _dl2):.4f}", flush=True)
                if occ_pred is not None and use_occ:
                    loss = loss + args.occ_w * net0.occ_loss(occ_pred.float(),
                                                             _fit(occ_gt, occ_pred))
                if ego_pred is not None and use_ego:
                    loss = loss + args.ego_w * net0.ego_loss(
                        ego_pred.float(), ego_gt,
                        intent_oh if args.model in (
                            "v43", "v44", "v45", "v46", "v47", "v48", "v49", "v51", "v52", "v53", "v54", "v55", "v56", "v63b", "v64r50", "v52r50", "v52r50s8", "v52rvgg", "v55rvgg", "v52s8") else None)
                if dlog is not None and use_depth:
                    loss = loss + args.depth_w * net0.depth_loss(
                        dlog.float(), depth_gt,
                        ent_w=args.depth_ent_w, far_w=args.depth_far_w,
                        band_bal=args.depth_band_balance)
                if args.pact_w > 0:
                    # Push alpha down to prune outliers. alpha is initialized from the
                    # measured max, so without this term it never moves.
                    loss = loss + args.pact_w * net0.pact_penalty()
                if seg2d is not None and use_seg2d:
                    loss = loss + args.seg2d_w * net0.seg2d_loss(seg2d.float(), seg2d_gt)
                if boxl is not None and use_box:
                    loss = loss + args.box_w * net0.box_loss(boxl.float(), box_gt)
                # Samples whose scene has NO 3D-box annotation carry det_n=-1
                # (dataset sentinel). Feeding them as "zero objects" is what
                # taught the detector to stay silent on x2gen2: heatmap score
                # 0.919 on the Japanese rig vs 0.080 there, boxes over the demo
                # threshold in 0 % of frames. Skip those rows in every loss
                # that reads box GT (det, traj, stationary, flow) -- absence of
                # a label is not a negative label.
                _bv = (det_n >= 0) if det_n is not None else None
                _bn = det_n.clamp(min=0) if det_n is not None else None
                # DDP SAFETY: every rank must build the SAME graph. Skipping a
                # loss on the ranks whose batch happens to be all-unannotated
                # (batch 2, x2gen2 oversampled x3) desynchronised the
                # all-reduce and hung the round with the GPUs at 100 % and no
                # step ever printed. So the loss is ALWAYS computed; when no row
                # is annotated it is multiplied by 0.0, which keeps the module
                # in the graph and contributes exactly zero gradient.
                _bsel = _bv if (_bv is not None and bool(_bv.any())) else \
                    (torch.ones_like(_bv) if _bv is not None else None)
                _bw = 1.0 if (_bv is not None and bool(_bv.any())) else 0.0
                if hm is not None and use_boxdet and _bsel is not None:
                    _gtw = None
                    # pl_gt, not lidbev: lidbev has already been
                    # zeroed by the modality dropout, and a dropped
                    # sample would then lose every box from its
                    # target instead of just its LiDAR input.
                    if args.gt_lidar_w > 0 and pl_gt is not None:
                        # channel 0 of lidar_bev is log1p(point count) on the
                        # 0.4 m grid; sum a 7x7 window (+-1.2 m) at each box
                        # centre and ramp the weight to 1.0 at --gt-lidar-w.
                        _cnt = torch.expm1(pl_gt[:, 0].float())
                        _pad = F.pad(_cnt, (3, 3, 3, 3))
                        _rr = ((80.0 - det_boxes[..., 1]) / DET_RES
                               ).long().clamp(0, 399)
                        _cc = ((50.0 - det_boxes[..., 2]) / DET_RES
                               ).long().clamp(0, 249)
                        _acc = torch.zeros_like(det_boxes[..., 0])
                        for _dr in range(7):
                            for _dc in range(7):
                                _acc = _acc + _pad[
                                    torch.arange(_pad.shape[0],
                                                 device=_pad.device
                                                 ).view(-1, 1),
                                    _rr + _dr, _cc + _dc]
                        _gtw = (_acc / args.gt_lidar_w).clamp(0.0, 1.0)
                    loss = loss + args.box_w * _bw * net0.boxdet_loss(
                        hm[_bsel], rg[_bsel], det_boxes[_bsel], _bn[_bsel],
                        gt_w=(None if _gtw is None else _gtw[_bsel]),
                        corner_w=args.box_corner_w)
                if args.model in ("v38", "v39", "v40", "v41", "v42", "v43", "v44", "v45", "v46", "v47", "v48", "v49", "v51", "v52", "v53", "v54", "v55", "v56", "v63b", "v64r50", "v52r50", "v52r50s8", "v52rvgg", "v55rvgg", "v52s8") and use_ego and ego_gt is not None:
                    loss = loss + 0.3 * net0.vprof_loss(ego_gt)
                if traj_pred is not None and use_traj and _bsel is not None:
                    loss = loss + args.traj_w * _bw * net0.traj_loss(
                        traj_pred[_bsel], det_boxes[_bsel], _bn[_bsel],
                        traj_gt[_bsel], tvalid_gt[_bsel])
                    if len(out) >= 11:      # v26 stationary-flag head
                        sw = (args.stat_w if args.stat_w is not None
                              else args.traj_w)
                        loss = loss + sw * _bw * net0.stat_loss(
                            out[10][_bsel], det_boxes[_bsel], _bn[_bsel],
                            traj_gt[_bsel], tvalid_gt[_bsel],
                            margin=args.stat_margin)
                if use_tl and len(out) >= 12:   # v27 traffic-light state
                    loss = loss + args.tl_w * net0.tl_loss(out[11], tl_gt)
                if use_risk and len(out) >= 13:  # v28 area risk map
                    loss = loss + args.risk_w * net0.risk_loss(out[12],
                                                               _fit(risk_gt, out[12]))
                if use_flow and len(out) >= 14 and traj_gt is not None \
                        and _bsel is not None:
                    loss = loss + args.flow_w * _bw * net0.flow_loss(
                        out[13][_bsel], det_boxes[_bsel], _bn[_bsel],
                        traj_gt[_bsel], tvalid_gt[_bsel])
                if use_lg and len(out) >= 17:
                    loss = loss + args.lanegraph_w * net0.lanegraph_loss(
                        out[14], out[15], out[16],
                        lg_pts_gt, lg_cls_gt, lg_n_gt, lg_adj_gt)
                if use_unk and len(out) >= 18:
                    loss = loss + args.unk_w * net0.unk_loss(
                        out[17], unk_c, unk_n)
                if use_unk_v2 and len(out) >= 18:
                    loss = loss + args.unk_dense_w * net0.unk_dense_loss(
                        out[17], _fit(unk_v2, out[17]))
                if use_rl and ego_pred is not None:
                    from bevlane.e2e_reward import (candidate_rewards,
                                                    grpo_mode_loss)
                    _K = EGO_K
                    _e = ego_pred.float()
                    _wp = _e[:, :12 * _K].view(-1, _K, 6, 2)
                    _lg = _e[:, 12 * _K:12 * _K + _K]
                    if intent_oh is not None:      # undo the command boost
                        _lg = _lg - getattr(net0, 'MODE_BOOST', 0.0) \
                            * intent_oh.to(_lg.dtype)
                    _rew, _parts = candidate_rewards(
                        _wp.detach(), gt=gt, boxes=det_boxes,
                        nbox=(det_n.clamp(min=0) if det_n is not None
                              else det_n),          # -1 sentinel -> no boxes

                        traj=traj_gt, tvalid=tvalid_gt,
                        tl=(tl_t if use_tlin else None),
                        v0=ego_gt[:, 12], ego_gt=ego_gt,
                        w={"imit": args.rl_imit_w, "tl": args.rl_tl_w})
                    _rl, _st = grpo_mode_loss(_lg, _rew,
                                              valid=ego_gt[:, 16] > 0.5,
                                              ent_w=args.rl_ent)
                    loss = loss + args.rl_w * _rl
                    if _st["n"] > 0:
                        rl_acc[0] += _st["rew_sel"] * _st["n"]
                        rl_acc[1] += _st["rew_best"] * _st["n"]
                        rl_acc[2] += _st["pick_acc"] * _st["n"]
                        rl_acc[3] += _st["n"]
                if use_pl and len(out) >= 19:
                    loss = loss + args.pseudo_lidar_w * \
                        net0.pseudo_lidar_loss(out[18], _fit(pl_gt, out[18]))
                    if pl_gt is not None and step % 20 == 0:
                        with torch.no_grad():
                            g = _fit(pl_gt, out[18]).to(device).float()
                            v = (g.abs().sum((1, 2, 3)) > 0)
                            if v.any():
                                og = (g[v, 3] > 0.5)
                                op = (out[18][v, 3].float() > 0)
                                pl_acc[0] += float((op & og).sum())
                                pl_acc[1] += float((op | og).sum())
                                pa = net0.pl_activate(out[18][v].float())
                                pl_acc[2] += float(
                                    ((pa[:, 1] - g[v, 1]).abs()
                                     * og.float()).sum())
                                pl_acc[3] += float(og.sum())
                if (args.intent_w > 0 and args.model in ("v43", "v44", "v45", "v46", "v47", "v48", "v49", "v51", "v52", "v53", "v54", "v55", "v56", "v63b", "v64r50", "v52r50", "v52r50s8", "v52rvgg", "v55rvgg", "v52s8")
                        and intent_oh is not None and ego_pred is not None):
                    loss = loss + args.intent_w * net0.intent_loss(
                        ego_pred, intent_oh, margin=args.intent_margin)
                if (args.offroad_w > 0 and intent_oh is not None
                        and ego_pred is not None and gt is not None):
                    _rows = intent_oh.amax(1) > 0.5      # only rows with a command
                    loss = loss + args.offroad_w * offroad_loss(
                        gt, ego_pred, rows=_rows)
                if (args.intent_mode_w > 0 and args.model in ("v44", "v45", "v46", "v47", "v48", "v49", "v51", "v52", "v53", "v54", "v55", "v56", "v63b", "v64r50", "v52r50", "v52r50s8", "v52rvgg", "v55rvgg", "v52s8")
                        and intent_oh is not None and ego_pred is not None):
                    loss = loss + args.intent_mode_w * net0.intent_mode_loss(
                        ego_pred, intent_oh)
                if hm2d is not None and use_bbox2d:
                    loss = loss + args.bbox2d_w * net0.bbox2d_loss(
                        hm2d, rg2d, bb2d, nb2d)
                # A rear-truncated grid makes the head emit fewer rows than the
                # 800-row label carries; keep the front rows that still exist.
                # No-op at the default extent.
                gt = _fit(gt, logits)
                if args.lane_sdf_w > 0 and \
                        getattr(net0, "_lane_sdf_pred", None) is not None:
                    import cv2 as _cv2
                    _pred = net0._lane_sdf_pred.float()
                    _pred = _pred[..., :gt.shape[-2], :]
                    with torch.no_grad():
                        _t = []
                        for _b in range(gt.shape[0]):
                            _lane = (gt[_b] == 4).to(torch.uint8).cpu().numpy()
                            _d = _cv2.distanceTransform(
                                1 - _lane, _cv2.DIST_L2, 3).astype("float32")
                            _t.append(torch.from_numpy(_d))
                        _sdf_raw = torch.stack(_t).to(_pred.device) * 0.2
                        _sdf = _sdf_raw.clamp(max=3.0)   # cells -> metres
                        # Consensus disagreement/unobserved cells must not
                        # supervise even the auxiliary distance field. Without
                        # this mask a lane beside 255 paints a smooth target
                        # back into the very region CE/clDice intentionally
                        # ignore.
                        # Build the band BEFORE clipping.  The old order
                        # clipped every distance to 2 m and then tested <3 m,
                        # accidentally supervising the entire raster and
                        # letting easy background dominate the lane signal.
                        _band = (_sdf_raw < 3.0) & (gt != 255)
                    _l = (torch.nn.functional.l1_loss(
                        _pred[:, 0], _sdf, reduction="none") * _band).sum() \
                        / _band.sum().clamp(min=1)
                    loss = loss + args.lane_sdf_w * _l
                if args.seg_w > 0:
                    ce = F.cross_entropy(logits, gt, weight=cw,
                                         ignore_index=ignore, reduction="none")
                    H2 = ce.shape[-2]
                    wmap = torch.ones_like(ce)
                    if args.far_w > 0:
                        # Upweight far cells BY METRIC DISTANCE FROM THE EGO,
                        # not by distance from the tensor centre. The two were
                        # the same until the grid became rear-truncated: with
                        # 80 m ahead / 20 m behind, the tensor centre sits at
                        # +30 m, so the old row formula put its minimum there --
                        # near-ego cells trained as if they were "far" (x1.6)
                        # while the +30 m band, where most laneline GT lives,
                        # got x1.0. Identical to the old behaviour on the
                        # symmetric default grid.
                        from bevlane.model import BEV_XF as _XF, BEV_XR as _XR
                        rows = torch.arange(H2, device=ce.device,
                                            dtype=ce.dtype)
                        _x = _XF - (rows + 0.5) * (_XF + _XR) / H2
                        wrow = 1 + args.far_w * _x.abs() / max(_XF, _XR)
                        wmap = wmap * wrow.view(1, -1, 1)
                    if args.boundary_w > 0:      # sharpen class boundaries
                        wmap = wmap * boundary_weight(
                            gt, radius=2, w=1 + args.boundary_w,
                            ignore_unlabeled=args.seg_ignore_unlabeled)
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
                if args.tversky_area_w > 0:
                    # S2 (2026-08-27): FP penalty on road/crosswalk. Precision request
                    loss = loss + args.tversky_area_w * tversky_loss(
                        logits.float(), gt, classes=(1, 3),
                        alpha=0.2, beta=0.8)
                if args.cldice_w > 0:
                    loss = loss + args.cldice_w * lane_cldice_loss(
                        logits.float(), gt)
            opt.zero_grad(set_to_none=True)
            # DDP: the skip decision must be UNANIMOUS. Deciding per rank
            # desynchronises the gradient all-reduce and the round dies with
            # "Expected to have finished reduction in the prior iteration"
            # -- which is exactly what happened the moment a round warm-started
            # a re-initialised head (r50/v49: fresh decoder, large early loss,
            # some ranks non-finite and some not).
            _fin = torch.tensor([float(torch.isfinite(loss))], device=device)
            if ddp:
                dist.all_reduce(_fin, op=dist.ReduceOp.MIN)
            if _fin.item() < 0.5:
                # the forward already moved the BN running stats; put them
                # back so one bad batch cannot poison eval-mode inference
                with torch.no_grad():
                    for _n, _b in (model.module if ddp else model
                                   ).named_buffers():
                        if _n in _bn_bak:
                            _b.copy_(_bn_bak[_n])
                opt.zero_grad(set_to_none=True)
                sched.step(); step += 1
                if is_main:
                    # Say WHICH tensor went bad. r51 skipped 2.0 % of its steps
                    # and the only clue was an "RL sel=+nan" line elsewhere in
                    # the log; the loss is accumulated over 26 sites so the
                    # scalar alone identifies nothing. The model outputs are the
                    # common origin, so name the non-finite ones here.
                    _bad = []
                    for _i, _o in enumerate(out if isinstance(out, tuple)
                                            else (out,)):
                        if not torch.is_tensor(_o):
                            continue
                        if not bool(torch.isfinite(_o.detach()).all()):
                            _nn = int((~torch.isfinite(_o.detach())).sum())
                            _bad.append(f"out[{_i}]"
                                        f"{tuple(_o.shape)}:{_nn}")
                    if _MODBAD:
                        print(f"[modprobe] first broken module: "
                              f"{_MODBAD[0]}", flush=True)
                        _MODBAD.clear()
                    print(f"ep{ep} step{step} SKIP non-finite loss "
                          "(BN stats restored) "
                          + (f"non-finite outputs: {', '.join(_bad)}"
                             if _bad else "outputs all finite -> the loss "
                             "itself diverged, not the forward"), flush=True)
                continue
            scaler.scale(loss).backward()
            _scale_before = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            # GradScaler silently skips optimizer.step on overflow. Advancing
            # OneCycleLR (and EMA) in that case desynchronises both from the
            # actual parameter updates and emits the scheduler-before-step
            # warning at startup. The logical data step still advances below.
            _optimizer_stepped = scaler.get_scale() >= _scale_before
            if _sp_pairs:
                if _sp_final and step >= args.sparse_ramp_steps:
                    # 1:4 -> 2:4 switch (once). From here on re-apply the 2:4 masks
                    _sp_pairs = [(_p, _mf) for (_p, _), _mf in zip(_sp_pairs, _sp_final)]
                    _sp_final = []
                    if is_main:
                        print(f"[sparse-24] step {step}: switched 1:4 -> 2:4", flush=True)
                with torch.no_grad():
                    for _p, _m in _sp_pairs:
                        _p.mul_(_m)
            if ema is not None and _optimizer_stepped:
                ema.update(model.module if ddp else model)
            if _optimizer_stepped:
                sched.step()
            elif is_main:
                print(f"ep{ep} step{step + 1} AMP-SKIP gradient overflow "
                      f"scale {_scale_before:g}->{scaler.get_scale():g}; "
                      "LR/EMA held", flush=True)
            step += 1
            # Preventive conv->BN renorm guard (--bn-guard, 2026-08-13): in this model
            # line seg_head.out's running_var runs away to ~1e7 within hours and
            # overflows fp16 (seen in r59/v63a/v63b/v65 x2). Apply the same
            # function-preserving rescale as the post-hoc fix (renorm_convbn) in-loop
            # when the threshold is exceeded. All ranks compute the same deterministic
            # result -> no communication. Adam state of the rescaled conv is reset (as on restart).
            if args.bn_guard > 0 and step % 200 == 0:
                with torch.no_grad():
                    _n0 = model.module if ddp else model
                    _mods = dict(_n0.named_modules())
                    for _bnm, _bn in _mods.items():
                        if not isinstance(_bn, (torch.nn.BatchNorm2d,
                                                torch.nn.SyncBatchNorm)):
                            continue
                        if _bn.running_var is None:
                            continue
                        _mx = float(_bn.running_var.max())
                        if _mx <= args.bn_guard:
                            continue
                        _h, _, _i = _bnm.rpartition(".")
                        if not _i.isdigit():
                            continue
                        _cv = _mods.get(f"{_h}.{int(_i) - 1}")
                        if not isinstance(_cv, torch.nn.Conv2d):
                            continue
                        _s = _mx ** 0.5
                        _cv.weight.div_(_s)
                        if _cv.bias is not None:
                            _cv.bias.div_(_s)
                        _bn.running_mean.div_(_s)
                        _bn.running_var.div_(_s * _s)
                        for _p in (_cv.weight, _cv.bias):
                            if _p is not None and _p in opt.state:
                                opt.state[_p] = {}
                        if is_main:
                            print(f"[bn-guard] {_bnm} var {_mx:.3g} -> 1.0 "
                                  f"(s={_s:.3g}) step{step}", flush=True)
            if is_main and step % 50 == 0:
                if os.environ.get("METEOR_MEM_PROBE"):
                    print(f"PEAK {torch.cuda.max_memory_allocated()/2**30:.2f}"
                          " GiB", flush=True)
                print(f"ep{ep} step{step}/{total_steps} loss={loss.item():.4f} "
                      f"lr={sched.get_last_lr()[0]:.2e} "
                      f"({(time.time() - t0) / step:.2f}s/it)", flush=True)
            # ---- step-level probe of the two priority metrics -------------
            # An epoch is ~75 min; a regression (or a fix) must be visible in
            # minutes, not hours. Rank 0 runs a small val slice while the
            # other ranks block on the next all-reduce, then a barrier
            # re-syncs everyone.
            if args.val_every and step % args.val_every == 0:
                # release this step's activations (18 output maps + graph
                # refs) BEFORE the eval spike: v48's extra head left only
                # tens of MB of headroom on 44 GB cards
                # Dropping `out` alone is not enough: the unpacked views
                # (logits/dlog/seg2d/hm/rg/hm2d/rg2d/occ_pred/traj_pred) still
                # reference the step's graph, so empty_cache freed almost
                # nothing and the E2E probe -- which every rank runs -- OOM'd
                # on whichever ranks were most fragmented (measured at batch 2:
                # ranks 2,3,4,7 skipped at step 300 while 0,1,5,6 passed).
                out = ego_pred = loss = None
                logits = dlog = seg2d = boxl = hm = rg = None
                hm2d = rg2d = occ_pred = traj_pred = None
                torch.cuda.empty_cache()
                _bad = sanitize_bn(model.module if ddp else model)
                if _bad and is_main:
                    print(f"ep{ep} step{step} BN REPAIRED: {len(_bad)} "
                          f"buffers, first={_bad[:3]}", flush=True)
                if is_main:
                  try:
                    torch.cuda.empty_cache()
                    netq = model.module if ddp else model
                    # 5, was 10. Only rank 0 runs the probe; the other six sit
                    # at the barrier for its whole duration, so every
                    # batch here is paid seven times over. Sampled GPU
                    # utilisation shows all six at 0 % while this runs.
                    iq = evaluate(netq, dv, device, max_batches=5)
                    mq = float(np.nanmean(list(iq.values())))
                    msg = (f"[probe ep{ep} step{step}] mIoU={mq:.3f} "
                           f"road={iq.get('road', float('nan')):.3f} "
                           f"lane={iq.get('laneline', float('nan')):.3f}")
                    if use_lidar and dv_lid is not None:
                        il = evaluate(netq, dv_lid, device, max_batches=5,
                                      use_lidar=True)
                        msg += (f" | +lidar mIoU="
                                f"{float(np.nanmean(list(il.values()))):.3f}")
                    try:
                        _th = thickness_ratio(netq, dv, device, max_batches=3)
                        msg += (" | thickness lane={:.2f} stop={:.2f} edge={:.2f}"
                                .format(_th[4], _th[5], _th[6]))
                    except Exception as _e:
                        msg += f" | thickness n/a({str(_e)[:20]})"
                    if use_rl and rl_acc[3] > 0:
                        msg += (f" | RL sel={rl_acc[0] / rl_acc[3]:+.3f}"
                                f" best={rl_acc[1] / rl_acc[3]:+.3f}"
                                f" pick={rl_acc[2] / rl_acc[3]:.2f}")
                        rl_acc[:] = [0.0, 0.0, 0.0, 0.0]
                    if use_pl and pl_acc[1] > 0:
                        msg += (f" | PL occIoU={pl_acc[0] / pl_acc[1]:.3f}"
                                f" zMAE={pl_acc[2] / max(pl_acc[3], 1):.2f}m")
                        pl_acc[:] = [0.0, 0.0, 0.0, 0.0]
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
                                                     + int(use_unk_v2)
                                                     + int(use_lidarbev)
                                                     + int(use_sdmap) + int(use_tlin))
                                            if use_temporal else None)
                        msg += (f" | vehRn={dq['vehn']:.2f} P={dq['veh'][0]:.2f}"
                                f" vruRn={dq['vrun']:.2f}"
                                f" yaw={dq['veh_yaw']:.1f}deg")
                    print(msg, flush=True)
                  except torch.cuda.OutOfMemoryError:
                    # a probe is diagnostics: never let it kill a
                    # multi-day run (v48 raised the peak enough that
                    # the eval spike on top of the training reservation
                    # tipped over 44 GB and the watchdog looped)
                    print(f"[probe ep{ep} step{step}] SKIPPED (OOM)",
                          flush=True)
                    torch.cuda.empty_cache()
                # distributed ADE/ADEc probe: every rank evaluates its own
                # shard of the val set (the other 7 GPUs used to idle here),
                # sums are all-reduced, rank 0 prints -> 8x coverage at the
                # same wall time
                if use_ego and use_temporal:
                    netq2 = model.module if ddp else model
                    torch.cuda.empty_cache()
                    try:
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
                                 + int(use_unk_v2) + int(use_lidarbev)
                                 + int(use_sdmap) + int(use_tlin)))
                    except torch.cuda.OutOfMemoryError:
                      # r48 died twice here: this eval sits outside the probe
                      # guard, and its spike on top of the training reservation
                      # tipped over 44 GB. The all_reduce below is collective,
                      # so a skipping rank MUST still contribute zeros or the
                      # other seven hang. evaluate_ego also leaves the model in
                      # eval mode when it throws mid-way.
                      print(f"[probeE2E ep{ep} step{step}] rank{rank} "
                            f"SKIPPED (OOM)", flush=True)
                      netq2.train()
                      torch.cuda.empty_cache()
                      sums = torch.zeros(8, dtype=torch.float64, device=device)
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
          try:
            torch.cuda.empty_cache()
            net = model.module if ddp else model
            ious = evaluate(net, dv_ep, device, max_batches=vcap(80))
            miou = float(np.nanmean(list(ious.values())))
            rp, rr, _ = class_pr(net, dv_ep, device, 1, max_batches=vcap(40))
            ep_, er_, eratio = class_pr(net, dv_ep, device, 6,
                                         max_batches=vcap(40))
            print(f"[val ep{ep}] mIoU={miou:.3f} " +
                  " ".join(f"{k}={v:.3f}" for k, v in ious.items()) +
                  f" | road P={rp:.3f} R={rr:.3f}" +
                  f" | redge P={ep_:.3f} R={er_:.3f} x{eratio:.2f}", flush=True)
            if use_seg2d:
                s2 = evaluate_seg2d(net, dv_ep, device, args.n_seg2d,
                                    max_batches=vcap(40))
                if s2:
                    key = {0: "bg", 8: "mark", 11: "road", 12: "swalk",
                           13: "lane", 20: "pole"}
                    val2d_miou = float(np.mean(list(s2.values())))
                    print(f"[val2d ep{ep}] mIoU={val2d_miou:.3f} "
                          + " ".join(f"{key[c]}={s2[c]:.3f}"
                                     for c in key if c in s2), flush=True)
            if use_occ:
                oc = evaluate_occ(net, dv_ep, device,
                                  4 + int(use_seg2d) + 4 * int(use_traj)
                                  + int(use_ego), max_batches=vcap(25))
                if oc:
                    onm = {0: "free", 2: "veh", 4: "ped", 5: "road",
                           7: "veg", 8: "bldg", 9: "pole"}
                    print(f"[valOCC ep{ep}] mIoU={np.mean(list(oc.values())):.3f} "
                          + " ".join(f"{onm[c]}={oc[c]:.3f}"
                                     for c in onm if c in oc), flush=True)
            vtmp = (4 + int(use_seg2d) + 4 * int(use_traj) + int(use_ego)
                    + int(use_occ) + int(use_tl) + int(use_risk)
                    + 4 * int(use_lg) + 2 * int(use_unk) + int(use_unk_v2)
                    + int(use_lidarbev) + int(use_sdmap) + int(use_tlin)) \
                if use_temporal else None
            if use_traj and use_boxdet:
                d3 = evaluate_det3d(net, dv_ep, device, 4 + int(use_seg2d),
                                    max_batches=vcap(30), tmp_idx=vtmp)
                print(f"[val3D ep{ep}] "
                      f"veh P={d3['veh'][0]:.2f} R={d3['veh'][1]:.2f} "
                      f"R50={d3['veh50']:.2f} Rn={d3['vehn']:.2f} "
                      f"err={d3['veh'][2]:.2f}m "
                      f"L/W={d3['veh_shape'][0]:.2f}/{d3['veh_shape'][1]:.2f}m "
                      f"corner={d3['veh_shape'][2]:.2f}m "
                      f"yaw={d3['veh_yaw']:.1f}deg flip={d3['veh_flip']:.2f} | "
                      f"vru P={d3['vru'][0]:.2f} R={d3['vru'][1]:.2f} "
                      f"R50={d3['vru50']:.2f} Rn={d3['vrun']:.2f} "
                      f"err={d3['vru'][2]:.2f}m", flush=True)
                if "veh_zones" in d3:
                    _zn = ("F0-40", "F40-80", "R0-40", "R40-80")
                    print("[val3D-zone ep{}] ".format(ep) + " ".join(
                        f"{nm}:R={v[0]:.2f}/e={v[1]:.2f}/n={v[2]}"
                        f"/corner={v[3]:.2f}/L={v[4]:.2f}/W={v[5]:.2f}"
                        f"/yaw={v[6]:.1f}"
                        for nm, v in zip(_zn, d3["veh_zones"])), flush=True)
                if "stat" in d3 and d3["stat"][4] > 0:
                    sp, sr, sa, sspec, sn = d3["stat"]
                    print(f"[valStat ep{ep}] P={sp:.2f} R={sr:.2f} "
                          f"acc={sa:.2f} movAcc={sspec:.2f} n={sn}",
                          flush=True)
                    _zn = ("F0-40", "F40-80", "R0-40", "R40-80")
                    print(f"[valStat-zone ep{ep}] " + " ".join(
                        f"{nm}:P={v[0]:.2f}/R={v[1]:.2f}/"
                        f"movAcc={v[2]:.2f}/n={v[3]}/statFrac={v[4]:.2f}"
                        for nm, v in zip(_zn, d3["stat_zones"])), flush=True)
                if "stat_cal" in d3:
                    def _stat_cal_fmt(name, v):
                        if v is None:
                            return f"{name}=unavailable"
                        return (f"{name}:t={v[0]:.3f}/P={v[1]:.2f}/"
                                f"R={v[2]:.2f}/acc={v[3]:.2f}/"
                                f"movAcc={v[4]:.2f}/bal={v[5]:.2f}")
                    print(f"[valStat-cal ep{ep}] " + " ".join(
                        _stat_cal_fmt(nm, d3["stat_cal"][nm])
                        for nm in ("balanced", "safe95")), flush=True)
            if use_traj:
                tj = evaluate_traj(net, dv_ep, device, 4 + int(use_seg2d),
                                   max_batches=vcap(40), tmp_idx=vtmp)
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
                tr_ = evaluate_tl(net, dv_ep, device, tl_idx, tmp_idx=vtmp,
                                  max_batches=vcap(40))
                if tr_:
                    print(f"[valTL ep{ep}] acc={tr_['acc']:.2f} "
                          + " ".join(f"{k}={tr_[k]:.2f}" for k in
                                     ("none", "green", "yellow", "red")
                                     if k in tr_), flush=True)
            if use_unk:
                uk_idx = (4 + int(use_seg2d) + 4 * int(use_traj)
                          + int(use_ego) + int(use_occ) + int(use_tl)
                          + int(use_risk) + 4 * int(use_lg))
                uk = evaluate_unknown(net, dv_ep, device, uk_idx, tmp_idx=vtmp,
                                  max_batches=vcap(20))
                if uk:
                    print(f"[valUnk ep{ep}] P={uk['p']:.2f} R={uk['r']:.2f}",
                          flush=True)
            if use_unk_v2:
                umv_idx = (4 + int(use_seg2d) + 4 * int(use_traj)
                           + int(use_ego) + int(use_occ) + int(use_tl)
                           + int(use_risk) + 4 * int(use_lg) + 2 * int(use_unk))
                ukd = evaluate_unknown_dense(net, dv_ep, device, umv_idx,
                                             tmp_idx=vtmp,
                                  max_batches=vcap(20))
                if ukd:
                    print(f"[valUnkD ep{ep}] P={ukd['p']:.2f} "
                          f"R={ukd['r']:.2f} Rfar={ukd['r_far']:.2f}",
                          flush=True)
            if use_risk:
                rk_idx = (4 + int(use_seg2d) + 4 * int(use_traj)
                          + int(use_ego) + int(use_occ) + int(use_tl))
                rk = evaluate_risk(net, dv_ep, device, rk_idx, tmp_idx=vtmp,
                                  max_batches=vcap(30))
                if rk:
                    print(f"[valRisk ep{ep}] L1={rk['l1']:.3f} "
                          f"L1(hi)={rk['l1_hi']:.3f}", flush=True)
            if use_flow:
                fl_ = evaluate_flow(net, dv_ep, device, 4 + int(use_seg2d),
                                    max_batches=vcap(20), tmp_idx=vtmp)
                if fl_:
                    print(f"[valFlow ep{ep}] EPE(mov)={fl_['epe_mov']:.2f} "
                          f"EPE(stat)={fl_['epe_stat']:.2f} m/s", flush=True)
            if use_lg:
                lg_idx = (4 + int(use_seg2d) + 4 * int(use_traj)
                          + int(use_ego) + int(use_occ) + int(use_tl)
                          + int(use_risk))
                lg_ = evaluate_lanegraph(net, dv_ep, device, lg_idx,
                                         tmp_idx=vtmp,
                                  max_batches=vcap(20))
                if lg_:
                    print(f"[valLane ep{ep}] P={lg_['p']:.2f} "
                          f"R={lg_['r']:.2f} adjAcc={lg_['adj']:.2f}",
                          flush=True)
            if use_ego:
                # ADEc reads nan when the capped slice contains no curve
                # frames -- exactly what --val-batch 1 caused (valE2E ep3/ep4
                # ADEc=nan). vcap keeps the sample count fixed.
                # Cover the WHOLE strided slice. vcap(40) capped this at 80
                # batches of the 167 the stride was built to span, which was
                # not making the number optimistic -- measured on r60 ep3, the
                # first 80 give ADE 0.659 and all 167 give 0.625 -- it was just
                # measuring half of it. E2E is the metric a round is selected
                # on; it gets the full slice.
                eg = evaluate_ego(net, dv_ep, device,
                                  4 + int(use_seg2d) + 4 * int(use_traj),
                                  max_batches=len(dv_ep) + 1, tmp_idx=vtmp)
                if eg:
                    print(f"[valE2E ep{ep}] ADE={eg['ade']:.2f}m "
                          f"ADEc={eg['ade_c']:.2f}m "
                          f"FDE={eg['fde']:.2f}m steer={eg['steer']:.3f}rad "
                          f"acc={eg['acc']:.2f}m/s2 brakeAcc={eg['brake']:.2f}",
                          flush=True)
                    print(f"[valE2Ed ep{ep}] oracle={eg['ade_o']:.2f} "
                          f"(selection loss {eg['ade'] - eg['ade_o']:+.2f}) "
                          f"moving={eg['ade_mv']:.2f} stopped={eg['ade_st']:.2f} "
                          f"const-vel-straight={eg['ade_cv']:.2f} "
                          f"hs-wp0-bias={eg['hs_bias']:+.3f}m "
                          f"(n={eg['hs_n']:.0f})", flush=True)
            if use_ego and eg and dv_hs is not None:
                _hs = evaluate_ego(net, dv_hs, device,
                                   4 + int(use_seg2d) + 4 * int(use_traj),
                                   max_batches=len(dv_hs) + 1, tmp_idx=vtmp)
                if _hs:
                    eg["hs_bias"], eg["hs_n"] = _hs["hs_bias"], _hs["hs_n"]
                    print(f"[valHS ep{ep}] high-speed wp0 bias={_hs['hs_bias']:+.3f}m "
                          f"ADE={_hs['ade']:.3f} (n={_hs['hs_n']:.0f})", flush=True)
            # The averaged weights get the SAME evaluation, on the same
            # slice, so "EMA is better" is a measurement rather than a habit.
            eg_e = None
            if ema is not None and use_ego:
                _bk = ema.swap_in(net)
                try:
                    eg_e = evaluate_ego(net, dv_ep, device,
                                        4 + int(use_seg2d) + 4 * int(use_traj),
                                        max_batches=len(dv_ep) + 1,
                                        tmp_idx=vtmp)
                    if eg_e and dv_hs is not None:
                        _hse = evaluate_ego(net, dv_hs, device,
                                            4 + int(use_seg2d) + 4 * int(use_traj),
                                            max_batches=len(dv_hs) + 1, tmp_idx=vtmp)
                        if _hse:
                            eg_e["hs_bias"], eg_e["hs_n"] = _hse["hs_bias"], _hse["hs_n"]
                            if is_main:
                                print(f"[valHS-ema ep{ep}] high-speed wp0 bias="
                                      f"{_hse['hs_bias']:+.3f}m ADE={_hse['ade']:.3f} "
                                      f"(n={_hse['hs_n']:.0f})", flush=True)
                    if eg_e and is_main:
                        print(f"[valE2E-ema ep{ep}] ADE={eg_e['ade']:.2f}m "
                              f"ADEc={eg_e['ade_c']:.2f}m "
                              f"FDE={eg_e['fde']:.2f}m "
                              f"oracle={eg_e['ade_o']:.2f} "
                              f"(selection loss {eg_e['ade'] - eg_e['ade_o']:+.2f}) "
                              f"hs-wp0-bias={eg_e['hs_bias']:+.3f}m",
                              flush=True)
                    # Run the seg2d survival guard on the EMA weights **themselves** (2026-09-04).
                    # Previously the EMA was saved based on the raw net's val2d, so the v131/v141
                    # best_e2e had a dead seg2d (failed SHIP-CHECK).
                    # If only the EMA side is dead, graft the raw seg_head and save
                    # (same as the manual graft done for v141; chain/ADEc verified unchanged).
                    ema_sd = None
                    if use_seg2d:
                        s2e = evaluate_seg2d(net, dv_ep, device, args.n_seg2d,
                                             max_batches=vcap(20))
                        v2e = float(np.mean(list(s2e.values()))) if s2e \
                            else float("nan")
                        if is_main:
                            print(f"[val2d-ema ep{ep}] mIoU={v2e:.3f}",
                                  flush=True)
                        if not (v2e > 0.30):
                            if val2d_miou is not None and val2d_miou > 0.30:
                                ema_sd = {k: v for k, v in
                                          net.state_dict().items()}
                                nrep = 0
                                for k in ema_sd:
                                    if k.startswith("seg_head.") and k in _bk:
                                        ema_sd[k] = _bk[k].to(ema_sd[k].dtype)
                                        nrep += 1
                                if is_main:
                                    print(f"[val2d-ema ep{ep}] seg2d collapsed -> "
                                          f"grafted raw seg_head ({nrep} tensor)",
                                          flush=True)
                            else:
                                ema_sd = False   # raw is dead too: do not save
                    if ema_sd is None:
                        ema_sd = net.state_dict()
                    if eg_e and eg_e.get("ade_c") == eg_e.get("ade_c") \
                            and eg_e["ade_c"] < best_e2e and is_main \
                            and ema_sd is not False:
                        best_e2e = eg_e["ade_c"]
                        torch.save({"model": ema_sd, "epoch": ep,
                                    "ious": ious, "ade_c": best_e2e,
                                    "ema": args.ema,
                                    "args": vars(args), "git": _GIT},
                                   os.path.join(args.out, "best_e2e.pt"))
                        print(f"[best-e2e] ep{ep} EMA ADEc={best_e2e:.3f}m "
                              "saved", flush=True)
                    # ckpt selected by the chain proxy (2026-09-04): ADEc + |high-speed wp0 bias|.
                    # The min-ADEc epoch is not necessarily best for chain divergence (v135/v132
                    # inversion), so add the longitudinal bias, the main chain driver, at equal weight.
                    if eg_e and is_main and ema_sd is not False \
                            and eg_e.get("ade_c") == eg_e.get("ade_c") \
                            and eg_e.get("hs_bias") == eg_e.get("hs_bias"):
                        cs = eg_e["ade_c"] + abs(eg_e["hs_bias"])
                        if cs < best_chain:
                            best_chain = cs
                            torch.save({"model": ema_sd, "epoch": ep,
                                        "ious": ious, "ade_c": eg_e["ade_c"],
                                        "hs_bias": eg_e["hs_bias"],
                                        "chain_score": cs, "ema": args.ema,
                                        "args": vars(args), "git": _GIT},
                                       os.path.join(args.out, "best_chain.pt"))
                            print(f"[best-chain] ep{ep} EMA ADEc="
                                  f"{eg_e['ade_c']:.3f} bias="
                                  f"{eg_e['hs_bias']:+.3f} score={cs:.3f} "
                                  "saved", flush=True)
                finally:
                    ema.swap_out(net, _bk)
            torch.save({"model": net.state_dict(), "epoch": ep,
                        "ious": ious, "args": vars(args), "git": _GIT},
                       os.path.join(args.out, "last.pt"))
            # composite best: BEV mIoU minus a small penalty for E2E curve
            # error, so "best" never selects a pre-curve-convergence epoch
            score = miou
            if use_ego and eg and eg.get("ade_c") == eg.get("ade_c"):
                score = miou - 0.01 * min(eg["ade_c"], 5.0)
            if score > best:
                best = score
                torch.save({"model": net.state_dict(), "epoch": ep,
                            "ious": ious, "args": vars(args), "git": _GIT},
                           os.path.join(args.out, "best.pt"))
            # E2E is the priority metric and the composite above is dominated
            # by mIoU: 0.01 * ADEc moves the score by ~0.010-0.018 while mIoU
            # itself spreads about that much, so an epoch with clearly better
            # driving loses to one with marginally better segmentation. r57
            # reached ADEc 0.92 m at ep4 -- the best any round has managed --
            # and kept neither that epoch nor anything close to it: best.pt is
            # ep0 (1.478 m on the fixed slice) and last.pt is ep7 (1.758 m).
            # Keep it explicitly, on ADEc alone.
            if use_ego and eg and eg.get("ade_c") == eg.get("ade_c") \
                    and (val2d_miou is None or val2d_miou > 0.30):
                # Forbid saving best_e2e on a seg2d-collapsed epoch (prevents a repeat of
                # v125/v128/v130, where the deployed ckpt's 2D seg was dead)
                if eg["ade_c"] < best_e2e:
                    best_e2e = eg["ade_c"]
                    torch.save({"model": net.state_dict(), "epoch": ep,
                                "ious": ious, "ade_c": best_e2e,
                                "args": vars(args), "git": _GIT},
                               os.path.join(args.out, "best_e2e.pt"))
                    print(f"[best-e2e] ep{ep} ADEc={best_e2e:.3f}m saved",
                          flush=True)
          except torch.cuda.OutOfMemoryError:
            # never let the epoch-end evaluation kill the round
            print(f'[val ep{ep}] SKIPPED (OOM)', flush=True)
            torch.cuda.empty_cache()
        if ddp:
            dist.barrier()
    if is_main:
        print(f"[done] best score={best:.3f} best_e2e ADEc={best_e2e:.3f}m", flush=True)
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

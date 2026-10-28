#!/usr/bin/env python3
"""METEORite trainer (with optional LiDAR input).

A focused multi-task trainer for bevlane/meteorite.py. Geometry runs through
METEORite.forward_geo (per-sample K/T, exact dense path); export later bakes the rig
table. Losses are reused from the METEOR line where the heads match bit-for-bit
(boxdet/traj/stat/ego from model.py as unbound methods, dice from train.py);
depth/seg2d/lane/tl/risk are written here for METEORite's 1/8 feature and D=48 bins.

Not in v1: 2D box loss (hm2d/reg2d
heads stay untrained), intent input, EMA, BEV rot aug / lat aug, occ head.

torchrun --nproc_per_node=8 bevlane/train_meteorite.py --root ... --train-list ... --out ...
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bevlane.dataset import BevLaneDataset, rmap_env                    # noqa: E402
from bevlane.model import DepthSegIPMNetV52 as DON, EGO_K               # noqa: E402
from bevlane.train import (EMA, EpochSubsetSampler, boundary_weight, dice_loss,   # noqa: E402
                           lovasz_softmax, split_scenes, tversky_loss)
from bevlane import meteorite as M                                      # noqa: E402

# batch layout for our dataset flags (imgs,K,T,gt + depth,seg2d + agent(4) + ego + tl + risk + lidar)
I_IMG, I_K, I_T, I_GT, I_DEP, I_SEG2D, I_AB, I_AN, I_AT, I_AV, I_EGO, I_TL, I_RISK, I_LB = range(14)
I_OCC = -1
I_B2 = I_C2 = None    # with --det2d-w > 0, 2D boxes (boxes [N,96,5] cls,cx,cy,w,h px / counts [N]) follow AV and shift later indices by 2


def function_preserving(cur, sd, ok):
    """Warm start across a capacity change WITHOUT changing the function (otherwise a widened BEV side
    restarts from random and loses accuracy). Rules, applied to shape-mismatched tensors:
      - same ndim: copy the overlapping block; for >=2-D weights zero the EXTRA INPUT channels (dim 1) so new
        channels feed nothing downstream; extra output channels keep their fresh init (their consumers read zeros)
      - BN of widened layers: overlap copied, new channels fresh (they are unread)
      - ego_mlp.0 (Linear on [BC1*20 flattened, v0]): old v0 column moved to the new last column
      - depth.1 (bins D_old -> D_new): logits interpolated over log-depth so the depth distribution is unchanged
      - new residual blocks (r18 -> r34: layerX.N absent in the ckpt): bn2 gamma = 0 -> the block is identity
    Returns (state dict to load, number of tensors handled here)."""
    out = dict(ok)
    n = 0
    for k, v in cur.items():
        if k in ok:
            continue
        if k in sd and sd[k].dim() == v.dim() and v.dim() > 0:
            o = sd[k]
            if k.startswith("depth.1.") and o.shape[0] != v.shape[0]:
                Do, Dn = o.shape[0], v.shape[0]
                src = torch.linspace(0, Do - 1, Dn)
                i0 = src.floor().long().clamp(max=Do - 1); i1 = (i0 + 1).clamp(max=Do - 1); f = src - i0
                fs = f.view(-1, *([1] * (o.dim() - 1)))
                out[k] = (o[i0] * (1 - fs) + o[i1] * fs).to(v.dtype)
                n += 1
                continue
            new = v.clone()
            if k == "ego_mlp.0.weight" and v.shape[1] < o.shape[1]:          # --ego-gap 1: input shrinks 2561 -> 129 (columns change meaning) -> keep fresh init
                continue
            if k == "ego_mlp.0.weight":
                new[:, :o.shape[1] - 1].zero_(); new[:, o.shape[1] - 1:].zero_()
                new[:o.shape[0], :o.shape[1] - 1] = o[:, :-1]
                new[:o.shape[0], -1] = o[:, -1]
                out[k] = new; n += 1
                continue
            if v.dim() >= 2 and v.shape[1] > o.shape[1]:
                new[:, o.shape[1]:] = 0
            sl = tuple(slice(0, min(a, b)) for a, b in zip(v.shape, o.shape))
            new[sl] = o[sl]
            out[k] = new
            n += 1
        elif k not in sd and k.endswith("bn2.weight") and ".layer" not in k and any(
                k.startswith(p) for p in ("stem.4.", "l2.", "l3.", "l4.")):
            out[k] = torch.zeros_like(v)                # new residual block = identity at start
            n += 1
    return out, n


def make_net(args, rank):
    net = M.METEORite(taps=None, vt="gather", pretrained=True, backbone=args.backbone)
    # ego-loss attributes of the production line (CE tau 0 = hard argmin)
    net.EPS_WTA = args.eps_wta                 # 0 lets candidates spread out (0.1 pulls them towards the mean)
    net.EGO_CE_MULT = args.ego_ce_mult         # multiplier on the selection-logit CE (counters the selector collapsing onto one candidate)
    net.EGO_TW = torch.tensor([1.5, 1.35, 1.2, 1.05, 0.95, 0.9])
    net.EGO_LONG_W = args.ego_long_w         # weight of the longitudinal (heading-direction) error, which dominates the total error
    net.EGO_FDE_W = args.ego_fde_w
    net.EGO_CE_TAU = args.ego_ce_tau
    # the borrowed losses call self.build_*_targets (staticmethods on the METEOR class)
    net.build_det_targets = DON.build_det_targets
    net.build_traj_targets = DON.build_traj_targets
    # The borrowed box loss reads BOX_FAR_W / VRU_FAR_BAND off self. They are class attributes of
    # the METEOR line (far veh >40 m x4; VRU 25-45 m x2, >50 m x0.3), absent on METEORite -> the
    # legacy branch DAMPS far veh >50 m x0.3 and VRU >40 m x0.2, hurting far recall.
    # --box-far 1 restores the production weighting.
    if args.box_far:
        net.BOX_FAR_W = DON.BOX_FAR_W
        net.VRU_FAR_BAND = DON.VRU_FAR_BAND
    return net


def depth_targets(d, dev, hw=None):
    """metric depth [B,N,h,w] -> bin index [B,N,FH,FW] (or hw) int64, 255 = ignore."""
    hw = hw or (M.FH, M.FW)
    if tuple(d.shape[-2:]) != tuple(hw):
        d = F.interpolate(d.to(dev), hw, mode="nearest")
    d = d.to(dev)
    step = (math.log(M.D_MAX) - math.log(M.D_MIN)) / (M.D - 1)
    ok = (d > M.D_MIN) & (d < M.D_MAX)
    b = ((d.clamp(min=M.D_MIN).log() - math.log(M.D_MIN)) / step).round().long()
    return torch.where(ok, b.clamp(0, M.D - 1), torch.full_like(b, 255))


_KD_MASK = {}


def _teacher_reg_yaw2(t_reg):
    """Convert the teacher (METEOR line) reg yaw channels (4=sin θ, 5=cos θ) to (sin 2θ, cos 2θ) to match a --yaw2 student."""
    t = t_reg.float().clone()
    sn, cs = t[:, 4:5], t[:, 5:6]
    n = (sn ** 2 + cs ** 2).sqrt().clamp(min=1e-6); sn, cs = sn / n, cs / n
    t[:, 4:5], t[:, 5:6] = 2 * sn * cs, cs ** 2 - sn ** 2
    return t


def kd_box_loss(hm, reg, t_hm, t_reg, rmin):
    """Far-box distillation. METEORite recalls far boxes (beyond ~30 m) much worse than the METEOR line.
    Uses the teacher's hm probability as a soft target with BCE in the band r >= rmin and |y| <= 15 m
    (teacher peaks weighted by 1+9p); where the teacher is confident (p > 0.3), reg (center offset /
    size / yaw) is also matched with L1. Zero inference cost."""
    key = (hm.shape[-2], hm.shape[-1], hm.device, rmin)
    if key not in _KD_MASK:
        rr = torch.arange(hm.shape[-2], device=hm.device).view(-1, 1).float()
        cc = torch.arange(hm.shape[-1], device=hm.device).view(1, -1).float()
        res = (M.C_XF + M.C_XR) / hm.shape[-2]          # box grid (0.4 m, also with BEV2); output is always the canonical grid (also with BEV_FRONT)
        xe = M.C_XF - (rr + 0.5) * res
        ye = M.C_YH - (cc + 0.5) * res
        _KD_MASK[key] = (((xe ** 2 + ye ** 2).sqrt() >= rmin) & (ye.abs() <= 15.0)).float()[None, None]
    m = _KD_MASK[key]
    tp = t_hm.float().sigmoid()
    w = m * (1.0 + 9.0 * tp)
    bce = F.binary_cross_entropy_with_logits(hm, tp, reduction="none")
    l_hm = (bce * w).sum() / w.sum().clamp(min=1)
    pk = m * (tp.amax(1, keepdim=True) > 0.3).float()
    l_reg = ((reg - t_reg.float()).abs() * pk).sum() / (pk.sum() * reg.shape[1]).clamp(min=1)
    return l_hm + 0.5 * l_reg


def losses(net, out, batch, dev, args):
    ls, log = {}, {}
    gt = batch[I_GT].to(dev)
    lane = out["lane"].float()
    if args.boundary_w > 0:          # boundary-weighted CE (implementation borrowed from train.py)
        ce = F.cross_entropy(lane, gt.clamp(max=8), reduction="none")
        bw = boundary_weight(gt, w=args.boundary_w)
        ok = (gt != 255).float()
        ce = (ce * bw * ok).sum() / (bw * ok).sum().clamp(min=1)
    elif args.lane_cw or args.seg_far_w > 0 or args.cls_w:   # thin-line class weights (CLASS_W), as in the METEOR line
        # --seg-far-w: weight pixels beyond the far threshold ahead by (1+w) to extend lane range; combinable with class weights
        from bevlane.train import CLASS_W
        Wc = CLASS_W.to(dev) if args.lane_cw else torch.ones(lane.shape[1], device=dev)
        if args.cls_w:                # explicit class weights (e.g. emphasize stop lines only: 0,1,1,1,1,6,1,1,1)
            Wc = torch.tensor([float(v) for v in args.cls_w.split(",")], device=dev)
        ok = (gt != 255)
        ce_px = F.cross_entropy(lane, gt, weight=Wc, ignore_index=255, reduction="none")
        wpx = Wc[gt.clamp(0, lane.shape[1] - 1)] * ok.float()
        if args.seg_far_w > 0:
            rr = torch.arange(gt.shape[-2], device=dev).view(1, -1, 1).float()
            xf = M.C_XF - (rr + 0.5) * ((M.C_XF + M.C_XR) / gt.shape[-2])
            fw = 1.0 + args.seg_far_w * ((xf > args.seg_far_x0) & (xf <= args.seg_far_x1)).float()   # band is configurable via --seg-far-x0/--seg-far-x1
            ce_px = ce_px * fw; wpx = wpx * fw
        ce = ce_px.sum() / wpx.sum().clamp(min=1.0)
    else:
        ce = F.cross_entropy(lane, gt, ignore_index=255)
    ls["lane"] = args.seg_w * (ce + 0.5 * dice_loss(lane, gt))
    if args.lovasz_w > 0:
        ls["lovasz"] = args.lovasz_w * lovasz_softmax(lane, gt)
    if args.tversky_w > 0:
        # The default (alpha 0.2 / beta 0.8) penalizes false positives strongly (thin, precise lanes) and misses
        #   lightly; alpha (misses) > beta (false positives), e.g. --tversky-ab 0.7,0.3, favours recall
        ta, tb = (float(v) for v in args.tversky_ab.split(","))
        ls["tversky"] = args.tversky_w * tversky_loss(lane, gt, classes=[3, 4, 5, 6], alpha=ta, beta=tb)
    if args.depth_hr and tuple(batch[I_DEP].shape[-2:]) != (M.FH, M.FW):
        # if the depth target (--depth-hw) is finer than the head grid, upsample logits bilinearly and compute CE at target resolution
        dt = depth_targets(batch[I_DEP].float(), dev, hw=tuple(batch[I_DEP].shape[-2:]))
        dl = out["dlog"].float()
        dl = F.interpolate(dl.flatten(0, 1), size=dt.shape[-2:], mode="bilinear", align_corners=False)
        if args.depth_edge_w > 0 or args.depth_far_w > 0:
            # edge-weighted depth loss: pixels on target depth edges (|∇log d| > 0.15, dilated by 1 px) are weighted (1 + w)
            #   so depth steps do not blur into slopes; other pixels unchanged
            dd = batch[I_DEP].float().to(dev).flatten(0, 1)                    # [BN, H, W] (m)
            lg = torch.log(dd.clamp(min=1.0))
            gx = torch.zeros_like(lg); gy = torch.zeros_like(lg)
            gx[..., 1:-1] = (lg[..., 2:] - lg[..., :-2]) * 0.5; gy[..., 1:-1, :] = (lg[..., 2:, :] - lg[..., :-2, :]) * 0.5
            vd = dd > 1.0
            ed = (((gx ** 2 + gy ** 2).sqrt() > 0.15) & vd).float()
            ed = F.max_pool2d(ed[:, None], 3, 1, 1)[:, 0]
            ce_px = F.cross_entropy(dl, dt.flatten(0, 1), ignore_index=255, reduction="none")
            okp = (dt.flatten(0, 1) != 255).float()
            wpx = (1.0 + args.depth_edge_w * ed) * okp
            if args.depth_far_w > 0:      # far road depth tends to collapse to the max range -> weight target pixels beyond 40 m by (1+w)
                wpx = wpx * (1.0 + args.depth_far_w * (dd > 40.0).float())
            ls["depth"] = args.depth_w * (ce_px * wpx).sum() / wpx.sum().clamp(min=1.0)
        else:
            ls["depth"] = args.depth_w * F.cross_entropy(dl, dt.flatten(0, 1), ignore_index=255)
    else:
        dt = depth_targets(batch[I_DEP].float(), dev)
        ls["depth"] = args.depth_w * F.cross_entropy(
            out["dlog"].float().flatten(0, 1), dt.flatten(0, 1), ignore_index=255)
    core = net.module if hasattr(net, "module") else net
    if args.seg2d_hr:                # 2D auxiliary supervision: upsample logits to the 1/4 GT; same class weights as the METEOR line
        g2 = batch[I_SEG2D].to(dev).long()
        sl = out["seg2d"].float()
        up = F.interpolate(sl.flatten(0, 1), size=g2.shape[-2:], mode="bilinear",
                           align_corners=False).view(*sl.shape[:3], *g2.shape[-2:])
        ls["seg2d"] = args.seg2d_w * DON.seg2d_loss(core, up, g2)
    if args.det2d_w > 0:             # 2D box auxiliary supervision (CenterNet style, same form as DepthSegIPMNetV52.bbox2d_loss)
        b2, c2 = batch[I_B2].float(), batch[I_C2]
        hm2, rg2 = out["hm2d"], out["reg2d"]
        B_, N_ = hm2.shape[:2]
        hw = hm2.shape[-2:]
        stride = batch[I_IMG].shape[-1] / hw[1]          # boxes are in data pixel coords (768 wide); r50 at 640 has upscaled input, so effective stride 5.33
        hm_t, reg_t, m_t = DON.build_det2d_targets(b2, c2, hw, "cpu", stride=stride)
        hm_t, reg_t, m_t = hm_t.to(dev), reg_t.to(dev), m_t.to(dev)
        pp = hm2.reshape(B_ * N_, -1, *hw).float().sigmoid().clamp(1e-4, 1 - 1e-4)
        pos = (hm_t > 0.99).float()
        fl = -(pos * (1 - pp) ** 2 * pp.log() + (1 - pos) * (1 - hm_t) ** 4 * pp ** 2 * (1 - pp).log()).sum() \
            / pos.sum().clamp(min=1)
        rl = (torch.abs(rg2.reshape(B_ * N_, 4, *hw).float() - reg_t) * m_t).sum() / m_t.sum().clamp(min=1) / 4
        ls["det2d"] = args.det2d_w * (fl + rl)
    s2 = F.interpolate(batch[I_SEG2D].to(dev).float().unsqueeze(2), (1, M.FH, M.FW),
                       mode="nearest").squeeze(2).long()
    if (s2 != 255).any() and not args.seg2d_hr:   # all-ignore batch (missing GT) -> CE would be NaN
        ls["seg2d"] = args.seg2d_w * F.cross_entropy(
            out["seg2d"].float().flatten(0, 1), s2.flatten(0, 1), ignore_index=255)
    ab, an = batch[I_AB].to(dev), batch[I_AN].to(dev)
    at, av = batch[I_AT].to(dev), batch[I_AV].to(dev)
    sel = an >= 0                                   # -1 = scene without box GT
    if sel.any():
        abx = ab
        if args.yaw2:                                   # axis only: wrap 2θ into [-π, π) and reuse the existing sin/cos target machinery
            abx = ab.clone(); abx[..., 5] = torch.remainder(2.0 * ab[..., 5] + math.pi, 2 * math.pi) - math.pi
        _pl = getattr(getattr(net, "module", net), "polar", None)
        if _pl is not None and "hm_polar" in out:       # polar BEV: boxes are trained on polar-grid targets (reg resampled to Cartesian loses meaning)
            _pg = (out["polar_kf"][sel], out["polar_tce"][sel]) if "polar_kf" in out else None   # per-sample rig geometry
            ls["box"] = args.box_w * _pl.det_loss(out["hm_polar"][sel], out["reg_polar"][sel], abx[sel], an[sel], geos=_pg)
        else:
            ls["box"] = args.box_w * DON.boxdet_loss(
                net, out["hm"].float()[sel], out["reg"].float()[sel], abx[sel], an[sel])
        ls["traj"] = args.traj_w * DON.traj_loss(
            net, out["traj"][sel], ab[sel], an[sel], at[sel], av[sel])
        ls["stat"] = args.stat_w * DON.stat_loss(
            out["stationary"].float()[sel], ab[sel], an[sel], at[sel], av[sel])
    if "kd_hm" in out:               # distill far hm/reg from the METEOR-line teacher
        ls["kd_box"] = args.kd_box_w * kd_box_loss(out["hm"].float(), out["reg"].float(),
                                                   out["kd_hm"], _teacher_reg_yaw2(out["kd_reg"]) if args.yaw2 else out["kd_reg"], args.kd_box_rmin)
    ego_gt = batch[I_EGO].to(dev)
    ls["ego"] = args.ego_w * DON.ego_loss(net, out["ego"].float(), ego_gt)
    if "pk_mask" in out and float(out["pk_mask"].sum()) > 0:      # privileged-information distillation (--priv-kd-*)
        w_ = out["pk_mask"]; nw = w_.sum()
        if args.priv_kd_feat_w > 0:
            def _fkd(a_, b_):
                d_ = ((a_ - b_) ** 2).flatten(1).mean(1) / (b_.flatten(1).var(1) + 1e-6)
                return (d_ * w_).sum() / nw
            ls["pkd_feat"] = args.priv_kd_feat_w * (_fkd(out["st_bev"], out["pk_bev"]) + _fkd(out["st_l1"], out["pk_l1"]))
        if args.priv_kd_ego_w > 0:
            Kq = EGO_K
            se, te = out["ego"].float(), out["pk_ego"]
            wp = F.smooth_l1_loss(se[:, :12 * Kq], te[:, :12 * Kq], reduction="none").mean(1)
            tau = args.priv_kd_tau
            kl = F.kl_div(F.log_softmax(se[:, 12 * Kq:13 * Kq] / tau, 1), F.softmax(te[:, 12 * Kq:13 * Kq] / tau, 1), reduction="none").sum(1) * tau * tau
            ls["pkd_ego"] = args.priv_kd_ego_w * ((wp + kl) * w_).sum() / nw
    elif "pk_mask" in out:                                                    # batch without a map: zero (no need to match graphs across ranks; grads are zero-filled in the reduce)
        ls["pkd_feat"] = out["ego"].sum() * 0.0
    if args.occ_w > 0 and "occ" in out:      # OCC head (same loss as the METEOR line)
        oc = out["occ"].float(); B_ = oc.shape[0]
        ls["occ"] = args.occ_w * DON.occ_loss(core, oc.view(B_, M.OCC_C, M.OCC_Z, *oc.shape[-2:]), batch[I_OCC].to(dev))
    tl_gt = batch[I_TL].to(dev)
    # a batch with only scenes lacking traffic-light GT (all 255) gives CE = nan (0/0) and a nan total loss
    #   (gradients stayed finite) -> use 0 when there is no valid target
    ls["tl"] = (args.tl_w * F.cross_entropy(out["tl"].float(), tl_gt, ignore_index=255)) if bool((tl_gt != 255).any()) \
        else out["tl"].float().sum() * 0.0
    r = batch[I_RISK].to(dev)
    rm = (r >= 0).float()
    if rm.sum() > 0:
        ls["risk"] = args.risk_w * ((out["risk"].float()[:, 0].sigmoid() - r).abs()
                                    * rm).sum() / rm.sum().clamp(min=1)
    if (args.lift_lr or args.rowband_lift or getattr(args, "polar", 0)) and args.lift_kd_w > 0 and "bev_geo" in out:   # distill the low-rank / row-band / polar lift from the geometric lift features
        a_, b_ = out["bev_lr"].float(), out["bev_geo"].float().detach()
        ls["liftkd"] = args.lift_kd_w * F.mse_loss(a_, b_) / (b_.var() + 1e-6)
        if "g_geo" in out:
            ga, gb = out["g_lr"].float(), out["g_geo"].float().detach()
            ls["gipmkd"] = args.lift_kd_w * F.mse_loss(ga, gb) / (gb.var() + 1e-6)
    for k, v in ls.items():
        log[k] = float(v.detach())
    if getattr(args, "_parts", None) is not None:
        args._parts.update(ls)
    return sum(ls.values()), log


def _set_route_map(net, batch, dev):
    """Route / map conditioning inputs: attach the last two dataset items (ego history [B,32|256], map/route raster [B,10,200,125]) to the net."""
    m_ = net.module if hasattr(net, "module") else net
    m_._rm_hist = batch[-2].to(dev, non_blocking=True).float(); m_._rm_ras = batch[-1].to(dev, non_blocking=True).float()


@torch.no_grad()
def validate(net, loader, dev, max_batches=200):
    net.eval()
    ade = fde = n = 0.0
    inter = torch.zeros(9, device=dev)
    union = torch.zeros(9, device=dev)
    for bi, batch in enumerate(loader):
        if bi >= max_batches:
            break
        e = batch[I_EGO].to(dev)
        if rmap_env("", "0") == "1":
            _set_route_map(net, batch, dev)
        with torch.autocast("cuda", torch.bfloat16):
            out = net.forward_geo(batch[I_IMG].to(dev), batch[I_K].to(dev),
                                  batch[I_T].to(dev), e[:, 12])   # val = camera only (gate condition)
        ego = out["ego"].float()
        wps = ego[:, :12 * EGO_K].view(-1, EGO_K, 6, 2)
        k = ego[:, 12 * EGO_K:12 * EGO_K + EGO_K].argmax(1)
        selw = wps[torch.arange(len(k), device=dev), k]
        gtw = e[:, :12].view(-1, 6, 2)
        v = e[:, 16] > 0
        if v.any():
            d = (selw[v] - gtw[v]).norm(dim=-1)
            ade += float(d.mean(1).sum()); fde += float(d[:, -1].sum()); n += int(v.sum())
        pred = out["lane"].float().argmax(1)
        gt = batch[I_GT].to(dev)
        ok = gt != 255
        for c in range(9):
            inter[c] += ((pred == c) & (gt == c) & ok).sum()
            union[c] += (((pred == c) | (gt == c)) & ok).sum()
    net.train()
    miou = float((inter / union.clamp(min=1)).mean())
    return ade / max(n, 1), fde / max(n, 1), miou, int(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--train-list", required=True)
    ap.add_argument("--mined", default="out/mined_scenes.txt")
    ap.add_argument("--mined-oversample", type=int, default=4)
    ap.add_argument("--eps-wta", type=float, default=0.1)
    ap.add_argument("--route-map-lr-mult", "--ae2e-lr-mult", type=float, default=20.0)
    # Privileged-information distillation ("HD map at training time only"): the teacher is a frozen --route-map checkpoint that
    #   sees the map / route raster and the ego history; the student sees images + v0 only (--route-map 0). The student's BEV
    #   features (fused BEV and the ego trunk l1) and its ego output (candidate waypoints and selection scores) are pulled
    #   towards the teacher, on samples that have a map only. The dataset appends the teacher inputs with METEOR_RMAP=1
    #   (they are not given to the student).
    ap.add_argument("--priv-kd-ckpt", default="", help="teacher checkpoint for privileged-information distillation (trained with --route-map 1)")
    ap.add_argument("--priv-kd-feat-w", type=float, default=0.0, help="weight of the BEV feature distillation (fused BEV + ego trunk)")
    ap.add_argument("--priv-kd-ego-w", type=float, default=0.0, help="weight of the ego output distillation (candidate waypoints smooth-L1 + selection-score KL)")
    ap.add_argument("--priv-kd-tau", type=float, default=1.0, help="temperature of the selection-score KL")
    ap.add_argument("--route-map", "--ae2e", type=int, default=0, help="route / map conditioned variant: add ego-motion history + HD-map / route raster inputs (dataset.rmap_inputs, METEOR_RMAP=1)")
    ap.add_argument("--ego-acc", type=int, default=0, help="add the current acceleration (GT acc, CAN-equivalent) as an ego_mlp input")
    ap.add_argument("--qat", type=int, default=0, help="fake-quant training: step at which to enable it (0 = off); observe only before that")
    ap.add_argument("--sparse-24", action="store_true", help="2:4 mask training (SP24_DENSE layers stay dense)")
    ap.add_argument("--sparse-nm", type=int, default=0, help="N:4 mask training, zeros per group of 4 (1 = 25%%, 2 = 50%% (= --sparse-24), 3 = 75%%); 0 = off")
    ap.add_argument("--sparse-ramp-steps", type=int, default=0, help="start with one fewer zero (e.g. 1:4) and switch to the final mask at this step")
    ap.add_argument("--ego-ce-mult", type=float, default=1.0)
    ap.add_argument("--ego-long-w", type=float, default=6.0, help="longitudinal weight of the ego loss (default 6.0 = previous behaviour)")
    ap.add_argument("--ego-gap", type=int, default=0, help="single-graph NPU export: feed ego/tl from GlobalAveragePool (BC1) instead of a fixed 5x4 pool + Flatten (MLP re-initialized)")
    ap.add_argument("--polar", type=int, default=0, help="polar BEV (bevlane/polar_lift): no view transform, Conv-only vertical collapse -> BEV net/heads on a [C,D,U] grid (single-graph NPU export); distilled from the geometric lift via liftkd")
    ap.add_argument("--rowband-lift", type=int, default=0, help="number of depth bins K of the row-band lift for edge accelerators (no Gather/MatMul); 0 = off")
    ap.add_argument("--rowband-pool", type=int, default=2, help="2D average-pool factor before prime")
    ap.add_argument("--rowband-zrows", type=int, default=100, help="number of BEV rows Z of the row-band lift (nearest upsample by LH/Z)")
    ap.add_argument("--rowband-grows", type=int, default=3, help="image-row window R of the ground lift")
    ap.add_argument("--rowband-static", type=int, default=0, help="1: static variant without depth (learned vertical Conv, no Mul)")
    ap.add_argument("--rowband-ground", type=int, default=1, help="0: keep the ground lift (gvt) as the original Gather and row-band only vt (split graph + host implementation; keeps lateral lane resolution)")
    ap.add_argument("--rowband-scale", type=float, default=1.0, help="magnitude correction of the initial Wcol (set by the calibration script)")
    ap.add_argument("--rowband-scale-file", default="", help="per-row scale correction [Z] as .npy (made by parity_rowband_lift.py --save-scale; overrides --rowband-scale)")
    ap.add_argument("--rowband-ref-scene", default=os.environ.get("METEOR_REF_SCENE", ""),
                    help="scene of the reference rig used for geometric initialization (default: $METEOR_REF_SCENE)")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit-train", type=int, default=60000)
    ap.add_argument("--val-batches", type=int, default=200)
    ap.add_argument("--lidar-drop", type=float, default=0.3)
    ap.add_argument("--cam-drop", type=float, default=0.25)
    ap.add_argument("--yaw-fix-deg", type=float, default=0.7)
    ap.add_argument("--grad-clip", type=float, default=100.0)
    ap.add_argument("--grad-probe", type=int, default=0, help="measure per-loss gradient norms (backbone/lift/BEV) over N batches, then exit")
    ap.add_argument("--seg-w", type=float, default=1.0)
    ap.add_argument("--depth-w", type=float, default=0.6)
    ap.add_argument("--seg2d-w", type=float, default=0.35)
    ap.add_argument("--box-w", type=float, default=1.3)
    ap.add_argument("--traj-w", type=float, default=0.5)
    ap.add_argument("--stat-w", type=float, default=1.0)
    ap.add_argument("--ego-w", type=float, default=2.0)
    ap.add_argument("--tl-w", type=float, default=0.6)
    ap.add_argument("--risk-w", type=float, default=0.3)
    ap.add_argument("--ego-fde-w", type=float, default=0.5)
    ap.add_argument("--ego-ce-tau", type=float, default=0.0)
    ap.add_argument("--fp-init", type=int, default=1, help="function-preserving warm start on capacity change (1 = on)")
    ap.add_argument("--kd-box-ckpt", default="", help="far-box distillation: teacher checkpoint (METEOR-line v52 family)")
    ap.add_argument("--kd-box-w", type=float, default=0.0, help="far-box distillation weight (0 = off)")
    ap.add_argument("--kd-box-rmin", type=float, default=30.0, help="minimum range for distillation [m] (only within |y| <= 15 m)")
    ap.add_argument("--det-head", type=int, default=0, help="width of a dedicated box head (0 = plain 1x1 only; can be grafted function-preservingly)")
    ap.add_argument("--paint-seg", default="", help="PointPainting (as in the METEOR line): append seg2d class probabilities to ctx before the lift (e.g. 2,3,4,5,6,7,8,13)")
    ap.add_argument("--det2d-w", type=float, default=0.0, help="2D box (hm2d/reg2d, 1/8) auxiliary loss weight (like bbox2d_w of the METEOR line); 0 = untrained")
    ap.add_argument("--head-v2", type=int, default=0, help="stronger heads (function-preserving graft): progressive BEV seg upsampling / post-painting mixing / depth & 2D residual blocks")
    ap.add_argument("--feat-stride", type=int, default=0, help="image feature stride (0 = default 8; 4 = depth/ctx/2D heads and lift columns at 1/4)")
    ap.add_argument("--seg2d-hr", type=int, default=0, help="train seg2d on the 1/4 GT (bilinearly upsampled logits) with METEOR-line class weights (thin lines x4)")
    ap.add_argument("--box-gscale", type=float, default=1.0, help="gradient scale from the 3D box head into shared layers (box-w 2.6 + 0.5 = shared layers see box-w 1.3)")
    ap.add_argument("--ego-gscale", type=float, default=1.0, help="gradient scale from the ego path into shared layers (0.1 makes ego comparable to box)")
    ap.add_argument("--cam-w", type=int, default=0, help="confidence-weighted camera fusion (zero init = plain average)")
    ap.add_argument("--bev2", type=int, default=0, help="BEV 0.2 m + 3-level FPN + task-specific heads")
    ap.add_argument("--bev2-c02", type=int, default=48, help="BEV2 width of the 0.2 m level")
    ap.add_argument("--bev2-dw", type=int, default=96, help="BEV2 width of the box tower")
    ap.add_argument("--box-far", type=int, default=0, help="1: METEOR-line far box loss weighting (veh >40 m x4 / VRU 25-45 m x2)")
    ap.add_argument("--gipm", action="store_true", help="ground IPM road branch (1/4 features -> 0.2 m grid -> lane head)")
    ap.add_argument("--seg-far-x0", type=float, default=40.0, help="near edge of the forward band weighted by --seg-far-w [m]")
    ap.add_argument("--seg-far-x1", type=float, default=999.0, help="far edge of that band [m]")
    ap.add_argument("--cls-w", default="", help="9 comma-separated class weights for the BEV seg CE (overrides --lane-cw)")
    ap.add_argument("--seg-far-w", type=float, default=0.0, help="CE weight for far BEV seg pixels in the --seg-far-x0/x1 band (x(1+w))")
    ap.add_argument("--lane-cw", action="store_true", help="use CLASS_W in the lane CE (thin lines x3-6, as in the METEOR line)")
    ap.add_argument("--lovasz-w", type=float, default=0.0)
    ap.add_argument("--tversky-w", type=float, default=0.0)
    ap.add_argument("--tversky-ab", default="0.2,0.8", help="Tversky alpha (misses), beta (false positives); 0.7,0.3 favours recall")
    ap.add_argument("--lift-lr", type=int, default=0, help="learned low-rank block mapping without geometric table lookup (view transform + ground projection); NPU-friendly: MatMul/Add/Concat only")
    ap.add_argument("--lr-blk", type=int, default=50); ap.add_argument("--lr-rank", type=int, default=32)
    ap.add_argument("--lr-blk-m", type=float, default=0.0, help="view-transform block side in metres (50 cells on the 0.4 m grid = 20 m); keeps the block count with BEV2 (0.2 m) so the node count does not grow; >0 overrides --lr-blk")
    ap.add_argument("--glr-blk", type=int, default=100); ap.add_argument("--glr-rank", type=int, default=16)
    ap.add_argument("--lift-kd-w", type=float, default=1.0, help="weight of distillation (MSE / variance) towards the geometric lift BEV features")
    ap.add_argument("--lr-rect", type=int, default=0, help="restrict low-rank layer inputs to the geometric rectangle of the reference rig (1/10 of the weights)")
    ap.add_argument("--lr-margin", type=float, default=0.15)
    ap.add_argument("--glr-half", type=int, default=0, help="build the ground projection on the 0.4 m grid and nearest-upsample x2 (1/4 weights/nodes)")
    ap.add_argument("--lr-geo-init", type=int, default=0, help="initialize rectangle-restricted low-rank U/V by truncated SVD of the geometric selection matrix (requires --lr-rect 1)")
    ap.add_argument("--lr-rows-only", type=int, default=0, help="use the full width for the rectangle columns (export is Slice + Conv1x1 only, accepted by strict NPU compilers)")
    ap.add_argument("--boundary-w", type=float, default=0.0)
    ap.add_argument("--ema", type=float, default=0.0, help="EMA decay (0 = off); val/best use EMA weights")
    ap.add_argument("--init-ckpt", default="", help="warm start (strict load of a mite ckpt)")
    ap.add_argument("--val-list", default="", help="list of validation scenes (e.g. for fine-tuning on another dataset)")
    ap.add_argument("--cp", type=int, default=0, help="prime/BEV channels (default 64)")
    ap.add_argument("--depth-bins", type=int, default=0, help="depth bins (default 48)")
    ap.add_argument("--bev-c0", type=int, default=0)
    ap.add_argument("--bev-c1", type=int, default=0)
    ap.add_argument("--fuse-c", type=int, default=0)
    ap.add_argument("--neck-c", type=int, default=0, help="image-side neck width (default 128; widened function-preservingly)")
    ap.add_argument("--head-c", type=int, default=0, help="hidden width of the depth/ctx heads (default 96)")
    ap.add_argument("--backbone", default="resnet18", choices=["resnet18", "resnet34", "resnet50"])
    ap.add_argument("--img-h", type=int, default=0)
    ap.add_argument("--seg2d-key", default="seg2d21", help="manifest key of the 2D seg target (seg2d21 = 108x192 / seg2d21_hr = finer grid)")
    ap.add_argument("--depth-hw", default="", help="depth target resolution H,W (default = feature grid FHxFW; e.g. 360,640 for targets finer than the head)")
    ap.add_argument("--depth-hr", type=int, default=0, help="when the target is finer than the head grid, upsample logits and compute depth CE at target resolution")
    ap.add_argument("--occ-gscale", type=float, default=1.0, help="gradient scale from the OCC head into shared BEV features (0 = train the head only, other tasks unchanged)")
    ap.add_argument("--occ-w", type=float, default=0.0, help="3D occupancy (OCC) head loss weight (0 = no head); METEOR-line occ_loss (class weights + near dynamic false-positive penalty), targets occ/ (16x200x200, 255 = unknown)")
    ap.add_argument("--yaw2", type=int, default=0, help="regress box yaw with period π (sin 2θ, cos 2θ). Auto-labels flip front/back of stationary vehicles between frames (axis is stable), so full-heading sin/cos collapses to zero and atan2 spins on noise. Inference halves the angle in MiteEval/demo (graph unchanged)")
    ap.add_argument("--depth-far-w", type=float, default=0.0, help="CE weight for depth target pixels beyond 40 m (x(1+w)); use with --depth-hr")
    ap.add_argument("--depth-edge-w", type=float, default=0.0, help="CE weight for depth edge pixels (x(1+w)); use with --depth-hr")
    ap.add_argument("--depth-key", default="depth4", help="manifest key of the depth target (depth4 = 108x192 / depth_hr = finer grid)")
    ap.add_argument("--bev-front", type=int, default=0, help="restrict the internal BEV grid to 0-80 m ahead x ±25 m (1/4 lift / BEV net); outputs are padded back to the canonical grid")
    ap.add_argument("--n-cams", type=int, default=8, help="use only the first n cameras (1 = CAM_FRONT_WIDE, monocular); sets M.N_CAM, restored from ckpt args")
    ap.add_argument("--front-mask", default="", help="restrict supervision to the front sector x0,x1,y0,y1 [m] (e.g. 0,80,-25,25); lane raster outside is ignored, box GT dropped (dataset.front_mask)")
    ap.add_argument("--hr-root", default="", help="image root of native-resolution (1280x720) frames; used where available, otherwise the 768-wide images under --root")
    ap.add_argument("--img-scale", type=float, default=1.0, help="image rescale in the dataset (hw x scale); e.g. 1.5 on 720p gives 648x1152 (downscale), 768-wide images are upscaled; K and 2D boxes follow")
    ap.add_argument("--resume", type=int, default=0, help="1: resume from the next epoch if --out/last.pt exists")
    ap.add_argument("--img-w", type=int, default=0, help="override the input resolution (default 352x640)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", type=int, default=0, help="steps per epoch for a smoke run")
    args = ap.parse_args()
    if args.route_map:
        os.environ["METEOR_RMAP"] = "1"     # dataset appends ego history and map/route raster (inherited by DataLoader workers)
    if args.det2d_w > 0:
        global I_B2, I_C2, I_EGO, I_TL, I_RISK, I_LB
        I_B2, I_C2, I_EGO, I_TL, I_RISK, I_LB = 10, 11, 12, 13, 14, 15
    if args.occ_w > 0:                       # dataset inserts occ right after ego -> shift later indices by 1
        global I_OCC
        I_OCC = I_EGO + 1; I_TL, I_RISK, I_LB = I_TL + 1, I_RISK + 1, I_LB + 1

    if args.img_h and args.img_w:            # resolution lever: module constants drive forward_geo and the targets
        M.IMG_H, M.IMG_W = args.img_h, args.img_w
    M.set_bev_front(int(args.bev_front))   # front-only internal BEV grid (before building the net)
    M.N_CAM = int(args.n_cams)              # monocular (1) / 8 cameras
    M.YAW2 = int(args.yaw2)
    M.OCC_HEAD = 1 if args.occ_w > 0 else 0
    if args.feat_stride:                     # image-side heads and lift columns at 1/4 (FS=4)
        M.FS = args.feat_stride
    M.HEAD_V2 = int(args.head_v2)
    M.EGO_GAP = int(getattr(args, "ego_gap", 0) or 0)   # ego/tl input via GAP (single-graph NPU export)
    if args.lr_blk_m > 0:
        args.lr_blk = int(round(args.lr_blk_m / (0.2 if args.bev2 else 0.4)))   # fixed physical size (ckpt args keep the resolved lr_blk)
    M.LIFT_LR = int(args.lift_lr); M.LR_BLK, M.LR_RANK, M.GLR_BLK, M.GLR_RANK = args.lr_blk, args.lr_rank, args.glr_blk, args.glr_rank
    M.LR_GEO_INIT = int(args.lr_geo_init); M.GLR_HALF = int(args.glr_half); M.LR_ROWS_ONLY = int(args.lr_rows_only)
    M.LR_RECT = int(args.lr_rect); M.LR_MARGIN = args.lr_margin
    M.FH, M.FW = M.IMG_H // M.FS, M.IMG_W // M.FS
    if args.cp: M.C_P = args.cp
    if args.depth_bins: M.D = args.depth_bins
    if args.bev_c0: M.BC0 = args.bev_c0
    if args.bev_c1: M.BC1 = args.bev_c1
    if args.fuse_c: M.BCF = args.fuse_c
    if args.neck_c: M.NECK_C = args.neck_c
    if args.head_c: M.HEAD_C = args.head_c
    M.GIPM = args.gipm
    M.DET_HEAD = args.det_head
    M.CAMW = args.cam_w
    M.EGO_GSCALE = args.ego_gscale
    M.OCC_GSCALE = args.occ_gscale
    M.BOX_GSCALE = args.box_gscale
    M.PAINT_CLS = tuple(int(c) for c in args.paint_seg.split(",") if c.strip())
    M.BEV2 = bool(args.bev2)
    M.BEV2_C02, M.BEV2_DW = args.bev2_c02, args.bev2_dw
    M.set_bev_res(0.2 if args.bev2 else 0.4)
    ddp = "RANK" in os.environ
    rank = int(os.environ.get("RANK", 0)); world = int(os.environ.get("WORLD_SIZE", 1))
    if ddp:
        dist.init_process_group("nccl", timeout=__import__("datetime").timedelta(
            seconds=int(os.environ.get("METEOR_NCCL_TIMEOUT_S", "3600"))))
        torch.cuda.set_device(rank % torch.cuda.device_count())
    dev = torch.device("cuda")
    os.makedirs(args.out, exist_ok=True)

    tr_scenes = [l.strip() for l in open(args.train_list) if l.strip()]
    mined = set(l.strip() for l in open(args.mined)) if os.path.exists(args.mined) else set()
    extra = [s for s in tr_scenes if s in mined] * max(args.mined_oversample - 1, 0)
    _, va_scenes = split_scenes(args.root)
    if args.val_list:          # explicit validation scenes (default: split_scenes)
        va_scenes = [l.strip() for l in open(args.val_list) if l.strip()]
        if rank == 0:
            print(f"[data] val list {args.val_list}: {len(va_scenes)} scenes", flush=True)
    # depth targets are read at the feature-grid resolution (FH x FW): 432p s4 = 108x192, 720p s4 = 180x320.
    #   --depth-key depth_hr prefers finer targets (extract_depth_dense.py --grid 180,320 --out-dir depth_gt_hr --key depth_hr); scenes without them upsample depth4
    dhw = tuple(int(v) for v in args.depth_hw.split(",")) if args.depth_hw else (M.FH, M.FW)
    kw = dict(gt_key="gt_cons", use_gt_valid=True, with_depth=True, depth_hw=dhw, depth_key=args.depth_key,
              with_seg2d=True, seg2d_key=args.seg2d_key, with_agenttraj=True, with_ego=True, with_tl=True,
              with_risk=True, with_lidarbev=True, n_cams=args.n_cams, front_mask=args.front_mask, with_bbox2d=args.det2d_w > 0, with_occ=args.occ_w > 0,
              yaw_fix_deg=args.yaw_fix_deg, hr_root=args.hr_root or None, img_scale=args.img_scale)
    tr = BevLaneDataset(args.root, tr_scenes + extra, augment=True,
                        cam_drop=args.cam_drop, **kw)
    va = BevLaneDataset(args.root, va_scenes, augment=False, **kw)
    if rank == 0:
        print(f"[data] train {len(tr)} samples ({len(tr_scenes)} scenes + mined x{args.mined_oversample}), "
              f"val {len(va)} ({len(va_scenes)} scenes)", flush=True)
    samp = EpochSubsetSampler(len(tr), min(args.limit_train, len(tr)), rank=rank, world=world)
    dl = torch.utils.data.DataLoader(tr, batch_size=args.batch, sampler=samp,
                                     num_workers=args.workers, pin_memory=True,
                                     drop_last=True, persistent_workers=args.workers > 0)
    dv = torch.utils.data.DataLoader(va, batch_size=4, shuffle=False, num_workers=2)

    torch.manual_seed(0)                 # identical fresh init on all ranks (redundant with the broadcast below)
    net = make_net(args, rank).to(dev)
    if args.route_map:
        net.enable_route_map()               # create before loading the init ckpt (older ckpts keep the zero init = function-preserving)
    if args.rowband_lift:                  # row-band lift: geometric init from the reference rig (overwritten later if init-ckpt has rowband.*)
        from bevlane.rowband_lift import attach as _hl_attach
        _ref = BevLaneDataset(args.root, [args.rowband_ref_scene], augment=False, max_per_scene=1, trim_start=3, trim_end=10, **kw)
        _b = _ref[0]
        _, _info = _hl_attach(net, vars(args), geometry=(_b[1][0].numpy(), _b[2][0].numpy(), tuple(_b[0].shape[-2:])))
        if rank == 0:
            print(f"[rowband] lift attached: {net.rowband.extra_repr()} ref={args.rowband_ref_scene} {_info}", flush=True)
    if getattr(args, "polar", 0):        # polar BEV: build the resampling back to the Cartesian grid (grid_sample) from the reference rig (polar.* in init-ckpt overwrites later)
        from bevlane.polar_lift import attach as _pl_attach
        _ref = BevLaneDataset(args.root, [args.rowband_ref_scene], augment=False, max_per_scene=1, trim_start=3, trim_end=10, **kw)
        _b = _ref[0]
        _, _info = _pl_attach(net, vars(args), geometry=(_b[1][0].numpy(), _b[2][0].numpy(), tuple(_b[0].shape[-2:])))
        if rank == 0:
            print(f"[polar] attached: {net.polar.extra_repr()} ref={args.rowband_ref_scene}", flush=True)
    if args.init_ckpt:
        ck = torch.load(args.init_ckpt, map_location="cpu", weights_only=False)
        sd = M.remap_legacy_keys({k.replace("module.", ""): v for k, v in ck["sd"].items()})
        if args.ego_acc and "ego_mlp.0.weight" in sd and sd["ego_mlp.0.weight"].shape[1] == net.ego_mlp[0].in_features + 1:
            net.enable_ego_acc()                 # ckpt already has the acc input -> widen first so shapes match
        cur = net.state_dict()
        ok = {k: v for k, v in sd.items() if k in cur and tuple(cur[k].shape) == tuple(v.shape)}
        bb_old = ck.get("args", {}).get("backbone", "resnet18")
        if (bb_old == "resnet50") != (args.backbone == "resnet50"):
            # basic block <-> bottleneck: the trained backbone/FPN laterals do not transfer -> keep ImageNet weights
            # for the new backbone and fresh laterals; only the BEV side / heads are warm-started
            BBK = ("stem.", "l2.", "l3.", "l4.", "lat2.", "lat3.", "lat4.", "paint.")
            ok = {k: v for k, v in ok.items() if not k.startswith(BBK)}
            sd = {k: v for k, v in sd.items() if not k.startswith(BBK)}
            if rank == 0:
                print(f"[init] backbone family change {bb_old} -> {args.backbone}: backbone = ImageNet, BEV side warm", flush=True)
        n_fp = 0
        if args.fp_init:
            ok, n_fp = function_preserving(cur, sd, ok)
        net.load_state_dict(ok, strict=False)
        if rank == 0:
            print(f"[init] warm start from {args.init_ckpt} (ep {ck.get('ep')}, ema={ck.get('ema', False)}; "
                  f"matched {len(ok) - n_fp}/{len(cur)} tensors, function-preserving {n_fp}"
                  f"{'' if len(ok) == len(cur) else ' - the rest freshly initialized'})", flush=True)
        if args.det2d_w > 0 and not float(ck.get("args", {}).get("det2d_w", 0) or 0):
            # hm2d/reg2d were never trained (random), so p=0.5 everywhere makes focal negatives explode
            # (huge backbone gradients) -> restart from the CenterNet init (bias -2.19 = p 0.1, small weights)
            with torch.no_grad():
                for hd in (net.hm2d, net.reg2d):
                    nn.init.normal_(hd.weight, std=1e-3); nn.init.zeros_(hd.bias)
                net.hm2d.bias.fill_(-2.19)
            if rank == 0:
                print("[init] det2d: hm2d/reg2d re-initialized (bias -2.19)", flush=True)
    if args.ego_acc:
        net.enable_ego_acc()                     # widen after loading the init ckpt (function-preserving; already matched above if the ckpt was wide)
        if rank == 0:
            print("[init] ego_acc: added a current-acceleration input column to ego_mlp", flush=True)
    start_ep = 0
    rp = os.path.join(args.out, "last.pt")
    if args.resume and os.path.exists(rp):
        # resume from the epoch after our own last.pt (EMA weights) instead of restarting from init.
        # optimizer state is not saved, so AdamW is re-initialized (lr continues from its schedule position)
        ck = torch.load(rp, map_location="cpu", weights_only=False)
        net.load_state_dict(M.remap_legacy_keys({k.replace("module.", ""): v for k, v in ck["sd"].items()}))
        start_ep = int(ck["ep"]) + 1
        if rank == 0:
            print(f"[resume] {rp} (ep {ck['ep']}) -> start ep {start_ep}", flush=True)
    sp_pairs, sp_final = [], []
    sp_zeros = int(getattr(args, "sparse_nm", 0) or 0) or (2 if args.sparse_24 else 0)   # 25/50/75% = 1/2/3 zeros per 4
    if sp_zeros or args.qat:
        from bevlane import robust_train as RT
    if sp_zeros:
        sp_pairs, sp_final, n_dn = RT.build_24_masks(net, ramp=args.sparse_ramp_steps > 0, zeros=sp_zeros)
        if rank == 0:
            print(f"[sparse-{sp_zeros}:4 = {25 * sp_zeros}%] {len(sp_pairs)} tensors masked, {n_dn} dense" + (f"; {max(1, sp_zeros - 1)}:4 for the first {args.sparse_ramp_steps} steps" if sp_final else ""), flush=True)
    if args.qat:
        n_fq = RT.attach_qat(net)
        if rank == 0:
            print(f"[qat] {n_fq} fake-quant points (conv/linear inputs + gather lift in/out); enable at step {args.qat}", flush=True)
    if ddp:
        # without a seed or broadcast, freshly initialized weights differ per rank (gradients are all-reduced,
        # weights are not). Sync everything to rank 0.
        with torch.no_grad():
            for t in list(net.parameters()) + list(net.buffers()):
                dist.broadcast(t.data, 0)
    ema = EMA(net, decay=args.ema) if args.ema > 0 else None
    teacher = None
    if args.kd_box_w > 0 and args.kd_box_ckpt:
        os.environ.setdefault("METEOR_ZERO_HIST", "1")       # same as the METEOR-line evaluation (zero history)
        from bevlane.probe_net import load_full
        teacher = load_full(args.kd_box_ckpt, mv="v52", verbose=rank == 0)
        for p_ in teacher.parameters():
            p_.requires_grad_(False)
        if rank == 0:
            print(f"[kd-box] teacher {args.kd_box_ckpt} (w {args.kd_box_w}, r >= {args.kd_box_rmin} m)", flush=True)
    priv = None
    if args.priv_kd_ckpt and (args.priv_kd_feat_w > 0 or args.priv_kd_ego_w > 0):
        assert rmap_env() == "1" and not args.route_map, "the dataset appends the teacher inputs with METEOR_RMAP=1; the student uses --route-map 0"
        import bevlane.model as _MD
        _y2 = getattr(_MD, "_YAW2", 0)                                    # load_mite_eval resets _YAW2 for evaluation -> restore it so the student's box encoding is unchanged
        priv = M.load_mite_eval(args.priv_kd_ckpt, device=str(dev), verbose=rank == 0).net.eval()
        _MD._YAW2 = _y2
        assert getattr(priv, "route_map", False), "the teacher must be a --route-map checkpoint (with map / route inputs)"
        for p_ in priv.parameters():
            p_.requires_grad_(False)
        if rank == 0:
            print(f"[priv-kd] teacher {args.priv_kd_ckpt} feat_w {args.priv_kd_feat_w} ego_w {args.priv_kd_ego_w} tau {args.priv_kd_tau}", flush=True)
    # no DDP wrapper: forward_geo is our own entry point, so gradients are synced
    # manually after backward (all ranks build the same graph; None grads skip equally)
    if args.route_map and args.route_map_lr_mult != 1.0:
        # the new zero-initialized route / map input layers barely move at the base lr -> separate group with lr x route_map_lr_mult
        _rm = [p_ for n_, p_ in net.named_parameters() if ".rm_" in n_ or n_.startswith("rm_")]
        _ids = {id(p_) for p_ in _rm}
        opt = torch.optim.AdamW([{"params": [p_ for p_ in net.parameters() if id(p_) not in _ids], "mult": 1.0},
                                 {"params": _rm, "mult": args.route_map_lr_mult}], lr=args.lr, weight_decay=1e-4)
        if rank == 0:
            print(f"[route-map] {len(_rm)} tensors at lr x{args.route_map_lr_mult}", flush=True)
    else:
        opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    steps_ep = len(dl)
    total = steps_ep * args.epochs
    warm = min(500, total // 20)

    def lr_at(step):
        if step < warm:
            return args.lr * step / max(warm, 1)
        p = (step - warm) / max(total - warm, 1)
        return 1e-6 + (args.lr - 1e-6) * 0.5 * (1 + math.cos(math.pi * p))

    best = 9e9
    step = start_ep * steps_ep
    for ep in range(start_ep, args.epochs):
        samp.set_epoch(ep)
        t0 = time.time()
        for bi, batch in enumerate(dl):
            if args.smoke and bi >= args.smoke:
                break
            for g in opt.param_groups:
                g["lr"] = lr_at(step) * g.get("mult", 1.0)
            e = batch[I_EGO].to(dev)
            lidar = batch[I_LB].to(dev)
            if args.lidar_drop > 0:
                keep = (torch.rand(lidar.shape[0], 1, 1, 1, device=dev)
                        >= args.lidar_drop).float()
                lidar = lidar * keep
            if args.route_map:
                _set_route_map(net, batch, dev)
            with torch.autocast("cuda", torch.bfloat16):
                out = net.forward_geo(batch[I_IMG].to(dev), batch[I_K].to(dev),
                                      batch[I_T].to(dev), e[:, 12], lidar=lidar, acc=e[:, 17] if args.ego_acc else None)   # input acceleration is the causal e[17] (e[13] looks ahead and is for targets only)
            if priv is not None:
                with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                    _set_route_map(priv, batch, dev)
                    pt = priv.forward_geo(batch[I_IMG].to(dev), batch[I_K].to(dev), batch[I_T].to(dev), e[:, 12])
                has_map = (batch[-1][:, :8].flatten(1).amax(1) > 0).to(dev)          # distil only samples that have an HD map
                out["pk_mask"] = has_map.float()
                out["pk_ego"], out["pk_bev"], out["pk_l1"] = pt["ego"].float(), priv._fused_bev.float(), priv._l1.float()
                out["st_bev"], out["st_l1"] = net._fused_bev.float(), net._l1.float()
            if teacher is not None:
                with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                    to = teacher(batch[I_IMG].to(dev), batch[I_K].to(dev), batch[I_T].to(dev), e[:, 12])
                out["kd_hm"], out["kd_reg"] = to[3].float(), to[4].float()
            if args.grad_probe:
                args._parts = {}
            loss, log = losses(net, out, batch, dev, args)
            if args.grad_probe:      # per-loss gradient norms per group (backbone / lift = ctx, depth / BEV and later)
                core = net.module if hasattr(net, "module") else net
                grp = {"bb": [], "lift": [], "bev": []}
                for n, p_ in core.named_parameters():
                    if not p_.requires_grad:
                        continue
                    k = "bb" if n.split(".")[0] in ("stem", "l2", "l3", "l4", "lat2", "lat3", "lat4", "smooth") else \
                        "lift" if n.split(".")[0] in ("ctx", "depth", "camw") else "bev"
                    grp[k].append(p_)
                row = []
                for k, v in list(args._parts.items()) + [("TOTAL", loss)]:
                    gs = {g: torch.autograd.grad(v, ps, retain_graph=True, allow_unused=True) for g, ps in grp.items()}
                    nr = {g: sum(float((x.float() ** 2).sum()) for x in gg if x is not None) ** 0.5 for g, gg in gs.items()}
                    row.append(f"{k}:bb={nr['bb']:.1f}/lift={nr['lift']:.1f}/bev={nr['bev']:.1f}")
                print(f"GRADPROBE b{bi} " + " ".join(row), flush=True)
                if bi + 1 >= args.grad_probe:
                    return
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if ddp:
                # a rank whose batch has only scenes without box GT gets None grads on the box head, so the number
                #   of all_reduce calls diverges across ranks and deadlocks. Reduce every requires_grad parameter in
                #   the same order, filling missing grads with zeros (same semantics as DDP).
                for p in net.parameters():
                    if not p.requires_grad:
                        continue
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
            gn = torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
            opt.step()
            if sp_pairs:
                if sp_final and step >= args.sparse_ramp_steps:
                    sp_pairs = [(p_, mf) for (p_, _), mf in zip(sp_pairs, sp_final)]; sp_final = []
                    if rank == 0:
                        print(f"[sparse-24] step {step}: switched 1:4 -> 2:4", flush=True)
                RT.apply_24(sp_pairs)
            if args.qat and step == args.qat:
                RT.set_fake_quant(net, True)
                if rank == 0:
                    print(f"[qat] step {step}: fake-quant enabled", flush=True)
            if ema is not None:
                ema.update(net)
            step += 1
            if rank == 0 and (step % 50 == 0 or step <= 5):
                parts = " ".join(f"{k}={v:.3f}" for k, v in sorted(log.items()))
                print(f"ep{ep} step{step}/{total} loss={float(loss):.3f} gn={float(gn):.0f} "
                      f"lr={lr_at(step):.2e} ({(time.time()-t0)/max(bi+1,1):.2f}s/it) {parts}",
                      flush=True)
        if rank == 0:
            bk = ema.swap_in(net) if ema is not None else None
            if sp_pairs:
                RT.apply_24(sp_pairs)                  # keep EMA weights strictly N:4 too (usable directly for eval/export)
            ade, fde, miou, nv = validate(net, dv, dev, args.val_batches)
            tag = "-ema" if ema is not None else ""
            print(f"[valE2E{tag} ep{ep}] ADE={ade:.2f}m FDE={fde:.2f}m (n={nv}) | "
                  f"[val{tag} ep{ep}] mIoU={miou:.3f}", flush=True)
            sd_save = RT.strip_qat(net.state_dict()) if args.qat else net.state_dict()
            torch.save({"sd": sd_save, "ep": ep, "mite": 1, "ema": ema is not None,
                        "args": vars(args)}, os.path.join(args.out, "last.pt"))
            if ade < best:
                best = ade
                torch.save({"sd": sd_save, "ep": ep, "mite": 1, "ema": ema is not None,
                            "args": vars(args)}, os.path.join(args.out, "best_e2e.pt"))
            if ema is not None:
                ema.swap_out(net, bk)
        if ddp:
            dist.barrier()
    if rank == 0:
        print(f"[done] best valE2E ADE={best:.3f}m", flush=True)


if __name__ == "__main__":
    main()

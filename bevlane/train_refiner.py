#!/usr/bin/env python3
"""Train post-hoc residual refiners for the three priority heads (roadmap 3f).

A MultiTaskRefiner (bevlane.model) refines the FROZEN main model's outputs:
  * BEV seg    -- U-Net residual on the 9-class logits (far-range completion,
                  black trained as a real class so road does not bleed)
  * BEV 3D box -- U-Net residual on the [hm+reg] det grid (peak sharpening,
                  box size/heading correction)
  * E2E plan   -- MLP residual on the waypoint vector, conditioned on v0 and a
                  pooled BEV summary (second-stage planner)

The base model is loaded from a round checkpoint and never updated (eval,
requires_grad=False, forward under no_grad), and every head is zero-init
residual, so all three tasks are preserved by construction and only added to.
Heads can be enabled independently (--do-seg / --do-box / --do-e2e).

8-GPU DDP:
    torchrun --nproc_per_node=8 bevlane/train_refiner.py \
        --ckpt out/bevlane_ckpt_r34/last.pt --model v39 --batch 4 \
        --do-box --do-e2e --out out/refiner_r34
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
from bevlane.dataset import BevLaneDataset                    # noqa: E402
from bevlane.model import (MODELS, MultiTaskRefiner,          # noqa: E402
                           N_CLASSES, EGO_K)
from bevlane.train import (CLASS_NAMES, CLASS_W, lovasz_softmax,  # noqa: E402
                           split_scenes)

EGO_DIM = 12 * EGO_K + EGO_K + 3
BANDS = [("40-80m", 0, 200), ("20-40m", 200, 300), ("0-20m", 300, 400)]


def _unpack(batch, device, a):
    """Mirror of dataset.py's append order. Every optional tensor is pulled
    in the SAME sequence the dataset writes it, and the running index is
    asserted against len(batch) at the end -- silent off-by-one in this
    function has cost us whole rounds before, so it is checked, not trusted.

    dataset order: imgs, K, Tc, gt | depth | seg2d | boxdet(2) or
    agenttraj(4) | bbox2d(2) | ego | occ | tl | risk | lanegraph(4) |
    unknown_v2 | lidar_bev
    """
    def g(i):
        return batch[i].to(device, non_blocking=True)
    imgs, K, Tc, gt = g(0), g(1), g(2), g(3)
    bi = 4
    d = {}
    if a.do_depth:
        d["depth"] = g(bi); bi += 1
    if a.do_seg2d:
        d["seg2d"] = g(bi); bi += 1
    if a.do_traj:                         # boxes + traj (+4)
        d["boxes"], d["nbox"] = g(bi), g(bi + 1)
        d["traj"], d["tvalid"] = g(bi + 2), g(bi + 3)
        bi += 4
    elif a.do_box:                        # boxes only (+2)
        d["boxes"], d["nbox"] = g(bi), g(bi + 1)
        bi += 2
    if a.do_det2d:
        d["bb2d"], d["nb2d"] = g(bi), g(bi + 1); bi += 2
    if a.do_e2e:
        d["ego"] = g(bi); bi += 1
    if a.do_occ:
        d["occ"] = g(bi); bi += 1
    if a.do_tl:
        d["tl"] = g(bi); bi += 1
    if a.do_risk:
        d["risk"] = g(bi); bi += 1
    if a.do_lg:
        d["lg_pts"], d["lg_cls"] = g(bi), g(bi + 1)
        d["lg_n"], d["lg_adj"] = g(bi + 2), g(bi + 3)
        bi += 4
    if a.do_unk:
        d["unk"] = g(bi); bi += 1
    if a.do_pl:
        d["lb"] = g(bi); bi += 1
    assert bi == len(batch), (
        f"batch layout mismatch: consumed {bi} of {len(batch)} tensors -- "
        "the dataset append order and _unpack have diverged")
    return imgs, K, Tc, gt, d


@torch.no_grad()
def evaluate(frozen, ref0, loader, device, args, max_b=40):
    frozen.eval(); ref0.eval()
    iR = np.zeros((len(BANDS), N_CLASSES)); uR = iR.copy()
    iF = iR.copy(); uF = iR.copy()
    ade_r = ade_f = nseen = 0.0
    box_r = ([0, 0, 0], [0, 0, 0], [], [])
    box_f = ([0, 0, 0], [0, 0, 0], [], [])
    adec_r = adec_f = orc_r = orc_f = 0.0
    nturn_r = 0
    hm_r = hm_f = 0.0
    utp_r = ufp_r = ufn_r = utp_f = ufp_f = ufn_f = 0
    s_cm = [[0, 0, 0], [0, 0, 0]]      # raw/refined tp,fp,fn
    pl_cm = [[0.0, 0.0], [0.0, 0.0]]   # raw/refined inter,union
    for bi, batch in enumerate(loader):
        if bi >= max_b:
            break
        imgs, K, Tc, gt, _d = _unpack(batch, device, args)
        det_boxes = _d.get("boxes"); det_n = _d.get("nbox")
        # -1 = scene has no 3D-box annotation (x2gen2): never a "zero
        # objects" label, so clamp for the counting loops and mask the
        # box-reading losses below.
        _bv = (det_n >= 0) if det_n is not None else None
        det_n = det_n.clamp(min=0) if det_n is not None else None
        traj_gt = _d.get("traj"); tvalid = _d.get("tvalid")
        ego_gt = _d.get("ego"); risk_gt = _d.get("risk")
        unk_gt = _d.get("unk"); lb_gt = _d.get("lb")
        v0 = ego_gt[:, 12] if args.do_e2e else None
        with torch.autocast("cuda", torch.float16):
            out = frozen(imgs, K, Tc, v0)
            seg = out[0].float()
            ctx = frozen.lane_input().float() if args.ctx else None
            fused = frozen._fused_bev.float() if args.do_e2e else None
            r = ref0(seg=seg if args.do_seg else None,
                     hm=out[3].float() if args.do_box else None,
                     reg=out[4].float() if args.do_box else None,
                     ego=out[7].float() if args.do_e2e else None,
                     v0=v0, fused=fused, seg_ctx=ctx,
                     traj=out[9].float() if args.do_traj else None,
                     risk=out[12].float() if args.do_risk else None,
                     unk=out[17].float() if args.do_unk else None,
                     stat=(out[10].float()
                           if args.do_stat and len(out) > 10 else None),
                     pl=(out[18].float()
                         if args.do_pl and len(out) > 18 else None))
        # ---- 3D BBox: 帯別 recall + 位置誤差 (raw vs refined) ----
        # 2026-08-20 追加: box の効果がこれまで一切可視化されておらず、
        # accept 判定にも残らなかった。
        if args.do_box and "hm" in r and det_boxes is not None:
            for _tag, (_hm, _rg) in (("r", (out[3], out[4])),
                                     ("f", (r["hm"], r["reg"]))):
                _d = frozen.decode_boxes(_hm.float().cpu(), _rg.float().cpu(),
                                         thresh=0.25)
                for _b in range(det_boxes.shape[0]):
                    _pv = [(float(x[2]), float(x[3])) for x in _d[_b]
                           if float(x[0]) < 1.5]
                    for _k in range(int(det_n[_b])):
                        _c, _xe, _ye, _ln = [float(v) for v in
                                             det_boxes[_b, _k, :4]]
                        if _ln <= 0 or _c >= 1.5:
                            continue
                        _rr = (_xe * _xe + _ye * _ye) ** 0.5
                        _bi = 0 if _rr < 20 else (1 if _rr < 40 else 2)
                        _best = None
                        for _px, _py in _pv:
                            _d2 = (_xe - _px) ** 2 + (_ye - _py) ** 2
                            if _d2 < 9.0 and (_best is None or _d2 < _best[0]):
                                _best = (_d2, _px, _py)
                        _acc = box_r if _tag == "r" else box_f
                        _acc[0][_bi] += 1
                        if _best:
                            _acc[1][_bi] += 1
                            _acc[2].append(abs(_best[1] - _xe))
                            _acc[3].append(abs(_best[2] - _ye))
        if args.do_stat and "stat" in r:
            for tag, lg in (("r", out[10]), ("f", r["stat"])):
                p = lg.float()[:, 0].sigmoid()
                for b in range(det_boxes.shape[0]):
                    for k in range(int(det_n[b])):
                        if det_boxes[b, k, 3] <= 0 or tvalid[b, k, 5] < 0.5 \
                                or det_boxes[b, k, 0] >= 1.5:
                            continue
                        d3 = float(traj_gt[b, k, 5].norm())
                        if 0.35 < d3 < 0.8:
                            continue
                        ri = int((80.0 - float(det_boxes[b, k, 1])) / 0.4)
                        ci = int((50.0 - float(det_boxes[b, k, 2])) / 0.4)
                        if not (0 <= ri < p.shape[-2] and 0 <= ci < p.shape[-1]):
                            continue
                        ps = float(p[b, ri, ci]) > 0.5
                        gs = d3 <= 0.35
                        j = 0 if tag == "r" else 1
                        s_cm[j][0] += int(ps and gs)
                        s_cm[j][1] += int(ps and not gs)
                        s_cm[j][2] += int((not ps) and gs)
        if args.do_pl and "pl" in r and lb_gt is not None:
            g4 = lb_gt.float()
            v = g4.abs().sum((1, 2, 3)) > 0
            if v.any():
                og = g4[v, 3] > 0.5
                for j, lg in enumerate((out[18], r["pl"])):
                    op = lg.float()[v, 3] > 0
                    pl_cm[j][0] += float((op & og).sum())
                    pl_cm[j][1] += float((op | og).sum())
        if args.do_unk:
            pos = unk_gt > 0.5
            neg = (unk_gt > -0.5) & ~pos       # visible free cells only
            for tag, logit in (("r", out[17]), ("f", r["unk"])):
                p = logit.float().sigmoid()[:, 0] > 0.3
                tp = (p & pos).sum().item()
                fp = (p & neg).sum().item()
                fn = (~p & pos).sum().item()
                if tag == "r":
                    utp_r += tp; ufp_r += fp; ufn_r += fn
                else:
                    utp_f += tp; ufp_f += fp; ufn_f += fn
        if args.do_seg:
            pr, pf = seg.argmax(1), r["seg"].argmax(1)
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
        if args.do_e2e:
            valid = ego_gt[:, 16] > 0.5
            if valid.any():
                gtw = ego_gt[:, :12].view(-1, 6, 2)
                # SELECTOR, not oracle. This used to be d.min(1) -- the
                # candidate closest to the log, chosen with knowledge of the
                # log. That is an upper bound no vehicle can reach at run time,
                # and it is the wrong thing to steer a decision by: the mode
                # selector picks the best candidate only 0.70-0.74 of the time
                # and loses to "always take candidate 0" (measured ADE 1.021 vs
                # 1.006), so a refiner that improves candidates the selector
                # never commits to would look like progress and deliver none.
                # Match what bevlane/train.py's valE2E reports: argmax of the
                # mode logits, and ADEc over turning frames alongside ADE.
                turn = gtw[:, -1, 1].abs() > 2.0
                for tag, ev in (("r", out[7].float()), ("f", r["ego"].float())):
                    wp = ev[:, :12 * EGO_K].view(-1, EGO_K, 6, 2)
                    d = (wp - gtw[:, None]).pow(2).sum(-1).sqrt().mean(2)
                    lg = ev[:, 12 * EGO_K:12 * EGO_K + EGO_K]
                    k = lg.argmax(1, keepdim=True)
                    sel = d.gather(1, k).squeeze(1)
                    ade = sel[valid].mean().item()
                    vt = valid & turn
                    adec = sel[vt].mean().item() if vt.any() else float("nan")
                    orc = d.min(1).values[valid].mean().item()
                    if tag == "r":
                        ade_r += ade
                        adec_r += 0.0 if adec != adec else adec
                        orc_r += orc
                        nturn_r += int(vt.any())
                    else:
                        ade_f += ade
                        adec_f += 0.0 if adec != adec else adec
                        orc_f += orc
                nseen += 1
    res = {"seg": (iR, uR, iF, uF)}
    if args.do_box and sum(box_r[0]):
        res["box"] = (box_r, box_f)
    if args.do_e2e and nseen:
        res["ade"] = (ade_r / nseen, ade_f / nseen)
        nt = max(nturn_r, 1)
        res["adec"] = (adec_r / nt, adec_f / nt)
        res["orc"] = (orc_r / nseen, orc_f / nseen)
    if args.do_unk:
        res["unk"] = ((utp_r / max(utp_r + ufp_r, 1),
                       utp_r / max(utp_r + ufn_r, 1)),
                      (utp_f / max(utp_f + ufp_f, 1),
                       utp_f / max(utp_f + ufn_f, 1)))
    if args.do_stat and sum(s_cm[0]) > 0:
        res["stat"] = tuple((c[0] / max(c[0] + c[1], 1),
                             c[0] / max(c[0] + c[2], 1)) for c in s_cm)
    if args.do_pl and pl_cm[0][1] > 0:
        res["pl"] = tuple(c[0] / max(c[1], 1) for c in pl_cm)
    ref0.train()
    return res


def _report(res, ep, step, tag=""):
    if "box" in res:
        _br, _bf = res["box"]
        _nm = ["0-20m", "20-40m", "40m+"]
        _line = ""
        for _i in range(3):
            if _br[0][_i]:
                _line += (f"  {_nm[_i]} R {_br[1][_i]/_br[0][_i]:.3f}->"
                          f"{_bf[1][_i]/_bf[0][_i]:.3f}")
        _dr = np.mean(_br[3]) if _br[3] else float("nan")
        _df = np.mean(_bf[3]) if _bf[3] else float("nan")
        print(f"[refBox ep{ep} step{step}] {tag} veh recall raw->refined"
              f"{_line} | |dy| {_dr:.3f}->{_df:.3f} m", flush=True)
    if "seg" in res:
        iR, uR, iF, uF = res["seg"]
        if uR.sum():
            print(f"[refBand ep{ep} step{step}] {tag} seg raw->refined IoU",
                  flush=True)
            for c in [1, 3, 4, 5, 6]:
                row = "  " + CLASS_NAMES[c].ljust(10)
                for bidx, (nm, _, _) in enumerate(BANDS):
                    r = iR[bidx, c] / uR[bidx, c] if uR[bidx, c] else float("nan")
                    f_ = iF[bidx, c] / uF[bidx, c] if uF[bidx, c] else float("nan")
                    row += f"  {nm} {r:.3f}->{f_:.3f}"
                print(row, flush=True)
    if "adec" in res:
        print(f"[refE2Ec ep{ep} step{step}]  ADEc(selector) "
              f"{res['adec'][0]:.3f}->{res['adec'][1]:.3f}  "
              f"oracle {res['orc'][0]:.3f}->{res['orc'][1]:.3f}", flush=True)
    if "ade" in res:
        print(f"[refE2E ep{ep} step{step}] {tag} ADE raw->refined "
              f"{res['ade'][0]:.3f}->{res['ade'][1]:.3f}", flush=True)
    if "stat" in res:
        (pr, rr_), (pf, rf) = res["stat"]
        print(f"[refStat ep{ep} step{step}] {tag} stationary P/R raw "
              f"{pr:.3f}/{rr_:.3f} -> refined {pf:.3f}/{rf:.3f}", flush=True)
    if "pl" in res:
        print(f"[refPL ep{ep} step{step}] {tag} pseudo-LiDAR occIoU raw "
              f"{res['pl'][0]:.3f} -> refined {res['pl'][1]:.3f}", flush=True)
    if "unk" in res:
        (pr, rr_), (pf, rf) = res["unk"]
        print(f"[refUnk ep{ep} step{step}] {tag} pix P/R raw "
              f"{pr:.3f}/{rr_:.3f} -> refined {pf:.3f}/{rf:.3f}", flush=True)


def _verdict(res):
    """Which heads actually BEAT the frozen model on val, per head.

    The refiner is zero-init, so at step 0 refined == raw exactly; a head that
    ends up worse than raw is a head that must not be applied. Measured on
    r45 and r47: the seg head trades far-range road / road_edge / stopline IoU
    away, so it fails this gate while E2E / stationary / pseudo-LiDAR pass.
    Stored in the ckpt as `accept` and honoured by demo_rgbd_bev.py, so a
    losing head can never silently degrade a priority task."""
    v = {}
    if "seg" in res:
        iR, uR, iF, uF = res["seg"]
        dl = []
        percls = {}
        for c in range(1, N_CLASSES):
            d_ = []
            for b_ in range(len(BANDS)):
                if uR[b_, c] and uF[b_, c]:
                    d_.append(iF[b_, c] / uF[b_, c] - iR[b_, c] / uR[b_, c])
            if d_:
                percls[c] = d_
                dl += d_
        # every band/class matters: accept only if the mean does not drop and
        # no single band/class loses more than 1 IoU point
        v["seg"] = bool(dl) and (sum(dl) / len(dl) >= 0.0) and min(dl) > -0.01
        # Per-class list, DIAGNOSTICS ONLY -- do not apply it at inference.
        # Swapping single channels into an otherwise-raw logit field is invalid:
        # the residual is learned jointly, so a lone refined channel sits below
        # its untouched competitors and the argmax drops the class (measured:
        # laneline IoU 0.125 -> 0.0045, predicted pixels 0.134 % -> 0.0013 %).
        # The seg head is applied whole or not at all.
        v["seg_classes"] = sorted(
            c for c, d_ in percls.items()
            if sum(d_) / len(d_) > 0.0 and min(d_) > -0.005)
    if "adec" in res:
        print(f"[refE2Ec ep{ep} step{step}]  ADEc(selector) "
              f"{res['adec'][0]:.3f}->{res['adec'][1]:.3f}  "
              f"oracle {res['orc'][0]:.3f}->{res['orc'][1]:.3f}", flush=True)
    if "ade" in res:
        v["e2e"] = res["ade"][1] <= res["ade"][0]        # lower ADE is better
    if "stat" in res:
        (pr, rr_), (pf, rf) = res["stat"]
        f1r = 2 * pr * rr_ / max(pr + rr_, 1e-6)
        f1f = 2 * pf * rf / max(pf + rf, 1e-6)
        v["stat"] = f1f >= f1r
    if "pl" in res:
        v["pl"] = res["pl"][1] >= res["pl"][0]
    if "unk" in res:
        (pr, rr_), (pf, rf) = res["unk"]
        f1r = 2 * pr * rr_ / max(pr + rr_, 1e-6)
        f1f = 2 * pf * rf / max(pf + rf, 1e-6)
        v["unk"] = f1f >= f1r
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="out/bevlane")
    ap.add_argument("--ckpt", required=True, help="frozen main-model ckpt")
    ap.add_argument("--model", default="v39")
    ap.add_argument("--gt-key", default="gt_cons")
    ap.add_argument("--n-seg2d", type=int, default=21)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.0)
    ap.add_argument("--width", type=int, default=48)
    ap.add_argument("--ctx", type=int, default=0)
    ap.add_argument("--far-w", type=float, default=3.0)
    ap.add_argument("--bg-w", type=float, default=0.5)
    ap.add_argument("--lovasz-w", type=float, default=0.3)
    ap.add_argument("--ce-w", type=float, default=1.0,
                    help="weight of the (far-row-weighted) CE term; drop it "
                         "below the lovasz weight to optimise what is measured")
    ap.add_argument("--class-iou-w", type=float, default=0.0,
                    help="per-class soft-IoU term on sidewalk/stopline/parking "
                         "-- the classes r48 regressed")
    ap.add_argument("--do-seg", action="store_true", default=True)
    ap.add_argument("--no-seg", dest="do_seg", action="store_false")
    ap.add_argument("--do-box", action="store_true", default=False)
    ap.add_argument("--do-e2e", action="store_true", default=False)
    ap.add_argument("--do-traj", action="store_true", default=False,
                    help="refine the other-agent trajectory field (out[9])")
    ap.add_argument("--do-risk", action="store_true", default=False,
                    help="refine the risk field (out[12])")
    ap.add_argument("--do-unk", action="store_true", default=False,
                    help="refine the dense unknown logit (out[17], v41+)")
    ap.add_argument("--unk-w", type=float, default=2.0)
    ap.add_argument("--unk-key", default="unknown_v3")
    ap.add_argument("--eval-samples", type=int, default=480,
                    help="val samples in the report/verdict slice; taken with "
                         "a stride so they span every val scene")
    ap.add_argument("--cam-drop", type=float, default=0.0,
                    help="probability of zeroing CAM_BACK_NARROW on an "
                         "8-camera sample (keeps the 7-camera rig calibrated)")
    ap.add_argument("--do-stat", action="store_true",
                    help="refine the stationary flag (out[10])")
    ap.add_argument("--stat-w", type=float, default=1.0)
    ap.add_argument("--do-pl", action="store_true",
                    help="refine the pseudo-LiDAR raster (v48 out[18])")
    ap.add_argument("--pl-w", type=float, default=1.0)
    ap.add_argument("--do-depth", action="store_true")
    ap.add_argument("--depth-w", type=float, default=0.6)
    ap.add_argument("--do-seg2d", action="store_true")
    ap.add_argument("--seg2d-key", default="seg2d21")
    ap.add_argument("--seg2d-w", type=float, default=0.35)
    ap.add_argument("--do-det2d", action="store_true")
    ap.add_argument("--bbox2d-w", type=float, default=0.25)
    ap.add_argument("--do-occ", action="store_true")
    ap.add_argument("--occ-w", type=float, default=0.4)
    ap.add_argument("--do-tl", action="store_true")
    ap.add_argument("--tl-w", type=float, default=0.6)
    ap.add_argument("--do-flow", action="store_true")
    ap.add_argument("--flow-w", type=float, default=0.3)
    ap.add_argument("--do-lg", action="store_true")
    ap.add_argument("--lanegraph-w", type=float, default=0.5)
    ap.add_argument("--yaw-fix-deg", type=float, default=0.0,
                    help="GT 層間回転の補正 (train.py と同じ、out/yawfix_plan.md)")
    ap.add_argument("--do-all", action="store_true",
                    help="refine every head the network emits")
    ap.add_argument("--box-w", type=float, default=1.0)
    ap.add_argument("--e2e-w", type=float, default=1.0)
    ap.add_argument("--traj-w", type=float, default=0.5)
    ap.add_argument("--risk-w", type=float, default=0.3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit-train", type=int, default=0)
    ap.add_argument("--out", default="out/refiner_r34")
    ap.add_argument("--shuffle-seed", type=int, default=0)
    ap.add_argument("--exclude-scenes", default=None,
                    help="file of scene names to drop from refiner training "
                         "(e.g. US batch without unknown_v3 / odd calib)")
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
    if args.exclude_scenes:
        excl = {l.strip() for l in open(args.exclude_scenes) if l.strip()}
        n0 = len(train_s)
        train_s = [s_ for s_ in train_s if s_ not in excl]
        print(f"[exclude] {n0 - len(train_s)} scenes dropped "
              f"({args.exclude_scenes})", flush=True)

    # frozen backbone (eval, no grad)
    frozen = MODELS[args.model](n_seg=args.n_seg2d).to(device)
    sd = torch.load(args.ckpt, map_location="cpu")
    sd = sd.get("model", sd)
    miss, unexp = frozen.load_state_dict(sd, strict=False)
    frozen.eval()
    for p in frozen.parameters():
        p.requires_grad_(False)

    if args.do_all:
        for _f in ("do_seg", "do_box", "do_e2e", "do_traj", "do_risk",
                   "do_unk", "do_stat", "do_pl", "do_depth", "do_seg2d",
                   "do_det2d", "do_occ", "do_tl", "do_flow", "do_lg"):
            setattr(args, _f, True)
    ref = MultiTaskRefiner(do_seg=args.do_seg, do_box=args.do_box,
                           do_e2e=args.do_e2e, do_traj=args.do_traj,
                           do_risk=args.do_risk, do_unk=args.do_unk,
                           do_stat=args.do_stat, do_pl=args.do_pl,
                           do_depth=args.do_depth, do_seg2d=args.do_seg2d,
                           do_det2d=args.do_det2d, do_occ=args.do_occ,
                           do_tl=args.do_tl, do_flow=args.do_flow,
                           do_lg=args.do_lg, n_seg2d=args.n_seg2d,
                           n_cls=N_CLASSES,
                           seg_width=args.width, seg_ctx=args.ctx,
                           ego_dim=EGO_DIM).to(device)
    if is_main:
        n_par = sum(p.numel() for p in ref.parameters()) / 1e6
        print(f"[frozen] {args.ckpt} missing={len(miss)} unexpected={len(unexp)}",
              flush=True)
        print(f"[refiner] heads: seg={args.do_seg} box={args.do_box} "
              f"e2e={args.do_e2e} traj={args.do_traj} risk={args.do_risk} "
              f"unk={args.do_unk}({args.unk_key}) "
              f"stat={args.do_stat} pl={args.do_pl} "
              f"depth={args.do_depth} seg2d={args.do_seg2d} "
              f"det2d={args.do_det2d} occ={args.do_occ} "
              f"tl={args.do_tl} flow={args.do_flow} "
              f"lg={args.do_lg} | "
              f"params={n_par:.2f}M (zero-init residual)", flush=True)
    # auto-resume: continue from a previous epoch save if present, replacing
    # any non-finite entries (poisoned BN stats) with safe defaults
    rck = os.path.join(args.out, "last.pt")
    if os.path.exists(rck):
        rsd = torch.load(rck, map_location="cpu").get("refiner", {})
        fixed = 0
        for k, v in rsd.items():
            m_ = ~torch.isfinite(v)
            if m_.any():
                v[m_] = 1.0 if "running_var" in k else 0.0
                fixed += 1
        miss_r, _ = ref.load_state_dict(rsd, strict=False)
        if is_main:
            print(f"[resume] {rck} missing={len(miss_r)} "
                  f"sanitized={fixed} tensors", flush=True)
    ref = DDP(ref, device_ids=[local]) if ddp else ref
    ref0 = ref.module if ddp else ref

    # with_agenttraj provides boxes+traj; else with_boxdet gives boxes.
    dkw = dict(with_depth=args.do_depth, with_agenttraj=args.do_traj,
               with_seg2d=args.do_seg2d, seg2d_key=args.seg2d_key,
               with_bbox2d=args.do_det2d, with_occ=args.do_occ,
               with_tl=args.do_tl, with_lanegraph=args.do_lg,
               with_boxdet=args.do_box and not args.do_traj,
               with_ego=args.do_e2e, with_risk=args.do_risk,
               with_unknown_v2=args.do_unk, unk2_key=args.unk_key,
               with_lidarbev=args.do_pl,
               yaw_fix_deg=args.yaw_fix_deg)
    tr = BevLaneDataset(args.root, train_s, gt_key=args.gt_key,
                        cam_drop=args.cam_drop, **dkw)
    va = BevLaneDataset(args.root, val_s, max_per_scene=4, gt_key=args.gt_key,
                        **dkw)
    sampler = DistributedSampler(tr) if ddp else None
    dl = DataLoader(tr, batch_size=args.batch, shuffle=sampler is None,
                    sampler=sampler, num_workers=args.workers, pin_memory=True,
                    drop_last=True, persistent_workers=args.workers > 0)
    # The reports (and the per-head ACCEPTANCE verdict written into the ckpt)
    # used to read the first max_b batches of an unshuffled loader: 480 samples
    # = 60 of 270 val scenes, all from the head of the list. A verdict decided
    # on the head of the list is not a verdict about the val set, and the seg
    # head was rejected on exactly that basis. Stride the slice so the same
    # evaluation cost spans every val scene; deterministic across epochs.
    _st = max(1, len(va) // max(args.eval_samples, 1))
    va_ev = torch.utils.data.Subset(va, list(range(0, len(va), _st)))
    dv = DataLoader(va_ev, batch_size=args.batch, shuffle=False,
                    num_workers=2, pin_memory=True)
    if is_main:
        _sc = {va.items[i][0] for i in va_ev.indices}
        print(f"[val] eval slice: {len(va_ev)} samples over {len(_sc)} of "
              f"{len(val_s)} scenes (stride {_st})", flush=True)
    if is_main:
        print(f"train {len(tr)} / {len(train_s)} scenes; "
              f"val {len(va)} / {len(val_s)} scenes; world={world} lr={lr:.1e}",
              flush=True)

    cw = CLASS_W.clone().to(device)
    cw[0] = args.bg_w
    opt = torch.optim.AdamW(ref0.parameters(), lr=lr, weight_decay=1e-4)
    steps = args.epochs * (args.limit_train or len(dl))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    scaler = torch.cuda.amp.GradScaler()

    step = 0
    for ep in range(args.epochs):
        if ddp:
            sampler.set_epoch(ep + args.shuffle_seed)
        for bidx, batch in enumerate(dl):
            if args.limit_train and bidx >= args.limit_train:
                break
            imgs, K, Tc, gt, _d = _unpack(batch, device, args)
            det_boxes = _d.get("boxes"); det_n = _d.get("nbox")
            # -1 = the scene has no 3D-box annotation (x2gen2): never a
            # "zero objects" label. Mask the box-reading losses instead.
            # DDP safety: same graph on every rank (see train.py) -- select
            # all rows and zero-weight the loss when none is annotated.
            _bv0 = (det_n >= 0) if det_n is not None else None
            _bw = 1.0 if (_bv0 is not None and bool(_bv0.any())) else 0.0
            _bv = _bv0 if _bw else (torch.ones_like(_bv0)
                                    if _bv0 is not None else None)
            det_n = det_n.clamp(min=0) if det_n is not None else None
            traj_gt = _d.get("traj"); tvalid = _d.get("tvalid")
            ego_gt = _d.get("ego"); risk_gt = _d.get("risk")
            unk_gt = _d.get("unk"); lb_gt = _d.get("lb")
            v0 = ego_gt[:, 12] if args.do_e2e else None
            has_box = args.do_box or args.do_traj      # det_boxes available
            with torch.no_grad(), torch.autocast("cuda", torch.float16):
                out = frozen(imgs, K, Tc, v0)
                seg = out[0].float()
                hm = out[3].float() if args.do_box else None
                reg = out[4].float() if args.do_box else None
                ego = out[7].float() if args.do_e2e else None
                traj = out[9].float() if args.do_traj else None
                risk = out[12].float() if args.do_risk else None
                unk = out[17].float() if args.do_unk else None
                stat_o = (out[10].float()
                          if args.do_stat and len(out) > 10 else None)
                pl_o = (out[18].float()
                        if args.do_pl and len(out) > 18 else None)
                ctx = frozen.lane_input().float() if args.ctx else None
                fused = frozen._fused_bev.float() if args.do_e2e else None
            # snapshot refiner BN stats: one pathological batch poisons them
            # IN the forward pass, before any loss check can catch it
            bn_bak = {k: v.detach().clone() for k, v in ref0.named_buffers()
                      if "running_" in k}
            # refiner runs in fp32: it is tiny (~8M params) and its BN
            # stats kept getting poisoned by fp16 overflow on outlier
            # frozen-output batches (r45 incident; fp16 gave no real speedup)
            with torch.autocast("cuda", enabled=False):
                r = ref(seg=seg if args.do_seg else None, hm=hm, reg=reg,
                        ego=ego, v0=v0, fused=fused, seg_ctx=ctx,
                        traj=traj, risk=risk, unk=unk,
                        stat=stat_o, pl=pl_o,
                        depth=out[1].float() if args.do_depth else None,
                        seg2d=out[2].float() if args.do_seg2d else None,
                        hm2d=([t.float() for t in out[5]]
                              if args.do_det2d else None),
                        reg2d=([t.float() for t in out[6]]
                               if args.do_det2d else None),
                        occ=out[8].float() if args.do_occ else None,
                        tl=out[11].float() if args.do_tl else None,
                        flow=out[13].float() if args.do_flow else None,
                        lg=((out[14].float(), out[15].float(),
                             out[16].float()) if args.do_lg else None))
                loss = seg.new_zeros(())
                if args.do_seg:
                    rs = r["seg"].float()
                    # The seg head is graded on BANDED PER-CLASS IoU, but was
                    # trained on far-row-weighted CE. Optimising CE with a x3
                    # far weight while measuring IoU is how the head ended up
                    # LOSING to the frozen model on r45 and r47 (road 40-80m
                    # 0.474->0.420, stopline 0-20m 0.251->0.208): CE rewards
                    # confident far-range road, IoU punishes the false
                    # positives that come with it. lovasz_softmax IS the IoU
                    # surrogate, so it leads and CE only regularises.
                    ce = F.cross_entropy(rs, gt, weight=cw,
                                         ignore_index=-100, reduction="none")
                    H2 = ce.shape[-2]
                    rows = torch.arange(H2, device=device, dtype=ce.dtype)
                    wrow = 1 + args.far_w * (1 - rows / (H2 - 1)).clamp(min=0)
                    loss = loss + args.ce_w * (ce * wrow.view(1, -1, 1)).mean()
                    if args.lovasz_w > 0:
                        loss = loss + args.lovasz_w * lovasz_softmax(
                            rs, gt, ignore=0)
                    if args.class_iou_w > 0:
                        # per-class soft IoU, so a class that the round
                        # regressed (r48: sidewalk 0.573->0.556, stopline
                        # 0.138->0.130, parking 0.264->0.248) gets its own
                        # gradient instead of being averaged away
                        p_ = rs.softmax(1)
                        for c in (2, 5, 8):
                            gi = (gt == c).float()
                            if gi.sum() < 1:
                                continue
                            pi = p_[:, c]
                            inter = (pi * gi).sum()
                            uni = pi.sum() + gi.sum() - inter
                            loss = loss + args.class_iou_w * (
                                1.0 - inter / uni.clamp(min=1.0))
                if args.do_box and _bv is not None:
                    loss = loss + args.box_w * _bw * ref0_boxloss(
                        frozen, r["hm"][_bv], r["reg"][_bv],
                        det_boxes[_bv], det_n[_bv])
                if args.do_e2e:
                    loss = loss + args.e2e_w * frozen.ego_loss(
                        r["ego"].float(), ego_gt)
                if args.do_traj and _bv is not None:
                    loss = loss + args.traj_w * _bw * frozen.traj_loss(
                        r["traj"].float()[_bv], det_boxes[_bv], det_n[_bv],
                        traj_gt[_bv], tvalid[_bv])
                if args.do_risk:
                    loss = loss + args.risk_w * frozen.risk_loss(
                        r["risk"].float(), risk_gt)
                if args.do_unk:
                    # v41+ alpha-focal w/ pos_weight; -1 = don't-care
                    loss = loss + args.unk_w * frozen.unk_dense_loss(
                        r["unk"].float(), unk_gt)
                if args.do_stat and "stat" in r:
                    loss = loss + args.stat_w * frozen.stat_loss(
                        r["stat"].float()[_bv], det_boxes[_bv], det_n[_bv],
                        traj_gt[_bv], tvalid[_bv])
                if args.do_pl and "pl" in r:
                    loss = loss + args.pl_w * frozen.pseudo_lidar_loss(
                        r["pl"].float(), lb_gt)
                if args.do_depth and "depth" in r:
                    loss = loss + args.depth_w * frozen.depth_loss(
                        r["depth"].float(), _d["depth"])
                if args.do_seg2d and "seg2d" in r:
                    loss = loss + args.seg2d_w * frozen.seg2d_loss(
                        r["seg2d"].float(), _d["seg2d"])
                if args.do_det2d and "hm2d" in r:
                    loss = loss + args.bbox2d_w * frozen.bbox2d_loss(
                        r["hm2d"], r["reg2d"], _d["bb2d"], _d["nb2d"])
                if args.do_occ and "occ" in r:
                    loss = loss + args.occ_w * frozen.occ_loss(
                        r["occ"].float(), _d["occ"])
                if args.do_tl and "tl" in r:
                    loss = loss + args.tl_w * frozen.tl_loss(
                        r["tl"].float(), _d["tl"])
                if args.do_flow and "flow" in r:
                    loss = loss + args.flow_w * _bw * frozen.flow_loss(
                        r["flow"].float()[_bv], det_boxes[_bv], det_n[_bv],
                        traj_gt[_bv], tvalid[_bv])
                if args.do_lg and "lg_pts" in r:
                    loss = loss + args.lanegraph_w * frozen.lanegraph_loss(
                        r["lg_pts"].float(), r["lg_meta"].float(),
                        r["lg_adj"].float(), _d["lg_pts"], _d["lg_cls"],
                        _d["lg_n"], _d["lg_adj"])
            opt.zero_grad(set_to_none=True)
            # DDP-safe non-finite guard: ALL ranks must agree, else a rank that
            # skips backward() deadlocks the others on the grad all-reduce.
            fin = torch.tensor([float(torch.isfinite(loss))], device=device)
            if ddp:
                dist.all_reduce(fin, op=dist.ReduceOp.MIN)   # 0 if any rank bad
            if fin.item() < 1.0:
                # restore pre-batch BN stats: the bad forward already
                # updated them (this, not the grads, was what kept
                # poisoning checkpoints)
                with torch.no_grad():
                    for k, v in ref0.named_buffers():
                        if "running_" in k:
                            v.copy_(bn_bak[k])
                sched.step(); step += 1
                nskip = getattr(main, "_nskip", 0) + 1
                main._nskip = nskip
                if is_main:
                    print(f"ep{ep} step{step} SKIP non-finite (all ranks)",
                          flush=True)
                # r39 lesson: once BN stats are poisoned EVERY step skips and
                # the normal-path detector below never runs -> check here too
                if nskip >= 20:
                    if is_main:
                        print(f"ep{ep} step{step} {nskip} consecutive skips "
                              "-- poisoned state, exiting for clean relaunch",
                              flush=True)
                    if ddp:
                        dist.destroy_process_group()
                    sys.exit(3)
                continue
            main._nskip = 0
            scaler.scale(loss).backward()
            scaler.unscale_(opt)                 # clip in true grad scale
            torch.nn.utils.clip_grad_norm_(ref0.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            step += 1
            if is_main and step % 100 == 0:
                print(f"ep{ep} step{step}/{steps} loss={loss.item():.4f} "
                      f"lr={sched.get_last_lr()[0]:.2e}", flush=True)
            if step % 200 == 0:
                # BN running stats are updated in FORWARD: one inf batch
                # poisons them permanently and the loss-skip guard cannot
                # help. Detect and hard-exit -> the retry wrapper relaunches
                # from a clean state instead of skip-looping forever.
                bad = any(not torch.isfinite(t).all()
                          for t in list(ref0.parameters())
                          + list(ref0.buffers()))
                flag = torch.tensor([float(bad)], device=device)
                if ddp:
                    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
                if flag.item() > 0:
                    if is_main:
                        print(f"ep{ep} step{step} POISONED PARAMS/BUFFERS "
                              "-- exiting for clean relaunch", flush=True)
                    if ddp:
                        dist.destroy_process_group()
                    sys.exit(3)
            if is_main and step % 1000 == 0:
                _report(evaluate(frozen, ref0, dv, device, args), ep, step)
                # periodic save: BN-poisoning incidents cost 20k steps when
                # only epoch-end saves existed. Never overwrite with a state
                # that is finite-but-huge (r43 incident: such a save poisoned
                # every subsequent resume) — keep the previous good save.
                sd_ = ref0.state_dict()
                ok_ = all(torch.isfinite(v).all()
                          and float(v.abs().max()) < 1e6
                          for v in sd_.values() if v.numel())
                if ok_:
                    torch.save({"refiner": sd_, "epoch": ep,
                                "step": step, "args": vars(args)},
                               os.path.join(args.out, "last.pt"))
                else:
                    print(f"ep{ep} step{step} SAVE SKIPPED "
                          "(non-finite/huge state)", flush=True)
        if is_main:
            torch.save({"refiner": ref0.state_dict(), "epoch": ep,
                        "args": vars(args)},
                       os.path.join(args.out, "last.pt"))
            print(f"[ckpt] saved epoch {ep}", flush=True)
    if is_main:
        fin = evaluate(frozen, ref0, dv, device, args,
                       max_b=len(dv))          # the whole strided slice
        _report(fin, args.epochs, step, tag="FINAL")
        acc = _verdict(fin)
        sc = acc.get("seg_classes")
        print(f"[refAccept] heads that beat the frozen model: "
              f"{ {k: v for k, v in acc.items() if k != 'seg_classes'} }",
              flush=True)
        if sc is not None:
            print(f"[refAccept] seg classes accepted: "
                  f"{[CLASS_NAMES[c] for c in sc]}", flush=True)
        pth = os.path.join(args.out, "last.pt")
        if os.path.exists(pth):
            _ck = torch.load(pth, map_location="cpu")
            _ck["accept"] = acc
            torch.save(_ck, pth)
            print(f"[refAccept] written into {pth}", flush=True)
        print("REFINER DONE", flush=True)
    if ddp:
        dist.destroy_process_group()


def ref0_boxloss(frozen, hm, reg, boxes, nbox):
    """Frozen model owns build_det_targets + the focal/L1 box loss; reuse it
    on the refined maps."""
    return frozen.boxdet_loss(hm.float(), reg.float(), boxes, nbox)


if __name__ == "__main__":
    main()

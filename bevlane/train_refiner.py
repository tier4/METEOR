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
    """Batch layout (counter identical to train.py). with_agenttraj provides
    boxes+traj (+4); else with_boxdet gives boxes (+2); then ego (+1), risk
    (+1)."""
    def g(i):
        return batch[i].to(device, non_blocking=True)
    imgs, K, Tc, gt = g(0), g(1), g(2), g(3)
    bi = 4
    det_boxes = det_n = traj_gt = tvalid = ego_gt = risk_gt = None
    if a.do_traj:                         # boxes + traj (+4)
        det_boxes, det_n = g(bi), g(bi + 1)
        traj_gt, tvalid = g(bi + 2), g(bi + 3)
        bi += 4
    elif a.do_box:                        # boxes only (+2)
        det_boxes, det_n = g(bi), g(bi + 1)
        bi += 2
    if a.do_e2e:
        ego_gt = g(bi); bi += 1
    if a.do_risk:
        risk_gt = g(bi); bi += 1
    unk_gt = None
    if a.do_unk:                          # dense unknown mask (+1)
        unk_gt = g(bi); bi += 1
    return (imgs, K, Tc, gt, det_boxes, det_n, traj_gt, tvalid, ego_gt,
            risk_gt, unk_gt)


@torch.no_grad()
def evaluate(frozen, ref0, loader, device, args, max_b=40):
    frozen.eval(); ref0.eval()
    iR = np.zeros((len(BANDS), N_CLASSES)); uR = iR.copy()
    iF = iR.copy(); uF = iR.copy()
    ade_r = ade_f = nseen = 0.0
    hm_r = hm_f = 0.0
    utp_r = ufp_r = ufn_r = utp_f = ufp_f = ufn_f = 0
    for bi, batch in enumerate(loader):
        if bi >= max_b:
            break
        (imgs, K, Tc, gt, det_boxes, det_n, traj_gt, tvalid,
         ego_gt, risk_gt, unk_gt) = _unpack(batch, device, args)
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
                     unk=out[17].float() if args.do_unk else None)
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
                for tag, ev in (("r", out[7].float()), ("f", r["ego"].float())):
                    wp = ev[:, :12 * EGO_K].view(-1, EGO_K, 6, 2)
                    d = (wp - gtw[:, None]).pow(2).sum(-1).sqrt().mean(2)  # [B,K]
                    ade = d.min(1).values[valid].mean().item()
                    if tag == "r":
                        ade_r += ade
                    else:
                        ade_f += ade
                nseen += 1
    res = {"seg": (iR, uR, iF, uF)}
    if args.do_e2e and nseen:
        res["ade"] = (ade_r / nseen, ade_f / nseen)
    if args.do_unk:
        res["unk"] = ((utp_r / max(utp_r + ufp_r, 1),
                       utp_r / max(utp_r + ufn_r, 1)),
                      (utp_f / max(utp_f + ufp_f, 1),
                       utp_f / max(utp_f + ufn_f, 1)))
    ref0.train()
    return res


def _report(res, ep, step, tag=""):
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
    if "ade" in res:
        print(f"[refE2E ep{ep} step{step}] {tag} ADE raw->refined "
              f"{res['ade'][0]:.3f}->{res['ade'][1]:.3f}", flush=True)
    if "unk" in res:
        (pr, rr_), (pf, rf) = res["unk"]
        print(f"[refUnk ep{ep} step{step}] {tag} pix P/R raw "
              f"{pr:.3f}/{rr_:.3f} -> refined {pf:.3f}/{rf:.3f}", flush=True)


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
    ap.add_argument("--box-w", type=float, default=1.0)
    ap.add_argument("--e2e-w", type=float, default=1.0)
    ap.add_argument("--traj-w", type=float, default=0.5)
    ap.add_argument("--risk-w", type=float, default=0.3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit-train", type=int, default=0)
    ap.add_argument("--out", default="out/refiner_r34")
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

    # frozen backbone (eval, no grad)
    frozen = MODELS[args.model](n_seg=args.n_seg2d).to(device)
    sd = torch.load(args.ckpt, map_location="cpu")
    sd = sd.get("model", sd)
    miss, unexp = frozen.load_state_dict(sd, strict=False)
    frozen.eval()
    for p in frozen.parameters():
        p.requires_grad_(False)

    ref = MultiTaskRefiner(do_seg=args.do_seg, do_box=args.do_box,
                           do_e2e=args.do_e2e, do_traj=args.do_traj,
                           do_risk=args.do_risk, do_unk=args.do_unk,
                           n_cls=N_CLASSES,
                           seg_width=args.width, seg_ctx=args.ctx,
                           ego_dim=EGO_DIM).to(device)
    if is_main:
        n_par = sum(p.numel() for p in ref.parameters()) / 1e6
        print(f"[frozen] {args.ckpt} missing={len(miss)} unexpected={len(unexp)}",
              flush=True)
        print(f"[refiner] heads: seg={args.do_seg} box={args.do_box} "
              f"e2e={args.do_e2e} traj={args.do_traj} risk={args.do_risk} "
              f"unk={args.do_unk}({args.unk_key}) | "
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
    dkw = dict(with_depth=False, with_agenttraj=args.do_traj,
               with_boxdet=args.do_box and not args.do_traj,
               with_ego=args.do_e2e, with_risk=args.do_risk,
               with_unknown_v2=args.do_unk, unk2_key=args.unk_key)
    tr = BevLaneDataset(args.root, train_s, gt_key=args.gt_key, **dkw)
    va = BevLaneDataset(args.root, val_s, max_per_scene=4, gt_key=args.gt_key,
                        **dkw)
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
    cw[0] = args.bg_w
    opt = torch.optim.AdamW(ref0.parameters(), lr=lr, weight_decay=1e-4)
    steps = args.epochs * (args.limit_train or len(dl))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    scaler = torch.cuda.amp.GradScaler()

    step = 0
    for ep in range(args.epochs):
        if ddp:
            sampler.set_epoch(ep)
        for bidx, batch in enumerate(dl):
            if args.limit_train and bidx >= args.limit_train:
                break
            (imgs, K, Tc, gt, det_boxes, det_n, traj_gt, tvalid,
             ego_gt, risk_gt, unk_gt) = _unpack(batch, device, args)
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
                ctx = frozen.lane_input().float() if args.ctx else None
                fused = frozen._fused_bev.float() if args.do_e2e else None
            with torch.autocast("cuda", torch.float16):
                r = ref(seg=seg if args.do_seg else None, hm=hm, reg=reg,
                        ego=ego, v0=v0, fused=fused, seg_ctx=ctx,
                        traj=traj, risk=risk, unk=unk)
                loss = seg.new_zeros(())
                if args.do_seg:
                    rs = r["seg"].float()
                    ce = F.cross_entropy(rs, gt, weight=cw,
                                         ignore_index=-100, reduction="none")
                    H2 = ce.shape[-2]
                    rows = torch.arange(H2, device=device, dtype=ce.dtype)
                    wrow = 1 + args.far_w * (1 - rows / (H2 - 1)).clamp(min=0)
                    loss = loss + (ce * wrow.view(1, -1, 1)).mean()
                    if args.lovasz_w > 0:
                        loss = loss + args.lovasz_w * lovasz_softmax(
                            rs, gt, ignore=0)
                if args.do_box:
                    loss = loss + args.box_w * ref0_boxloss(
                        frozen, r["hm"], r["reg"], det_boxes, det_n)
                if args.do_e2e:
                    loss = loss + args.e2e_w * frozen.ego_loss(
                        r["ego"].float(), ego_gt)
                if args.do_traj:
                    loss = loss + args.traj_w * frozen.traj_loss(
                        r["traj"].float(), det_boxes, det_n, traj_gt, tvalid)
                if args.do_risk:
                    loss = loss + args.risk_w * frozen.risk_loss(
                        r["risk"].float(), risk_gt)
                if args.do_unk:
                    # v41+ alpha-focal w/ pos_weight; -1 = don't-care
                    loss = loss + args.unk_w * frozen.unk_dense_loss(
                        r["unk"].float(), unk_gt)
            opt.zero_grad(set_to_none=True)
            # DDP-safe non-finite guard: ALL ranks must agree, else a rank that
            # skips backward() deadlocks the others on the grad all-reduce.
            fin = torch.tensor([float(torch.isfinite(loss))], device=device)
            if ddp:
                dist.all_reduce(fin, op=dist.ReduceOp.MIN)   # 0 if any rank bad
            if fin.item() < 1.0:
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
        if is_main:
            torch.save({"refiner": ref0.state_dict(), "epoch": ep,
                        "args": vars(args)},
                       os.path.join(args.out, "last.pt"))
            print(f"[ckpt] saved epoch {ep}", flush=True)
    if is_main:
        _report(evaluate(frozen, ref0, dv, device, args, max_b=120),
                args.epochs, step, tag="FINAL")
        print("REFINER DONE", flush=True)
    if ddp:
        dist.destroy_process_group()


def ref0_boxloss(frozen, hm, reg, boxes, nbox):
    """Frozen model owns build_det_targets + the focal/L1 box loss; reuse it
    on the refined maps."""
    return frozen.boxdet_loss(hm.float(), reg.float(), boxes, nbox)


if __name__ == "__main__":
    main()

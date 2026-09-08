#!/usr/bin/env python3
"""Rule-based rewards for the K E2E trajectory candidates + a GRPO-style
group-relative policy loss on the mode logits (r48 reinforcement stage).

Why this shape of RL: there is no reactive simulator, but the network
already emits EGO_K candidate trajectories and a mode logit per candidate,
and the imitation loss only teaches "the candidate closest to the log" --
never *which* candidate to commit to. Scoring the candidates with rules we
can evaluate on logged data (drivable area, time-resolved agent collision,
comfort, progress, red-light compliance from the v47 traffic-light input)
turns mode selection into a K-armed bandit, so a group-relative policy
gradient needs no value function, no simulator and no new parameters.

Safety: the reward includes the imitation error, so a rule-compliant but
absurd candidate cannot win, and the loss is additive to the existing
imitation objective -- priority tasks keep their supervision untouched.

All geometry is in the ego frame, x forward / y left, metres:
  wp     [B,K,T,2]  candidate waypoints (T=6 @0.5 s)
  gt     [B,800,500] BEV semantic GT at 0.2 m (row=(80-x)/0.2, col=(50-y)/0.2)
  boxes  [B,N,6]    agent boxes (cls, x, y, l, w, yaw)
  traj   [B,N,T,2]  agent displacement per step from its box centre
  tl     [B,8,7,27,48] per-camera traffic-light raster (v47 input)
"""
import torch
import torch.nn.functional as F

BEV_RES, BEV_XH, BEV_YH = 0.2, 80.0, 50.0
DRIVABLE = (1, 3, 4, 5, 7)          # road, crosswalk, laneline, stopline, marking
IGN = 255                            # don't-care cells never penalise
FRONT_CAMS = (0, 6)                  # CAM_FRONT_WIDE, CAM_FRONT_NARROW
COLL_R = 3.0                         # m: footprint + margin
REW_W = {"drive": 1.0, "coll": 1.5, "comfort": 0.4, "progress": 0.5,
         "tl": 1.0, "imit": 0.5}


def candidate_rewards(wp, gt=None, boxes=None, nbox=None, traj=None,
                      tvalid=None, tl=None, v0=None, ego_gt=None, w=None):
    """-> rewards [B,K] (higher is better) and a dict of scalar parts."""
    w = dict(REW_W, **(w or {}))
    wp = wp.float()
    B, K, T, _ = wp.shape
    dev = wp.device
    x, y = wp[..., 0], wp[..., 1]
    parts = {}

    # --- drivable area -------------------------------------------------
    if gt is not None:
        H, W = gt.shape[-2:]
        r = ((BEV_XH - x) / BEV_RES).long()
        c = ((BEV_YH - y) / BEV_RES).long()
        inwin = (r >= 0) & (r < H) & (c >= 0) & (c < W)
        idx = (r.clamp(0, H - 1) * W + c.clamp(0, W - 1)).view(B, -1)
        cls = torch.gather(gt.reshape(B, -1), 1, idx).view(B, K, T)
        ok = torch.zeros_like(cls, dtype=torch.bool)
        for d in DRIVABLE:
            ok |= cls == d
        ok |= cls == IGN                     # unlabelled: no opinion
        ok &= inwin
        drive = ok.float().mean(-1) - 1.0    # 0 (all on-road) .. -1
    else:
        drive = torch.zeros(B, K, device=dev)
    parts["drive"] = float(drive.mean())

    # --- time-resolved agent collision ---------------------------------
    if boxes is not None and traj is not None and tvalid is not None:
        bx = boxes.float()
        ap = bx[:, :, 1:3].unsqueeze(2) + traj.float()[:, :, :T]     # [B,N,T,2]
        live = (bx[:, :, 3] > 0).unsqueeze(-1) & (tvalid[:, :, :T] > 0.5)
        d = (wp.unsqueeze(1) - ap.unsqueeze(2)).norm(dim=-1)          # B,N,K,T
        # Two ranges. The hard term is the original 3 m footprint penalty; the
        # soft one keeps a gradient out to 9 m. Measured: with the hard term
        # alone, 60 % of frames gave all K candidates the SAME collision score
        # and the term contributed 1 % of what separated them -- the candidates
        # are only 0.82 m apart, so they all sit inside or outside the 3 m
        # circle together. The heaviest-weighted term in the reward (1.5) was
        # deciding almost nothing.
        pen = ((1.0 - d / COLL_R).clamp(min=0) ** 2
               + 0.15 * (1.0 - d / (3.0 * COLL_R)).clamp(min=0) ** 2)
        pen = pen * live.unsqueeze(2).float()
        coll = pen.amax(dim=1).mean(-1)                               # [B,K]
    else:
        coll = torch.zeros(B, K, device=dev)
    parts["coll"] = float(coll.mean())

    # --- comfort: second difference (accel/jerk proxy) -----------------
    if T >= 3:
        d2 = wp[:, :, 2:] - 2 * wp[:, :, 1:-1] + wp[:, :, :-2]
        # Clamped. Unbounded, this term decided the ranking on its own:
        # measured over 200 val frames, the spread between the K candidates was
        # 4.35 for comfort against 1.25 for imit, 0.38 for progress, 0.21 for
        # drive and 0.016 for collision, so 68 % of what separates candidates
        # was jerk. The selector it produced loses to "always take candidate 0"
        # on ADE (1.042 vs 1.012 m), which is what picking the smoothest path
        # rather than the right one looks like. 1.0 here is already a harsh
        # second difference (0.5 m per 0.5 s step); past that a candidate is
        # simply unacceptable and there is nothing to gain by ranking degrees
        # of unacceptable.
        comfort = (d2.norm(dim=-1).mean(-1) / 0.5).clamp(max=1.0)
    else:
        comfort = torch.zeros(B, K, device=dev)
    parts["comfort"] = float(comfort.mean())

    # --- progress (normalised by what the current speed allows) --------
    if v0 is not None:
        horizon = (v0.float().view(B, 1) * (0.5 * T)).clamp(min=2.0)
        prog = (x[:, :, -1] / horizon).clamp(-0.5, 1.2)
    else:
        prog = torch.zeros(B, K, device=dev)
    parts["progress"] = float(prog.mean())

    # --- red-light compliance (vehicle lamps in the front cameras) -----
    if tl is not None:
        t = tl.float()[:, FRONT_CAMS]                     # [B,c,7,h,w]
        veh_red = (t[:, :, 0] * (1.0 - t[:, :, 3])).amax(dim=(1, 2, 3))
        red = (veh_red > 0.5).float().view(B, 1)
    else:
        red = torch.zeros(B, 1, device=dev)
    parts["red_frac"] = float(red.mean())

    # --- imitation anchor: keeps rule-compliant nonsense from winning --
    if ego_gt is not None:
        g = ego_gt.float()
        gwp = g[:, :12].view(B, 1, 6, 2)[:, :, :T]
        err = (wp - gwp).abs()
        imit = ((err[..., 0] + 4.0 * err[..., 1]).mean(-1) / 2.5)
        imit = imit * (g[:, 16:17] > 0.5).float()
    else:
        imit = torch.zeros(B, K, device=dev)
    parts["imit"] = float(imit.mean())

    rew = (w["drive"] * drive
           - w["coll"] * coll
           - w["comfort"] * comfort
           + w["progress"] * prog * (1.0 - red)
           - w["tl"] * red * prog.clamp(min=0)
           - w["imit"] * imit)
    return rew, parts


def grpo_mode_loss(mode_logits, rewards, valid=None, ent_w=0.0):
    """Group-relative policy gradient over the K candidates (no critic).

    advantage_k = (r_k - mean_k r) / std_k r  ->  loss = -E[A * log pi]
    Returns (loss, stats). `valid` [B] gates rows without usable rewards.
    """
    lg = mode_logits.float()
    r = rewards.detach().float()
    adv = (r - r.mean(1, keepdim=True)) / r.std(1, keepdim=True).clamp(min=1e-3)
    logp = F.log_softmax(lg, 1)
    per = -(adv * logp).sum(1)
    if ent_w > 0:
        per = per + ent_w * (logp.exp() * logp).sum(1)      # -entropy
    if valid is None:
        loss = per.mean()
        n = per.numel()
    else:
        v = valid.float().view(-1)
        n = float(v.sum())
        loss = (per * v).sum() / max(n, 1.0)
    pick = lg.argmax(1)
    best = r.argmax(1)
    stats = {
        "rew_sel": float(r.gather(1, pick[:, None]).mean()),
        "rew_best": float(r.max(1).values.mean()),
        "rew_mean": float(r.mean()),
        "pick_acc": float((pick == best).float().mean()),
        "n": n,
    }
    return loss, stats

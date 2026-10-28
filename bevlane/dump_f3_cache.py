#!/usr/bin/env python3
"""F3 cache dump: the training data of the F3 trajectory refinement head (train_f3.py).

Runs a frozen checkpoint (METEOR v52 line or METEOR2, via probe_net.load_full) over scenes and stores,
per scene, consecutive-frame sequences of: pooled fused BEV features (fp16, 64x40, or an integer
AveragePool with F3_POOLK for the NPU head), the baseline 42-d ego output (cand), GT path, v0, causal
acceleration / brake, and the relative 2D pose to the previous frame (for make_warp_theta).

Usage: dump_f3_cache.py <ckpt> <scene_list> <out_dir> [shard] [nshard]
  env: METEOR_BEV_ROOT (default out/bevlane), F3_MPS (frames per scene, default 16), F3_POOLK,
       METEOR_HR_ROOT / METEOR_IMG_SCALE / METEOR_N_CAMS / METEOR_FRONT_MASK (same as the evaluation)
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bevlane.dataset import BevLaneDataset          # noqa: E402
from bevlane.probe_net import load_full             # noqa: E402

ck, lst, outd = sys.argv[1], sys.argv[2], sys.argv[3]
SHARD = int(sys.argv[4]) if len(sys.argv) > 4 else 0
NSHARD = int(sys.argv[5]) if len(sys.argv) > 5 else 1
POOL = (64, 40)
POOLK = int(os.environ.get("F3_POOLK", "0"))        # NPU head: integer AveragePool (k=5: fused BEV 400x250 -> 80x50); 0 = adaptive 64x40
MPS = int(os.environ.get("F3_MPS", "16"))          # frames per scene (consecutive); raise it for long recordings
root = os.environ.get("METEOR_BEV_ROOT", "out/bevlane")
os.makedirs(outd, exist_ok=True)
scenes = [s for i, s in enumerate(open(lst).read().split()) if i % NSHARD == SHARD]
m = load_full(ck).cuda().eval()
done = 0
for si, s in enumerate(scenes):
    op = os.path.join(outd, s + ".npz")
    if os.path.exists(op):
        continue
    try:
        em = np.load(os.path.join(root, s, "ego_motion.npz"), allow_pickle=True)
        pose = np.asarray(em["pose"], np.float64)
        ds = BevLaneDataset(root, [s], gt_key="gt_cons", with_ego=True,
                            n_cams=int(os.environ.get("METEOR_N_CAMS", "8") or 8),            # METEOR2 Nano (1 front camera): same env as the evaluation
                            **({"front_mask": os.environ["METEOR_FRONT_MASK"]} if os.environ.get("METEOR_FRONT_MASK") else {}),
                            max_per_scene=MPS, trim_start=3, trim_end=10,
                            # METEOR2 648p: native-resolution input (same env as the evaluation)
                            hr_root=os.environ.get("METEOR_HR_ROOT") or None,
                            img_scale=float(os.environ.get("METEOR_IMG_SCALE", "1") or 1))
    except Exception as e:
        print("skip", s, e, flush=True)
        continue
    R = {k: [] for k in ("frame", "bev", "cand", "gt", "v0", "rel", "acc", "brk")}   # acc/brk (CAN-equivalent) as optional extra F3 inputs
    for i in range(len(ds)):
        b = ds[i]
        if b is None:
            continue
        _, f = ds.items[i]
        fr = int(f["frame"])
        e = b[4]
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            out = m(b[0][None].cuda(), b[1][None].cuda(), b[2][None].cuda(),
                    e[12][None].cuda())
            fb = m._fused_bev.float()
            bev = (F.avg_pool2d(fb, POOLK) if POOLK else F.adaptive_avg_pool2d(fb, POOL))[0].half().cpu().numpy()
            if done == 0 and not R["frame"]:
                print(f"[pool] fused {tuple(fb.shape)} -> {bev.shape} (POOLK={POOLK})", flush=True)
        # relative pose current->previous frame for make_warp_theta:
        # p_prev = R(dyaw) p_cur + t with dyaw = yaw_t - yaw_p, t = R(-yaw_p)(x_t - x_p)
        if 0 <= fr - 1 < len(pose):
            xt, yt, at = pose[fr]
            xp, yp, ap = pose[fr - 1]
            c, sn = np.cos(-ap), np.sin(-ap)
            tx = c * (xt - xp) - sn * (yt - yp)
            ty = sn * (xt - xp) + c * (yt - yp)
            rel = np.array([tx, ty, at - ap], np.float32)
        else:
            rel = np.zeros(3, np.float32)
        R["frame"].append(fr)
        R["bev"].append(bev)
        R["cand"].append(out[7][0].float().cpu().numpy())
        R["gt"].append(e[:12].numpy())
        R["v0"].append(float(e[12]))
        R["acc"].append(float(e[17])); R["brk"].append(float(e[17] < -0.5))   # inputs use the causal e[17] (e[13]/e[15] look ahead and are targets only)
        R["rel"].append(rel)
    if len(R["frame"]) >= 4:
        np.savez_compressed(op, **{k: np.asarray(v) for k, v in R.items()})
        done += 1
    if si % 20 == 0:
        print(f"[{SHARD}] {si}/{len(scenes)} saved={done}", flush=True)
print(f"[{SHARD}] DONE scenes={len(scenes)} saved={done}", flush=True)

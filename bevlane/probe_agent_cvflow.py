#!/usr/bin/env python3
"""CV-from-flow probe (2026-08-27): upper bound of the "CV + residual" lever without training.

Follow-up to probe_agent_cv.py. The model's (v125) agent trajectory error of 2.14 m
loses badly to oracle-CV at 0.8 m. So how good is a realistic CV built from the
flow head's velocity estimate? If good, it sets the expected value for a round
that reparameterizes the traj head as CV(flow) + residual (v128 candidate).
"""
import argparse
import math
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, ".")
from bevlane.dataset import BevLaneDataset                      # noqa: E402
from bevlane.model import DepthSegIPMNetV52                     # noqa: E402
from bevlane.ckpt_load import load_net                          # noqa: E402
from bevlane.train import _temporal_inputs                      # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="out/v125_final_best_e2e.pt")
ap.add_argument("--root", default="out/bevlane")
ap.add_argument("--val-list", default="val.lst")
ap.add_argument("--scenes", type=int, default=24)
ap.add_argument("--max-per-scene", type=int, default=5)
a = ap.parse_args()

dev = "cuda"
sc = [l.strip() for l in open(a.val_list) if l.strip()][:a.scenes]
ds = BevLaneDataset(a.root, sc, gt_key="gt_cons",
                    with_boxdet=True, with_agenttraj=True,
                    with_temporal=True, max_per_scene=a.max_per_scene)
dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)
net = DepthSegIPMNetV52(n_seg=21).to(dev).eval()
load_net(net, a.ckpt)

S = {}


def add(k, v):
    S.setdefault(k, []).append(float(v))


steps = torch.arange(1, 7, device=dev).view(6, 1).float()
n_ag = 0
with torch.no_grad():
    for batch in dl:
        imgs = batch[0].to(dev)
        K = batch[1].to(dev)
        Tc = batch[2].to(dev)
        # agenttraj quadruple (boxes[64,6], count, traj[64,6,2], tvalid[64,6])
        tj = bx = None
        tensors = [t for t in batch if torch.is_tensor(t)]
        for j in range(len(tensors) - 3):
            t2 = tensors[j + 2]
            if t2.dim() == 4 and tuple(t2.shape[2:]) == (6, 2) and \
                    tensors[j].dim() == 3 and tensors[j].shape[1] == t2.shape[1]:
                bx, nbT, tj, tv = (tensors[j][0], tensors[j + 1],
                                   tensors[j + 2][0], tensors[j + 3][0])
                break
        if tj is None:
            continue
        pb, th = _temporal_inputs(net, batch, dev, None)
        with torch.autocast("cuda", torch.float16):
            out = net(imgs, K, Tc, None, pb, th)
        tp = out[9][0].float()                       # [39,H,W] det grid
        # eager output order: ...traj9 stationary10 tl11 risk12 flow13.
        # "first tensor with C==2" would wrongly grab hm (2-ch heatmap).
        flow = out[13]
        assert torch.is_tensor(flow) and flow.dim() == 4 \
            and flow.shape[1] == 2, f"flow index mismatch: {flow.shape}"
        flow = flow[0].float()                       # [2,FH,FW] +-40m window
        Hh, Ww = tp.shape[-2:]
        FH, FW = flow.shape[-2:]
        for k in range(max(0, int(nbT))):
            if float(bx[k, 3]) <= 0:
                continue
            v = tv[k]
            if v.sum() == 0:
                continue
            x, y = float(bx[k, 1]), float(bx[k, 2])
            ri = int((80.0 - x) / 0.4)
            ci = int((50.0 - y) / 0.4)
            if not (0 <= ri < Hh and 0 <= ci < Ww):
                continue
            g = tj[k].to(dev)
            vv = v.to(dev)
            cls = "veh" if float(bx[k, 0]) < 1.5 else "vru"
            # --- model prediction (winning mode)
            vec = tp[:, ri, ci]
            kb = int(vec[36:39].argmax())
            p = vec[kb * 12:(kb + 1) * 12].view(6, 2)
            add(f"model_{cls}", ((p - g).norm(dim=1) * vv).sum() / vv.sum())
            # --- CV(flow): flow window +-40m
            rf = int((40.0 - x) / (80.0 / FH))
            cf = int((40.0 - y) / (80.0 / FW))
            if 0 <= rf < FH and 0 <= cf < FW:
                vel = flow[:, rf, cf]                # displacement per 0.5s
                pcv = steps * vel.view(1, 2)
                add(f"cvflow_{cls}",
                    ((pcv - g).norm(dim=1) * vv).sum() / vv.sum())
                if vv[5] > 0 and float(g[5].norm()) > 1.0:
                    if float(pcv[5].norm()) > 0.3:
                        ga = math.atan2(float(g[5, 1]), float(g[5, 0]))
                        pa = math.atan2(float(pcv[5, 1]), float(pcv[5, 0]))
                        add(f"cvflow_head_{cls}", math.degrees(
                            abs((pa - ga + math.pi) % (2 * math.pi) - math.pi)))
                    # error of the velocity estimate itself (0.5s displacement)
                    add(f"flowerr_{cls}", float((vel - g[0].to(dev)).norm()))
            n_ag += 1

print(f"agents={n_ag} scenes={len(sc)} ckpt={a.ckpt}")
for k in sorted(S):
    vv = np.array(S[k])
    print(f"  {k:<18} mean={vv.mean():.3f}  n={len(vv)}")

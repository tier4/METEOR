"""Paired test of whether temporal memory (history) pushes the E2E trajectory to the right.

Orin measurement: with history -1.15 m / without -0.29 m (r64 weights, same frames).
If torch reproduces this, the cause is not the engine but the history-warp
convention itself (make_warp_theta / rel_pose).

Method: run the same scene in time order and at each frame infer
  (A) with real history (raw_bev at t-0.4/1.2/2.8 s + ego warp)
  (B) with zero history
The mean difference in selected-mode trajectory y (points 2/4/6) = the push from history.
Same images and weights for both, so the difference is the history path alone.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bevlane.dataset import BevLaneDataset                        # noqa: E402
from bevlane.model import (MODELS, EGO_K, BEV_H, BEV_W,           # noqa: E402
                           make_warp_theta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", default="v52")
    ap.add_argument("--list", default="val.lst")
    ap.add_argument("--root", default="out/bevlane")
    ap.add_argument("--scenes", type=int, default=12)
    ap.add_argument("--offs", default="2,6,14",
                    help="history offsets (frames). Orin path_orin runs at stride 2, "
                         "so this tests the hypothesis that the effective offsets are 4,12,28")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    scenes = [l.strip() for l in open(a.list) if l.strip()][:a.scenes]
    ds = BevLaneDataset(a.root, scenes, gt_key="gt_cons", with_ego=True,
                        max_per_scene=200, n_cams=8, trim_start=3,
                        trim_end=10)
    m = MODELS[a.model](n_seg=21).cuda().eval()
    sd = torch.load(a.ckpt, map_location="cpu")
    sd = {k.replace("module.", ""): v for k, v in sd.get("model", sd).items()}
    cur = m.state_dict()
    m.load_state_dict({k: v for k, v in sd.items()
                       if k in cur and cur[k].shape == v.shape}, strict=False)

    by_scene = {}
    for i, (s, f) in enumerate(ds.items):
        by_scene.setdefault(s, []).append((int(f["frame"]), i))

    HOR = [1, 3, 5]
    ya, yb, dgt = [], [], []          # y with history / y without / GT y
    n_hist = 0
    for s, lst in by_scene.items():
        lst.sort()
        pose = None
        p = os.path.join(a.root, s, "ego_motion.npz")
        if os.path.exists(p):
            try:
                pose = np.load(p)["pose"]
            except Exception:
                pose = None
        bevq = {}
        for fi, di in lst:
            b = ds[di]
            if b is None:
                continue
            eg = b[4] if len(b) > 4 and torch.is_tensor(b[4]) else None
            ims = b[0][None].cuda()
            Kk = b[1][None].cuda()
            Tc = b[2][None].cuda()

            def rel(fj):
                if pose is None or fj < 0 or fj >= len(pose):
                    return None
                pc_, pp_ = pose[fi], pose[fj]
                if abs(pc_).sum() == 0 or abs(pp_).sum() == 0:
                    return None
                cp, sp = np.cos(pp_[2]), np.sin(pp_[2])
                dx, dy = pc_[0] - pp_[0], pc_[1] - pp_[1]
                return torch.tensor([[cp * dx + sp * dy,
                                      -sp * dx + cp * dy,
                                      float(pc_[2] - pp_[2])]],
                                    dtype=torch.float32).cuda()

            v0t = torch.tensor([float(eg[12])] if eg is not None
                               else [0.0]).cuda()
            pbs, ths, got = [], [], 0
            for off in tuple(int(x) for x in a.offs.split(",")):
                hb, rl = bevq.get(fi - off), rel(fi - off)
                if hb is None or rl is None:
                    pbs.append(torch.zeros(1, 96, BEV_H, BEV_W,
                                           device="cuda"))
                    ths.append(make_warp_theta(
                        torch.zeros(1, 3, device="cuda")))
                else:
                    pbs.append(hb)
                    ths.append(make_warp_theta(rl))
                    got += 1
            pb, th = torch.stack(pbs, 1), torch.stack(ths, 1)
            z_pb = torch.zeros_like(pb)
            z_th = torch.stack([make_warp_theta(
                torch.zeros(1, 3, device="cuda"))] * 3, 1)

            with torch.no_grad(), torch.autocast("cuda", torch.float16):
                oa = m(ims, Kk, Tc, v0=v0t, prev_bev=pb, warp_theta=th)
                bevq[fi] = m._last_bev.detach().float()
                ob = m(ims, Kk, Tc, v0=v0t, prev_bev=z_pb, warp_theta=z_th)
            for kk in [k for k in bevq if k != "scene"
                       and isinstance(k, int) and k < fi - 28]:
                bevq.pop(kk)
            if got == 0 or eg is None or float(eg[16]) == 0:
                continue

            def sel_y(o):
                e = o[7].float()[0]
                wp = e[:12 * EGO_K].view(EGO_K, 6, 2)
                k = int(e[12 * EGO_K:12 * EGO_K + EGO_K].argmax())
                return [float(wp[k, h, 1]) for h in HOR]
            ya.append(sel_y(oa))
            yb.append(sel_y(ob))
            dgt.append([float(eg[:12].view(6, 2)[h, 1]) for h in HOR])
            n_hist += 1

    ya, yb, dgt = map(np.array, (ya, yb, dgt))
    print(f"\n=== {a.tag or a.ckpt} history paired comparison ({n_hist} frames) ===")
    print("(+y = left / -y = right)")
    for j, h in enumerate(HOR):
        d = ya[:, j] - yb[:, j]
        print(f"[pt{h+1}/6] with hist {ya[:, j].mean():+.3f}"
              f" | no hist {yb[:, j].mean():+.3f}"
              f" | diff(with-no) {d.mean():+.3f}±{d.std():.3f}"
              f" | GT {dgt[:, j].mean():+.3f}")
    print("PROBE_HIST_BIAS_DONE")


if __name__ == "__main__":
    main()

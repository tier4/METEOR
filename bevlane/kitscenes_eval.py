#!/usr/bin/env python3
"""KITScenes trajectory evaluation of METEOR2 checkpoints: ADE@1s / ADE@2s / ADE@3s / FDE@3s.

Evaluation sets (on KITScenes val, scenes converted by tools/kitscenes/kitscenes_to_bevlane.py):
  kf202 : per val scene, 10 Hz keyframes a = 40 + 90 k with a + 30 <= U - 1 (4 s of past and 3 s of future
          available, U = number of 10 Hz poses) -> 202 keyframes on the official val split
  dense : every frame a = 40 .. U - 51. The converted images are the even 10 Hz frames only (stride 2), so only
          even a are evaluated (~5.5k frames)
Ground truth: from the raw 10 Hz poses (poses10.txt, TUM format), pos[a+1 : a+31] - pos[a] rotated into the ego frame
  (x forward, y left) by the yaw of pose a -> 30 points at 0.1 s.
Prediction: the selected candidate of the METEOR2 ego head (6 points at 0.5 s), linearly interpolated from (0, 0) at
  0.1 s -> 30 points. ADE@1/2/3s = mean L2 error over the first 10 / 20 / 30 points, FDE@3s = error of point 30.
Inputs follow the checkpoint: a plain checkpoint sees images + ego speed only; a --route-map checkpoint additionally gets
  the ego history and the HD-map / route raster that the dataset appends (this script sets METEOR_RMAP=1). The
  conditioning switches must match training, e.g. METEOR_RMAP_ROUTE=lanelet METEOR_RMAP_HIST=past64 METEOR_ABSENT_CAMS=6
  (and METEOR_RMAP_NOMAP=1 for the no-HD-map variant); see docs/KITSCENES.md.

usage: python bevlane/kitscenes_eval.py NAME=CKPT [NAME=CKPT ...] --val-list out/kitscenes/scenes_val.txt
           [--root out/bevlane] [--hr-root <hr720 root>] [--img-scale 1.5] [--set kf202|dense|both] [--dump DIR]
"""
import argparse
import os
import sys

os.environ.setdefault("METEOR_EVAL_MODEL", "v52")
os.environ.setdefault("METEOR_ZERO_HIST", "1")
os.environ["METEOR_RMAP"] = "1"                 # the dataset appends the conditioning inputs (ignored by plain checkpoints)
# The "trace" route draws the recorded future trajectory itself; evaluation only allows the lane-level route.
_route = os.environ.get("METEOR_RMAP_ROUTE") or os.environ.get("METEOR_AE2E_ROUTE") or "lanelet"
if _route != "lanelet":
    sys.exit(f"kitscenes_eval: METEOR_RMAP_ROUTE={_route} is not allowed (it encodes the future trajectory); use lanelet")
os.environ["METEOR_RMAP_ROUTE"] = "lanelet"

import numpy as np   # noqa: E402
import torch         # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bevlane import meteorite as M              # noqa: E402
from bevlane.dataset import BevLaneDataset      # noqa: E402


def load_poses(root, s):
    P = np.loadtxt(os.path.join(root, s, "poses10.txt"))
    qx, qy, qz, qw = P[:, 4], P[:, 5], P[:, 6], P[:, 7]
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy ** 2 + qz ** 2))
    return P[:, 1:3], yaw


def gt_future(pos, yaw, a, n=30):
    d = pos[a + 1:a + 1 + n] - pos[a]; c, s_ = np.cos(yaw[a]), np.sin(yaw[a])
    return np.stack([d[:, 0] * c + d[:, 1] * s_, -d[:, 0] * s_ + d[:, 1] * c], 1)


def interp10(wp6):
    t = np.arange(0, 3.01, 0.5); P = np.vstack([[0, 0], wp6]); tt = np.arange(1, 31) * 0.1
    return np.stack([np.interp(tt, t, P[:, 0]), np.interp(tt, t, P[:, 1])], 1)


def anchors_of(root, scenes):
    out = {"kf202": [], "dense": []}
    for s in scenes:
        U = len(np.loadtxt(os.path.join(root, s, "poses10.txt"), usecols=0))
        a = 40
        while a + 30 <= U - 1:
            out["kf202"].append((s, a)); a += 90
        for a in range(40, U - 50):
            if a % 2 == 0:
                out["dense"].append((s, a))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpts", nargs="+", help="NAME=CKPT pairs")
    ap.add_argument("--root", default=os.environ.get("METEOR_BEV_ROOT", "out/bevlane"))
    ap.add_argument("--hr-root", default=os.environ.get("METEOR_HR_ROOT", ""), help="native-resolution image root (as in training)")
    ap.add_argument("--img-scale", type=float, default=float(os.environ.get("METEOR_IMG_SCALE", "1.5")))
    ap.add_argument("--val-list", default=os.environ.get("METEOR_VAL_LIST", "out/kitscenes/scenes_val.txt"),
                    help="KITScenes val scenes (KITS_<uuid>, one per line)")
    ap.add_argument("--set", default="kf202", help="kf202 | dense | both (comma-separated)")
    ap.add_argument("--dump", default=os.environ.get("KITS_EVAL_DUMP", ""), help="save per-keyframe errors (30 points) to DIR/<name>__<set>.npz")
    ap.add_argument("--zero-ch", default="", help="ablation: zero these raster channels (e.g. 8,9 = no route / destination)")
    ap.add_argument("--lidar", action="store_true", help="also feed the LiDAR pillar raster (lidar_bev/) as the optional BEV input")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    sets = ["kf202", "dense"] if a.set == "both" else a.set.split(",")
    zero_ch = [int(c) for c in a.zero_ch.split(",") if c]

    scenes = [l.strip() for l in open(a.val_list) if l.strip()]
    anchors = anchors_of(a.root, scenes)
    print({k: len(v) for k, v in anchors.items() if k in sets}, "(dense: even 10 Hz frames only)", flush=True)
    ds = BevLaneDataset(a.root, scenes, gt_key="gt", gt_optional=True, with_ego=True, with_boxdet=True, n_cams=M.N_CAM,
                        hr_root=a.hr_root or None, img_scale=a.img_scale, with_lidarbev=a.lidar)
    byk = {(s, f["frame"]): i for i, (s, f) in enumerate(ds.items)}
    poses = {s: load_poses(a.root, s) for s in scenes}
    K = getattr(M, "EGO_K", 3)
    for spec in a.ckpts:
        name, ck = spec.split("=", 1)
        w = M.load_mite_eval(ck, device=a.device); net = w.net.eval()
        cond = getattr(net, "route_map", False)
        for st in sets:
            E, miss, keys, route = [], 0, [], []
            for s, fa in anchors[st]:
                i = byk.get((s, fa // 2))
                b = ds[i] if i is not None else None
                if b is None:
                    miss += 1; continue
                ego = b[6]
                if cond:
                    net._rm_hist = b[-2][None].to(a.device).float(); net._rm_ras = b[-1][None].to(a.device).float()
                    for ch in zero_ch:
                        net._rm_ras[:, ch] = 0
                lid = None
                if a.lidar:                                   # the raster is the only [4, 400, 250] item of the sample
                    lid = next((t[None].to(a.device).float() for t in b if torch.is_tensor(t) and tuple(t.shape) == (4, 400, 250)), None)
                with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                    o = net.forward_geo(b[0][None].to(a.device), b[1][None].to(a.device), b[2][None].to(a.device),
                                        ego[12].view(1).to(a.device).float(), lidar=lid)
                e = o["ego"].float()[0].cpu().numpy()
                wps = e[:12 * K].reshape(K, 6, 2); k = int(e[12 * K:13 * K].argmax())
                pos, yaw = poses[s]
                E.append(np.linalg.norm(interp10(wps[k]) - gt_future(pos, yaw, fa), axis=1))
                keys.append(f"{s}:{fa}"); route.append(bool(b[-1][8].sum() > 0))
            E = np.array(E)
            if a.dump:
                os.makedirs(a.dump, exist_ok=True)
                np.savez(os.path.join(a.dump, f"{name}__{st}.npz"), err=E.astype(np.float32), key=np.array(keys), route=np.array(route))
            print(f"KITS name={name} set={st} n={len(E)} miss={miss} ADE@1s={E[:, :10].mean():.3f} ADE@2s={E[:, :20].mean():.3f} "
                  f"ADE@3s={E[:, :30].mean():.3f} FDE@3s={E[:, 29].mean():.3f} (6-point ADE@3s {E[:, 4::5].mean():.3f})", flush=True)
        del w, net
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

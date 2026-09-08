"""Formal A/B toggling LiDAR on/off in one process on the same frames:
  (1) GUARD VETO (STATIC obstacle) rate and truthfulness
  (2) 3D BBox accuracy (recall per band + signed error)
Follows the optional-input A/B rule (same process, same frames, only the kwarg toggled).
"""
import argparse, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bevlane.dataset import BevLaneDataset
from bevlane.model import MODELS, EGO_K
from bevlane.guardrail import check_path, _occ_ground
from bevlane.ckpt_load import load_net

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--list", required=True)
ap.add_argument("--root", default="out/bevlane")
ap.add_argument("--frames", type=int, default=200)
ap.add_argument("--tag", default="")
a = ap.parse_args()

scenes = [l.strip() for l in open(a.list) if l.strip()]
ds = BevLaneDataset(a.root, scenes, gt_key="gt_cons", with_boxdet=True,
                    with_ego=True, with_occ=True, with_lidarbev=True,
                    max_per_scene=8, n_cams=8, trim_start=3, trim_end=10)
m = MODELS["v52"](n_seg=21).cuda().eval()
# paint-seg / paint-det branches do not exist until enable_* is called; a plain
# strict=False load silently dropped them (found 2026-08-22). Always grow them via the shared loader.
if os.environ.get("METEOR_NO_PAINT"):
    print("[ab] METEOR_NO_PAINT=1: measuring with paint branches dropped (reproduces the old probe)")
    _sd = torch.load(a.ckpt, map_location="cpu")
    _sd = {k.replace("module.", ""): v for k, v in _sd.get("model", _sd).items()}
    _cur = m.state_dict()
    m.load_state_dict({k: v for k, v in _sd.items()
                       if k in _cur and _cur[k].shape == v.shape}, strict=False)
else:
    load_net(m, a.ckpt)

# output order: imgs,K,Tc,gt, boxes,n, ego, occ, lidar_bev (assumed verified against flag order)
BANDS = [(0, 20), (20, 40), (40, 60)]
res = {}
for mode in ("cam", "lidar"):
    res[mode] = dict(veto=0, veto_static=0, n=0,
                     gt_n={b: 0 for b in BANDS}, hit={b: 0 for b in BANDS},
                     dy=[], phantom=0)
step = max(1, len(ds) // a.frames)
done = 0
for i in range(0, len(ds), step):
    b = ds[i]
    if b is None: continue
    bx, nb, eg, occ_gt = b[4], int(b[5]), b[6], b[7]
    lb = b[8] if len(b) > 8 else None
    v0t = torch.tensor([float(eg[12]) if torch.is_tensor(eg) else 8.0]).cuda()
    for mode in ("cam", "lidar"):
        kw = {}
        if mode == "lidar" and torch.is_tensor(lb):
            kw["lidar_bev"] = lb[None].cuda().float()
        with torch.no_grad(), torch.autocast("cuda", torch.float16):
            out = m(b[0][None].cuda(), b[1][None].cuda(), b[2][None].cuda(),
                    v0=v0t, **kw)
        dets = m.decode_boxes(out[3].float().cpu(), out[4].float().cpu(),
                              thresh=0.25)[0]
        occ_pred = out[8].float().argmax(1)[0].cpu().numpy().astype(np.uint8)
        lane_am = out[0].float().argmax(1)[0].cpu().numpy().astype(np.uint8)
        e = out[7].float()[0]
        wp = e[:12 * EGO_K].view(EGO_K, 6, 2)
        k = int(e[12 * EGO_K:12 * EGO_K + EGO_K].argmax())
        path = wp[k].cpu().numpy()
        d_in = [(int(d[0]), float(d[1]), *[float(v) for v in d[2:7]])
                for d in dets]
        offs = [np.zeros((6, 2), np.float32)] * len(d_in)
        tlp = np.array([1.0, 0, 0, 0], np.float32)
        g = check_path(path, occ_pred, d_in, offs, tlp, lane_am,
                       float(v0t.item()))
        R = res[mode]; R["n"] += 1
        if g["verdict"] == "VETO":
            R["veto"] += 1
            if "STATIC" in g["reason"]:
                R["veto_static"] += 1
                # truth check: is there a static object in the 3x3 at the same coords of GT occ (255=ignore)
                px, py = g["p_event"]
                og = np.where(occ_gt.numpy() == 255, 0, occ_gt.numpy())
                blk_gt = np.isin(og, (1, 8)).any(0)
                r0, c0 = int((40 - px) / 0.4), int((40 - py) / 0.4)
                if 2 <= r0 < 198 and 2 <= c0 < 198 \
                        and blk_gt[r0-1:r0+2, c0-1:c0+2].sum() < 1:
                    R["phantom"] += 1
        # 3D BBox recall + signed lateral error (vehicles)
        pv = [(x, y) for (c, s, x, y, l, w, yw) in d_in if c == 0]
        for kk in range(max(nb, 0)):
            cls, xe, ye, ln = [float(v) for v in bx[kk][:4]]
            if ln <= 0 or cls >= 1.5: continue
            r_ = (xe*xe + ye*ye) ** 0.5
            best = None
            for px2, py2 in pv:
                d2 = (xe-px2)**2 + (ye-py2)**2
                if d2 < 9.0 and (best is None or d2 < best[0]):
                    best = (d2, px2, py2)
            for lo, hi in BANDS:
                if lo <= r_ < hi:
                    R["gt_n"][(lo,hi)] += 1
                    if best: R["hit"][(lo,hi)] += 1
            if best: R["dy"].append(best[2] - ye)
    done += 1
    if done >= a.frames: break

print(f"\n=== {a.tag} LiDAR A/B ({done} frames, same frames) ===")
for mode in ("cam", "lidar"):
    R = res[mode]
    rec = " ".join(f"{lo}-{hi}m R={R['hit'][(lo,hi)]/max(R['gt_n'][(lo,hi)],1):.3f}"
                   for lo, hi in BANDS)
    dy = np.array(R["dy"])
    print(f"[{mode:5}] VETO {R['veto']}/{R['n']} (STATIC {R['veto_static']}, "
          f"phantom {R['phantom']}) | veh {rec} | lat err {dy.mean():+.3f}±{dy.std():.3f}")
print("GUARD_AB_DONE")

"""Detection recall per range band, front and rear separately (rear not cut at 20 m)."""
import argparse, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bevlane.dataset import BevLaneDataset
from bevlane.model import MODELS

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--model", default="v52")
ap.add_argument("--n-cams", type=int, default=8); ap.add_argument("--list", default="val.lst")
ap.add_argument("--root", default="out/bevlane"); ap.add_argument("--frames", type=int, default=150)
ap.add_argument("--zero-cams", default=""); ap.add_argument("--tag", default="")
a = ap.parse_args()
from bevlane.dataset import CAMS as DSC
zc = [DSC.index(c) for c in a.zero_cams.split(",") if c] if a.zero_cams else []
scenes = [l.strip() for l in open(a.list) if l.strip()][:60]
ds = BevLaneDataset(a.root, scenes, gt_key="gt_cons", with_boxdet=True,
                    n_cams=8, trim_start=25, trim_end=10)
m = MODELS[a.model](n_seg=21).cuda().eval()
sd = torch.load(a.ckpt, map_location="cpu")
sd = {k.replace("module.", ""): v for k, v in sd.get("model", sd).items()}
cur = m.state_dict()
m.load_state_dict({k: v for k, v in sd.items() if k in cur and cur[k].shape == v.shape}, strict=False)
TH = 0.25
BANDS = [(0,20),(20,40),(40,60),(60,80)]
gt_n = {("front",b):0 for b in BANDS} | {("rear",b):0 for b in BANDS}
hit = dict(gt_n)
step = max(1, len(ds)//a.frames); done = 0
for i in range(0, len(ds), step):
    b = ds[i]
    if b is None: continue
    ims = b[0][None][:, :a.n_cams].clone()
    for ci in zc:
        if ci < ims.shape[1]: ims[:, ci] = 0
    with torch.no_grad(), torch.autocast("cuda", torch.float16):
        out = m(ims.cuda(), b[1][None][:, :a.n_cams].cuda(), b[2][None][:, :a.n_cams].cuda())
    dets = m.decode_boxes(out[3].float().cpu(), out[4].float().cpu(), thresh=TH)[0]
    pred = [(float(d[2]), float(d[3])) for d in dets if int(d[0]) == 0]
    bx, nb = b[4], int(b[5]); used = set()
    for k in range(max(nb,0)):
        cls, xe, ye, ln, wd, yw = [float(v) for v in bx[k][:6]]
        if ln <= 0 or cls >= 1.5 or abs(ye) > 50: continue
        r = (xe*xe + ye*ye) ** 0.5
        side = "front" if xe > 0 else "rear"
        band = next((bb for bb in BANDS if bb[0] <= r < bb[1]), None)
        if band is None: continue
        gt_n[(side,band)] += 1
        best = None
        for j,(dx,dy) in enumerate(pred):
            if j in used: continue
            d2 = (xe-dx)**2 + (ye-dy)**2
            if d2 < 4.0 and (best is None or d2 < best[0]): best = (d2, j)
        if best is not None:
            used.add(best[1]); hit[(side,band)] += 1
    done += 1
    if done >= a.frames: break
print(f"\n=== {a.tag or a.ckpt} ({done} frames, thr {TH}) ===")
print("band      " + "  ".join(f"{b[0]:2d}-{b[1]:2d}m" for b in BANDS))
for side in ("front","rear"):
    print(f"  {side} recall " + "  ".join(f"{hit[(side,b)]/max(gt_n[(side,b)],1):6.3f}" for b in BANDS))
    print(f"       GT boxes " + "  ".join(f"{gt_n[(side,b)]:6d}" for b in BANDS))

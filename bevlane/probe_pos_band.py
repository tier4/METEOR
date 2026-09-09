"""3D BBox position error per front/rear range band (for cross-generation comparison)."""
import argparse, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bevlane.dataset import BevLaneDataset
from bevlane.model import MODELS

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--model", default="v52")
ap.add_argument("--n-cams", type=int, default=8)
ap.add_argument("--list", default="val.lst")
ap.add_argument("--root", default="out/bevlane")
ap.add_argument("--frames", type=int, default=150)
ap.add_argument("--thresh", type=float, default=0.25)
ap.add_argument("--tag", default="")
a = ap.parse_args()

scenes = [l.strip() for l in open(a.list) if l.strip()][:60]
ds = BevLaneDataset(a.root, scenes, gt_key="gt_cons", with_boxdet=True,
                    max_per_scene=4, n_cams=8, trim_start=3, trim_end=10)
m = MODELS[a.model](n_seg=21).cuda().eval()
sd = torch.load(a.ckpt, map_location="cpu")
sd = {k.replace("module.", ""): v for k, v in sd.get("model", sd).items()}
cur = m.state_dict()
m.load_state_dict({k: v for k, v in sd.items()
                   if k in cur and cur[k].shape == v.shape}, strict=False)

BANDS = [("front 0-20m", 0, 20, 1), ("front 20-40m", 20, 40, 1), ("front 40-80m", 40, 80, 1),
         ("rear 0-20m", 0, 20, -1), ("rear 20-40m", 20, 40, -1)]
err = {b[0]: [] for b in BANDS}
dx_e = {b[0]: [] for b in BANDS}
step = max(1, len(ds) // a.frames); done = 0
for i in range(0, len(ds), step):
    b = ds[i]
    if b is None: continue
    with torch.no_grad(), torch.autocast("cuda", torch.float16):
        out = m(b[0][None][:, :a.n_cams].cuda(), b[1][None][:, :a.n_cams].cuda(),
                b[2][None][:, :a.n_cams].cuda())
    dets = m.decode_boxes(out[3].float().cpu(), out[4].float().cpu(), thresh=a.thresh)[0]
    pred = [(float(d[2]), float(d[3])) for d in dets if int(d[0]) == 0]
    bx, nb = b[4], int(b[5])
    for k in range(max(nb, 0)):
        cls, xe, ye, ln, wd, yw = [float(v) for v in bx[k][:6]]
        if ln <= 0 or cls >= 1.5: continue
        r = (xe * xe + ye * ye) ** 0.5
        best = None
        for dx, dy in pred:
            d2 = (xe - dx) ** 2 + (ye - dy) ** 2
            if d2 < 9.0 and (best is None or d2 < best[0]): best = (d2, dx, dy)
        if best is None: continue
        for nm, lo, hi, sgn in BANDS:
            if lo <= r < hi and (xe > 0) == (sgn > 0):
                err[nm].append(best[0] ** 0.5)
                dx_e[nm].append(abs(best[1] - xe))    # longitudinal (range) error
    done += 1
    if done >= a.frames: break

print(f"=== {a.tag or a.ckpt} ({done} frames) ===")
print("band        n    median pos err  median range err")
for nm, *_ in BANDS:
    e, d = err[nm], dx_e[nm]
    if len(e) >= 3:
        print(f"  {nm:<10} {len(e):4d}   {np.median(e):5.2f} m      {np.median(d):5.2f} m")

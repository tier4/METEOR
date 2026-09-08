"""For rear GT boxes at 20-40 m, measure distance and direction to the nearest prediction.

Tests the depth-bias hypothesis: if boxes are detected but systematically shifted
longitudinally, that explains both the recall drop (missing the 2 m match) and the
matched-set error plateau at 0.7 m (survivor bias) at once.
"""
import argparse, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bevlane.dataset import BevLaneDataset
from bevlane.model import MODELS

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--model", default="v52")
ap.add_argument("--frames", type=int, default=150)
a = ap.parse_args()
scenes = [l.strip() for l in open("val.lst") if l.strip()][:60]
ds = BevLaneDataset("out/bevlane", scenes, gt_key="gt_cons", with_boxdet=True,
                    n_cams=8, trim_start=25, trim_end=10)
m = MODELS[a.model](n_seg=21).cuda().eval()
sd = torch.load(a.ckpt, map_location="cpu")
sd = {k.replace("module.", ""): v for k, v in sd.get("model", sd).items()}
cur = m.state_dict()
m.load_state_dict({k: v for k, v in sd.items() if k in cur and cur[k].shape == v.shape}, strict=False)

rows = {"front": [], "rear": []}
step = max(1, len(ds)//a.frames); done = 0
for i in range(0, len(ds), step):
    b = ds[i]
    if b is None: continue
    with torch.no_grad(), torch.autocast("cuda", torch.float16):
        out = m(b[0][None].cuda(), b[1][None].cuda(), b[2][None].cuda())
    # include weak detections (threshold lowered to 0.10 to separate shift from absence)
    dets = m.decode_boxes(out[3].float().cpu(), out[4].float().cpu(), thresh=0.10)[0]
    pred = [(float(d[2]), float(d[3])) for d in dets if int(d[0]) == 0]
    bx, nb = b[4], int(b[5])
    for k in range(max(nb, 0)):
        cls, xe, ye, ln, wd, yw = [float(v) for v in bx[k][:6]]
        if ln <= 0 or cls >= 1.5 or abs(ye) > 45: continue
        r = (xe*xe + ye*ye) ** 0.5
        if not (20 <= r < 40): continue
        best = None
        for dx, dy in pred:
            d2 = (xe-dx)**2 + (ye-dy)**2
            if best is None or d2 < best[0]: best = (d2, dx, dy)
        side = "front" if xe > 0 else "rear"
        if best is None:
            rows[side].append((99.0, 0.0)); continue
        dist = best[0] ** 0.5
        # signed radial (range) shift: + = prediction farther than GT
        ux, uy = xe / max(r, 1e-6), ye / max(r, 1e-6)
        dr = (best[1]-xe)*ux + (best[2]-ye)*uy
        rows[side].append((dist, dr))
    done += 1
    if done >= a.frames: break

print(f"=== 20-40m band: GT box to nearest prediction (thr 0.10, {done} frames) ===")
for side in ("front", "rear"):
    A = np.array(rows[side])
    d, dr = A[:,0], A[:,1]
    print(f"\n{side} (GT {len(A)} boxes)")
    for lo, hi, nm in ((0,1,"0-1m"),(1,2,"1-2m"),(2,3,"2-3m"),(3,5,"3-5m"),(5,98,">5m"),(98,100,"none")):
        s = (d>=lo)&(d<hi)
        print(f"  nearest {nm:<6}: {100*s.mean():5.1f}%")
    near = d < 5
    if near.sum() >= 5:
        print(f"  range shift of boxes within 5m: median {np.median(dr[near]):+.2f}m "
              f"(+=predicted farther) / farther {100*(dr[near]>1).mean():.0f}% nearer {100*(dr[near]<-1).mean():.0f}%")

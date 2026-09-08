#!/usr/bin/env python3
"""Rendering (compose_frame) bottleneck measurement (2026-09-05). Infer N frames -> cProfile compose_frame.
Usage: python3 deploy/profile_render.py eng/xxx.engine [root] [n]"""
import sys, os, json, time, cProfile, pstats, io, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deploy.runtime import MeteorRT
import deploy.orin_render as R
eng = sys.argv[1]; root = sys.argv[2] if len(sys.argv) > 2 else "mixed"; N = int(sys.argv[3]) if len(sys.argv) > 3 else 8
CAMS = ["CAM_FRONT_WIDE", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK_WIDE", "CAM_BACK_LEFT", "CAM_BACK_RIGHT", "CAM_FRONT_NARROW", "CAM_BACK_NARROW"]
R.CAMS = CAMS; R.CAM_DRAW = list(CAMS)
rt = MeteorRT(eng, skip_outputs=("flow", "unk", "pl", "tl", "lg_pts", "lg_meta", "lg_adj"), n_out_slots=2)
if rt.shapes.get("lane") is not None: R.set_bev_extent(rt.shapes["lane"][-2])
scenes = sorted(s for s in os.listdir(root) if os.path.isfile(os.path.join(root, s, "manifest.json")))
s = scenes[0]; d = os.path.join(root, s); m = json.load(open(os.path.join(d, "manifest.json")))
K = np.stack([np.array(m["cams"][c]["K"], np.float32) for c in CAMS])[None]
Tc = np.stack([np.linalg.inv(np.array(m["cams"][c]["T_ego_cam"], np.float32)) for c in CAMS])[None]
z = np.load(os.path.join(d, "ego_motion.npz")); v0s = z["v0"]; poses = z["pose"] if "pose" in z else None
frames = m["frames"][10:10 + N]
outs = []; t_inf = []
for f in frames:
    raw = {c: cv2.imread(os.path.join(d, f["imgs"][c])) for c in CAMS}
    imgs = np.ascontiguousarray(np.stack([raw[c][:, :, ::-1].transpose(2, 0, 1) for c in CAMS])[None])
    fi = int(f["frame"]); v0 = float(v0s[fi]); po = tuple(float(x) for x in poses[fi]) if poses is not None else None
    t = time.perf_counter(); out = rt.infer(imgs, K[0][None], Tc[0][None], v0=v0, pose=po, out_slot=0); t_inf.append((time.perf_counter() - t) * 1000)
    outs.append((raw, v0, {k: (v.copy() if hasattr(v, "copy") else v) for k, v in out.items()}, po))
print(f"infer {np.mean(t_inf):.1f} ms/frame (n={N})")
# raw rendering time
ts = []
for raw, v0, out, po in outs:
    t = time.perf_counter(); R.compose_frame(raw, K, Tc, v0, out, 80.0, fps_now=5.0, pose=po); ts.append((time.perf_counter() - t) * 1000)
print(f"compose_frame {np.mean(ts):.0f} ms/frame (min {np.min(ts):.0f} max {np.max(ts):.0f})")
# with OCC removed
occ_bak = [o[2].pop("occ", None) for o in outs]
ts2 = []
for raw, v0, out, po in outs:
    t = time.perf_counter(); R.compose_frame(raw, K, Tc, v0, out, 80.0, fps_now=5.0, pose=po); ts2.append((time.perf_counter() - t) * 1000)
print(f"compose_frame without occ {np.mean(ts2):.0f} ms/frame")
os.environ["METEOR_SEG2D_OVERLAY"]="1"; R._SEG2D_OVERLAY=True; ts3=[]
for raw, v0, out, po in outs:
    t = time.perf_counter(); R.compose_frame(raw, K, Tc, v0, out, 80.0, fps_now=5.0, pose=po); ts3.append((time.perf_counter() - t) * 1000)
print(f"compose_frame with seg2d overlay {np.mean(ts3):.0f} ms/frame"); R._SEG2D_OVERLAY=False
for o, b in zip(outs, occ_bak):
    if b is not None: o[2]["occ"] = b
def _run(label, env):
    bak = {k: os.environ.get(k) for k in env}
    os.environ.update(env); R._SEG2D_OVERLAY = os.environ.get("METEOR_SEG2D_OVERLAY", "0") == "1"
    tt = []
    for raw, v0, out, po in outs:
        t = time.perf_counter(); R.compose_frame(raw, K, Tc, v0, out, 80.0, fps_now=5.0, pose=po); tt.append((time.perf_counter() - t) * 1000)
    print(f"[cfg] {label:48s} {np.mean(tt):5.0f} ms (min {np.min(tt):.0f} max {np.max(tt):.0f})")
    for k, v in bak.items():
        if v is None: os.environ.pop(k, None)
        else: os.environ[k] = v
_run("occ off", {"METEOR_OCC_PANEL": "0"})
_run("occ off + depth off", {"METEOR_OCC_PANEL": "0", "METEOR_DEPTH_PANEL": "0"})
_run("occ off + depth off + no thin", {"METEOR_OCC_PANEL": "0", "METEOR_DEPTH_PANEL": "0", "METEOR_NO_THIN": "1"})
_run("occ off + depth off + no thin + no segfuse", {"METEOR_OCC_PANEL": "0", "METEOR_DEPTH_PANEL": "0", "METEOR_NO_THIN": "1", "METEOR_SEG_FUSE": "0"})
# cProfile (per function)
pr = cProfile.Profile(); pr.enable()
for raw, v0, out, po in outs: R.compose_frame(raw, K, Tc, v0, out, 80.0, fps_now=5.0, pose=po)
pr.disable(); sio = io.StringIO(); ps = pstats.Stats(pr, stream=sio).sort_stats("cumulative"); ps.print_stats(28); txt = sio.getvalue()
for l in txt.splitlines():
    if "ncalls" in l or ("orin_render" in l or "cv2" in l or "numpy" in l or "{built-in" in l) and l.strip(): print(l[:150])

#!/usr/bin/env python3
"""Dump 2D BBoxes at a low threshold (for threshold sweeps, 2026-09-06). Usage: python3 det2d_dump.py <engine> <root> <out.json> [stride]"""
import sys, os, json, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deploy.runtime import MeteorRT
from deploy.viz_np import decode_boxes2d_ms_np
CAMS = ["CAM_FRONT_WIDE", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK_WIDE", "CAM_BACK_LEFT", "CAM_BACK_RIGHT", "CAM_FRONT_NARROW", "CAM_BACK_NARROW"]
eng, root, outp = sys.argv[1], sys.argv[2], sys.argv[3]; ST = int(sys.argv[4]) if len(sys.argv) > 4 else 4
rt = MeteorRT(eng, skip_outputs=("flow", "unk", "pl", "tl", "lg_pts", "lg_meta", "lg_adj", "depth", "seg2d", "occ", "traj", "lane"), n_out_slots=1)
res = []
for s in sorted(d for d in os.listdir(root) if os.path.isfile(os.path.join(root, d, "manifest.json"))):
    m = json.load(open(os.path.join(root, s, "manifest.json")))
    K = np.stack([np.array(m["cams"][c]["K"], np.float32) for c in CAMS])[None]
    Tc = np.stack([np.linalg.inv(np.array(m["cams"][c]["T_ego_cam"], np.float32)) for c in CAMS])[None]
    z = np.load(os.path.join(root, s, "ego_motion.npz")); v0s = z["v0"]
    for f in m["frames"][::ST]:
        raw = [cv2.imread(os.path.join(root, s, f["imgs"][c])) for c in CAMS]
        if any(r is None for r in raw): continue
        imgs = np.ascontiguousarray(np.stack([r[:, :, ::-1].transpose(2, 0, 1) for r in raw])[None]).astype(np.uint8)
        fi = int(f["frame"]); v0 = float(v0s[fi]) if fi < len(v0s) else 8.0
        out = rt.infer(imgs, K[0][None], Tc[0][None], v0=v0, pose=None, out_slot=0)
        b2d = decode_boxes2d_ms_np([out[f"hm2d_s{i}"][0] for i in range(3)], [out[f"reg2d_s{i}"][0] for i in range(3)], thresh=0.05, topk=96)
        res.append({"scene": s, "frame": fi, "boxes": [[[float(x) for x in b] for b in cam] for cam in b2d]})
json.dump(res, open(outp, "w")); print("DUMPED", len(res), "frames ->", outp)

#!/usr/bin/env python3
"""Turn 3D box auto-labels of KITScenes scenes into METEOR box GT (KITScenes itself ships no 3D boxes).

The labels come from an offline LiDAR 3D detector run over the converted scenes (in our pipeline: the CoMET
auto-labeler on the 7 merged LiDARs). They are pseudo-labels, not human annotations; detection scores on KITScenes
measured against them are reference numbers only.

Input  <labels>/<scene>/<label>  (default name boxes3d.json), one file per converted scene KITS_<uuid>:
         {"frames": [{"frame": <bevlane frame index>, "boxes": [[x, y, z, l, w, h, yaw, score], ...],
                      "names": ["car", "pedestrian", ...]}, ...]}
       boxes in the bevlane ego frame of that scene (x forward, y left, z up, road at z=0; see kitscenes_to_bevlane.py).
       Instead of "frame", a frame may carry "k" plus a <labels>/<scene>/kits_meta.json with
         {"frames": [{"k": ..., "bevlane_frame": ...}, ...]}.
Output <root>/<scene>/bev_box_c/NNNN.png  (BEV raster 800x500 @0.2 m, 1 = vehicle, 2 = VRU)
                      bev_box_c/NNNN.npz  (boxes [N,6] = cls, x, y, l, w, yaw)
                      agent_traj_c/NNNN.npz (boxes [64,6] nearest first, count; traj / tvalid = 0: no future)
       The scene's own bev_box_p / agent_traj and manifest are left untouched; training reads the _c files only
       with METEOR_KITS_BOX=1 (bevlane/dataset.py).
Classes: vehicles (car / truck / bus / trailer / construction_vehicle) as labelled, VRU (pedestrian / bicycle /
       motorcycle) with score >= --vru-score. There are no track ids, so no future trajectories (tvalid = 0: the
       forecasting loss is off, only detection is trained).

usage: python tools/kitscenes/kits_box_gt.py --labels <label root> [--root out/bevlane] [--label boxes3d.json] [--workers 16]
"""
import argparse, glob, json, os
from concurrent.futures import ProcessPoolExecutor
import numpy as np, cv2

VEH = {"car", "truck", "bus", "trailer", "construction_vehicle"}; VRU = {"pedestrian", "bicycle", "motorcycle"}
BEV_XH, BEV_YH, RES, H, W, KMAX = 80.0, 50.0, 0.2, 800, 500, 64


def ego_to_pix(xy):
    return np.stack([(BEV_YH - xy[:, 1]) / RES - 0.5, (BEV_XH - xy[:, 0]) / RES - 0.5], 1)


def one(args):
    s, labels, label, root, vru_s = args
    try:
        L = json.load(open(f"{labels}/{s}/{label}"))["frames"]
        meta = f"{labels}/{s}/kits_meta.json"
        k2f = {m["k"]: m["bevlane_frame"] for m in json.load(open(meta))["frames"]} if os.path.isfile(meta) else {}
        D = os.path.join(root, s); os.makedirs(f"{D}/bev_box_c", exist_ok=True); os.makedirs(f"{D}/agent_traj_c", exist_ok=True)
        nb = 0
        for fr in L:
            fi = int(fr["frame"]) if "frame" in fr else k2f[fr["k"]]; bx = []
            for b, n in zip(fr["boxes"], fr["names"]):
                cls = 1 if n in VEH else 2 if n in VRU else 0
                if not cls or (cls == 2 and b[7] < vru_s) or abs(b[0]) > BEV_XH or abs(b[1]) > BEV_YH:
                    continue
                bx.append([cls, b[0], b[1], b[3], b[4], b[6]])
            bx.sort(key=lambda r: r[1] ** 2 + r[2] ** 2)
            bimg = np.zeros((H, W), np.uint8)
            for cls, x, y, l, w, yaw in bx:
                c, sn = np.cos(yaw), np.sin(yaw)
                cor = np.array([[l / 2, w / 2], [l / 2, -w / 2], [-l / 2, -w / 2], [-l / 2, w / 2]]) @ np.array([[c, sn], [-sn, c]]) + [x, y]
                cv2.fillPoly(bimg, [np.round(ego_to_pix(cor) * 8).astype(np.int32)], int(cls), cv2.LINE_8, 3)
            cv2.imwrite(f"{D}/bev_box_c/{fi:04d}.png", bimg)
            B = np.asarray(bx, np.float32).reshape(-1, 6)
            np.savez_compressed(f"{D}/bev_box_c/{fi:04d}.npz", boxes=B)
            TB = np.zeros((KMAX, 6), np.float32); TB[:min(len(B), KMAX)] = B[:KMAX]
            np.savez_compressed(f"{D}/agent_traj_c/{fi:04d}.npz", boxes=TB, count=np.int64(min(len(B), KMAX)),
                                traj=np.zeros((KMAX, 6, 2), np.float32), tvalid=np.zeros((KMAX, 6), np.float32))
            nb += len(B)
        json.dump({"label": label, "vru_score": vru_s, "frames": len(L), "boxes": nb}, open(f"{D}/bev_box_c/SOURCE.json", "w"))
        return f"[ok] {s} {len(L)} {nb}"
    except Exception as e:
        return f"[fail] {s}: {e!r}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True, help="root with <scene>/<label> auto-label files")
    ap.add_argument("--root", default="out/bevlane", help="bevlane data root with the converted KITS_* scenes")
    ap.add_argument("--label", default="boxes3d.json")
    ap.add_argument("--vru-score", type=float, default=0.3, help="minimum score of VRU boxes")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    S = sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(f"{a.labels}/KITS_*/{a.label}"))
    fail = 0
    with ProcessPoolExecutor(a.workers) as ex:
        for i, r in enumerate(ex.map(one, [(s, a.labels, a.label, a.root, a.vru_score) for s in S])):
            fail += r.startswith("[fail")
            if i % 100 == 0 or r.startswith("[fail"):
                print(f"{i + 1}/{len(S)} {r}", flush=True)
    print("DONE", len(S), "fail", fail)


if __name__ == "__main__":
    main()

"""配布時 (INT8/最適化後) の精度を GT に対して測る。

同じ指標コードを 2 つの経路で使う:
  --engine  TensorRT エンジン (Orin 実機。torch 不要)
  --ckpt    PyTorch の重み (ローカル GPU。基準値づくり)
どちらも同じフレーム列・同じ GT・同じ判定で測るので、
「PyTorch -> fp16 -> INT8 -> 各最適化」の劣化を段階ごとに切り分けられる。

指標:
  BEV Seg  クラス別 IoU と mIoU (255 は無視)
  3D Det   車両/VRU の再現率・適合率・位置誤差・yaw 誤差
           (しきい値を振って、誤検出率を揃えた比較ができるようにする)

GT の規約: bev_box の npz は [cls, x, y, l, w, yaw] で cls=1 が車両、
cls=2 が VRU。モデルのヒートマップは 0=車両, 1=VRU なので cls-1 で対応する。
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CAMS = ["CAM_FRONT_WIDE", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
        "CAM_BACK_WIDE", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
        "CAM_FRONT_NARROW", "CAM_BACK_NARROW"]
SEG_NAMES = ["背景", "road", "sidewalk", "crosswalk", "laneline",
             "stopline", "road_edge", "marking", "parking"]
THS = [0.15, 0.25, 0.35, 0.45]
MATCH_R = 2.0                       # GT と予測の対応付け半径 [m]


def frames(root, scenes, n_cams, stride, limit):
    """(imgs uint8, K, Tc, v0, pose, gt_seg, boxes) を順に返す。"""
    got = 0
    for s in scenes:
        d = os.path.join(root, s)
        mp = os.path.join(d, "manifest.json")
        if not os.path.isfile(mp):
            continue
        m = json.load(open(mp))
        cams = [c for c in CAMS[:n_cams] if c in m["cams"]]
        if len(cams) < n_cams:
            continue
        # シーンの端は蓄積 LiDAR が無く GT が弱いので、学習側と同じく落とす
        fr = m["frames"]
        if len(fr) > 20:
            fr = fr[3:len(fr) - 10]
        for f in fr[::stride]:
            if "gt_cons" not in f or "bev_box_p" not in f:
                continue
            try:
                ims = np.stack([cv2.imread(os.path.join(d, f["imgs"][c]))[:, :, ::-1]
                                for c in cams])
                gt = cv2.imread(os.path.join(d, f["gt_cons"]),
                                cv2.IMREAD_UNCHANGED)
                bx = np.load(os.path.join(d, f["bev_box_p"]))["boxes"]
            except Exception:
                continue
            if gt is None:
                continue
            K = np.stack([np.array(m["cams"][c]["K"], np.float32) for c in cams])
            # マニフェストは T_ego_cam を持つので反転して T_cam_ego にする
            Tc = np.stack([np.linalg.inv(np.array(m["cams"][c]["T_ego_cam"],
                                                  np.float32))
                           for c in cams])
            i = int(f["frame"])
            v0, pose = np.float32(0), np.zeros(3, np.float32)
            try:
                em = np.load(os.path.join(d, "ego_motion.npz"))
                if "v0" in em and i < len(em["v0"]):
                    v0 = np.float32(em["v0"][i])
                if "pose" in em and i < len(em["pose"]):
                    pose = np.asarray(em["pose"][i], np.float32)
            except Exception:
                pass
            yield (np.ascontiguousarray(ims.transpose(0, 3, 1, 2))[None],
                   K[None], Tc[None], np.array([v0], np.float32),
                   np.asarray(pose, np.float32), gt, bx)
            got += 1
            if got >= limit:
                return


def seg_update(conf, pred, gt):
    """pred/gt は同じ大きさのクラス index マップ。255 は無視。"""
    h = min(pred.shape[0], gt.shape[0])
    p, g = pred[:h].ravel(), gt[:h].ravel()
    ok = g != 255
    p, g = p[ok], g[ok]
    n = conf.shape[0]
    ok = (p < n) & (g < n)
    np.add.at(conf, (g[ok], p[ok]), 1)


def det_update(acc, dets, boxes, xf, xr):
    """dets: [(cls, score, x, y, l, w, yaw)], boxes: GT [cls,x,y,l,w,yaw]."""
    for ci, gcls in ((0, 1), (1, 2)):                    # 0=車両, 1=VRU
        # 判定窓は従来計測と同じ (前 50 m / 後 20 m, 半径 50 m 以内)。
        # ここを広げると遠方の難しい箱が入って再現率が実力より低く出る。
        gt = []
        for b in boxes:
            if int(b[0]) != gcls or float(b[3]) <= 0:
                continue
            x, y = float(b[1]), float(b[2])
            r = (x * x + y * y) ** 0.5
            # 2026-08-18: 後方 20 m 固定も旧測定の残骸。格子の実範囲で判定する
            if x > 78 or x < -(xr - 2.0) or r > 78:
                continue
            gt.append(b)
        acc[ci]["gt"] += len(gt)
        for t in THS:
            pred = [d for d in dets if int(d[0]) == ci and float(d[1]) > t]
            used = set()
            for b in gt:
                best = None
                for j, d in enumerate(pred):
                    if j in used:
                        continue
                    d2 = (float(b[1]) - d[2]) ** 2 + (float(b[2]) - d[3]) ** 2
                    if d2 < MATCH_R ** 2 and (best is None or d2 < best[0]):
                        best = (d2, j, d)
                if best is None:
                    continue
                used.add(best[1])
                d = best[2]
                a = acc[ci][t]
                a["hit"] += 1
                a["pos"].append(best[0] ** 0.5)
                de = abs((d[6] - float(b[5]) + np.pi) % (2 * np.pi) - np.pi)
                a["yaw"].append(np.degrees(min(de, np.pi - de)))
            acc[ci][t]["fp"] += len(pred) - len(used)


def new_acc():
    return [{"gt": 0, **{t: {"hit": 0, "fp": 0, "pos": [], "yaw": []}
                         for t in THS}} for _ in range(2)]


def report(conf, acc, nfr, tag):
    print(f"\n===== {tag} ({nfr} フレーム) =====")
    inter = np.diag(conf).astype(np.float64)
    union = conf.sum(1) + conf.sum(0) - np.diag(conf)
    iou = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
    present = union > 0
    print(f"BEV Seg mIoU={np.nanmean(iou[present]):.3f}  " +
          "  ".join(f"{SEG_NAMES[i]}={iou[i]:.3f}"
                    for i in range(len(iou)) if present[i]))
    for ci, nm in ((0, "車両"), (1, "VRU")):
        g = acc[ci]["gt"]
        if not g:
            continue
        print(f"  {nm} (GT {g} 箱)")
        for t in THS:
            a = acc[ci][t]
            r = a["hit"] / max(g, 1)
            p = a["hit"] / max(a["hit"] + a["fp"], 1)
            pos = np.median(a["pos"]) if a["pos"] else float("nan")
            yaw = np.median(a["yaw"]) if a["yaw"] else float("nan")
            print(f"    th={t:.2f}  再現率 {r:5.3f}  適合率 {p:5.3f}  "
                  f"誤検出/フレーム {a['fp'] / max(nfr, 1):5.2f}  "
                  f"位置誤差 {pos:.2f}m  yaw {yaw:5.1f}度")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine")
    ap.add_argument("--ckpt")
    ap.add_argument("--model", default="v52")
    ap.add_argument("--root", default="fast")
    ap.add_argument("--n-cams", type=int, default=8)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--scenes-file",
                    help="評価するシーン名の一覧 (Orin と同一フレームで測るため)")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    if a.scenes_file:
        scenes = [l.strip() for l in open(a.scenes_file) if l.strip()]
    else:
        scenes = sorted(s for s in os.listdir(a.root)
                        if os.path.isfile(os.path.join(a.root, s,
                                                       "manifest.json")))
    conf = np.zeros((9, 9), np.int64)
    acc = new_acc()
    nfr = 0

    if a.engine:
        from deploy.runtime import MeteorRT, decode_boxes
        rt = MeteorRT(a.engine, n_out_slots=1)
        lane_h = rt.shapes["lane"][-2]
        xr = lane_h * 0.2 - 80.0
        for imgs, K, Tc, v0, pose, gt, bx in frames(
                a.root, scenes, a.n_cams, a.stride, a.limit):
            o = rt.infer(imgs, K, Tc, v0, pose=pose)
            lane = o["lane"][0]
            if lane.ndim == 3:
                lane = lane.argmax(0)
            seg_update(conf, lane.astype(np.int64), gt.astype(np.int64))
            # ランタイムの decode_boxes は dict を返すので、指標側の
            # (クラス番号, スコア, x, y, l, w, yaw) に揃える。
            dets = [(0 if d["cls"] == "vehicle" else 1, d["score"],
                     d["x"], d["y"], d["l"], d["w"], d["yaw"])
                    for d in decode_boxes(o["hm"], o["reg"], thresh=min(THS))]
            det_update(acc, dets, bx, 80.0, xr)
            nfr += 1
    else:
        import torch
        from bevlane.model import MODELS
        net = MODELS[a.model](n_seg=21).cuda().eval()
        sd = torch.load(a.ckpt, map_location="cpu")
        sd = {k.replace("module.", ""): v for k, v in sd.get("model", sd).items()}
        cur = net.state_dict()
        net.load_state_dict({k: v for k, v in sd.items()
                             if k in cur and cur[k].shape == v.shape},
                            strict=False)
        xr = 80.0
        for imgs, K, Tc, v0, pose, gt, bx in frames(
                a.root, scenes, a.n_cams, a.stride, a.limit):
            ims = torch.from_numpy(imgs).cuda().float() / 255.0
            with torch.no_grad(), torch.autocast("cuda", torch.float16):
                out = net(ims, torch.from_numpy(K).cuda(),
                          torch.from_numpy(Tc).cuda())
            lane = out[0][0].float().argmax(0).cpu().numpy()
            seg_update(conf, lane.astype(np.int64), gt.astype(np.int64))
            dets = net.decode_boxes(out[3].float().cpu(), out[4].float().cpu(),
                                    thresh=min(THS))[0]
            det_update(acc, [tuple(float(v) for v in d) for d in dets],
                       bx, 80.0, xr)
            nfr += 1

    report(conf, acc, nfr, a.tag or (a.engine or a.ckpt))


if __name__ == "__main__":
    main()

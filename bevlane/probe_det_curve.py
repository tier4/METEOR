"""しきい値を振って再現率と誤検出数を出す (モデル間の検出性能比較用)。

固定しきい値 1 点での検出数比較はモデル間のスコア較正差に汚染される
(同じ 0.25 でも出方が違う)。曲線で見ないと「劣化したのか、しきい値が
合っていないだけなのか」が分けられない。
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bevlane.dataset import BevLaneDataset                        # noqa: E402
from bevlane.model import MODELS                                  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-cams", type=int, default=8)
    ap.add_argument("--list", default="val.lst")
    ap.add_argument("--root", default="out/bevlane")
    ap.add_argument("--scenes", type=int, default=60)
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--range-th", default="",
                    help="距離別しきい値 例 20:0.25,40:0.18,80:0.12 "
                         "(距離までの上限:しきい値)")
    ap.add_argument("--zero-cams", default="",
                    help="指定カメラを黒画像にして寄与を測る "
                         "(例: CAM_FRONT_NARROW)")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    ths = [0.10, 0.15, 0.20, 0.25, 0.35, 0.45]
    scenes = [l.strip() for l in open(a.list) if l.strip()][:a.scenes]
    ds = BevLaneDataset(a.root, scenes, gt_key="gt_cons", with_boxdet=True,
                        max_per_scene=4, n_cams=8, trim_start=3, trim_end=10)
    m = MODELS[a.model](n_seg=21).cuda().eval()
    sd = torch.load(a.ckpt, map_location="cpu")
    sd = {k.replace("module.", ""): v for k, v in sd.get("model", sd).items()}
    cur = m.state_dict()
    m.load_state_dict({k: v for k, v in sd.items()
                       if k in cur and cur[k].shape == v.shape}, strict=False)

    from bevlane.dataset import CAMS as _DSCAMS
    zc = [_DSCAMS.index(c) for c in a.zero_cams.split(",") if c] \
        if a.zero_cams else []
    rth = []
    for part in (a.range_th.split(",") if a.range_th else []):
        lim, th = part.split(":")
        rth.append((float(lim), float(th)))
    BANDS = [(0, 20), (20, 40), (40, 60), (60, 80)]
    rt_hit = {b: 0 for b in BANDS}
    rt_fp = 0
    band_gt = {b: 0 for b in BANDS}
    band_hit = {(b, t): 0 for b in BANDS for t in ths}
    hit = {t: 0 for t in ths}
    fp = {t: 0 for t in ths}
    yaw = {t: [] for t in ths}
    ngt = 0
    step = max(1, len(ds) // a.frames)
    done = 0
    for i in range(0, len(ds), step):
        b = ds[i]
        if b is None:
            continue
        ims = b[0][None][:, :a.n_cams].clone()
        for _ci in zc:
            if _ci < ims.shape[1]:
                ims[:, _ci] = 0                     # そのカメラだけ黒にする
        with torch.no_grad(), torch.autocast("cuda", torch.float16):
            out = m(ims.cuda(),
                    b[1][None][:, :a.n_cams].cuda(),
                    b[2][None][:, :a.n_cams].cuda())
        hm, reg = out[3].float().cpu(), out[4].float().cpu()
        gts = []
        bx, nb = b[4], int(b[5])
        for k in range(max(nb, 0)):
            cls, xe, ye, ln, wd, yw = [float(v) for v in bx[k][:6]]
            if ln <= 0 or cls >= 1.5:
                continue
            r = (xe * xe + ye * ye) ** 0.5
            # 距離帯別に見るため、前方は 80 m まで拾う
            if 0 < xe <= 80 or (-20 <= xe <= 0 and r <= 50):
                gts.append((xe, ye, yw))
        ngt += len(gts)
        for (xe, ye, _y) in gts:
            rr = (xe * xe + ye * ye) ** 0.5
            for b in BANDS:
                if b[0] <= rr < b[1]:
                    band_gt[b] += 1
        if rth:
            low = min(t for _l, t in rth)
            dd = [d for d in m.decode_boxes(hm, reg, thresh=low)[0]
                  if int(d[0]) == 0]
            keep = []
            for d in dd:
                rr = (float(d[2]) ** 2 + float(d[3]) ** 2) ** 0.5
                th_here = rth[-1][1]
                for lim, th in rth:
                    if rr <= lim:
                        th_here = th
                        break
                if float(d[1]) > th_here:
                    keep.append((float(d[2]), float(d[3])))
            used_r = set()
            for (xe, ye, _yw) in gts:
                bst = None
                for j, (dx, dy) in enumerate(keep):
                    if j in used_r:
                        continue
                    d2 = (xe - dx) ** 2 + (ye - dy) ** 2
                    if d2 < 4.0 and (bst is None or d2 < bst[0]):
                        bst = (d2, j)
                if bst is not None:
                    used_r.add(bst[1])
                    rr = (xe * xe + ye * ye) ** 0.5
                    for bb in BANDS:
                        if bb[0] <= rr < bb[1]:
                            rt_hit[bb] += 1
            rt_fp += len(keep) - len(used_r)
        for t in ths:
            dets = m.decode_boxes(hm, reg, thresh=t)[0]
            pred = [(float(d[2]), float(d[3]), float(d[6]))
                    for d in dets if int(d[0]) == 0]
            used = set()
            for (xe, ye, yw) in gts:
                best = None
                for j, (dx, dy, dyaw) in enumerate(pred):
                    if j in used:
                        continue
                    d2 = (xe - dx) ** 2 + (ye - dy) ** 2
                    if d2 < 4.0 and (best is None or d2 < best[0]):
                        best = (d2, j, dyaw)
                if best is not None:
                    used.add(best[1])
                    hit[t] += 1
                    rr = (xe * xe + ye * ye) ** 0.5
                    for b in BANDS:
                        if b[0] <= rr < b[1]:
                            band_hit[(b, t)] += 1
                    de = abs((best[2] - yw + np.pi) % (2 * np.pi) - np.pi)
                    yaw[t].append(np.degrees(min(de, np.pi - de)))
            fp[t] += len(pred) - len(used)
        done += 1
        if done >= a.frames:
            break

    if rth:
        # 距離別しきい値: 低いしきい値で一度デコードし、箱ごとの距離に応じて
        # 採否を決める。遠方だけスコア基準を緩めたときの再現率と誤検出を見る。
        print(f"=== {a.tag} 距離別しきい値 {a.range_th} ===")
        print("  帯別再現率: " + "  ".join(
            f"{b[0]}-{b[1]}m {rt_hit[b] / max(band_gt[b], 1):5.3f}"
            for b in BANDS))
        print(f"  誤検出/フレーム {rt_fp / max(done, 1):5.2f}  "
              f"全体再現率 {sum(rt_hit.values()) / max(ngt, 1):5.3f}")
    print(f"=== {a.tag or a.ckpt} ({done} フレーム, GT 車両 {ngt} 箱) ===")
    print("しきい値  再現率      誤検出/フレーム  yaw中央値")
    for t in ths:
        rc = hit[t] / max(ngt, 1)
        ym = np.median(yaw[t]) if yaw[t] else float("nan")
        print(f"  {t:.2f}   {rc:5.3f} ({hit[t]:4d})   "
              f"{fp[t] / max(done, 1):5.2f}          {ym:5.1f} 度")
    print("\n距離帯別の再現率 (GT 箱数)")
    print("  しきい値 " + "  ".join(f"{b[0]:2d}-{b[1]:2d}m" for b in BANDS))
    for b in BANDS:
        pass
    for t in ths:
        row = "  ".join(f"{band_hit[(b, t)] / max(band_gt[b], 1):6.3f}"
                        for b in BANDS)
        print(f"    {t:.2f}   {row}")
    print("  GT 箱数 " + "  ".join(f"{band_gt[b]:6d}" for b in BANDS))


if __name__ == "__main__":
    main()

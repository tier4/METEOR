"""depth_gt4n (望遠 2 台の深度教師) が GT 箱と整合しているかを検証する。

v1 は全カメラで |差| 16-27 m と出た: これは教師ではなく検証側の投影が
壊れている値 (FRONT_WIDE ですら 25 m)。外部パラメータの向き (T_ego_cam が
ego->cam か cam->ego か) と深度の定義 (z 深度か光線距離か) を仮定せず、
既知の正解 (前方カメラには前方の点が写る) で規約を自動判定してから測る。

occlusion 対策: 箱中心の画素が手前の別車で隠れることがあるので、
|差| < 4 m を「整合」とみなし、整合率と整合集合の系統ずれを報告する。
教師のずれは「整合集合の中央値の前後差」に現れる。
"""
import argparse
import json
import os

import numpy as np

CHECK = ["CAM_FRONT_WIDE", "CAM_BACK_WIDE",
         "CAM_FRONT_NARROW", "CAM_BACK_NARROW"]
IMG_W, IMG_H = 768, 432
DS = 4


def pick_convention(M, K, is_front):
    """ego->cam 行列を返す。前方カメラなら (20,0,1)、後方なら (-20,0,1) が
    Z>0 かつ画像内に入る向きを選ぶ。"""
    p = np.array([20.0 if is_front else -20.0, 0.0, 1.0, 1.0])
    for cand in (M, np.linalg.inv(M)):
        q = cand @ p
        if q[2] <= 0:
            continue
        u = K[0][0] * q[0] / q[2] + K[0][2]
        v = K[1][1] * q[1] / q[2] + K[1][2]
        if 0 <= u < IMG_W and 0 <= v < IMG_H:
            return cand
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", default="val.lst")
    ap.add_argument("--root", default="out/bevlane")
    ap.add_argument("--scenes", type=int, default=40)
    ap.add_argument("--frames", type=int, default=250)
    a = ap.parse_args()

    scenes = [l.strip() for l in open(a.list) if l.strip()][:a.scenes]
    # (カメラ, 深度定義) 別の差分。定義: "z" = z 深度, "ray" = 光線距離
    diffs = {(c, m): [] for c in CHECK for m in ("z", "ray")}
    conv_note = {}
    nfr = 0
    for s in scenes:
        d = os.path.join(a.root, s)
        try:
            man = json.load(open(os.path.join(d, "manifest.json")))
        except Exception:
            continue
        if any(c not in man["cams"] for c in CHECK):
            continue
        Ms, Ks = {}, {}
        ok = True
        for c in CHECK:
            K = np.array(man["cams"][c]["K"], np.float64)
            M = pick_convention(np.array(man["cams"][c]["T_ego_cam"],
                                         np.float64), K, "FRONT" in c)
            if M is None:
                ok = False
                break
            Ms[c], Ks[c] = M, K
            conv_note.setdefault(c, "inv" if not np.allclose(
                M, np.array(man["cams"][c]["T_ego_cam"])) else "そのまま")
        if not ok:
            continue
        fr = man["frames"]
        if len(fr) > 20:
            fr = fr[3:len(fr) - 10]
        for f in fr[::10]:
            if "bev_box_p" not in f or "depth4" not in f:
                continue
            try:
                bx = np.load(os.path.join(d, f["bev_box_p"]))["boxes"]
                dg6 = np.load(os.path.join(d, f["depth4"]))["depth"]
                dgn = np.load(os.path.join(d, f["depth4n"]))["depth"]
            except Exception:
                continue
            dmap = {"CAM_FRONT_WIDE": dg6[0], "CAM_BACK_WIDE": dg6[3],
                    "CAM_FRONT_NARROW": dgn[0], "CAM_BACK_NARROW": dgn[1]}
            for r in bx:
                if int(r[0]) != 1 or float(r[3]) <= 0:
                    continue
                x, y = float(r[1]), float(r[2])
                rr = (x * x + y * y) ** 0.5
                if not (15 <= rr <= 45):
                    continue
                p = np.array([x, y, 1.0, 1.0])
                for c in CHECK:
                    q = Ms[c] @ p
                    Zc = float(q[2])
                    if Zc < 5.0:
                        continue
                    u = Ks[c][0][0] * q[0] / Zc + Ks[c][0][2]
                    v = Ks[c][1][1] * q[1] / Zc + Ks[c][1][2]
                    if not (0 <= u < IMG_W and 0 <= v < IMG_H):
                        continue
                    uu, vv = int(u / DS), int(v / DS)
                    g = dmap[c][max(0, vv - 2):vv + 3, max(0, uu - 2):uu + 3]
                    g = g[g > 0.5]
                    if len(g) < 3:
                        continue
                    est = float(np.median(g))
                    ray = float(np.linalg.norm(q[:3]))
                    diffs[(c, "z")].append(est - Zc)
                    diffs[(c, "ray")].append(est - ray)
            nfr += 1
            if nfr >= a.frames:
                break
        if nfr >= a.frames:
            break

    print(f"=== 外部パラメータの規約判定 ===")
    for c, note in conv_note.items():
        print(f"  {c:<20} T_ego_cam を {note} 使用")
    # 深度の定義は FRONT_WIDE の整合率が高い方を採用し、全カメラに適用
    def inl(v):
        v = np.array(v)
        m = np.abs(v) < 4.0
        return m.mean() if len(v) else 0.0, v[m]
    fz, _ = inl(diffs[("CAM_FRONT_WIDE", "z")])
    fray, _ = inl(diffs[("CAM_FRONT_WIDE", "ray")])
    mode = "z" if fz >= fray else "ray"
    print(f"\n深度の定義: {'z 深度' if mode == 'z' else '光線距離'} を採用 "
          f"(FRONT_WIDE の整合率 z={100*fz:.0f}% / ray={100*fray:.0f}%)")

    print(f"\n=== 深度教師と GT 箱距離の整合 ({nfr} フレーム, 15-45 m の車両) ===")
    print("カメラ                 n    整合率(|差|<4m)  整合集合の系統ずれ")
    print("  ※ LiDAR は車体の手前面に当たるため -1〜-2 m の負が正常")
    for c in CHECK:
        v = np.array(diffs[(c, mode)])
        if len(v) < 10:
            print(f"  {c:<20} {len(v):5d}  標本不足")
            continue
        frac, iv = inl(v)
        print(f"  {c:<20} {len(v):5d}   {100*frac:5.1f}%          "
              f"{np.median(iv):+6.2f} m")


if __name__ == "__main__":
    main()

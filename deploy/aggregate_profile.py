"""trtexec --exportProfile の JSON をブロック別に集計する。

    python3 deploy/aggregate_profile.py out/prof_v95.json [out/prof_r73sb.json]
"""
import json
import re
import sys

GROUPS = [
    ("画像バックボーン", r"image_feats|backbone|resnet|/net/(layer|conv1|bn1|maxpool|fpn)|stem"),
    ("深度ヘッド (64bin)", r"depth_head|depth_up|sharpen|dprob"),
    ("2D セグ/塗り", r"seg2d|paint|painter"),
    ("ctx 圧縮", r"/ctx|ctx/"),
    ("リフト (プラグイン)", r"MeteorLift|lift|project_bev|frustum"),
    ("BEV 追加畳込み", r"bev_extra"),
    ("時系列融合", r"temporal|hist|warp|fuse"),
    ("BEV セグ dec", r"/dec|seg_head|lane"),
    ("3D det ヘッド", r"det|/hm|/reg|box"),
    ("E2E/ego 系", r"ego|intent|kin|vprof|gate|mode"),
    ("occ ヘッド", r"occ"),
    ("traj/tl/risk 他", r"traj|/tl|risk|flow|stat"),
]


def agg(path):
    data = json.load(open(path))
    rows = [d for d in data if isinstance(d, dict) and "name" in d]
    tot = sum(d.get("averageMs", 0.0) for d in rows)
    out = {g: 0.0 for g, _ in GROUPS}
    out["その他"] = 0.0
    unmatched = []
    for d in rows:
        nm = d["name"]
        ms = d.get("averageMs", 0.0)
        for g, pat in GROUPS:
            if re.search(pat, nm, re.I):
                out[g] += ms
                break
        else:
            out["その他"] += ms
            unmatched.append((ms, nm))
    return tot, out, sorted(unmatched, reverse=True)[:12], rows


def top_layers(rows, n=15):
    return sorted(rows, key=lambda d: -d.get("averageMs", 0))[:n]


if __name__ == "__main__":
    for p in sys.argv[1:]:
        tot, out, unm, rows = agg(p)
        print(f"\n=== {p}  合計 {tot:.2f} ms")
        for g, ms in sorted(out.items(), key=lambda x: -x[1]):
            if ms > 0.05:
                print(f"  {g:<16} {ms:6.2f} ms ({100*ms/tot:4.1f}%)")
        print("  --- 個別上位:")
        for d in top_layers(rows, 12):
            print(f"    {d.get('averageMs', 0):6.3f} ms  {d['name'][:90]}")
        if unm:
            print("  --- 未分類上位:", [u[1][:40] for u in unm[:5]])

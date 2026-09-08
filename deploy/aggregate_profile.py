"""Aggregate trtexec --exportProfile JSON by block.

    python3 deploy/aggregate_profile.py out/prof_v95.json [out/prof_r73sb.json]
"""
import json
import re
import sys

GROUPS = [
    ("image backbone", r"image_feats|backbone|resnet|/net/(layer|conv1|bn1|maxpool|fpn)|stem"),
    ("depth head (64bin)", r"depth_head|depth_up|sharpen|dprob"),
    ("2D seg/paint", r"seg2d|paint|painter"),
    ("ctx compress", r"/ctx|ctx/"),
    ("lift (plugin)", r"MeteorLift|lift|project_bev|frustum"),
    ("BEV extra conv", r"bev_extra"),
    ("temporal fusion", r"temporal|hist|warp|fuse"),
    ("BEV seg dec", r"/dec|seg_head|lane"),
    ("3D det head", r"det|/hm|/reg|box"),
    ("E2E/ego", r"ego|intent|kin|vprof|gate|mode"),
    ("occ head", r"occ"),
    ("traj/tl/risk misc", r"traj|/tl|risk|flow|stat"),
]


def agg(path):
    data = json.load(open(path))
    rows = [d for d in data if isinstance(d, dict) and "name" in d]
    tot = sum(d.get("averageMs", 0.0) for d in rows)
    out = {g: 0.0 for g, _ in GROUPS}
    out["other"] = 0.0
    unmatched = []
    for d in rows:
        nm = d["name"]
        ms = d.get("averageMs", 0.0)
        for g, pat in GROUPS:
            if re.search(pat, nm, re.I):
                out[g] += ms
                break
        else:
            out["other"] += ms
            unmatched.append((ms, nm))
    return tot, out, sorted(unmatched, reverse=True)[:12], rows


def top_layers(rows, n=15):
    return sorted(rows, key=lambda d: -d.get("averageMs", 0))[:n]


if __name__ == "__main__":
    for p in sys.argv[1:]:
        tot, out, unm, rows = agg(p)
        print(f"\n=== {p}  total {tot:.2f} ms")
        for g, ms in sorted(out.items(), key=lambda x: -x[1]):
            if ms > 0.05:
                print(f"  {g:<16} {ms:6.2f} ms ({100*ms/tot:4.1f}%)")
        print("  --- top layers:")
        for d in top_layers(rows, 12):
            print(f"    {d.get('averageMs', 0):6.3f} ms  {d['name'][:90]}")
        if unm:
            print("  --- top unmatched:", [u[1][:40] for u in unm[:5]])

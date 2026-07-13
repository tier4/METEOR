#!/usr/bin/env python3
"""Find consecutive completed scene chains, merge each, and vectorize (v2)."""
import argparse
import os
import re
import subprocess
import sys
from collections import defaultdict

PROD = "out/production"
OUT = "out/merged_chains"


def find_chains(min_len):
    byrec = defaultdict(list)
    for d in os.listdir(PROD):
        if os.path.exists(os.path.join(PROD, d, "bev_counts.npz")):
            m = re.match(r"(.+)_(\d+)$", d)
            byrec[m.group(1)].append(int(m.group(2)))
    chains = []
    for rec, idxs in sorted(byrec.items()):
        idxs = sorted(set(idxs))
        s = p = idxs[0]
        for i in idxs[1:] + [10 ** 9]:
            if i == p + 1:
                p = i
                continue
            if p - s + 1 >= min_len:
                chains.append((rec, s, p))
            s = p = i
    return chains


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-len", type=int, default=6)
    ap.add_argument("--max-scenes", type=int, default=20, help="cap per chain")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    chains = find_chains(args.min_len)
    print(f"{len(chains)} chains found", flush=True)
    for rec, s, e in chains:
        e = min(e, s + args.max_scenes - 1)
        name = f"{rec.split('_')[-1]}_{s}_{e}"
        out = os.path.join(OUT, name)
        if os.path.exists(os.path.join(out, "nuscenes_map.json")):
            print(f"[skip] {name}", flush=True)
            continue
        inputs = [os.path.join(PROD, f"{rec}_{i}") for i in range(s, e + 1)]
        try:
            subprocess.run([sys.executable, "merge_scenes.py", "--inputs", *inputs,
                            "--out", out], check=True, capture_output=True)
            subprocess.run([sys.executable, "vectorize_bev.py", out],
                           check=True, capture_output=True)
            print(f"[ok] {name} ({e - s + 1} scenes)", flush=True)
        except subprocess.CalledProcessError as ex:
            print(f"[fail] {name}: {ex.stderr.decode()[-300:]}", flush=True)


if __name__ == "__main__":
    main()

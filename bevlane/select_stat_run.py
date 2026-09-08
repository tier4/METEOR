#!/usr/bin/env python3
"""Select a stationary-head A/B run from validation logs.

Safety comes first: compare recall at >=95% moving accuracy, then balanced
accuracy.  Candidates more than 1 point behind the best BEV mIoU are gated
out so a stationary-only gain cannot silently damage the shared perception
backbone.  Prints ``checkpoint margin`` for direct use by a shell chain.
"""
import argparse
import os
import re


CAL = re.compile(
    r"\[valStat-cal ep\d+\].*?balanced:.*?/bal=([0-9.]+)"
    r".*?safe95:.*?/R=([0-9.]+).*?/bal=([0-9.]+)")
FIXED = re.compile(
    r"\[valStat ep\d+\].*?R=([0-9.]+).*?movAcc=([0-9.]+)")
MIOU = re.compile(r"\[val ep\d+\] mIoU=([0-9.]+)")


def parse(spec):
    name, log, ckpt, margin = spec.split(",", 3)
    text = open(log, encoding="utf-8", errors="replace").read()
    cm = CAL.findall(text)
    mm = MIOU.findall(text)
    if cm:
        balanced, safe_recall, safe_balanced = map(float, cm[-1])
    else:
        fm = FIXED.findall(text)
        if not fm:
            raise RuntimeError(f"no stationary validation metric in {log}")
        recall, moving = map(float, fm[-1])
        balanced = safe_balanced = (recall + moving) / 2
        safe_recall = recall if moving >= 0.95 else -1.0
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(ckpt)
    return {"name": name, "log": log, "ckpt": ckpt,
            "margin": float(margin), "miou": float(mm[-1]) if mm else 0.,
            "safe_recall": safe_recall, "balanced": balanced,
            "safe_balanced": safe_balanced}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", action="append", required=True,
                    help="NAME,LOG,CHECKPOINT,MARGIN")
    ap.add_argument("--report", default="")
    args = ap.parse_args()
    runs = [parse(s) for s in args.candidate]
    best_miou = max(r["miou"] for r in runs)
    eligible = [r for r in runs if r["miou"] >= best_miou - 0.01]
    winner = max(eligible, key=lambda r: (r["safe_recall"],
                                           r["safe_balanced"],
                                           r["balanced"], r["miou"]))
    report = "\n".join(
        f"{r['name']}: mIoU={r['miou']:.3f} safe95_R={r['safe_recall']:.3f} "
        f"safe_bal={r['safe_balanced']:.3f} bal={r['balanced']:.3f}"
        for r in runs) + f"\nWINNER={winner['name']}\n"
    if args.report:
        with open(args.report, "w") as fp:
            fp.write(report)
    print(winner["ckpt"], f"{winner['margin']:g}")


if __name__ == "__main__":
    main()

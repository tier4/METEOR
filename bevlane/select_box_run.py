#!/usr/bin/env python3
"""Select a box-geometry A/B checkpoint by physical corner error."""
import argparse
import os
import re

VAL = re.compile(r"\[val ep\d+\] mIoU=([0-9.]+)")
DET = re.compile(r"\[val3D ep\d+\] veh P=([0-9.]+) R=([0-9.]+).*?"
                 r"err=([0-9.]+)m.*?corner=([0-9.]+)m")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", action="append", required=True,
                    help="NAME,LOG,CHECKPOINT,CORNER_W")
    ap.add_argument("--report", default="")
    args = ap.parse_args()
    runs = []
    for spec in args.candidate:
        name, log, ckpt, weight = spec.split(",", 3)
        text = open(log, errors="replace").read()
        vm, dm = VAL.findall(text), DET.findall(text)
        if not vm or not dm:
            raise RuntimeError(f"no detailed box validation metric in {log}")
        precision, recall, centre, corner = map(float, dm[-1])
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(ckpt)
        runs.append(dict(name=name, ckpt=ckpt, weight=float(weight),
                         miou=float(vm[-1]), precision=precision,
                         recall=recall, centre=centre, corner=corner))
    best_miou = max(r["miou"] for r in runs)
    best_recall = max(r["recall"] for r in runs)
    eligible = [r for r in runs if r["miou"] >= best_miou - 0.01 and
                r["recall"] >= best_recall - 0.02]
    winner = min(eligible, key=lambda r: (r["corner"], r["centre"],
                                           -r["precision"]))
    report = "\n".join(
        f"{r['name']}: mIoU={r['miou']:.3f} P/R={r['precision']:.3f}/"
        f"{r['recall']:.3f} centre={r['centre']:.3f} corner={r['corner']:.3f}"
        for r in runs) + f"\nWINNER={winner['name']}\n"
    if args.report:
        with open(args.report, "w") as fp:
            fp.write(report)
    print(winner["ckpt"], f"{winner['weight']:g}")


if __name__ == "__main__":
    main()

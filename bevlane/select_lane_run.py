#!/usr/bin/env python3
"""Select a Lane A/B checkpoint while gating shared-BEV regression."""
import argparse
import os
import re

VAL = re.compile(r"\[val ep\d+\] mIoU=([0-9.]+).*?laneline=([0-9.]+)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", action="append", required=True,
                    help="NAME,LOG,CHECKPOINT,CLDICE_W,SDF_W")
    ap.add_argument("--report", default="")
    args = ap.parse_args()
    runs = []
    for spec in args.candidate:
        name, log, ckpt, cldice, sdf = spec.split(",", 4)
        vals = VAL.findall(open(log, errors="replace").read())
        if not vals:
            raise RuntimeError(f"no Lane validation metric in {log}")
        miou, lane = map(float, vals[-1])
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(ckpt)
        runs.append(dict(name=name, ckpt=ckpt, cldice=float(cldice),
                         sdf=float(sdf), miou=miou, lane=lane))
    best_miou = max(r["miou"] for r in runs)
    eligible = [r for r in runs if r["miou"] >= best_miou - 0.01]
    winner = max(eligible, key=lambda r: (r["lane"], r["miou"]))
    report = "\n".join(f"{r['name']}: mIoU={r['miou']:.3f} "
                         f"lane={r['lane']:.3f}" for r in runs)
    report += f"\nWINNER={winner['name']}\n"
    if args.report:
        with open(args.report, "w") as fp:
            fp.write(report)
    print(winner["ckpt"], f"{winner['cldice']:g}", f"{winner['sdf']:g}")


if __name__ == "__main__":
    main()

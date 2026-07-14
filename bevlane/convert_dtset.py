#!/usr/bin/env python3
"""Convert DTSET scenes end-to-end into training data (all stages per scene).

Per scene: autolabel production -> vectorize -> extract_gt(+narrow) -> gt_vec
-> depth (surround+narrow, fixed logic) -> bev_box -> ... -> tl_state. Scene-parallel; each stage
skips work that already exists, so the driver is resumable.
"""
import argparse
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

B = "/home/umedan/work/BevLane"
PY = "/home/umedan/comet_venv/bin/python3"
ENV = dict(os.environ, BEVLANE_ROOT=f"{B}/out/allroot", OMP_NUM_THREADS="1",
           OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")


def run(cmd):
    r = subprocess.run(cmd, env=ENV, cwd=B, capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr)[-400:]


def convert_scene(scene):
    try:
        prod = f"{B}/out/production/{scene}"
        if not os.path.exists(f"{prod}/bev_label_masked.npy"):
            rc, out = run([PY, "run_batch.py", "--scenes", scene,
                           "--out", "out/production", "--workers", "1"])
            if rc or not os.path.exists(f"{prod}/bev_label_masked.npy"):
                return f"[fail-prod] {scene}: {out[-150:]}"
        if not os.path.exists(f"{prod}/vector_map.json"):
            rc, out = run([PY, "vectorize_bev.py", f"out/production/{scene}"])
            if rc:
                return f"[fail-vec] {scene}: {out[-150:]}"
        stages = [
            [PY, "bevlane/extract_gt.py", "--scenes", scene, "--stride", "2",
             "--workers", "1"],
            [PY, "-c", "import sys,os;sys.path.insert(0,'.');"
             "os.environ['BEVLANE_ROOT']=r'%s/out/allroot';"
             "from bevlane.add_narrow_cams import process_scene;"
             "print(process_scene(('%s',2)))" % (B, scene)],
            [PY, "bevlane/render_vector_gt.py", "--scenes", scene,
             "--stride", "2", "--workers", "1"],
            [PY, "bevlane/extract_depth_dense.py", "--scenes", scene,
             "--stride", "2", "--workers", "1"],
            [PY, "bevlane/extract_depth_narrow.py", "--scenes", scene,
             "--stride", "2", "--workers", "1"],
            [PY, "bevlane/extract_bev_box.py", "--scenes", scene,
             "--stride", "2", "--workers", "1"],
            [PY, "bevlane/annotate_gtcov.py", "--scenes", scene,
             "--workers", "1"],
            [PY, "bevlane/extract_seg2d.py", "--scenes", scene,
             "--stride", "2", "--workers", "1"],
            [PY, "bevlane/extract_bbox2d.py", "--scenes", scene,
             "--stride", "2", "--workers", "1"],
            [PY, "bevlane/extract_ego.py", "--scenes", scene,
             "--stride", "2", "--workers", "1"],
            [PY, "bevlane/extract_occ.py", "--scenes", scene,
             "--stride", "2", "--workers", "1"],
            [PY, "bevlane/extract_agent_traj.py", "--scenes", scene,
             "--stride", "2", "--workers", "1"],
            [PY, "bevlane/extract_tl.py", "--scenes", scene,
             "--stride", "2", "--workers", "1"],
            [PY, "bevlane/extract_risk.py", "--scenes", scene,
             "--workers", "1"],
            [PY, "bevlane/annotate_indoor.py", "--scenes", scene,
             "--workers", "1"],
        ]
        for st in stages:
            rc, out = run(st)
            if rc:
                return f"[fail] {scene} @{st[1].split('/')[-1]}: {out[-150:]}"
        # space hygiene: drop intermediates not needed downstream (~30MB/scene)
        for junk in ("bev_label.npy", "bev_counts.npz"):
            p = os.path.join(prod, junk)
            if os.path.exists(p):
                os.remove(p)
        return f"[ok] {scene}"
    except Exception as e:
        return f"[fail] {scene}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", required=True, help="file with scene names")
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    scenes = open(args.scenes).read().split()
    print(f"{len(scenes)} scenes, {args.workers} workers", flush=True)
    ok = fail = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(convert_scene, scenes)):
            ok += r.startswith("[ok]")
            fail += not r.startswith("[ok]")
            if i % 10 == 0 or not r.startswith("[ok]"):
                print(f"{i + 1}/{len(scenes)} {r}", flush=True)
    print(f"DONE ok={ok} fail={fail}", flush=True)


if __name__ == "__main__":
    main()

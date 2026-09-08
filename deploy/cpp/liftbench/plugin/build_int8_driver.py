#!/usr/bin/env python3
"""Companion-calibrated INT8 build of a MeteorLift-plugin ONNX on the Orin.

Same flow as deploy/orin_build_int8.py (real-frame IInt8EntropyCalibrator2
fed from the fp16 companion's device buffers), plus:
  - loads the plugin .so (RTLD_GLOBAL) before TensorRT parses/builds
  - tolerates pruned unused inputs in either engine at check time

usage (on the Orin, from ~/meteor):
  python3 liftbench/plugin/build_int8_driver.py \
      --onnx out/meteor_v61noref_liftfold.onnx \
      --companion eng/v61_fp16s.engine \
      --out eng/v61noref_liftfold_int8s.engine --calib 64 --check 24
"""
import argparse
import ctypes
import os
import sys

sys.path.insert(0, os.path.expanduser("~/meteor"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--companion", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--roots", nargs="+", default=["calib", "fast"])
    ap.add_argument("--calib", type=int, default=64)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--check", type=int, default=0)
    ap.add_argument("--workspace", type=int, default=8)
    ap.add_argument("--fp16-keep", dest="fp16_keep", default="")
    ap.add_argument("--no-sparse", action="store_true")
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--cams8", action="store_true",
                    help="8-camera rig (CAM_BACK_NARROW appended, r64 line)")
    ap.add_argument("--so",
                    default="liftbench/plugin/build/libmeteor_lift.so")
    a = ap.parse_args()
    a.sparse = not a.no_sparse

    ctypes.CDLL(a.so, mode=ctypes.RTLD_GLOBAL)
    print(f"[plugin] loaded {a.so}", flush=True)

    import numpy as np
    from deploy import runtime as R
    from deploy import orin_build_int8 as OB

    nc = 7
    if a.cams8:
        nc = 8
        if len(OB.CAMS) == 7:
            OB.CAMS = OB.CAMS + ["CAM_BACK_NARROW"]
        print("[driver] 8-camera feed enabled")

    _orig_init = R.MeteorRT.__init__

    def _init(self, *args, **kw):
        _orig_init(self, *args, **kw)
        for nm, shp in (("K", (1, nc, 3, 3)), ("T_cam_ego", (1, nc, 4, 4))):
            if nm not in self.host:
                self.shapes[nm] = shp
                self.host[nm] = R.cuda.pagelocked_empty(
                    int(np.prod(shp)), dtype=np.float32)
                self.dev[nm] = R.cuda.mem_alloc(self.host[nm].nbytes)
                print(f"[driver] '{nm}' pruned from engine; dummy buffer")

    R.MeteorRT.__init__ = _init

    if not a.skip_build:
        if os.path.isfile(a.out + ".calib"):
            os.remove(a.out + ".calib")     # force a REAL recalibration
            print("[driver] removed stale calib cache", flush=True)
        OB.build(a)
    if a.check:
        OB.check(a)


if __name__ == "__main__":
    main()

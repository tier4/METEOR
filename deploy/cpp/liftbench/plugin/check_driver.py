#!/usr/bin/env python3
"""Numerical check of a MeteorLift-plugin engine against the fp16 companion.

Wraps deploy/orin_build_int8.py's check() (same 24-frame protocol: argmax
agreement on lane/seg2d/depth, ego L2, hm sigmoid max|d|) with two additions:
  - loads the plugin .so (RTLD_GLOBAL) before tensorrt deserializes
  - tolerates engines whose unused K / T_cam_ego inputs were pruned by the
    builder (MeteorRT.infer feeds them unconditionally)

usage (on the Orin, from ~/meteor):
  python3 liftbench/plugin/check_driver.py \
      --engine eng/v61_liftplugin_int8s.engine \
      --companion eng/v61_fp16s.engine --frames 24
"""
import argparse
import ctypes
import os
import sys

sys.path.insert(0, os.path.expanduser("~/meteor"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--companion", required=True)
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--roots", nargs="+", default=["calib", "fast"])
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--so",
                    default="liftbench/plugin/build/libmeteor_lift.so")
    a = ap.parse_args()

    if os.path.isfile(a.so):
        ctypes.CDLL(a.so, mode=ctypes.RTLD_GLOBAL)
        print(f"[plugin] loaded {a.so}")

    import numpy as np
    from deploy import runtime as R
    from deploy import orin_build_int8 as OB

    # If the builder pruned the (now unused) K / T_cam_ego inputs, give the
    # runtime harmless landing buffers so its unconditional feed still works.
    _orig_init = R.MeteorRT.__init__

    def _init(self, *args, **kw):
        _orig_init(self, *args, **kw)
        for nm, shp in (("K", (1, 7, 3, 3)), ("T_cam_ego", (1, 7, 4, 4)),
                        ("hist_theta", (1, 3, 2, 3)), ("v0", (1,)),
                        ("imgs", (1, 7, 3, 432, 768))):
            if nm not in self.host:
                self.shapes[nm] = shp
                self.host[nm] = R.cuda.pagelocked_empty(
                    int(np.prod(shp)), dtype=np.float32)
                self.dev[nm] = R.cuda.mem_alloc(self.host[nm].nbytes)
                print(f"[check] engine lacks input '{nm}' "
                      f"(pruned); dummy buffer attached")

    R.MeteorRT.__init__ = _init

    class A:
        pass

    args = A()
    args.companion = a.companion
    args.out = a.engine
    args.roots = a.roots
    # check() strides frame_stream by (stride*2+1) to use held-back frames
    args.stride = a.stride
    args.check = a.frames
    OB.check(args)


if __name__ == "__main__":
    main()

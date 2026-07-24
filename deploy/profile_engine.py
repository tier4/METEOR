#!/usr/bin/env python3
"""Per-module latency profile of a METEOR TensorRT engine.

Builds (once, cached) a DETAILED-verbosity profiling engine from the ONNX,
runs N frames with a layer profiler attached, and aggregates per-layer times
into architecture modules (backbone / BEV / each head). Myelin ForeignNodes
are attributed by the first known token in their fused-op name list.

Usage:
  CUDA_VISIBLE_DEVICES=7 python3 deploy/profile_engine.py \
      --onnx out/meteor_v41.onnx --iters 50
"""
import argparse
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# module attribution: ordered (first match wins)
GROUPS = [
    ("backbone(8cam)", ["stem/", "layer1", "layer2", "layer3", "layer4",
                        "lat1", "lat2", "lat3", "lat4", "/m/fuse"]),
    ("depth-head", ["depth_head"]),
    ("seg2d-head", ["seg2d", "dec/"]),
    ("bev-projection", ["grid_sample", "GridSample", "ipm", "proj"]),
    ("temporal-fuse", ["tfuse", "tgate", "/m/ctx", "lidar_stem",
                       "lid_alpha"]),
    ("bevseg-head", ["seg_head"]),
    ("det3d-head", ["det_stem", "hm_head", "reg_head", "stat_head",
                    "vprof"]),
    ("det2d-head", ["det2d", "hm2d", "reg2d"]),
    ("e2e-head", ["ego_", "dec_head", "dec_gate", "kin_delta",
                  "intent_delta"]),
    ("traj-head", ["traj_", "agent_"]),
    ("lanegraph-head", ["lgdec", "lg_", "lgq", "lg_tower"]),
    ("tl-head", ["tl_"]),
    ("risk-head", ["risk_"]),
    ("occ-head", ["occ_"]),
    ("unknown-head", ["unk_"]),
    ("flow-head", ["flow_"]),
    ("refiner", ["refiner"]),
]


def attribute(name):
    for g, toks in GROUPS:
        if any(t in name for t in toks):
            return g
    return "other/fused-misc"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    import pycuda.autoinit  # noqa: F401
    import pycuda.driver as cuda
    import tensorrt as trt

    prof_eng = os.path.splitext(args.onnx)[0] + "_prof.engine"
    logger = trt.Logger(trt.Logger.WARNING)
    if not (os.path.exists(prof_eng)
            and os.path.getmtime(prof_eng) >= os.path.getmtime(args.onnx)):
        print(f"[build] profiling engine {prof_eng} (~minutes)", flush=True)
        builder = trt.Builder(logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)
        with open(args.onnx, "rb") as f:
            assert parser.parse(f.read()), "parse failed"
        config = builder.create_builder_config()
        # modest workspace: profiling engines are built alongside training
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 3 << 30)
        config.set_flag(trt.BuilderFlag.FP16)
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        blob = builder.build_serialized_network(network, config)
        assert blob is not None, "build failed"
        open(prof_eng, "wb").write(blob)
        print("[build] saved", flush=True)

    rt_ = trt.Runtime(logger)
    engine = rt_.deserialize_cuda_engine(open(prof_eng, "rb").read())
    ctx = engine.create_execution_context()

    class Prof(trt.IProfiler):
        def __init__(self):
            super().__init__()
            self.t = defaultdict(float)
            self.n = 0

        def report_layer_time(self, layer_name, ms):
            self.t[layer_name] += ms

    host, dev = {}, {}
    for i in range(engine.num_io_tensors):
        nm = engine.get_tensor_name(i)
        shp = tuple(engine.get_tensor_shape(nm))
        dt = trt.nptype(engine.get_tensor_dtype(nm))
        host[nm] = cuda.pagelocked_empty(int(np.prod(shp)), dtype=dt)
        host[nm][:] = 0
        dev[nm] = cuda.mem_alloc(host[nm].nbytes)
        cuda.memcpy_htod(dev[nm], host[nm])
        ctx.set_tensor_address(nm, int(dev[nm]))
    stream = cuda.Stream()
    # realistic-ish inputs: random images, identity thetas
    rng = np.random.default_rng(0)
    host["imgs"][:] = rng.standard_normal(host["imgs"].shape[0]) \
        .astype(np.float32)
    cuda.memcpy_htod(dev["imgs"], host["imgs"])
    for _ in range(10):                     # warmup, no profiler
        ctx.execute_async_v3(stream.handle)
    stream.synchronize()
    p = Prof()
    ctx.profiler = p
    for _ in range(args.iters):
        ctx.execute_async_v3(stream.handle)
        stream.synchronize()
        p.n += 1
    agg = defaultdict(float)
    for lname, ms in p.t.items():
        agg[attribute(lname)] += ms / p.n
    tot = sum(agg.values())
    print(f"\n=== per-module latency (avg over {p.n} runs, "
          f"total {tot:.1f} ms) ===")
    for g, ms in sorted(agg.items(), key=lambda x: -x[1]):
        print(f"{g:18s} {ms:7.2f} ms  {100 * ms / tot:5.1f}%")
    # top unattributed layers for transparency
    misc = [(n, ms / p.n) for n, ms in p.t.items()
            if attribute(n) == "other/fused-misc"]
    misc.sort(key=lambda x: -x[1])
    if misc:
        print("\n--- top 'other' layers ---")
        for n, ms in misc[:8]:
            print(f"{ms:7.2f} ms  {n[:120]}")


if __name__ == "__main__":
    main()

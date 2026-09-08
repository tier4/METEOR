#!/usr/bin/env python3
"""Raw per-layer TensorRT profile, classified — no module guessing.

The module table in build_and_bench.py attributes layers by name prefix, which
silently lumps every Myelin ForeignNode whose name starts with an unrecognised
token into "other / reformat". That made a 12 ms block of real compute look
like data-movement overhead. This script classifies by what the layer IS:

  reformat/copy : TensorRT-inserted format conversions and copies -- pure
                  overhead, the thing worth attacking
  foreign       : Myelin-fused blocks -- real compute, reported with the op
                  range they fused so they can be identified
  plain         : ordinary layers

    CUDA_VISIBLE_DEVICES=7 python3 deploy/profile_layers.py \
        --engine out/trt_nolg/meteor_v48_fp16.engine --iters 30
"""
import argparse
import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--top", type=int, default=22)
    a = ap.parse_args()
    import tensorrt as trt

    rt = trt.Runtime(trt.Logger(trt.Logger.ERROR))
    rt.engine_host_code_allowed = True
    eng = rt.deserialize_cuda_engine(open(a.engine, "rb").read())
    ctx = eng.create_execution_context()
    bufs = []
    for i in range(eng.num_io_tensors):
        n = eng.get_tensor_name(i)
        shp = tuple(eng.get_tensor_shape(n))
        dt = {"DataType.FLOAT": torch.float32, "DataType.HALF": torch.float16,
              "DataType.INT32": torch.int32,
              "DataType.INT8": torch.int8}[str(eng.get_tensor_dtype(n))]
        t = torch.zeros(*shp, dtype=dt, device="cuda")
        bufs.append(t)
        ctx.set_tensor_address(n, int(t.data_ptr()))

    rows = {}

    class P(trt.IProfiler):
        def __init__(s):
            trt.IProfiler.__init__(s)

        def report_layer_time(s, name, ms):
            r = rows.setdefault(name, [0.0, 0])
            r[0] += ms
            r[1] += 1

    ptrs = [int(t.data_ptr()) for t in bufs]
    for _ in range(10):
        ctx.execute_v2(ptrs)
    ctx.profiler = P()
    for _ in range(a.iters):
        ctx.execute_v2(ptrs)
    ctx.profiler = None

    def cls(n):
        if "Reformat" in n or "copy" in n.lower() or "CopyNode" in n:
            return "reformat/copy"
        if "ForeignNode" in n:
            return "foreign (fused compute)"
        return "plain layer"

    per = {}
    for n, (tot, cnt) in rows.items():
        k = cls(n)
        e = per.setdefault(k, [0.0, 0])
        e[0] += tot / a.iters
        e[1] += 1
    total = sum(v[0] for v in per.values())
    print(f"\n{os.path.basename(a.engine)}  —  {total:.2f} ms of layer time "
          f"over {a.iters} runs, {len(rows)} layers")
    print(f"\n{'category':26s} {'ms':>7s} {'%':>6s} {'layers':>7s}")
    for k, (ms, n) in sorted(per.items(), key=lambda kv: -kv[1][0]):
        print(f"{k:26s} {ms:7.2f} {100 * ms / total:5.1f}% {n:7d}")

    print(f"\ntop {a.top} layers")
    for n, (tot, cnt) in sorted(rows.items(), key=lambda kv: -kv[1][0])[:a.top]:
        ms = tot / a.iters
        tag = {"reformat/copy": "COPY", "foreign (fused compute)": "FUSED",
               "plain layer": "    "}[cls(n)]
        # for a fused node, show the first and last op it swallowed
        m = re.match(r"\{ForeignNode\[(.+?)\.\.\.(.+?)\]\}", n)
        label = f"{m.group(1)}  ...  {m.group(2)}" if m else n
        print(f"  {tag} {ms:7.3f} ms  {label[:118]}")

    rf = [(n, tot / a.iters) for n, (tot, cnt) in rows.items()
          if cls(n) == "reformat/copy"]
    if rf:
        print(f"\nevery reformat/copy layer ({sum(v for _, v in rf):.2f} ms)")
        for n, ms in sorted(rf, key=lambda kv: -kv[1]):
            print(f"  {ms:7.3f} ms  {n[:112]}")


if __name__ == "__main__":
    main()

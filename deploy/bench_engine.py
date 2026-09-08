#!/usr/bin/env python3
"""Time an existing TensorRT engine. No rebuild, no calibration.

Needed because build_and_bench.py only reports latency as part of a build, and
one of those numbers (34.7 ms) was taken while a demo was rendering on the same
GPU -- the timing loop was contended and the figure was meaningless. This runs
alone, times with CUDA events, and reports the spread so contention is visible.

    CUDA_VISIBLE_DEVICES=7 python3 deploy/bench_engine.py --engine E [E ...]
"""
import argparse
import os

import numpy as np
import torch


def bench(path, iters, warmup):
    import tensorrt as trt
    rt = trt.Runtime(trt.Logger(trt.Logger.ERROR))
    rt.engine_host_code_allowed = True
    eng = rt.deserialize_cuda_engine(open(path, "rb").read())
    ctx = eng.create_execution_context()
    bufs = []
    for i in range(eng.num_io_tensors):
        n = eng.get_tensor_name(i)
        dt = {"DataType.FLOAT": torch.float32, "DataType.HALF": torch.float16,
              "DataType.INT32": torch.int32, "DataType.INT8": torch.int8,
              "DataType.BOOL": torch.bool}[str(eng.get_tensor_dtype(n))]
        t = torch.zeros(*tuple(eng.get_tensor_shape(n)), dtype=dt, device="cuda")
        bufs.append(t)
        ctx.set_tensor_address(n, int(t.data_ptr()))
    ptrs = [int(t.data_ptr()) for t in bufs]
    for _ in range(warmup):
        ctx.execute_v2(ptrs)
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        ctx.execute_v2(ptrs)
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return np.array(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", nargs="+", required=True)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    a = ap.parse_args()
    print(f"{'engine':40s} {'mean':>8s} {'p50':>8s} {'p90':>8s} "
          f"{'min':>8s} {'FPS':>6s}")
    for e in a.engine:
        if not os.path.exists(e):
            print(f"{os.path.basename(e):40s}   見つかりません")
            continue
        t = bench(e, a.iters, a.warmup)
        print(f"{os.path.basename(os.path.dirname(e)) + '/' + os.path.basename(e):40s} "
              f"{t.mean():7.2f} {np.percentile(t,50):7.2f} "
              f"{np.percentile(t,90):7.2f} {t.min():7.2f} {1000/t.mean():6.1f}")


if __name__ == "__main__":
    main()

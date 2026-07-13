#!/usr/bin/env python3
"""Export IPMSegNet to ONNX, build TensorRT fp16 / INT8 engines, benchmark.

INT8 uses entropy calibration over real val samples (images + calib tensors).
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bevlane.dataset import BevLaneDataset  # noqa: E402
from bevlane.model import MODELS  # noqa: E402
from bevlane.train import split_scenes  # noqa: E402

SHAPES = {"imgs": (1, 8, 3, 288, 512), "K": (1, 8, 3, 3), "T": (1, 8, 4, 4)}


def export_onnx(model, path):
    n = SHAPES["imgs"][1]
    dummy = (torch.randn(*SHAPES["imgs"]), torch.eye(3).view(1, 1, 3, 3)
             .repeat(1, n, 1, 1) * torch.tensor([224., 195., 1.]).view(3, 1),
             torch.eye(4).view(1, 1, 4, 4).repeat(1, n, 1, 1))
    torch.onnx.export(model, dummy, path, opset_version=17,
                      input_names=["imgs", "K", "T"], output_names=["logits", "depth"] if "v8" in path or "lss" in path else ["logits"],
                      do_constant_folding=True)
    print(f"[onnx] {path} ({os.path.getsize(path) / 1e6:.1f} MB)", flush=True)


class Calibrator:
    """Entropy calibrator streaming real samples."""

    def __init__(self, samples, cache):
        import tensorrt as trt

        class _C(trt.IInt8EntropyCalibrator2):
            def __init__(self, samples, cache):
                super().__init__()
                self.samples = samples
                self.i = 0
                self.cache = cache
                self.bufs = {k: torch.zeros(*v, device="cuda")
                             for k, v in SHAPES.items()}

            def get_batch_size(self):
                return 1

            def get_batch(self, names):
                if self.i >= len(self.samples):
                    return None
                imgs, K, T = self.samples[self.i]
                self.i += 1
                self.bufs["imgs"].copy_(imgs)
                self.bufs["K"].copy_(K)
                self.bufs["T"].copy_(T)
                return [int(self.bufs[n].data_ptr()) for n in names]

            def read_calibration_cache(self):
                if os.path.exists(self.cache):
                    return open(self.cache, "rb").read()

            def write_calibration_cache(self, c):
                open(self.cache, "wb").write(c)

        self.impl = _C(samples, cache)


def build_engine(onnx_path, engine_path, int8_samples=None):
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(open(onnx_path, "rb").read()):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise RuntimeError("onnx parse failed")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    config.set_flag(trt.BuilderFlag.FP16)
    if int8_samples is not None:
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = Calibrator(
            int8_samples, engine_path + ".calib").impl
    engine = builder.build_serialized_network(network, config)
    open(engine_path, "wb").write(engine)
    print(f"[engine] {engine_path} ({os.path.getsize(engine_path) / 1e6:.1f} MB)",
          flush=True)


def bench_engine(engine_path, n=100):
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    rt = trt.Runtime(logger)
    engine = rt.deserialize_cuda_engine(open(engine_path, "rb").read())
    ctx = engine.create_execution_context()
    bufs, out = {}, None
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = tuple(engine.get_tensor_shape(name))
        dt = {"DataType.FLOAT": torch.float32, "DataType.HALF": torch.float16,
              "DataType.INT32": torch.int32}[str(engine.get_tensor_dtype(name))]
        t = torch.zeros(*shape, dtype=dt, device="cuda")
        bufs[name] = t
        ctx.set_tensor_address(name, int(t.data_ptr()))
        if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
            if name == "logits" or out is None:
                out = t
    stream = torch.cuda.Stream()
    for _ in range(10):
        ctx.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    t0 = time.time()
    for _ in range(n):
        ctx.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    ms = (time.time() - t0) / n * 1000
    return ms, bufs, ctx, out, stream


def engine_ious(engine_path, samples, gts):
    """IoU of engine predictions against GT for parity checking."""
    import tensorrt as trt  # noqa: F401
    ms, bufs, ctx, out, stream = bench_engine(engine_path, n=1)
    inter = np.zeros(9)
    union = np.zeros(9)
    for (imgs, K, T), gt in zip(samples, gts):
        bufs["imgs"].copy_(imgs)
        bufs["K"].copy_(K)
        bufs["T"].copy_(T)
        ctx.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        pred = out.float().argmax(1)[0].cpu().numpy()
        g = gt.numpy()
        if pred.shape != g.shape:
            continue
        m = g > 0
        for c in range(1, 9):
            pi, gi = (pred == c) & m, g == c
            inter[c] += (pi & gi).sum()
            union[c] += (pi | gi).sum()
    return {c: inter[c] / union[c] for c in range(1, 9) if union[c]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", default="v2")
    ap.add_argument("--out", default="out/trt")
    ap.add_argument("--calib-n", type=int, default=128)
    ap.add_argument("--eval-n", type=int, default=64)
    ap.add_argument("--root", default="out/bevlane")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    model = MODELS[args.model]().eval()
    if os.path.exists(args.ckpt):
        model.load_state_dict(torch.load(args.ckpt, map_location="cpu")["model"])
        print(f"[ckpt] {args.ckpt}", flush=True)
    onnx_path = os.path.join(args.out, f"bevlane_{args.model}.onnx")
    export_onnx(model, onnx_path)

    # real samples for calibration / parity
    _, val_s = split_scenes(args.root)
    ds = BevLaneDataset(args.root, val_s[:40], max_per_scene=4, gt_key="gt_vec")
    step = max(1, len(ds) // max(args.calib_n, args.eval_n))
    samples, gts = [], []
    for i in range(0, len(ds), step):
        imgs, K, T, gt = ds[i]
        samples.append((imgs[None], K[None], T[None]))
        gts.append(gt)
        if len(samples) >= args.calib_n:
            break
    print(f"[calib] {len(samples)} samples", flush=True)

    fp16_path = os.path.join(args.out, f"bevlane_{args.model}_fp16.engine")
    build_engine(onnx_path, fp16_path)
    ms16, *_ = bench_engine(fp16_path)
    print(f"[bench] fp16: {ms16:.2f} ms", flush=True)

    int8_path = os.path.join(args.out, f"bevlane_{args.model}_int8.engine")
    build_engine(onnx_path, int8_path, int8_samples=samples)
    ms8, *_ = bench_engine(int8_path)
    print(f"[bench] int8: {ms8:.2f} ms", flush=True)

    ev = samples[:args.eval_n]
    evg = gts[:args.eval_n]
    iou16 = engine_ious(fp16_path, ev, evg)
    iou8 = engine_ious(int8_path, ev, evg)
    names = ["", "road", "sidewalk", "crosswalk", "laneline", "stopline",
             "road_edge", "marking", "parking"]
    print("[parity] class fp16 int8")
    for c in sorted(iou16):
        print(f"  {names[c]:10s} {iou16[c]:.3f} {iou8.get(c, float('nan')):.3f}",
              flush=True)


if __name__ == "__main__":
    main()

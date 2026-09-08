#!/usr/bin/env python3
"""Build paired TensorRT engines to measure whether camera FiLM fuses into Conv."""
from pathlib import Path
import sys

import numpy as np
import tensorrt as trt


LOG = trt.Logger(trt.Logger.WARNING)
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
OUT.mkdir(parents=True, exist_ok=True)


def build(path: Path, with_film: bool) -> None:
    builder = trt.Builder(LOG)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    net = builder.create_network(flags)
    cfg = builder.create_builder_config()
    cfg.set_flag(trt.BuilderFlag.FP16)
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)

    x = net.add_input("f", trt.float16, (8, 160, 108, 192))
    t = x
    if with_film:
        # Camera-specific constants make this stricter than a batch-shared
        # channel scale. Zero values retain numerical identity at build time.
        gamma = np.zeros((8, 160, 1, 1), dtype=np.float16)
        beta = np.zeros((8, 160, 1, 1), dtype=np.float16)
        one = np.ones((8, 160, 1, 1), dtype=np.float16)
        g = net.add_constant(gamma.shape, gamma).get_output(0)
        b = net.add_constant(beta.shape, beta).get_output(0)
        o = net.add_constant(one.shape, one).get_output(0)
        scale = net.add_elementwise(g, o, trt.ElementWiseOperation.SUM).get_output(0)
        t = net.add_elementwise(t, scale, trt.ElementWiseOperation.PROD).get_output(0)
        t = net.add_elementwise(t, b, trt.ElementWiseOperation.SUM).get_output(0)

    # First depth/context-style 3x3 tower layer.
    rng = np.random.default_rng(7)
    weight = (rng.standard_normal((256, 160, 3, 3)) * 0.01).astype(np.float32)
    bias = np.zeros(256, dtype=np.float32)
    conv = net.add_convolution_nd(t, 256, (3, 3), weight, bias)
    conv.padding_nd = (1, 1)
    y = conv.get_output(0)
    y.name = "y"
    net.mark_output(y)

    blob = builder.build_serialized_network(net, cfg)
    if blob is None:
        raise RuntimeError(f"TensorRT build failed: {path}")
    path.write_bytes(blob)
    print(path, path.stat().st_size)


build(OUT / "conv_base_fp16.engine", False)
build(OUT / "conv_film_fp16.engine", True)

#!/usr/bin/env python3
"""NPU portability lint for an ONNX graph (METEORite P0, 2026-10-09).

Tier A = operators that practically every NPU compiler maps natively. Tier B = common but
vendor-dependent (constant-index Gather, ArgMax, Div, LayerNorm, ...). Anything else, dynamic
shapes, data-dependent indices or extra graph inputs fail the lint.

usage: python deploy/npu_lint.py model.onnx [--tier A|B] [--target npu|conv1x1]
"""
import argparse
import collections
import sys

import onnx

TIER_A = {"Conv", "Relu", "Add", "Sub", "Mul", "Sigmoid", "Tanh", "Softmax", "MatMul", "Gemm",
          "Transpose", "Reshape", "Concat", "Slice", "Resize", "AveragePool", "MaxPool",
          "GlobalAveragePool", "Clip", "ReduceSum", "ReduceMean", "Flatten", "Cast", "Constant",
          "Unsqueeze", "Squeeze", "BatchNormalization", "Identity", "Pad"}
TIER_B = {"Gather", "ArgMax", "Div", "Reciprocal", "LayerNormalization", "GridSample", "Expand",
          "Exp", "Log", "Sqrt", "Erf", "Split", "ReduceMax", "Neg", "Abs", "Pow", "ConvTranspose",
          "LeakyRelu", "HardSwish", "Where"}
FORBIDDEN_HINT = {"ScatterND": "lift index_add_ (to be replaced by a plugin)", "GatherElements": "runtime index",
                  "Shape": "dynamic-shape plumbing (from K/T inputs)", "Range": "dynamic index", "NonZero": "data-dependent",
                  "TopK": "data-dependent", "CumSum": "kinematic anchor (use a constant lower-triangular MatMul)", "Sin": "trigonometric",
                  "Cos": "trigonometric", "Atan": "trigonometric", "IsNaN": "NaN guard", "IsInf": "NaN guard",
                  "Loop": "control flow", "If": "control flow", "ConstantOfShape": "dynamic shape"}


# conv1x1 target: a Conv-centric edge accelerator paired with a GPU. No MatMul / Gather / Cast;
# Conv kernel <=32 dilation <=32, Pool kernel <=8 stride <=16 pad <=7, Concat on the channel axis only, integer Resize,
# Softmax on the channel axis, every tensor dim <=8192. Unsupported layers fall back to the GPU (more accelerator<->GPU transitions).
CONV1X1_OPS = {"Conv", "ConvTranspose", "Relu", "Sigmoid", "Tanh", "Clip", "LeakyRelu", "Add", "Sub", "Mul", "Max", "Min",
           "Div", "Pow", "AveragePool", "MaxPool", "Softmax", "Concat", "Resize", "Transpose", "Reshape", "Flatten",
           "Unsqueeze", "Squeeze", "Identity", "Constant", "BatchNormalization", "Gemm", "ReduceMax", "ReduceSum",
           "Slice", "Abs", "Sin", "Cos", "Floor", "Ceil", "Equal", "Greater", "Less"}
CONV1X1_GPU_ONLY = {"MatMul": "no MatMul on the accelerator (1x1 Conv if one side is constant, GPU if both are dynamic)", "Gather": "no Gather on the accelerator (GPU)",
                "Cast": "uint8->fp Cast is GPU preprocessing (accelerator input is fp16/int8)", "ReduceMean": "GPU", "Where": "GPU",
                "GridSample": "GPU", "ScatterND": "GPU", "LayerNormalization": "GPU", "Exp": "GPU", "Log": "GPU", "Sqrt": "GPU"}


def _attr(n, name, default=None):
    for a in n.attribute:
        if a.name == name:
            return list(a.ints) if a.type == onnx.AttributeProto.INTS else a.i
    return default


def lint_conv1x1(path):
    """Offline dry run: decide per layer whether it fits the accelerator, and count GPU fallbacks and accelerator<->GPU transitions."""
    m = onnx.load(path, load_external_data=False)
    g = m.graph
    inits = {i.name: i for i in g.initializer}
    gpu, why = [], collections.Counter()
    for n in g.node:
        if n.op_type == "Constant":
            continue
        r = None
        if n.op_type in CONV1X1_GPU_ONLY:
            r = CONV1X1_GPU_ONLY[n.op_type]
        elif n.op_type not in CONV1X1_OPS:
            r = "operator not supported on the accelerator"
        elif n.op_type == "Conv":
            k = _attr(n, "kernel_shape", [1]); d = _attr(n, "dilations", [1])
            if max(k) > 32 or max(d) > 32:
                r = f"Conv kernel {k} / dilation {d} exceeds the limit of 32"
        elif n.op_type in ("AveragePool", "MaxPool"):
            k = _attr(n, "kernel_shape", [1]); s = _attr(n, "strides", [1]); p = _attr(n, "pads", [0])
            if max(k) > 8 or max(s) > 16 or max(p) > 7:
                r = f"Pool kernel {k} > 8 (stride {s}, pad {p}): split into two stages"
        elif n.op_type == "Softmax":
            ax = _attr(n, "axis", -1)
            if ax not in (1,):
                r = f"Softmax axis={ax}: use the 4-D channel axis (axis=1)"
        elif n.op_type == "Concat":
            if _attr(n, "axis", 0) not in (1, -3):
                r = f"Concat axis={_attr(n, 'axis', 0)}: channel axis only"
        elif n.op_type == "Resize":
            mode = [a.s.decode() for a in n.attribute if a.name == "mode"]
            if mode and mode[0] not in ("nearest", "linear"):
                r = f"Resize mode {mode[0]}"
        if r:
            gpu.append((n.name or n.output[0], n.op_type, r)); why[f"{n.op_type}: {r}"] += 1
    # transitions: count ACC/GPU switches in node order (approximates topological order)
    seq = [("GPU" if (n.name or n.output[0]) in {x[0] for x in gpu} else "ACC") for n in g.node if n.op_type != "Constant"]
    trans = sum(1 for a, b in zip(seq, seq[1:]) if a != b)
    n_all = len(seq)
    print(f"== {path}  [conv1x1 accelerator dry run]")
    print(f"layers {n_all}: accelerator {n_all - len(gpu)}, GPU fallback {len(gpu)}, accelerator<->GPU transitions ~{trans}")
    for k, v in sorted(why.items(), key=lambda x: -x[1]):
        print(f"  GPU x{v}: {k}")
    return 0 if not gpu else 1


def lint(path, tier="A"):
    m = onnx.load(path, load_external_data=False)
    g = m.graph
    inits = {i.name for i in g.initializer} | {o for n in g.node if n.op_type == "Constant" for o in n.output}
    hist = collections.Counter(n.op_type for n in g.node)
    allowed = TIER_A | (TIER_B if tier == "B" else set())
    problems = []
    for op, k in sorted(hist.items(), key=lambda x: -x[1]):
        if op not in allowed:
            tag = "Tier B" if op in TIER_B else "forbidden"
            problems.append(f"{tag} operator {op} x{k}" + (f" — {FORBIDDEN_HINT[op]}" if op in FORBIDDEN_HINT else ""))
    for i in g.input:
        dims = [d.dim_value for d in i.type.tensor_type.shape.dim]
        if not all(dims):
            problems.append(f"input {i.name} has dynamic dims {dims}")
    for n in g.node:
        if n.op_type == "Gather" and n.input[1] not in inits:
            problems.append(f"Gather {n.name}: index is not constant")
        if n.op_type == "GridSample" and n.input[1] not in inits:
            problems.append(f"GridSample {n.name}: grid is not constant")
        if n.op_type == "Resize":
            scales = n.input[2] if len(n.input) > 2 else ""
            if scales and scales not in inits:
                problems.append(f"Resize {n.name}: scales are not constant")
        if n.op_type == "Softmax":
            ax = [a.i for a in n.attribute if a.name == "axis"]
            if ax and ax[0] not in (-1,):
                problems.append(f"Softmax {n.name}: axis={ax[0]} (not the last axis; many NPUs only support the last axis)")
    print(f"== {path}")
    print(f"nodes={len(g.node)} inputs={[i.name for i in g.input]} outputs={len(g.output)}")
    print("ops:", ", ".join(f"{op} {k}" for op, k in sorted(hist.items(), key=lambda x: -x[1])))
    if problems:
        print(f"NPU lint (Tier {tier}): FAIL")
        for p in problems:
            print("  -", p)
        return 1
    print(f"NPU lint (Tier {tier}): PASS")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx")
    ap.add_argument("--tier", default="A", choices=["A", "B"])
    ap.add_argument("--target", default="npu", choices=["npu", "conv1x1"], help="conv1x1: offline dry run against Conv-centric accelerator layer restrictions")
    a = ap.parse_args()
    sys.exit(lint_conv1x1(a.onnx) if a.target == "conv1x1" else lint(a.onnx, a.tier))

#!/usr/bin/env python3
"""Build fp16 / INT8 engines from a METEOR ONNX and measure both.

Reports, per configuration: build time, engine size, latency (mean/p50/p90),
throughput, and BEV-seg IoU + E2E ADE against the SAME val frames the PyTorch
model is scored on, so a speed win can be read next to what it costs.

INT8 uses entropy calibration over real val frames (images + K/T + the temporal
inputs), never random tensors: a calibrator fed noise picks activation ranges
that do not exist in the data.

    CUDA_VISIBLE_DEVICES=7 python3 deploy/build_and_bench.py \
        --onnx out/meteor_v48.onnx --ckpt out/bevlane_ckpt_r48/last.pt \
        --calib 64 --eval 40 --iters 100
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bevlane.dataset import BevLaneDataset                       # noqa: E402
from bevlane.model import BEV_H, BEV_W                           # noqa: E402

HIST_N = 3
N_CLASSES = 9


def _denorm_u8(t):
    """正規化済み float 画像 [1,N,3,H,W] -> 生の uint8 画素。

    uint8-in グラフは正規化を自分の中に持つ。正規化済みテンソルを uint8 へ
    キャストすると値がほぼ 0/255 に潰れ、BEV Seg は「それらしく」出るのに
    3D 検出だけ静かに壊れる (orin_render.py で 2026-08-24 に実害)。"""
    import numpy as _np
    from bevlane.dataset import MEAN as _M, STD as _S
    mm = torch.tensor(_np.asarray(_M), dtype=t.dtype).view(1, 1, 3, 1, 1)
    ss = torch.tensor(_np.asarray(_S), dtype=t.dtype).view(1, 1, 3, 1, 1)
    return ((t * ss + mm) * 255.0).round().clamp(0, 255).to(torch.uint8)


def sample_inputs(root, val_list, n, hist_ckpt=None, hist_model="v50",
                  n_seg2d=21):
    """Real frames in the engine's input layout.

    hist_bev used to be handed over as zeros, which is wrong for INT8 in two
    ways. It made the partial-INT8 build fail outright -- every layer fed only
    by the history saw an identically-zero activation over all calibration
    batches, so its scale came out 0 and bias quantisation tripped
    "weightConvertors.cpp::quantizeBiasCommon::310: Assertion getter(i) != 0".
    And more quietly, it means the temporal-fusion branch of the full-INT8
    engine was calibrated against a tensor it never sees at runtime. With
    --hist-ckpt the history is the model's own BEV feature for the same frame,
    which at 10 Hz is what the previous frame actually supplies.
    """
    scenes = [l.strip() for l in open(val_list) if l.strip()][:8]
    ds = BevLaneDataset(root, scenes, gt_key="gt_cons", max_per_scene=8,
                        with_ego=True)
    net = None
    if hist_ckpt:
        from bevlane.model import MODELS
        from bevlane.ckpt_load import load_net
        net = MODELS[hist_model](n_seg=n_seg2d).cuda().eval()
        # ckpt_load 経由で depth-slim / sem_ego / delta-stat 等を自動検出
        # (素の v52 への strict なし load は slim ckpt で形状不一致に落ちる)
        load_net(net, hist_ckpt)
        net = net.cuda().eval()
    st = max(1, len(ds) // n)
    out = []
    for i in list(range(0, len(ds), st))[:n]:
        b = ds[i]
        if b is None:
            continue
        if net is not None:
            with torch.no_grad(), torch.autocast("cuda", torch.float16):
                net(b[0][None].cuda(), b[1][None].cuda(), b[2][None].cuda())
            hb = net._last_bev.detach().float().cpu()
            hist = hb.unsqueeze(1).repeat(1, HIST_N, 1, 1, 1).contiguous()
        else:
            hist = torch.zeros(1, HIST_N, 96, BEV_H, BEV_W)
        out.append(dict(
            imgs=b[0][None].contiguous(), K=b[1][None].contiguous(),
            T=b[2][None].contiguous(),
            v0=b[4][12].view(1).float().contiguous(),
            hist_bev=hist,
            hist_theta=torch.tensor([[[1., 0., 0.], [0., 1., 0.]]])
            .repeat(HIST_N, 1, 1)[None].contiguous(),
            gt=b[3], ego=b[4]))
    if net is not None:
        del net
        torch.cuda.empty_cache()
    return out


# engine input name -> key in the sample dict. The first version keyed the
# calibration buffers by the SAMPLE names, so TensorRT asking for
# "T_cam_ego" got a null pointer back and calibration died inside
# executeV2 with "Assertion context->executeV2(&bindings[0]) failed".
NAME2KEY = {"imgs": "imgs", "K": "K", "T_cam_ego": "T", "v0": "v0",
            "hist_bev": "hist_bev", "hist_theta": "hist_theta"}


class Calib:
    def __init__(self, samples, cache, in_dtypes=None):
        _in_dtypes = dict(in_dtypes or {})
        import tensorrt as trt

        class _C(trt.IInt8EntropyCalibrator2):
            def __init__(s):
                trt.IInt8EntropyCalibrator2.__init__(s)
                s.samples = samples
                s.i = 0
                s.cache = cache
                s.bufs = {}          # keyed by ENGINE input name

            def get_batch_size(s):
                return 1

            def get_batch(s, names):
                if s.i >= len(s.samples):
                    return None
                smp = s.samples[s.i]
                s.i += 1
                out = []
                for n in names:
                    key = NAME2KEY.get(n)
                    if key is None or key not in smp:
                        raise RuntimeError(
                            f"calibrator has no data for engine input '{n}' "
                            f"(known: {sorted(NAME2KEY)})")
                    src = smp[key]
                    if key == "imgs" and _in_dtypes.get(n) == "DataType.UINT8":
                        src = _denorm_u8(src)
                    if n not in s.bufs:
                        s.bufs[n] = src.cuda().contiguous().clone()
                    s.bufs[n].copy_(src.cuda())
                    out.append(int(s.bufs[n].data_ptr()))
                if s.i == 1:
                    print(f"[calib] bindings {list(names)}", flush=True)
                return out

            def read_calibration_cache(s):
                if os.path.exists(s.cache):
                    return open(s.cache, "rb").read()
                return None

            def write_calibration_cache(s, c):
                open(s.cache, "wb").write(c)

        self.impl = _C()


def is_ampere_plus():
    """True when every visible GPU is SM 8.0+ (what AMPERE_PLUS covers)."""
    import torch
    if not torch.cuda.is_available():
        return False
    return all(torch.cuda.get_device_capability(i)[0] >= 8
               for i in range(torch.cuda.device_count()))


# Substrings of layer names that stay in fp16 when building a partial-INT8
# engine. Chosen from what the full-INT8 build actually complained about:
#   - the built-in refiner's E2E path produced dozens of
#     "Missing scale and zero-point for tensor /net/refiner/e2e/..." warnings,
#     fell back to fp16 anyway, and dragged a quantisation boundary with it
#   - Scatter/Gather is the frustum lift's index_add_ / index_select: there is
#     nothing to gain from quantising data movement, and every boundary costs a
#     reformat
#   - GridSample interpolates; INT8 sampling of a probability volume is where
#     accuracy goes first
# Pinning them also avoids the reformat the full-INT8 build died on
# ("Assertion w != 0.F failed" in reformatBuilder).
FP16_KEEP = ("refiner/e2e", "Scatter", "Gather", "GridSample", "grid_sample",
             "ScatterND", "ScatterElements")


def build(onnx, engine, int8_samples=None, workspace_gb=8,
          portable=False, direct_io=False, strict_types=False,
          fp16_keep=(), sparse=False):
    # workspace_gb is the builder's scratch limit, not the runtime cost. On a
    # small card it must come down or the build fails outright: an 8 GB laptop
    # GPU cannot hand TensorRT an 8 GiB workspace. 2 GiB is enough for this
    # graph (measured runtime need: 1.5 GiB of activation workspace).
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    net = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(net, logger)
    if not parser.parse(open(onnx, "rb").read()):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise RuntimeError("onnx parse failed")
    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,
                              workspace_gb << 30)
    cfg.set_flag(trt.BuilderFlag.FP16)
    if direct_io:
        # Measured on the fp16 engine: 7.19 ms of the 49.4 ms of layer time is
        # TensorRT-inserted reformatting, and 2.2 ms of that is two copies of
        # the 96x800x500 BEV tensor between the lift's fused node and tfuse3.
        # DIRECT_IO forbids reformats on the ENGINE BOUNDARY (the host must hand
        # over tensors in the engine's preferred layout); PREFER_PRECISION_
        # CONSTRAINTS keeps TensorRT from inserting casts it does not need.
        cfg.set_flag(trt.BuilderFlag.DIRECT_IO)
    if strict_types:
        cfg.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
    if sparse:
        # 2:4 structured sparsity on the tensor cores. The flag alone changes
        # nothing -- TensorRT checks each weight tensor for the pattern and
        # falls back to the dense kernel where it is absent -- so this is only
        # worth setting on weights that were pruned to the pattern. Both
        # targets have the hardware: the workstation GPU is SM 8.9 (Ada), AGX Orin is SM 8.7.
        cfg.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)
    tag = "fp16"
    if portable:
        # Version- and hardware-compatible plan: the engine carries a lean
        # runtime and is built for the Ampere-and-newer instruction set, so the
        # same file loads on another machine and another TensorRT minor version
        # instead of only on the card it was built on. Same pair of settings
        # CoMET's tensorrt_common.cpp uses (kVERSION_COMPATIBLE +
        # kAMPERE_PLUS), gated the same way on SM >= 8.0.
        if not is_ampere_plus():
            raise RuntimeError("--portable needs SM 8.0+ (Ampere or newer)")
        cfg.set_flag(trt.BuilderFlag.VERSION_COMPATIBLE)
        cfg.hardware_compatibility_level = \
            trt.HardwareCompatibilityLevel.AMPERE_PLUS
        tag += "+portable"
    if int8_samples is not None:
        cfg.set_flag(trt.BuilderFlag.INT8)
        _idt = {net.get_input(i).name: str(net.get_input(i).dtype)
                for i in range(net.num_inputs)}
        cfg.int8_calibrator = Calib(int8_samples, engine + ".calib",
                                    in_dtypes=_idt).impl
        tag = "int8+fp16"
        # A layer whose weights are ALL zero has a per-channel weight scale of
        # zero, and bias quantisation divides by it:
        # "weightConvertors.cpp::quantizeBiasCommon::310: Assertion
        # getter(i) != 0". This model contains such layers on purpose -- the
        # optional-input branches are zero-initialised so that feeding zeros is
        # bit-identical to not having the input at all -- and any of them that a
        # given checkpoint never trained is still exactly zero at export time.
        # (Found this way: the calibration cache had no zero-scale activation,
        # so the zero had to be in the weights; the ONNX scan named
        # depth_head.sig, which the distilled checkpoint predates.) Such a layer
        # outputs a constant, so fp16 costs nothing.
        n_zero = 0
        # net.get_layer returns a base ILayer, which has no .kernel: the weights
        # are only reachable after re-classing to the concrete layer type.
        _WEIGHTED = {}
        for _tn, _cn in (("CONVOLUTION", "IConvolutionLayer"),
                         ("DECONVOLUTION", "IDeconvolutionLayer"),
                         ("FULLY_CONNECTED", "IFullyConnectedLayer")):
            if hasattr(trt.LayerType, _tn) and hasattr(trt, _cn):
                _WEIGHTED[getattr(trt.LayerType, _tn)] = getattr(trt, _cn)
        for i in range(net.num_layers):
            lay = net.get_layer(i)
            cls = _WEIGHTED.get(lay.type)
            if cls is None:
                continue
            try:
                lay.__class__ = cls
                arr = np.array(lay.kernel)
            except Exception:
                continue
            if arr.size and float(np.abs(arr).max()) == 0.0:
                lay.precision = trt.float16
                n_zero += 1
        if n_zero:
            print(f"[int8] {n_zero} all-zero-weight layer(s) pinned to fp16 "
                  f"(their INT8 weight scale would be 0)", flush=True)
        if fp16_keep:
            cfg.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
            n_pin = n_skip = 0
            for i in range(net.num_layers):
                lay = net.get_layer(i)
                if not any(k in lay.name for k in fp16_keep):
                    continue
                outs = [lay.get_output(j) for j in range(lay.num_outputs)]
                # A shape tensor MUST stay INT32 -- forcing fp16 on one is what
                # killed this build: "shapeLayer.h::setOutputType::46,
                # condition: dataType == DataType::kINT32". The FP16_KEEP names
                # are substrings ("Gather", "Concat"), and ONNX gives the same
                # names to the layers that compute shapes as to the ones that
                # move data, so the keep-list has to be filtered by what the
                # output actually IS, not by what it is called.
                if any(o is None or o.is_shape_tensor
                       or o.dtype not in (trt.float32, trt.float16)
                       for o in outs):
                    n_skip += 1
                    continue
                lay.precision = trt.float16
                # Some layer types already carry their output type in their own
                # parameters, and re-declaring it is an error even when the type
                # agrees: "castLayer.cpp::validate::34, Assertion
                # !mOutputTypes.at(0).hasValue() || ... == params.toType". For
                # those, the compute-precision constraint alone keeps the layer
                # out of INT8, which is all we need.
                _fixed = {getattr(trt.LayerType, _n) for _n in
                          ("CAST", "CONSTANT", "SHAPE", "TOPK", "NON_ZERO",
                           "IDENTITY") if hasattr(trt.LayerType, _n)}
                if lay.type not in _fixed:
                    for j in range(len(outs)):
                        lay.set_output_type(j, trt.float16)
                n_pin += 1
            print(f"[partial-int8] {n_pin} of {net.num_layers} layers pinned "
                  f"to fp16 ({', '.join(fp16_keep)}); {n_skip} name matches "
                  f"skipped as shape/integer layers", flush=True)
            tag = "partial-int8"
    t0 = time.time()
    ser = builder.build_serialized_network(net, cfg)
    if ser is None:
        raise RuntimeError(f"{tag} build failed")
    open(engine, "wb").write(ser)
    dt = time.time() - t0
    print(f"[build] {tag:9s} {engine} {os.path.getsize(engine) / 2**20:.0f} MiB "
          f"in {dt / 60:.1f} min", flush=True)
    return dt


# Layer-name prefix -> reported module. First match wins. Myelin fuses many
# layers into one node whose name lists the fused ops, so the same table works
# for both fused and unfused builds.
MODULES = [
    ("backbone (8 cam)", ("/net/m/", "stem", "layer1", "layer2", "layer3",
                          "layer4", "lat1", "lat2", "lat3", "lat4")),
    ("depth head", ("depth_head", "depth_up")),
    ("2D seg head", ("seg_head",)),
    ("BEV lift (projection)", ("GridSample", "grid_sample", "ipm", "Gather",
                               "ScatterND")),
    ("temporal fusion", ("tfuse", "tgate")),
    ("BEV seg decoder", ("/net/dec/", "dec.")),
    ("3D det", ("det_stem", "hm_head", "reg_head", "stat_head")),
    ("2D det", ("det2d",)),
    ("E2E plan", ("ego_stem", "ego_mlp", "ego_attn", "ego_delta", "kin_delta")),
    ("occupancy", ("occ_stem", "occ_head")),
    ("traj / flow", ("traj_stem", "traj_head", "flow_head", "agent_")),
    ("traffic light", ("tl_head", "tl_fc")),
    ("risk", ("risk_head",)),
    ("lane graph", ("lg_", "lgdec", "lgq")),
    ("unknown dense", ("unk_dense",)),
    ("pseudo-LiDAR", ("pl_head",)),
    ("built-in refiner", ("refiner",)),
]


def module_of(layer_name):
    for mod, keys in MODULES:
        if any(k in layer_name for k in keys):
            return mod
    return "other / reformat"


class LayerProfiler:
    """trt.IProfiler that sums per-layer time, like CoMET's SimpleProfiler.

    Keeps count and min as well as the total, because TensorRT reports each
    layer once per execution and a single slow first call would otherwise be
    read as a slow layer.
    """

    def __init__(self):
        import tensorrt as trt
        self.rows = {}

        class _P(trt.IProfiler):
            def __init__(s, owner):
                trt.IProfiler.__init__(s)
                s.o = owner

            def report_layer_time(s, name, ms):
                r = s.o.rows.setdefault(name, [0.0, 0, 1e9])
                r[0] += ms
                r[1] += 1
                r[2] = min(r[2], ms)

        self.impl = _P(self)

    def table(self, runs):
        per_mod = {}
        for name, (tot, cnt, mn) in self.rows.items():
            per_mod.setdefault(module_of(name), [0.0, 0])
            per_mod[module_of(name)][0] += tot / max(runs, 1)
            per_mod[module_of(name)][1] += 1
        total = sum(v[0] for v in per_mod.values())
        return total, sorted(per_mod.items(), key=lambda kv: -kv[1][0])

    def top_layers(self, runs, n=12):
        rows = [(name, tot / max(runs, 1), cnt)
                for name, (tot, cnt, mn) in self.rows.items()]
        return sorted(rows, key=lambda r: -r[1])[:n]


def run_engine(engine, samples, iters, profile=False):
    import tensorrt as trt
    rt = trt.Runtime(trt.Logger(trt.Logger.ERROR))
    # A version-compatible plan ships host code with it; the runtime refuses to
    # deserialise it unless that is explicitly allowed.
    rt.engine_host_code_allowed = True
    eng = rt.deserialize_cuda_engine(open(engine, "rb").read())
    ctx = eng.create_execution_context()
    bufs = {}
    for i in range(eng.num_io_tensors):
        n = eng.get_tensor_name(i)
        shp = tuple(eng.get_tensor_shape(n))
        dt = {"DataType.FLOAT": torch.float32, "DataType.HALF": torch.float16,
              "DataType.INT32": torch.int32, "DataType.UINT8": torch.uint8,
              "DataType.BOOL": torch.bool,
              "DataType.INT8": torch.int8}[str(eng.get_tensor_dtype(n))]
        t = torch.zeros(*shp, dtype=dt, device="cuda")
        bufs[n] = t
        ctx.set_tensor_address(n, int(t.data_ptr()))
    stream = torch.cuda.Stream()

    def feed(s):
        for k, nm in (("imgs", "imgs"), ("K", "K"), ("T", "T_cam_ego"),
                      ("v0", "v0"), ("hist_bev", "hist_bev"),
                      ("hist_theta", "hist_theta")):
            if nm in bufs:
                src = s[k]
                if k == "imgs" and bufs[nm].dtype == torch.uint8:
                    src = _denorm_u8(src)
                bufs[nm].copy_(src.to(bufs[nm].dtype).cuda())

    feed(samples[0])
    # locate outputs by SHAPE, not by trusting the name list: OUT_NAMES has one
    # more entry than the model has outputs, so a name-indexed lookup silently
    # picked the wrong tensor (ego read a 103,680-element map).
    seg_n = next((n for n, t in bufs.items()
                  if t.dim() == 4 and t.shape[1] == N_CLASSES
                  and t.shape[-2] == BEV_H), "lane")
    ego_n = next((n for n, t in bufs.items() if t.numel() == 42), None)
    print(f"[io] seg='{seg_n}' {tuple(bufs[seg_n].shape)}  "
          f"ego='{ego_n}'", flush=True)
    for _ in range(20):
        ctx.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    lat = []
    for _ in range(iters):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        ctx.execute_async_v3(stream.cuda_stream)
        e1.record()
        stream.synchronize()
        lat.append(e0.elapsed_time(e1))

    prof = None
    if profile:
        # profiling adds per-layer synchronisation, so it runs AFTER the timing
        # loop and its numbers are relative, not absolute latency
        prof = LayerProfiler()
        ctx.profiler = prof.impl
        runs = 20
        for _ in range(runs):
            ctx.execute_v2([int(t.data_ptr()) for t in bufs.values()])
        # ctx.profiler cannot be cleared -- the setter rejects null with
        # "executionContext.h::setProfiler::161, condition: (profiler) !=
        # nullptr". The context is not reused after this point, so leave it
        # attached rather than logging an error on every profiled build.
        tot, rows = prof.table(runs)
        print(f"\n  per-module layer time ({os.path.basename(engine)}, "
              f"{tot:.1f} ms of layer time over {runs} runs)")
        for mod, (ms, nl) in rows:
            print(f"    {mod:24s} {ms:7.2f} ms {100 * ms / max(tot, 1e-9):5.1f}% "
                  f"({nl} layers)")
        print("  slowest individual layers")
        for nm, ms, cnt in prof.top_layers(runs):
            print(f"    {ms:7.2f} ms  {nm[:88]}")

    inter = np.zeros(N_CLASSES)
    union = np.zeros(N_CLASSES)
    ade = n = 0.0
    for s in samples:
        feed(s)
        ctx.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        pred = bufs[seg_n].float().argmax(1)[0].cpu().numpy()
        g = s["gt"].numpy()
        msk = g > 0
        for c in range(1, N_CLASSES):
            pi, gi = (pred == c) & msk, g == c
            inter[c] += (pi & gi).sum()
            union[c] += (pi | gi).sum()
        eg = s["ego"]
        if float(eg[16]) > 0.5 and ego_n is not None:
            wp = bufs[ego_n].float().reshape(-1)[:12]\
                .view(6, 2).cpu().numpy()
            ade += float(np.linalg.norm(wp - eg[:12].view(6, 2).numpy(),
                                        axis=1).mean())
            n += 1
    iou = {c: (inter[c] / union[c] if union[c] else float("nan"))
           for c in range(1, N_CLASSES)}
    return dict(mean=float(np.mean(lat)), p50=float(np.percentile(lat, 50)),
                p90=float(np.percentile(lat, 90)),
                iou=iou, miou=float(np.nanmean(list(iou.values()))),
                ade=(ade / n if n else float("nan")), n_eval=len(samples))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--root", default="out/bevlane")
    ap.add_argument("--val-list", default="val.lst")
    ap.add_argument("--out-dir", default="out/trt_v48")
    ap.add_argument("--calib", type=int, default=64)
    ap.add_argument("--eval", type=int, default=40)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--partial-int8", action="store_true",
                    help="INT8 except the layers in FP16_KEEP (refiner E2E, "
                         "scatter/gather, grid-sample): avoids the reformat the "
                         "full-INT8 build asserts on and keeps the plan in fp16")
    ap.add_argument("--direct-io", action="store_true",
                    help="forbid reformats at the engine boundary")
    ap.add_argument("--strict-types", action="store_true",
                    help="PREFER_PRECISION_CONSTRAINTS: no extra casts")
    ap.add_argument("--portable", action="store_true",
                    help="version- + hardware-compatible plan (Ampere+): the "
                         "engine file can be moved to another machine")
    ap.add_argument("--profile", action="store_true",
                    help="per-layer timing, aggregated per module")
    ap.add_argument("--workspace", type=int, default=8,
                    help="builder scratch limit in GiB; use 2 on an 8 GB GPU")
    ap.add_argument("--skip-int8", action="store_true")
    ap.add_argument("--skip-fp16", action="store_true")
    ap.add_argument("--fp16-keep", default=None,
                    help="comma-separated substrings to pin to fp16 under "
                         "--partial-int8; overrides FP16_KEEP. Fewer pins "
                         "means fewer precision boundaries and less "
                         "reformat time (measured: 365 pins cost 8.73 ms "
                         "of reformat against fp16's 3.47 ms)")
    ap.add_argument("--sparse", action="store_true",
                    help="set SPARSE_WEIGHTS; only helps if the weights are "
                         "already in the 2:4 pattern (see sparse_probe.py)")
    ap.add_argument("--hist-ckpt", default=None,
                    help="checkpoint used to synthesise a real "
                         "hist_bev for calibration instead of zeros")
    ap.add_argument("--hist-model", default="v50")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    print("[data] loading real frames", flush=True)
    ev = sample_inputs(a.root, a.val_list, a.eval,
                       hist_ckpt=a.hist_ckpt, hist_model=a.hist_model)
    cal = sample_inputs(a.root, a.val_list, a.calib,
                        hist_ckpt=a.hist_ckpt, hist_model=a.hist_model)
    print(f"[data] {len(ev)} eval / {len(cal)} calib frames", flush=True)

    res = {}
    sfx = "_portable" if a.portable else ""
    fp16 = os.path.join(a.out_dir, f"meteor_v48_fp16{sfx}.engine")
    if not a.skip_fp16:
        if not os.path.exists(fp16):
            build(a.onnx, fp16, workspace_gb=a.workspace,
                  portable=a.portable, direct_io=a.direct_io,
                  strict_types=a.strict_types, sparse=a.sparse)
        res["fp16"] = run_engine(fp16, ev, a.iters, a.profile)
    i8 = os.path.join(a.out_dir,
                      f"meteor_v48_int8{'_partial' if a.partial_int8 else ''}{sfx}.engine")
    if not a.skip_int8:
        if not os.path.exists(i8):
            build(a.onnx, i8, int8_samples=cal, sparse=a.sparse,
                  workspace_gb=a.workspace, portable=a.portable,
                  fp16_keep=(tuple(x for x in
                                (a.fp16_keep or ",".join(FP16_KEEP)
                                 ).split(",") if x)
                                if a.partial_int8 else ()))
        res["int8"] = run_engine(i8, ev, a.iters, a.profile)

    print(f"\n{'config':10s} {'ms':>7s} {'p90':>7s} {'FPS':>6s} {'mIoU':>7s} "
          f"{'lane':>7s} {'road':>7s} {'ADE':>7s}")
    for k, r in res.items():
        print(f"{k:10s} {r['mean']:7.1f} {r['p90']:7.1f} "
              f"{1000 / r['mean']:6.1f} {r['miou']:7.4f} "
              f"{r['iou'][4]:7.4f} {r['iou'][1]:7.4f} {r['ade']:7.3f}")
    if "fp16" in res and "int8" in res:
        f, i = res["fp16"], res["int8"]
        print(f"\nINT8 vs fp16: {f['mean'] / i['mean']:.2f}x faster, "
              f"mIoU {i['miou'] - f['miou']:+.4f}, ADE {i['ade'] - f['ade']:+.3f} m")
    json.dump(res, open(os.path.join(a.out_dir, "perf.json"), "w"), indent=1)
    print(f"\n-> {a.out_dir}/perf.json")


if __name__ == "__main__":
    main()

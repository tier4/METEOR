#!/usr/bin/env python3
"""Build an INT8 engine ON the Orin from a METEOR ONNX — torch-free (R3).

build_and_bench.py owns INT8 on the workstation but imports torch and the
training dataset, neither of which exists on the Orin. This script rebuilds
just the pieces the device needs: an IInt8EntropyCalibrator2 fed with REAL
frames (never random tensors — a calibrator fed noise picks activation ranges
that do not exist in the data), and the same real-history trick the runtime
uses for hist_bev.

hist_bev is the trap. Feeding zeros calibrates the temporal-fusion path on a
tensor that only occurs at scene starts. Instead a COMPANION engine (fp16,
same graph) runs the calibration stream first, frame by frame, maintaining
its device-resident raw_bev ring exactly as MeteorRT does at runtime; the
calibrator then hands TensorRT the companion's own DEVICE input pointers —
imgs/K/T/v0/hist_theta as fed from the host, hist_bev as spliced on-device
from real history. No host round-trip of the 345 MB tensor, no dumps on disk.

    sudo jetson_clocks   # first, always
    python3 deploy/orin_build_int8.py --onnx out/meteor_v61_prod.onnx \
        --companion eng/v61_fp16s.engine --roots calib fast \
        --out eng/v61_int8s.engine --sparse --calib 64 --check 24
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import tensorrt as trt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deploy.runtime import MeteorRT, cuda                        # noqa: E402
from deploy.orin_render import CAMS                              # noqa: E402


def frame_stream(roots, stride=4, max_per_scene=0):
    """Yield (scene_key, imgs_u8 [1,N,3,H,W], K, Tc, v0, pose) over all
    scenes under the given roots. pose is (x, y, yaw) when the scene carries
    ego_motion.npz with a pose track, else None (runtime-equivalent)."""
    for root in roots:
        for s in sorted(os.listdir(root)):
            mp = os.path.join(root, s, "manifest.json")
            if not os.path.isfile(mp):
                continue
            m = json.load(open(mp))
            K = np.stack([np.array(m["cams"][c]["K"], np.float32)
                          for c in CAMS])[None]
            Tc = np.stack([np.linalg.inv(np.array(
                m["cams"][c]["T_ego_cam"], np.float32)) for c in CAMS])[None]
            v0s = pose = None
            try:
                z = np.load(os.path.join(root, s, "ego_motion.npz"))
                v0s = z["v0"]
                pose = z["pose"] if "pose" in z else None
            except Exception:
                pass
            yielded = 0
            for f in m["frames"][::stride]:
                raw = []
                ok = True
                for c in CAMS:
                    im = cv2.imread(os.path.join(root, s,
                                                 f["imgs"].get(c, "_")))
                    if im is None:
                        ok = False
                        break
                    raw.append(im)
                if not ok:
                    continue
                fi = f["frame"]
                v0 = float(v0s[fi]) if v0s is not None and fi < len(v0s) \
                    else 8.0
                po = tuple(float(x) for x in pose[fi]) \
                    if pose is not None and fi < len(pose) else None
                imgs = np.ascontiguousarray(np.stack(
                    [im[:, :, ::-1].transpose(2, 0, 1) for im in raw])[None])
                lb = None
                if os.environ.get("METEOR_LIDAR", "0") == "1":   # feed real LiDAR to calibration too (zero calibration collapses the range)
                    _lp = f.get("lidar_bev") or f"lidar_bev/{int(fi):04d}.npz"
                    try:
                        lb = np.load(os.path.join(root, s, _lp))["lb"].astype(np.float32)[None]
                    except Exception:
                        lb = None
                yield s, imgs, K, Tc, v0, po, lb
                yielded += 1
                if max_per_scene and yielded >= max_per_scene:
                    break


# 2026-08-22: make the difference between calibration methods measurable. TensorRT
# offers EntropyCalibrator2 (default, clips outliers by KL minimization) / MinMaxCalibrator
# (uses the raw max) / LegacyCalibrator. Made selectable to isolate whether the
# INT8 ego freeze seen from v98 onwards depends on the calibration method.
CALIB_BASE = {
    "entropy2": trt.IInt8EntropyCalibrator2,
    "minmax": trt.IInt8MinMaxCalibrator,
    "entropy": trt.IInt8EntropyCalibrator,
    "legacy": trt.IInt8LegacyCalibrator,
}
_CALIB_KIND = os.environ.get("METEOR_CALIB_KIND", "entropy2")
_BASE = CALIB_BASE.get(_CALIB_KIND, trt.IInt8EntropyCalibrator2)


class RealFrameCalibrator(_BASE):
    """Feeds the network's six inputs from the companion engine's device
    buffers, one real frame per get_batch call."""

    def __init__(self, companion, stream, n_calib, cache):
        _BASE.__init__(self)
        self.rt = companion
        self.it = iter(stream)
        self.n = n_calib
        self.done = 0
        self.cache = cache
        self.scene = None
        # dtype conversion buffer for hist_bev (2026-08-26 calibration contamination fix).
        # The network being built reads hist_bev as fp32 per the ONNX declaration, but
        # the companion engine's buffer is fp16. Passing the pointer directly misreads
        # the fp16 bit pattern as fp32, yielding ~1e9 garbage, and the hist scales were
        # contaminated 5e7x (v124's delta-stat was the first INT8 case that bit us; older
        # generations stayed latent since tfuse was always fp16-keep and never used the scale).
        self._h16 = None

    def get_batch_size(self):
        return 1

    def get_batch(self, names):
        if self.done >= self.n:
            return None
        try:
            s, imgs, K, Tc, v0, pose, lb = next(self.it)
        except StopIteration:
            return None
        if s != self.scene:
            self.rt.reset()
            self.scene = s
        # companion fills every device input buffer (incl. the on-device
        # hist_bev splice) as a side effect of running the frame
        self.rt.infer(imgs, K, Tc, v0, pose=pose, lidar_bev=lb)
        self.done += 1
        if self.done % 8 == 0:
            print(f"[calib] {self.done}/{self.n}", flush=True)
        # convert the companion's fp16 hist_bev to fp32 and serve it from a dedicated buffer
        if "hist_bev" in self.rt.dev and \
                self.rt.host["hist_bev"].dtype == np.float16:
            if self._h16 is None:
                _n = int(np.prod(self.rt.shapes["hist_bev"]))
                self._h16 = np.empty(_n, np.float16)
                self._h32 = np.empty(_n, np.float32)
                self._dev32 = cuda.mem_alloc(_n * 4)
            cuda.memcpy_dtoh_async(self._h16, self.rt.dev["hist_bev"],
                                   self.rt.stream)
            self.rt.stream.synchronize()
            self._h32[:] = self._h16.astype(np.float32)
            cuda.memcpy_htod_async(self._dev32, self._h32, self.rt.stream)
            self.rt.stream.synchronize()
        ptrs = []
        for nm in names:
            # ONNX baked with --split-hist has three inputs hist_bev0/1/2.
            # The companion (pre-split) engine's hist_bev holds the 3 slots
            # contiguously, so passing that offset yields identical contents.
            if nm.startswith("hist_bev") and nm not in self.rt.dev \
                    and "hist_bev" in self.rt.dev:
                i = int(nm[len("hist_bev"):])
                _sh = self.rt.shapes["hist_bev"]
                _stride = int(np.prod(_sh[2:])) * 4      # fp32 after conversion
                ptrs.append(int(self._dev32) + i * _stride)
                continue
            if nm == "hist_bev" and self._h16 is not None:
                ptrs.append(int(self._dev32))
                continue
            assert nm in self.rt.dev, \
                f"calibrator has no data for engine input '{nm}'"
            ptrs.append(int(self.rt.dev[nm]))
        return ptrs

    def read_calibration_cache(self):
        if os.path.isfile(self.cache):
            return open(self.cache, "rb").read()
        return None

    def write_calibration_cache(self, cache):
        open(self.cache, "wb").write(cache)


def build(args):
    logger = trt.Logger(trt.Logger.WARNING)
    # The lift plugin must be registered before the ONNX is parsed.
    # 2026-08-15: with ctypes.CDLL alone the IPluginV3 creator was invisible to
    # the parser and it failed with "Plugin not found" (the equivalent of
    # trtexec's --staticPlugins is plugin_registry.load_library).
    _so = os.environ.get("METEOR_PLUGIN_SO")
    if _so:
        import ctypes
        ctypes.CDLL(_so, mode=ctypes.RTLD_GLOBAL)
        trt.init_libnvinfer_plugins(logger, "")
        try:
            trt.get_plugin_registry().load_library(_so)
        except Exception as _e:                       # not available in older TRT
            print(f"[plugin] load_library unavailable ({_e}) -- continuing with CDLL only",
                  flush=True)
    builder = trt.Builder(logger)
    net = builder.create_network(1 << int(
        trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(net, logger)
    ok = parser.parse(open(args.onnx, "rb").read())
    if not ok:
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        sys.exit(1)
    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,
                              args.workspace << 30)
    cfg.set_flag(trt.BuilderFlag.FP16)
    cfg.set_flag(trt.BuilderFlag.INT8)
    if args.builder_opt >= 0:
        # Level 5 takes ~2x build time but searches tactics deeper and can
        # yield an engine measured a few % faster (2026-08-28 lever 5)
        cfg.builder_optimization_level = args.builder_opt
    if args.max_aux_streams >= 0:
        # For CUDA Graph: limit the aux streams. Naively stream-capturing the
        # default engine (6 streams) leaves some work outside the graph and the
        # output freezes (bit us 2026-08-15). Capping at 0 makes the capture complete.
        cfg.max_aux_streams = args.max_aux_streams
    if args.sparse:
        cfg.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)
    if args.fp16_keep:
        # partial INT8: layers matching these name patterns stay fp16. The
        # first full-INT8 build showed visible degradation on exactly the
        # per-pixel regression/argmax heads (depth 92.4 % agreement, hm
        # sigmoid |d| 0.13) -- protect them, quantise the rest.
        pats = [p for p in args.fp16_keep.split(",") if p]
        cfg.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
        # name matching alone also hits Shape/Constant/Slice helpers that
        # inherit the node path -- setPrecision on those is an API error that
        # fails the whole build. Pin compute layers only.
        _OK = {trt.LayerType.CONVOLUTION, trt.LayerType.DECONVOLUTION,
               trt.LayerType.MATRIX_MULTIPLY, trt.LayerType.ELEMENTWISE,
               trt.LayerType.ACTIVATION, trt.LayerType.SCALE,
               trt.LayerType.POOLING, trt.LayerType.SOFTMAX}
        kept = 0
        for i in range(net.num_layers):
            lay = net.get_layer(i)
            if lay.type in _OK and any(p in lay.name for p in pats):
                try:
                    lay.precision = trt.DataType.HALF
                    kept += 1
                except Exception:
                    pass
        print(f"[fp16-keep] {kept} layers pinned fp16 "
              f"({','.join(pats)})", flush=True)
    comp = MeteorRT(args.companion)
    cfg.int8_calibrator = RealFrameCalibrator(
        comp, frame_stream(args.roots, args.stride, args.calib_per_scene),
        args.calib,
        args.out + ".calib")
    t0 = time.time()
    plan = builder.build_serialized_network(net, cfg)
    assert plan is not None, "build failed"
    open(args.out, "wb").write(plan)
    print(f"[build] {args.out} {os.path.getsize(args.out) / 2**20:.0f} MB "
          f"in {(time.time() - t0) / 60:.1f} min", flush=True)
    del comp


def check(args):
    """INT8 vs fp16 on held-back frames: argmax agreement on the uint8 maps,
    L2 on the ego vector, max|Δ| on the detection heatmap."""
    a = MeteorRT(args.companion)
    b = MeteorRT(args.out)
    agree = {"lane": [], "depth": [], "seg2d": []}
    ego_d, hm_d = [], []
    stat_agree, stat_corr, stat_std = [], [], []
    n = 0
    scene = None
    for s, imgs, K, Tc, v0, pose, lb in frame_stream(args.roots,
                                                 args.stride * 2 + 1):
        if s != scene:
            a.reset()
            b.reset()
            scene = s
        oa = a.infer(imgs, K, Tc, v0, pose=pose, lidar_bev=lb)
        ob = b.infer(imgs, K, Tc, v0, pose=pose, lidar_bev=lb)
        for k in agree:
            agree[k].append(float((oa[k] == ob[k]).mean()))
        ego_d.append(float(np.abs(oa["ego"][0, :12] -
                                  ob["ego"][0, :12]).max()))
        hm_d.append(float(np.abs(1 / (1 + np.exp(-oa["hm"]))
                                 - 1 / (1 + np.exp(-ob["hm"]))).max()))
        sa = oa["stationary"].astype(np.float32).ravel()
        sb = ob["stationary"].astype(np.float32).ravel()
        confident = np.abs(sa) > 0.25
        if confident.any():
            stat_agree.append(float(((sa > 0) == (sb > 0))[confident].mean()))
        if sa.std() > 1e-6 and sb.std() > 1e-6:
            stat_corr.append(float(np.corrcoef(sa, sb)[0, 1]))
        stat_std.append((float(sa.std()), float(sb.std())))
        n += 1
        if n >= args.check:
            break
    print(f"[check] {n} frames, INT8 vs fp16:")
    for k, v in agree.items():
        print(f"  {k:6s} argmax agreement {np.mean(v):.4f}")
    print(f"  ego  wp max|d| {np.mean(ego_d):.3f} m   "
          f"hm sigmoid max|d| {np.mean(hm_d):.3f}")
    if stat_std:
        fp_std = np.mean([x[0] for x in stat_std])
        q_std = np.mean([x[1] for x in stat_std])
        print(f"  stat logit std fp16={fp_std:.4f} int8={q_std:.4f} "
              f"corr={np.mean(stat_corr) if stat_corr else float('nan'):.4f} "
              f"sign={np.mean(stat_agree) if stat_agree else float('nan'):.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--companion", required=True,
                    help="fp16 engine of the SAME graph (trtexec build)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--roots", nargs="+", default=["calib", "fast"])
    ap.add_argument("--calib", type=int, default=64)
    ap.add_argument("--max-aux-streams", type=int, default=-1)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--builder-opt", type=int, default=-1,
                    help="builder optimization level (-1=default, max 5)")
    ap.add_argument("--calib-per-scene", type=int, default=8,
                    help="cap calibration frames per scene to prevent the "
                         "activation ranges being dominated by 2-3 drives")
    ap.add_argument("--check", type=int, default=0,
                    help="after building, compare N frames vs the companion")
    ap.add_argument("--sparse", action="store_true")
    ap.add_argument("--workspace", type=int, default=8)
    ap.add_argument("--fp16-keep", default="",
                    help="comma-separated layer-name substrings to keep fp16")
    ap.add_argument("--skip-build", action="store_true",
                    help="only run --check on existing engines")
    ap.add_argument("--cams8", action="store_true",
                    help="feed 8 cameras (dataset order: CAM_BACK_NARROW "
                         "appended) for full-rig engines like r64")
    args = ap.parse_args()
    if args.cams8 and len(CAMS) == 7:
        CAMS = CAMS + ["CAM_BACK_NARROW"]
    if not args.skip_build:
        build(args)
    if args.check:
        check(args)

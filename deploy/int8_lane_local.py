#!/usr/bin/env python3
"""INT8 lane-thinning investigation on the workstation (TensorRT 8.6, pycuda) — 2026-09-10.

build: plain ONNX -> fp16 / INT8 engine. INT8 uses IInt8EntropyCalibrator2 (or minmax) fed with real
       8-camera frames; --fp16-keep pins layer-name patterns to fp16 (bisection of the culprit);
       --sparse enables 2:4 kernels. Calibration cache is written next to the engine.
probe: runs a reference engine and a test engine over the SAME anonymised demo scenes the Orin demo
       uses and reports per-class BEV pixel ratio + IoU (test vs reference) — the Orin measurement.

  python3 deploy/int8_lane_local.py build --onnx out/meteor_v157c3Z_prod.onnx --out out/local_trt/v157_fp16.engine
  python3 deploy/int8_lane_local.py build --onnx ... --out out/local_trt/v157_int8.engine --int8 [--sparse] [--calibrator minmax] [--fp16-keep dec,lane_branch]
  python3 deploy/int8_lane_local.py probe --ref out/local_trt/v157_fp16.engine --eng out/local_trt/v157_int8.engine --tag v157_int8
"""
import argparse, glob, json, os, sys, time
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
B = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAMS = ["CAM_FRONT_WIDE", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK_WIDE", "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT", "CAM_FRONT_NARROW", "CAM_BACK_NARROW"]
NAMES = {1: "road", 3: "crosswalk", 4: "laneline", 5: "stopline", 6: "road_edge"}


def demo_scene_dirs():
    """The nine anonymised demo scenes (public 6 + city 3), in demo order."""
    order = ["highway_day", "mountain_day", "arterial_day", "valday", "valcurve", "fast", "city_shopping", "city_intersection", "city_crossing"]
    out = []
    for r in order:
        for base in ("out/hf_stage/data", "out/hf_stage/data_city"):
            p = os.path.join(B, base, r)
            if os.path.isdir(p):
                s = open(os.path.join(p, "scenes.txt")).read().split()[0]; out.append(os.path.join(p, s)); break
    return out


def calib_scene_dirs(n=8):
    """Converted val scenes not in the demo set (training-format dirs share the manifest layout)."""
    demo = {os.path.basename(d) for d in demo_scene_dirs()}
    sc = [l.strip() for l in open(os.path.join(B, "val.lst")) if l.strip()]
    return [os.path.join(B, "out/bevlane", s) for s in sc if s not in demo and os.path.isdir(os.path.join(B, "out/bevlane", s, "img"))][:n]


def frames(dirs, stride, per_scene=0, start=3):
    for d in dirs:
        m = json.load(open(os.path.join(d, "manifest.json"))); v0s = np.load(os.path.join(d, "ego_motion.npz"))["v0"]
        K = np.stack([np.array(m["cams"][c]["K"], np.float32) for c in CAMS])[None]
        T = np.stack([np.linalg.inv(np.array(m["cams"][c]["T_ego_cam"], np.float32)) for c in CAMS])[None]
        n = 0
        for fi in range(start, len(m["frames"]) - 1, stride):
            if per_scene and n >= per_scene:
                break
            f = m["frames"][fi]; ims = [cv2.imread(os.path.join(d, f["imgs"][c])) for c in CAMS]
            if any(im is None for im in ims):
                continue
            imgs = np.ascontiguousarray(np.stack([im[:, :, ::-1].transpose(2, 0, 1) for im in ims])[None].astype(np.uint8))
            yield os.path.basename(d), fi, imgs, K, T, np.array([float(v0s[fi])], np.float32); n += 1


def sanitize_cache(path, floor=1e-4):
    """Replace scale 0 / inf / nan entries in a TensorRT calibration cache; returns True if anything changed."""
    import struct, math
    lines = open(path).read().splitlines(); out = [lines[0]]; changed = 0; finite = []
    for l in lines[1:]:
        n, h = l.rsplit(": ", 1)
        try: v = struct.unpack(">f", bytes.fromhex(h))[0]
        except Exception: v = 0.0
        if math.isfinite(v) and v > 0: finite.append(v)
    big = max(finite) if finite else 1.0
    for l in lines[1:]:
        n, h = l.rsplit(": ", 1)
        try: v = struct.unpack(">f", bytes.fromhex(h))[0]
        except Exception: v = 0.0
        if not math.isfinite(v): v2 = big
        elif v <= 0: v2 = floor
        else: v2 = v
        if v2 != v: changed += 1
        out.append(f"{n}: {struct.pack('>f', v2).hex()}")
    if changed:
        open(path, "w").write("\n".join(out) + "\n"); print(f"[cache] sanitised {changed} scales in {path}", flush=True)
    return changed > 0


def build(a):
    import tensorrt as trt, pycuda.driver as cuda
    cuda.init(); ctx = cuda.Device(0).retain_primary_context(); ctx.push()
    lg = trt.Logger(trt.Logger.WARNING)
    if a.plugin:      # ONNX after make_plugin_onnx.py surgery: register the MeteorLift plugin library first
        import ctypes; ctypes.CDLL(a.plugin, mode=ctypes.RTLD_GLOBAL); trt.init_libnvinfer_plugins(lg, "")
    b = trt.Builder(lg)
    net = b.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)); p = trt.OnnxParser(net, lg)
    assert p.parse(open(a.onnx, "rb").read()), [p.get_error(i) for i in range(p.num_errors)]
    cfg = b.create_builder_config(); cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)
    cfg.set_flag(trt.BuilderFlag.FP16); cfg.builder_optimization_level = a.opt_level
    if a.sparse:
        cfg.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)
    if a.int8:
        cfg.set_flag(trt.BuilderFlag.INT8)
        it = frames(calib_scene_dirs(), 4, a.calib // 8 + 1)
        Base = trt.IInt8MinMaxCalibrator if a.calibrator == "minmax" else trt.IInt8EntropyCalibrator2

        class Cal(Base):
            def __init__(s2):
                Base.__init__(s2); s2.done = 0
                s2.dev = {"imgs": cuda.mem_alloc(8 * 3 * 432 * 768), "K": cuda.mem_alloc(8 * 9 * 4), "T_cam_ego": cuda.mem_alloc(8 * 16 * 4), "v0": cuda.mem_alloc(4)}
            def get_batch_size(s2): return 1
            def get_batch(s2, names):
                if s2.done >= a.calib: return None
                try: _, _, imgs, K, T, v0 = next(it)
                except StopIteration: return None
                for nm, arr in (("imgs", imgs), ("K", K), ("T_cam_ego", T), ("v0", v0)):
                    cuda.memcpy_htod(s2.dev[nm], np.ascontiguousarray(arr))
                s2.done += 1
                if s2.done % 16 == 0: print(f"[calib] {s2.done}/{a.calib}", flush=True)
                return [int(s2.dev[nm]) for nm in names]
            def read_calibration_cache(s2):
                return open(a.cache, "rb").read() if a.cache and os.path.isfile(a.cache) else None
            def write_calibration_cache(s2, c):
                if a.cache: open(a.cache, "wb").write(c)
        cfg.int8_calibrator = Cal()
        if a.fp16_keep:
            pats = [x for x in a.fp16_keep.split(",") if x]
            cfg.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
            OK = {trt.LayerType.CONVOLUTION, trt.LayerType.DECONVOLUTION, trt.LayerType.MATRIX_MULTIPLY, trt.LayerType.ELEMENTWISE,
                  trt.LayerType.ACTIVATION, trt.LayerType.SCALE, trt.LayerType.POOLING, trt.LayerType.SOFTMAX, trt.LayerType.REDUCE}
            kept = 0
            for i in range(net.num_layers):
                lay = net.get_layer(i)
                if lay.type in OK and any(pt in lay.name for pt in pats):
                    lay.precision = trt.DataType.HALF
                    for j in range(lay.num_outputs):
                        if lay.get_output(j).dtype == trt.DataType.FLOAT: lay.set_output_type(j, trt.DataType.HALF)
                    kept += 1
            print(f"[fp16-keep] {kept} layers pinned to fp16 for {pats}", flush=True)
    t0 = time.time(); ser = b.build_serialized_network(net, cfg)
    if ser is None and a.int8 and a.cache and os.path.isfile(a.cache) and sanitize_cache(a.cache):
        # TensorRT 8.6: constant all-zero tensors get scale 0 (and index tensors get inf) in the entropy cache,
        # and reformatBuilder asserts on them. Rewrite those entries and build again from the cache.
        print("[build] retry with sanitised calibration cache", flush=True)
        ser = b.build_serialized_network(net, cfg)
    assert ser is not None, "build failed"
    open(a.out, "wb").write(ser); print(f"built {a.out} {ser.nbytes/1e6:.0f} MB in {(time.time()-t0)/60:.1f} min", flush=True)


class OrtRef:
    """fp32 onnxruntime reference (the Orin comparison was made against fp32)."""
    def __init__(self, onnx):
        import onnxruntime as ort
        prov = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
        self.s = ort.InferenceSession(onnx, providers=prov); print("[ort] providers", prov, flush=True)
    def infer(self, imgs, K, T, v0):
        return {"lane": self.s.run(["lane"], {"imgs": imgs, "K": K, "T_cam_ego": T, "v0": np.array([v0], np.float32)})[0]}


def probe(a):
    from deploy.runtime import MeteorRT
    ref = OrtRef(a.ref) if a.ref.endswith(".onnx") else MeteorRT(a.ref); eng = MeteorRT(a.eng)
    dirs = demo_scene_dirs()
    if a.scenes: dirs = [d for d in dirs if any(d.endswith(x) for x in a.scenes.split(","))]
    per = {os.path.basename(d): {k: [0, 0, 0, 0] for k in NAMES} for d in dirs}   # ref px, test px, inter, union
    agree = tot = n = 0
    for s, fi, imgs, K, T, v0 in frames(dirs, a.stride):
        l1 = np.asarray(ref.infer(imgs, K, T, float(v0[0]))["lane"]).reshape(800, 500)
        l2 = np.asarray(eng.infer(imgs, K, T, float(v0[0]))["lane"]).reshape(800, 500)
        for k in NAMES:
            m1, m2 = l1 == k, l2 == k; c = per[s][k]
            c[0] += int(m1.sum()); c[1] += int(m2.sum()); c[2] += int((m1 & m2).sum()); c[3] += int((m1 | m2).sum())
        agree += int((l1 == l2).sum()); tot += l1.size; n += 1
    tot_c = {k: [0, 0, 0, 0] for k in NAMES}
    for s in per:
        for k in NAMES:
            for j in range(4): tot_c[k][j] += per[s][k][j]
    def fmt(c): return "\t".join(f"{NAMES[k]}:{c[k][1]/max(c[k][0],1):.3f}/{c[k][2]/max(c[k][3],1):.3f}" for k in NAMES)
    print(f"{a.tag}\tALL\t{n}\t{agree/max(tot,1):.4f}\t{fmt(tot_c)}", flush=True)
    for s in per:
        print(f"{a.tag}\t{s[-8:]}\t-\t-\t{fmt(per[s])}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--onnx", required=True); b.add_argument("--out", required=True)
    b.add_argument("--int8", action="store_true"); b.add_argument("--sparse", action="store_true"); b.add_argument("--fp16-keep", default="")
    b.add_argument("--calib", type=int, default=64); b.add_argument("--cache", default=""); b.add_argument("--calibrator", default="entropy2")
    b.add_argument("--opt-level", type=int, default=3); b.add_argument("--plugin", default="", help="path to libmeteor_lift.so for a plugin-surgery ONNX")
    p = sub.add_parser("probe"); p.add_argument("--ref", required=True); p.add_argument("--eng", required=True); p.add_argument("--tag", default="")
    p.add_argument("--stride", type=int, default=6); p.add_argument("--scenes", default="")
    a = ap.parse_args(); build(a) if a.cmd == "build" else probe(a)

#!/usr/bin/env python3
"""Real-TensorRT INT8 lane probe on an x86 GPU (2026-09-10).

Why: on the Orin the INT8 engine of the 2:4-sparse v157 loses 23-56 % of its laneline pixels
vs fp16 while dense v151 keeps them; a PyTorch fake-quant proxy did not reproduce it. This
builds fp16 / INT8 (/ INT8+sparse) engines from the PLAIN ONNX with an entropy calibrator fed
with real frames, then counts BEV lane-class pixels per engine on val frames (the same
measurement made on the Orin: pixel ratio int8/fp16 and per-class agreement).

  python3 deploy/int8_lane_x86.py build --onnx out/meteor_v157c3Z_prod.onnx --out eng/v157_fp16.plan
  python3 deploy/int8_lane_x86.py build --onnx ... --out eng/v157_int8.plan --int8 [--sparse] [--fp16-keep dec,lane_branch,refiner.seg]
  python3 deploy/int8_lane_x86.py probe --ref eng/v157_fp16.plan --eng eng/v157_int8.plan [--tag ...]
"""
import argparse, json, os, sys, time
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CAMS = ["CAM_FRONT_WIDE", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK_WIDE", "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT", "CAM_FRONT_NARROW", "CAM_BACK_NARROW"]
ROOT = os.environ.get("METEOR_BEV_ROOT", "/data/dataset/bevlane")
LIST = os.path.expanduser("~/work/BevLane/val.lst")
NAMES = {1: "road", 3: "crosswalk", 4: "laneline", 5: "stopline", 6: "road_edge"}


def frames(scenes, stride, per_scene, start=3):
    for s in scenes:
        d = os.path.join(ROOT, s)
        try:
            m = json.load(open(os.path.join(d, "manifest.json"))); v0s = np.load(os.path.join(d, "ego_motion.npz"))["v0"]
        except Exception:
            continue
        K = np.stack([np.array(m["cams"][c]["K"], np.float32) for c in CAMS])[None]
        T = np.stack([np.linalg.inv(np.array(m["cams"][c]["T_ego_cam"], np.float32)) for c in CAMS])[None]
        n = 0
        for fi in range(start, len(m["frames"]) - 1, stride):
            if per_scene and n >= per_scene:
                break
            f = m["frames"][fi]
            ims = [cv2.imread(os.path.join(d, f["imgs"][c])) for c in CAMS]
            if any(im is None for im in ims):
                continue
            imgs = np.stack([im[:, :, ::-1].transpose(2, 0, 1) for im in ims])[None].astype(np.uint8)
            yield s, fi, np.ascontiguousarray(imgs), K, T, np.array([float(v0s[fi])], np.float32); n += 1


def build(a):
    import tensorrt as trt
    from cuda.bindings import runtime as rt
    lg = trt.Logger(trt.Logger.WARNING); b = trt.Builder(lg)
    net = b.create_network(0); p = trt.OnnxParser(net, lg)
    assert p.parse(open(a.onnx, "rb").read()), [p.get_error(i) for i in range(p.num_errors)]
    cfg = b.create_builder_config(); cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 16 << 30)
    cfg.set_flag(trt.BuilderFlag.FP16)
    cfg.builder_optimization_level = a.opt_level   # 0-1: fast builds on a shared GPU (numerics come from calibration, not tactics)
    if a.sparse:
        cfg.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)
    if a.int8:
        cfg.set_flag(trt.BuilderFlag.INT8)
        scenes = [l.strip() for l in open(LIST) if l.strip()][a.n_eval:a.n_eval + 8]
        it = frames(scenes, 4, a.calib // 8 + 1)

        class Cal(trt.IInt8EntropyCalibrator2):
            def __init__(s2):
                trt.IInt8EntropyCalibrator2.__init__(s2); s2.done = 0; s2.dev = {}
                for nm, nbytes in (("imgs", 8 * 3 * 432 * 768), ("K", 8 * 9 * 4), ("T_cam_ego", 8 * 16 * 4), ("v0", 4)):
                    e, ptr = rt.cudaMalloc(nbytes); assert int(e) == 0; s2.dev[nm] = ptr
            def get_batch_size(s2): return 1
            def get_batch(s2, names):
                if s2.done >= a.calib: return None
                try: _, _, imgs, K, T, v0 = next(it)
                except StopIteration: return None
                for nm, arr in (("imgs", imgs), ("K", K), ("T_cam_ego", T), ("v0", v0)):
                    arr = np.ascontiguousarray(arr); rt.cudaMemcpy(s2.dev[nm], arr.ctypes.data, arr.nbytes, rt.cudaMemcpyKind.cudaMemcpyHostToDevice)
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
            cfg.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
            OK = {trt.LayerType.CONVOLUTION, trt.LayerType.DECONVOLUTION, trt.LayerType.MATRIX_MULTIPLY, trt.LayerType.ELEMENTWISE,
                  trt.LayerType.ACTIVATION, trt.LayerType.SCALE, trt.LayerType.POOLING, trt.LayerType.SOFTMAX}
            kept = 0
            for i in range(net.num_layers):
                lay = net.get_layer(i)
                if lay.type in OK and any(pt in lay.name for pt in pats):
                    lay.precision = trt.DataType.HALF; kept += 1
            print(f"[fp16-keep] {kept} layers pinned to fp16 for {pats}", flush=True)
    t0 = time.time(); ser = b.build_serialized_network(net, cfg); assert ser is not None, "build failed"
    open(a.out, "wb").write(ser); print(f"built {a.out} {ser.nbytes/1e6:.0f} MB in {(time.time()-t0)/60:.1f} min", flush=True)


def probe(a):
    from deploy.runtime import MeteorRT
    ref = MeteorRT(a.ref); eng = MeteorRT(a.eng)
    scenes = [l.strip() for l in open(LIST) if l.strip()][:a.n_eval]
    cnt = {k: [0, 0] for k in NAMES}; inter = {k: 0 for k in NAMES}; union = {k: 0 for k in NAMES}; n = 0; agree = 0; tot = 0
    for s, fi, imgs, K, T, v0 in frames(scenes, a.stride, 0):
        o1 = ref.infer(imgs, K, T, float(v0[0])); o2 = eng.infer(imgs, K, T, float(v0[0]))
        l1 = np.asarray(o1["lane"]).reshape(800, 500); l2 = np.asarray(o2["lane"]).reshape(800, 500)
        for k in NAMES:
            m1, m2 = l1 == k, l2 == k; cnt[k][0] += int(m1.sum()); cnt[k][1] += int(m2.sum())
            inter[k] += int((m1 & m2).sum()); union[k] += int((m1 | m2).sum())
        agree += int((l1 == l2).sum()); tot += l1.size; n += 1
    row = [a.tag or os.path.basename(a.eng), str(n), f"{agree/max(tot,1):.4f}"]
    for k, nm in NAMES.items():
        row.append(f"{nm}:{cnt[k][1]/max(cnt[k][0],1):.3f}/{inter[k]/max(union[k],1):.3f}")
    print("\t".join(row), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--onnx", required=True); b.add_argument("--out", required=True)
    b.add_argument("--int8", action="store_true"); b.add_argument("--sparse", action="store_true"); b.add_argument("--fp16-keep", default="")
    b.add_argument("--calib", type=int, default=64); b.add_argument("--opt-level", type=int, default=1); b.add_argument("--cache", default=""); b.add_argument("--n-eval", type=int, default=6)
    p = sub.add_parser("probe"); p.add_argument("--ref", required=True); p.add_argument("--eng", required=True); p.add_argument("--tag", default="")
    p.add_argument("--n-eval", type=int, default=6); p.add_argument("--stride", type=int, default=6)
    a = ap.parse_args(); build(a) if a.cmd == "build" else probe(a)

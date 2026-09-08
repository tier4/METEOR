#!/usr/bin/env python3
"""Build a plugin-free fp16 TensorRT engine from the released METEOR ONNX (workstation reproduction)."""
import sys, time, tensorrt as trt
onnx, out = sys.argv[1], sys.argv[2]
ws_gb = int(sys.argv[3]) if len(sys.argv) > 3 else 8
log = trt.Logger(trt.Logger.INFO)
b = trt.Builder(log); net = b.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
p = trt.OnnxParser(net, log)
with open(onnx, "rb") as f:
    ok = p.parse(f.read())
if not ok:
    for i in range(p.num_errors): print("PARSE ERROR", p.get_error(i))
    sys.exit(1)
print("inputs:", [(net.get_input(i).name, net.get_input(i).shape, net.get_input(i).dtype) for i in range(net.num_inputs)])
print("outputs:", net.num_outputs, [net.get_output(i).name for i in range(net.num_outputs)])
cfg = b.create_builder_config()
cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, ws_gb << 30)
cfg.set_flag(trt.BuilderFlag.FP16)
t0 = time.time()
ser = b.build_serialized_network(net, cfg)
assert ser is not None, "build failed"
open(out, "wb").write(ser)
print(f"saved {out} ({ser.nbytes/1e6:.0f} MB) in {time.time()-t0:.0f}s, TensorRT {trt.__version__}")

#!/usr/bin/env python3
"""Surgery to convert GridSample fp32 islands to fp16 (2026-08-30, vs 2.04 ms Orin GridSample copies).

GridSample_2/3/4 of the temporal hist warp are fp32 islands due to explicit
Cast(FLOAT) at export. Switch the feature-side Cast to FLOAT16 and insert a
Cast(FLOAT) on the output so the outer graph dtype is unchanged (the only math
difference is fp16 quantization of features). grid (coords) stays fp32 - no positional loss.
Per the lever-4 lesson, adopt only via on-device build equivalence check + bench.
"""
import argparse
import onnx
from onnx import TensorProto, helper

ap = argparse.ArgumentParser()
ap.add_argument("--onnx", default="out/meteor_v130c3R_final.onnx")
ap.add_argument("--out", default="out/meteor_v130gs16_final.onnx")
ap.add_argument("--nodes", default="/net/GridSample_2,/net/GridSample_3,/net/GridSample_4")
a = ap.parse_args()

m = onnx.load(a.onnx)
g = m.graph
prod = {o: n for n in g.node for o in n.output}
targets = set(a.nodes.split(","))
n_done = 0
for n in list(g.node):
    if n.op_type != "GridSample" or n.name not in targets:
        continue
    feat_cast = prod[n.input[0]]
    assert feat_cast.op_type == "Cast", f"{n.name}: feat producer is {feat_cast.op_type}"
    for att in feat_cast.attribute:
        if att.name == "to":
            att.i = TensorProto.FLOAT16
    old_out = n.output[0]
    gs16_out = old_out + "_fp16"
    n.output[0] = gs16_out
    back = helper.make_node("Cast", [gs16_out], [old_out],
                            name=n.name + "_castback", to=TensorProto.FLOAT)
    # insert right after GridSample (keeps topological order)
    idx = list(g.node).index(n)
    g.node.insert(idx + 1, back)
    n_done += 1
print(f"[gs16] {n_done} GridSample converted to fp16 (grid stays fp32)")
try:
    onnx.checker.check_model(m, full_check=False)
except Exception as e:
    print(f"[gs16] checker (original file is non-compliant too): {str(e)[:80]}")
onnx.save(m, a.out)
print(f"[gs16] wrote {a.out}")

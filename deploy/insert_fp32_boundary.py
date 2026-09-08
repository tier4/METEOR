"""Insert Cast(fp32) at the ego / stationary branch entries to create a precision boundary.

2026-08-22: the v98/v103 INT8 engines froze the ego output (frame-to-frame diff
0.06 vs 2.05 for fp16) and halved the stationary logit range (std 4.79 vs 9.31),
so every object got a "stationary" verdict and the E2E path did not move. The
--layerPrecisions fp16 setting never reached the Myelin-fused layers, so we
insert explicit Casts on the ONNX side to tell TensorRT the precision boundary.
"""
import argparse
import onnx
from onnx import helper, TensorProto

ap = argparse.ArgumentParser()
ap.add_argument("--onnx", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--entries", default="/net/ego_stem/ego_stem.0/Conv,/net/stat_head2/Conv")
a = ap.parse_args()

m = onnx.load(a.onnx)
g = m.graph
targets = [t for t in a.entries.split(",") if t]
by_name = {n.name: n for n in g.node}
new_nodes = []
done = 0
for t in targets:
    n = by_name.get(t)
    if n is None:
        print(f"[warn] node not found: {t}")
        continue
    src = n.input[0]
    cast_out = f"{src}__fp32b"
    cast = helper.make_node("Cast", [src], [cast_out],
                            name=f"cast_fp32_{done}", to=TensorProto.FLOAT)
    new_nodes.append((n, cast, cast_out))
    print(f"[ok] {t}: {src[:45]} -> Cast(fp32)")
    done += 1

for n, cast, cast_out in new_nodes:
    n.input[0] = cast_out
    idx = list(g.node).index(n)
    g.node.insert(idx, cast)

onnx.save(m, a.out, save_as_external_data=False)
print(f"inserted precision boundary at {done} sites -> {a.out}")

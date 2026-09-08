"""ego / stationary ブランチの入口に Cast(fp32) を挿入して精度境界を作る。

2026-08-22: v98/v103 の INT8 エンジンは ego 出力が凍結し (フレーム間差
0.06 vs fp16 の 2.05)、stationary のロジット幅も半減する (std 4.79 vs 9.31)
ため、全物体が「停止」判定になり E2E パスが動かない。--layerPrecisions の
fp16 指定は Myelin 融合層に届かず無効だったので、ONNX 側で明示的な
Cast を入れて TensorRT に精度境界を伝える。
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
        print(f"[warn] ノードなし: {t}")
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
print(f"{done} 箇所に精度境界を挿入 -> {a.out}")

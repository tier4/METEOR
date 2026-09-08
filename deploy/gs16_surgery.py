#!/usr/bin/env python3
"""GridSample fp32 島の fp16 化手術 (2026-08-30, Orin GridSample コピー 2.04ms 対策)。

時間系 hist ワープの GridSample_2/3/4 は輸出時の明示 Cast(FLOAT) で fp32 島に
なっている。特徴側 Cast を FLOAT16 へ付け替え、出力に Cast(FLOAT) を挿んで
グラフの外側 dtype は不変に保つ (数学的差は特徴量の fp16 量子化のみ)。
grid (座標) 側は fp32 のまま — 位置精度は落とさない。
レバー4 の教訓により、採否は実機ビルドの等価チェック + bench でのみ判定する。
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
    # GridSample 直後に挿入 (トポロジカル順維持)
    idx = list(g.node).index(n)
    g.node.insert(idx + 1, back)
    n_done += 1
print(f"[gs16] {n_done} GridSample を fp16 化 (grid は fp32 のまま)")
try:
    onnx.checker.check_model(m, full_check=False)
except Exception as e:
    print(f"[gs16] checker (元ファイルも非準拠): {str(e)[:80]}")
onnx.save(m, a.out)
print(f"[gs16] wrote {a.out}")

"""paint-seg 経路の Gather をpaint_proj Conv に畳み込む (2026-08-27)。

Orin プロファイルで Softmax+Gather が Myelin ForeignNode に固まり、その
入出力 Reformat が 3.78 + 0.69x3 ms を食っていた。Gather(axis=1, 定数
インデックス) + 1x1 Conv は「入力チャネルを並べ替えた 1x1 Conv」と恒等
なので、paint_proj の重みを [96, 8] -> [96, C_full] (未選択列ゼロ) に
拡張して Gather/Cast_1/Constant を削除する。数学的に同値・精度影響ゼロ。

使い方: python3 deploy/fold_paint_gather.py <in.onnx> <out.onnx>
"""
import sys

import numpy as np
import onnx
from onnx import numpy_helper

src, dst = sys.argv[1], sys.argv[2]
m = onnx.load(src)
g = m.graph
nodes = {n.name: n for n in g.node}
init = {i.name: i for i in g.initializer}

ga = nodes["/net/Gather"]
c1 = nodes["/net/Cast_1"]
pp = nodes["/net/paint_proj/Conv"]
const = next(n for n in g.node if ga.input[1] in n.output)
idx = numpy_helper.to_array(next(a.t for a in const.attribute
                                 if a.name == "value"))
sm_out = ga.input[0]                      # Softmax_1 の出力

# softmax のチャネル数 = softmax へ至る conv の出力チャネル (確実な出所)
prod = {o: n for n in g.node for o in n.output}
cur = prod[sm_out]                       # Softmax_1
while cur.op_type != "Conv":
    cur = prod[cur.input[0]]
C_full = numpy_helper.to_array(init[cur.input[1]]).shape[0]
print(f"fold: idx={idx.tolist()} C_full={C_full} (from {cur.name})")

w = numpy_helper.to_array(init[pp.input[1]])          # [96, 8, 1, 1]
assert w.shape[1] == len(idx)
w2 = np.zeros((w.shape[0], C_full, 1, 1), w.dtype)
for j, c in enumerate(idx):
    w2[:, int(c)] = w[:, j]
new_w = numpy_helper.from_array(w2, pp.input[1])
for i, t in enumerate(g.initializer):
    if t.name == pp.input[1]:
        g.initializer.pop(i)
        g.initializer.insert(i, new_w)
        break

pp.input[0] = sm_out                       # Cast_1 を飛ばして直結
for n in (ga, c1, const):
    g.node.remove(n)
onnx.save(m, dst)
print(f"wrote {dst}  paint_proj W {w.shape} -> {w2.shape}, "
      f"removed Gather/Cast_1/Constant")

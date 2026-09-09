#!/usr/bin/env python3
"""Replace the v61 graph's dense lift with one MeteorLift plugin node.

The lift subgraph (156 nodes: the T_cam_ego x bev_pts projection math, two
GridSamples over ctx/dprob, the depth-bin GatherElements, the valid mask and
the camera reduction, ending in the [1,96,150,125] reshape) is replaced by:

    MeteorLift(dprob = /net/Softmax_output_0, ctx = /net/ctx/Conv_output_0)
        -> lift_bev [1,96,150,125]

with the pair tables (dump_lift.py) baked as node attributes (INTS/FLOATS --
the ONNX parser's fallback plugin importer converts these to PluginFields).
The Resize to 600x500 stays in the graph. K / T_cam_ego remain as (now
unused) inputs so the engine I/O signature is unchanged if the parser
tolerates it; --drop-kt removes them instead.

usage: make_plugin_onnx.py [--onnx out/meteor_v61_prod.onnx]
                           [--tables deploy/cpp/liftbench/tables]
                           [--out out/meteor_v61_lift_plugin.onnx]
"""
import argparse
import json
import os
from collections import deque

import numpy as np
import onnx
import onnx_graphsurgeon as gs

DPROB = "/net/Softmax_output_0"
CTX = "/net/ctx/Conv_output_0"
RESIZE = "/net/Resize_3"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="out/meteor_v61_prod.onnx")
    ap.add_argument("--tables", default="deploy/cpp/liftbench/tables")
    ap.add_argument("--out", default="out/meteor_v61_lift_plugin.onnx")
    ap.add_argument("--drop-kt", action="store_true",
                    help="remove K/T_cam_ego from the graph inputs")
    ap.add_argument("--fold", action="store_true",
                    help="fold the depth Softmax and the 150x125->600x500 "
                         "Resize INTO the plugin: input 0 becomes the raw "
                         "depth logits, the plugin's fp32 output becomes "
                         "raw_bev directly (Softmax / Resize_3 removed)")
    a = ap.parse_args()

    meta = json.load(open(os.path.join(a.tables, "meta.json")))
    P, G2 = meta["P"], meta["G2"]
    tb = {
        "rowptr": np.fromfile(f"{a.tables}/csr_rowptr.bin", np.int32),
        "col": np.fromfile(f"{a.tables}/csr_col.bin", np.int32),
        "cam": np.fromfile(f"{a.tables}/pair_cam.bin", np.int32),
        "b0": np.fromfile(f"{a.tables}/pair_b0.bin", np.int32),
        "ix": np.fromfile(f"{a.tables}/pair_ix.bin", np.float32),
        "iy": np.fromfile(f"{a.tables}/pair_iy.bin", np.float32),
        "fr": np.fromfile(f"{a.tables}/pair_fr.bin", np.float32),
    }
    assert len(tb["rowptr"]) == G2 + 1 and len(tb["col"]) == P

    g = gs.import_onnx(onnx.load(a.onnx))
    tmap = g.tensors()
    ctx = tmap[CTX]
    # PointPainting (2026-08-17): in ckpts with paint enabled, an Add
    # "ctx + paint_proj(seg2d probs)" follows ctx and downstream uses its output.
    # Grabbing the raw ctx would get the paint path removed as dead code, so
    # if an Add takes ctx as input, pass its output to the plugin instead.
    _adds = [n for n in g.nodes
             if n.op == "Add" and any(t is ctx for t in n.inputs)]
    if _adds:
        ctx = _adds[0].outputs[0]
        print("[surgery] using painted ctx (PointPainting path preserved)")
    resize = [n for n in g.nodes if n.name == RESIZE]
    assert len(resize) == 1, "Resize_3 not found"
    resize = resize[0]

    # attrs: ints/floats as python lists -> INTS/FLOATS attributes
    attrs = dict(n_cams=meta["N"], depth_bins=meta["D"], cc=meta["Cc"],
                 hf=meta["Hf"], wf=meta["Wf"],
                 lift_h=meta["lift_h"], lift_w=meta["lift_w"],
                 rowptr=tb["rowptr"].tolist(), col=tb["col"].tolist(),
                 cam=tb["cam"].tolist(), b0=tb["b0"].tolist(),
                 ix=tb["ix"].tolist(), iy=tb["iy"].tolist(),
                 fr=tb["fr"].tolist())
    if a.fold:
        # input 0 = the RAW depth logits (Softmax's input); output = the
        # Resize_3 output tensor itself (raw_bev), fp32, full 600x500
        sm = [n for n in g.nodes if n.name == "/net/Softmax"]
        assert len(sm) == 1, "/net/Softmax not found"
        dlog = sm[0].inputs[0]
        raw_bev = resize.outputs[0]
        attrs.update(fold_softmax=1, out_h=int(meta["out_h"]),
                     out_w=int(meta["out_w"]), out_fp32=1)
        node = gs.Node(op="MeteorLift", name="MeteorLift_0",
                       inputs=[dlog, ctx], outputs=[raw_bev], attrs=attrs)
        g.nodes.append(node)
        resize.outputs = []          # orphan Resize_3 (+ Softmax upstream)
        print(f"[surgery] fold: in0={dlog.name}, out={raw_bev.name} "
              f"[1,{meta['Cc']},{meta['out_h']},{meta['out_w']}] fp32")
    else:
        dprob = tmap[DPROB]
        old_out = resize.inputs[0]          # /net/Reshape_5_output_0
        lift_out = gs.Variable("lift_bev", dtype=np.float16,
                               shape=[1, meta["Cc"], meta["lift_h"],
                                      meta["lift_w"]])
        node = gs.Node(op="MeteorLift", name="MeteorLift_0",
                       inputs=[dprob, ctx], outputs=[lift_out], attrs=attrs)
        g.nodes.append(node)
        # rewire every consumer of the old lift output (Resize_3 + its Shape)
        for n in list(g.nodes):
            n.inputs = [lift_out if t is old_out else t for t in n.inputs]

    before = len(g.nodes)
    g.cleanup(remove_unused_graph_inputs=False)
    print(f"[surgery] nodes {before} -> {len(g.nodes)} "
          f"(removed {before - len(g.nodes)})")
    if a.drop_kt:
        g.inputs = [i for i in g.inputs if i.name not in ("K", "T_cam_ego")]
        print("[surgery] dropped inputs K, T_cam_ego")
    print("[surgery] inputs:", [i.name for i in g.inputs])

    m = gs.export_onnx(g)
    onnx.save(m, a.out, save_as_external_data=m.ByteSize() > (2 << 30))
    print(f"[surgery] wrote {a.out} "
          f"{os.path.getsize(a.out) / 2**20:.1f} MB")


if __name__ == "__main__":
    main()

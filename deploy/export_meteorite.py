#!/usr/bin/env python3
"""Export the METEORite skeleton to ONNX for one real rig, lint it, check ORT parity, count MACs.

usage: python deploy/export_meteorite.py --out out/meteorite [--vt gather,matmul] [--scene <id>]
The rig (K / T_cam_ego of 8 cameras) is read from one training sample so the baked tables are real.
"""
import argparse
import collections
import os
import sys
import time

import numpy as np
import onnx
import torch
from onnx import shape_inference

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bevlane import meteorite as M                       # noqa: E402


def load_rig(root, scene):
    from bevlane.dataset import BevLaneDataset
    ds = BevLaneDataset(root, [scene], gt_key="gt_cons", with_ego=True, n_cams=8, max_per_scene=1,
                        trim_start=3, trim_end=10)
    b = ds[0]
    K, T = b[1].numpy(), b[2].numpy()
    hw = tuple(b[0].shape[-2:])
    return K, T, hw


def count_macs(path):
    m = shape_inference.infer_shapes(onnx.load(path))
    shapes = {}
    for vi in list(m.graph.value_info) + list(m.graph.input) + list(m.graph.output):
        shapes[vi.name] = [d.dim_value for d in vi.type.tensor_type.shape.dim]
    inits = {i.name: list(i.dims) for i in m.graph.initializer}
    macs = collections.Counter(); unk = 0
    for n in m.graph.node:
        if n.op_type == "Conv":
            w = inits.get(n.input[1]); o = shapes.get(n.output[0])
            if w and o and all(o):
                macs["Conv"] += int(np.prod(o)) * w[1] * w[2] * w[3]
            else:
                unk += 1
        elif n.op_type in ("MatMul", "Gemm"):
            a = inits.get(n.input[0]) or shapes.get(n.input[0]); b = inits.get(n.input[1]) or shapes.get(n.input[1])
            if a and b and all(a) and all(b):
                macs[n.op_type] += int(np.prod(a)) * b[-1]
            else:
                unk += 1
    return macs, unk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/meteorite")
    ap.add_argument("--vt", default="gather,matmul")
    ap.add_argument("--root", default=os.environ.get("METEOR_BEV_ROOT", "out/bevlane"))
    ap.add_argument("--scene", default=None)
    ap.add_argument("--conv1x1", action="store_true", help="conv1x1 layer layout (softmax on the channel axis, two-stage pooling); adds _conv1x1 to the output name")
    ap.add_argument("--prime", default="matmul", choices=["matmul", "conv_exact", "conv_learned"])
    ap.add_argument("--blk", type=int, default=None, help="blockconv/matmul block size (default 10; 25 for the conv1x1 target)")
    ap.add_argument("--float-input", action="store_true", help="take images as float (removes the Cast from the graph; leaves input formatting to the accelerator runtime)")
    ap.add_argument("--with-lidar", action="store_true", help="add lidar_bev [1,4,400,250] as an input (all zeros = exactly off)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.blk:
        M.BLK = a.blk
    scene = a.scene or [l.strip() for l in open("val.lst") if l.strip()][0]
    K, T, hw = load_rig(a.root, scene)
    print(f"[rig] scene {scene}: K image size {hw}, cx/cy of cam0 = {K[0,0,2]:.1f}/{K[0,1,2]:.1f}", flush=True)
    t0 = time.time()
    taps = M.bake_taps(K, T, hw)
    n_tap = sum(len(t) for t in taps); n_vis = sum(1 for t in taps if t)
    print(f"[bake] {len(taps)} cells, {n_vis} visible ({100*n_vis/len(taps):.0f}%), {n_tap} taps "
          f"(mean {n_tap/max(n_vis,1):.1f}/visible cell), kmax {max(len(t) for t in taps)}  [{time.time()-t0:.1f}s]")
    np.save(os.path.join(a.out, "rig_K.npy"), K); np.save(os.path.join(a.out, "rig_T_cam_ego.npy"), T)

    torch.manual_seed(0)
    imgs = torch.randint(0, 256, (1, M.N_CAM, 3, M.IMG_H, M.IMG_W), dtype=torch.uint8)
    if a.float_input:
        imgs = imgs.float()
    v0 = torch.tensor([8.0])
    ref = None
    for vt in a.vt.split(","):
        net = M.METEORite(taps, vt=vt, conv1x1=a.conv1x1, prime=a.prime).eval()
        net.float_input = a.float_input
        if vt in ("matmul", "blockconv"):
            n_lay = len(getattr(net.vt, "weights", getattr(net.vt, "convs", [])))
            print(f"[vt={vt}] BLK {M.BLK} blocks {len(net.vt.blocks)}, block-camera layers {n_lay}, "
                  f"constant weights {net.vt.n_params/1e6:.1f}M, {net.vt.n_mac/1e9:.2f} GMAC")
        # share the learned (random) weights across variants so the two VTs can be compared
        if ref is None:
            ref = {k: v for k, v in net.state_dict().items() if not k.startswith("vt.")}
        else:
            net.load_state_dict(ref, strict=False)
        lid = torch.zeros(1, 4, M.LH, M.LW) if a.with_lidar else None
        with torch.no_grad():
            outs = net(imgs, v0, lid)
        path = os.path.join(a.out, f"meteorite_{vt}{'_conv1x1' if a.conv1x1 else ''}{'' if a.prime == 'matmul' else '_' + a.prime}.onnx")
        if a.prime == "conv_exact":          # same weights as the matmul variant -> the prime maps must agree numerically
            ref_net = M.METEORite(taps, vt=vt, conv1x1=a.conv1x1, prime="matmul").eval(); ref_net.load_state_dict(net.state_dict(), strict=False); ref_net.float_input = a.float_input
            with torch.no_grad():
                o_ref = ref_net(imgs, v0, lid)
            d = max(float((x - y).abs().max()) for x, y in zip(o_ref, outs))
            print(f"[prime] conv_exact vs matmul (all outputs, same weights): max|diff| {d:.2e}")
        t0 = time.time()
        M.export(net, path, with_lidar=a.with_lidar)
        print(f"[export] {path}: {os.path.getsize(path)/1e6:.1f} MB  [{time.time()-t0:.0f}s]")
        import onnxruntime as ort
        s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        t0 = time.time()
        feed = {"imgs": imgs.numpy(), "v0": v0.numpy()}
        if a.with_lidar:
            feed["lidar_bev"] = lid.numpy()
        o = s.run(None, feed)
        dt = time.time() - t0
        worst = max(float((torch.from_numpy(x) - y).abs().max()) for x, y in zip(o, outs))
        print(f"[parity] onnxruntime(CPU) vs PyTorch: max|diff| {worst:.2e} over {len(o)} outputs; CPU run {dt*1e3:.0f} ms")
        macs, unk = count_macs(path)
        print(f"[macs] {sum(macs.values())/1e9:.1f} GMAC  ({', '.join(f'{k} {v/1e9:.1f}' for k, v in macs.items())}; "
              f"unknown-shape nodes {unk})")
        if vt == "blockconv":
            torch.manual_seed(1); P = torch.randn(M.N_CAM, M.C_P, M.D * M.FW)
            d = float((net.vt(P) - M.VTGather(taps)(P)).abs().max())
            print(f"[vt] blockconv vs gather view transform on the same prime map: max|diff| {d:.2e}")
        if vt == "gather":
            net_m = M.METEORite(taps, vt="matmul", pretrained=False)
            torch.manual_seed(1); P = torch.randn(M.N_CAM, M.C_P, M.D * M.FW)
            d = float((net.vt(P) - net_m.vt(P)).abs().max())
            print(f"[vt] gather vs matmul view transform on the same prime map: max|diff| {d:.2e}")
    print("[done]")


if __name__ == "__main__":
    main()

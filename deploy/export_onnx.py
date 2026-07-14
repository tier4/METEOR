#!/usr/bin/env python3
"""Export METEOR v26 to ONNX for TensorRT deployment.

Wraps the training model into a deployment graph:
  inputs : imgs [1,8,3,432,768], K [1,8,3,3], T_cam_ego [1,8,4,4],
           v0 [1], prev_bev [1,96,800,500], warp_theta [1,2,3]
  outputs: the 11 task outputs + raw_bev [1,96,800,500]

raw_bev is fed back as next frame's prev_bev (streaming temporal BEV) —
the recurrence lives OUTSIDE the engine, so the graph stays static.
F.affine_grid is not exportable; the warp grid is built manually from
warp_theta with basic ops (matmul over a constant base grid).
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bevlane.model import MODELS, BEV_H, BEV_W  # noqa: E402

OUT_NAMES = ["lane", "depth", "seg2d", "hm", "reg", "hm2d", "reg2d",
             "ego", "occ", "traj", "stationary", "raw_bev"]


class MeteorDeploy(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        # normalized base grid for manual affine_grid (align_corners=False)
        ys = (torch.arange(BEV_H, dtype=torch.float32) + 0.5) / BEV_H * 2 - 1
        xs = (torch.arange(BEV_W, dtype=torch.float32) + 0.5) / BEV_W * 2 - 1
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        base = torch.stack([gx, gy, torch.ones_like(gx)], -1)   # [H,W,3]
        self.register_buffer("base_grid", base.reshape(1, -1, 3))

    def forward(self, imgs, K, T_cam_ego, v0, prev_bev, warp_theta):
        net = self.net
        bev = net.project_bev(*net._encode(imgs, K, T_cam_ego))
        net._last_bev = bev
        # manual affine_grid: grid = base @ theta^T  (theta [1,2,3])
        grid = torch.matmul(self.base_grid, warp_theta.transpose(1, 2))
        grid = grid.reshape(1, BEV_H, BEV_W, 2)
        warped = F.grid_sample(prev_bev, grid, align_corners=False)
        fused = bev + net.tfuse(torch.cat([bev, warped], 1))
        net._fused_bev = fused
        det = net.det_stem(net._last_bev)          # v25 routing: det on RAW
        net._det_feat = det
        lane = net.dec(net._last_bev)
        seg2d, dlog = net._aux2d
        hm2d, reg2d = net.det2d_forward(net._f_s4, 1, imgs.shape[1])
        pooled = net.ego_stem(fused).flatten(1)
        ego = net.ego_mlp(torch.cat([pooled, v0.view(1, 1)], 1))
        crop = net._last_bev[:, :, 200:600, 50:450]
        o = net.occ_head(net.occ_stem(crop))
        occ = o.view(1, 10, 16, o.shape[-2], o.shape[-1])
        tfeat = net.traj_stem(fused)
        return (lane, dlog, seg2d, net.hm_head(det), net.reg_head(det),
                hm2d[0], reg2d[0], ego, occ, net.traj_head(tfeat),
                net.stat_head(det), bev)


def build(ckpt):
    net = MODELS["v26"](n_seg=21)
    sd = torch.load(ckpt, map_location="cpu")["model"]
    net.load_state_dict(sd)
    net.eval()

    # expose the pieces the deploy wrapper needs as plain methods
    def _encode(imgs, K, T_cam_ego):
        B, N, _, H, W = imgs.shape
        f = net.image_feats(imgs)
        net._f_s4 = f
        seg2d = net.seg_head(f)
        dlog = net.depth_head(net.depth_up(f))
        net._aux2d = (seg2d.view(B, N, -1, seg2d.shape[-2], seg2d.shape[-1]),
                      dlog)
        ctx = net.ctx(f)
        return dlog.softmax(1), ctx, K, T_cam_ego, B, N, H, W

    net._encode = _encode
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="out/meteor_v26.onnx")
    ap.add_argument("--fp16", action="store_true",
                    help="also write a weight-fp16 copy next to --out")
    ap.add_argument("--check", action="store_true",
                    help="verify onnxruntime outputs against torch")
    args = ap.parse_args()

    net = build(args.ckpt)
    dep = MeteorDeploy(net).eval()
    imgs = torch.randn(1, 8, 3, 432, 768)
    K = torch.eye(3).repeat(1, 8, 1, 1)
    K[:, :, 0, 0] = K[:, :, 1, 1] = 600.0
    K[:, :, 0, 2], K[:, :, 1, 2] = 384.0, 216.0
    Tc = torch.eye(4).repeat(1, 8, 1, 1)
    v0 = torch.tensor([8.0])
    pb = torch.zeros(1, 96, BEV_H, BEV_W)
    th = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.005]]])
    ex = (imgs, K, Tc, v0, pb, th)

    with torch.no_grad():
        ref = dep(*ex)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.onnx.export(
        dep, ex, args.out, opset_version=17,
        input_names=["imgs", "K", "T_cam_ego", "v0", "prev_bev", "warp_theta"],
        output_names=OUT_NAMES, do_constant_folding=True)
    print("exported", args.out, "%.1f MB" % (os.path.getsize(args.out) / 2**20))

    if args.check:
        import numpy as np
        import onnxruntime as ort
        s = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
        outs = s.run(None, {"imgs": imgs.numpy(), "K": K.numpy(),
                            "T_cam_ego": Tc.numpy(), "v0": v0.numpy(),
                            "prev_bev": pb.numpy(), "warp_theta": th.numpy()})
        for nm, o, r in zip(OUT_NAMES, outs, ref):
            d = float(np.abs(o - r.numpy()).max())
            print(f"  {nm:10s} max|diff|={d:.2e} {'OK' if d < 2e-3 else 'FAIL'}")

    if args.fp16:
        import onnx
        from onnxconverter_common import float16
        m = onnx.load(args.out)
        m16 = float16.convert_float_to_float16(
            m, keep_io_types=True, disable_shape_infer=True)
        p16 = args.out.replace(".onnx", "_fp16.onnx")
        onnx.save(m16, p16)
        print("exported", p16, "%.1f MB" % (os.path.getsize(p16) / 2**20))


if __name__ == "__main__":
    main()

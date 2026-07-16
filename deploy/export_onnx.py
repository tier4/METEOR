#!/usr/bin/env python3
"""Export METEOR v29 to ONNX for TensorRT deployment.

Wraps the training model into a deployment graph:
  inputs : imgs [1,8,3,432,768], K [1,8,3,3], T_cam_ego [1,8,4,4], v0 [1],
           hist_bev [1,3,96,800,500], hist_theta [1,3,2,3]
  outputs: the 16 task tensors + raw_bev [1,96,800,500]

raw_bev is fed back as the next frame's newest history slot (streaming
temporal BEV) — the recurrence lives OUTSIDE the engine, so the graph
stays static. F.affine_grid is not exportable; the warp grid is built
manually from each slot's theta with basic ops (matmul over a constant
base grid).
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bevlane.model import MODELS, BEV_H, BEV_W  # noqa: E402

OUT_NAMES = ["lane", "depth", "seg2d", "hm", "reg", "hm2d", "reg2d",
             "ego", "occ", "traj", "stationary", "tl", "risk", "flow",
             "lg_pts", "lg_meta", "lg_adj", "raw_bev"]
HIST_N = 3


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
        # lane-graph anchors/grid are model buffers; copy them so the export
        # graph holds constants rather than attribute lookups
        self.register_buffer("lg_grid", net.lg_grid.clone())
        self.register_buffer("lg_anchors_n", (net.lg_anchors / 30.0)[None])
        self.register_buffer("lg_anchor_xy", net.lg_anchors.view(1, 24, 1, 2))

    def forward(self, imgs, K, T_cam_ego, v0, hist_bev, hist_theta):
        net = self.net
        bev = net.project_bev(*net._encode(imgs, K, T_cam_ego))
        net._last_bev = bev
        # manual affine_grid per history slot: grid = base @ theta^T
        cat = [bev]
        for i in range(HIST_N):
            grid = torch.matmul(self.base_grid,
                                hist_theta[:, i].transpose(1, 2))
            grid = grid.reshape(1, BEV_H, BEV_W, 2)
            cat.append(F.grid_sample(hist_bev[:, i], grid,
                                     align_corners=False))
        fused = bev + net.tfuse3(torch.cat(cat, 1))
        net._fused_bev = fused
        det = net.det_stem(net._last_bev)          # v25 routing: det on RAW
        net._det_feat = det
        lane = net.dec(net._last_bev)
        seg2d, dlog = net._aux2d
        hm2d, reg2d = net.det2d_forward(net._f_s4, 1, imgs.shape[1])
        pooled = net.ego_stem(fused).flatten(1)
        ego = net.ego_mlp(torch.cat([pooled, v0.view(1, 1)], 1))
        crop = net._last_bev[:, :, 200:600, 50:450]
        of = net.occ_stem(crop)
        o = net.occ_head(of)
        occ = o.view(1, 10, 16, o.shape[-2], o.shape[-1])
        flow = net.flow_head(of)
        # traj head is class-conditioned: motion stem + detection feature
        tfeat = torch.cat([net.traj_stem(fused), det], 1)
        # TL head reads the front WIDE + NARROW s4 features
        f = net._f_s4.view(1, imgs.shape[1], -1, *net._f_s4.shape[-2:])
        tl = net.tl_fc(net.tl_head(torch.cat([f[:, 0], f[:, 6]], 1)).flatten(1))
        risk = net.risk_head(fused[:, :, 200:600, 125:375])
        # lane-graph slot decoder
        roi = net._last_bev[:, :, 100:450, 125:375]
        lf = net.lg_tower(roi)
        emb = F.grid_sample(lf, self.lg_grid, align_corners=False)[..., 0]
        emb = emb.transpose(1, 2)
        emb = net.lg_mlp(torch.cat([emb, self.lg_anchors_n], 2))
        lg_pts = net.lg_pts(emb).view(1, 24, 12, 2) * 30.0 + self.lg_anchor_xy
        lg_meta = net.lg_meta(emb)
        pair = torch.cat([emb.unsqueeze(2).expand(-1, -1, 24, -1),
                          emb.unsqueeze(1).expand(-1, 24, -1, -1)], 3)
        lg_adj = net.lg_adj(pair)[..., 0]
        return (lane, dlog, seg2d, net.hm_head(det), net.reg_head(det),
                hm2d[0], reg2d[0], ego, occ, net.traj_head(tfeat),
                net.stat_head(det), tl, risk, flow,
                lg_pts, lg_meta, lg_adj, bev)


def build(ckpt):
    net = MODELS["v29"](n_seg=21)
    sd = torch.load(ckpt, map_location="cpu")["model"]
    cur = net.state_dict()
    sd = {k: v for k, v in sd.items()
          if k in cur and cur[k].shape == v.shape}
    net.load_state_dict(sd, strict=False)
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
    ap.add_argument("--out", default="out/meteor_v29.onnx")
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
    pb = torch.zeros(1, HIST_N, 96, BEV_H, BEV_W)
    th = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.005]]]
                      ).repeat(HIST_N, 1, 1)[None]
    ex = (imgs, K, Tc, v0, pb, th)

    with torch.no_grad():
        ref = dep(*ex)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.onnx.export(
        dep, ex, args.out, opset_version=17,
        input_names=["imgs", "K", "T_cam_ego", "v0", "hist_bev",
                     "hist_theta"],
        output_names=OUT_NAMES, do_constant_folding=True)
    print("exported", args.out, "%.1f MB" % (os.path.getsize(args.out) / 2**20))

    if args.check:
        import numpy as np
        import onnxruntime as ort
        s = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
        outs = s.run(None, {"imgs": imgs.numpy(), "K": K.numpy(),
                            "T_cam_ego": Tc.numpy(), "v0": v0.numpy(),
                            "hist_bev": pb.numpy(),
                            "hist_theta": th.numpy()})
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

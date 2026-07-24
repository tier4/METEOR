#!/usr/bin/env python3
"""Export the v36 model (12 tasks + optional LiDAR + kinematics) to ONNX.

Direct trace of the real forward (all optional branches resolved at trace
time): inputs gain lidar_bev [1,4,400,250] and kin [1,3,3] next to the
v29-era six; feeding zeros for both is bit-equal to camera-only (v31/v32
gates + zero-init kin residual), so ONE engine serves every sensor config.
Outputs = the 18 v36 heads + raw_bev for the streaming ring.

Usage: python3 deploy/export_v36.py --ckpt out/bevlane_ckpt_r30/last.pt \
           --out out/meteor_v36.onnx [--check]
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bevlane.model import MODELS  # noqa: E402

OUT_NAMES = ["lane", "depth", "seg2d", "hm", "reg", "hm2d", "reg2d",
             "ego", "occ", "traj", "stationary", "tl", "risk", "flow",
             "lg_pts", "lg_meta", "lg_adj", "unk", "raw_bev"]
IN_NAMES = ["imgs", "K", "T_cam_ego", "v0", "hist_bev", "hist_theta",
            "lidar_bev", "kin"]


_F_avg = None


def _manual_adaptive_pool(x, out):
    """Export-safe adaptive_avg_pool2d: centre-crop to a divisible size,
    then fixed-kernel avg_pool. Sub-percent numeric difference on the
    attention-pool branches; parity is reported by --check."""
    th, tw = out if isinstance(out, (tuple, list)) else (out, out)
    if th == 1 and tw == 1:
        # global average -> ReduceMean: exact, and TensorRT rejects an
        # avg_pool kernel of 400x250 (> MAX_KERNEL_DIMS_PRODUCT)
        return x.mean((-2, -1), keepdim=True)
    H, W = int(x.shape[-2]), int(x.shape[-1])
    kh, kw = H // th, W // tw
    ch, cw_ = kh * th, kw * tw
    y0, x0 = (H - ch) // 2, (W - cw_) // 2
    x = x[..., y0:y0 + ch, x0:x0 + cw_]
    return _F_avg(x, (kh, kw), (kh, kw))


def _manual_affine_grid(theta, size, align_corners=False):
    """ONNX-exportable replacement for F.affine_grid (align_corners=False
    convention), matching PyTorch exactly for 4-D inputs."""
    N, C, H, W = size
    dev, dt = theta.device, theta.dtype
    xs = (torch.arange(W, device=dev, dtype=dt) + 0.5) * 2.0 / W - 1.0
    ys = (torch.arange(H, device=dev, dtype=dt) + 0.5) * 2.0 / H - 1.0
    base = torch.stack([xs.view(1, -1).expand(H, W),
                        ys.view(-1, 1).expand(H, W),
                        torch.ones(H, W, device=dev, dtype=dt)], -1)
    return torch.einsum("hwk,nok->nhwo", base, theta)


class Wrap(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, imgs, K, T, v0, hist_bev, hist_theta, lidar_bev, kin):
        out = list(self.m(imgs, K, T, v0, hist_bev, hist_theta,
                          lidar=None, lidar_bev=lidar_bev, kin=kin))
        # 2D det heads are 3-scale lists; export the finest scale only so
        # the 19 output names stay aligned (viz uses single-scale decode)
        if isinstance(out[5], (list, tuple)):
            out[5] = out[5][0]
        if isinstance(out[6], (list, tuple)):
            out[6] = out[6][0]
        return tuple(out) + (self.m._last_bev,)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", default="v36")
    ap.add_argument("--out", default="out/meteor_v36.onnx")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    m = MODELS[args.model](n_seg=21).eval()
    sd = torch.load(args.ckpt, map_location="cpu")["model"]
    cur = m.state_dict()
    sd = {k: v for k, v in sd.items() if k in cur and cur[k].shape == v.shape}
    missing, _ = m.load_state_dict(sd, strict=False)
    print(f"loaded {args.ckpt} (missing={len(missing)})", flush=True)
    import torch.nn.functional as F
    global _F_avg
    _F_avg = F.avg_pool2d
    F.affine_grid = _manual_affine_grid          # export-safe warp
    F.adaptive_avg_pool2d = _manual_adaptive_pool
    w = Wrap(m).eval()
    ex = (torch.randn(1, 8, 3, 432, 768),
          torch.randn(1, 8, 3, 3),
          torch.randn(1, 8, 4, 4),
          torch.tensor([5.0]),
          torch.zeros(1, 3, 96, 800, 500),
          torch.tensor([[[[1.0, 0, 0], [0, 1.0, 0]]] * 3]),
          torch.zeros(1, 4, 400, 250),
          torch.zeros(1, 3, 3))
    with torch.no_grad():
        ref = w(*ex)
    print(f"{len(ref)} outputs, exporting...", flush=True)
    torch.onnx.export(w, ex, args.out, opset_version=17,
                      input_names=IN_NAMES, output_names=OUT_NAMES,
                      do_constant_folding=True)
    print(f"saved {args.out}", flush=True)
    if args.check:
        import onnxruntime as ort
        sess = ort.InferenceSession(args.out,
                                    providers=["CPUExecutionProvider"])
        outs = sess.run(None, {n: e.numpy() for n, e in zip(IN_NAMES, ex)})
        worst = 0.0
        for r, o in zip(ref, outs):
            worst = max(worst, float(np.abs(r.detach().numpy() - o).max()))
        print(f"ORT parity max diff: {worst:.2e}", flush=True)


if __name__ == "__main__":
    main()

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

import math
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bevlane.model import MODELS, BEV_H, BEV_W  # noqa: E402

# Legacy (hand-written wrapper) names.
OUT_NAMES = ["lane", "depth", "seg2d", "hm", "reg", "hm2d", "reg2d",
             "ego", "occ", "traj", "stationary", "tl", "risk", "flow",
             "lg_pts", "lg_meta", "lg_adj", "unk", "pl", "raw_bev"]
# Names for the model's OWN forward (MeteorExport). The 2D detection head is
# MULTI-SCALE: out[5] and out[6] are 3-tuples, which ONNX flattens, so the
# graph has 23 outputs and a 20-name list silently mislabelled everything after
# `reg` -- the tensor called "ego" was really a 2D heatmap and the ego vector
# came out as "tl". Names must match the flattened order.
OUT_NAMES_MODEL = (
    ["lane", "depth", "seg2d", "hm", "reg"]
    + [f"hm2d_s{i}" for i in range(3)] + [f"reg2d_s{i}" for i in range(3)]
    + ["ego", "occ", "traj", "stationary", "tl", "risk", "flow",
       "lg_pts", "lg_meta", "lg_adj", "unk", "pl", "raw_bev"])
# Without the lane graph. Not returning those three outputs makes the whole
# query-decoder subgraph dead code, which the exporter prunes -- and it is the
# single most expensive thing in the INT8 engine (10.07 ms of 31.8 ms, in a
# Transformer that INT8 cannot quantise) while the head itself has never left
# its floor (P/R ~0.01, ROADMAP A1).
OUT_NAMES_NO_LG = [n for n in OUT_NAMES_MODEL
                   if n not in ("lg_pts", "lg_meta", "lg_adj")]
HIST_N = 3


_POOL_CACHE = {}


def _pool_matrix(n_in, n_out, dtype, device):
    """Build the average-pooling matrix as a **constant**.

    2026-08-14: this used to be filled into a torch tensor by a Python loop,
    so tracing recorded one ScatterND node per assignment and the engine
    ended up with a serial chain of 86 ScatterND nodes (myelin cluster
    31 ms / 48 ms for 8cam) baked in. The values depend only on (n_in, n_out),
    so build the constant in numpy and make a single tensor. Bit-identical.
    """
    key = (n_in, n_out)
    m = _POOL_CACHE.get(key)
    if m is None:
        import numpy as _np
        a = _np.zeros((n_out, n_in), dtype=_np.float64)
        for j in range(n_out):
            s0 = (j * n_in) // n_out
            e0 = -(-((j + 1) * n_in) // n_out)      # ceil
            a[j, s0:e0] = 1.0 / (e0 - s0)
        m = torch.from_numpy(a)
        _POOL_CACHE[key] = m
    return m.to(device=device, dtype=dtype)


def exact_adaptive_avg_pool2d(x, out_size):
    """adaptive_avg_pool2d as two MatMuls -- exportable and bit-exact.

    ONNX refuses adaptive_avg_pool2d when the output is not a factor of the
    input, and the model pools 800x500 -> 25x16 (500/16 = 31.25). For a STATIC
    input shape the operator is linear: out = Wy @ x @ Wx^T with Wy/Wx holding
    the same averaging ranges, so this is not an approximation.
    """
    oh, ow = (out_size, out_size) if isinstance(out_size, int) else out_size
    H, W = x.shape[-2], x.shape[-1]
    if oh == H and ow == W:
        return x
    wy = _pool_matrix(H, oh, x.dtype, x.device)
    wx = _pool_matrix(W, ow, x.dtype, x.device)
    return torch.matmul(wy, torch.matmul(x, wx.transpose(0, 1)))


class _PatchPool:
    """Swap F.adaptive_avg_pool2d for the matmul form during export."""

    def __enter__(self):
        self.orig = F.adaptive_avg_pool2d
        F.adaptive_avg_pool2d = exact_adaptive_avg_pool2d
        torch.nn.functional.adaptive_avg_pool2d = exact_adaptive_avg_pool2d
        return self

    def __exit__(self, *a):
        F.adaptive_avg_pool2d = self.orig
        torch.nn.functional.adaptive_avg_pool2d = self.orig


class MeteorExport(torch.nn.Module):
    """Export wrapper that runs the MODEL'S OWN forward.

    The hand-written MeteorDeploy below reimplemented the forward for ONNX and
    fell behind the model every time a round added something: measured against
    v48 it differed on lane (78 logits, the built-in refiner graft), hm/reg,
    stationary, the lane-graph outputs (a query decoder replaced the anchor
    MLP), unk, traj and ego. Reimplementation is the wrong tool -- the only
    genuinely ONNX-hostile op is F.affine_grid in temporal_fuse, so override
    just that and let every other line be the model's.

    The optional-input stems are simply not exercised (lidar / lidar_bev /
    sdmap / tl are never passed), which is bit-identical to their absence and
    is the camera-only configuration the vehicle runs.
    """

    def __init__(self, net, drop_lg=False, drop=(), uint8_in=False,
                 argmax_out=False, lane_logits=False, no_hist=False,
                 depth_mean=False, bev_tokens=False):
        super().__init__()
        self.net = net
        self.drop_lg = drop_lg
        # --no-hist (2026-09-05): if the history slots are always zero (training
        # condition = METEOR_ZERO_HIST=1 on the vehicle), the 1x1 convs tgate /
        # tfuse3[0] on cat[bev,0,0,0] are mathematically identical using only the
        # first BEV_CH input-channel weights. Drop the 3 grid_samples, the hist
        # input (460MB/frame of IO) and the raw_bev output; accuracy unchanged by construction (ORT equivalence check).
        self.no_hist = no_hist
        if no_hist:
            import copy
            C = net.tfuse3[0].in_channels // (1 + HIST_N)
            tg = copy.deepcopy(net.tgate)
            tg_slim = torch.nn.Conv2d(C, tg.out_channels, 1, bias=tg.bias is not None)
            tg_slim.weight.data.copy_(tg.weight.data[:, :C])
            if tg.bias is not None:
                tg_slim.bias.data.copy_(tg.bias.data)
            tf = copy.deepcopy(net.tfuse3)
            c0 = tf[0]
            c0_slim = torch.nn.Conv2d(C, c0.out_channels, 1, bias=c0.bias is not None)
            c0_slim.weight.data.copy_(c0.weight.data[:, :C])
            if c0.bias is not None:
                c0_slim.bias.data.copy_(c0.bias.data)
            tf[0] = c0_slim
            self.tgate_slim = tg_slim.eval()
            self.tfuse3_slim = tf.eval()
        # Bake the host-side pre/post work INTO the graph. Measured on the
        # Orin realtime loop: CPU normalise costs ~30 ms and D2H of the fat
        # fp32 outputs (depth 37 MB + seg2d 24 MB) plus their CPU argmax
        # another ~30-50 ms. A uint8 input with /255 inside, and ArgMax+Cast
        # on lane/seg2d/depth at the tail, remove both without a line of
        # custom CUDA -- the same reduce-before-transfer idea the workstation
        # realtime pipeline proved (40.5 MB -> 166 KB).
        self.uint8_in = uint8_in
        self.argmax_out = argmax_out
        self.lane_logits = lane_logits
        self.depth_mean = depth_mean
        # --bev-tokens (2026-09-10): the fused BEV pooled to 25x16 (96 x 400 fp16) as an extra
        # tail output -- the interface the METEOR-VLA reads (same pooling as the VLA training dump)
        self.bev_tokens = bev_tokens
        # Dropping an output makes the subgraph that only feeds it dead code,
        # which the exporter prunes -- the same mechanism --no-lanegraph uses,
        # and the reason that one is worth 10 ms. This generalises it so the
        # cost of each head can be MEASURED instead of argued about.
        self.drop = set(drop)
        ys = (torch.arange(BEV_H, dtype=torch.float32) + 0.5) / BEV_H * 2 - 1
        xs = (torch.arange(BEV_W, dtype=torch.float32) + 0.5) / BEV_W * 2 - 1
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        base = torch.stack([gx, gy, torch.ones_like(gx)], -1)
        self.register_buffer("base_grid", base.reshape(1, -1, 3))

    def forward(self, imgs, K, T_cam_ego, v0, hist_bev=None, hist_theta=None,
                hist_bev1=None, hist_bev2=None, lidar_bev=None, lidar_flag=None):
        net = self.net
        bg = self.base_grid
        # --with-lidar (2026-09-08): optional input lidar_bev [1,4,400,250] is added to the BEV
        # through lidar_stem (same as the model's bev_extra; zero input == camera-only).
        _lb_kw = {"lidar_bev": lidar_bev} if lidar_bev is not None else {}
        if lidar_bev is not None:
            # Orin measurement (2026-09-08): the model's bev_extra broadcasts flag [B,1,1,1] to
            # 800x500x96 before multiplying, costing 14.8 ms in a single Myelin ReplAbsSum node.
            # Multiply by flag at 400x250 before interpolation (linear, so numerically identical) -> overhead ~1 ms.
            def _bev_extra_export(bev, _net=net):
                lb = _net._lidar_bev
                if lb is None:
                    return bev
                lb = lb.to(bev.dtype)
                # Second profile (2026-09-08): the in-graph abs().sum() reduction (ReplAbsSum) costs 14.8 ms.
                # The host knows the flag (whether LiDAR was passed), so take it as a scalar input lidar_flag [1].
                flag = lidar_flag.to(bev.dtype).view(1, 1, 1, 1)
                res = F.interpolate(_net.lidar_stem(lb) * flag, bev.shape[-2:],
                                    mode="bilinear", align_corners=False)
                return bev + res
            net.bev_extra = _bev_extra_export
        if self.no_hist:
            def temporal_fuse_nohist(bev):
                # Static-shape zero constant (zeros_like depends on Shape; in the graph with the LiDAR input the
                # abs().sum check on traj_feat was not constant-folded and cost 14.8 ms in Myelin; 2026-09-08)
                net._warped0 = torch.zeros(1, int(bev.shape[1]), BEV_H, BEV_W,
                                           dtype=bev.dtype, device=bev.device)
                net._prefuse_bev = bev
                g = self.tgate_slim(bev).softmax(1)
                return bev + self.tfuse3_slim(bev * (4.0 * g[:, 0:1]))
            net.temporal_fuse = temporal_fuse_nohist
            if self.uint8_in:
                imgs = imgs.to(torch.float32) / 255.0
            out = list(net(imgs, K, T_cam_ego, v0, **_lb_kw))
            return self._finish(out, imgs, with_raw=False)
        # --split-hist: the 3 slots arrive as separate inputs. We only index this
        # list instead of the tensor, so the compute graph is identical to the unsplit one.
        _hs = ([hist_bev, hist_bev1, hist_bev2]
               if hist_bev1 is not None else None)

        def temporal_fuse(bev):
            cat = [bev]
            for i in range(HIST_N):
                grid = torch.matmul(bg.to(bev.dtype),
                                    hist_theta[:, i].to(bev.dtype)
                                    .transpose(1, 2))
                grid = grid.reshape(1, BEV_H, BEV_W, 2)
                _h = _hs[i] if _hs is not None else hist_bev[:, i]
                cat.append(F.grid_sample(_h.to(bev.dtype), grid,
                                         align_corners=False))
            net._warped0 = cat[1]
            net._prefuse_bev = bev             # read by delta-stat
            g = net.tgate(torch.cat(cat, 1)).softmax(1)
            cat = [c * (4.0 * g[:, i:i + 1]) for i, c in enumerate(cat)]
            return bev + net.tfuse3(torch.cat(cat, 1))

        net.temporal_fuse = temporal_fuse          # instance-level override
        if self.uint8_in:
            imgs = imgs.to(torch.float32) / 255.0
        # net() has temporal_fuse overridden, so the contents of hist_bev are
        # not used (only the shape has to match).
        out = list(net(imgs, K, T_cam_ego, v0, hist_bev, hist_theta, **_lb_kw))
        return self._finish(out, imgs, with_raw=True)

    def _finish(self, out, imgs, with_raw=True):
        net = self.net
        _lane_logit = out[0] if self.lane_logits else None
        _depth_mean = None
        if self.depth_mean:
            # Expected depth [B,N,h,w] (fp16): renormalised expectation over the modal bin +-2 (as in the old demo_rgbd_bev).
            # The plain full expectation is not used: at object edges foreground/background mix into a bogus mid-range depth.
            # Fix for the 2D unknown-object BEV projection (unk2d) missing the range with a single argmax-bin pixel (2026-09-07).
            # Computed at half resolution after a 2x2 mean pool (+2.5 ms -> ~0.6 ms; enough for the unk2d range lookup, 2026-09-08)
            _B0, _N0, _D, _h0, _w0 = out[1].shape
            _dl = torch.nn.functional.avg_pool2d(out[1].float().reshape(_B0 * _N0, _D, _h0, _w0), 2)
            _dl = _dl.reshape(_B0, _N0, _D, _dl.shape[-2], _dl.shape[-1])   # [B,N,D,h/2,w/2]
            _bins = torch.exp(torch.linspace(math.log(1.0), math.log(79.75), _D,
                                             device=_dl.device)).view(1, 1, _D, 1, 1)
            _pr = _dl.softmax(2)
            _pk = _pr.argmax(2, keepdim=True)
            _ar = torch.arange(_D, device=_dl.device).view(1, 1, _D, 1, 1)
            _pw = _pr * ((_ar - _pk).abs() <= 2).to(_pr.dtype)
            _depth_mean = ((_pw * _bins).sum(2) / _pw.sum(2).clamp(min=1e-6)).half()
        if self.argmax_out:
            # lane [B,9,H,W] / depth [B,N,64,h,w] / seg2d [B,N,21,h,w]
            out[0] = out[0].argmax(1).to(torch.uint8)
            out[1] = out[1].argmax(2).to(torch.uint8)
            out[2] = out[2].argmax(2).to(torch.uint8)
        if self.drop_lg:
            del out[14:17]          # lg_pts / lg_meta / lg_adj
        if self.drop:
            # eager slot names (hm2d/reg2d are one tuple slot each). Zipping the
            # flat names misaligns and drops the wrong outputs (bit us in practice, 2026-08-13).
            eager = ["lane", "depth", "seg2d", "hm", "reg", "hm2d", "reg2d",
                     "ego", "occ", "traj", "stationary", "tl", "risk",
                     "flow", "lg_pts", "lg_meta", "lg_adj", "unk", "pl"]
            if self.drop_lg:
                eager = [n for n in eager
                         if n not in ("lg_pts", "lg_meta", "lg_adj")]
            assert len(eager) == len(out), \
                f"eager names {len(eager)} != outputs {len(out)}"
            out = [o for o, n in zip(out, eager) if n not in self.drop]
        # The streaming memory needs the PRE-fusion BEV back out, otherwise a
        # host running the engine cannot fill hist_bev on the next frame and
        # has to feed zeros (losing the temporal fusion entirely).
        # raw_bev is appended at the tail on return, so lane_logit goes after
        # it (appending to out shifts the names by one -- 2026-08-19 real
        # failure: raw_bev and lane_logit swapped and the history ring was disabled).
        tail = (net._last_bev,) if with_raw else ()
        if _lane_logit is not None:
            tail = tail + (_lane_logit,)
        if _depth_mean is not None:
            tail = tail + (_depth_mean,)          # tail (after lane_logit)
        if self.bev_tokens:
            tail = tail + (F.adaptive_avg_pool2d(net._fused_bev, (25, 16)).half(),)   # bev_tok [B,96,25,16]
        return tuple(out) + tail


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
        # B2 gated temporal fusion: the four slots are re-weighted by a
        # softmax gate before tfuse3. The v29-era wrapper summed them
        # unweighted, which is why `lane` differed from the model by 78 logits.
        g = net.tgate(torch.cat(cat, 1)).softmax(1)
        cat_g = [c * (4.0 * g[:, i:i + 1]) for i, c in enumerate(cat)]
        fused = bev + net.tfuse3(torch.cat(cat_g, 1))
        net._fused_bev = fused
        det = net.det_stem(net._last_bev)          # v25 routing: det on RAW
        net._det_feat = det
        lane = net.dec(net._last_bev)
        seg2d, dlog = net._aux2d
        hm2d, reg2d = net.det2d_forward(net._f_s4, 1, imgs.shape[1])
        pooled = net.ego_stem(fused).flatten(1)
        ego = net.ego_mlp(torch.cat([pooled, v0.view(1, 1)], 1))
        # B3 ego residual: attention pooling over the fused BEV
        tok = F.adaptive_avg_pool2d(fused, (25, 16)).flatten(2).transpose(1, 2)
        qa, _ = net.ego_attn(net.ego_q.weight.unsqueeze(0), tok, tok)
        ego = ego + net.ego_delta(qa.flatten(1))
        # v36 kinematic-history residual. It is NOT optional: even with no
        # history the feature carries v0/15 and the trained layer contributes a
        # bias, so omitting it shifted the exported plan by 0.42 m.
        kf = torch.zeros(1, 7, dtype=ego.dtype, device=ego.device)
        kf[:, 0] = v0.view(-1).to(ego.dtype) / 15.0
        ego = ego + net.kin_delta(kf)
        crop = net._last_bev[:, :, 200:600, 50:450]
        of = net.occ_stem(crop)
        o = net.occ_head(of)
        occ = o.view(1, 10, 16, o.shape[-2], o.shape[-1])
        flow = net.flow_head(of)
        # traj head is class-conditioned: motion stem + detection feature.
        # traj_stem takes [fused | motion] (192ch) since v30 -- the v29-era
        # wrapper fed only `fused` (96ch) and the export failed outright once a
        # current checkpoint was used. `motion` is the difference against the
        # FIRST warped history slot, zeroed when that slot is missing, exactly
        # as model.traj_feat() computes it.
        warped0 = cat[1]
        valid = (warped0.abs().sum(1, keepdim=True) > 0).to(bev.dtype)
        mot = (net._last_bev - warped0) * valid
        reg = net.reg_head(det)
        # v33+ appends two channels of the regression head (the velocity pair)
        # to the trajectory feature: traj_head is Conv2d(258, ...).
        tfeat = torch.cat([net.traj_stem(torch.cat([fused, mot], 1)), det,
                           reg[:, 4:6]], 1)
        _traj_in = None
        # TL head reads the front WIDE + NARROW s4 features
        f = net._f_s4.view(1, imgs.shape[1], -1, *net._f_s4.shape[-2:])
        tl = net.tl_fc(net.tl_head(torch.cat([f[:, 0], f[:, 6]], 1)).flatten(1))
        risk = net.risk_head(fused[:, :, 200:600, 125:375])
        # B1 lane graph: a query decoder over a pooled ROI memory replaced the
        # v29 tower + anchor-MLP path (out[14..16] are overwritten there), so
        # the old path exported a graph the model no longer uses.
        roi = net._last_bev[:, :, 100:450, 125:375]
        mem = F.adaptive_avg_pool2d(net.lg_in(roi), (22, 16)) \
            .flatten(2).transpose(1, 2)
        emb = net.lgdec(net.lgq.weight.unsqueeze(0), mem)
        lg_pts = net.lg_pts2(emb).view(1, 24, 12, 2) * 30.0 + self.lg_anchor_xy
        lg_meta = net.lg_meta2(emb)
        pair = torch.cat([emb.unsqueeze(2).expand(-1, -1, 24, -1),
                          emb.unsqueeze(1).expand(-1, 24, -1, -1)], 3)
        lg_adj = net.lg_adj(pair)[..., 0]
        # v43+: dense unknown-object head reads the trajectory feature;
        # v48: pseudo-LiDAR raster predicted from the camera-only BEV. Both are
        # heads of the shipped model, so both are exported. The optional-INPUT
        # stems (lidar/sdmap/tl) are not in this graph at all -- zeros through
        # them are bit-identical to their absence and the deployed rig is
        # camera-only, so they would only add dead inputs.
        # B4 lite: an agent-attention token shifts the traj/stat feature
        dt = F.adaptive_avg_pool2d(det, (25, 16)).flatten(2).transpose(1, 2)
        ag, _ = net.agent_attn(net.agent_q.weight.unsqueeze(0), dt, dt)
        tf256 = tfeat[:, :256] + net.agent_delta(ag.mean(1)[:, :, None, None])
        net._tf = tfeat
        unk = net.unk_dense(tfeat[:, :256])
        pl = net.pl_head(net._last_bev)
        # v42+ grafts a MultiTaskRefiner INSIDE the model and overwrites
        # seg / hm / reg / ego with its refined values. The v29-era wrapper had
        # no idea, which is why `lane` differed from the model by 78 logits.
        # This is part of the shipped network, not the optional post-hoc
        # refiner, so it belongs in the engine.
        hm = net.hm_head(det)
        rr = net.refiner(seg=lane.float(), hm=hm.float(), reg=reg.float(),
                         ego=ego.float(), v0=v0.view(1), fused=fused.float(),
                         seg_ctx=None)
        lane, hm, reg, ego = rr["seg"], rr["hm"], rr["reg"], rr["ego"]
        return (lane, dlog, seg2d, hm, reg,
                hm2d[0], reg2d[0], ego, occ,
                net.traj_head(torch.cat([tf256, reg[:, 4:6]], 1)),
                # v33+: the stationary flag comes from stat_head2 on the
                # trajectory feature, not the frozen v26 stat_head on `det`.
                net.stat_head2(tf256), tl, risk, flow,
                lg_pts, lg_meta, lg_adj, unk, pl, bev)


def build(ckpt, mv=None, seg_bias=""):
    """Deployment graph for the shipped model version.

    Was hardcoded to v29: loading a v48 checkpoint into it silently dropped
    every layer added since (unknown-dense and pseudo-LiDAR heads, the optional
    stems), so the exported engine was NOT the model being shipped. The version
    now comes from the checkpoint.

    The three OPTIONAL-input stems (lidar_stem / sdmap_stem / tl_stem) are
    deliberately left OUT of the graph: each is zero-init and additive, feeding
    zeros is bit-identical to not having them (verified per round), and the
    vehicle configuration measured here is camera-only. Removing them also
    removes their inputs from the engine signature.
    """
    try:
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch < 2.0
        ck = torch.load(ckpt, map_location="cpu")
    mv = mv or (ck.get("args") or {}).get("model") or "v29"
    net = MODELS[mv](n_seg=21)
    sd = ck["model"]
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    if any(k.startswith("lane_branch.") for k in sd):
        # the ckpt was trained with the thin-class residual branch; without
        # this the key filter below silently drops it from the export
        from bevlane.model import enable_lane_branch
        enable_lane_branch(net)
        print("[build] lane_branch attached (ckpt carries branch weights)")
    # v115 onwards: the ego global average pooling is replaced by a depthwise conv
    # (root fix for the INT8 collapse where the output reused the input's scale).
    # If the ckpt carries those weights, convert to the same form before loading.
    _cp = [k for k, v in sd.items()
           if k.startswith("ego_stem.") and k.endswith(".weight")
           and getattr(v, "dim", lambda: 0)() == 4 and v.shape[1] == 1]
    if _cp:
        _hw = tuple(int(x) for x in sd[_cp[0]].shape[-2:])
        net.convert_ego_pool(_hw, verbose=False)
        print(f"[build] detected conv-ified ego global pooling -> converting with {_hw}")

    if "depth_head.0.0.weight" in sd and hasattr(net, "depth_head"):
        _wck = tuple(sd[f"depth_head.{i}.0.weight"].shape[0]
                     for i in range(4)
                     if f"depth_head.{i}.0.weight" in sd)
        _wcur = tuple(m[0].out_channels for m in net.depth_head[:-1])
        if len(_wck) == 4 and _wck != _wcur:
            from bevlane.model import enable_depth_slim
            enable_depth_slim(net, widths=_wck)
            print(f"[build] detected depth-slim widths {_wck} -> rebuilding")
    if any(k.startswith("sem_ego.") for k in sd):
        from bevlane.model import enable_semantic_ego
        enable_semantic_ego(net)
        print("[build] detected semantic-ego residual -> enabling")
    if any(k.startswith("delta_stat.") for k in sd):
        from bevlane.model import enable_delta_stat
        enable_delta_stat(net)
        print("[build] detected temporal-delta stat head -> stationary is output from the delta")
    if any(k.startswith("stat_head2.proj.") for k in sd):
        from bevlane.model import enable_quant_stat_head
        _cap = float((ck.get("args") or {}).get("stat_quant_head") or 8.0)
        enable_quant_stat_head(net, _cap)
        print(f"[build] quant-robust stationary head detected (cap={_cap:g})")

    if any(k.startswith("stat_head2.proj.") for k in sd):
        # v119 line: the stationary head is bounded (QuantRobustStatHead) from training.
        # Enable it before loading, otherwise the proj/out weights are silently dropped.
        from bevlane.model import enable_quant_stat_head
        enable_quant_stat_head(net, 8.0)
        print("[build] detected bounded stat head (trained) -> converting and loading")
    if any(k.startswith("paint_proj.") for k in sd):
        # PointPainting (2026-08-17): if the ckpt carries the paint projection, enable
        # the path with the same class set before loading. The hook's softmax/gather/1x1
        # conv enter the traced graph as-is (BEV width 96 and the lift boundary are unchanged).
        _pc = (ck.get("args") or {}).get("paint_seg") or "2,3,4,5,6,7"
        net.enable_paint_seg([int(x) for x in str(_pc).split(",")])
        print(f"[build] paint-seg enabled (classes={_pc})")
    cur = net.state_dict()
    sd = {k: v for k, v in sd.items()
          if k in cur and cur[k].shape == v.shape}
    miss = net.load_state_dict(sd, strict=False)
    if seg_bias:
        # Decision-boundary calibration (probe_seg_bias confirmed fit/val splits agree).
        # Subtracting a constant from the logits = subtracting from the final 1x1 bias. Zero runtime cost.
        with torch.no_grad():
            _b = net.dec.out[3].bias
            for _part in seg_bias.split(","):
                _c, _v = _part.split(":")
                _b[int(_c)] -= float(_v)
        print(f"[build] seg-bias baked in: {seg_bias}")
    print(f"[build] {mv}: loaded, missing {len(miss.missing_keys)} "
          f"unexpected {len(miss.unexpected_keys)}")
    net.eval()
    if hasattr(net, "rvgg"):
        # RepVGG-style nets are saved in training form (3x3+1x1+identity branches). Deploy
        # converts to the re-parameterised pure 3x3 stack (the form measured on the Orin ladder).
        import timm.utils
        net.rvgg = timm.utils.reparameterize_model(net.rvgg)
        print("[build] rvgg reparameterized to deploy form")

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


def _out_names(args):
    """Output names for this export, after --no-lanegraph and --drop."""
    if args.legacy_wrapper:
        return OUT_NAMES
    base = OUT_NAMES_NO_LG if args.no_lanegraph else list(OUT_NAMES_MODEL)
    drop = {x for x in args.drop.split(",") if x}
    # OUT_NAMES_MODEL already ends with raw_bev; do not append it again.
    names = [n for n in base if n not in drop or n == "raw_bev"]
    if getattr(args, "no_hist", False):
        names = [n for n in names if n != "raw_bev"]
    if getattr(args, "lane_logits", False):
        names.append("lane_logit")
    if getattr(args, "depth_mean", False):
        names.append("depth_mean")
    if getattr(args, "bev_tokens", False):
        names.append("bev_tok")
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="out/meteor_v29.onnx")
    ap.add_argument("--fp16", action="store_true",
                    help="also write a weight-fp16 copy next to --out")
    ap.add_argument("--model", default=None,
                    help="model version; default reads args.model from the "
                         "checkpoint and falls back to v29, which silently "
                         "exported the wrong architecture for checkpoints that "
                         "do not carry args (the distillation output)")
    ap.add_argument("--seg-bias", default="",
                    help="bake the BEV Seg decision-boundary calibration into the final bias. "
                         "e.g. 4:0.75,5:1.0,6:0.5 (measured: mIoU +1.0-1.4%, "
                         "laneline +7-15%, line-width ratio 3.1->1.05)")
    ap.add_argument("--drop", default="",
                    help="comma-separated output names to leave out, e.g. "
                         "occ,pl,unk. Their subgraphs then get pruned.")
    ap.add_argument("--no-lanegraph", action="store_true", default=True,
                    help="(default) drop the lane-graph outputs so its query "
                         "decoder is pruned from the graph; the head was "
                         "retired (P/R ~0.01) and is not part of the release")
    ap.add_argument("--with-lanegraph", dest="no_lanegraph", action="store_false",
                    help="keep the retired lane-graph outputs in the export")
    ap.add_argument("--frustum", default=None,
                    help="bake the frustum lift for the rig of this scene "
                         "(<root>/<scene>): the exported graph then samples "
                         "only the cells each camera sees, and is valid for "
                         "THAT calibration only")
    ap.add_argument("--root", default="out/bevlane")
    ap.add_argument("--legacy-wrapper", action="store_true",
                    help="use the hand-written v29-era graph (diverges from "
                         "the model; kept only for comparison)")
    ap.add_argument("--check", action="store_true",
                    help="verify onnxruntime outputs against torch")
    ap.add_argument("--uint8-in", action="store_true",
                    help="graph takes uint8 images and divides by 255 inside "
                         "(kills the ~30 ms CPU normalise and 4x of H2D)")
    ap.add_argument("--lane-logits", action="store_true",
                    help="also emit the lane logits at the tail in addition to the argmax output "
                         "(for the full Orin seg-fuse)")
    ap.add_argument("--argmax-out", action="store_true",
                    help="lane/depth/seg2d leave the graph as uint8 argmax "
                         "(kills ~60 MB of D2H and the CPU argmax)")
    ap.add_argument("--quant-stat-head", type=float, default=0.0,
                    help="replace stat_head2 with the bounded equivalent form (QuantRobustStatHead) "
                         "before exporting. relu(z)-relu(-z)=z, so the function is "
                         "preserved for |z|<=cap. Gives TensorRT a head-specific activation range "
                         "and keeps INT8 fusion from crushing the logits (cap 8 recommended)")
    ap.add_argument("--split-hist", action="store_true",
                    help="split hist_bev [1,3,96,800,500] into 3 inputs hist_bev0/1/2. "
                         "The runtime can then bind the ring addresses directly, "
                         "which removes the per-frame 230MB D2D copy "
                         "(measured 6.2ms) entirely. Numerically identical "
                         "(indexing just becomes input splitting)")
    ap.add_argument("--with-lidar", action="store_true",
                    help="include the optional input lidar_bev [1,4,400,250] fp32 (no-hist only, 2026-09-08)")
    ap.add_argument("--bev-tokens", action="store_true",
                    help="append bev_tok [B,96,25,16] fp16 (fused BEV pooled to 25x16) -- the METEOR-VLA input (2026-09-10)")
    ap.add_argument("--depth-mean", action="store_true",
                    help="append the expected depth depth_mean [B,N,h,w] fp16 to the tail outputs (for unk2d, 2026-09-07)")
    ap.add_argument("--no-hist", action="store_true",
                    help="export without history inputs (history forced to zero = training condition). Equivalent graph "
                         "using only the first 96ch of the tgate/tfuse3 weights. The 3 grid_samples,"
                         " the hist_bev input and the raw_bev output disappear (2026-09-05)")
    ap.add_argument("--gather-lift", action="store_true",
                    help="export the lift in gather form (needs --frustum). "
                         "Aimed at Orin, where ScatterND atomics dominate.")
    ap.add_argument("--n-cams", type=int, default=8,
                    help="cameras in the exported graph. CAMS ends with "
                         "CAM_BACK_NARROW, so 7 drops exactly that one.")
    args = ap.parse_args()

    net = build(args.ckpt, mv=args.model, seg_bias=args.seg_bias)
    if args.quant_stat_head > 0:
        from bevlane.model import enable_quant_stat_head
        enable_quant_stat_head(net, args.quant_stat_head)
        print(f"[build] stat_head2 replaced with the bounded form (cap={args.quant_stat_head:g})")
    dep = (MeteorDeploy(net) if args.legacy_wrapper
           else MeteorExport(net, drop_lg=args.no_lanegraph,
                             drop=[x for x in args.drop.split(',') if x],
                             uint8_in=args.uint8_in,
                             argmax_out=args.argmax_out,
                             lane_logits=args.lane_logits,
                             depth_mean=args.depth_mean,
                             no_hist=args.no_hist,
                             bev_tokens=args.bev_tokens)).eval()
    # The network is camera-count agnostic (verified: a 7-camera forward runs
    # and emits depth for 7), so dropping CAM_BACK_NARROW is purely an
    # input-side change worth 1/8 of the backbone and the depth tower.
    NC = args.n_cams
    imgs = (torch.rand(1, NC, 3, 432, 768) * 255).to(torch.uint8) \
        if args.uint8_in else torch.randn(1, NC, 3, 432, 768)
    K = torch.eye(3).repeat(1, NC, 1, 1)
    K[:, :, 0, 0] = K[:, :, 1, 1] = 600.0
    K[:, :, 0, 2], K[:, :, 1, 2] = 384.0, 216.0
    Tc = torch.eye(4).repeat(1, NC, 1, 1)
    v0 = torch.tensor([8.0])
    pb = torch.zeros(1, HIST_N, 96, BEV_H, BEV_W)
    th = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.005]]]
                      ).repeat(HIST_N, 1, 1)[None]
    if args.frustum:
        # real calibration from a scene, then bake -- the dummy K/T above are
        # not a rig and would keep the wrong cells
        from bevlane.dataset import BevLaneDataset
        _ds = BevLaneDataset(args.root, [args.frustum], gt_key="gt_vec")
        _b = _ds[0]
        K, Tc = _b[1][None][:, :NC], _b[2][None][:, :NC]
        net.bake_frustum(K, Tc, imgs.shape[-2], imgs.shape[-1])
        if args.gather_lift:
            # scatter -> gather reformulation of the lift. Same maths up to
            # fp16 addition order (measured: depth exact, seg logits 2e-2);
            # exists because the Orin profile shows the ScatterND nodes as the
            # slowest layers of both engines (atomics on a 204 GB/s part).
            net.bake_gather()
    ex = (imgs, K, Tc, v0, pb, th)
    _in_names = ["imgs", "K", "T_cam_ego", "v0", "hist_bev", "hist_theta"]
    if args.no_hist:
        ex = (imgs, K, Tc, v0)
        _in_names = ["imgs", "K", "T_cam_ego", "v0"]
        if args.with_lidar:
            # lidar_bev is the last forward argument (hist args stay None) -> pass Nones to reach it positionally
            ex = (imgs, K, Tc, v0, None, None, None, None, torch.zeros(1, 4, 400, 250),
                  torch.ones(1))
            _in_names = ["imgs", "K", "T_cam_ego", "v0", "lidar_bev", "lidar_flag"]
    if args.split_hist:
        _h0 = torch.zeros(1, 96, BEV_H, BEV_W)
        ex = (imgs, K, Tc, v0, _h0, th, _h0.clone(), _h0.clone())
        _in_names = ["imgs", "K", "T_cam_ego", "v0", "hist_bev0",
                     "hist_theta", "hist_bev1", "hist_bev2"]

    with torch.no_grad():
        ref = dep(*ex)
        with _PatchPool():
            alt = dep(*ex)
        dmax = max(float((x if not isinstance(x, tuple) else x[0]).float()
                         .reshape(-1)
                         .sub((y if not isinstance(y, tuple) else y[0]).float()
                              .reshape(-1)).abs().max())
                   for x, y in zip(ref, alt))
        print(f"[pool] matmul adaptive-pool vs native: max|diff| {dmax:.3e}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with _PatchPool():
      # torch>=2.6 defaults to the dynamo exporter, which fails tracing
      # project_bev (bit us with torch 2.11 on the remote). Force the legacy tracer.
      _kw = {}
      import inspect as _ins
      if "dynamo" in _ins.signature(torch.onnx.export).parameters:
          _kw["dynamo"] = False
      torch.onnx.export(
        dep, ex, args.out, opset_version=17, **_kw,
        input_names=_in_names,
        output_names=_out_names(args),
        do_constant_folding=True)
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

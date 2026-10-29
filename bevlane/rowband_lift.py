#!/usr/bin/env python3
"""Row-band lift for strict NPU / edge-accelerator compilers: no Gather and no activation x activation MatMul.

  prime : average-pool ctx [B,C,V,U] and depth logits [B,D,V,U] by pool x pool -> K depth bins (1x1 Conv + Softmax)
          -> tiled product (a single Mul) -> vertical mean (at export: depthwise Conv (10,1)s10 -> (9,1)s9)
          = P [B, C*K, 1, Up]   (channel order c*K + k)
  vt    : per BEV row z, a column map Wcol[z] (Up -> X, 1x1 Conv) and depth-bin interpolation Wk[z,x,k]
          (depthwise Conv (1,K) stride K) -> bev [B, C, Z, X] -> nearest upsample x(LH/Z)
  gvt   : per BEV row z, Wg[z] (X, R, U) applied to R consecutive paint rows (starting at r0[z])
          -> g [B, CG, Z, X] -> nearest upsample x(GH/Z, GW/X)
  static: no depth; P is built with K learned row-band Convs along the vertical axis (no Mul).

Weights are initialised from the reference rig geometry (K_cam, T_cam_ego, image size) to match the Gather
version (bake_taps / ground_pix) before training. Training computes the same thing with PyTorch einsum; at export
time it is unrolled into per-row Convs (standard ONNX operators only, static shapes).
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import bevlane.meteorite as M


class RowBandLift(nn.Module):
    def __init__(self, K=4, pool=2, zrows=100, grows=3, static=False):
        super().__init__()
        self.K, self.pool, self.Z, self.R, self.static = int(K), int(pool), int(zrows), int(grows), bool(static)
        self.C, self.D, self.V, self.U = M.C_P, M.D, M.FH, M.FW
        self.Vp, self.Up = self.V // self.pool, self.U // self.pool
        self.X, self.LH = M.LW, M.LH
        self.CG, self.GH, self.GW = M.GIPM_C, M.GH, M.GW
        self.H4, self.W4 = -(-M.IMG_H // 4), -(-M.IMG_W // 4)
        assert self.LH % self.Z == 0 and self.GH % self.Z == 0 and self.GW % self.X == 0, (self.LH, self.GH, self.GW, self.Z, self.X)
        if self.static:
            self.vconv = nn.Parameter(torch.zeros(self.K, self.Vp))          # vertical weights of row band k (shared across channels)
        else:
            self.depthK = nn.Conv2d(self.D, self.K, 1)                        # 64-bin logits -> K-bin logits
        self.Wcol = nn.Parameter(torch.zeros(self.Z, self.X, self.Up))        # column map (row z, output column x, input column u')
        self.Wk = nn.Parameter(torch.zeros(self.Z, self.X, self.K))           # depth-bin interpolation
        self.Wg = nn.Parameter(torch.zeros(self.Z, self.X, self.R, self.U))   # ground: row window R x column U
        self.register_buffer("r0", torch.zeros(self.Z, dtype=torch.long))     # first row of the paint row window
        self.geo_ok = False

    # ------------------------------------------------------------------ geometry init
    @torch.no_grad()
    def init_geometry(self, Kc, T, img_hw, scale=1.0):
        """Build the same bilinear / nearest weights as the Gather version from one reference-rig camera
        (Kc [3,3] for images of size img_hw, T [4,4] ego -> camera).
        scale is a magnitude correction applied to Wcol (scale gap vs. the Gather prime; default 1, set by a calibration script)."""
        Kc = np.asarray(Kc, np.float64); T = np.asarray(T, np.float64)
        h, w = img_hw
        sx, sy = M.IMG_W / w, M.IMG_H / h
        fx, cx = Kc[0, 0] * sx, Kc[0, 2] * sx
        fy, cy = Kc[1, 1] * sy, Kc[1, 2] * sy
        D, K, FW, FS = self.D, self.K, self.U, M.FS
        # ---- vt: lift grid cells -> (pooled column u', depth-group coordinate gk)
        pts = M.lift_cell_centres()                                          # [LH*LW, 4] row 0 = far
        pc = pts @ T.T
        Xc, Zc = pc[:, 0], pc[:, 2]
        ok = Zc > M.D_MIN
        Zs = np.where(ok, Zc, 1.0)
        u = ((fx * Xc / Zs + cx) + 0.5) / FS - 0.5                           # feature column (FW scale)
        step = (math.log(M.D_MAX) - math.log(M.D_MIN)) / (D - 1)
        b = (np.log(Zs) - math.log(M.D_MIN)) / step                           # continuous fine-bin index
        ok &= (u >= 0) & (u <= FW - 1) & (b >= 0) & (b <= D - 1)
        up = (u + 0.5) / self.pool - 0.5                                      # pooled column coordinate
        gk = (b + 0.5) * (K / D) - 0.5                                        # group coordinate (group k = bins [k*D/K, (k+1)*D/K))
        Wcol = np.zeros((self.Z, self.X, self.Up)); Wk = np.zeros((self.Z, self.X, K))
        rows_per = self.LH // self.Z
        LH, LW = self.LH, self.X
        for i in range(LH):
            z = i // rows_per
            for j in range(LW):
                c = i * LW + j
                if not ok[c]:
                    continue
                u0 = int(np.floor(up[c])); fu = up[c] - u0
                for uu, wu in ((u0, 1 - fu), (u0 + 1, fu)):
                    if 0 <= uu < self.Up and wu > 1e-6:
                        Wcol[z, j, uu] += wu / rows_per
                g0 = int(np.floor(gk[c])); fg = gk[c] - g0
                for gg, wg in ((g0, 1 - fg), (g0 + 1, fg)):
                    gg = min(max(gg, 0), K - 1)
                    if wg > 1e-6:
                        Wk[z, j, gg] += wg / rows_per
        sc = np.asarray(scale, np.float64)                                   # a per-row scale [Z] is also accepted (compensates far rows whose magnitude drops to ~1/24)
        if sc.ndim:
            assert sc.shape == (self.Z,), (sc.shape, self.Z)
            sc = sc.reshape(self.Z, 1, 1)
        self.Wcol.copy_(torch.from_numpy(Wcol * sc).float())
        self.Wk.copy_(torch.from_numpy(Wk).float())
        # ---- gvt: ground cells -> nearest (v, u) at 1/4 resolution
        gpts = M.ground_cell_centres()                                       # [GH*GW, 4] row 0 = far
        pc = gpts @ T.T
        Xg, Yg, Zg = pc[:, 0], pc[:, 1], pc[:, 2]
        okg = Zg > M.D_MIN
        Zs = np.where(okg, Zg, 1.0)
        ug = (fx * Xg / Zs + cx + 0.5) / 4 - 0.5
        vg = (fy * Yg / Zs + cy + 0.5) / 4 - 0.5
        ui, vi = np.rint(ug).astype(int), np.rint(vg).astype(int)
        okg &= (ui >= 0) & (ui < self.W4) & (vi >= 0) & (vi < self.H4)
        GH, GW = self.GH, self.GW
        gr, gc = GH // self.Z, GW // self.X
        Wg = np.zeros((self.Z, self.X, self.R, self.U)); r0 = np.zeros(self.Z, np.int64)
        for z in range(self.Z):
            rows = np.arange(z * gr, (z + 1) * gr)
            vs = [vi[r * GW + j] for r in rows for j in range(GW) if okg[r * GW + j]]
            vmed = int(np.median(vs)) if vs else 0
            r0[z] = min(max(vmed - self.R // 2, 0), self.H4 - self.R)
            for r in rows:
                for j in range(GW):
                    c = r * GW + j
                    if not okg[c]:
                        continue
                    rr = min(max(vi[c] - r0[z], 0), self.R - 1)              # clamp rows outside the window to its edge (near-range approximation)
                    Wg[z, j // gc, rr, ui[c]] += 1.0 / (gr * gc)
        self.Wg.copy_(torch.from_numpy(Wg).float())
        self.r0.copy_(torch.from_numpy(r0))
        if not self.static:
            w = torch.zeros(K, D, 1, 1)
            for k in range(K):
                w[k, k * D // K:(k + 1) * D // K] = float(K) / D              # mean of in-group logits (a sum saturates the Softmax and zeroes the far groups)
            self.depthK.weight.copy_(w); self.depthK.bias.zero_()
        else:
            vc = torch.zeros(K, self.Vp)
            for k in range(K):                                               # row band k = k-th of K vertical image slices (top = far ... bottom = near)
                vc[k, k * self.Vp // K:(k + 1) * self.Vp // K] = 1.0 / (self.Vp // K)
            self.vconv.copy_(vc)
        self.geo_ok = True
        return dict(vt_cells=int(ok.sum()), g_cells=int(okg.sum()), r0=r0.tolist()[:5])

    # ------------------------------------------------------------------ forward pieces
    def prime(self, ctx, dlog):
        """ctx [B,C,V,U], dlog [B,D,V,U] (float32) -> Pt [B, Up, C*K] (channel order c*K + k)"""
        B = ctx.shape[0]
        ctxp = F.avg_pool2d(ctx, self.pool) if self.pool > 1 else ctx                      # [B,C,Vp,Up]
        if self.static:
            P = torch.einsum("kv,bcvu->bcku", self.vconv, ctxp)                            # [B,C,K,Up]
        else:
            dK = F.softmax(self.depthK(dlog), 1)
            dKp = F.avg_pool2d(dK, self.pool) if self.pool > 1 else dK                     # [B,K,Vp,Up]
            P = (ctxp.unsqueeze(2) * dKp.unsqueeze(1)).mean(3)                             # Mul -> vertical mean -> [B,C,K,Up]
        return P.reshape(B, self.C * self.K, self.Up).permute(0, 2, 1)                    # [B,Up,C*K]

    def lift(self, ctx, dlog):
        Pt = self.prime(ctx, dlog)
        y = torch.einsum("zxu,buq->bzxq", self.Wcol, Pt)                                   # [B,Z,X,C*K]
        y = y.view(y.shape[0], self.Z, self.X, self.C, self.K)
        bev = torch.einsum("bzxck,zxk->bcz x".replace(" ", ""), y, self.Wk)                # [B,C,Z,X]
        if self.Z != self.LH:
            bev = F.interpolate(bev, scale_factor=(self.LH // self.Z, 1), mode="nearest")
        return bev

    def ground(self, paint):
        """paint [B,CG,H4,W4] (float32) -> g [B,CG,GH,GW]"""
        idx = self.r0[:, None] + torch.arange(self.R, device=paint.device)[None]          # [Z,R]
        sl = paint[:, :, idx, :]                                                           # [B,CG,Z,R,U]
        g = torch.einsum("bczru,zxru->bczx", sl, self.Wg)                                  # [B,CG,Z,X]
        return F.interpolate(g, size=(self.GH, self.GW), mode="nearest")

    def extra_repr(self):
        return f"K={self.K} pool={self.pool} Z={self.Z} X={self.X} Up={self.Up} R={self.R} static={self.static}"


def attach(net, args_dict, geometry=None):
    """Create net.rowband. Geometry-initialised if geometry=(Kc, T, img_hw) is given. args_dict holds the rowband_* settings from the ckpt/CLI."""
    hl = RowBandLift(K=int(args_dict.get("rowband_lift") or 4), pool=int(args_dict.get("rowband_pool") or 2),
                     zrows=int(args_dict.get("rowband_zrows") or 100), grows=int(args_dict.get("rowband_grows") or 3),
                     static=bool(args_dict.get("rowband_static") or 0))
    hl.use_ground = bool(int(args_dict.get("rowband_ground", 1) if args_dict.get("rowband_ground") is not None else 1))   # 0: ground lift uses the original Gather (Wg kept but unused)
    if not hl.use_ground:
        hl.Wg.requires_grad_(False)                                          # exclude the unused parameter from DDP gradient sync
    info = None
    if geometry is not None:
        sf = args_dict.get("rowband_scale_file")
        scale = np.load(sf).astype(np.float64) if sf else float(args_dict.get("rowband_scale") or 1.0)
        info = hl.init_geometry(*geometry, scale=scale)
        if sf:
            info["scale_file"] = sf; info["scale_minmax"] = [float(scale.min()), float(scale.max())]
    net.rowband = hl.to(next(net.parameters()).device)
    return hl, info

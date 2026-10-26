"""Polar BEV: a lift-free design that compiles to a single compiled graph on strict NPU compilers.

Strict NPU / edge-accelerator compilers often cannot place a lift built from Gather, activation x activation
MatMul or many-branch Concat. Here the view transform is dropped altogether: camera features [C, V, U] are
collapsed vertically while keeping the column axis, giving a "row x image column" polar BEV [C_P, R, U] on which
the BEV network and heads run directly (Conv-only graph).
  mode 1 (depth bins): six vertically strided Convs collapse V=180 -> 1, a 1x1 Conv produces C*D channels, and a
                       one-hot ConvTranspose unpacks them to [C, D, U]. Not used: the NPU compiler failed to place the
                       unpack (grouped and non-grouped). Code kept for reference.
  mode 2 (image rows): a single vertical stride (2,1) stage turns [C, R=V/2=90, U] directly into the polar BEV
                       (no unpack, 3x3 Convs only). Row r <-> image row 2r (monotonic in distance for ground points).
                       Depth is supplied as dlog input channels and used implicitly by the network.
  to_cart(x, kind)   : resamples the dense polar outputs (lane / stationary / risk / traj) back to the Cartesian
                       forward grid (lift 0.4 m 200x128, lane 0.2 m 400x256) with grid_sample. Training/evaluation
                       only; it is not part of the deployed graph (the fixed remap runs on the host, or is not needed).
  boxes (hm / reg)   : trained and decoded on the polar grid (resampling Cartesian-cell regressions with grid_sample
                       performed much worse). Targets: box centre (x,y,0) -> (u, r) cell, Gaussian hm,
                       reg = (du, dr, log l, log w, sin, cos). Decoding: peak + offset -> inverse geometry
                       (pixel -> intersection with the ground plane) gives (x, y). For evaluation compatibility,
                       _finish_geo re-renders the decoded boxes as Cartesian hm/reg maps.
The geometry is fixed per rig (the reference scene's K/T are stored as buffers and saved in the ckpt)."""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import meteorite as M


def _cbr(cin, cout, k, s):
    kh, kw = k if isinstance(k, tuple) else (k, k)
    sh, sw = s if isinstance(s, tuple) else (s, s)
    return nn.Sequential(nn.Conv2d(cin, cout, (kh, kw), (sh, sw), (kh // 2, kw // 2), bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class PolarLift(nn.Module):
    def __init__(self, mode=2, use_paint=True, chans=(128, 160, 192, 256, 256, 256)):
        super().__init__()
        self.mode = int(mode)
        self.C, self.D, self.U, self.V = M.C_P, M.D, M.FW, M.FH
        self.R = self.D if self.mode == 1 else (self.V if self.mode == 3 else self.V // 2)   # polar BEV rows (mode 3: all 180 image rows = 2x far-range resolution)
        self.rdiv = 1.0 if self.mode == 3 else 2.0                          # feature row -> polar row scale (mode 2/3)
        cin = M.C_P + M.D + (M.GIPM_C if use_paint else 0)
        self.use_paint = use_paint
        if self.mode == 1:
            layers, c = [], cin
            for co in chans:
                layers.append(_cbr(c, co, 3, (2, 1))); c = co
            self.vstack = nn.Sequential(*layers)
            self.collapse = nn.Sequential(nn.Conv2d(c, c, (3, 1), 1, 0, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True))
            self.proj = nn.Conv2d(c, self.C * self.D, 1)
            nn.init.normal_(self.proj.weight, std=0.01); nn.init.zeros_(self.proj.bias)
            wu = torch.zeros(self.C * self.D, 1, self.D, 1)
            for cc in range(self.C):
                for d in range(self.D):
                    wu[cc * self.D + d, 0, d, 0] = 1.0
            self.register_buffer("unpack_w", wu, persistent=False)
        else:
            s0 = (1, 1) if self.mode == 3 else (2, 1)
            self.vstack = nn.Sequential(_cbr(cin, 128, 3, s0), _cbr(128, 128, 3, 1), _cbr(128, self.C, 3, 1))   # [C, 90 (mode 2) | 180 (mode 3), U]
        self.export_unpack = False
        self.register_buffer("grid_lift", torch.zeros(1, M.LH, M.LW, 2))                 # saved in the ckpt (evaluation can restore it without geometry)
        self.register_buffer("valid_lift", torch.zeros(1, 1, M.LH, M.LW))
        self.register_buffer("grid_lane", torch.zeros(1, 2 * M.LH, 2 * M.LW, 2))
        self.register_buffer("valid_lane", torch.zeros(1, 1, 2 * M.LH, 2 * M.LW))
        self.register_buffer("Kf", torch.zeros(4))                                       # fx, cx, fy, cy (in IMG_W x IMG_H pixels)
        self.register_buffer("Tce", torch.eye(4))                                        # cam <- ego
        self.geo_info = None
        import os as _os
        self.det_th = float(_os.environ.get("METEOR_POLAR_DET_TH", "0.3"))               # box decoding threshold at evaluation (can be lowered via env)
        self.sc_pow = float(_os.environ.get("METEOR_POLAR_SC_POW", "1.0"))               # score^pow in render_cart (monotonic: ranking unchanged, more boxes pass the eval-side 0.3 threshold)

    def has_geometry(self):
        return bool(self.valid_lift.abs().sum() > 0)

    # ---- geometry (fixed per rig) ----
    def init_geometry(self, Kc, T, img_hw):
        """Kc [3,3] (intrinsics for img_hw), T [4,4] (cam <- ego), img_hw = (h, w) of the dataset images."""
        h0, w0 = img_hw
        sx = M.IMG_W / float(w0); sy = M.IMG_H / float(h0)
        fx, cx = float(Kc[0, 0]) * sx, float(Kc[0, 2]) * sx
        fy, cy = float(Kc[1, 1]) * sy, float(Kc[1, 2]) * sy
        self.Kf.copy_(torch.tensor([fx, cx, fy, cy])); self.Tce.copy_(torch.as_tensor(np.asarray(T, np.float32)))
        out = {}
        for kind, (H, W, res) in {"lift": (M.LH, M.LW, M.LIFT_RES), "lane": (2 * M.LH, 2 * M.LW, M.LIFT_RES / 2)}.items():
            xs = M.BEV_XF - (np.arange(H) + 0.5) * res
            ys = M.BEV_YH - (np.arange(W) + 0.5) * res
            gx, gy = np.meshgrid(xs, ys, indexing="ij")
            u, r, valid = self._ground_to_polar_np(gx.reshape(-1), gy.reshape(-1))
            grid = np.stack([u / (M.FW - 1) * 2 - 1, r / (self.R - 1) * 2 - 1], -1).reshape(1, H, W, 2).astype(np.float32)
            getattr(self, f"grid_{kind}").copy_(torch.from_numpy(grid))
            getattr(self, f"valid_{kind}").copy_(torch.from_numpy(valid.reshape(1, 1, H, W).astype(np.float32)))
            out[kind] = float(valid.mean())
        self.geo_info = {"mode": self.mode, "R": self.R, "fx": fx, "cx": cx, "fy": fy, "cy": cy, "valid_frac": out}
        return self.geo_info

    def _ground_to_polar_np(self, x, y):
        """numpy: ego-frame ground point (x forward, y left) -> (feature column u, polar row r, valid)"""
        fx, cx, fy, cy = self.Kf.tolist(); T = self.Tce.cpu().numpy().astype(np.float64)
        pts = np.stack([x, y, np.zeros_like(x), np.ones_like(x)], -1) @ T.T
        X, Y, Z = pts[:, 0], pts[:, 1], pts[:, 2]
        Zs = np.clip(Z, M.D_MIN, None)
        u = ((fx * X / Zs + cx) + 0.5) / M.FS - 0.5
        if self.mode == 1:
            r = (np.log(Zs) - math.log(M.D_MIN)) / ((math.log(M.D_MAX) - math.log(M.D_MIN)) / (M.D - 1))
        else:
            r = (((fy * Y / Zs + cy) + 0.5) / M.FS - 0.5) / self.rdiv
        valid = (Z > M.D_MIN) & (u >= 0) & (u <= M.FW - 1) & (r >= 0) & (r <= self.R - 1)
        return u, r, valid

    def ground_to_polar(self, x, y):
        """torch: ego-frame ground points [N] -> (u, r, valid) [N] (training only, image-row modes)."""
        fx, cx, fy, cy = self.Kf
        T = self.Tce.to(x.dtype)
        X = T[0, 0] * x + T[0, 1] * y + T[0, 3]; Y = T[1, 0] * x + T[1, 1] * y + T[1, 3]; Z = T[2, 0] * x + T[2, 1] * y + T[2, 3]
        Zs = Z.clamp(min=M.D_MIN)
        u = ((fx * X / Zs + cx) + 0.5) / M.FS - 0.5
        r = (((fy * Y / Zs + cy) + 0.5) / M.FS - 0.5) / self.rdiv
        valid = (Z > M.D_MIN) & (u >= 0) & (u <= M.FW - 1) & (r >= 0) & (r <= self.R - 1)
        return u, r, valid

    def polar_to_ground(self, u, r):
        """torch: polar cell (u, r) [N] -> intersection (x, y) [N] with the ground plane (ego z=0). Image-row modes (pixel -> ray -> plane)."""
        fx, cx, fy, cy = self.Kf
        px = (u + 0.5) * M.FS - 0.5; py = (self.rdiv * r + 0.5) * M.FS - 0.5
        d = torch.stack([(px - cx) / fx, (py - cy) / fy, torch.ones_like(px)], -1)      # ray direction in the camera frame [N,3]
        T = self.Tce.to(u.dtype); Rm = T[:3, :3]; t = T[:3, 3]
        # ego point p = R^T (lam d - t); p_z = 0 -> lam = (R^T t)_z / (R^T d)_z
        Rt = Rm.t(); dz = (d @ Rm)[:, 2]; tz = (Rt @ t)[2]                            # (R^T d)_z, (R^T t)_z (= -camera height, negative)
        dz = torch.where(dz.abs() < 1e-3, torch.full_like(dz, -1e-3), dz)             # rays hitting the ground have dz < 0; guard near-horizontal rays (small |dz|) that go to infinity
        lam = tz / dz                                                                 # same sign -> lam > 0
        p = (lam.unsqueeze(1) * d - t) @ Rm                                           # ego point = R^T (lam d - t) (row-vector form)
        return p[:, 0], p[:, 1]

    # ---- lift ----
    def lift(self, ctx, dlog, paint=None):
        x = [ctx.float(), dlog.float()]
        if self.use_paint:
            assert paint is not None, "use_paint=True requires paint features"
            x.append(paint.float())
        x = torch.cat(x, 1)
        if self.mode in (2, 3):
            return self.vstack(x)
        y = self.proj(self.collapse(self.vstack(x)))
        if self.export_unpack:
            return F.conv_transpose2d(y, self.unpack_w.to(y.dtype), stride=(self.D, 1), groups=self.C)
        return y.view(y.shape[0], self.C, self.D, self.U)

    # ---- polar -> Cartesian (dense outputs, training/evaluation only) ----
    def to_cart(self, x, kind="lift", fill=None):
        g = getattr(self, f"grid_{kind}"); v = getattr(self, f"valid_{kind}")
        B = x.shape[0]
        y = F.grid_sample(x.float(), g.expand(B, -1, -1, -1), mode="bilinear", align_corners=True, padding_mode="zeros")
        if fill is not None:
            y = y * v + fill.to(y.dtype).view(1, -1, 1, 1) * (1.0 - v)
        else:
            y = y * v
        return y.to(x.dtype)

    # ---- boxes: targets, loss and decoding on the polar grid ----
    def det_targets(self, boxes, nbox, device):
        """boxes [B,K,6] (cls, xe, ye, l, w, yaw (already folded to 2*theta)), nbox [B] -> hm_t [B,2,R,U], reg_t [B,6,R,U], msk [B,1,R,U]"""
        B = boxes.shape[0]; R, U = self.R, self.U
        hm = torch.zeros(B, 2, R, U, device=device); reg = torch.zeros(B, 6, R, U, device=device); msk = torch.zeros(B, 1, R, U, device=device)
        rr = torch.arange(R, device=device, dtype=torch.float32).view(-1, 1); cc = torch.arange(U, device=device, dtype=torch.float32).view(1, -1)
        for bi in range(B):
            n = int(nbox[bi])
            if n <= 0: continue
            bx = boxes[bi, :n].float()
            u, r, ok = self.ground_to_polar(bx[:, 1], bx[:, 2])
            for k in range(n):
                if not bool(ok[k]): continue
                cls = 0 if float(bx[k, 0]) < 1.5 else 1                                 # as in model.build_det_targets: 1 = vehicle -> ch0, >= 2 (VRU) -> ch1
                uk, rk = float(u[k]), float(r[k]); ci, ri = int(round(uk)), int(round(rk))
                ci = min(max(ci, 0), U - 1); ri = min(max(ri, 0), R - 1)
                # Gaussian radius: horizontally the box width in columns; vertically see sr below
                _, (xq, yq) = None, self.polar_to_ground(torch.tensor([uk, uk + 1.0], device=device), torch.tensor([rk, rk], device=device))
                cell_w = float((xq[1] - xq[0]).abs() + (yq[1] - yq[0]).abs()) + 1e-3        # lateral width of one column [m]
                su = max(1.0, 0.5 * float(bx[k, 4]) / cell_w / 2.0)
                # row-direction width: half the row span between the box's front and rear ends (x +/- l/2); vehicles span
                # dozens of image rows, and a fixed sigma_r of 1.5 produced multiple peaks for one vehicle
                hl = 0.5 * float(bx[k, 3]); _, r2, _ = self.ground_to_polar(torch.tensor([float(bx[k, 1]) - hl, float(bx[k, 1]) + hl], device=device), torch.tensor([float(bx[k, 2])] * 2, device=device))
                sr = max(1.5, 0.5 * float((r2[0] - r2[1]).abs()))
                g = torch.exp(-((rr - ri) ** 2) / (2 * sr ** 2) - ((cc - ci) ** 2) / (2 * su ** 2))
                hm[bi, cls] = torch.maximum(hm[bi, cls], g)
                yaw = float(bx[k, 5])
                for dr_ in (-1, 0, 1):
                    for dc_ in (-1, 0, 1):
                        r2, c2 = ri + dr_, ci + dc_
                        if 0 <= r2 < R and 0 <= c2 < U and msk[bi, 0, r2, c2] == 0:
                            reg[bi, :, r2, c2] = torch.tensor([rk - r2, uk - c2, math.log(max(float(bx[k, 3]), 0.1)), math.log(max(float(bx[k, 4]), 0.1)), math.sin(yaw), math.cos(yaw)], device=device)
                reg[bi, :, ri, ci] = torch.tensor([rk - ri, uk - ci, math.log(max(float(bx[k, 3]), 0.1)), math.log(max(float(bx[k, 4]), 0.1)), math.sin(yaw), math.cos(yaw)], device=device)
                msk[bi, 0, ri, ci] = 1.0
        return hm, reg, msk

    def det_loss(self, hm, reg, boxes, nbox):
        """CenterNet-style focal loss + L1 on centre cells (polar version of model.boxdet_loss, plus a far-range positive weight)"""
        hm_t, reg_t, msk = self.det_targets(boxes, nbox, hm.device)
        p = hm.float().sigmoid().clamp(1e-4, 1 - 1e-4)
        pos = (hm_t > 0.99).float(); neg_w = (1 - hm_t) ** 4
        fw = self._far_weight(hm.device)                                              # up-weight far positives (recall beyond 30 m dropped otherwise)
        lpos = -(pos * fw * (1 - p) ** 2 * p.log()).sum(); lneg = -((1 - pos) * neg_w * p ** 2 * (1 - p).log()).sum()
        npos = pos.sum().clamp(min=1.0)
        lreg = ((reg.float() - reg_t).abs() * msk * fw).sum() / npos
        return (lpos + lneg) / npos + lreg

    def _far_weight(self, device):
        """Positive weight [1,1,R,U] by the cell's ground distance: x2.5 beyond 40 m, x1.5 for 25-40 m (like BOX_FAR_W in model.boxdet_loss). Built once and cached."""
        w = getattr(self, "_fw", None)
        if w is None or w.device != device:
            rr = torch.arange(self.R, device=device, dtype=torch.float32).view(-1, 1).expand(self.R, self.U).reshape(-1)
            cc = torch.arange(self.U, device=device, dtype=torch.float32).view(1, -1).expand(self.R, self.U).reshape(-1)
            x, y = self.polar_to_ground(cc, rr); d = torch.sqrt(x * x + y * y).clamp(max=200.0)
            w = torch.where(d > 40.0, 2.5, torch.where(d > 25.0, 1.5, 1.0)).view(1, 1, self.R, self.U)
            self._fw = w
        return w

    @torch.no_grad()
    def decode(self, hm, reg, th=None, topk=64):
        """polar hm [B,2,R,U] / reg [B,6,R,U] -> per-batch list of (cls, score, xe, ye, l, w, yaw) (yaw = atan2 of sin/cos, halved if YAW2)"""
        th = self.det_th if th is None else th
        p = hm.float().sigmoid(); pm = F.max_pool2d(p, (11, 5), 1, (5, 2)); p = p * (pm == p)   # NMS wider along rows (+/-5 rows, +/-2 columns): vehicles appear elongated along rows
        B, C, R, U = p.shape; out = []
        for bi in range(B):
            flat = p[bi].reshape(-1); sc, idx = flat.topk(min(topk, flat.numel())); keep = sc > th; sc, idx = sc[keep], idx[keep]
            if len(sc) == 0: out.append([]); continue
            cls = idx // (R * U); rc = idx % (R * U); ri = (rc // U).float(); ci = (rc % U).float()
            o = reg[bi, :, (rc // U), (rc % U)].float()                                    # [6, n]
            x, y = self.polar_to_ground(ci + o[1], ri + o[0])
            l, w = o[2].exp(), o[3].exp(); yaw = torch.atan2(o[4], o[5]) * (0.5 if M.YAW2 else 1.0)
            cand = sorted([(int(cls[j]), float(sc[j]), float(x[j]), float(y[j]), float(l[j]), float(w[j]), float(yaw[j])) for j in range(len(sc))], key=lambda b: -b[1])
            kept = []                                                                       # Cartesian duplicate suppression: drop lower-score boxes of the same class within 2.5 m
            for b in cand:
                if all(b[0] != k[0] or math.hypot(b[2] - k[2], b[3] - k[3]) > 2.5 for k in kept):
                    kept.append(b)
            out.append(kept)
        return out

    @torch.no_grad()
    def render_cart(self, boxes_list, B, device, dtype):
        """Re-render decoded boxes as hm/reg maps on the Cartesian lift grid (LH x LW, 0.4 m), compatible with the evaluation decode_boxes: reg = (dr, dc, log l, log w, sin, cos)"""
        hm = torch.full((B, 2, M.LH, M.LW), -10.0, device=device, dtype=torch.float32); reg = torch.zeros(B, 6, M.LH, M.LW, device=device, dtype=torch.float32)
        for bi, bl in enumerate(boxes_list):
            for cls, sc, xe, ye, l, w, yaw in bl:
                r = (M.BEV_XF - xe) / M.LIFT_RES; c = (M.BEV_YH - ye) / M.LIFT_RES; ri, ci = int(round(r)), int(round(c))
                if not (0 <= ri < M.LH and 0 <= ci < M.LW): continue
                sc_ = min(max(sc ** self.sc_pow, 1e-4), 1 - 1e-4); logit = math.log(sc_ / (1 - sc_))
                if logit > float(hm[bi, cls, ri, ci]):
                    hm[bi, cls, ri, ci] = logit
                    yy = yaw * (2.0 if M.YAW2 else 1.0)
                    reg[bi, :, ri, ci] = torch.tensor([r - ri, c - ci, math.log(max(l, 0.1)), math.log(max(w, 0.1)), math.sin(yy), math.cos(yy)], device=device)
        return hm.to(dtype), reg.to(dtype)

    def extra_repr(self):
        return f"polar mode={self.mode} C={self.C} R={self.R} U={self.U} V={self.V} paint={self.use_paint} geo={self.geo_info}"


def attach(net, args_dict, geometry=None):
    """Create net.polar and replace ego_stem's fixed pool for the polar grid (R/4 x U/4 -> 5x4). geometry=(Kc, T, img_hw). mode = args["polar"] (1|2|3)."""
    mode = int(args_dict.get("polar") or 2)
    pl = PolarLift(mode=mode, use_paint=bool(getattr(net, "gipm", False)))
    info = pl.init_geometry(*geometry) if geometry is not None else None
    dev = next(net.parameters()).device
    net.polar = pl.to(dev)
    if not isinstance(net.ego_stem[2], nn.AdaptiveAvgPool2d):           # keep as is with EGO_GAP (GlobalAveragePool)
        h8, w8 = -(-pl.R // 8), -(-M.FW // 8)
        assert h8 >= 5 and w8 >= 4, (h8, w8)
        net.ego_stem[2] = nn.AvgPool2d((h8 - 4, w8 // 4), (1, w8 // 4)).to(dev)
    return pl, info


def ensure_geometry(net, root=None, scene=None):
    """Evaluation side: if the ckpt has no grid (older ckpt) or it is all zeros, rebuild the geometry from the reference scene's K/T. root defaults to METEOR_BEV_ROOT."""
    pl = getattr(net, "polar", None)
    if pl is None or pl.has_geometry():
        return None
    import os
    from .dataset import BevLaneDataset
    root = root or os.environ.get("METEOR_BEV_ROOT", "out/bevlane")
    scene = scene or os.environ.get("METEOR_REF_SCENE", "")
    if not scene:
        raise SystemExit("polar lift: pass the reference-rig scene (scene=... or METEOR_REF_SCENE)")
    kw = {}
    hr = os.environ.get("METEOR_HR_ROOT"); sc = os.environ.get("METEOR_IMG_SCALE")
    if hr: kw["hr_root"] = hr
    if sc: kw["img_scale"] = float(sc)
    ds = BevLaneDataset(root, [scene], augment=False, max_per_scene=1, trim_start=3, trim_end=10, n_cams=1, **kw)
    b = ds[0]
    info = pl.init_geometry(b[1][0].numpy(), b[2][0].numpy(), tuple(b[0].shape[-2:]))
    print(f"[polar] geometry rebuilt from {scene}: {info}", flush=True)
    return info

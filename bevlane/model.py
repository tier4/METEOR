"""IPM-based multi-camera BEV segmentation network.

Camera features (ResNet18, stride-8) are sampled onto a BEV grid via
ground-plane (z=0) projection using known intrinsics/extrinsics, fused across
cameras, and decoded by a small BEV U-Net into semantic logits.
"""
import math

import os
import torch
import torch.nn as nn

# 推論グラフ最適化 (2026-08-19, export 時のみ METEOR_EXPORT_FAST=1 で有効):
#   1) seg2d の nan_to_num を省略 (学習時の fp16 発散対策。固定重みの推論では
#      NaN は発生せず、Orin 実測で Isnan/Select 鎖が 4.0 ms を消費していた)
#   2) 時系列ワープ有効判定の abs-sum を 96ch -> 16ch に簡約 (実測 4.0 ms)。
#      ゼロ埋めスロット/ワープ外の検出という目的には十分。
# どちらも採否は「出力一致 + パス/det 指標不変」のゲートで判定する。
_EXPORT_FAST = os.environ.get("METEOR_EXPORT_FAST", "0") == "1"
import torch.nn.functional as F
import torchvision

N_CLASSES = 9
FEAT_GRID = 200        # legacy square feature grid (0.3 m)
BEV_SIZE = 400         # legacy square output grid (0.15 m)
BEV_HALF = 30.0
# rectangular long-range BEV (v3s/v8): +-80 m fwd, +-50 m lateral @ 0.2 m
# Overridable so the cost of a smaller BEV can be PRICED before a round is spent
# on it. Latency does not care whether the rows removed are at the front or the
# rear -- 500 rows is 500 rows -- so a symmetric shrink measures exactly what an
# asymmetric "keep 80 m ahead, cut the rear to 20 m" grid would cost, and it
# needs no dataset or loss changes to measure. Everything below derives from
# these, including DET_H/DET_W.
BEV_XF = float(os.environ.get("METEOR_BEV_XF", "80.0"))   # forward extent [m]
BEV_XR = float(os.environ.get("METEOR_BEV_XR", "80.0"))   # rear extent [m]
BEV_YH = float(os.environ.get("METEOR_BEV_YH", "50.0"))
# BEV feature channel width: the ctx conv's out_channels and therefore the
# width of the post-lift BEV feature, the temporal history slots and every
# BEV-side head input. Overridable so the R6 "ctx 96 -> 64" latency axis can
# be built without touching code (METEOR_BEV_CH=64).
BEV_CH = int(os.environ.get("METEOR_BEV_CH", "96"))
BEV_RES = 0.2
# Row 0 is the far FRONT, so BEV_XH means "where the grid starts" everywhere it
# is used and stays the forward extent when the rear is truncated.
BEV_XH = BEV_XF
BEV_H = int((BEV_XF + BEV_XR) / BEV_RES)
BEV_W = int(2 * BEV_YH / BEV_RES)                          # 800x500 by default


def crop_rows(t, h):
    """Crop a BEV-space raster's rows to h. Row 0 is the far front everywhere.

    Optional INPUT rasters (lidar_bev, sdmap) arrive from the label factory on
    the full-length grid regardless of what the network was built for, so a
    rear-truncated network has to take the front h rows of them. Cropping is
    correct rather than resampling because row 0 and the resolution are shared:
    the rows that remain are the same ground cells they always were.
    """
    if t is None or not torch.is_tensor(t) or t.shape[-2] <= h:
        return t
    return t[..., :h, :]


def bev_rows(x_hi, x_lo, h=None):
    """Row slice covering forward distance x_hi..x_lo, clamped to the grid.

    Several heads crop a fixed metric window out of the BEV feature, and they
    used to do it with absolute row numbers baked for the 800-row grid
    (occ 200:600 = +-40 m, risk 200:600, lane-graph 100:450 = +60..-10 m).
    Those numbers are silently wrong the moment the grid changes length -- on a
    rear-truncated 500-row grid, 200:600 runs off the end and the head would
    quietly receive a shorter, differently-centred window. Derive them from the
    geometry instead.
    """
    h = BEV_H if h is None else h
    res = (BEV_XF + BEV_XR) / h
    r0 = max(0, int(round((BEV_XF - x_hi) / res)))
    r1 = min(h, int(round((BEV_XF - x_lo) / res)))
    return r0, max(r1, r0 + 1)


def make_bev_points_rect(device):
    """Ego-frame ground points, rect grid. Row 0 = front(+80), col 0 = left(+50)."""
    xs = BEV_XH - (torch.arange(BEV_H, device=device) + 0.5) * BEV_RES
    ys = BEV_YH - (torch.arange(BEV_W, device=device) + 0.5) * BEV_RES
    gx = xs.view(-1, 1).expand(BEV_H, BEV_W)
    gy = ys.view(1, -1).expand(BEV_H, BEV_W)
    pts = torch.stack([gx, gy, torch.zeros_like(gx), torch.ones_like(gx)], -1)
    return pts.view(-1, 4)


def make_bev_points(device, grid=None):
    """Ego-frame ground points for the feature grid. Row 0 = front, col 0 = left."""
    g = grid or FEAT_GRID
    r = 2 * BEV_HALF / g
    ys = BEV_HALF - (torch.arange(g, device=device) + 0.5) * r  # x fwd
    xs = BEV_HALF - (torch.arange(g, device=device) + 0.5) * r  # y left
    gx = ys.view(-1, 1).expand(g, g)   # x (forward)
    gy = xs.view(1, -1).expand(g, g)   # y (left)
    pts = torch.stack([gx, gy, torch.zeros_like(gx), torch.ones_like(gx)], -1)
    return pts.view(-1, 4)                              # [G*G, 4]


class ConvBlock(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


def ipm_project(feats, K, T_cam_ego, pts, H, W, out_hw=None):
    """Sample camera features at BEV ground points. Export-friendly (no einsum).

    feats [B,N,C,fh,fw]; K [B,N,3,3]; T_cam_ego [B,N,4,4]; pts [G2,4].
    Returns fused BEV features [B,C,G,G].
    """
    B, N, C, fh, fw = feats.shape
    G2 = pts.shape[0]
    pc = torch.matmul(T_cam_ego.reshape(B * N, 4, 4),
                      pts.t().unsqueeze(0).expand(B * N, 4, G2))  # [BN,4,G2]
    x, y, z = pc[:, 0], pc[:, 1], pc[:, 2]                        # [BN,G2]
    Kf = K.reshape(B * N, 3, 3)
    valid = z > 0.5
    zc = z.clamp(min=0.5)
    u = Kf[:, 0, 0].unsqueeze(-1) * x / zc + Kf[:, 0, 2].unsqueeze(-1)
    v = Kf[:, 1, 1].unsqueeze(-1) * y / zc + Kf[:, 1, 2].unsqueeze(-1)
    valid = valid & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    valid = valid & (x * x + y * y + z * z < 95.0 ** 2)

    gu = (u / (W - 1) * 2 - 1).clamp(-2, 2)
    gv = (v / (H - 1) * 2 - 1).clamp(-2, 2)
    grid = torch.stack([gu, gv], -1).unsqueeze(2)                 # [BN,G2,1,2]
    samp = F.grid_sample(feats.reshape(B * N, C, fh, fw), grid,
                         align_corners=False).squeeze(-1)         # [BN,C,G2]
    vf = valid.unsqueeze(1).to(samp.dtype)
    samp = (samp * vf).view(B, N, C, G2).sum(1)
    cnt = vf.view(B, N, 1, G2).sum(1).clamp(min=1.0)
    if out_hw is None:
        g = int(G2 ** 0.5)
        out_hw = (g, g)
    return (samp / cnt).view(B, C, out_hw[0], out_hw[1])


class IPMSegNet(nn.Module):
    def __init__(self, n_cams=6, feat_ch=64):
        super().__init__()
        rn = torchvision.models.resnet18(weights="IMAGENET1K_V1")
        self.stem = nn.Sequential(rn.conv1, rn.bn1, rn.relu, rn.maxpool,
                                  rn.layer1, rn.layer2)   # stride 8, 128 ch
        self.reduce = nn.Sequential(nn.Conv2d(128, feat_ch, 1, bias=False),
                                    nn.BatchNorm2d(feat_ch), nn.ReLU(inplace=True))
        self.bev_pts = None
        self.dec = nn.Sequential(
            ConvBlock(feat_ch, 128), ConvBlock(128, 128),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBlock(128, 64),
            nn.Conv2d(64, N_CLASSES, 1))

    def forward(self, imgs, K, T_cam_ego):
        """imgs [B,N,3,H,W]; K [B,N,3,3]; T_cam_ego [B,N,4,4] (ego->cam)."""
        B, N, _, H, W = imgs.shape
        feats = self.reduce(self.stem(imgs.view(B * N, 3, H, W)))
        feats = feats.view(B, N, *feats.shape[1:])
        if self.bev_pts is None or self.bev_pts.device != imgs.device:
            self.bev_pts = make_bev_points(imgs.device)
        bev = ipm_project(feats, K, T_cam_ego, self.bev_pts, H, W)
        return self.dec(bev)


class IPMSegNetV2(nn.Module):
    """Capacity-scaled variant: full ResNet18 + FPN fusion to stride 8,
    128-ch BEV features, deeper decoder. ~19M params / ~460 GFLOPs."""

    def __init__(self, n_cams=6, feat_ch=128):
        super().__init__()
        rn = torchvision.models.resnet18(weights="IMAGENET1K_V1")
        self.stem = nn.Sequential(rn.conv1, rn.bn1, rn.relu, rn.maxpool, rn.layer1)
        self.layer2, self.layer3, self.layer4 = rn.layer2, rn.layer3, rn.layer4
        self.lat2 = nn.Conv2d(128, feat_ch, 1)
        self.lat3 = nn.Conv2d(256, feat_ch, 1)
        self.lat4 = nn.Conv2d(512, feat_ch, 1)
        self.fuse = nn.Sequential(nn.Conv2d(feat_ch, feat_ch, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(feat_ch), nn.ReLU(inplace=True))
        self.register_buffer("bev_pts", make_bev_points(torch.device("cpu")),
                             persistent=False)
        self.dec = nn.Sequential(
            ConvBlock(feat_ch, 256), ConvBlock(256, 256), ConvBlock(256, 256),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBlock(256, 128),
            nn.Conv2d(128, N_CLASSES, 1))

    def forward(self, imgs, K, T_cam_ego):
        B, N, _, H, W = imgs.shape
        x1 = self.stem(imgs.reshape(B * N, 3, H, W))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        f = self.lat2(x2) \
            + F.interpolate(self.lat3(x3), scale_factor=2, mode="bilinear",
                            align_corners=False) \
            + F.interpolate(self.lat4(x4), scale_factor=4, mode="bilinear",
                            align_corners=False)
        f = self.fuse(f)
        feats = f.view(B, N, *f.shape[1:])
        bev = ipm_project(feats, K, T_cam_ego, self.bev_pts, H, W)
        return self.dec(bev)


class LSSDepthNet(nn.Module):
    """LSS-style lift with supervised depth (BEVDepth-lite).

    Per-camera depth distribution (D bins) is predicted and supervised with
    LiDAR depth GT; features are lifted along rays weighted by the depth
    distribution and splatted onto the BEV grid.
    Returns (seg_logits, depth_logits).
    """
    D = 48
    D_MIN, D_STEP = 2.0, 1.0     # bins: 2 .. 50 m

    def _d2b(self, x):
        return (x - self.D_MIN) / self.D_STEP
    FH, FW = 36, 64              # stride-8 feature grid for 512x288

    def __init__(self, n_cams=6, feat_ch=128, ctx_ch=80):
        super().__init__()
        rn = torchvision.models.resnet18(weights="IMAGENET1K_V1")
        self.stem = nn.Sequential(rn.conv1, rn.bn1, rn.relu, rn.maxpool, rn.layer1)
        self.layer2, self.layer3, self.layer4 = rn.layer2, rn.layer3, rn.layer4
        self.lat2 = nn.Conv2d(128, feat_ch, 1)
        self.lat3 = nn.Conv2d(256, feat_ch, 1)
        self.lat4 = nn.Conv2d(512, feat_ch, 1)
        self.fuse = nn.Sequential(nn.Conv2d(feat_ch, feat_ch, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(feat_ch), nn.ReLU(inplace=True))
        self.depth_head = nn.Sequential(
            nn.Conv2d(feat_ch, feat_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(feat_ch), nn.ReLU(inplace=True),
            nn.Conv2d(feat_ch, self.D, 1))
        self.ctx = nn.Conv2d(feat_ch, ctx_ch, 1)
        self.dec = nn.Sequential(
            ConvBlock(ctx_ch, 256), ConvBlock(256, 256), ConvBlock(256, 256),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBlock(256, 128),
            nn.Conv2d(128, N_CLASSES, 1))
        # pixel-center rays at stride 8 (image-plane homogeneous coords)
        vs, us = torch.meshgrid(torch.arange(self.FH), torch.arange(self.FW),
                                indexing="ij")
        px = (us.float() + 0.5) * 8.0
        py = (vs.float() + 0.5) * 8.0
        self.register_buffer("pix", torch.stack(
            [px, py, torch.ones_like(px)], 0).view(3, -1), persistent=False)
        ds = self.D_MIN + torch.arange(self.D).float() * self.D_STEP
        self.register_buffer("bin_d", ds, persistent=False)

    def forward(self, imgs, K, T_cam_ego):
        B, N, _, H, W = imgs.shape
        x1 = self.stem(imgs.reshape(B * N, 3, H, W))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        f = self.lat2(x2) \
            + F.interpolate(self.lat3(x3), scale_factor=2, mode="bilinear",
                            align_corners=False) \
            + F.interpolate(self.lat4(x4), scale_factor=4, mode="bilinear",
                            align_corners=False)
        f = self.fuse(f)                                   # [BN,128,FH,FW]
        dlog = self.depth_head(f)                          # [BN,D,FH,FW]
        dprob = dlog.softmax(1)
        ctx = self.ctx(f)                                  # [BN,C,FH,FW]
        C = ctx.shape[1]
        P = self.FH * self.FW

        # rays in ego frame: ray_cam = K^-1 @ pix ; p_ego = R*(d*ray) + t
        Kf = K.reshape(B * N, 3, 3)
        T_ego_cam = torch.inverse(T_cam_ego.reshape(B * N, 4, 4))
        ray = torch.matmul(torch.inverse(Kf), self.pix.unsqueeze(0))  # [BN,3,P]
        Rm = T_ego_cam[:, :3, :3]
        tm = T_ego_cam[:, :3, 3:]
        # p[d] = Rm@ray * d + t : [BN,3,D,P]
        rr = torch.matmul(Rm, ray).unsqueeze(2)            # [BN,3,1,P]
        p = rr * self.bin_d.view(1, 1, self.D, 1) + tm.unsqueeze(2)
        gx, gy = p[:, 0], p[:, 1]                          # ego fwd / left
        res = 2 * BEV_HALF / FEAT_GRID
        row = ((BEV_HALF - gx) / res).long()
        col = ((BEV_HALF - gy) / res).long()
        valid = (row >= 0) & (row < FEAT_GRID) & (col >= 0) & (col < FEAT_GRID)
        idx = (row.clamp(0, FEAT_GRID - 1) * FEAT_GRID
               + col.clamp(0, FEAT_GRID - 1))              # [BN,D,P]

        w = dprob.view(B * N, self.D, P) * valid.to(dprob.dtype)
        feat = ctx.view(B * N, C, 1, P) * w.unsqueeze(1)   # [BN,C,D,P]
        feat = feat.reshape(B, N, C, -1)
        idx = idx.reshape(B, N, -1)
        bev = feat.new_zeros(B, C, FEAT_GRID * FEAT_GRID)
        for b in range(B):
            bev[b].index_add_(1, idx[b].reshape(-1),
                              feat[b].reshape(C, -1))
        bev = bev.view(B, C, FEAT_GRID, FEAT_GRID)
        return self.dec(bev), dlog.view(B, N, self.D, self.FH, self.FW)

    def depth_loss(self, dlog, depth_gt):
        """CE over depth bins where GT valid. depth_gt [B,N,FH,FW] metres."""
        tgt = self._d2b(depth_gt).round().long()
        valid = (depth_gt > 0.5) & (tgt >= 0) & (tgt < self.D)
        tgt = tgt.clamp(0, self.D - 1)
        tgt[~valid] = -1
        return F.cross_entropy(dlog.flatten(0, 1),
                               tgt.flatten(0, 1), ignore_index=-1)


class IPMSegNetV3(nn.Module):
    """High-resolution BEV sampling variant: IPM directly at 400x400 (0.15 m),
    decoder fully at output resolution. Targets thin-structure fidelity."""

    def __init__(self, n_cams=6, feat_ch=128):
        super().__init__()
        rn = torchvision.models.resnet18(weights="IMAGENET1K_V1")
        self.stem = nn.Sequential(rn.conv1, rn.bn1, rn.relu, rn.maxpool, rn.layer1)
        self.layer2, self.layer3, self.layer4 = rn.layer2, rn.layer3, rn.layer4
        self.lat2 = nn.Conv2d(128, feat_ch, 1)
        self.lat3 = nn.Conv2d(256, feat_ch, 1)
        self.lat4 = nn.Conv2d(512, feat_ch, 1)
        self.fuse = nn.Sequential(nn.Conv2d(feat_ch, feat_ch, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(feat_ch), nn.ReLU(inplace=True))
        self.register_buffer("bev_pts", make_bev_points_rect(torch.device("cpu")),
                             persistent=False)
        self.dec = nn.Sequential(
            ConvBlock(feat_ch, 160), ConvBlock(160, 160), ConvBlock(160, 128),
            nn.Conv2d(128, N_CLASSES, 1))

    def forward(self, imgs, K, T_cam_ego):
        B, N, _, H, W = imgs.shape
        x1 = self.stem(imgs.reshape(B * N, 3, H, W))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        f = self.lat2(x2) \
            + F.interpolate(self.lat3(x3), scale_factor=2, mode="bilinear",
                            align_corners=False) \
            + F.interpolate(self.lat4(x4), scale_factor=4, mode="bilinear",
                            align_corners=False)
        f = self.fuse(f)
        feats = f.view(B, N, *f.shape[1:])
        bev = ipm_project(feats, K, T_cam_ego, self.bev_pts, H, W,
                          out_hw=(BEV_H, BEV_W))
        return self.dec(bev)


class DepthGatedIPMNet(nn.Module):
    """TRT-friendly depth-aware BEV net (pull-based; no scatter).

    A stride-4 depth distribution is predicted per camera (supervised with
    dense LiDAR+panoptic depth). For every BEV ground cell we sample both the
    context feature and the depth probability at the projected pixel, and
    gate the feature by the probability that the pixel's depth matches the
    cell's ray distance (soft visibility). Ops: conv / grid_sample / gather —
    all natively supported by TensorRT.
    """
    D = 64
    D_MIN, D_STEP = 1.0, 1.25      # bins 1.0 .. 79.75 m; last bin = sky/far

    # D5 (2026-08-29): 対数ビン対応。enable_depth_logbins() が DEPTH_CENTERS
    # を張ると、以下のヘルパ経由の全経路 (損失・リフト可視率・期待深度・
    # LiDAR ブレンド) が対数間隔で動く。未設定なら従来式とビット同値。
    def _d2b(self, x):
        """metric 深度 -> 連続 bin 座標。"""
        c = getattr(self, "DEPTH_CENTERS", None)
        if c is None:
            return (x - self.D_MIN) / self.D_STEP
        c = c.to(device=x.device, dtype=x.dtype)
        i = torch.bucketize(x.detach(), c).clamp(1, self.D - 1)
        lo = c[i - 1]
        hi = c[i]
        return (i - 1).to(x.dtype) + ((x - lo) / (hi - lo)).clamp(0, 1)

    def _dbins(self, device=None, dtype=None):
        c = getattr(self, "DEPTH_CENTERS", None)
        if c is None:
            c = self.D_MIN + torch.arange(self.D).float() * self.D_STEP
        if device is not None:
            c = c.to(device=device, dtype=dtype if dtype is not None
                     else c.dtype)
        return c

    def __init__(self, n_cams=6, feat_ch=160, ctx_ch=BEV_CH):
        super().__init__()
        rn = torchvision.models.resnet34(weights="IMAGENET1K_V1")
        self.stem = nn.Sequential(rn.conv1, rn.bn1, rn.relu, rn.maxpool)
        self.layer1, self.layer2 = rn.layer1, rn.layer2
        self.layer3, self.layer4 = rn.layer3, rn.layer4
        self.lat1 = nn.Conv2d(64, feat_ch, 1)
        self.lat2 = nn.Conv2d(128, feat_ch, 1)
        self.lat3 = nn.Conv2d(256, feat_ch, 1)
        self.lat4 = nn.Conv2d(512, feat_ch, 1)
        self.fuse = nn.Sequential(nn.Conv2d(feat_ch, feat_ch, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(feat_ch), nn.ReLU(inplace=True))
        self.depth_head = nn.Sequential(
            nn.Conv2d(feat_ch, feat_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(feat_ch), nn.ReLU(inplace=True),
            nn.Conv2d(feat_ch, self.D, 1))
        self.ctx = nn.Conv2d(feat_ch, ctx_ch, 1)
        self.register_buffer("bev_pts", make_bev_points_rect(torch.device("cpu")),
                             persistent=False)
        self.dec = nn.Sequential(
            ConvBlock(ctx_ch, 160), ConvBlock(160, 160), ConvBlock(160, 128),
            nn.Conv2d(128, N_CLASSES, 1))

    def image_feats(self, imgs):
        B, N, _, H, W = imgs.shape
        x0 = self.stem(imgs.reshape(B * N, 3, H, W))
        x1 = self.layer1(x0)      # stride 4
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        sz = x1.shape[-2:]        # handle input sizes not divisible by 32
        up = lambda t: F.interpolate(t, size=sz, mode="bilinear", align_corners=False)
        f = self.lat1(x1) + up(self.lat2(x2)) + up(self.lat3(x3)) + up(self.lat4(x4))
        return self.fuse(f)       # [BN, C, H/4, W/4]

    def forward(self, imgs, K, T_cam_ego):
        B, N, _, H, W = imgs.shape
        f = self.image_feats(imgs)
        dlog = self.depth_head(f)                      # [BN,D,fh,fw]
        dprob = dlog.softmax(1)
        ctx = self.ctx(f)                              # [BN,Cc,fh,fw]
        Cc = ctx.shape[1]
        fh, fw = f.shape[-2:]

        pts = self.bev_pts
        G2 = pts.shape[0]
        pc = torch.matmul(T_cam_ego.reshape(B * N, 4, 4),
                          pts.t().unsqueeze(0).expand(B * N, 4, G2))
        x, y, z = pc[:, 0], pc[:, 1], pc[:, 2]
        Kf = K.reshape(B * N, 3, 3)
        valid = z > 0.5
        zc = z.clamp(min=0.5)
        u = Kf[:, 0, 0].unsqueeze(-1) * x / zc + Kf[:, 0, 2].unsqueeze(-1)
        v = Kf[:, 1, 1].unsqueeze(-1) * y / zc + Kf[:, 1, 2].unsqueeze(-1)
        valid = valid & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        dist = torch.sqrt(x * x + y * y + z * z)
        valid = valid & (dist < 90.0)

        gu = (u / (W - 1) * 2 - 1).clamp(-2, 2)
        gv = (v / (H - 1) * 2 - 1).clamp(-2, 2)
        grid = torch.stack([gu, gv], -1).unsqueeze(2)  # [BN,G2,1,2]
        ctx_s = F.grid_sample(ctx, grid, align_corners=False).squeeze(-1)
        prob_s = F.grid_sample(dprob, grid, align_corners=False).squeeze(-1)
        # soft visibility: linear interp of P(depth == cell ray distance)
        b = self._d2b(dist).clamp(0, self.D - 1 - 1e-4)
        b0 = b.floor().long()
        fr = (b - b0.float()).unsqueeze(1)
        w0 = torch.gather(prob_s, 1, b0.unsqueeze(1))
        w1 = torch.gather(prob_s, 1, (b0 + 1).clamp(max=self.D - 1).unsqueeze(1))
        w = (w0 * (1 - fr) + w1 * fr) + 0.05           # floor keeps IPM signal
        w = w * valid.unsqueeze(1).to(w.dtype)

        num = (ctx_s.view(B, N, Cc, G2) * w.view(B, N, 1, G2)).sum(1)
        den = w.view(B, N, 1, G2).sum(1).clamp(min=1e-4)
        bev = (num / den).view(B, Cc, BEV_H, BEV_W)
        return self.dec(bev), dlog.view(B, N, self.D, fh, fw)

    def depth_loss(self, dlog, depth_gt):
        tgt = self._d2b(depth_gt).round().long()
        valid = (depth_gt > 0.1) & (tgt >= 0)
        tgt = tgt.clamp(0, self.D - 1)
        tgt[~valid] = -1
        return F.cross_entropy(dlog.flatten(0, 1),
                               tgt.flatten(0, 1), ignore_index=-1)


N_SEG = 12   # compact 2D semantic classes for auxiliary image-space seg


class DepthSegIPMNet(nn.Module):
    """v13: depth-gated IPM with higher-res input, stride-2 depth head, and a
    2D semantic-segmentation auxiliary head (intermediate representation).

    forward -> (bev_logits, depth_logits[stride-2], seg2d_logits[stride-4]).
    Ops remain conv / grid_sample / gather (TensorRT-friendly). The 2D seg head
    and depth head are auxiliary supervision that shape the shared backbone.
    """
    D = 64
    D_MIN, D_STEP = 1.0, 1.25

    # D5: 対数ビン対応ヘルパ (DepthGatedIPMNet と同一実装)
    def _d2b(self, x):
        c = getattr(self, "DEPTH_CENTERS", None)
        if c is None:
            return (x - self.D_MIN) / self.D_STEP
        c = c.to(device=x.device, dtype=x.dtype)
        i = torch.bucketize(x.detach(), c).clamp(1, self.D - 1)
        lo = c[i - 1]
        hi = c[i]
        return (i - 1).to(x.dtype) + ((x - lo) / (hi - lo)).clamp(0, 1)

    def _dbins(self, device=None, dtype=None):
        c = getattr(self, "DEPTH_CENTERS", None)
        if c is None:
            c = self.D_MIN + torch.arange(self.D).float() * self.D_STEP
        if device is not None:
            c = c.to(device=device, dtype=dtype if dtype is not None
                     else c.dtype)
        return c

    def __init__(self, n_cams=8, feat_ch=160, ctx_ch=BEV_CH, n_seg=N_SEG):
        super().__init__()
        rn = torchvision.models.resnet34(weights="IMAGENET1K_V1")
        self.stem = nn.Sequential(rn.conv1, rn.bn1, rn.relu, rn.maxpool)
        self.layer1, self.layer2 = rn.layer1, rn.layer2
        self.layer3, self.layer4 = rn.layer3, rn.layer4
        self.lat1 = nn.Conv2d(64, feat_ch, 1)
        self.lat2 = nn.Conv2d(128, feat_ch, 1)
        self.lat3 = nn.Conv2d(256, feat_ch, 1)
        self.lat4 = nn.Conv2d(512, feat_ch, 1)
        self.fuse = nn.Sequential(nn.Conv2d(feat_ch, feat_ch, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(feat_ch), nn.ReLU(inplace=True))
        # depth at stride-2 (upsample fused stride-4 feature x2 then predict)
        self.depth_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(feat_ch, feat_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(feat_ch), nn.ReLU(inplace=True))
        self.depth_head = nn.Conv2d(feat_ch, self.D, 1)
        # 2D semantic seg head at stride-4 (auxiliary)
        self.seg_head = nn.Sequential(
            nn.Conv2d(feat_ch, feat_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(feat_ch), nn.ReLU(inplace=True),
            nn.Conv2d(feat_ch, n_seg, 1))
        self.ctx = nn.Conv2d(feat_ch, ctx_ch, 1)
        self.register_buffer("bev_pts", make_bev_points_rect(torch.device("cpu")),
                             persistent=False)
        self.dec = nn.Sequential(
            ConvBlock(ctx_ch, 160), ConvBlock(160, 160), ConvBlock(160, 128),
            nn.Conv2d(128, N_CLASSES, 1))

    grad_ckpt = False          # trade ~15% speed for backbone activations

    def image_feats(self, imgs):
        B, N, _, H, W = imgs.shape
        x0 = self.stem(imgs.reshape(B * N, 3, H, W))
        if self.grad_ckpt and self.training:
            # 8 cameras of backbone activations dominate the peak; with the
            # full multi-task loss set the run sat ~1 GB under the 44 GB
            # ceiling and OOM'd on scene-dependent spikes. Recompute them in
            # the backward pass instead of storing them.
            from torch.utils.checkpoint import checkpoint as _ck
            x1 = _ck(self.layer1, x0, use_reentrant=False)
            x2 = _ck(self.layer2, x1, use_reentrant=False)
            x3 = _ck(self.layer3, x2, use_reentrant=False)
            x4 = _ck(self.layer4, x3, use_reentrant=False)
        else:
            x1 = self.layer1(x0)
            x2 = self.layer2(x1)
            x3 = self.layer3(x2)
            x4 = self.layer4(x3)
        # FUSE_STRIDE=8 (R7 解像度軸): x2 入力を stride-8 で融合する。特徴
        # グリッドは x1@stride4 と同じ 108x192 になり、lat2/3/4 の入力
        # チャンネルも既存のまま = 成熟した重みを丸ごと引き継げる。
        # クラス属性で切り替えるのは、image_feats を丸ごと上書きすると
        # 上位クラスが積んだ副作用 (_last_f / tl_stem 残差) が消えるため。
        if getattr(self, "FUSE_STRIDE", 4) == 8:
            sz = x2.shape[-2:]
            up = lambda t: F.interpolate(t, size=sz, mode="bilinear",
                                         align_corners=False)
            f = self.lat2(x2) + up(self.lat3(x3)) + up(self.lat4(x4))
            return self.fuse(f)
        sz = x1.shape[-2:]
        up = lambda t: F.interpolate(t, size=sz, mode="bilinear", align_corners=False)
        f = self.lat1(x1) + up(self.lat2(x2)) + up(self.lat3(x3)) + up(self.lat4(x4))
        return self.fuse(f)

    def forward(self, imgs, K, T_cam_ego):
        B, N, _, H, W = imgs.shape
        f = self.image_feats(imgs)                     # [BN,C,H/4,W/4]
        # Clamped. The 2D seg logits are the one thing that goes non-finite in
        # these rounds: r59 discarded 0 % of steps up to 9k, 5 % to 12k and
        # 29 % by 13.5k, every one of them reported as out[2] alone with the BEV
        # outputs clean, and r53/r56/r57 showed the same climb. In fp16 a logit
        # only has to pass 65504 to become inf, and CE saturates long before
        # +-30, so this changes nothing about what the loss sees while removing
        # the failure -- and since clamp has zero gradient outside the range it
        # also stops the head being pushed further out once it gets there.
        # clamp だけでは NaN が素通しする (clamp(nan)=nan)。fp16 で head 内部が
        # 一度 inf になると BN の (x-mean)/sqrt(var) で NaN が生まれ、
        # 出力段の clamp では消せない。2026-08-17: nan_to_num を前置し、
        # out[2] 由来の SKIP (r59 で 29%, r72 で 148 回, r73 で 16 回) を断つ。
        if _EXPORT_FAST:
            seg2d = self.seg_head(f).clamp(-30.0, 30.0)
        else:
            seg2d = torch.nan_to_num(self.seg_head(f), nan=0.0,
                                     posinf=30.0, neginf=-30.0
                                     ).clamp(-30.0, 30.0)                        # [BN,n_seg,H/4,W/4]
        dlog = self.depth_head(self.depth_up(f))       # [BN,D,H/2,W/2]
        dprob = dlog.softmax(1)
        ctx = self.ctx(f)
        Cc = ctx.shape[1]

        pts = self.bev_pts
        G2 = pts.shape[0]
        pc = torch.matmul(T_cam_ego.reshape(B * N, 4, 4),
                          pts.t().unsqueeze(0).expand(B * N, 4, G2))
        x, y, z = pc[:, 0], pc[:, 1], pc[:, 2]
        Kf = K.reshape(B * N, 3, 3)
        valid = z > 0.5
        zc = z.clamp(min=0.5)
        u = Kf[:, 0, 0].unsqueeze(-1) * x / zc + Kf[:, 0, 2].unsqueeze(-1)
        v = Kf[:, 1, 1].unsqueeze(-1) * y / zc + Kf[:, 1, 2].unsqueeze(-1)
        valid = valid & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        dist = torch.sqrt(x * x + y * y + z * z)
        valid = valid & (dist < 90.0)
        gu = (u / (W - 1) * 2 - 1).clamp(-2, 2)
        gv = (v / (H - 1) * 2 - 1).clamp(-2, 2)
        grid = torch.stack([gu, gv], -1).unsqueeze(2)
        ctx_s = F.grid_sample(ctx, grid, align_corners=False).squeeze(-1)
        prob_s = F.grid_sample(dprob, grid, align_corners=False).squeeze(-1)
        b = self._d2b(dist).clamp(0, self.D - 1 - 1e-4)
        b0 = b.floor().long()
        fr = (b - b0.float()).unsqueeze(1)
        w0 = torch.gather(prob_s, 1, b0.unsqueeze(1))
        w1 = torch.gather(prob_s, 1, (b0 + 1).clamp(max=self.D - 1).unsqueeze(1))
        w = (w0 * (1 - fr) + w1 * fr) + 0.05
        w = w * valid.unsqueeze(1).to(w.dtype)
        num = (ctx_s.view(B, N, Cc, G2) * w.view(B, N, 1, G2)).sum(1)
        den = w.view(B, N, 1, G2).sum(1).clamp(min=1e-4)
        bev = (num / den).view(B, Cc, BEV_H, BEV_W)
        fh2, fw2 = dlog.shape[-2:]
        sh, sw = seg2d.shape[-2:]
        return (self.dec(bev), dlog.view(B, N, self.D, fh2, fw2),
                seg2d.view(B, N, seg2d.shape[1], sh, sw))

    def depth_loss(self, dlog, depth_gt):
        tgt = self._d2b(depth_gt).round().long()
        valid = (depth_gt > 0.1) & (tgt >= 0)
        tgt = tgt.clamp(0, self.D - 1)
        tgt[~valid] = -1
        return F.cross_entropy(dlog.flatten(0, 1), tgt.flatten(0, 1), ignore_index=-1)

    def seg2d_loss(self, seg2d, seg_gt):
        """2D semantic seg CE. seg_gt [B,N,sh,sw] long, 255 = ignore.

        All-ignore batches (scene not yet covered by a rolling re-extraction)
        make mean-CE 0/0 = nan; return a graph-preserving zero instead.
        For the 21-class csv taxonomy, weight up thin/rare classes (lane 13,
        marking 8, light 9, sign 10, pole 20, VRU 5/6/7) and damp the
        dominant background.
        """
        if (seg_gt != 255).sum() == 0:
            return seg2d.sum() * 0.0
        w = None
        C = seg2d.shape[2]
        if C == 21:
            if getattr(self, "_seg21_w", None) is None \
                    or self._seg21_w.device != seg2d.device:
                w21 = torch.ones(21, device=seg2d.device)
                w21[0] = 0.4
                w21[[8, 13]] = 4.0
                w21[[9, 10, 20]] = 2.0
                w21[[5, 6, 7]] = 1.5
                self._seg21_w = w21
            w = self._seg21_w
        B, N = seg_gt.shape[:2]
        ce = F.cross_entropy(seg2d.flatten(0, 1), seg_gt.flatten(0, 1).long(),
                             ignore_index=255, weight=w, reduction="none")
        # side cameras (FL/FR/BL/BR = 1,2,4,5) x1.6: their precision lags
        camw = torch.ones(N, device=seg2d.device)
        camw[[1, 2, 4, 5]] = 1.6
        camw = camw.repeat(B).view(-1, 1, 1)
        valid = (seg_gt.flatten(0, 1) != 255).float()
        return (ce * camw).sum() / (valid * camw).sum().clamp(min=1)


class DepthSegIPMNetS4(DepthSegIPMNet):
    """v13 variant with a stride-4 depth head (matches stride-4 depth GT).

    Same as DepthSegIPMNet but the depth head predicts at the stride-4 feature
    resolution directly (no x2 upsample), so depth GT at 108x192 aligns.
    """
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.depth_up = nn.Identity()     # depth head runs on stride-4 feature


class DepthSegIPMNetV14(DepthSegIPMNetS4):
    """v14d: fundamental depth upgrade, BEV path untouched.

    - depth branch replaced by a real decoder (4 ConvBlocks, ~1.6M params vs
      the previous single 1x1 conv) at stride-4, TRT-safe convs only.
    - depth loss = smoothed bin-CE + L1 on the expected (metric) depth, so the
      distribution is pushed toward metric accuracy, not just bin hits.
    Context head / IPM gating / BEV decoder are identical to v13d, so BEV
    behaviour is preserved (gate only gets a better depth distribution).
    """
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        ch = 160
        self.depth_up = nn.Identity()
        self.depth_head = nn.Sequential(
            ConvBlock(ch, 256), ConvBlock(256, 256),
            ConvBlock(256, 192), ConvBlock(192, 128),
            nn.Conv2d(128, self.D, 1))

    def depth_loss(self, dlog, depth_gt, ent_w=0.0, far_w=0.0, band_bal=0.0):
        """深度 CE + 鋭化項 (2026-08-21 追加の ent_w / far_w)。

        実測: 30-60 m 帯で深度分布の最大確率が 0.07-0.085、エントロピー
        3.44-3.62 (64 ビン一様は 4.16) = 約 ±17 m に滲んでいる。リフトは
        この分布に従って BEV セルへ特徴を配るので、遠方物体の証拠が
        薄く塗り広げられ、ヒートマップのピークが立たない (cam-only の
        veh recall 20-40m が 0.50 で頭打ちになる直接の原因)。
        ent_w: 分布のエントロピーを罰して鋭くする。
        far_w: 遠方画素の CE 重みを上げる (LiDAR 点は近傍に偏るため)。
        """
        logits = dlog.flatten(0, 1)                       # [BN,D,h,w]
        gt = depth_gt.flatten(0, 1)                       # [BN,h,w]
        tgt = self._d2b(gt).round().long()
        valid = (gt > 0.1) & (tgt >= 0) & (tgt < self.D)
        tgt = tgt.clamp(0, self.D - 1)
        tgt[~valid] = -1
        # edge-aware weighting: depth discontinuities (object boundaries)
        # are a handful of pixels and plain CE lets them smear; weight
        # boundary pixels up to 3x (gradient of GT, valid neighbours only)
        gx = (gt[:, :, 1:] - gt[:, :, :-1]).abs()
        gx = gx * (valid[:, :, 1:] & valid[:, :, :-1])
        gy = (gt[:, 1:, :] - gt[:, :-1, :]).abs()
        gy = gy * (valid[:, 1:, :] & valid[:, :-1, :])
        g = torch.zeros_like(gt)
        g[:, :, 1:] = torch.maximum(g[:, :, 1:], gx)
        g[:, :, :-1] = torch.maximum(g[:, :, :-1], gx)
        g[:, 1:, :] = torch.maximum(g[:, 1:, :], gy)
        g[:, :-1, :] = torch.maximum(g[:, :-1, :], gy)
        wpx = 1.0 + 2.0 * (g / 3.0).clamp(max=1.0)
        # カメラ別の重み (2026-08-17 登録): METEOR_DEPTH_CAM_W="6:3,7:3" の形式。
        # 深度 CE は全カメラ・全画素の等価平均なので、8 台中 2 台の望遠 (特に
        # BACK_NARROW) への圧力が路面画素に埋もれる。教師は箱と整合している
        # (probe_depthgt_align: 整合率 71%) のにモデル側バイアスが +2.3->+2.66 m
        # と学習で縮まなかったのはこのため。既定は未設定 = 従来と完全同一。
        _spec = os.environ.get("METEOR_DEPTH_CAM_W", "")
        if _spec:
            _N = depth_gt.shape[1]
            _wc = torch.ones(_N, device=logits.device, dtype=wpx.dtype)
            for _part in _spec.split(","):
                _i, _w = _part.split(":")
                if 0 <= int(_i) < _N:
                    _wc[int(_i)] = float(_w)
            _B = depth_gt.shape[0]
            wpx = wpx * _wc.repeat(_B).view(-1, 1, 1)
        # 鋭化を狙う場合は label smoothing を外す (平滑化と目的が逆)
        _ls = 0.0 if ent_w > 0 else 0.05
        ce_px = F.cross_entropy(logits, tgt, ignore_index=-1,
                                label_smoothing=_ls, reduction="none")
        if far_w > 0:
            wpx = wpx * (1.0 + far_w * (gt / 60.0).clamp(0.0, 1.0))
        if band_bal > 0:
            # 距離帯の逆頻度重み (2026-08-22)。実測で深度 GT の有効画素は
            # 0-10m が 58.9%、40-60m は合計 4.6% しかない (13 倍の不均衡)。
            # far_w は最大でも 2 倍にしかならず釣り合わないので、10m 帯ごとに
            # 「その帯の画素数の逆数」で正規化して各帯の寄与を揃える。
            # バッチ自身のヒストグラムから作るので外部テーブルは不要。
            _bi = (gt / 10.0).clamp(0, 7).long()
            _cnt = torch.bincount(_bi[valid].flatten(), minlength=8).float()
            _inv = valid.sum().float() / (8.0 * _cnt.clamp(min=1.0))
            _inv = _inv.clamp(0.2, 10.0)
            _inv = 1.0 + band_bal * (_inv - 1.0)      # band_bal=1 で完全均等
            # 平均重みを 1 に正規化する。これをしないと「配分を変える」と
            # 「深度損失の係数を上げる」が同時に起きて A/B の変数が 2 つに
            # なる (band_bal=1 で損失が 2.3 倍になっていた)。
            _mean = (_cnt * _inv).sum() / valid.sum().clamp(min=1).float()
            _inv = _inv / _mean.clamp(min=1e-6)
            wpx = wpx * _inv[_bi]
        ce = (ce_px * wpx)[valid].mean() if valid.any() \
            else logits.sum() * 0.0
        # L1 on expected depth (metres) -> metric accuracy, sharper distributions
        prob = logits.softmax(1)
        bins = self._dbins(logits.device, prob.dtype).view(1, -1, 1, 1)
        exp_d = (prob * bins).sum(1)
        if valid.any():
            l1 = (exp_d - gt).abs()[valid].mean()
        else:
            l1 = exp_d.sum() * 0
        out = ce + 0.1 * l1
        if ent_w > 0 and valid.any():
            _e = -(prob.clamp_min(1e-6) * prob.clamp_min(1e-6).log()).sum(1)
            out = out + ent_w * _e[valid].mean()
        return out


N_BOX = 3    # 0 bg, 1 vehicle, 2 VRU (camera-visible 3D boxes on BEV)


class DepthSegIPMNetV15(DepthSegIPMNetV14):
    """v15: v14d + multi-task BEV 3D-box occupancy head.

    A parallel decoder on the shared BEV feature predicts camera-visible
    object footprints (vehicle / VRU) rasterised from LiDAR 3D boxes.
    Lane-seg decoder, depth branch and IPM gating are unchanged; TRT-safe.
    forward -> (seg, depth_logits, seg2d, box_logits).
    """
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.box_dec = nn.Sequential(
            ConvBlock(96, 128), ConvBlock(128, 96),
            nn.Conv2d(96, N_BOX, 1))

    def forward(self, imgs, K, T_cam_ego):
        B, N, _, H, W = imgs.shape
        f = self.image_feats(imgs)
        # Clamped. The 2D seg logits are the one thing that goes non-finite in
        # these rounds: r59 discarded 0 % of steps up to 9k, 5 % to 12k and
        # 29 % by 13.5k, every one of them reported as out[2] alone with the BEV
        # outputs clean, and r53/r56/r57 showed the same climb. In fp16 a logit
        # only has to pass 65504 to become inf, and CE saturates long before
        # +-30, so this changes nothing about what the loss sees while removing
        # the failure -- and since clamp has zero gradient outside the range it
        # also stops the head being pushed further out once it gets there.
        # clamp だけでは NaN が素通しする (clamp(nan)=nan)。fp16 で head 内部が
        # 一度 inf になると BN の (x-mean)/sqrt(var) で NaN が生まれ、
        # 出力段の clamp では消せない。2026-08-17: nan_to_num を前置し、
        # out[2] 由来の SKIP (r59 で 29%, r72 で 148 回, r73 で 16 回) を断つ。
        if _EXPORT_FAST:
            seg2d = self.seg_head(f).clamp(-30.0, 30.0)
        else:
            seg2d = torch.nan_to_num(self.seg_head(f), nan=0.0,
                                     posinf=30.0, neginf=-30.0
                                     ).clamp(-30.0, 30.0)
        dlog = self.depth_head(self.depth_up(f))
        dprob = dlog.softmax(1)
        ctx = self.ctx(f)
        Cc = ctx.shape[1]
        pts = self.bev_pts
        G2 = pts.shape[0]
        pc = torch.matmul(T_cam_ego.reshape(B * N, 4, 4),
                          pts.t().unsqueeze(0).expand(B * N, 4, G2))
        x, y, z = pc[:, 0], pc[:, 1], pc[:, 2]
        Kf = K.reshape(B * N, 3, 3)
        valid = z > 0.5
        zc = z.clamp(min=0.5)
        u = Kf[:, 0, 0].unsqueeze(-1) * x / zc + Kf[:, 0, 2].unsqueeze(-1)
        v = Kf[:, 1, 1].unsqueeze(-1) * y / zc + Kf[:, 1, 2].unsqueeze(-1)
        valid = valid & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        dist = torch.sqrt(x * x + y * y + z * z)
        valid = valid & (dist < 90.0)
        gu = (u / (W - 1) * 2 - 1).clamp(-2, 2)
        gv = (v / (H - 1) * 2 - 1).clamp(-2, 2)
        grid = torch.stack([gu, gv], -1).unsqueeze(2)
        ctx_s = F.grid_sample(ctx, grid, align_corners=False).squeeze(-1)
        prob_s = F.grid_sample(dprob, grid, align_corners=False).squeeze(-1)
        b = self._d2b(dist).clamp(0, self.D - 1 - 1e-4)
        b0 = b.floor().long()
        fr = (b - b0.float()).unsqueeze(1)
        w0 = torch.gather(prob_s, 1, b0.unsqueeze(1))
        w1 = torch.gather(prob_s, 1, (b0 + 1).clamp(max=self.D - 1).unsqueeze(1))
        w = (w0 * (1 - fr) + w1 * fr) + 0.05
        w = w * valid.unsqueeze(1).to(w.dtype)
        num = (ctx_s.view(B, N, Cc, G2) * w.view(B, N, 1, G2)).sum(1)
        den = w.view(B, N, 1, G2).sum(1).clamp(min=1e-4)
        bev = (num / den).view(B, Cc, BEV_H, BEV_W)
        fh2, fw2 = dlog.shape[-2:]
        sh, sw = seg2d.shape[-2:]
        return (self.dec(bev), dlog.view(B, N, self.D, fh2, fw2),
                seg2d.view(B, N, seg2d.shape[1], sh, sw), self.box_dec(bev))

    def box_loss(self, box_logits, box_gt, w_bg=0.2, w_veh=1.0, w_vru=2.0):
        cw = torch.tensor([w_bg, w_veh, w_vru], device=box_logits.device,
                          dtype=box_logits.dtype)
        ce = F.cross_entropy(box_logits, box_gt, weight=cw)
        # dice on object classes for shape quality
        prob = box_logits.softmax(1)
        dice = 0.0
        for c in (1, 2):
            p = prob[:, c]
            t = (box_gt == c).to(prob.dtype)
            inter = (p * t).sum((1, 2))
            dice = dice + (1 - (2 * inter + 1.0)
                           / (p.sum((1, 2)) + t.sum((1, 2)) + 1.0)).mean()
        return ce + 0.5 * dice


DET_S = 2                      # detection grid stride on the BEV (400x250)
DET_H, DET_W = BEV_H // DET_S, BEV_W // DET_S
DET_RES = 0.2 * DET_S          # metres per det cell


class DepthSegIPMNetV16(DepthSegIPMNetV14):
    """v16: oriented-BBox detection head (CenterPoint-style) instead of the
    v15 occupancy mask. Multi-task with lane seg + depth; TRT-safe ops.

    Det head on the shared BEV feature at stride-2 (400x250):
      hm  [B,2,h,w]  class center heatmaps (vehicle, VRU) - focal loss
      reg [B,6,h,w]  (off_r, off_c, log l, log w, sin yaw, cos yaw) - L1@centers
    forward -> (seg, depth_logits, seg2d, hm, reg).
    """
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.det_stem = nn.Sequential(
            nn.Conv2d(BEV_CH, 128, 3, stride=DET_S, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            ConvBlock(128, 128))
        self.hm_head = nn.Conv2d(128, 2, 1)
        self.reg_head = nn.Conv2d(128, 6, 1)
        nn.init.constant_(self.hm_head.bias, -2.19)   # focal init (p~0.1)

    def bake_frustum(self, K, T_cam_ego, H, W):
        """Freeze the per-camera visible-cell lists for ONE rig calibration.

        `nonzero` is data-dependent and cannot be exported to ONNX, so the
        deployment graph needs the index lists as CONSTANTS. K / T_cam_ego are
        fixed for a vehicle, so baking them is legitimate -- but the resulting
        graph is then valid for THAT rig only, which is why it is opt-in and
        recorded on the module.
        """
        B_N = K.reshape(-1, 3, 3).shape[0]
        pts = self.bev_pts
        G2 = pts.shape[0]
        pc = torch.matmul(T_cam_ego.reshape(B_N, 4, 4),
                          pts.t().unsqueeze(0).expand(B_N, 4, G2))
        x, y, z = pc[:, 0], pc[:, 1], pc[:, 2]
        Kf = K.reshape(B_N, 3, 3)
        zc = z.clamp(min=0.5)
        u = Kf[:, 0, 0].unsqueeze(-1) * x / zc + Kf[:, 0, 2].unsqueeze(-1)
        v = Kf[:, 1, 1].unsqueeze(-1) * y / zc + Kf[:, 1, 2].unsqueeze(-1)
        dist = torch.sqrt(x * x + y * y + z * z)
        valid = ((z > 0.5) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
                 & (dist < 90.0))
        self._fr_static = []
        for i in range(B_N):
            ii = valid[i].nonzero(as_tuple=True)[0].contiguous()
            self.register_buffer(f"fr_idx_{i}", ii, persistent=False)
            self._fr_static.append(ii)
        self.frustum_lift = True
        n = sum(int(t.numel()) for t in self._fr_static)
        print(f"[frustum] baked {B_N} cameras, {n} of {B_N * G2} "
              f"(camera, cell) pairs kept = {100 * n / (B_N * G2):.1f}%",
              flush=True)
        return self._fr_static

    def bake_gather(self):
        """Invert the baked frustum lists into a per-cell gather table.

        Measured on the AGX Orin (32 GB devkit, TensorRT 10.16): the slowest
        layers of both engines are the lift's Scatter nodes -- index_add_ turns
        into ScatterND with atomic collisions wherever two cameras write one
        cell, and on a 204 GB/s part that is the single largest block (the
        Myelin lift cluster is ~53 % of base, ~51 % of light). The same sum
        can be read instead of written: concatenate every camera's weighted
        pair outputs into one flat buffer and let EACH CELL gather its own
        contributors from a static table. Same maths, addition order aside;
        atomics gone; reads coalesced.

        Requires bake_frustum first (the table inverts those lists). Like the
        frustum, the table is valid for ONE rig only.
        """
        st = getattr(self, "_fr_static", None)
        assert st is not None, "bake_frustum を先に呼ぶこと"
        G2 = self.bev_pts.shape[0]
        owners = [[] for _ in range(G2)]
        off = 0
        for ii in st:                       # pair k of camera i -> flat off+k
            for k, cell in enumerate(ii.tolist()):
                owners[cell].append(off + k)
            off += int(ii.numel())
        kmax = max((len(o) for o in owners), default=1)
        tab = torch.full((G2, kmax), off, dtype=torch.long)   # off = pad slot
        for c, o in enumerate(owners):
            for j, p in enumerate(o):
                tab[c, j] = p
        # register_buffer after .cuda() leaves the buffer on CPU; place it
        # where the frustum lists already live
        tab = tab.to(st[0].device if len(st) else "cpu")
        self.register_buffer("gather_tab", tab, persistent=False)
        self._gather_pairs = off
        self.gather_lift = True
        n = sum(len(o) for o in owners)
        print(f"[gather] {G2} cells, kmax={kmax}, pairs={off} "
              f"(mean {n / G2:.2f}/cell)", flush=True)

    def _project_bev_frustum_gather(self, dprob, ctx, K, T_cam_ego,
                                    B, N, H, W):
        """Gather-form lift: identical numbers to the scatter form up to
        addition order (verified against it before deployment)."""
        Cc = ctx.shape[1]
        pts = self.bev_pts
        G2 = pts.shape[0]
        pc = torch.matmul(T_cam_ego.reshape(B * N, 4, 4),
                          pts.t().unsqueeze(0).expand(B * N, 4, G2))
        x, y, z = pc[:, 0], pc[:, 1], pc[:, 2]
        Kf = K.reshape(B * N, 3, 3)
        zc = z.clamp(min=0.5)
        u = Kf[:, 0, 0].unsqueeze(-1) * x / zc + Kf[:, 0, 2].unsqueeze(-1)
        v = Kf[:, 1, 1].unsqueeze(-1) * y / zc + Kf[:, 1, 2].unsqueeze(-1)
        dist = torch.sqrt(x * x + y * y + z * z)
        gu = (u / (W - 1) * 2 - 1).clamp(-2, 2)
        gv = (v / (H - 1) * 2 - 1).clamp(-2, 2)
        grid = torch.stack([gu, gv], -1)
        vals, wgts = [], []
        for i in range(B * N):
            ii = self._fr_static[i]
            if ii.numel() == 0:
                continue
            g = grid[i].index_select(0, ii).view(1, -1, 1, 2).to(ctx.dtype)
            cs = F.grid_sample(ctx[i:i + 1], g,
                               align_corners=False).squeeze(-1)
            ps = F.grid_sample(dprob[i:i + 1], g,
                               align_corners=False).squeeze(-1)
            d_ = dist[i].index_select(0, ii)
            b_ = self._d2b(d_).clamp(0, self.D - 1 - 1e-4)
            b0 = b_.floor().long()
            fr = (b_ - b0.float()).view(1, 1, -1).to(ps.dtype)
            w = (torch.gather(ps, 1, b0.view(1, 1, -1)) * (1 - fr)
                 + torch.gather(ps, 1, (b0 + 1).clamp(max=self.D - 1)
                                .view(1, 1, -1)) * fr) + 0.05
            vals.append((cs * w)[0])
            wgts.append(w[0])
        val = torch.cat(vals, 1)                       # [Cc, P]
        wgt = torch.cat(wgts, 1)                       # [1, P]
        # pad slot P: zero contribution
        val = torch.cat([val, val.new_zeros(Cc, 1)], 1)
        wgt = torch.cat([wgt, wgt.new_zeros(1, 1)], 1)
        tab = self.gather_tab.reshape(-1)              # [G2*kmax]
        num = val.index_select(1, tab).view(Cc, G2, -1).sum(-1)
        den = wgt.index_select(1, tab).view(1, G2, -1).sum(-1)
        return (num / den.clamp(min=1e-4)).view(B, Cc, BEV_H, BEV_W)

    def _frustum_idx(self, valid, key):
        st = getattr(self, "_fr_static", None)
        if st is not None:
            return st
        """Per-camera list of BEV cells that camera can actually see.

        Depends only on K / T_cam_ego, which are constant for a rig, so the
        lists are cached and computed once per calibration.
        """
        cache = getattr(self, "_fr_cache", None)
        if cache is None:
            cache = self._fr_cache = {}
        hit = cache.get(key)
        if hit is not None:
            return hit
        idx = [valid[i].nonzero(as_tuple=True)[0].contiguous()
               for i in range(valid.shape[0])]
        if len(cache) > 8:
            cache.clear()
        cache[key] = idx
        return idx

    def project_bev(self, dprob, ctx, K, T_cam_ego, B, N, H, W):
        """Depth-gated lift of image features into the BEV grid.

        The dense form samples EVERY camera at EVERY BEV cell and throws the
        misses away with a mask. Measured on a real rig: only 22.6 % of the
        (camera, cell) pairs are inside an image (721,761 of 3,200,000), so
        77 % of the sampling and the reduction is discarded work. With
        `frustum_lift` the cells are gathered per camera first, which is
        mathematically the SAME computation -- verified fp32-exact against the
        dense path, fp16 within one ULP (max 9.8e-4) -- and needs no retraining.
        11.6 -> 3.4 ms on the workstation GPU.

        Kept off during training: the augmentation changes T_cam_ego per sample
        so the index cache would thrash, and the dense path is already the
        gradient-tested one.
        """
        if getattr(self, "frustum_lift", False) and not self.training:
            if getattr(self, "gather_lift", False):
                return self._project_bev_frustum_gather(
                    dprob, ctx, K, T_cam_ego, B, N, H, W)
            return self._project_bev_frustum(dprob, ctx, K, T_cam_ego,
                                             B, N, H, W)
        Cc = ctx.shape[1]
        pts = self.bev_pts
        G2 = pts.shape[0]
        pc = torch.matmul(T_cam_ego.reshape(B * N, 4, 4),
                          pts.t().unsqueeze(0).expand(B * N, 4, G2))
        x, y, z = pc[:, 0], pc[:, 1], pc[:, 2]
        Kf = K.reshape(B * N, 3, 3)
        valid = z > 0.5
        zc = z.clamp(min=0.5)
        u = Kf[:, 0, 0].unsqueeze(-1) * x / zc + Kf[:, 0, 2].unsqueeze(-1)
        v = Kf[:, 1, 1].unsqueeze(-1) * y / zc + Kf[:, 1, 2].unsqueeze(-1)
        valid = valid & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        dist = torch.sqrt(x * x + y * y + z * z)
        valid = valid & (dist < 90.0)
        gu = (u / (W - 1) * 2 - 1).clamp(-2, 2)
        gv = (v / (H - 1) * 2 - 1).clamp(-2, 2)
        grid = torch.stack([gu, gv], -1).unsqueeze(2)
        ctx_s = F.grid_sample(ctx, grid, align_corners=False).squeeze(-1)
        prob_s = F.grid_sample(dprob, grid, align_corners=False).squeeze(-1)
        b = self._d2b(dist).clamp(0, self.D - 1 - 1e-4)
        b0 = b.floor().long()
        fr = (b - b0.float()).unsqueeze(1)
        w0 = torch.gather(prob_s, 1, b0.unsqueeze(1))
        w1 = torch.gather(prob_s, 1, (b0 + 1).clamp(max=self.D - 1).unsqueeze(1))
        wgt = (w0 * (1 - fr) + w1 * fr) + 0.05
        wgt = wgt * valid.unsqueeze(1).to(wgt.dtype)
        num = (ctx_s.view(B, N, Cc, G2) * wgt.view(B, N, 1, G2)).sum(1)
        den = wgt.view(B, N, 1, G2).sum(1).clamp(min=1e-4)
        return (num / den).view(B, Cc, BEV_H, BEV_W)

    def _project_bev_frustum(self, dprob, ctx, K, T_cam_ego, B, N, H, W):
        Cc = ctx.shape[1]
        pts = self.bev_pts
        G2 = pts.shape[0]
        pc = torch.matmul(T_cam_ego.reshape(B * N, 4, 4),
                          pts.t().unsqueeze(0).expand(B * N, 4, G2))
        x, y, z = pc[:, 0], pc[:, 1], pc[:, 2]
        Kf = K.reshape(B * N, 3, 3)
        zc = z.clamp(min=0.5)
        u = Kf[:, 0, 0].unsqueeze(-1) * x / zc + Kf[:, 0, 2].unsqueeze(-1)
        v = Kf[:, 1, 1].unsqueeze(-1) * y / zc + Kf[:, 1, 2].unsqueeze(-1)
        dist = torch.sqrt(x * x + y * y + z * z)
        valid = ((z > 0.5) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
                 & (dist < 90.0))
        gu = (u / (W - 1) * 2 - 1).clamp(-2, 2)
        gv = (v / (H - 1) * 2 - 1).clamp(-2, 2)
        grid = torch.stack([gu, gv], -1)
        key = (int(K.data_ptr()), int(T_cam_ego.data_ptr()), B * N, H, W,
               float(K.reshape(-1)[0]), float(T_cam_ego.reshape(-1)[3]))
        idx = self._frustum_idx(valid, key)
        num = torch.zeros(B, Cc, G2, device=ctx.device, dtype=ctx.dtype)
        den = torch.zeros(B, 1, G2, device=ctx.device, dtype=ctx.dtype)
        for i in range(B * N):
            ii = idx[i]
            if ii.numel() == 0:
                continue
            g = grid[i].index_select(0, ii).view(1, -1, 1, 2).to(ctx.dtype)
            cs = F.grid_sample(ctx[i:i + 1], g,
                               align_corners=False).squeeze(-1)
            ps = F.grid_sample(dprob[i:i + 1], g,
                               align_corners=False).squeeze(-1)
            d_ = dist[i].index_select(0, ii)
            b_ = self._d2b(d_).clamp(0, self.D - 1 - 1e-4)
            b0 = b_.floor().long()
            fr = (b_ - b0.float()).view(1, 1, -1).to(ps.dtype)
            w = (torch.gather(ps, 1, b0.view(1, 1, -1)) * (1 - fr)
                 + torch.gather(ps, 1, (b0 + 1).clamp(max=self.D - 1)
                                .view(1, 1, -1)) * fr) + 0.05
            num[i // N].index_add_(1, ii, (cs * w)[0].to(num.dtype))
            den[i // N].index_add_(1, ii, w[0].to(den.dtype))
        return (num / den.clamp(min=1e-4)).view(B, Cc, BEV_H, BEV_W)

    def compute_bev(self, imgs, K, T_cam_ego):
        """Images -> raw (pre-fusion) BEV feature; used for the temporal
        previous-frame pass and for streaming deployment."""
        B, N, _, H, W = imgs.shape
        f = self.image_feats(imgs)
        dprob = self.sharpen_dprob(self.depth_head(self.depth_up(f))
                                   .softmax(1))
        return self.bev_extra(self.project_bev(dprob, self.ctx(f), K,
                                               T_cam_ego, B, N, H, W))

    def sharpen_dprob(self, dprob):
        return dprob                    # v31 blends in LiDAR when present

    def bev_extra(self, bev):
        return bev                      # v32 adds the LiDAR pillar residual

    def enable_paint_seg(self, classes):
        """PointPainting (2026-08-17): seg2d の確率をリフト前の ctx に注入する。

        歩行者は 30 m で幅 1.4 特徴画素だが高さは 4 画素あり、2D セグは
        その縦の柱を使って画素単位で塗れる。BEV への持ち上げは足跡 1 セルに
        潰すため縦の証拠が消える -- 塗った確率をリフトで運べば BEV セルに
        「ここは歩行者/車」の明示的な証拠が届く。

        連結ではなく 1x1 ゼロ初期化射影の加算にする理由:
          - ctx のチャネル数 (96) は hist_bev や時間融合など下流全体に波及
            するので、幅を変えると機能保存にならず配布(リフトプラグインの
            Cc=96)も壊れる。加算なら開始時は現行と完全同一。
          - painter (seg_head) は detach し、自身の 2D セグ損失だけで学習させる。
        """
        self._paint_cls = [int(c) for c in classes]
        _convs = [mm for mm in self.ctx.modules()
                  if isinstance(mm, nn.Conv2d)]
        cc = _convs[-1].out_channels
        self.paint_proj = nn.Conv2d(len(self._paint_cls), cc, 1)
        nn.init.zeros_(self.paint_proj.weight)
        nn.init.zeros_(self.paint_proj.bias)
        # モデルが既に GPU 上にあるとき、新設の conv が CPU に残ると DDP が
        # 「cpu と cuda が混在」で落ちる (v91 で実際に発生)
        self.paint_proj = self.paint_proj.to(next(self.parameters()).device)

        def _stash(_m, _i, _o):
            self._paint_buf = _o

        def _mix(_m, _i, _o):
            pb = getattr(self, "_paint_buf", None)
            if pb is None:
                return _o
            pb = pb.float() if _EXPORT_FAST else torch.nan_to_num(
                pb.float(), nan=0.0, posinf=30.0,
                                  neginf=-30.0)
            pb = pb.softmax(1)[:, self._paint_cls].detach().to(_o.dtype)
            if pb.shape[-2:] != _o.shape[-2:]:
                pb = F.interpolate(pb, size=_o.shape[-2:], mode="bilinear",
                                   align_corners=False)
            if pb.shape[0] != _o.shape[0]:
                if os.environ.get("METEOR_PAINT_DEBUG"):
                    print(f"[paint] バッチ不一致 {pb.shape} 対 {_o.shape}")
                return _o
            _d = self.paint_proj(pb)
            if os.environ.get("METEOR_PAINT_DEBUG"):
                print(f"[paint] pb={tuple(pb.shape)} 加算ノルム={float(_d.abs().mean()):.5f}")
            return _o + _d

        self._paint_buf = None
        self.seg_head.register_forward_hook(_stash)
        self.ctx.register_forward_hook(_mix)

    def convert_ego_pool(self, hw, mean=None, var=None, verbose=True):
        """ego の大域平均プーリングを INT8 に耐える形へ置き換える (2026-08-24)。

        **なぜ必要か** (Orin 実測): INT8 は値を 127 段階でしか表せず、目盛りの
        間隔はそのテンソルの最大値で決まる。プーリング**手前**の特徴マップは
        0-19.3 に散らばるので目盛りは 0.155。ところが 400 セルを平均すると
        出力は 0-0.29 の狭い範囲に集まる (66 分の 1)。TensorRT は平均
        プーリングの出力に**入力と同じ目盛りを流用する**ため、出力は 127 段階
        のうち約 2 段階しか使えない。フレーム間の変化 0.0102 は 0.065 段階 =
        四捨五入で完全に消える。だから真っ黒でも乱数でも ego 出力がビット一致
        する「凍結」が起きていた。occ/traj/seg/det が無事なのは、空間ヘッドは
        セルの値を直接読むから (ego だけが大域平均を取る唯一のヘッド)。

        **対策**: 平均を「深さ方向畳み込み + BatchNorm」に置き換える。
        畳み込みは INT32 で累算してから**自前の較正済み目盛り**で測り直すので
        流用が起きない。さらに BN がチャネル毎に平均と分散を揃えるため、
        一部の高水準チャネルが目盛りを独占しなくなる。

        実測 (変動 / 1 段階): 現行 0.065 -> conv 化 4.41 -> conv+BN 23.51。

        **機能保存**: 畳み込みの重みを 1/(h*w) で初期化し、BN の統計を実測値
        (mean/var) にしたうえで、ego_mlp の第 1 層を W'=W*sigma, b'=b+W@mu と
        補正するので、**変換直後の出力は変換前と一致する**。
        """
        seq = self.ego_stem
        idx = [i for i, m in enumerate(seq)
               if isinstance(m, nn.AdaptiveAvgPool2d)]
        if not idx:
            if verbose:
                print("[ego-pool] AdaptiveAvgPool が無い (変換済み?)", flush=True)
            return False
        i = idx[-1]
        c = None
        for m in list(seq)[:i][::-1]:
            if isinstance(m, nn.Conv2d):
                c = m.out_channels
                break
            if isinstance(m, nn.BatchNorm2d):
                c = m.num_features
                break
        assert c is not None, "プール直前のチャネル数が取れない"
        h, w = int(hw[0]), int(hw[1])
        dev = next(self.parameters()).device
        # BatchNorm は使わない: 出力が 1x1 なのでバッチ内 1 チャネルあたり
        # 標本が batch 数しか無く統計が取れない (batch=1 では例外になる)。
        # 正規化は定数なので **畳み込みの重みとバイアスに畳み込める**:
        #   出力 = (平均 - mu) / sigma
        #        = sum(x) * 1/(h*w*sigma)  +  (-mu/sigma)
        # 1 層で済み、TensorRT はこの conv 出力に自前の較正スケールを付ける。
        # 重み自体は学習可能なので、以後の微調整で自由に動く。
        conv = nn.Conv2d(c, c, kernel_size=(h, w), groups=c, bias=True)
        with torch.no_grad():
            _mu = torch.zeros(c) if mean is None else torch.as_tensor(mean).float()
            _va = torch.ones(c) if var is None else torch.as_tensor(var).float()
            _sig = _va.clamp(min=1e-8).sqrt()
            conv.weight.copy_((1.0 / (h * w * _sig)).view(c, 1, 1, 1)
                              .expand(c, 1, h, w).contiguous())
            conv.bias.copy_(-_mu / _sig)
            # ego_mlp の第 1 層で正規化を打ち消す (機能保存)
            lin = None
            for m in self.ego_mlp:
                if isinstance(m, nn.Linear):
                    lin = m
                    break
            if lin is not None:
                W = lin.weight[:, :c]
                lin.bias.add_(W @ _mu.to(W.device))
                lin.weight[:, :c] = W * _sig.to(W.device)
        seq[i] = conv.to(dev)
        self.to(dev)
        if verbose:
            print(f"[ego-pool] 大域平均 ({h}x{w}) を depthwise conv + BN に置換 "
                  f"(ch {c}, 機能保存)", flush=True)
        return True

    def enable_pact(self, patterns, alpha_init=None, verbose=True):
        """PACT (2026-08-23): 指定 ReLU を「学習可能な上限つき ReLU」に置換。

        INT8 で ego が凍結する原因を実測で特定した結果: ego の直前にある
        時間融合の ReLU が **外れ値比 48 倍** (tfuse3.3.5 は max 136.9 に対し
        p99.9 が 2.86) だった。per-tensor の INT8 スケールは max で決まるので、
        信号の 99.9% が 127 段階のうち 3 段階に潰れる。ego 出力が定数化する
        一方で std だけ大きく見える現象と整合する。較正方式を 3 種
        (entropy2 / MinMax / legacy) 試して ego 出力が小数第 5 位まで
        一致したのは、スケールの選び方ではなく **活性の分布自体** が
        原因だから。

        ReLU6 (固定値 6) は使えない: 同じネットの中に p99.9 が 27 の層
        (pl_head) と 2.9 の層 (tfuse) が同居しており、一律に切ると正常な
        信号を壊す。よって層ごとに学習可能な上限 alpha を持たせる。

        alpha は **実測 max x 1.10 で初期化する**ので、開始時点の出力は
        素の ReLU と完全に一致する (機能保存)。そこから L1 の減衰
        (--pact-w) で押し下げ、外れ値だけを刈る。

        推論時は定数の Clip 1 個に落ちるので **レイテンシ増はゼロ**
        (前段の conv+BN に融合される)。QAT と違い ONNX のグラフ構造も
        量子化方式も変えないので、MeteorLift プラグイン経路のカーネル選択に
        影響しない。
        """
        import fnmatch as _fn
        init = dict(alpha_init or {})
        tgt = [t.strip() for t in (patterns.split(",")
                                   if isinstance(patterns, str) else patterns)]

        def _match(n):
            return any(n.startswith(t) or _fn.fnmatch(n, t) for t in tgt)

        made = []
        for mn, mod in list(self.named_modules()):
            for cn, ch in list(mod.named_children()):
                if not isinstance(ch, nn.ReLU):
                    continue
                full = f"{mn}.{cn}" if mn else cn
                if not _match(full):
                    continue
                a0 = float(init.get(full, 64.0))
                setattr(mod, cn, PACTReLU(a0, inplace=ch.inplace))
                made.append((full, a0))
        self._pact_names = [m[0] for m in made]
        dev = next(self.parameters()).device
        self.to(dev)
        if verbose:
            print(f"[pact] {len(made)} 層を上限つき ReLU に置換 "
                  f"(alpha は実測 max の安全率つきで初期化 = 機能保存)", flush=True)
            for n_, a_ in made:
                print(f"[pact]   {n_:<22} alpha_init {a_:8.2f}", flush=True)
        return made

    def pact_penalty(self):
        """alpha の L1。これで上限を押し下げ、外れ値だけを刈る。"""
        acc = None
        for m in self.modules():
            if isinstance(m, PACTReLU):
                v = m.alpha.abs()
                acc = v if acc is None else acc + v
        return acc if acc is not None else torch.zeros((), device=next(
            self.parameters()).device)

    def enable_paint_det(self, classes):
        """paint-det (2026-08-22): 2D 検出ヒートマップをリフト前の ctx へ注入。

        深度分布の鋭化は 30-60m で飽和し (ent_w 0.02 と 0.1 で最大確率
        0.090/0.091、目標 0.15 に届かず)、cam-only の遠方 veh recall は
        0.50 で頭打ち。一方 2D 画像では遠方車両は見えており、2D det の
        ヒートマップはピークが立つ。その「ここに車がいる」証拠をリフトで
        BEV セルへ届ける。paint-seg と同じくゼロ初期化 1x1 射影の加算なので
        導入時点の出力は完全に不変 (BEV 幅 96ch もリフトプラグインも不変)。

        det2d は本来 forward の最後で _last_f から計算されるが、ctx より
        前に計算しても等価なので、ここで前倒しして結果をキャッシュし、
        後段の det2d_forward がそれを再利用する (計算は 1 回だけ)。
        """
        self._paint_det_cls = [int(c) for c in classes]
        _convs = [mm for mm in self.ctx.modules()
                  if isinstance(mm, nn.Conv2d)]
        cc = _convs[-1].out_channels
        self.paint_det_proj = nn.Conv2d(len(self._paint_det_cls), cc, 1)
        nn.init.zeros_(self.paint_det_proj.weight)
        nn.init.zeros_(self.paint_det_proj.bias)
        self.paint_det_proj = self.paint_det_proj.to(
            next(self.parameters()).device)

        def _mix_det(_m, _i, _o):
            f = getattr(self, "_last_f", None)
            if f is None or f.shape[0] != _o.shape[0]:
                return _o
            d = self.det2d_stem(f)
            hm = self.hm2d_head(d)
            self._det2d_cache = (d, hm)          # 後段で再利用
            p = hm.sigmoid()[:, self._paint_det_cls].detach().to(_o.dtype)
            if p.shape[-2:] != _o.shape[-2:]:
                p = F.interpolate(p, size=_o.shape[-2:], mode="bilinear",
                                  align_corners=False)
            _d = self.paint_det_proj(p)
            if os.environ.get("METEOR_PAINT_DEBUG"):
                print(f"[paint-det] p={tuple(p.shape)} "
                      f"加算ノルム={float(_d.abs().mean()):.5f}")
            return _o + _d

        self._det2d_cache = None
        self.ctx.register_forward_hook(_mix_det)

    def temporal_fuse(self, bev):
        return bev                      # identity below v22

    def lane_input(self):
        return self._fused_bev          # v24 reroutes to the raw BEV

    def occ_input(self):
        return self._fused_bev

    def det_input(self):
        return self._fused_bev          # v25 reroutes to the raw BEV

    def forward(self, imgs, K, T_cam_ego):
        B, N, _, H, W = imgs.shape
        f = self.image_feats(imgs)
        self._f_s4 = f                 # v27 TL head reads the front cameras
        # Clamped. The 2D seg logits are the one thing that goes non-finite in
        # these rounds: r59 discarded 0 % of steps up to 9k, 5 % to 12k and
        # 29 % by 13.5k, every one of them reported as out[2] alone with the BEV
        # outputs clean, and r53/r56/r57 showed the same climb. In fp16 a logit
        # only has to pass 65504 to become inf, and CE saturates long before
        # +-30, so this changes nothing about what the loss sees while removing
        # the failure -- and since clamp has zero gradient outside the range it
        # also stops the head being pushed further out once it gets there.
        # clamp だけでは NaN が素通しする (clamp(nan)=nan)。fp16 で head 内部が
        # 一度 inf になると BN の (x-mean)/sqrt(var) で NaN が生まれ、
        # 出力段の clamp では消せない。2026-08-17: nan_to_num を前置し、
        # out[2] 由来の SKIP (r59 で 29%, r72 で 148 回, r73 で 16 回) を断つ。
        if _EXPORT_FAST:
            seg2d = self.seg_head(f).clamp(-30.0, 30.0)
        else:
            seg2d = torch.nan_to_num(self.seg_head(f), nan=0.0,
                                     posinf=30.0, neginf=-30.0
                                     ).clamp(-30.0, 30.0)
        dlog = self.depth_head(self.depth_up(f))
        dprob = self.sharpen_dprob(dlog.softmax(1))
        ctx = self.ctx(f)
        bev = self.bev_extra(
            self.project_bev(dprob, ctx, K, T_cam_ego, B, N, H, W))
        self._last_bev = bev
        bev = self.temporal_fuse(bev)
        self._fused_bev = bev          # consumed by ego / occ / traj heads
        # Optional lane SDF auxiliary (enable_lane_sdf). Prediction is stashed
        # on the module rather than appended to the output tuple: the tuple is
        # positional with 19 consumers and a probe must not renumber them.
        if getattr(self, "lane_sdf", None) is not None:
            # the head is float32 and this point is reached both inside and
            # outside autocast (temporal-history passes disable it); cast the
            # input to the head's own dtype so both paths work
            _w = self.lane_sdf[0].weight
            # Lane geometry is a static task and must use the same RAW BEV as
            # the segmentation decoder. The temporally fused BEV contains
            # warp residuals/moving-object ghosts and is reserved for motion.
            self._lane_sdf_pred = self.lane_sdf(
                self.lane_input().to(_w.dtype))

        det = self.det_stem(self.det_input())
        dd = getattr(self, "det_deep", None)
        if dd is not None:
            det = det + dd(det.to(dd[0].weight.dtype)).to(det.dtype)
        self._det_feat = det
        lane_bev = self.lane_input()
        fh2, fw2 = dlog.shape[-2:]
        sh, sw = seg2d.shape[-2:]
        rg_out = self.reg_head(det)
        self._det_reg = rg_out          # v33 traj head reads sin/cos yaw
        return (self._seg_with_lane(lane_bev),
                dlog.view(B, N, self.D, fh2, fw2),
                seg2d.view(B, N, seg2d.shape[1], sh, sw),
                self.hm_head(det), rg_out)

    def _seg_with_lane(self, lane_bev):
        seg = self.dec(lane_bev)
        sd = getattr(self, "seg_deep", None)
        if sd is not None:
            seg = seg + sd(lane_bev.to(sd[0].weight.dtype)).to(seg.dtype)
        br = getattr(self, "lane_branch", None)
        if br is None:
            return seg
        res = br(lane_bev.to(br[0].weight.dtype)).to(seg.dtype)
        return torch.cat([seg[:, :4], seg[:, 4:7] + res, seg[:, 7:]], 1)

    @staticmethod
    def build_det_targets(boxes, nbox, device, dtype=torch.float32,
                          gt_w=None):
        """boxes [B,Kmax,6] (cls,xe,ye,l,w,yaw), nbox [B] -> (hm_t, reg_t, m_t).

        hm_t [B,2,h,w] Gaussian center heatmaps; reg_t [B,6,h,w]; m_t [B,1,h,w]
        1 at box-center cells (reg supervised there only).
        """
        B = boxes.shape[0]
        hm = torch.zeros(B, 2, DET_H, DET_W, device=device, dtype=dtype)
        reg = torch.zeros(B, 6, DET_H, DET_W, device=device, dtype=dtype)
        msk = torch.zeros(B, 1, DET_H, DET_W, device=device, dtype=dtype)
        ys = torch.arange(DET_H, device=device, dtype=dtype)
        xs = torch.arange(DET_W, device=device, dtype=dtype)
        # two passes: neighbours first, centres last (centres always win) --
        # decode reads reg at the heatmap PEAK cell, which can sit 1 cell off
        # the GT centre, so the 3x3 neighbourhood must carry valid targets
        # (per-cell offsets; size/yaw shared) or yaw/size come out untrained
        centres = []
        for bi in range(B):
            for k in range(int(nbox[bi])):
                cls, xe, ye, l, w, yaw = boxes[bi, k].tolist()
                # Per-GT confidence from LiDAR support. Measured on val: a GT
                # vehicle box carries 857 LiDAR points at 0-20 m, 232 at 20-40,
                # 59 at 40-60 and 24 at 60-80, where 13 % of the boxes have
                # none at all. Restricting the evaluation to boxes with >= 40
                # points lifts 40-60 m recall from 0.34 to 0.44, so a large
                # part of what looks like a miss is a box the sensors never
                # actually saw. Training on those as hard positives teaches the
                # detector to fire where there is no evidence, which is where
                # the far-range precision goes (0.86 at 20-40 m, 0.56 at
                # 40-60 m). gw scales the peak of the Gaussian, so an
                # unsupported box neither demands a detection nor counts as
                # background.
                gw = 1.0 if gt_w is None else float(gt_w[bi, k])
                if gw <= 0.0:
                    continue
                r = (BEV_XF - xe) / DET_RES
                c = (BEV_YH - ye) / DET_RES
                ri, ci = int(r), int(c)
                if not (0 <= ri < DET_H and 0 <= ci < DET_W):
                    continue
                rad = min(max(2.0, 0.7 * max(l, w) / DET_RES / 2), 4.0)
                g = torch.exp(-(((ys - r) ** 2).view(-1, 1)
                                + ((xs - c) ** 2).view(1, -1)) / (2 * rad ** 2))
                ch = 0 if cls < 1.5 else 1
                hm[bi, ch] = torch.maximum(hm[bi, ch], g * gw)
                ll, lw = math.log(max(l, .1)), math.log(max(w, .1))
                sy, cy = math.sin(yaw), math.cos(yaw)
                # crossing vehicles (lateral yaw) are rare and their yaw
                # regresses toward the along-road prior -> weight by
                # lateralness (mask doubles as the per-cell reg weight)
                wy = 1.0 + 2.0 * abs(sy)
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        r2, c2 = ri + dr, ci + dc
                        if dr == dc == 0                                 or not (0 <= r2 < DET_H and 0 <= c2 < DET_W):
                            continue
                        reg[bi, :, r2, c2] = torch.tensor(
                            [r - r2, c - c2, ll, lw, sy, cy],
                            device=device, dtype=dtype)
                        msk[bi, 0, r2, c2] = wy
                centres.append((bi, ri, ci, wy,
                                torch.tensor([r - ri, c - ci, ll, lw, sy, cy],
                                             device=device, dtype=dtype)))
        for bi, ri, ci, wy, t in centres:
            reg[bi, :, ri, ci] = t
            msk[bi, 0, ri, ci] = wy
        return hm, reg, msk

    def boxdet_loss(self, hm, reg, boxes, nbox, gt_w=None, corner_w=0.0):
        hm_t, reg_t, m = self.build_det_targets(boxes, nbox, hm.device,
                                                torch.float32, gt_w=gt_w)
        p = hm.float().sigmoid().clamp(1e-4, 1 - 1e-4)
        pos = (hm_t > 0.99).float()
        neg_w = (1 - hm_t) ** 4
        # near-range VRU emphasis: VRU channel x2.5, positives within 20 m
        # of ego x2 (user: near bicycles/bikes/pedestrians are weak)
        if getattr(self, "_det_posw", None) is None \
                or self._det_posw.device != hm.device:
            rr = torch.arange(DET_H, device=hm.device).view(-1, 1)
            cc = torch.arange(DET_W, device=hm.device).view(1, -1)
            xe = BEV_XF - rr * DET_RES
            ye = BEV_YH - cc * DET_RES
            r = (xe ** 2 + ye ** 2).sqrt()
            # near-range recall boost, per class: the veh boost is softened
            # (x3 overfired -> near duplicate/phantom FPs); VRU keeps x3
            near_veh = 1.0 + 0.25 * (r < 20.0).float() + 0.25 * (r < 12.0).float()
            near_vru = 1.0 + (r < 20.0).float() + (r < 12.0).float()
            near = torch.stack([near_veh, near_vru])
            # v131' (2026-08-28): VRU クラス重みをフラグ化 (既定 5.0 = 従来値)
            cw = torch.tensor([2.0, float(getattr(self, "VRU_CW", 5.0))],
                              device=hm.device).view(2, 1, 1)
            # far positives are unresolvable at 768x432 (a 60 m pedestrian is
            # ~10 px); full-weight unlearnable positives push the focal loss
            # to suppress confidence everywhere -> damp them instead.
            # v41 (BOX_FAR_W): VEHICLES are resolvable at range by the narrow
            # cameras (~4.7 px/cell at 60 m), so BOOST far veh positives to
            # lift recall; keep VRU far-damping (far pedestrians truly ~10 px).
            far_w = getattr(self, "BOX_FAR_W", None)
            if getattr(self, "VRU_FAR_BAND", False):
                # v42: the narrow cams resolve a 40 m pedestrian (~60 px),
                # so the old >40 m x0.2 damp was discarding learnable GT.
                # Boost the 25-45 m band x2, damp only past 50 m.
                d_vru = ((1.0 + ((r > 25.0) & (r < 45.0)).float())
                         * torch.where(r > 50.0, 0.3, 1.0))
                damp = torch.stack([torch.where(r > 40.0, float(far_w or 2.0),
                                                1.0), d_vru])
            elif far_w:
                damp = torch.stack([torch.where(r > 40.0, float(far_w), 1.0),
                                    torch.where(r > 40.0, 0.2, 1.0)])
            else:
                damp = torch.stack([torch.where(r > 50.0, 0.3, 1.0),
                                    torch.where(r > 40.0, 0.2, 1.0)])
            # laterally distant objects are out of scope -> nearly ignore
            damp = damp * torch.where(ye.abs() > 15.0, 0.2, 1.0)
            self._det_posw = (near * cw * damp).unsqueeze(0)
        _neg = (1 - pos) * neg_w
        if getattr(self, "_det_sup_km", None) is not None:
            _neg = _neg * self._det_sup_km      # 窓外の負例も無監督に
        floss = -(pos * self._det_posw * (1 - p) ** 2 * p.log()
                  + _neg * p ** 2 * (1 - p).log()).sum() \
            / (pos * self._det_posw).sum().clamp(min=1)
        # yaw channels (sin/cos) x3: orientation error is the weakest output
        if getattr(self, "_reg_cw", None) is None \
                or self._reg_cw.device != hm.device:
            self._reg_cw = torch.tensor([1., 1., 1., 1., 3., 3.],
                                        device=hm.device).view(1, 6, 1, 1)
        # yaw-balance (v68+ 事前登録): 予測が軸平行モードに崩壊した実測
        # (斜めGT 15-75deg で誤差 31-43deg) への的絞りレバー。GT の斜め度で
        # yaw チャネル (4:6) の重みを持ち上げる。
        #
        # 2026-08-15 修正: 重みが sin(2*yaw)^2 = (2 sin cos)^2 だったため、
        # 真横 (yaw=90deg, sin=1 cos=0) で 0 になり、自車と平行な車 (0deg) と
        # 同じ「重み無し」扱いになっていた。ペア比較 (両モデルが検出できた
        # 共通 801 箱) で軽量版の yaw 誤差中央値は 0-15deg 帯 2.1deg に対し
        # 75-90deg 帯 39.4deg と、まさに重みが消える帯で最大に壊れている。
        # 単調な sin^2 に直す (0deg -> 0, 45deg -> 0.5, 90deg -> 1.0)。
        # これは「斜めほど重く」という登録済みの意図どおりの形であり、
        # レバー自体は METEOR_YAW_BALANCE_W>0 のときだけ効く (既定は無効)。
        # 形の指数は METEOR_YAW_BALANCE_P で選ぶ (既定 2 = sin^2)。
        # p=1 の |sin| は 15-45deg 帯にも効くが正対帯への影響が大きい。
        # どちらが良いかは実測で決める (out/probe_yawlever*.log)。
        # 2026-08-15 追加: METEOR_YAW_BALANCE_SAT=<度> を指定すると、その角度
        # 以上の斜めを一律で最大重みにする飽和形になる。v76 (p=1, w=2) の
        # 本走行で真横 34.3->6.5 度と直った一方、15-45 度が 35.5->47.4 度へ
        # 悪化した。|sin| は 90 度に重みが集中し 15-45 度に薄い (15 度で 0.26)
        # ため。飽和形なら斜めの全帯を同じ重さで扱える。
        ybw = float(os.environ.get("METEOR_YAW_BALANCE_W", "0"))
        ybp = float(os.environ.get("METEOR_YAW_BALANCE_P", "2"))
        ybs = float(os.environ.get("METEOR_YAW_BALANCE_SAT", "0"))
        _yw = 1.0
        if ybw > 0:
            if ybs > 0:
                _s = math.sin(math.radians(ybs))
                obl = (reg_t[:, 4:5].abs() / _s).clamp(max=1.0)
            else:
                obl = reg_t[:, 4:5].abs() ** ybp       # |sin(yaw)|^p
            _yw = torch.ones_like(reg_t)
            _yw[:, 4:6] = 1.0 + ybw * obl.expand(-1, 2, -1, -1)
        rloss = (torch.abs(reg.float() - reg_t) * m * _yw * self._reg_cw).sum() \
            / m.sum().clamp(min=1) / 10
        corner_loss = reg.float().sum() * 0.0
        if corner_w > 0:
            # Joint metric-space geometry constraint. Independent encoded L1
            # terms can trade a centre error against size/yaw and still look
            # cheap; matching all four physical corners couples those errors.
            # It is training-only and adds no operation to the exported graph.
            bi, ri, ci = (m[:, 0] > 0).nonzero(as_tuple=True)
            if bi.numel():
                pr = reg.float()[bi, :, ri, ci]
                tr = reg_t[bi, :, ri, ci]

                def _corners(v):
                    centre = torch.stack([
                        (ri.float() + v[:, 0]) * DET_RES,
                        (ci.float() + v[:, 1]) * DET_RES], dim=-1)
                    length = v[:, 2].clamp(-4.0, 5.0).exp()
                    width = v[:, 3].clamp(-4.0, 5.0).exp()
                    norm = (v[:, 4].square() + v[:, 5].square()
                            ).clamp_min(1e-6).sqrt()
                    sn, cs = v[:, 4] / norm, v[:, 5] / norm
                    signs = v.new_tensor([[1., 1.], [1., -1.],
                                          [-1., -1.], [-1., 1.]])
                    lx = length[:, None] * signs[None, :, 0] * 0.5
                    ly = width[:, None] * signs[None, :, 1] * 0.5
                    dx = lx * cs[:, None] - ly * sn[:, None]
                    dy = lx * sn[:, None] + ly * cs[:, None]
                    return centre[:, None, :] + torch.stack([dx, dy], -1)

                corner_loss = F.smooth_l1_loss(
                    _corners(pr), _corners(tr), beta=0.25)
        return floss + rloss + float(corner_w) * corner_loss

    @staticmethod
    def decode_boxes(hm, reg, thresh=0.3, topk=64):
        """-> list per batch of (cls, score, xe, ye, l, w, yaw)."""
        p = hm.sigmoid()
        pmax = F.max_pool2d(p, 5, 1, 2)
        p = p * (pmax == p)                        # 5x5 NMS (2 m): near-range
        # duplicate peaks on large vehicles were the top veh-FP source
        B, C, Hh, Ww = p.shape
        out = []
        for bi in range(B):
            flat = p[bi].reshape(-1)
            sc, idx = flat.topk(min(topk, flat.numel()))
            keep = sc > thresh
            sc, idx = sc[keep], idx[keep]
            cls = idx // (Hh * Ww)
            rc = idx % (Hh * Ww)
            ri = (rc // Ww).float()
            ci = (rc % Ww).float()
            boxes = []
            for j in range(len(sc)):
                rr, cc = int(ri[j]), int(ci[j])
                o = reg[bi, :, rr, cc]
                r = ri[j] + o[0]
                c = ci[j] + o[1]
                xe = BEV_XF - float(r) * DET_RES
                ye = BEV_YH - float(c) * DET_RES
                l = float(o[2].exp())
                w = float(o[3].exp())
                yaw = float(torch.atan2(o[4], o[5]))
                boxes.append((int(cls[j]), float(sc[j]), xe, ye, l, w, yaw))
            out.append(boxes)
        return out


DET2D_S = 4                    # 2D det grid stride on the cached image (108x192)
class PACTReLU(nn.Module):
    """上限が学習可能な ReLU。y = clamp(x, 0, alpha)。

    学習時は 0.5*(|x| - |x-a| + a) の形で書く (これは clamp と数値的に
    同一で、x > a の領域から alpha へ勾配が流れる = PACT の要点)。
    推論時は定数の clamp にするので ONNX には Clip が 1 個出るだけ。
    """

    def __init__(self, alpha_init=64.0, inplace=True):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.inplace = bool(inplace)

    def forward(self, x):
        # 値は必ず hardtanh の in-place 1 発で作る。置換元の ReLU が
        # inplace=True なので、非 in-place にすると別名参照の見え方が変わり
        # 補助ヘッドで相対 5-7% ずれる (実測)。hardtanh + inplace なら
        # 素の ReLU と **ビット一致** することを確認済み。
        # ONNX には Clip が 1 個出るだけなので TensorRT が前段の conv+BN に
        # 融合する = 推論コストはゼロ。
        a = self.alpha.abs().clamp(min=1e-3)
        if self.training:
            # alpha へ勾配を流す (PACT): 上限に当たった画素で d/da = 1。
            # 値は 0 を足すだけなので数値は hardtanh と完全に同じ。
            with torch.no_grad():
                clipped = (x > a).to(x.dtype)
            y = F.hardtanh(x, 0.0, float(a), inplace=self.inplace)
            return y + (a.to(x.dtype) - float(a)) * clipped
        return F.hardtanh(x, 0.0, float(a), inplace=self.inplace)

    def extra_repr(self):
        return f"alpha={float(self.alpha):.3f}"


N_DET2D = 10                   # 10-class 2D instance taxonomy (comlops-instance-2510.csv)


class DepthSegIPMNetV17(DepthSegIPMNetV16):
    """v17: + per-camera 10-class 2D-bbox detection head (CenterNet-style)
    on the shared stride-4 image feature. TRT-safe ops only.

      hm2d  [B,N,10,108,192] class center heatmaps - focal loss
      reg2d [B,N,4,108,192]  (off_x, off_y, log w, log h) in cells - L1@centers
    forward -> (seg, depth_logits, seg2d, hm, reg, hm2d, reg2d).
    """
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.det2d_stem = ConvBlock(160, 128)
        self.hm2d_head = nn.Conv2d(128, N_DET2D, 1)
        self.reg2d_head = nn.Conv2d(128, 4, 1)
        nn.init.constant_(self.hm2d_head.bias, -2.19)   # focal init

    def forward(self, imgs, K, T_cam_ego):
        B, N = imgs.shape[:2]
        out = super().forward(imgs, K, T_cam_ego)
        return out + self.det2d_forward(self._last_f, B, N)

    def det2d_forward(self, f, B, N):
        _c = getattr(self, "_det2d_cache", None)
        if _c is not None:               # paint-det が前倒し計算済み
            d, hm2d = _c
            self._det2d_cache = None
        else:
            d = self.det2d_stem(f)
            hm2d = self.hm2d_head(d)
        reg2d = self.reg2d_head(d)
        fh, fw = hm2d.shape[-2:]
        return (hm2d.view(B, N, N_DET2D, fh, fw),
                reg2d.view(B, N, 4, fh, fw))

    def image_feats(self, imgs):
        f = super().image_feats(imgs)
        self._last_f = f           # reuse the shared feature for the 2D head
        return f

    @staticmethod
    def build_det2d_targets(boxes, nbox, hw, device, dtype=torch.float32,
                            stride=DET2D_S):
        """boxes [B,N,K,5] (cls,cx,cy,w,h in cached px), nbox [B,N] ->
        (hm_t [BN,10,h,w], reg_t [BN,4,h,w], m_t [BN,1,h,w]).
        Boxes with w<=0 (zeroed by a scale filter) are skipped."""
        B, N = boxes.shape[:2]
        h, w = hw
        BN = B * N
        hm = torch.zeros(BN, N_DET2D, h, w, device=device, dtype=dtype)
        reg = torch.zeros(BN, 4, h, w, device=device, dtype=dtype)
        msk = torch.zeros(BN, 1, h, w, device=device, dtype=dtype)
        ys = torch.arange(h, device=device, dtype=dtype)
        xs = torch.arange(w, device=device, dtype=dtype)
        bb = boxes.view(BN, -1, 5)
        nn_ = nbox.view(BN)
        DET2D_S = stride
        for i in range(BN):
            for k in range(int(nn_[i])):
                cls, cx, cy, bw, bh = bb[i, k].tolist()
                if bw <= 0 or bh <= 0:
                    continue
                r, c = cy / DET2D_S, cx / DET2D_S
                ri, ci = int(r), int(c)
                if not (0 <= ri < h and 0 <= ci < w):
                    continue
                rad = max(1.0, 0.35 * max(bw, bh) / DET2D_S / 2)
                g = torch.exp(-(((ys - r) ** 2).view(-1, 1)
                                + ((xs - c) ** 2).view(1, -1)) / (2 * rad ** 2))
                ch = int(cls)
                hm[i, ch] = torch.maximum(hm[i, ch], g)
                reg[i, :, ri, ci] = torch.tensor(
                    [r - ri, c - ci, math.log(max(bw / DET2D_S, .25)),
                     math.log(max(bh / DET2D_S, .25))],
                    device=device, dtype=dtype)
                msk[i, 0, ri, ci] = 1
        return hm, reg, msk

    def bbox2d_loss(self, hm2d, reg2d, boxes, nbox):
        B, N = hm2d.shape[:2]
        hw = hm2d.shape[-2:]
        hm_t, reg_t, m = self.build_det2d_targets(boxes, nbox, hw,
                                                  hm2d.device, torch.float32)
        p = hm2d.view(B * N, N_DET2D, *hw).float().sigmoid().clamp(1e-4, 1 - 1e-4)
        pos = (hm_t > 0.99).float()
        neg_w = (1 - hm_t) ** 4
        floss = -(pos * (1 - p) ** 2 * p.log()
                  + (1 - pos) * neg_w * p ** 2 * (1 - p).log()).sum() \
            / pos.sum().clamp(min=1)
        rloss = (torch.abs(reg2d.view(B * N, 4, *hw).float() - reg_t)
                 * m).sum() / m.sum().clamp(min=1) / 4
        return floss + rloss

    @staticmethod
    def decode_boxes2d(hm2d, reg2d, thresh=0.3, topk=48):
        """[N,10,h,w],[N,4,h,w] -> per-cam list of (cls,score,cx,cy,w,h) px."""
        p = hm2d.sigmoid()
        pmax = F.max_pool2d(p, 3, 1, 1)
        p = p * (pmax == p)
        N, C, h, w = p.shape
        out = []
        for ni in range(N):
            flat = p[ni].reshape(-1)
            sc, idx = flat.topk(min(topk, flat.numel()))
            keep = sc > thresh
            sc, idx = sc[keep], idx[keep]
            cls = idx // (h * w)
            rc = idx % (h * w)
            boxes = []
            for j in range(len(sc)):
                rr, cc = int(rc[j] // w), int(rc[j] % w)
                o = reg2d[ni, :, rr, cc]
                cy = (rr + float(o[0])) * DET2D_S
                cx = (cc + float(o[1])) * DET2D_S
                boxes.append((int(cls[j]), float(sc[j]), cx, cy,
                              float(o[2].exp()) * DET2D_S,
                              float(o[3].exp()) * DET2D_S))
            out.append(boxes)
        return out


EGO_HORIZON = 6                # trajectory waypoints @ 0.5 s (3 s)
EGO_OUT = EGO_HORIZON * 2 + 3  # wp(12) + steer + accel + brake-logit


class DepthSegIPMNetV18(DepthSegIPMNetV17):
    """v18: + E2E ego head (trajectory / steering / accel / brake) pooled
    from the BEV feature, conditioned on current speed v0. TRT-safe.

    forward(imgs, K, T, v0=None) ->
      (..., v17 outputs ..., ego [B, 15]) where ego =
      [wp_x1,wp_y1,...,wp_x6,wp_y6, steer(rad), accel(m/s^2), brake_logit]
    """
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.ego_stem = nn.Sequential(
            nn.Conv2d(BEV_CH, 64, 3, stride=4, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),      # 200x125
            ConvBlock(64, 96),
            nn.Conv2d(96, 96, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(96), nn.ReLU(inplace=True),      # 100x63
            nn.AdaptiveAvgPool2d(1))
        self.ego_mlp = nn.Sequential(
            nn.Linear(96 + 1, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 256), nn.ReLU(inplace=True),
            nn.Linear(256, EGO_OUT))

    def forward(self, imgs, K, T_cam_ego, v0=None):
        B = imgs.shape[0]
        out = super().forward(imgs, K, T_cam_ego)
        g = self.ego_stem(self._fused_bev).flatten(1)       # [B,96]
        if v0 is None:
            v0 = torch.zeros(B, device=imgs.device, dtype=g.dtype)
        ego = self.ego_mlp(torch.cat([g, v0.view(B, 1).to(g.dtype)], 1))
        return out + (ego,)

    @staticmethod
    def ego_loss(ego, gt):
        """gt [B,17] = wp(12), v0, acc, steer, brake, valid.

        Curves are rare (17% of frames have |lat@3s|>2m) and lateral offsets
        are ~18x smaller than longitudinal ones, so a plain 12-dim L1 learns
        a go-straight prior. Counter both: lateral errors weighted x4 and a
        per-sample curvature weight 1+|lat@3s|/1.5 (up to x5).
        """
        valid = gt[:, 16:17]
        n = valid.sum().clamp(min=1)
        err = torch.abs(ego[:, :12] - gt[:, :12]).view(-1, 6, 2)
        wp_e = (err[:, :, 0] + 4.0 * err[:, :, 1]).mean(1, keepdim=True) / 2.5
        cw = 1.0 + gt[:, 11:12].abs().clamp(max=6.0) / 1.5   # curve weight
        nw = (cw * valid).sum().clamp(min=1)
        wl = (wp_e * cw * valid).sum() / nw
        sl = (torch.abs(ego[:, 12:13] - gt[:, 14:15]) * cw * valid).sum() / nw
        al = (torch.abs(ego[:, 13:14] - gt[:, 13:14]) * valid).sum() / n
        p = ego[:, 14:15].clamp(-15, 15)
        bl = (F.binary_cross_entropy_with_logits(
            p, gt[:, 15:16], reduction="none") * valid).sum() / n
        return wl + 2.0 * sl + al + 0.5 * bl


class SegHeadED(nn.Module):
    """Encoder-decoder 2D seg head: heavy convs at s8/s16 (cheap px), light
    skip at s4. ~4.2M params at roughly the FLOPs of two s4 convs."""
    def __init__(self, cin, n_seg):
        super().__init__()
        self.skip = nn.Conv2d(cin, 96, 1)
        self.d1 = nn.Sequential(
            nn.Conv2d(cin, 192, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(192), nn.ReLU(inplace=True), ConvBlock(192, 192))
        self.d2 = nn.Sequential(
            nn.Conv2d(192, 320, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(320), nn.ReLU(inplace=True), ConvBlock(320, 320))
        self.u1 = nn.Conv2d(320, 192, 1)
        self.m1 = ConvBlock(192, 192)
        self.u2 = nn.Conv2d(192, 96, 1)
        self.out = nn.Sequential(
            nn.Conv2d(96, 96, 3, padding=1, bias=False),
            nn.BatchNorm2d(96), nn.ReLU(inplace=True),
            nn.Conv2d(96, n_seg, 1))

    def forward(self, f):
        x1 = self.d1(f)
        x2 = self.d2(x1)
        y1 = self.m1(x1 + F.interpolate(self.u1(x2), size=x1.shape[-2:],
                                        mode="bilinear", align_corners=False))
        y0 = self.skip(f) + F.interpolate(self.u2(y1), size=f.shape[-2:],
                                          mode="bilinear", align_corners=False)
        return self.out(y0)


class DepthSegIPMNetV19(DepthSegIPMNetV18):
    """v19: capacity re-balance (user request: params up, FLOPs bounded).

    - 2D seg: SegHeadED encoder-decoder (0.23M -> ~4.2M)
    - 2D det: YOLO-like 3-scale pyramid on the shared s4 feature:
        s4 (small: cones/lights/far objects), s8 (mid), s16 (large).
        GT assigned by max(w,h): <40 px -> s4, <120 -> s8, else s16.
    - E2E ego head: deeper pyramid + wider MLP (0.37M -> ~4.0M).
    forward -> same 8-tuple as v18; hm2d/reg2d entries are 3-tuples.
    """
    DET2D_SPLIT = (40.0, 120.0)
    DET2D_STRIDES = (4, 8, 16)

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        n_seg = self.seg_head[-1].out_channels
        self.seg_head = SegHeadED(160, n_seg)
        # multi-scale 2D det (det2d_stem/hm2d_head/reg2d_head = s4 scale)
        self.det2d_d8 = nn.Sequential(
            nn.Conv2d(128, 192, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(192), nn.ReLU(inplace=True), ConvBlock(192, 192))
        self.det2d_d16 = nn.Sequential(
            nn.Conv2d(192, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True), ConvBlock(256, 256))
        self.hm2d_head8 = nn.Conv2d(192, N_DET2D, 1)
        self.reg2d_head8 = nn.Conv2d(192, 4, 1)
        self.hm2d_head16 = nn.Conv2d(256, N_DET2D, 1)
        self.reg2d_head16 = nn.Conv2d(256, 4, 1)
        for m in (self.hm2d_head8, self.hm2d_head16):
            nn.init.constant_(m.bias, -2.19)
        # E2E: deeper pyramid + wider MLP (most important task)
        self.ego_stem = nn.Sequential(
            nn.Conv2d(BEV_CH, 128, 3, stride=4, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),     # 200x125
            ConvBlock(128, 128),
            nn.Conv2d(128, 192, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(192), nn.ReLU(inplace=True),     # 100x63
            ConvBlock(192, 192),
            nn.Conv2d(192, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),     # 50x32
            ConvBlock(256, 256),
            nn.Conv2d(256, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),     # 25x16
            nn.AdaptiveAvgPool2d(1))
        self.ego_mlp = nn.Sequential(
            nn.Linear(256 + 1, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 256), nn.ReLU(inplace=True),
            nn.Linear(256, EGO_OUT))

    def det2d_forward(self, f, B, N):
        # paint-det が有効なときは s4 段 (det2d_stem + hm2d_head) をリフト前に
        # 計算済み。同じ f から出るので等価であり、ここで再利用して二重計算を
        # 避ける (キャッシュは 1 回で使い切る)。
        _c = getattr(self, "_det2d_cache", None)
        if _c is not None and _c[0].shape[0] == f.shape[0]:
            d4, _hm4 = _c
            self._det2d_cache = None
        else:
            d4, _hm4 = self.det2d_stem(f), None
        d8 = self.det2d_d8(d4)
        d16 = self.det2d_d16(d8)
        hms, regs = [], []
        for _si, (d, hh, rr) in enumerate((
                (d4, self.hm2d_head, self.reg2d_head),
                (d8, self.hm2d_head8, self.reg2d_head8),
                (d16, self.hm2d_head16, self.reg2d_head16))):
            hm = _hm4 if (_si == 0 and _hm4 is not None) else hh(d)
            rg = rr(d)
            fh, fw = hm.shape[-2:]
            hms.append(hm.view(B, N, N_DET2D, fh, fw))
            regs.append(rg.view(B, N, 4, fh, fw))
        return (tuple(hms), tuple(regs))

    def bbox2d_loss(self, hm2d, reg2d, boxes, nbox):
        """Multi-scale: assign boxes to a scale by max(w,h) px."""
        size = boxes[..., 3:5].max(-1).values          # [B,N,K]
        lo = (0.0,) + self.DET2D_SPLIT
        hi = self.DET2D_SPLIT + (1e9,)
        total = 0.0
        for si, (hm, reg) in enumerate(zip(hm2d, reg2d)):
            sel = (size >= lo[si]) & (size < hi[si]) & (boxes[..., 3] > 0)
            bsel = boxes * sel.unsqueeze(-1)           # others w=0 -> skipped
            B, N = hm.shape[:2]
            hw = hm.shape[-2:]
            hm_t, reg_t, m = self.build_det2d_targets(
                bsel, nbox, hw, hm.device, torch.float32,
                stride=self.DET2D_STRIDES[si])
            p = hm.reshape(B * N, N_DET2D, *hw).float().sigmoid() \
                .clamp(1e-4, 1 - 1e-4)
            pos = (hm_t > 0.99).float()
            neg_w = (1 - hm_t) ** 4
            # rare/small classes: obs(cone/unknown) x3, 2-wheelers x2,
            # traffic light x2, ped/sign x1.5
            if getattr(self, "_det2d_cw", None) is None \
                    or self._det2d_cw.device != hm.device:
                self._det2d_cw = torch.tensor(
                    [3.0, 1.0, 1.0, 1.5, 2.0, 2.0, 1.5, 1.0, 2.0, 1.5],
                    device=hm.device).view(1, N_DET2D, 1, 1)
            fl = -(pos * self._det2d_cw * (1 - p) ** 2 * p.log()
                   + (1 - pos) * neg_w * p ** 2 * (1 - p).log()).sum() \
                / (pos * self._det2d_cw).sum().clamp(min=1)
            rl = (torch.abs(reg.reshape(B * N, 4, *hw).float() - reg_t)
                  * m).sum() / m.sum().clamp(min=1) / 4
            total = total + fl + rl
        return total / len(hm2d)

    @classmethod
    def decode_boxes2d_ms(cls, hms, regs, thresh=0.3, topk=48):
        """Merge per-scale decodes -> per-cam list of (cls,score,cx,cy,w,h)."""
        out = None
        for si, (hm, reg) in enumerate(zip(hms, regs)):
            s = cls.DET2D_STRIDES[si]
            p = hm.sigmoid()
            pmax = F.max_pool2d(p, 3, 1, 1)
            p = p * (pmax == p)
            N, C, h, w = p.shape
            if out is None:
                out = [[] for _ in range(N)]
            for ni in range(N):
                flat = p[ni].reshape(-1)
                sc, idx = flat.topk(min(topk, flat.numel()))
                keep = sc > thresh
                sc, idx = sc[keep], idx[keep]
                ccls = idx // (h * w)
                rc = idx % (h * w)
                for j in range(len(sc)):
                    rr, cc = int(rc[j] // w), int(rc[j] % w)
                    o = reg[ni, :, rr, cc]
                    out[ni].append((int(ccls[j]), float(sc[j]),
                                    (cc + float(o[1])) * s,
                                    (rr + float(o[0])) * s,
                                    float(o[2].exp()) * s,
                                    float(o[3].exp()) * s))
        return out


OCC_Z, OCC_C = 16, 10          # z bins / classes (0 free + 9, 255 ignore)


class DepthSegIPMNetV20(DepthSegIPMNetV19):
    """v20: + 3D semantic occupancy head.

    BEV feature (96ch 800x500 @0.2m) cropped to +-40 x +-40 m (rows 200:600,
    cols 50:450) -> stride-2 stem -> 1x1 to OCC_Z*OCC_C channels ->
    [B, C=10, Z=16, 200, 200] logits @0.4m voxels, aligned with the
    extract_occ grid. TRT-safe (conv only). +~0.7M params / ~55 GFLOPs.
    forward -> v19 outputs + (occ,).
    """
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.occ_stem = nn.Sequential(
            nn.Conv2d(BEV_CH, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            ConvBlock(128, 192))
        self.occ_head = nn.Conv2d(192, OCC_Z * OCC_C, 1)

    def forward(self, imgs, K, T_cam_ego, v0=None):
        out = super().forward(imgs, K, T_cam_ego, v0)
        _r0, _r1 = bev_rows(40.0, -40.0)
        crop = self.occ_input()[:, :, _r0:_r1, 50:450]
        of = self.occ_stem(crop)
        self._occ_feat = of                    # v29 flow head reads this
        o = self.occ_head(of)
        B = o.shape[0]
        return out + (o.view(B, OCC_C, OCC_Z, o.shape[-2], o.shape[-1]),)

    def occ_loss(self, occ, occ_gt):
        """occ [B,C,Z,H,W], occ_gt [B,Z,H,W] uint8 (255 = unknown)."""
        if (occ_gt != 255).sum() == 0:
            return occ.sum() * 0.0
        if getattr(self, "_occ_w", None) is None \
                or self._occ_w.device != occ.device:
            w = torch.ones(OCC_C, device=occ.device)
            w[0] = 0.3                     # free dominates the carved volume
            w[2] = 1.5                     # vehicle (2.0 caused near halos)
            w[1] = 3.0                     # obstacle/unknown (cones etc.)
            w[[3, 4]] = 4.0                # 2-wheelers / pedestrians
            self._occ_w = w
            yy, xx = torch.meshgrid(
                torch.arange(occ.shape[-2], device=occ.device),
                torch.arange(occ.shape[-1], device=occ.device),
                indexing="ij")
            cy, cx = occ.shape[-2] / 2.0, occ.shape[-1] / 2.0
            self._occ_near = (((yy - cy) ** 2 + (xx - cx) ** 2)
                              <= 30.0 ** 2)          # 12 m at 0.4 m cells
        ce = F.cross_entropy(occ, occ_gt.long(), weight=self._occ_w,
                             ignore_index=255)
        # near-ego dynamic false positives (fragmentary phantom vehicles
        # beside the ego) get a dedicated penalty: where GT says FREE
        # inside the 12 m zone, push down the summed dynamic-class prob
        p = occ.float().softmax(1)
        pdyn = p[:, 2:5].sum(1)                       # veh + 2wheel + ped
        fp_mask = (occ_gt == 0) & self._occ_near
        if fp_mask.any():
            ce = ce + 0.5 * (-torch.log1p(
                -pdyn[fp_mask].clamp(max=0.999))).mean()
        return ce


TRAJ_H = 6                     # agent-forecast waypoints @0.5 s


class DepthSegIPMNetV21(DepthSegIPMNetV20):
    """v21: + one-shot agent trajectory forecasting.

    A 1x1 conv on the shared 3D-detection stem regresses, at every BEV
    detection cell, the agent's future offsets for 6 x 0.5 s horizons
    (metres, current ego frame). Constant cost in the number of agents
    (+0.05M params). forward -> v20 outputs + (traj [B,12,400,250],).
    """
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.traj_head = nn.Conv2d(128, TRAJ_H * 2, 1)

    def traj_feat(self):
        return self._det_feat           # v25: separate stem on the fused BEV

    def forward(self, imgs, K, T_cam_ego, v0=None):
        out = super().forward(imgs, K, T_cam_ego, v0)
        return out + (self.traj_head(self.traj_feat()),)

    @staticmethod
    def build_traj_targets(boxes, nbox, traj, tvalid, device):
        """-> (t [B,12,h,w], m [B,12,h,w]) at box-centre cells.

        The mask doubles as a class weight: VRU cells count x2.5 because a
        pedestrian's 3 s displacement is half a vehicle's (2.58 m vs 5.51 m
        on val), so their contribution was being drowned out by traffic.
        """
        Bn = boxes.shape[0]
        t = torch.zeros(Bn, TRAJ_H * 2, DET_H, DET_W, device=device)
        m = torch.zeros(Bn, TRAJ_H * 2, DET_H, DET_W, device=device)
        for bi in range(Bn):
            for k in range(int(nbox[bi])):
                cls, xe, ye = boxes[bi, k, 0], boxes[bi, k, 1], boxes[bi, k, 2]
                if boxes[bi, k, 3] <= 0:
                    continue
                ri = int((BEV_XF - float(xe)) / DET_RES)
                ci = int((BEV_YH - float(ye)) / DET_RES)
                if not (0 <= ri < DET_H and 0 <= ci < DET_W):
                    continue
                w = 2.5 if float(cls) >= 1.5 else 1.0      # VRU emphasis
                # oncoming vehicles: rare direction, and the failure the
                # user sees (heading flipped to ego-forward) concentrates
                # there -> same emphasis as VRUs
                yaw_k = float(boxes[bi, k, 5])
                if float(cls) < 1.5 and \
                        abs((yaw_k + math.pi) % (2 * math.pi) - math.pi) \
                        > 2.36:
                    w = 2.5
                t[bi, :, ri, ci] = traj[bi, k].reshape(-1)
                m[bi, :, ri, ci] = tvalid[bi, k].repeat_interleave(2) * w
        return t, m

    def traj_loss(self, tr_pred, boxes, nbox, traj, tvalid):
        t, m = self.build_traj_targets(boxes, nbox, traj, tvalid,
                                       tr_pred.device)
        if m.sum() == 0:
            return tr_pred.sum() * 0.0
        return (torch.abs(tr_pred.float() - t) * m).sum() / m.sum()


def enable_lane_branch(net, ch=64):
    """Dedicated decoder branch for the THIN classes (laneline / stopline /
    road_edge), residual on the shared 9-class logits.

    Why now: the Orin profile removed the cost objection -- every head together
    is ~18 ms, dec+seg 7.3, and this branch prices at 1-2 ms. What it buys that
    loss reweighting could not (four attempts, refuted by the gradient probe):
    the thin classes currently ride a decoder whose BatchNorm statistics and
    gradient budget are dominated by road/sidewalk, and their decision boundary
    sits inside a 9-way softmax where road competes at every lane pixel. A
    separate branch gives the thin classes their own normalisation statistics
    and their own capacity.

    The last conv is ZERO-INITIALISED: at attach time the network's function is
    bit-identical, so it can be dropped onto a trained checkpoint and measured
    as a pure residual. The 9-class output interface is unchanged -- the branch
    ADDS to channels 4..6 -- so every existing metric, demo and export reads it
    with no change.
    """
    import torch.nn as nn
    dev = next(net.parameters()).device
    net.lane_branch = nn.Sequential(
        nn.Conv2d(BEV_CH, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(ch, 3, 1)).to(dev)
    nn.init.zeros_(net.lane_branch[-1].weight)
    nn.init.zeros_(net.lane_branch[-1].bias)
    return net


class QuantRobustStatHead(nn.Module):
    """Function-preserving, bounded reparameterisation of a binary 1x1 head.

    For z=w*x+b, relu(z)-relu(-z)=z. Clamping both positive branches to
    ``cap`` bounds the deployed logit to +/-cap and gives TensorRT a compact,
    head-specific activation range instead of letting an implicit INT8 fusion
    collapse the one-channel output to zero.
    """
    def __init__(self, old, cap=8.0):
        super().__init__()
        self.cap = float(cap)
        self.proj = nn.Conv2d(old.in_channels, 2, 1, bias=True)
        self.out = nn.Conv2d(2, 1, 1, bias=False)
        with torch.no_grad():
            self.proj.weight[0].copy_(old.weight[0])
            self.proj.weight[1].copy_(-old.weight[0])
            b = old.bias[0] if old.bias is not None else old.weight.new_zeros(())
            self.proj.bias[0].copy_(b)
            self.proj.bias[1].copy_(-b)
            self.out.weight.zero_()
            self.out.weight[0, 0, 0, 0] = 1.0
            self.out.weight[0, 1, 0, 0] = -1.0

    def forward(self, x):
        p = self.proj(x).clamp(0.0, self.cap)
        return self.out(p)



def enable_kinematic_anchor(net):
    """v139: ego waypoint に等速直進項 g_t*[v0*t,0] をゲート付きで加算する。
    ゼロ初期化で関数保存。学習後は近距離の速度バイアスを構造的に除去する。"""
    import torch.nn as nn
    if getattr(net, "kin_gate", None) is not None:
        return net
    dev = next(net.parameters()).device
    net.kin_gate = nn.Parameter(torch.zeros(6, device=dev))
    return net

def enable_semantic_ego(net, ch=64):
    """ego の INT8 耐性化 (2026-08-25): ego_stem の入力を「意味出力」に変える。

    根本原因 (実測): ego は融合 BEV の生特徴を大域平均で読む。INT8 では
    (1) 平均後の値が入力スケールの流用で潰れ (真っ黒でも乱数でも出力が
    ビット一致)、(2) 上流 tfuse 帯の微細信号も破壊される。層精度・較正・
    PACT・conv 化のすべてが不合格で、fp16-keep 65 層 (+18.6ms) が唯一の
    回避策だった。
    対処: ego が読む量を **INT8 で健全と実証済みのテンソル** に変える ---
    seg ロジット (9ch)・det ヒートマップ (2ch)・占有の接地面 (合計 14ch 程度)。
    これらは意味の決定境界を持つためダイナミックレンジが大きく、量子化に
    構造的に強い (occ/traj/seg/det が INT8 で健全な理由そのもの)。
    旧 ego_stem/ego_mlp は残し、出力に**ゼロ初期化の残差**として加算する
    (導入時点は機能保存。学習が意味経路へ重みを移せば、fp16-keep から
    tfuse/ego を外しても E2E が生きる)。"""
    if getattr(net, "sem_ego", None) is not None:
        return net
    dev = next(net.parameters()).device
    # 入力: seg 9 (800x500 を 2x 平均して det 格子へ) + det hm 2 = 11ch
    # (occ は BEV と座標範囲が異なるため使わない)
    net.sem_ego = nn.Sequential(
        nn.Conv2d(11, ch, 3, stride=4, padding=1, bias=False),
        nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
        nn.Conv2d(ch, ch, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
        nn.Conv2d(ch, ch, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d((5, 4)), nn.Flatten(),
        nn.Linear(ch * 20, 256), nn.ReLU(inplace=True),
        nn.Linear(256, net.ego_mlp[-1].out_features)).to(dev)
    nn.init.zeros_(net.sem_ego[-1].weight)
    nn.init.zeros_(net.sem_ego[-1].bias)
    return net


def enable_delta_stat(net, ch=48):
    """停止判定の時間差分ヘッド (2026-08-25)。

    根本原因 (実測): stat_head2 が頼る微細な見えの差は INT8 バックボーンの
    深部で破壊され、ヘッド近傍の対処 5 種 (margin / 有界ヘッド後付け /
    有界ヘッド学習+拡張keep / traj 由来 / トラッキング=忠実代替) すべてが
    ゲート不合格だった。
    対処: 差分 |bev - warp(prev_bev)| を明示的にテンソルにし、そこから
    停止判定を出す。静止物体は差分 ~0、移動物体は差分大 --- 量子化スケールが
    動き信号そのもので決まるため、微小信号の目盛り消失が構造的に起きない。
    ワープは時間融合の実装 (配備済み) を流用。追加計算は pool + 小 conv 3 層。
    旧 stat_head2 は残す (fp16 実行では従来どおり比較できる)。"""
    if getattr(net, "delta_stat", None) is not None:
        return net
    dev = next(net.parameters()).device
    net.delta_stat = nn.Sequential(
        nn.AvgPool2d(2),                       # 800x500 -> 400x250 (det 格子)
        nn.Conv2d(BEV_CH, ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
        nn.Conv2d(ch, ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
        nn.Conv2d(ch, 1, 1)).to(dev)
    nn.init.zeros_(net.delta_stat[-1].bias)
    # out[10] を上書きされる stat_head2 は勾配を受けない。DDP は登録済みの
    # 全 requires_grad パラメータに勾配を要求するので、凍結しないと step2 の
    # reducer 再構築で落ちる (v124 初回起動で実害)。重みは fp16 比較用に残す。
    for _p in net.stat_head2.parameters():
        _p.requires_grad_(False)
    return net


def enable_depth_slim(net, scale=0.75, widths=None):
    """深度ヘッドの幅を縮める (Orin 100ms 台帳のレバー3, 2026-08-27)。

    v125fp のプロファイルで深度ヘッド (conv 5 本) が 12.8 ms と最大の単一
    ブロック。幅 256/256/192/128 を scale 倍 (8 の倍数へ丸め) にした新しい
    ヘッドへ置き換える。重みは新規初期化 — LiDAR 由来の深度 GT 監督が
    強いので蒸留は使わず、ラウンド内で学習し直す。
    widths を渡した場合はそれをそのまま使う (チェックポイント自動検出用)。"""
    old = net.depth_head
    ch = old[0][0].in_channels
    D = old[-1].out_channels
    if widths is None:
        widths = tuple(max(8, int(round(w * scale / 8)) * 8)
                       for w in (256, 256, 192, 128))
    dev = old[-1].weight.device
    net.depth_head = nn.Sequential(
        ConvBlock(ch, widths[0]), ConvBlock(widths[0], widths[1]),
        ConvBlock(widths[1], widths[2]), ConvBlock(widths[2], widths[3]),
        nn.Conv2d(widths[3], D, 1)).to(dev)
    return net


def enable_traj_flow(net, ch=256):
    """他車軌跡ヘッドへ flow 場を接続 (A レバー A1, 2026-08-27)。

    実測 (probe_agent_cvflow): flow の向きは 11.3° と traj ヘッド (27°) より
    大幅に良いのに、traj ヘッドは flow を読んでいない。flow (out[13],
    ±40m クロップの occ 格子) を det/traj 格子へ再配置し、ゼロ初期化 1x1 の
    残差として traj 特徴にのみ加算する (stat_head2 の入力は変えない)。
    detach 供給なので flow ヘッド自身の学習は汚さない。A2 (--flow-w 1.0) は
    flowerr 2.69m≈基準 2.64m で棄却 — 速度は損失重み律速ではない。"""
    if getattr(net, "traj_flow", None) is not None:
        return net
    dev = next(net.parameters()).device
    net.traj_flow = nn.Conv2d(2, ch, 1).to(dev)
    nn.init.zeros_(net.traj_flow.weight)
    nn.init.zeros_(net.traj_flow.bias)
    return net


def enable_det_temporal(net, ch=64):
    """D7 (2026-08-28): 3D 検出 hm へ時間特徴のゼロ初期化残差。

    遠方 recall と VRU は容量(D1)・露出(D2'/pV2)・重み(D3/pV1) の全レバーで
    動かず、残る仮説は「単フレームの証拠不足」。unk_head2 が時間特徴化で
    0.4m コーンを検出できるようになった前例と同型の接ぎ木。"""
    if getattr(net, "det_tmp", None) is not None:
        return net
    dev = next(net.parameters()).device
    net.det_tmp = nn.Sequential(
        nn.Conv2d(256, ch, 3, padding=1), nn.ReLU(inplace=True),
        nn.Conv2d(ch, 2, 1)).to(dev)
    nn.init.zeros_(net.det_tmp[-1].weight)
    nn.init.zeros_(net.det_tmp[-1].bias)
    return net


def enable_traj_cv(net):
    """A6 (2026-08-28): 他車軌跡を CV(v̂)+残差 に再パラメータ化。

    実測: agentADE 2.13m は oracle-CV 0.88m に大差負け = ヘッドが速度を
    因数分解できていない。ゼロ初期化の速度予測 v̂ [2ch] を足し、
    各モード k の waypoint に v̂*t_i を加算する (導入時は機能保存)。
    残差学習で v̂ に運動が寄れば CV 構造が内在化される。"""
    if getattr(net, "traj_vel", None) is not None:
        return net
    dev = next(net.parameters()).device
    net.traj_vel = nn.Conv2d(256, 2, 1).to(dev)
    nn.init.zeros_(net.traj_vel.weight)
    nn.init.zeros_(net.traj_vel.bias)
    return net


def enable_depth_logbins(net):
    """D5 (2026-08-29): 深度ビンを対数間隔化 (geomspace 1.0..79.75m x64)。
    遠方 recall の真因 (距離での深度証拠の質) への構造レバー。
    注意: 配備時はリフトプラグインのテーブル再生成が必要。"""
    import math
    ds = torch.exp(torch.linspace(math.log(1.0), math.log(79.75), 64))
    try:
        ds = ds.to(next(net.parameters()).device)
    except StopIteration:
        pass
    net.register_buffer("DEPTH_CENTERS", ds, persistent=False)
    return net


def enable_mode_scorer(net, ch=64):
    """E3 (2026-08-29): 経路条件付きモード選択スコアラ。

    実測: 選択ロス +0.15m は soft CE (tau 0.2/0.4) でも risk 積分でも回収
    できず、「候補を識別する入力の欠如」が律速。各候補の waypoint 位置で
    BEV 時間特徴を双線形サンプルし、小 MLP のスコアをモード logit へ
    ゼロ初期化残差として加算する (導入時は機能保存)。"""
    if getattr(net, "mode_scorer", None) is not None:
        return net
    dev = next(net.parameters()).device
    net.mode_scorer = nn.Sequential(
        nn.Linear(256, ch), nn.ReLU(inplace=True),
        nn.Linear(ch, 1)).to(dev)
    nn.init.zeros_(net.mode_scorer[-1].weight)
    nn.init.zeros_(net.mode_scorer[-1].bias)
    return net


def enable_quant_stat_head(net, cap=8.0):
    """Replace stat_head2 with its bounded, INT8-robust equivalent."""
    if isinstance(net.stat_head2, QuantRobustStatHead):
        return net
    old = net.stat_head2
    dev, dtype = old.weight.device, old.weight.dtype
    net.stat_head2 = QuantRobustStatHead(old, cap=cap).to(device=dev,
                                                              dtype=dtype)
    return net


def enable_seg_deep(net, ch=None, n=3):
    """Deepen the BEV Seg head: an n-block residual tower on the fused BEV
    feature, added to ALL 9 class logits (lane_branch covers only the thin 3).

    Rationale, measured not argued: the Orin profile prices every head
    together at ~18 ms of a 69-76 ms engine -- head capacity is nearly free
    at inference -- while the lane_branch probe showed BEV-side decoder
    capacity IS a binding constraint (+9 % stopline for 1-2 ms). Same
    zero-init-last-conv contract: attach to a trained checkpoint is
    function-preserving; measure as a pure residual. Call BEFORE DDP."""
    import torch.nn as nn
    dev = next(net.parameters()).device
    c = ch or BEV_CH
    blocks = []
    for i in range(n):
        blocks += [nn.Conv2d(BEV_CH if i == 0 else c, c, 3, padding=1,
                             bias=False), nn.BatchNorm2d(c),
                   nn.ReLU(inplace=True)]
    blocks += [nn.Conv2d(c, N_CLASSES, 1)]
    net.seg_deep = nn.Sequential(*blocks).to(dev)
    nn.init.zeros_(net.seg_deep[-1].weight)
    nn.init.zeros_(net.seg_deep[-1].bias)
    return net


def enable_det_deep(net, n=3):
    """Deepen the 3D detection trunk: an n-block zero-init residual tower on
    the det_stem features, so hm_head / reg_head (and the traj head, which
    reads the same features) see a richer representation. Detection runs at
    300x250 -- a quarter of the seg grid -- so each block prices well under
    1 ms on the Orin. Call BEFORE DDP."""
    import torch.nn as nn
    dev = next(net.parameters()).device
    c = net.hm_head[0].in_channels if hasattr(net.hm_head, "__getitem__")         else 128
    blocks = []
    for _ in range(n):
        blocks += [nn.Conv2d(c, c, 3, padding=1, bias=False),
                   nn.BatchNorm2d(c), nn.ReLU(inplace=True)]
    blocks += [nn.Conv2d(c, c, 1)]
    net.det_deep = nn.Sequential(*blocks).to(dev)
    nn.init.zeros_(net.det_deep[-1].weight)
    nn.init.zeros_(net.det_deep[-1].bias)
    return net


def enable_lane_sdf(net, ch=48):
    """Attach the lane signed-distance auxiliary head (probe, flag-gated).

    Why this and not another loss reweighting: laneline IoU has been pinned at
    0.09-0.13 for many rounds, the gradient probe showed every seg loss already
    pushes the ring thinner, and the identified ceiling is SUB-CELL LABEL
    JITTER -- a 0.2 m raster of a ~0.6 m line whose position noise is about
    half a cell (docs/MILESTONES.md M5). Per-cell CE on a jittered label learns
    the jitter. Regressing the DISTANCE FIELD to the nearest lane cell is
    smooth under that jitter, so across many scenes the regression target
    averages to the mean lane position -- sub-cell denoising on the model side,
    without waiting for the GT factory re-run.

    Call BEFORE the DDP wrap (it adds parameters)."""
    import torch.nn as nn
    dev = next(net.parameters()).device
    net.lane_sdf = nn.Sequential(
        nn.Conv2d(BEV_CH, ch, 3, padding=1), nn.ReLU(inplace=True),
        nn.Conv2d(ch, 1, 1)).to(dev)
    net._lane_sdf_pred = None
    return net


def make_warp_theta(rel):
    """rel [B,3] = (tx, ty, dyaw): current-frame BEV point p_c maps to the
    previous frame as p_p = R(dyaw) p_c + t. Returns affine theta [B,2,3]
    for F.affine_grid on the 800x500 BEV (row=(80-x)/0.2, col=(50-y)/0.2),
    so grid_sample pulls the previous feature into the current frame."""
    B = rel.shape[0]
    a = 0.1 * (BEV_H - 1)              # x = (80-a) - a*v
    b = 0.1 * (BEV_W - 1)              # y = (50-b) - b*u
    cd, sd = torch.cos(rel[:, 2]), torch.sin(rel[:, 2])
    tx, ty = rel[:, 0], rel[:, 1]
    Cx = cd * (80 - a) - sd * (50 - b) + tx
    Cy = sd * (80 - a) + cd * (50 - b) + ty
    th = torch.zeros(B, 2, 3, device=rel.device, dtype=rel.dtype)
    th[:, 0, 0] = cd                   # u_p = cd*u + (a*sd/b)*v + ...
    th[:, 0, 1] = a * sd / b
    th[:, 0, 2] = ((50 - b) - Cy) / b
    th[:, 1, 0] = -(b / a) * sd        # v_p = -(b*sd/a)*u + cd*v + ...
    th[:, 1, 1] = cd
    th[:, 1, 2] = ((80 - a) - Cx) / a
    return th


class DepthSegIPMNetV22(DepthSegIPMNetV21):
    """v22: streaming temporal BEV (TensorRT-safe).

    The previous frame's raw BEV feature, ego-motion-warped into the current
    frame (affine_grid + grid_sample), is fused residually with the current
    BEV. At deployment prev_bev / warp theta are graph INPUTS and the raw
    current BEV is an extra OUTPUT -> static feed-forward graph, no RNN.
    Improves velocity observability for E2E and agent forecasting.
    forward(imgs, K, T, v0, prev_bev, warp_theta) -> v21 outputs.
    """
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.tfuse = nn.Sequential(
            nn.Conv2d(2 * BEV_CH, BEV_CH, 1, bias=False), nn.BatchNorm2d(BEV_CH),
            nn.ReLU(inplace=True), ConvBlock(BEV_CH, BEV_CH))
        self._prev = (None, None)

    def forward(self, imgs, K, T_cam_ego, v0=None, prev_bev=None,
                warp_theta=None):
        self._prev = (prev_bev, warp_theta)
        return super().forward(imgs, K, T_cam_ego, v0)

    def temporal_fuse(self, bev):
        pb, th = self._prev
        if pb is None:
            pb = torch.zeros_like(bev)
            warped = pb
        else:
            grid = F.affine_grid(th.to(bev.dtype), list(bev.shape),
                                 align_corners=False)
            warped = F.grid_sample(pb.to(bev.dtype), grid,
                                   align_corners=False)
        return bev + self.tfuse(torch.cat([bev, warped], 1))


class LaneDecED(nn.Module):
    """Encoder-decoder BEV lane decoder: params at s2/s4 of the BEV, thin
    structures preserved by a full-resolution skip. ~4.1M params at ~0.8x
    the FLOPs of the flat full-res stack it replaces."""
    def __init__(self, cin, n_cls, w=1.0):
        """`w` scales every internal width. w=0.5 is 1.06M params instead of
        4.10M and 3.28 ms instead of 7.72 ms at 800x500 (measured, fp16):
        9 output classes did not need 320 channels at s4. Only BEV seg is
        affected -- the shared BEV feature `cin` is untouched."""
        super().__init__()
        c0, c1, c2 = (max(8, int(round(c * w))) for c in (64, 192, 320))
        self.skip = ConvBlock(cin, c0)
        self.d1 = nn.Sequential(
            nn.Conv2d(cin, c1, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c1), nn.ReLU(inplace=True), ConvBlock(c1, c1))
        self.d2 = nn.Sequential(
            nn.Conv2d(c1, c2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c2), nn.ReLU(inplace=True), ConvBlock(c2, c2))
        self.u1 = nn.Conv2d(c2, c1, 1)
        self.m1 = ConvBlock(c1, c1)
        self.u2 = nn.Conv2d(c1, c0, 1)
        self.out = nn.Sequential(
            nn.Conv2d(c0, c0, 3, padding=1, bias=False),
            nn.BatchNorm2d(c0), nn.ReLU(inplace=True),
            nn.Conv2d(c0, n_cls, 1))

    def forward(self, x):
        s = self.skip(x)
        x1 = self.d1(x)
        x2 = self.d2(x1)
        y1 = self.m1(x1 + F.interpolate(self.u1(x2), size=x1.shape[-2:],
                                        mode="bilinear", align_corners=False))
        y0 = s + F.interpolate(self.u2(y1), size=s.shape[-2:],
                               mode="bilinear", align_corners=False)
        return self.out(y0)


class DepthSegIPMNetV23(DepthSegIPMNetV22):
    """v23: BEV-priority capacity round.

    - tfuse last BN zero-initialised -> temporal fusion starts as identity
      and cannot perturb the pretrained BEV (fixes the v22 mIoU dip).
    - BEV lane decoder -> encoder-decoder (LaneDecED, 1.16M -> ~4.1M params
      at ~0.8x FLOPs).
    - 3D det stem gains an s4 256ch tower fused back at s2 (+1.9M).
    """
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        n_cls = self.dec[-1].out_channels
        self.dec = LaneDecED(BEV_CH, n_cls)
        # stronger det stem: old (conv s2 96->128 + ConvBlock) + s4 tower
        self.det_d8 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True), ConvBlock(256, 256))
        self.det_u = nn.Conv2d(256, 128, 1)
        self.det_m = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        base_stem = self.det_stem

        class _DetStemED(nn.Module):
            def __init__(self, stem, d8, u, m):
                super().__init__()
                self.stem, self.d8, self.u, self.m = stem, d8, u, m

            def forward(self, x):
                d4 = self.stem(x)
                d8 = self.d8(d4)
                return self.m(d4 + F.interpolate(
                    self.u(d8), size=d4.shape[-2:], mode="bilinear",
                    align_corners=False))

        self.det_stem = _DetStemED(base_stem, self.det_d8, self.det_u,
                                   self.det_m)
        # identity-start temporal fusion: zero the last BN of tfuse
        last_bn = self.tfuse[-1][-2]
        nn.init.zeros_(last_bn.weight)
        nn.init.zeros_(last_bn.bias)


class DepthSegIPMNetV24(DepthSegIPMNetV23):
    """v24: task-routed temporal BEV.

    Static tasks (BEV lanes, occupancy) read the RAW BEV — isolated from
    warp noise and moving-object ghosts of the temporal fusion — while
    motion tasks (3D det, agent forecast, E2E) keep the FUSED BEV where
    velocity cues live. Protects the top-priority mIoU without giving up
    the temporal gains."""

    def lane_input(self):
        return self._last_bev

    def occ_input(self):
        return self._last_bev


class DepthSegIPMNetV25(DepthSegIPMNetV24):
    """v25: BEV-geometry group on the RAW BEV (lanes + 3D boxes + occupancy
    — BEV seg and 3D detection share geometry), motion group on the FUSED
    BEV (E2E; agent forecasting via its own light stem keeps velocity)."""
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.traj_stem = nn.Sequential(
            nn.Conv2d(BEV_CH, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True), ConvBlock(128, 128))
        self.traj_head = nn.Conv2d(256, TRAJ_H * 2, 1)   # +det feat (class)

    def det_input(self):
        return self._last_bev           # 3D det joins the BEV-geometry group

    def traj_feat(self):
        # Motion comes from the FUSED BEV (velocity lives there), but the
        # shallow traj_stem cannot tell a crossing pedestrian from a car
        # following the road, so it regressed the dominant road-direction
        # prior for everyone: measured on val at r20 ep3, VRU GT headings
        # average 87.9 deg off ego-forward (crossing) while predictions
        # averaged 29.7 deg (along the road), a 75.9 deg mean error.
        # Concatenate the DETACHED detection feature -- it already encodes
        # class (it feeds the class heatmap) -- so the head can condition on
        # what the agent is. Detached: the trajectory loss must not perturb
        # 3D detection or the shared BEV.
        return torch.cat([self.traj_stem(self._fused_bev),
                          self._det_feat.detach()], 1)


class DepthSegIPMNetV26(DepthSegIPMNetV25):
    """v26: + explicit stationary-flag supervision for detected agents.

    A 1x1 conv on the (raw-BEV) detection stem predicts a per-cell
    stationary logit; supervised with BCE at GT box centres where the
    3 s future exists (label = |GT displacement@3s| < 0.5 m). Replaces the
    threshold-on-forecast heuristic for parked/stopped colouring.
    forward -> v25 outputs + (stat [B,1,400,250],)."""
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.stat_head = nn.Conv2d(128, 1, 1)
        nn.init.zeros_(self.stat_head.bias)

    def forward(self, imgs, K, T_cam_ego, v0=None, prev_bev=None,
                warp_theta=None):
        out = super().forward(imgs, K, T_cam_ego, v0, prev_bev, warp_theta)
        return out + (self.stat_head(self._det_feat),)

    @staticmethod
    def stat_loss(stat, boxes, nbox, traj, tvalid, margin=0.0):
        """Signed-margin logistic loss at unambiguous GT vehicle centres.

        ``margin=0`` is exactly BCE-with-logits.  A positive margin requires
        stationary logits above +margin and moving logits below -margin,
        preserving decision threshold zero while widening the separation that
        INT8 quantisation needs.  Creep (0.35--0.8 m at 3 s) is ignored.
        """
        B = boxes.shape[0]
        num = stat.sum() * 0.0
        den = 0
        for b in range(B):
            for k in range(int(nbox[b])):
                if boxes[b, k, 3] <= 0 or tvalid[b, k, 5] < 0.5:
                    continue
                ri = int((80.0 - float(boxes[b, k, 1])) / DET_RES)
                ci = int((50.0 - float(boxes[b, k, 2])) / DET_RES)
                if not (0 <= ri < DET_H and 0 <= ci < DET_W):
                    continue
                d3 = float(traj[b, k, 5].norm())
                if 0.35 < d3 < 0.8:      # ambiguous creep band -> no label
                    continue
                lbl = float(d3 <= 0.35)
                z = stat[b, 0, ri, ci].float().clamp(-15, 15)
                signed = z if lbl > 0.5 else -z
                num = num + F.softplus(float(margin) - signed)
                den += 1
        return num / max(den, 1)


TL_CLASSES = 4                 # none / green / yellow / red


class DepthSegIPMNetV27(DepthSegIPMNetV26):
    """v27: + ego-relevant traffic-light state (whole-image classification).

    Reads the shared s4 features of the two forward cameras (FRONT_WIDE,
    FRONT_NARROW), fuses them with a small conv tower and predicts one
    4-way state: none / green / yellow / red. GT comes from the CoMET TLR
    autolabels reduced to an ego-relevance heuristic (extract_tl.py).
    forward -> v26 outputs + (tl [B,4],)."""
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.tl_head = nn.Sequential(
            nn.Conv2d(320, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),      # 54x96
            ConvBlock(128, 128),
            nn.Conv2d(128, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),      # 27x48
            ConvBlock(128, 128),
            nn.AdaptiveAvgPool2d(1))
        self.tl_fc = nn.Linear(128, TL_CLASSES)

    def forward(self, imgs, K, T_cam_ego, v0=None, prev_bev=None,
                warp_theta=None):
        out = super().forward(imgs, K, T_cam_ego, v0, prev_bev, warp_theta)
        B, N = imgs.shape[:2]
        f = self._f_s4.view(B, N, -1, *self._f_s4.shape[-2:])
        ff = torch.cat([f[:, 0], f[:, 6]], 1)   # FRONT_WIDE + FRONT_NARROW
        return out + (self.tl_fc(self.tl_head(ff).flatten(1)),)

    def tl_loss(self, tl_pred, tl_gt):
        """tl_gt [B] int64 (255 = no GT extracted). Class-weighted CE:
        'none' dominates and red >> yellow among lit frames."""
        if getattr(self, "_tl_w", None) is None                 or self._tl_w.device != tl_pred.device:
            self._tl_w = torch.tensor([0.25, 1.5, 6.0, 1.5],
                                      device=tl_pred.device)
        if (tl_gt != 255).sum() == 0:
            return tl_pred.sum() * 0.0
        return F.cross_entropy(tl_pred.float(), tl_gt.long(),
                               weight=self._tl_w, ignore_index=255)


class DepthSegIPMNetV28(DepthSegIPMNetV27):
    """v28: + near-range area risk map (+-40 x +-25 m @ 0.2 m).

    A small conv head on the FUSED BEV crop (risk encodes motion: lobes
    grow/lead with agent speed, so it needs the temporal feature) predicts
    the potential-field risk GT of bevlane/risk_field.py. Independent head,
    modest weight -> minimal interference with the other tasks.
    forward -> v27 outputs + (risk logits [B,1,400,250],)."""
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.risk_head = nn.Sequential(
            ConvBlock(BEV_CH, 64), ConvBlock(64, 64), nn.Conv2d(64, 1, 1))
        nn.init.constant_(self.risk_head[-1].bias, -2.0)   # start near 0 risk

    def forward(self, imgs, K, T_cam_ego, v0=None, prev_bev=None,
                warp_theta=None):
        out = super().forward(imgs, K, T_cam_ego, v0, prev_bev, warp_theta)
        _r0, _r1 = bev_rows(40.0, -40.0)       # +-40 x +-25 m
        crop = self._fused_bev[:, :, _r0:_r1, 125:375]
        return out + (self.risk_head(crop),)

    @staticmethod
    def risk_loss(pred, gt):
        """pred logits [B,1,400,250]; gt [B,400,250] in [0,1] (-1 = no GT).
        L1 on sigmoid with high-risk emphasis (weight 1 + 4*gt)."""
        m = (gt >= 0).float()
        if m.sum() == 0:
            return pred.sum() * 0.0
        p = pred[:, 0].float().sigmoid()
        g = gt.clamp(min=0)
        w = (1.0 + 4.0 * g) * m
        return (w * (p - g).abs()).sum() / w.sum().clamp(min=1)


EGO_K = 3                      # trajectory hypotheses (WTA-trained)
HIST_N = 3                     # temporal memory slots (0.4 / 1.2 / 2.8 s)
LG_M, LG_P = 24, 12            # lane-graph slots / points per chain


class DepthSegIPMNetV29(DepthSegIPMNetV28):
    """v29 (DESIGN_v29): multimodal trajectories (K=3, winner-takes-all),
    3-slot temporal memory queue, vector lane-graph slot decoder, occupancy
    flow. forward(imgs, K, T, v0, hist_bev [B,3,BEV_CH,800,500],
    hist_theta [B,3,2,3]) -> v28 outputs (with wider ego/traj) +
    (flow [B,2,200,200], lg_pts [B,24,12,2], lg_meta [B,24,4],
     lg_adj [B,24,24])."""
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        # 1. multimodal heads (WTA)
        self.ego_mlp[-1] = nn.Linear(256, 12 * EGO_K + EGO_K + 3)
        self.traj_head = nn.Conv2d(256, TRAJ_H * 2 * EGO_K + EGO_K, 1)
        # 2. temporal memory queue replaces the single-frame tfuse
        del self.tfuse
        self.tfuse3 = nn.Sequential(
            nn.Conv2d(BEV_CH * (1 + HIST_N), BEV_CH, 1, bias=False),
            nn.BatchNorm2d(BEV_CH), nn.ReLU(inplace=True),
            ConvBlock(BEV_CH, BEV_CH))
        last_bn = self.tfuse3[-1][-2]
        nn.init.zeros_(last_bn.weight)
        nn.init.zeros_(last_bn.bias)
        # 3. lane-graph slot decoder on the RAW BEV ROI (x -10..60, |y|<=25)
        self.lg_tower = nn.Sequential(
            nn.Conv2d(BEV_CH, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True), ConvBlock(128, 128),
            nn.Conv2d(128, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True), ConvBlock(128, 128))
        ax = torch.linspace(-4.0, 54.0, 6)     # anchor x (fwd)
        ay = torch.linspace(-18.75, 18.75, 4)  # anchor y (left)
        gx, gy = torch.meshgrid(ax, ay, indexing="ij")
        anchors = torch.stack([gx.reshape(-1), gy.reshape(-1)], 1)  # [24,2]
        self.register_buffer("lg_anchors", anchors)
        # normalized sample coords on the ROI crop (rows x 60..-10, cols y)
        u = (25.0 - anchors[:, 1]) / 50.0 * 2 - 1
        v = (60.0 - anchors[:, 0]) / 70.0 * 2 - 1
        self.register_buffer("lg_grid",
                             torch.stack([u, v], 1).view(1, LG_M, 1, 2))
        self.lg_mlp = nn.Sequential(nn.Linear(130, 256), nn.ReLU(inplace=True),
                                    nn.Linear(256, 256), nn.ReLU(inplace=True))
        self.lg_pts = nn.Linear(256, LG_P * 2)
        # start each slot as a short straight segment on its anchor: the
        # x30 output scale makes default-init offsets +-15 m of noise and
        # the Hungarian matching never converges from there
        nn.init.normal_(self.lg_pts.weight, std=1e-3)
        with torch.no_grad():
            b = torch.zeros(LG_P, 2)
            b[:, 0] = torch.linspace(-4.0, 4.0, LG_P)     # +-4 m along x
            self.lg_pts.bias.copy_((b / 30.0).reshape(-1))
        self.lg_meta = nn.Linear(256, 4)       # exist + 3-class logits
        self.lg_adj = nn.Sequential(nn.Linear(512, 128), nn.ReLU(inplace=True),
                                    nn.Linear(128, 1))
        # 4. occupancy flow
        self.flow_head = nn.Conv2d(192, 2, 1)

    def temporal_fuse(self, bev):
        hb, th = self._prev
        if hb is None:
            cat = [bev] + [torch.zeros_like(bev)] * HIST_N
        else:
            cat = [bev]
            for i in range(HIST_N):
                grid = F.affine_grid(th[:, i].to(bev.dtype), list(bev.shape),
                                     align_corners=False)
                cat.append(F.grid_sample(hb[:, i].to(bev.dtype), grid,
                                         align_corners=False))
        # warped t-0.4s slot, kept for the trajectory head's motion residual
        self._warped0 = cat[1]
        return bev + self.tfuse3(torch.cat(cat, 1))

    def forward(self, imgs, K, T_cam_ego, v0=None, prev_bev=None,
                warp_theta=None):
        out = super().forward(imgs, K, T_cam_ego, v0, prev_bev, warp_theta)
        flow = self.flow_head(self._occ_feat)
        # detached: the lane-graph loss is large early and its gradients
        # through the shared raw BEV wrecked every other task in r20 --
        # the slot decoder learns on frozen features, interference-free
        _r0, _r1 = bev_rows(60.0, -10.0)
        roi = self._last_bev.detach()[:, :, _r0:_r1, 125:375]
        f = self.lg_tower(roi)
        B = f.shape[0]
        emb = F.grid_sample(f, self.lg_grid.expand(B, -1, -1, -1),
                            align_corners=False)[..., 0].transpose(1, 2)
        emb = self.lg_mlp(torch.cat(
            [emb, self.lg_anchors.unsqueeze(0).expand(B, -1, -1) / 30.0], 2))
        pts = self.lg_pts(emb).view(B, LG_M, LG_P, 2) * 30.0             + self.lg_anchors.view(1, LG_M, 1, 2)
        meta = self.lg_meta(emb)
        pair = torch.cat([emb.unsqueeze(2).expand(-1, -1, LG_M, -1),
                          emb.unsqueeze(1).expand(-1, LG_M, -1, -1)], 3)
        adj = self.lg_adj(pair)[..., 0]
        return out + (flow, pts, meta, adj)

    # ---- 1. WTA losses -------------------------------------------------
    # Pure winner-takes-all starves the losing modes: with modes warm-started
    # as near-copies of one trained trajectory, whichever wins first takes
    # every gradient and the others never differentiate. Measured on val at
    # r20 ep3: mode 1 won 80/80 samples, hypotheses 0.74 m apart, minADE only
    # 0.10 m better than top-1 -- i.e. K=3 was dead weight. EPS_WTA keeps the
    # losers alive with a small share of the loss so they can specialise.
    EPS_WTA = 0.1

    def ego_loss(self, ego, gt, intent=None):
        valid = gt[:, 16:17]
        n = valid.sum().clamp(min=1)
        Kn = EGO_K
        wps = ego[:, :12 * Kn].view(-1, Kn, 6, 2)
        err = torch.abs(wps - gt[:, :12].view(-1, 1, 6, 2))
        tw = getattr(self, "EGO_TW", None)
        if tw is not None:                      # v36: near horizons weighted
            err = err * tw.to(err.device).view(1, 1, 6, 1)
        lw = getattr(self, "EGO_LONG_W", 1.0)
        # normalise by the weight sum so changing the long/lat RATIO does
        # not change the loss MAGNITUDE -- otherwise raising lw silently
        # rescales E2E against the other 25 tasks. (lw=2 gives 6/2.4=2.5,
        # exactly the constant this replaced.)
        _n = (lw + 4.0) / 2.4
        wp_ek = (lw * err[..., 0] + 4.0 * err[..., 1]).mean(2) / _n
        best = wp_ek.detach().argmin(1)
        # MODE BINDING. v44 adds MODE_BOOST to the commanded mode's logit and
        # claimed the switch was "guaranteed by construction". It was not:
        # `best` is the argmin of the WAYPOINT error, so the winner is
        # whichever mode already happens to be closest and the command has no
        # say at all. Measured on r45 (50 val turn frames): the three modes
        # differ by 1.08 m and reverse the turn on 0 % of them -- commanding
        # "left" on a right-turn frame still turns right. Routing the winner
        # by the command is what makes mode j accumulate manoeuvre-j
        # gradients. The command is derived from the GT future (build_intent),
        # so command == manoeuvre and this never fights the waypoint target.
        if intent is not None:
            cmd = intent.sum(1) > 0.5                 # rows carrying a command
            best = torch.where(cmd, intent.argmax(1), best)
        e = self.EPS_WTA
        wp_e = ((1.0 - e) * wp_ek.gather(1, best[:, None])
                + e * wp_ek.mean(1, keepdim=True))
        # Endpoint term, on the SELECTED candidate only, added on top of the
        # per-step loss rather than redistributing it. Measured on val with the
        # profile above: the first five steps track the log to 0.047-0.111 m of
        # lateral error while the last is off by 0.538 m, and on turning frames
        # (GT lateral > 2 m) the first 2.5 s are accurate to 0.64 m while the
        # endpoint misses by 3.75 m -- the "path bends only at the tip" the
        # demo shows.
        fde_w = getattr(self, "EGO_FDE_W", 0.0)
        if fde_w:
            f_err = err[:, :, -1]                      # [B,K,2], tw applied
            f_ek = (lw * f_err[..., 0] + 4.0 * f_err[..., 1]) / _n
            wp_e = wp_e + fde_w * f_ek.gather(1, best[:, None])
        mlog = ego[:, 12 * Kn:12 * Kn + Kn]
        # Train the SELECTOR on the RAW logits. forward() already added
        # MODE_BOOST to mlog, so a CE against `best` on the boosted logits
        # teaches the network to cancel the very mechanism it must obey.
        # intent_mode_loss already subtracts the boost; this did not.
        mb = getattr(self, "MODE_BOOST", 0.0)
        if intent is not None and mb:
            mlog = mlog - mb * intent.to(mlog.dtype)
        # SOFT selector target. The hard argmin is a coin flip most of the
        # time: measured on r61 best_e2e over 720 val frames, the gap between
        # the best and second-best candidate has a MEDIAN of 0.231 m, 43.6 % of
        # frames are inside 0.20 m and 96.4 % inside 0.50 m. Teaching cross
        # entropy that one of two near-identical paths is "the" answer is
        # teaching noise, and the selector shows it: it picks the best mode on
        # 37 % of frames against a 33 % chance baseline, and the 0.171 m it
        # loses to the oracle is 28 % of the whole ADE.
        #
        # softmax(-error / tau) says instead "prefer the winner in proportion to
        # how much better it actually is", so a 0.01 m win contributes almost
        # nothing and a 0.5 m win is nearly one-hot. No sample is discarded.
        # A driving COMMAND still overrides completely -- that binding was
        # deliberate (docs/FIX_COMMAND_BINDING.md) and is not what is broken.
        tau = getattr(self, "EGO_CE_TAU", 0.0)
        if tau > 0:
            with torch.no_grad():
                soft = F.softmax(-wp_ek.detach().float() / tau, dim=1)
                if intent is not None:
                    soft = torch.where(cmd[:, None],
                                       intent.to(soft.dtype), soft)
            ce = -(soft * F.log_softmax(mlog.float(), 1)).sum(1)[:, None]
        else:
            ce = F.cross_entropy(mlog, best, reduction="none")[:, None]
        cw = 1.0 + gt[:, 11:12].abs().clamp(max=6.0) / 1.5
        _sw = getattr(self, "EGO_SPEED_W", 0.0)
        if _sw > 0:   # v139b: 高速フレーム (ego 教師の少数派) の重み押し上げ
            cw = cw * (1.0 + gt[:, 12:13].abs() / _sw).clamp(max=4.0)
        nw = (cw * valid).sum().clamp(min=1)
        wl = (wp_e * cw * valid).sum() / nw
        cl = (ce * valid).sum() / n * getattr(self, "EGO_CE_MULT", 1.0)
        # r22: mode-confidence CE weighted up 0.3 -> 0.6 (winner confidence
        # was stuck near 1/3 = undecided while modes specialise)
        o = 12 * Kn + Kn
        sl = (torch.abs(ego[:, o:o + 1] - gt[:, 14:15]) * cw * valid).sum() / nw
        al = (torch.abs(ego[:, o + 1:o + 2] - gt[:, 13:14]) * valid).sum() / n
        p = ego[:, o + 2:o + 3].clamp(-15, 15)
        bl = (F.binary_cross_entropy_with_logits(
            p, gt[:, 15:16], reduction="none") * valid).sum() / n
        return wl + 0.6 * cl + 2.0 * sl + al + 0.5 * bl

    def traj_loss(self, tr_pred, boxes, nbox, traj, tvalid):
        t, m = self.build_traj_targets(boxes, nbox, traj, tvalid,
                                       tr_pred.device)
        if m.sum() == 0:
            return tr_pred.sum() * 0.0
        Kn = EGO_K
        B, _, Hh, Ww = tr_pred.shape
        wps = tr_pred[:, :12 * Kn].view(B, Kn, 12, Hh, Ww).float()
        ml = tr_pred[:, 12 * Kn:].float()                    # [B,K,H,W]
        ek = (torch.abs(wps - t.unsqueeze(1)) * m.unsqueeze(1)).sum(2)
        best = ek.detach().argmin(1, keepdim=True)           # [B,1,H,W]
        cell = m.sum(1) > 0                                  # [B,H,W]
        e = self.EPS_WTA
        ek_w = ((1.0 - e) * ek.gather(1, best).squeeze(1)
                + e * ek.mean(1))
        wl = ek_w[cell].sum() / m.sum()
        ce = F.cross_entropy(ml, best.squeeze(1), reduction="none")
        cl = ce[cell].mean()
        # heading term: L1 keeps magnitudes honest but under-penalises a
        # flipped direction (user-visible on oncoming traffic) -> add
        # 1 - cos between the winning mode's 3 s displacement and GT,
        # weighted by the cell mask (which already carries the oncoming
        # and VRU emphasis), on clearly-moving agents only
        bidx = best.unsqueeze(2).expand(-1, -1, 12, -1, -1)
        wb = wps.gather(1, bidx).squeeze(1)          # [B,12,H,W]
        pd, gd = wb[:, 10:12], t[:, 10:12]
        gn = gd.norm(dim=1)
        # 1.0 m: pedestrians average 2.58 m @3 s — a 2.0 m gate excluded
        # half of them from direction supervision (vruHead stuck ~79 deg)
        mov = (gn > 1.0) & (m[:, 10] > 0)
        if mov.any():
            cos = (pd * gd).sum(1) / (pd.norm(dim=1) * gn + 1e-3)
            dl = ((1.0 - cos) * m[:, 10])[mov].mean()
            wl = wl + 0.5 * dl
        return wl + 0.3 * cl

    # ---- 4. occupancy-flow loss ----------------------------------------
    @staticmethod
    def flow_loss(flow, boxes, nbox, traj, tvalid):
        """target: per-cell ego-frame velocity of the covering agent box
        (traj[0]/0.5 s); stationary boxes supervise (0,0)."""
        import cv2
        import numpy as np
        B = flow.shape[0]
        # The occupancy/flow window is +-40 m at 0.4 m -> 200x200, but a
        # rear-truncated grid gives the head fewer rows (the crop is clamped in
        # bev_rows), so build the target at whatever the prediction actually is.
        FH, FW = int(flow.shape[-2]), int(flow.shape[-1])
        tgt = np.zeros((B, 2, FH, FW), np.float32)
        msk = np.zeros((B, 1, FH, FW), np.float32)
        bx = boxes.detach().cpu().numpy()
        tj = traj.detach().cpu().numpy()
        tv = tvalid.detach().cpu().numpy()
        for b in range(B):
            for k2 in range(int(nbox[b])):
                cls, xe, ye, l, w, yaw = bx[b, k2]
                if l <= 0 or abs(xe) > 42 or abs(ye) > 42:
                    continue
                vx = vy = 0.0
                if tv[b, k2, 0] > 0.5:
                    vx, vy = tj[b, k2, 0] / 0.5
                c, s = np.cos(yaw), np.sin(yaw)
                pts = []
                for lx, wy in ((l / 2, w / 2), (l / 2, -w / 2),
                               (-l / 2, -w / 2), (-l / 2, w / 2)):
                    px = xe + lx * c - wy * s
                    py = ye + lx * s + wy * c
                    pts.append([int((40 - py) / 0.4), int((40 - px) / 0.4)])
                mm = np.zeros((FH, FW), np.uint8)
                cv2.fillPoly(mm, [np.array(pts, np.int32).reshape(-1, 1, 2)], 1)
                tgt[b, 0][mm > 0] = vx
                tgt[b, 1][mm > 0] = vy
                msk[b, 0][mm > 0] = 1
        tgt_t = torch.from_numpy(tgt).to(flow.device)
        msk_t = torch.from_numpy(msk).to(flow.device)
        if msk_t.sum() == 0:
            return flow.sum() * 0.0
        return (torch.abs(flow.float() - tgt_t) * msk_t).sum()             / msk_t.sum().clamp(min=1) / 2

    # ---- 3. lane-graph loss (train-time Hungarian) ---------------------
    @staticmethod
    def lanegraph_loss(pts, meta, adj, gt_pts, gt_cls, gt_n, gt_adj):
        from scipy.optimize import linear_sum_assignment
        B = pts.shape[0]
        total = pts.sum() * 0.0
        nb = 0
        for b in range(B):
            n = int(gt_n[b])
            ex_t = torch.zeros(LG_M, device=pts.device)
            if n == 0:
                total = total + F.binary_cross_entropy_with_logits(
                    meta[b, :, 0].float(), ex_t)
                # graph-preserving zeros: DDP needs every head in the loss
                total = total + (adj[b].sum() + pts[b].sum()
                                 + meta[b, :, 1:].sum()) * 0.0
                nb += 1
                continue
            g = gt_pts[b, :n].float()                        # [n,P,2]
            p = pts[b].float()                               # [M,P,2]
            d1 = (p.unsqueeze(1) - g.unsqueeze(0)).abs().mean((2, 3))
            d2 = (p.unsqueeze(1) - g.flip(1).unsqueeze(0)).abs().mean((2, 3))
            cost = torch.minimum(d1, d2)                     # [M,n]
            # The assignment is done on CPU by scipy, which raises on a NaN or
            # inf instead of returning a bad match -- and that kills the whole
            # round. A non-finite cost means the lane-graph head produced
            # non-finite points this step (fp16 overflow while a warm-started
            # head settles: r51 died here on step 1). Fall back to a
            # gradient-preserving zero for this sample; the caller's non-finite
            # loss guard handles anything worse.
            if not bool(torch.isfinite(cost.detach()).all()):
                total = total + cost.nan_to_num(0.0, 0.0, 0.0).sum() * 0.0
                nb += 1
                continue
            ri, ci = linear_sum_assignment(cost.detach().cpu()
                                           .numpy())
            ri = torch.as_tensor(ri, device=pts.device)
            ci = torch.as_tensor(ci, device=pts.device)
            total = total + cost[ri, ci].mean()
            ex_t[ri] = 1.0
            total = total + F.binary_cross_entropy_with_logits(
                meta[b, :, 0].float(), ex_t)
            total = total + F.cross_entropy(meta[b, ri, 1:].float(),
                                            gt_cls[b][ci].long())
            a_t = gt_adj[b][ci][:, ci].float()
            a_p = adj[b][ri][:, ri].float()
            total = total + 0.5 * F.binary_cross_entropy_with_logits(a_p, a_t)
            nb += 1
        return total / max(nb, 1)


class DepthSegIPMNetV30(DepthSegIPMNetV29):
    """v30: + unknown-object (cone/pole/debris) detection, fixed size.

    Separate 1ch centre heatmap on the (raw-BEV) detection stem; GT centres
    come from small occupancy-obstacle blobs (extract_unknown.py) — the
    annotation set has no unknown boxes, so size is fixed at decode
    (0.4x0.4 m). forward -> v29 outputs + (hm_unk [B,1,400,250],)."""
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.unk_head = nn.Conv2d(128, 1, 1)
        nn.init.constant_(self.unk_head.bias, -2.19)
        # motion-residual input for the trajectory head: current BEV minus
        # the ego-warped t-0.4s slot. Oncoming cars read as a signed dipole
        # along their true motion; without it the only direction cue is the
        # fused-BEV smear, and predictions collapsed to the ego-forward
        # majority prior (measured: oncoming heading flips, worst at launch).
        self.traj_stem = nn.Sequential(
            nn.Conv2d(2 * BEV_CH, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True), ConvBlock(128, 128))

    def traj_feat(self):
        # zero residual when the slot is missing (all-zero warped feature),
        # otherwise "everything just appeared" reads as fake motion
        _w0 = self._warped0[:, :16] if _EXPORT_FAST else self._warped0
        valid = (_w0.abs().sum(1, keepdim=True) > 0).to(
            self._last_bev.dtype)
        mot = (self._last_bev - self._warped0) * valid
        self._tf = torch.cat(
            [self.traj_stem(torch.cat([self._fused_bev, mot], 1)),
             self._det_feat.detach()], 1)
        return self._tf

    def forward(self, imgs, K, T_cam_ego, v0=None, prev_bev=None,
                warp_theta=None):
        out = super().forward(imgs, K, T_cam_ego, v0, prev_bev, warp_theta)
        return out + (self.unk_head(self._det_feat),)

    @staticmethod
    def decode_unknown(hm_unk, thresh=0.25, topk=64):
        """-> per-batch list of (cls=2, score, xe, ye, 0.4, 0.4, 0.0):
        unknown objects join the BEV 3D box stream as a third class with a
        fixed footprint (no size GT exists for them)."""
        p = hm_unk.sigmoid()
        pmax = F.max_pool2d(p, 3, 1, 1)
        p = p * (pmax == p)
        B, _, Hh, Ww = p.shape
        out = []
        for bi in range(B):
            flat = p[bi, 0].reshape(-1)
            sc, idx = flat.topk(min(topk, flat.numel()))
            keep = sc > thresh
            boxes = []
            for s_, i_ in zip(sc[keep].tolist(), idx[keep].tolist()):
                ri, ci = divmod(i_, Ww)
                boxes.append((2, s_, 80.0 - ri * DET_RES,
                              50.0 - ci * DET_RES, 0.4, 0.4, 0.0))
            out.append(boxes)
        return out

    @staticmethod
    def unk_loss(hm, centers, n):
        """penalty-reduced focal on 1ch heatmap; Gaussian radius 1.5 cells.

        n is PACKED (dataset): low byte = positives, high byte = occluded
        blobs stored after them in `centers` — those get an ignore disk
        (no negative loss) instead of counting as background."""
        B = hm.shape[0]
        dev = hm.device
        hm_t = torch.zeros(B, 1, DET_H, DET_W, device=dev)
        ign = torch.zeros(B, 1, DET_H, DET_W, device=dev)
        ys = torch.arange(DET_H, device=dev, dtype=torch.float32)
        xs = torch.arange(DET_W, device=dev, dtype=torch.float32)
        npos = 0
        for b in range(B):
            nv = int(n[b]) & 0xFF
            ni = (int(n[b]) >> 8) & 0xFF
            for k in range(nv + ni):
                xe, ye = float(centers[b, k, 0]), float(centers[b, k, 1])
                r = (BEV_XF - xe) / DET_RES
                c = (BEV_YH - ye) / DET_RES
                if not (0 <= r < DET_H and 0 <= c < DET_W):
                    continue
                if k >= nv:                     # occluded -> ignore disk
                    d2 = ((ys - r) ** 2).view(-1, 1) \
                        + ((xs - c) ** 2).view(1, -1)
                    ign[b, 0] = torch.maximum(ign[b, 0],
                                              (d2 < 3.0 ** 2).float())
                    continue
                g = torch.exp(-(((ys - r) ** 2).view(-1, 1)
                                + ((xs - c) ** 2).view(1, -1)) / (2 * 2.0 ** 2))
                hm_t[b, 0] = torch.maximum(hm_t[b, 0], g)
                npos += 1
        p = hm.float().sigmoid().clamp(1e-4, 1 - 1e-4)
        pos = (hm_t > 0.99).float()
        neg_w = (1 - hm_t) ** 4
        # KMAX-saturated frames have unlabeled real positives beyond the
        # cap; punishing their cells as negatives suppresses the whole
        # head's scores (measured: max score 0.21 after 8 epochs). Ignore
        # negatives on saturated frames — positives still supervise.
        sat = ((n & 0xFF).float() >= 64).view(-1, 1, 1, 1).to(p.dtype)
        # sparse-positive rebalance: ~2 positives vs 100k cells after the
        # v3 GT cleanup; unscaled negatives suppress the whole head
        loss = -(pos * (1 - p) ** 2 * p.log()
                 + 0.25 * (1 - pos) * (1 - ign) * (1 - sat) * neg_w
                 * p ** 2 * (1 - p).log()).sum()
        return loss / max(npos, 1)


class DepthSegIPMNetV31(DepthSegIPMNetV30):
    """v31: optional LiDAR depth input (roadmap C6a) — one set of weights
    serves camera-only AND LiDAR-assisted inference.

    Input: LiDAR points projected to the 6 IPM cameras as a sparse metric
    depth map [B,6,hd,wd] (0 = no return) — the exact format of the depth4
    GT, so training reuses the batch's depth tensor as the input. Where a
    return exists, the predicted depth softmax is blended toward the
    measured bin (triangular two-bin interpolation, elementwise only —
    TRT-safe, no scatter):

        dprob' = (1 - a*m) * dprob + a*m * tri(d)      a = sigmoid(w_a)

    Feeding zeros makes m == 0 everywhere -> bit-equal to the camera-only
    network: one ONNX/engine for both modes. Train with modality dropout
    (--lidar-drop) so BN statistics stay calibrated for both. History
    slots always run camera-only (self._lidar is cleared after forward),
    matching a runtime whose memory ring stores raw camera BEVs."""
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.lid_alpha = nn.Parameter(torch.tensor(1.5))  # sigmoid -> 0.82
        self._lidar = None

    def sharpen_dprob(self, dprob):
        if self._lidar is None:
            return dprob
        BN, D, fh, fw = dprob.shape
        d = self._lidar.reshape(BN, 1, *self._lidar.shape[-2:]).to(
            dprob.dtype)
        if d.shape[-2:] != (fh, fw):
            d = F.interpolate(d, (fh, fw), mode="nearest")
        _c = self._dbins(dprob.device, dprob.dtype)
        m = ((d > _c[0]) & (d < _c[-2])).to(dprob.dtype)
        bins = _c.view(1, D, 1, 1)
        _w = torch.gradient(_c)[0].clamp(min=1e-3).view(1, D, 1, 1)
        tri = (1.0 - (d - bins).abs() / _w).clamp(min=0)
        a = torch.sigmoid(self.lid_alpha) * m
        return (1.0 - a) * dprob + a * tri

    def forward(self, imgs, K, T_cam_ego, v0=None, prev_bev=None,
                warp_theta=None, lidar=None):
        self._lidar = lidar
        try:
            return super().forward(imgs, K, T_cam_ego, v0, prev_bev,
                                   warp_theta)
        finally:
            self._lidar = None          # history compute_bev stays cam-only


class DepthSegIPMNetV32(DepthSegIPMNetV31):
    """v32: + optional LiDAR pillar branch (roadmap C6b), switchable at
    inference on the same weights.

    Input: a host-side BEV raster of the current LiDAR sweep
    [B,4,400,250] at 0.4 m (log-count, max z, mean z, occupancy — see
    extract_lidar_bev.py; the deployment runtime computes the identical
    raster from the raw pcd). A small conv stem turns it into a 96-ch
    residual added to the raw BEV *before* every head and the temporal
    queue:

        bev' = bev + flag * up2(stem(raster))

    flag is derived from the raster itself (any nonzero cell), so feeding
    zeros makes the residual EXACTLY zero -> bit-equal to the camera-only
    network: ON/OFF is decided per frame by what you feed, one engine, one
    checkpoint. The stem's last conv is zero-initialised, so warm starts
    from a non-LiDAR checkpoint are behaviour-preserving; train with the
    same modality dropout as v31 (the C6a depth input and this raster are
    dropped together)."""
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.lidar_stem = nn.Sequential(
            nn.Conv2d(4, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, BEV_CH, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(BEV_CH, BEV_CH, 1))
        nn.init.zeros_(self.lidar_stem[-1].weight)
        nn.init.zeros_(self.lidar_stem[-1].bias)
        self._lidar_bev = None

    def bev_extra(self, bev):
        if self._lidar_bev is None:
            return bev
        lb = self._lidar_bev.to(bev.dtype)
        flag = (lb.abs().sum((1, 2, 3), keepdim=True) > 0).to(bev.dtype)
        res = F.interpolate(self.lidar_stem(lb), bev.shape[-2:],
                            mode="bilinear", align_corners=False)
        return bev + flag * res

    def forward(self, imgs, K, T_cam_ego, v0=None, prev_bev=None,
                warp_theta=None, lidar=None, lidar_bev=None):
        self._lidar_bev = lidar_bev
        try:
            return super().forward(imgs, K, T_cam_ego, v0, prev_bev,
                                   warp_theta, lidar=lidar)
        finally:
            self._lidar_bev = None      # history compute_bev stays cam-only


class DepthSegIPMNetV33(DepthSegIPMNetV32):
    """v33: precision pass on user-visible failures.

    1. Stationary flag reads the TEMPORAL trajectory feature (motion
       residual included) instead of the single-frame detection stem — one
       frame cannot tell parked from stopped from creeping, which capped
       statAcc at ~0.68. Graft: old 128-ch weights map onto the det-feat
       half of the 256-ch input, zeros elsewhere -> behaviour-preserving.
    2. Trajectory head additionally sees the detached detection yaw
       (sin/cos of the reg head) — a direct orientation prior for oncoming
       traffic instead of re-deriving it from BEV smears.
    (Occupancy near-FP suppression and crossing-yaw weighting live in the
    shared losses/GT; see occ_loss, build_det_targets, filter v2.)"""
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        # separate module: the inherited forward still runs the old
        # 128-ch stat_head on the det stem; out[10] is then replaced.
        # Freeze the old head — its output never reaches a loss, and DDP
        # refuses parameters that produce no gradient.
        for p_ in self.stat_head.parameters():
            p_.requires_grad_(False)
        self.stat_head2 = nn.Conv2d(256, 1, 1)
        nn.init.zeros_(self.stat_head2.bias)
        self.traj_head = nn.Conv2d(258, TRAJ_H * 2 * EGO_K + EGO_K, 1)

    def traj_feat(self):
        base = super().traj_feat()                    # caches self._tf too
        self._tf = torch.cat([base, self._det_reg[:, 4:6].detach()], 1)
        return self._tf

    def forward(self, imgs, K, T_cam_ego, v0=None, prev_bev=None,
                warp_theta=None, lidar=None, lidar_bev=None):
        out = super().forward(imgs, K, T_cam_ego, v0, prev_bev, warp_theta,
                              lidar=lidar, lidar_bev=lidar_bev)
        # stationary flag (out[10]) recomputed on the temporal feature;
        # _tf excludes the yaw channels appended for the traj head
        out = list(out)
        out[10] = self.stat_head2(self._tf[:, :256])
        return tuple(out)


class DepthSegIPMNetV34(DepthSegIPMNetV33):
    """v34: unknown detection reworked (roadmap: fundamental fix).

    The 1x1 head on the single-frame det stem could not accumulate
    evidence for 0.4 m objects; the new head runs a small conv stem on the
    TEMPORAL trajectory feature (motion residual + det feature, 256 ch) —
    static cones integrate over the 2.8 s memory. Old head frozen (DDP).
    Pair with radius-2.0 positives and --unk-w 1.0."""
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        for p_ in self.unk_head.parameters():
            p_.requires_grad_(False)
        self.unk_stem = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=1), nn.ReLU(inplace=True),
            ConvBlock(64, 64))
        self.unk_head2 = nn.Conv2d(64, 1, 1)
        nn.init.constant_(self.unk_head2.bias, -2.19)

    def forward(self, imgs, K, T_cam_ego, v0=None, prev_bev=None,
                warp_theta=None, lidar=None, lidar_bev=None):
        out = list(super().forward(imgs, K, T_cam_ego, v0, prev_bev,
                                   warp_theta, lidar=lidar,
                                   lidar_bev=lidar_bev))
        out[17] = self.unk_head2(self.unk_stem(self._tf[:, :256]))
        return tuple(out)


class DepthSegIPMNetV35(DepthSegIPMNetV34):
    """v35: the TRT-safe transformer quartet (roadmap B1-B4).

    B1 lane graph: 24 learned queries, 2 decoder layers (self-attn +
       cross-attn to 352 pooled BEV-ROI tokens) replace the anchored MLP.
    B2 temporal: per-cell softmax gate over [cur|t-.4|t-1.2|t-2.8];
       zero-init -> uniform -> exactly today's fusion at start.
    B3 E2E: K=3 queries attend to the pooled BEV grid; zero-init residual
       added to the ego output.
    B4 agents (lite): scene-level interaction token from det features,
       zero-init residual into the trajectory stem output.
    All dense attention with static token counts: MatMul/Softmax only."""
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        # old lane-graph decoder is replaced -> freeze (DDP refuses
        # parameters that never receive gradient; lg_adj is reused live)
        for mod_ in (self.lg_tower, self.lg_mlp, self.lg_pts, self.lg_meta):
            for p_ in mod_.parameters():
                p_.requires_grad_(False)
        # B2
        self.tgate = nn.Conv2d(BEV_CH * 4, 4, 1)
        nn.init.zeros_(self.tgate.weight); nn.init.zeros_(self.tgate.bias)
        # B1
        self.lgq = nn.Embedding(LG_M, 256)
        dl = nn.TransformerDecoderLayer(256, 4, 512, batch_first=True,
                                        dropout=0.0)
        self.lgdec = nn.TransformerDecoder(dl, 2)
        self.lg_in = nn.Conv2d(BEV_CH, 256, 1)
        self.lg_pts2 = nn.Linear(256, LG_P * 2)
        nn.init.normal_(self.lg_pts2.weight, std=1e-3)
        with torch.no_grad():
            b = torch.zeros(LG_P, 2)
            b[:, 0] = torch.linspace(-4.0, 4.0, LG_P)
            self.lg_pts2.bias.copy_((b / 30.0).reshape(-1))
        self.lg_meta2 = nn.Linear(256, 4)
        # B3
        self.ego_q = nn.Embedding(3, BEV_CH)
        self.ego_attn = nn.MultiheadAttention(BEV_CH, 4, batch_first=True)
        self.ego_delta = nn.Linear(3 * BEV_CH, 12 * EGO_K + EGO_K + 3)
        nn.init.zeros_(self.ego_delta.weight); nn.init.zeros_(self.ego_delta.bias)
        # B4 lite
        self.agent_q = nn.Embedding(4, 128)
        self.agent_attn = nn.MultiheadAttention(128, 4, batch_first=True)
        self.agent_delta = nn.Conv2d(128, 256, 1)
        nn.init.zeros_(self.agent_delta.weight); nn.init.zeros_(self.agent_delta.bias)

    def temporal_fuse(self, bev):                      # B2
        hb, th = self._prev
        cat = [bev]
        if hb is None:
            cat += [torch.zeros_like(bev)] * HIST_N
        else:
            for i in range(HIST_N):
                grid = F.affine_grid(th[:, i].to(bev.dtype), list(bev.shape),
                                     align_corners=False)
                cat.append(F.grid_sample(hb[:, i].to(bev.dtype), grid,
                                         align_corners=False))
        self._warped0 = cat[1]
        self._prefuse_bev = bev                        # delta-stat が読む
        g = self.tgate(torch.cat(cat, 1)).softmax(1)   # [B,4,H,W]
        cat = [c * (4.0 * g[:, i:i + 1]) for i, c in enumerate(cat)]
        return bev + self.tfuse3(torch.cat(cat, 1))

    def forward(self, imgs, K, T_cam_ego, v0=None, prev_bev=None,
                warp_theta=None, lidar=None, lidar_bev=None):
        out = list(super().forward(imgs, K, T_cam_ego, v0, prev_bev,
                                   warp_theta, lidar=lidar,
                                   lidar_bev=lidar_bev))
        B = out[0].shape[0]
        # B3: ego residual from attention pooling over the fused BEV
        tok = F.adaptive_avg_pool2d(self._fused_bev, (25, 16))             .flatten(2).transpose(1, 2)                # [B,400,96]
        qa, _ = self.ego_attn(self.ego_q.weight.unsqueeze(0).expand(B, -1, -1),
                              tok, tok)
        out[7] = out[7] + self.ego_delta(qa.flatten(1))
        if getattr(self, "sem_ego", None) is not None:
            # 意味出力 (INT8 健全) から ego 残差。detach で painter 側と同じく
            # 意味ヘッドの学習を汚さない。
            _sem = torch.cat([
                F.avg_pool2d(out[0].detach().float().softmax(1), 2),  # seg 9ch @400x250
                out[3].detach().float().sigmoid()], 1)                # hm 2ch @400x250
            out[7] = out[7] + self.sem_ego(_sem.to(out[7].dtype))
        if getattr(self, "kin_gate", None) is not None and v0 is not None:
            # v139 運動学アンカー (2026-09-02): 全 K モードの waypoint に
            # g_t * [v0*t, 0] を加算。g_t (6 個) はゼロ初期化 = 導入時関数保存。
            # 根拠: 0.5s 目 waypoint の縦バイアスが v0 に比例 (8-15 m/s で −10%)。
            _t = torch.arange(1, 7, device=out[7].device, dtype=torch.float32) * 0.5
            _cv = (v0.view(-1, 1).float() * (_t * self.kin_gate.float()).view(1, 6))  # [B,6]
            _e = out[7].float()
            _wp = _e[:, :12 * EGO_K].view(-1, EGO_K, 6, 2).clone()
            _wp[..., 0] = _wp[..., 0] + _cv.view(-1, 1, 6)
            out[7] = torch.cat([_wp.view(-1, 12 * EGO_K), _e[:, 12 * EGO_K:]], 1).to(out[7].dtype)
        # B4 lite: scene interaction token -> traj/stat features rerun
        dt = F.adaptive_avg_pool2d(self._det_feat.detach(), (25, 16))             .flatten(2).transpose(1, 2)                # [B,400,128]
        ag, _ = self.agent_attn(self.agent_q.weight.unsqueeze(0)
                                .expand(B, -1, -1), dt, dt)
        ctx = self.agent_delta(ag.mean(1)[:, :, None, None])
        tf = self._tf[:, :256] + ctx
        tfq = tf
        if getattr(self, "traj_flow", None) is not None:
            # A1: flow 場 (±40m クロップ, stride2) を traj 格子へ再配置し、
            # ゼロ初期化 1x1 の残差として traj 入力にのみ加算 (stat は不変)。
            _fl = out[13].detach().to(tf.dtype)
            _cv = tf.new_zeros(tf.shape[0], 2, tf.shape[2], tf.shape[3])
            _fr0, _ = bev_rows(40.0, -40.0)
            _cv[:, :, _fr0 // 2:_fr0 // 2 + _fl.shape[2],
                25:25 + _fl.shape[3]] = _fl
            tfq = tf + self.traj_flow(_cv)
        out[9] = self.traj_head(torch.cat(
            [tfq, self._det_reg[:, 4:6].detach()], 1))
        if getattr(self, "traj_vel", None) is not None:
            # A6: CV 再パラメータ化 — 各モードの waypoint に v̂*t を加算
            _v = self.traj_vel(tfq)                       # [B,2,H,W]
            _o9 = out[9]
            _t = torch.arange(1, 7, device=_v.device,
                              dtype=_v.dtype) * 0.5      # 0.5..3.0s
            # traj GT [6,2] の (dx_i, dy_i) 交互並びに合わせて 12ch を構成
            _cvk = torch.stack([_v[:, 0] * s for s in _t]
                               + [_v[:, 1] * s for s in _t], 1)
            _idx = [i for p_ in range(6) for i in (p_, 6 + p_)]
            _cvk = _cvk[:, _idx]                          # [B,12,H,W] (x,y)交互
            for _k in range(EGO_K):
                _o9 = torch.cat([_o9[:, :_k * 12],
                                 _o9[:, _k * 12:(_k + 1) * 12] + _cvk,
                                 _o9[:, (_k + 1) * 12:]], 1)
            out[9] = _o9
        if getattr(self, "det_tmp", None) is not None:
            # D7: hm へ時間特徴残差 (ゼロ初期化)
            out[3] = out[3] + self.det_tmp(tf)
        if getattr(self, "mode_scorer", None) is not None:
            # E3: 各候補経路に沿った特徴でモード logit を補正
            _p7 = out[7]
            _B = _p7.shape[0]
            _wk = _p7[:, :12 * EGO_K].detach().view(_B, EGO_K, 6, 2)
            # (x 前方, y 左) [m] -> det 格子 (row = (80-x)/0.4, col = (50-y)/0.4)
            _H2, _W2 = tfq.shape[-2:]
            _gr = (80.0 - _wk[..., 0]) / 0.4 / max(_H2 - 1, 1) * 2 - 1
            _gc = (50.0 - _wk[..., 1]) / 0.4 / max(_W2 - 1, 1) * 2 - 1
            _grid = torch.stack([_gc, _gr], -1).view(_B, EGO_K * 6, 1, 2)
            _sam = F.grid_sample(tfq.detach(), _grid.to(tfq.dtype),
                                 align_corners=True)      # [B,256,K*6,1]
            _sam = _sam.view(_B, 256, EGO_K, 6).mean(-1)  # [B,256,K]
            _sc = self.mode_scorer(_sam.permute(0, 2, 1)).squeeze(-1)  # [B,K]
            _o7 = _p7.clone()
            _o7[:, 12 * EGO_K:12 * EGO_K + EGO_K] = \
                _p7[:, 12 * EGO_K:12 * EGO_K + EGO_K] + _sc
            out[7] = _o7
        out[10] = self.stat_head2(tf)
        if getattr(self, "delta_stat", None) is not None:
            # 時間差分ヘッド (2026-08-25): 停止判定を |bev - warp(prev_bev)|
            # から出す。動きの信号がテンソルのレンジそのものになるため、
            # INT8 の目盛りが動き信号で決まり「大きな DC に乗った微小 AC が
            # 量子化で消える」構造 (stat_head2 が 5 段の対処すべてで死んだ
            # 根本原因) が原理的に消える。
            _d = (self._prefuse_bev - self._warped0).abs()
            out[10] = self.delta_stat(_d)
        # B1: query-decoder lane graph replaces out[14..16]
        _r0, _r1 = bev_rows(60.0, -10.0)
        roi = self._last_bev.detach()[:, :, _r0:_r1, 125:375]
        mem = F.adaptive_avg_pool2d(self.lg_in(roi), (22, 16))             .flatten(2).transpose(1, 2)                # [B,352,256]
        emb = self.lgdec(self.lgq.weight.unsqueeze(0).expand(B, -1, -1), mem)
        out[14] = self.lg_pts2(emb).view(B, LG_M, LG_P, 2) * 30.0             + self.lg_anchors.view(1, LG_M, 1, 2)
        out[15] = self.lg_meta2(emb)
        pair = torch.cat([emb.unsqueeze(2).expand(-1, -1, LG_M, -1),
                          emb.unsqueeze(1).expand(-1, LG_M, -1, -1)], 3)
        out[16] = self.lg_adj(pair)[..., 0]
        return tuple(out)


class DepthSegIPMNetV36(DepthSegIPMNetV35):
    """v36 (ADE round, r30): the planner finally learns longitudinal state.

    1. Kinematic-history input: per-slot ego displacements/yaw-rates from
       the memory-queue rel poses (already in the batch) + v0 -> 7-dim
       feature -> zero-init residual on the ego output. Without it the
       planner guessed accel/brake state (measured: 0.43 m longitudinal
       vs 0.19 m lateral error).
    2. Mode-selection pressure: EGO_CE_MULT 2x (best-of-3 is 0.52 m but
       the picked mode is 0.76 m -> selection gap 0.24 m).
    3. Time-weighted waypoints: near horizons x1.5 -> ADE-aligned."""
    EGO_CE_MULT = 2.0
    # Reverted. Flattening this to all-ones (r56) made E2E worse on every
    # measure -- ADE 1.162 -> 1.367 m, ADEc 1.542 -> 1.819 m, FDE 2.271 ->
    # 2.577 m on the 240-sample slice, and monotonically worse epoch over epoch
    # in training val (0.78 / 0.89 / 1.10). The observation that drove it was
    # right (the +3.0 s point carries the largest error and had the smallest
    # weight) but the remedy was not: with an absolute-error loss the far point
    # already owns the largest gradient share by having the largest error, so
    # flattening let it crowd out the near waypoints that actually steer. This
    # profile is what keeps that balance. The endpoint gets its own ADDITIVE
    # term instead (--ego-fde-w), which adds supervision without taking any
    # away.
    EGO_TW = torch.tensor([1.5, 1.35, 1.2, 1.05, 0.95, 0.9])

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.kin_delta = nn.Linear(7, 12 * EGO_K + EGO_K + 3)
        nn.init.zeros_(self.kin_delta.weight)
        nn.init.zeros_(self.kin_delta.bias)

    def forward(self, imgs, K, T_cam_ego, v0=None, prev_bev=None,
                warp_theta=None, lidar=None, lidar_bev=None, kin=None):
        out = list(super().forward(imgs, K, T_cam_ego, v0, prev_bev,
                                   warp_theta, lidar=lidar,
                                   lidar_bev=lidar_bev))
        B = out[0].shape[0]
        f = torch.zeros(B, 7, device=out[0].device, dtype=out[7].dtype)
        if v0 is not None:
            f[:, 0] = v0.view(-1).to(f.dtype) / 15.0
        if kin is not None:                     # [B,3,3] rel (tx,ty,dyaw)
            dts = torch.tensor([0.4, 1.2, 2.8], device=f.device,
                               dtype=f.dtype)
            f[:, 1:4] = kin[..., :2].norm(dim=2).to(f.dtype) / dts / 15.0
            f[:, 4:7] = kin[..., 2].to(f.dtype) / dts
        out[7] = out[7] + self.kin_delta(f)
        return tuple(out)


class DepthSegIPMNetV37(DepthSegIPMNetV36):
    """v37 (roadmap E6): route-intent conditioning for the planner.

    intent [B,3] one-hot (straight / left / right); zero vector = no
    navigation available (trained with 30% intent dropout so the
    unconditioned mode stays strong). Zero-init delta on the ego output:
    the intent biases BOTH the waypoints and the K=3 mode logits — the
    measured 0.24 m selection gap is mostly wrong-branch picks at
    intersections, which navigation resolves for free at runtime."""
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.intent_delta = nn.Linear(3, 12 * EGO_K + EGO_K + 3)
        nn.init.zeros_(self.intent_delta.weight)
        nn.init.zeros_(self.intent_delta.bias)

    def forward(self, imgs, K, T_cam_ego, v0=None, prev_bev=None,
                warp_theta=None, lidar=None, lidar_bev=None, kin=None,
                intent=None):
        out = list(super().forward(imgs, K, T_cam_ego, v0, prev_bev,
                                   warp_theta, lidar=lidar,
                                   lidar_bev=lidar_bev, kin=kin))
        if intent is not None:
            out[7] = out[7] + self.intent_delta(intent.to(out[7].dtype))
        return tuple(out)


class DepthSegIPMNetV38(DepthSegIPMNetV37):
    """v38 (ADE P2+P3): risk-aware mode selection + longitudinal focus.

    P2: each of the K=3 hypotheses integrates the model's own risk field
    along its path; a zero-init learnable gate feeds -risk into the mode
    logits, so the selector learns confidence x safety jointly (C1 moved
    into training).
    P3: longitudinal waypoint error weight 1.0 -> 2.0 (measured error is
    0.43 m longitudinal vs 0.19 m lateral) and an auxiliary speed-profile
    head (per-horizon speed regression on the pooled BEV) shapes the
    representation the planner reads."""
    # 6.0, was 2.0. Measured on 120 val frames: 92 % of ADE is longitudinal
    # (1.059 m of 1.153) and nearly all of that is a one-sided bias -- the
    # predicted path is short by 0.127 / 0.311 / 0.546 / 0.737 / 0.895 /
    # 1.152 m at +0.5 .. +3.0 s, while lateral error is only 0.228 m and has no
    # such drift. The loss weighted lateral 4.0 against longitudinal 2.0, which
    # is right about which error is dangerous and wrong about which one is
    # actually happening: it let a systematic under-travel accumulate unpunished.
    EGO_LONG_W = 6.0

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.risk_gate = nn.Parameter(torch.zeros(1))
        self.vprof_head = nn.Linear(BEV_CH, 6)
        nn.init.zeros_(self.vprof_head.weight)
        nn.init.zeros_(self.vprof_head.bias)

    def forward(self, imgs, K, T_cam_ego, v0=None, prev_bev=None,
                warp_theta=None, lidar=None, lidar_bev=None, kin=None,
                intent=None):
        out = list(super().forward(imgs, K, T_cam_ego, v0, prev_bev,
                                   warp_theta, lidar=lidar,
                                   lidar_bev=lidar_bev, kin=kin,
                                   intent=intent))
        e = out[7]
        B = e.shape[0]
        # P2: per-mode risk line integral (grid_sample on own risk head)
        risk = out[12].float().sigmoid()               # [B,1,400,250]
        wps = e[:, :36].view(B, 3, 6, 2).detach()      # coords only
        gx = (25.0 - wps[..., 1]) / 25.0 - 0.0         # y -> [-1,1] approx
        gx = -wps[..., 1] / 25.0
        gy = (40.0 - wps[..., 0]) / 40.0 - 1.0         # x 0..80 -> [-1,1]
        grid = torch.stack([gx, gy], -1).view(B, 3, 6, 2)
        rs = F.grid_sample(risk, grid, align_corners=False,
                           padding_mode="border")      # [B,1,3,6]
        rint = rs.mean(3).squeeze(1)                   # [B,3]
        e = e.clone()
        e[:, 36:39] = e[:, 36:39] - self.risk_gate * rint.to(e.dtype)
        out[7] = e
        # P3: auxiliary speed profile (read by vprof_loss)
        g = F.adaptive_avg_pool2d(self._fused_bev, 1).flatten(1)
        self._vprof = self.vprof_head(g.float())
        return tuple(out)

    def vprof_loss(self, ego_gt):
        """per-horizon speed regression vs GT waypoint arc steps."""
        wp = ego_gt[:, :12].view(-1, 6, 2)
        steps = torch.cat([wp[:, :1], wp[:, 1:] - wp[:, :-1]], 1)
        v_gt = steps.norm(dim=2) / 0.5                 # m/s per horizon
        valid = ego_gt[:, 16:17]
        return (torch.abs(self._vprof - v_gt) * valid).sum() \
            / valid.sum().clamp(min=1) / 6.0


class DepthSegIPMNetV39(DepthSegIPMNetV38):
    """v39: truly DECOUPLED E2E head (shape x speed composition).

    A parallel decoder predicts, per mode, a heading profile phi(t) [3,6]
    and a speed profile v(t) [3,6]; waypoints are composed by cumulative
    integration wp_i = wp_{i-1} + 0.5 * v_i * (cos phi_i, sin phi_i)
    (cumsum only -> TRT-safe). A zero-init gate blends it with the
    original entangled regression, so longitudinal accuracy can be
    optimised independently of path shape without losing the r32
    behaviour at start."""
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.dec_head = nn.Linear(BEV_CH, 36)  # [3 modes x (6 phi + 6 v)]
        nn.init.zeros_(self.dec_head.weight)
        with torch.no_grad():
            b = torch.zeros(36)
            b[18:] = 5.0                       # v init ~5 m/s
            self.dec_head.bias.copy_(b)
        self.dec_gate = nn.Parameter(torch.zeros(1))

    def forward(self, imgs, K, T_cam_ego, v0=None, prev_bev=None,
                warp_theta=None, lidar=None, lidar_bev=None, kin=None,
                intent=None):
        out = list(super().forward(imgs, K, T_cam_ego, v0, prev_bev,
                                   warp_theta, lidar=lidar,
                                   lidar_bev=lidar_bev, kin=kin,
                                   intent=intent))
        e = out[7]
        B = e.shape[0]
        g = F.adaptive_avg_pool2d(self._fused_bev, 1).flatten(1).float()
        d = self.dec_head(g).view(B, 3, 12)
        phi = d[:, :, :6]
        # fp16-safe softplus: log(1+exp(x)) overflows fp16 at x>~11 (a
        # 15 m/s speed logit), which made the TRT fp16 engine emit NaN
        # waypoints at ~55 km/h. exp(-|x|)<=1 never overflows; identical
        # values. Speeds capped at 25 m/s (90 km/h): physically sane and
        # exact identity below the cap.
        xv = d[:, :, 6:]
        v = (xv.clamp(min=0)
             + torch.log1p(torch.exp(-xv.abs()))).clamp(max=25.0)
        step = 0.5 * v
        dx = step * torch.cos(phi)
        dy = step * torch.sin(phi)
        wp_dec = torch.stack([dx.cumsum(2), dy.cumsum(2)], -1)  # [B,3,6,2]
        wps = e[:, :36].view(B, 3, 6, 2)
        gate = torch.tanh(self.dec_gate)
        e = e.clone()
        e[:, :36] = (wps + gate * (wp_dec.to(e.dtype) - wps)).reshape(B, 36)
        out[7] = e
        return tuple(out)


class BEVSegRefiner(nn.Module):
    """Post-hoc BEV-seg sharpener / far-range completer (roadmap 3f, r33).

    The single-frame lane head drops out past ~50 m: the measured failure is
    not blur but *missing* prediction (recall collapse), because the wide
    cameras give <2 px per 0.2 m cell there. This module takes the FROZEN
    main model's BEV-seg logits and predicts a residual correction. Its U-Net
    has a wide receptive field (down to s16 = 3.2 m/cell) so it can propagate
    the confident near-range structure forward into the far field, using the
    learned prior that lanes / edges / crosswalks are spatially continuous.
    The far-range GT it is trained against is trustworthy: gt_cons is built by
    accumulating each place over the whole drive, so 50-80 m is fully labelled
    even though a single frame can't see it.

    Design guarantees:
      * residual + zero-init last conv  -> identity at init, so it can only
        add to the frozen output; near-range accuracy is preserved by
        construction (the "never degrade Seg/Det/E2E" priority).
      * input is the 9-channel logit map (+ a forward-range channel), so it is
        a standalone net: deployable as its own ONNX/TRT engine chained after
        the main graph, and trainable without touching the frozen backbone.
      * optional raw-BEV context (ctx_ch>0) lets it also read the weak far
        evidence the seg head discarded; off by default for deployability.
    """

    def __init__(self, n_cls=N_CLASSES, ctx_ch=0, width=48):
        super().__init__()
        self.n_cls = n_cls
        self.ctx_ch = ctx_ch
        cin = n_cls + 1 + ctx_ch          # +1 = forward-range position channel

        def enc(ci, co):
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(co), nn.ReLU(inplace=True), ConvBlock(co, co))

        self.stem = nn.Sequential(
            nn.Conv2d(cin, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width), nn.ReLU(inplace=True))
        self.d1 = enc(width, width * 2)          # s2
        self.d2 = enc(width * 2, width * 3)       # s4
        self.d3 = enc(width * 3, width * 4)       # s8
        self.d4 = enc(width * 4, width * 4)       # s16 (wide RF ~ >80 m)
        self.u4 = nn.Conv2d(width * 4, width * 4, 1)
        self.m3 = ConvBlock(width * 4, width * 4)
        self.u3 = nn.Conv2d(width * 4, width * 3, 1)
        self.m2 = ConvBlock(width * 3, width * 3)
        self.u2 = nn.Conv2d(width * 3, width * 2, 1)
        self.m1 = ConvBlock(width * 2, width * 2)
        self.u1 = nn.Conv2d(width * 2, width, 1)
        self.out = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width), nn.ReLU(inplace=True),
            nn.Conv2d(width, n_cls, 1))
        nn.init.zeros_(self.out[-1].weight)      # identity at init
        nn.init.zeros_(self.out[-1].bias)
        self._rng_cache = {}

    def _range_chan(self, h, w, device, dtype):
        key = (h, w, device, dtype)
        c = self._rng_cache.get(key)
        if c is None:
            # row r -> forward distance x = 80 - r*0.2 (BEV convention);
            # encode x/80 in [~-1,1], broadcast across columns so the net
            # knows where the ~50 m dropout boundary sits
            x = (BEV_XH - (torch.arange(h, device=device, dtype=dtype) + 0.5)
                 * ((BEV_XF + BEV_XR) / h)) / BEV_XF
            c = x.view(1, 1, h, 1).expand(1, 1, h, w).contiguous()
            self._rng_cache[key] = c
        return c

    def forward(self, seg_logits, ctx=None):
        """seg_logits [B,n_cls,H,W] from the frozen model (detached upstream).
        ctx: optional [B,ctx_ch,H,W] raw-BEV context. Returns refined logits
        (same shape) = frozen logits + learned residual."""
        B, _, H, W = seg_logits.shape
        rng = self._range_chan(H, W, seg_logits.device,
                               seg_logits.dtype).expand(B, 1, H, W)
        parts = [seg_logits, rng]
        if self.ctx_ch and ctx is not None:
            parts.append(ctx)
        x = torch.cat(parts, 1)
        s0 = self.stem(x)
        s1 = self.d1(s0)
        s2 = self.d2(s1)
        s3 = self.d3(s2)
        s4 = self.d4(s3)

        def up(u, skip):
            return F.interpolate(u, size=skip.shape[-2:], mode="bilinear",
                                 align_corners=False)
        y3 = self.m3(s3 + up(self.u4(s4), s3))
        y2 = self.m2(s2 + up(self.u3(y3), s2))
        y1 = self.m1(s1 + up(self.u2(y2), s1))
        y0 = s0 + up(self.u1(y1), s0)
        res = self.out(y0)
        # fp16 safety: one inf activation poisons BN running stats forever
        # (r39 refiner: seg.out BN died at step 12.6k); bound the residual
        return seg_logits + 8.0 * torch.tanh(res / 8.0)


class BEVBoxRefiner(nn.Module):
    """Residual refiner for the 3D-box head (roadmap 3f, multi-task).

    The box head is a dense BEV representation too (CenterPoint: a centre
    heatmap + per-cell box regression on the 400x250 det grid), so the same
    residual-U-Net recipe as the seg refiner applies. It sharpens the centre
    peaks (recall/precision, esp. far range) and corrects the box regression
    (size / heading). Input is the FROZEN [hm(2) + reg(6)] maps + a range
    channel; zero-init last conv => identity at start, so it can only add to
    the frozen detection (never degrades it)."""

    def __init__(self, width=32):
        super().__init__()
        cin = 8 + 1                      # hm(2) + reg(6) + forward-range

        def enc(ci, co):
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(co), nn.ReLU(inplace=True), ConvBlock(co, co))

        self.stem = nn.Sequential(
            nn.Conv2d(cin, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width), nn.ReLU(inplace=True))
        self.d1 = enc(width, width * 2)          # s2
        self.d2 = enc(width * 2, width * 3)       # s4
        self.d3 = enc(width * 3, width * 4)       # s8 (wide RF over the grid)
        self.u3 = nn.Conv2d(width * 4, width * 3, 1)
        self.m2 = ConvBlock(width * 3, width * 3)
        self.u2 = nn.Conv2d(width * 3, width * 2, 1)
        self.m1 = ConvBlock(width * 2, width * 2)
        self.u1 = nn.Conv2d(width * 2, width, 1)
        self.out = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width), nn.ReLU(inplace=True),
            nn.Conv2d(width, 8, 1))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)
        self._rng = {}

    def _range(self, h, w, device, dtype):
        c = self._rng.get((h, w, device, dtype))
        if c is None:
            x = (BEV_XH - (torch.arange(h, device=device, dtype=dtype) + 0.5)
                 * ((BEV_XF + BEV_XR) / h)) / BEV_XF
            c = x.view(1, 1, h, 1).expand(1, 1, h, w).contiguous()
            self._rng[(h, w, device, dtype)] = c
        return c

    def forward(self, hm, reg):
        x = torch.cat([hm, reg], 1)
        B, _, H, W = x.shape
        rng = self._range(H, W, x.device, x.dtype).expand(B, 1, H, W)
        y = torch.cat([x, rng], 1)
        s0 = self.stem(y)
        s1 = self.d1(s0)
        s2 = self.d2(s1)
        s3 = self.d3(s2)

        def up(u, skip):
            return F.interpolate(u, size=skip.shape[-2:], mode="bilinear",
                                 align_corners=False)
        y2 = self.m2(s2 + up(self.u3(s3), s2))
        y1 = self.m1(s1 + up(self.u2(y2), s1))
        y0 = s0 + up(self.u1(y1), s0)
        res = self.out(y0)
        res = 6.0 * torch.tanh(res / 6.0)     # fp16-safe bounded residual
        return hm + res[:, :2], reg + res[:, 2:]


class E2ERefiner(nn.Module):
    """Residual second-stage planner for the E2E head (roadmap 3f, multi-task).

    The E2E output is a low-dim vector (K hypotheses x 6 waypoints x 2 +
    confidences + controls), not a raster, so the refiner is an MLP rather
    than a U-Net. It sees the predicted plan, the current speed v0, and a
    pooled summary of the fused BEV (scene context), and predicts a residual
    correction on the waypoints. Zero-init last layer => identity at start,
    so the base planner's ADE is preserved and can only improve."""

    def __init__(self, ego_dim, k=EGO_K, ctx_ch=BEV_CH, hidden=256, max_res=3.0):
        super().__init__()
        self.k = k
        self.max_res = max_res           # bound the waypoint correction (m)
        # LayerNorm the concatenated input: the pooled BEV context can have a
        # large / uncalibrated magnitude, which under fp16 + a high LR blows
        # the MLP up to inf -> NaN. Normalising + a tanh-bounded residual keeps
        # the second-stage planner numerically stable.
        self.norm = nn.LayerNorm(ego_dim + 1 + ctx_ch)
        self.mlp = nn.Sequential(
            nn.Linear(ego_dim + 1 + ctx_ch, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 12 * k))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, ego, v0, fused_bev):
        ctx = F.adaptive_avg_pool2d(fused_bev, 1).flatten(1)
        v = v0.view(-1, 1)
        # do the MLP in fp32 for stability (ego_loss has small denominators)
        x = self.norm(torch.cat([ego.float(), v.float(), ctx.float()], 1))
        res = torch.tanh(self.mlp(x)) * self.max_res   # zero-init => 0 at start
        out = ego.clone().float()
        out[:, :12 * self.k] = ego[:, :12 * self.k].float() + res
        return out


class BEVDenseRefiner(nn.Module):
    """Generic residual U-Net for any dense BEV map on the det grid
    (400x250). Used for the agent-trajectory field (other-agent behaviour,
    39 ch) and the risk field (1 ch). Same recipe as the box refiner: a range
    channel, encoder to s8, decoder with skips, zero-init last conv so it is
    identity at start and can only add a correction to the frozen map."""

    def __init__(self, cin, width=32, bound=None):
        super().__init__()
        self.cin = cin
        self.bound = bound          # tanh-bound on the residual (fp16 safety)

        def enc(ci, co):
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(co), nn.ReLU(inplace=True), ConvBlock(co, co))

        self.stem = nn.Sequential(
            nn.Conv2d(cin + 1, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width), nn.ReLU(inplace=True))
        self.d1 = enc(width, width * 2)
        self.d2 = enc(width * 2, width * 3)
        self.d3 = enc(width * 3, width * 4)
        self.u3 = nn.Conv2d(width * 4, width * 3, 1)
        self.m2 = ConvBlock(width * 3, width * 3)
        self.u2 = nn.Conv2d(width * 3, width * 2, 1)
        self.m1 = ConvBlock(width * 2, width * 2)
        self.u1 = nn.Conv2d(width * 2, width, 1)
        self.out = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width), nn.ReLU(inplace=True),
            nn.Conv2d(width, cin, 1))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)
        self._rng = {}

    def _range(self, h, w, device, dtype):
        c = self._rng.get((h, w, device, dtype))
        if c is None:
            x = (BEV_XH - (torch.arange(h, device=device, dtype=dtype) + 0.5)
                 * ((BEV_XF + BEV_XR) / h)) / BEV_XF
            c = x.view(1, 1, h, 1).expand(1, 1, h, w).contiguous()
            self._rng[(h, w, device, dtype)] = c
        return c

    def forward(self, x):
        B, _, H, W = x.shape
        rng = self._range(H, W, x.device, x.dtype).expand(B, 1, H, W)
        s0 = self.stem(torch.cat([x, rng], 1))
        s1 = self.d1(s0)
        s2 = self.d2(s1)
        s3 = self.d3(s2)

        def up(u, skip):
            return F.interpolate(u, size=skip.shape[-2:], mode="bilinear",
                                 align_corners=False)
        y2 = self.m2(s2 + up(self.u3(s3), s2))
        y1 = self.m1(s1 + up(self.u2(y2), s1))
        y0 = s0 + up(self.u1(y1), s0)
        res = self.out(y0)
        if self.bound:
            # one fp16-inf activation permanently poisons the BN running
            # stats (r38 refiner died at step 19.5k); a bounded residual
            # cannot amplify itself into overflow
            res = self.bound * torch.tanh(res / self.bound)
        return x + res


TRAJ_CH = TRAJ_H * 2 * EGO_K + EGO_K      # 39: dense agent-forecast channels



class ImgDenseRefiner(nn.Module):
    """Residual refiner for per-camera image-space maps (depth logits, 2D
    seg, 2D det heads). Input arrives as [B,N,C,h,w] or [B*N,C,h,w]; the
    cameras are folded into the batch so one small module serves all of
    them. Three convs, zero-init last -> identity at start, and a
    tanh bound keeps fp16 safe like the BEV refiners."""

    def __init__(self, cin, width=32, bound=6.0):
        super().__init__()
        self.bound = bound
        self.body = nn.Sequential(
            nn.Conv2d(cin, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width), nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width), nn.ReLU(inplace=True),
            nn.Conv2d(width, cin, 1))
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, x):
        sh = x.shape
        z = x.reshape(-1, *sh[-3:]) if x.dim() == 5 else x
        res = self.body(z)
        if self.bound:
            res = self.bound * torch.tanh(res / self.bound)
        return (z + res).reshape(sh)


class VecRefiner(nn.Module):
    """Residual refiner for vector/set outputs (traffic-light state, lane
    graph points / meta / adjacency). Operates on the last dimension so
    [B,D], [B,M,D] and [B,M,P,2] all work. Zero-init last layer."""

    def __init__(self, dim, hidden=64, bound=None):
        super().__init__()
        self.bound = bound
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(inplace=True),
                                 nn.Linear(hidden, dim))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x):
        res = self.mlp(x.float())
        if self.bound:
            res = self.bound * torch.tanh(res / self.bound)
        return x + res.to(x.dtype)


class OccRefiner(nn.Module):
    """Residual refiner for the 3D occupancy logits [B,cls,Z,H,W]: the z
    slices are folded into the channel dim so a 2D U-Net-free conv stack
    can correct them at 200x200 without a 3D kernel."""

    def __init__(self, n_cls=10, z=16, width=64, bound=6.0):
        super().__init__()
        self.n_cls, self.z, self.bound = n_cls, z, bound
        c = n_cls * z
        self.body = nn.Sequential(
            nn.Conv2d(c, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width), nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width), nn.ReLU(inplace=True),
            nn.Conv2d(width, c, 1))
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, x):
        B, C, Z, H, W = x.shape
        res = self.body(x.reshape(B, C * Z, H, W))
        if self.bound:
            res = self.bound * torch.tanh(res / self.bound)
        return x + res.reshape(B, C, Z, H, W)


class MultiTaskRefiner(nn.Module):
    """Post-hoc residual refiners for the three priority heads, sharing the
    single frozen-model forward. Each enabled head is an independent zero-init
    residual module (BEV seg U-Net, 3D-box U-Net, E2E MLP), so none shares
    weights with another or with the frozen base -- every task is preserved by
    construction and only added to. Heads can be enabled independently and
    deployed separately."""

    def __init__(self, do_seg=True, do_box=True, do_e2e=True,
                 do_traj=False, do_risk=False, do_unk=False,
                 do_stat=False, do_pl=False, do_depth=False,
                 do_seg2d=False, do_det2d=False, do_occ=False, do_tl=False,
                 do_flow=False, do_lg=False, n_seg2d=21, n_depth=64,
                 n_cls=N_CLASSES, seg_width=48, box_width=32, seg_ctx=0,
                 ego_dim=None, ego_k=EGO_K):
        super().__init__()
        self.seg = BEVSegRefiner(n_cls, ctx_ch=seg_ctx,
                                 width=seg_width) if do_seg else None
        self.box = BEVBoxRefiner(width=box_width) if do_box else None
        self.e2e = (E2ERefiner(ego_dim, k=ego_k)
                    if (do_e2e and ego_dim) else None)
        # new heads: other-agent trajectory field + risk field
        self.traj = BEVDenseRefiner(TRAJ_CH, width=32) if do_traj else None
        self.risk = BEVDenseRefiner(1, width=24) if do_risk else None
        # dense unknown-obstacle logit refiner (v41+ out[17], 1ch 400x250)
        self.unk = BEVDenseRefiner(1, width=32, bound=6.0) if do_unk else None
        # r48: stationary flag (v26+ out[10], 1ch) and the pseudo-LiDAR
        # raster (v48 out[18], 4ch) get their own residual refiners
        self.stat = BEVDenseRefiner(1, width=24, bound=6.0) if do_stat else None
        self.pl = BEVDenseRefiner(4, width=32, bound=8.0) if do_pl else None
        # remaining heads, so every task the network emits can be refined:
        # image-space maps fold the cameras into the batch, the occupancy
        # logits fold z into channels, and the vector/set outputs get MLPs
        self.depth = (ImgDenseRefiner(n_depth, width=48, bound=8.0)
                      if do_depth else None)
        self.seg2d = (ImgDenseRefiner(n_seg2d, width=48, bound=8.0)
                      if do_seg2d else None)
        self.det2d_hm = (nn.ModuleList([ImgDenseRefiner(10, width=24,
                                                       bound=6.0)
                                        for _ in range(3)])
                         if do_det2d else None)
        self.det2d_reg = (nn.ModuleList([ImgDenseRefiner(4, width=24, bound=6.0)
                                         for _ in range(3)])
                          if do_det2d else None)
        self.occ = OccRefiner(width=64, bound=6.0) if do_occ else None
        self.tl = VecRefiner(4, hidden=64, bound=6.0) if do_tl else None
        self.flow = BEVDenseRefiner(2, width=24, bound=6.0) if do_flow else None
        self.lg_pts = VecRefiner(2, hidden=64, bound=8.0) if do_lg else None
        self.lg_meta = VecRefiner(4, hidden=64, bound=6.0) if do_lg else None
        self.lg_adj = VecRefiner(24, hidden=64, bound=6.0) if do_lg else None

    def forward(self, seg=None, hm=None, reg=None, ego=None, v0=None,
                fused=None, seg_ctx=None, traj=None, risk=None, unk=None,
                stat=None, pl=None, depth=None, seg2d=None, hm2d=None,
                reg2d=None, occ=None, tl=None, flow=None, lg=None):
        """Refine whichever frozen outputs are provided; returns a dict. Called
        through DDP so every enabled head's params are tracked each step."""
        out = {}
        if self.seg is not None and seg is not None:
            out["seg"] = self.seg(seg.clamp(-20.0, 20.0), seg_ctx)
        if self.box is not None and hm is not None:
            out["hm"], out["reg"] = self.box(hm.clamp(-15.0, 15.0),
                                             reg.clamp(-20.0, 20.0))
        if self.e2e is not None and ego is not None:
            out["ego"] = self.e2e(ego, v0, fused)
        if self.traj is not None and traj is not None:
            out["traj"] = self.traj(traj)
        if self.risk is not None and risk is not None:
            out["risk"] = self.risk(risk)
        if self.unk is not None and unk is not None:
            out["unk"] = self.unk(unk.clamp(-12.0, 12.0))
        if self.stat is not None and stat is not None:
            out["stat"] = self.stat(stat.clamp(-15.0, 15.0))
        if self.pl is not None and pl is not None:
            out["pl"] = self.pl(pl.clamp(-15.0, 15.0))
        if self.depth is not None and depth is not None:
            out["depth"] = self.depth(depth.clamp(-20.0, 20.0))
        if self.seg2d is not None and seg2d is not None:
            out["seg2d"] = self.seg2d(seg2d.clamp(-20.0, 20.0))
        if self.det2d_hm is not None and hm2d is not None:
            out["hm2d"] = [r(h.clamp(-15.0, 15.0))
                           for r, h in zip(self.det2d_hm, hm2d)]
            out["reg2d"] = [r(g.clamp(-20.0, 20.0))
                            for r, g in zip(self.det2d_reg, reg2d)]
        if self.occ is not None and occ is not None:
            out["occ"] = self.occ(occ.clamp(-15.0, 15.0))
        if self.tl is not None and tl is not None:
            out["tl"] = self.tl(tl.clamp(-15.0, 15.0))
        if self.flow is not None and flow is not None:
            out["flow"] = self.flow(flow.clamp(-30.0, 30.0))
        if self.lg_pts is not None and lg is not None:
            pts, meta, adj = lg
            out["lg_pts"] = self.lg_pts(pts)
            out["lg_meta"] = self.lg_meta(meta.clamp(-15.0, 15.0))
            out["lg_adj"] = self.lg_adj(adj.clamp(-15.0, 15.0))
        return out


class DepthSegIPMNetV40(DepthSegIPMNetV39):
    """v40 (method A): the multi-task refiner GRAFTED onto the model as
    trainable post-heads. The seg/box/e2e refiners that were trained as a
    frozen post-processor (train_refiner.py) become part of the network and
    are fine-tuned end-to-end in r35, initialised from r34 (48 M base) + the
    trained refiner (5 M heads). The refiner's learned weights carry forward,
    the whole 53 M model trains together, and deployment needs no separate
    refiner engine. Each refiner head is a residual (zero-init when fresh), so
    at the graft point the combined model reproduces base+refiner behaviour and
    fine-tuning only adapts it."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.refiner = MultiTaskRefiner(
            do_seg=True, do_box=True, do_e2e=True, n_cls=N_CLASSES,
            ego_dim=12 * EGO_K + EGO_K + 3)

    def forward(self, imgs, K, T_cam_ego, v0=None, prev_bev=None,
                warp_theta=None, lidar=None, lidar_bev=None, kin=None,
                intent=None):
        out = list(super().forward(imgs, K, T_cam_ego, v0, prev_bev,
                                   warp_theta, lidar=lidar,
                                   lidar_bev=lidar_bev, kin=kin, intent=intent))
        B = out[0].shape[0]
        if os.environ.get("METEOR_NOREF") == "1":
            # diagnostic export switch: price the built-in refiner on the
            # target device (its fp16-era measurement said latency-free; the
            # INT8 profile's ~20 ms of elementwise Myelin chains say check)
            return tuple(out)
        v0r = v0 if v0 is not None else out[0].new_zeros(B)
        r = self.refiner(seg=out[0].float(), hm=out[3].float(),
                         reg=out[4].float(), ego=out[7].float(),
                         v0=v0r, fused=self._fused_bev.float(), seg_ctx=None)
        out[0] = r["seg"]
        out[3], out[4] = r["hm"], r["reg"]
        out[7] = r["ego"]
        return tuple(out)


class _DenseObstacleHead(nn.Module):
    """Dense small-static-obstacle occupancy head (v41). Compact U-Net on the
    temporal feature -> a per-cell logit on the 400x250 det grid, trained on
    the LiDAR-accumulated dense GT (unknown_v2) with dense focal loss. Dense
    supervision avoids the sparse-positive collapse of the old peak head."""

    def __init__(self, cin, width=48):
        super().__init__()

        def enc(ci, co):
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(co), nn.ReLU(inplace=True), ConvBlock(co, co))

        self.stem = nn.Sequential(
            nn.Conv2d(cin, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width), nn.ReLU(inplace=True), ConvBlock(width, width))
        self.d1 = enc(width, width * 2)
        self.d2 = enc(width * 2, width * 3)
        self.u2 = nn.Conv2d(width * 3, width * 2, 1)
        self.m1 = ConvBlock(width * 2, width * 2)
        self.u1 = nn.Conv2d(width * 2, width, 1)
        self.out = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width), nn.ReLU(inplace=True),
            nn.Conv2d(width, 1, 1))
        nn.init.constant_(self.out[-1].bias, -2.0)   # rare-positive prior

    def forward(self, x):
        s0 = self.stem(x)
        s1 = self.d1(s0)
        s2 = self.d2(s1)

        def up(u, skip):
            return F.interpolate(u, size=skip.shape[-2:], mode="bilinear",
                                 align_corners=False)
        y1 = self.m1(s1 + up(self.u2(s2), s1))
        y0 = s0 + up(self.u1(y1), s0)
        return self.out(y0)


class DepthSegIPMNetV41(DepthSegIPMNetV40):
    """v41 (next-stage perception): two fundamental fixes on the r35 base.

    (1) Unknown detector redesign: the sparse-peak head (recall ~0.11) is
        replaced by a DENSE small-static-obstacle occupancy head trained on
        the LiDAR-accumulated dense GT (unknown_v2) with dense focal loss.
    (2) Far-range vehicle 3D-box recall: box_loss gains a range weight that
        up-weights far positives (BOX_FAR_W), so distant vehicles -- small in
        BEV and camera-sparse -- are pushed harder (recall was the weak point).
    """
    # 3.0, was 2.0. Measured on val: vehicle recall is 0.564 / 0.402 / 0.356 /
    # 0.089 over 0-20 / 20-40 / 40-60 / 60-80 m, and the far range is not a
    # sensing limit -- zeroing the two NARROW cameras drops 40-60 m recall from
    # 0.34 to 0.04, so those detections come almost entirely from the telephoto
    # pair and the evidence is there. Restricting the GT to boxes with >= 40
    # LiDAR points lifts 40-60 m recall to 0.44, i.e. a large share of the
    # misses are boxes the sensors never saw. What is left is worth pushing on.
    BOX_FAR_W = 4.0                       # extra weight on far-range box GT

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        # the sparse-peak unknown head (unk_stem/unk_head2) is superseded by
        # the dense head; drop its params (param-free Identity) so DDP does
        # not flag them as unused -> avoids find_unused_parameters overhead.
        self.unk_stem = nn.Identity()
        self.unk_head2 = nn.Identity()
        self.unk_dense = _DenseObstacleHead(256)

    def forward(self, imgs, K, T_cam_ego, v0=None, prev_bev=None,
                warp_theta=None, lidar=None, lidar_bev=None, kin=None,
                intent=None):
        out = list(super().forward(imgs, K, T_cam_ego, v0, prev_bev,
                                   warp_theta, lidar=lidar,
                                   lidar_bev=lidar_bev, kin=kin, intent=intent))
        out[17] = self.unk_dense(self._tf[:, :256].float())
        return tuple(out)

    def unk_dense_loss(self, pred, mask):
        """pred logits [B,1,H,W]; mask [B,H,W] float (1 = small obstacle,
        0 = free, -1 = no GT for this frame -> ignored).
        alpha-balanced focal on the dense mask, averaged over valid pixels."""
        p = pred[:, 0].float()
        g = mask.float()
        valid = (g >= 0).float()          # -1 sentinel frames contribute 0
        denom = valid.sum().clamp(min=1)
        g = g.clamp(min=0)
        # pos_weight lifts the focal equilibrium: with 0.3% positives and
        # alpha .75 alone, the calibrated positive prob plateaued at ~0.25
        # (never crossing any usable threshold, r36 valUnkD stuck at 0)
        pw = torch.tensor(8.0, device=p.device)
        bce = F.binary_cross_entropy_with_logits(p, g, reduction="none",
                                                 pos_weight=pw)
        pt = torch.exp(-bce.clamp(max=20))
        alpha = torch.where(g > 0.5, 0.75, 0.25)
        return (alpha * (1 - pt) ** 2 * bce * valid).sum() / denom * 100.0


class DepthSegIPMNetV42(DepthSegIPMNetV41):
    """v42 (r37): TRT-safety + far-VRU round.

    1. fp16-safe E2E decoder (stable softplus + 25 m/s cap in the v39 dec
       path -- fix lives in V39.forward, shared by all descendants): no
       overflow on ANY TensorRT precision.
    2. VRU_FAR_BAND: 25-45 m pedestrians boosted x2 in the box loss (was
       damped x0.2 past 40 m); recall at range was the weak point.
    Zero new parameters: r36/r37_init checkpoints load with missing=0.
    """
    VRU_FAR_BAND = True


class DepthSegIPMNetV43(DepthSegIPMNetV42):
    """v43 (r39): Driving-Command that actually steers.

    r38 finding: the v37 intent_delta is an INPUT-INDEPENDENT 3->39 bias --
    a 'left' command shifts the final waypoint by <=0.105 m whatever the
    scene, so commands visibly do nothing. Two fixes:
    1. Context-dependent conditioning: intent_mlp([pooled fused BEV, intent])
       -> ego-output delta (zero-init last layer, tanh-bounded +-8 m,
       fp16-safe): the shift can now depend on the junction geometry.
    2. Train with --intent-w: a consistency hinge on the (soft) selected
       mode's final lateral displacement vs the command direction
       (train.py intent_loss).
    """
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.intent_mlp = nn.Sequential(
            nn.Linear(BEV_CH + 3, 64), nn.ReLU(),
            nn.Linear(64, 12 * EGO_K + EGO_K + 3))
        nn.init.zeros_(self.intent_mlp[-1].weight)
        nn.init.zeros_(self.intent_mlp[-1].bias)

    def forward(self, imgs, K, T_cam_ego, v0=None, prev_bev=None,
                warp_theta=None, lidar=None, lidar_bev=None, kin=None,
                intent=None):
        out = list(super().forward(imgs, K, T_cam_ego, v0, prev_bev,
                                   warp_theta, lidar=lidar,
                                   lidar_bev=lidar_bev, kin=kin,
                                   intent=intent))
        if intent is not None:
            g = F.adaptive_avg_pool2d(self._fused_bev, 1).flatten(1).float()
            d = self.intent_mlp(torch.cat([g, intent.float()], 1))
            d = 8.0 * torch.tanh(d / 8.0)          # bounded, fp16-safe
            gate = intent.float().amax(1, keepdim=True)   # 0-vec = no nav
            out[7] = out[7] + (d * gate).to(out[7].dtype)
        return tuple(out)

    @staticmethod
    def intent_loss(ego_pred, intent, margin=1.5):
        """Command-consistency hinge: the (softmax-soft) selected mode's
        final lateral displacement must agree with the commanded direction.
        intent [B,3] one-hot (straight,left,right); rows dropped to zero by
        the training dropout contribute nothing.

        `margin` is the whole strength of the counterfactual signal and must
        be set from the acceptance bar, not left at a token value. The hinge
        is EXACTLY ZERO once the commanded mode reaches `margin` metres in the
        commanded direction, so margin=1.5 caps the achievable command spread
        at ~3 m however long it trains -- measured on r48 ep0: spread 1.24 m,
        sign reversal 20 %/4 % against a 5 m / 60 % bar. Meanwhile the same
        mode keeps taking waypoint gradients toward the real (opposite)
        manoeuvre from the 85 % of correctly-commanded rows, so the
        equilibrium sits at or below the margin. Pass margin >= half the
        required spread (r48: 3.5)."""
        B = ego_pred.shape[0]
        e = ego_pred.float()
        wp = e[:, :12 * EGO_K].view(B, EGO_K, 6, 2)
        probs = e[:, 12 * EGO_K:12 * EGO_K + EGO_K].softmax(1)
        lat = (probs * wp[:, :, -1, 1]).sum(1)          # soft-selected y
        dire = intent[:, 1] - intent[:, 2]              # left=+1 right=-1
        m = (dire.abs() > 0.5).float()
        pen = F.relu(margin - dire * lat)
        return (pen * m).sum() / m.sum().clamp(min=1)


class DepthSegIPMNetV44(DepthSegIPMNetV43):
    """v44 (r40): Driving Command -> mode BINDING (structural, zero params).

    v43's context delta moved the path by only ~0.2 m: with GT-following
    losses, a residual correction is never forced to matter. v44 instead
    assigns SEMANTICS to the K=3 modes (0=straight, 1=left, 2=right) by
    adding +MODE_BOOST to the commanded mode's logit inside forward, so a
    command always SELECTS its mode.

    Selection was never the problem; what the mode selected meant was. The
    original claim here -- that routing the selection-dependent losses made
    the binding emerge "by construction" -- was wrong for the WTA waypoint
    loss, whose winner is the argmin of the waypoint error and never looks
    at the logits. Measured on r45 over 50 val turn frames: the K=3 paths
    differ by 1.08 m and reverse the turn on 0 % of them, i.e. commanding
    "left" on a right-turn frame still turned right; the modes had learned
    magnitude, not direction. ego_loss(intent=...) now routes the WTA winner
    by the command (and de-boosts the logits before the selector CE), which
    is what actually accumulates manoeuvre-j gradients in mode j.
    intent_mode_loss additionally aligns the RAW logits with the manoeuvre
    so the no-command mode selection improves too."""
    MODE_BOOST = 8.0
    # J6 sensor-config fine-tune: indices of cameras to hard-zero at every
    # image entry point (train + all evals + demo see the same 7-cam world)
    zero_cams = ()

    def _mask_cams(self, imgs):
        if not self.zero_cams:
            return imgs
        imgs = imgs.clone()
        imgs[:, list(self.zero_cams)] = 0
        return imgs

    def compute_bev(self, imgs, K, T_cam_ego, *a, **k):
        return super().compute_bev(self._mask_cams(imgs), K, T_cam_ego,
                                   *a, **k)

    def forward(self, imgs, K, T_cam_ego, v0=None, prev_bev=None,
                warp_theta=None, lidar=None, lidar_bev=None, kin=None,
                intent=None):
        imgs = self._mask_cams(imgs)
        out = list(super().forward(imgs, K, T_cam_ego, v0, prev_bev,
                                   warp_theta, lidar=lidar,
                                   lidar_bev=lidar_bev, kin=kin,
                                   intent=intent))
        if intent is not None:
            e = out[7].clone()
            e[:, 12 * EGO_K:12 * EGO_K + EGO_K] = \
                e[:, 12 * EGO_K:12 * EGO_K + EGO_K] \
                + self.MODE_BOOST * intent.to(e.dtype)
            out[7] = e
        return tuple(out)

    def intent_mode_loss(self, ego_pred, intent):
        """CE on the RAW (pre-boost) mode logits toward the commanded mode,
        on rows where a command is present."""
        lg = (ego_pred[:, 12 * EGO_K:12 * EGO_K + EGO_K].float()
              - self.MODE_BOOST * intent.float())
        m = intent.sum(1) > 0.5
        if not m.any():
            return ego_pred.new_zeros(())
        return F.cross_entropy(lg[m], intent[m].argmax(1))


class DepthSegIPMNetV45(DepthSegIPMNetV44):
    """v45 (r43): lateral-departure recovery + INT8 robustness.

    1. Recovery: train-side SE(2) lateral offset of the ego frame
       (bev_rotation_aug lat_max/lat_p) with hermite-smoothed
       return-to-lane E2E targets -- exact under the depth-lifted
       projection, ChauffeurNet-style. No model change needed here.
    2. INT8-robust features (quant_noise > 0): forward hooks on the
       image-feature fuse and the temporal BEV fuse inject per-channel
       uniform noise of one int8 rounding step (ch_absmax/127), making
       activations tolerant to post-training quantization.
    Zero new parameters."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.quant_noise = 0.0
        self.bev_dropblock = 0.0       # r46: MAE-like BEV block masking
        self.bev_wedgedrop = 0.0       # 2026-08-30: 角度セクタ欠損の模擬
        self.fuse.register_forward_hook(self._qnoise_hook)
        self.tfuse3.register_forward_hook(self._qnoise_hook)
        self.tfuse3.register_forward_hook(self._dropblock_hook)
        self.tfuse3.register_forward_hook(self._wedgedrop_hook)
        self.bev_ringdrop = 0.0        # 2026-08-30: 距離帯リング零化
        self.bev_chandrop = 0.0        # 2026-08-30: チャネルドロップ
        self.tfuse3.register_forward_hook(self._ringdrop_hook)
        self.tfuse3.register_forward_hook(self._chandrop_hook)

    def _qnoise_hook(self, module, inp, out):
        if not self.training or self.quant_noise <= 0:
            return None
        amax = out.detach().abs().amax(dim=(0, 2, 3), keepdim=True)
        delta = amax / 127.0
        return out + (torch.rand_like(out) - 0.5) * delta \
            * (2.0 * self.quant_noise)

    def _dropblock_hook(self, module, inp, out):
        """MAE-like masking on the temporal-fused BEV feature: zero a few
        16-40 m rectangles per sample so every head must inpaint them from
        surrounding context + temporal memory (GT stays complete). Train
        only; probability per sample = self.bev_dropblock."""
        if not self.training or self.bev_dropblock <= 0:
            return None
        B, _, H, W = out.shape
        m = torch.ones(B, 1, H, W, device=out.device, dtype=out.dtype)
        hit = False
        for b in range(B):
            if torch.rand(()) >= self.bev_dropblock:
                continue
            hit = True
            for _ in range(int(torch.randint(2, 5, ()))):
                bh = int(torch.randint(H // 10, H // 4 + 1, ()))
                bw = int(torch.randint(W // 10, W // 4 + 1, ()))
                r0 = int(torch.randint(0, H - bh + 1, ()))
                c0 = int(torch.randint(0, W - bw + 1, ()))
                m[b, :, r0:r0 + bh, c0:c0 + bw] = 0
        return out * m if hit else None

    def _wedgedrop_hook(self, module, inp, out):
        """極座標くさびドロップ (2026-08-30): 自車を原点に 40-90° の角度
        セクタを BEV 特徴ごと零化し、カメラ 1 本分の視野欠損を特徴レベルで
        模擬する。入力レベルの cam-drop と違い、塗り込み済み LiDAR 特徴や
        時間メモリ経由の残存も含めて「その方角が見えない」状態を作る。
        GT は完全なまま = 周辺文脈と時間記憶からの補完を学習させる。
        train のみ、サンプル毎確率 = self.bev_wedgedrop。"""
        if not self.training or getattr(self, "bev_wedgedrop", 0) <= 0:
            return None
        B, _, H, W = out.shape
        dev = out.device
        ys = torch.arange(H, device=dev, dtype=torch.float32) - (H - 1) / 2
        xs = torch.arange(W, device=dev, dtype=torch.float32) - (W - 1) / 2
        ang = torch.atan2(ys[:, None], xs[None, :])          # [-pi, pi]
        m = torch.ones(B, 1, H, W, device=dev, dtype=out.dtype)
        hit = False
        for b in range(B):
            if torch.rand(()) >= self.bev_wedgedrop:
                continue
            hit = True
            c = (torch.rand(()) * 2 - 1) * torch.pi          # 中心角
            half = torch.deg2rad(20 + torch.rand(()) * 25)   # 半幅 20-45°
            d = (ang - c + torch.pi) % (2 * torch.pi) - torch.pi
            m[b, 0][d.abs() < half] = 0
        return out * m if hit else None

    def _ringdrop_hook(self, module, inp, out):
        """距離帯リング零化 (2026-08-30): 自車からの距離 r0..r0+w の帯を
        BEV 特徴ごと零化。遠方ヘッドに「手前が見えない」補完を、近傍
        ヘッドに「中間帯の欠損」への頑健性を要求する。GT は完全なまま。
        train のみ、サンプル毎確率 = self.bev_ringdrop。"""
        if not self.training or getattr(self, "bev_ringdrop", 0) <= 0:
            return None
        B, _, H, W = out.shape
        dev = out.device
        ys = torch.arange(H, device=dev, dtype=torch.float32) - (H - 1) / 2
        xs = torch.arange(W, device=dev, dtype=torch.float32) - (W - 1) / 2
        rr = torch.sqrt(ys[:, None] ** 2 + xs[None, :] ** 2)
        rmax = rr.max()
        m = torch.ones(B, 1, H, W, device=dev, dtype=out.dtype)
        hit = False
        for b in range(B):
            if torch.rand(()) >= self.bev_ringdrop:
                continue
            hit = True
            r0 = torch.rand(()) * rmax * 0.7                 # 内径 0-70%
            w = (0.08 + torch.rand(()) * 0.12) * rmax        # 幅 8-20%
            m[b, 0][(rr >= r0) & (rr < r0 + w)] = 0
        return out * m if hit else None

    def _chandrop_hook(self, module, inp, out):
        """BEV チャネルドロップ (2026-08-30): tfuse3 出力のチャネルを
        SpatialDropout 風にサンプル毎へ確率 self.bev_chandrop で零化
        (面ではなく特徴軸の冗長性を強制)。生存チャネルは 1/(1-p) 補償。"""
        if not self.training or getattr(self, "bev_chandrop", 0) <= 0:
            return None
        p = float(self.bev_chandrop)
        B, C, _, _ = out.shape
        keep = (torch.rand(B, C, 1, 1, device=out.device) >= p).to(out.dtype)
        return out * keep / (1.0 - p)

    @staticmethod
    def stat_loss(stat, boxes, nbox, traj, tvalid, margin=0.0):
        """r46 stationary supervision v2: paint the label over the WHOLE
        rotated box footprint on the det grid (v26 used only the centre
        cell -> ~20x sparser signal), keep the 0.35-0.8 m creep dead-band,
        and balance the classes per batch (parked cars dominate)."""
        B = boxes.shape[0]
        dev = stat.device
        lbl = torch.full((B, DET_H, DET_W), -1.0, device=dev)
        for b in range(B):
            for k in range(int(nbox[b])):
                if boxes[b, k, 3] <= 0 or tvalid[b, k, 5] < 0.5:
                    continue
                d3 = float(traj[b, k, 5].norm())
                if 0.35 < d3 < 0.8:          # ambiguous creep band
                    continue
                xe, ye = float(boxes[b, k, 1]), float(boxes[b, k, 2])
                l_ = float(boxes[b, k, 3]); w_ = float(boxes[b, k, 4])
                yaw = float(boxes[b, k, 5]) if boxes.shape[2] > 5 else 0.0
                rc = (BEV_XF - xe) / DET_RES
                cc = (BEV_YH - ye) / DET_RES
                half = max(l_, w_) / (2 * DET_RES) + 1
                r0, r1 = int(max(0, rc - half)), int(min(DET_H, rc + half + 1))
                c0, c1 = int(max(0, cc - half)), int(min(DET_W, cc + half + 1))
                if r0 >= r1 or c0 >= c1:
                    continue
                rr = torch.arange(r0, r1, device=dev, dtype=torch.float32)
                cx = torch.arange(c0, c1, device=dev, dtype=torch.float32)
                X = 80.0 - (rr[:, None] + 0.5) * DET_RES - xe
                Y = 50.0 - (cx[None, :] + 0.5) * DET_RES - ye
                ca, sa = math.cos(yaw), math.sin(yaw)
                u = X * ca + Y * sa
                v = -X * sa + Y * ca
                inside = (u.abs() <= l_ / 2) & (v.abs() <= w_ / 2)
                lbl[b, r0:r1, c0:c1][inside] = float(d3 <= 0.35)
        m = lbl >= 0
        if not m.any():
            return stat.sum() * 0.0
        logits = stat[:, 0].float().clamp(-15, 15)[m]
        target = lbl[m]
        n_pos = float(target.sum()); n_neg = float(len(target)) - n_pos
        # inverse-frequency weight, clamped: parked (pos) usually dominates
        w_pos = min(max(n_neg / max(n_pos, 1.0), 0.5), 4.0)
        w = torch.where(target > 0.5, torch.full_like(target, w_pos),
                        torch.ones_like(target))
        loss = F.binary_cross_entropy_with_logits(logits, target, weight=w)
        if margin > 0:
            # stationary=+1, moving=-1. BCE gets the class right; this term
            # additionally keeps correct logits away from the zero threshold,
            # where one INT8 rounding step can flip the deployed decision.
            signed = target.mul(2.0).sub(1.0) * logits
            loss = loss + F.relu(float(margin) - signed).mean()
        return loss


class DepthSegIPMNetV46(DepthSegIPMNetV45):
    """v46 (r44): FREE SD-map (OpenStreetMap) prior as an OPTIONAL input.

    Identical recipe to the v32 LiDAR raster: sdmap [B,4,400,250] (road
    area / centerline / intersections / crossings+signals, rendered into
    the ego frame from per-pose GNSS + OSM) -> zero-init conv stem ->
    flag-gated residual on the BEV. An all-zero input is bit-equal to
    no-map, and training drops the map on half the samples, so ONE set of
    weights serves both GNSS-less and map-assisted operation."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.sdmap_stem = nn.Sequential(
            nn.Conv2d(4, 48, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(48, BEV_CH, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(BEV_CH, BEV_CH, 1))
        nn.init.zeros_(self.sdmap_stem[-1].weight)
        nn.init.zeros_(self.sdmap_stem[-1].bias)
        self._sdmap = None

    def bev_extra(self, bev):
        bev = super().bev_extra(bev)
        if self._sdmap is None:
            return bev
        sd = self._sdmap.to(bev.dtype)
        flag = (sd.abs().sum((1, 2, 3), keepdim=True) > 0).to(bev.dtype)
        res = F.interpolate(self.sdmap_stem(sd), bev.shape[-2:],
                            mode="bilinear", align_corners=False)
        return bev + flag * res

    def forward(self, imgs, K, T_cam_ego, v0=None, prev_bev=None,
                warp_theta=None, lidar=None, lidar_bev=None, kin=None,
                intent=None, sdmap=None):
        self._sdmap = sdmap
        try:
            return super().forward(imgs, K, T_cam_ego, v0, prev_bev,
                                   warp_theta, lidar=lidar,
                                   lidar_bev=lidar_bev, kin=kin,
                                   intent=intent)
        finally:
            self._sdmap = None          # history compute_bev stays map-free


class DepthSegIPMNetV47(DepthSegIPMNetV46):
    """v47 (r45): per-camera BOX-LEVEL traffic-light states as an OPTIONAL
    input (from an external recognizer; dummy = dataset color_shape ann).

    Input tl [B,N,7,27,48]: per camera a raster painted inside each light
    element's bbox -- channels [red, yellow, green, is_ped, is_arrow,
    sin(orient), cos(orient)] (orientation: 0=up, +pi/2=right, clockwise).
    Injection: bias-free zero-init stem added onto the per-camera image
    feature, so the depth-lift carries the states into BEV with the
    camera's own geometry. Bias-free + linear head guarantees an all-zero
    raster (or tl=None) is BIT-EQUAL to a no-input run -- ON/OFF safe."""

    TL_CH, TL_H, TL_W = 7, 27, 48

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.tl_stem = nn.Sequential(
            nn.Conv2d(self.TL_CH, 48, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, self.ctx.in_channels, 1, bias=False))
        nn.init.zeros_(self.tl_stem[-1].weight)
        self._tl = None

    def image_feats(self, imgs):
        f = super().image_feats(imgs)
        tl = self._tl
        if tl is None and self.training and torch.is_grad_enabled():
            # DDP: the stem must join every grad pass (bias-free, so an
            # all-zero raster contributes exactly nothing numerically)
            B, N = imgs.shape[:2]
            tl = imgs.new_zeros(B, N, self.TL_CH, self.TL_H, self.TL_W)
        if tl is not None:
            self._tl = None            # one-shot: history passes stay TL-free
            B, N = tl.shape[:2]
            res = self.tl_stem(tl.reshape(B * N, self.TL_CH,
                                          self.TL_H, self.TL_W).to(f.dtype))
            f = f + F.interpolate(res, f.shape[-2:], mode="bilinear",
                                  align_corners=False)
        return f

    def forward(self, imgs, K, T_cam_ego, v0=None, prev_bev=None,
                warp_theta=None, lidar=None, lidar_bev=None, kin=None,
                intent=None, sdmap=None, tl=None):
        self._tl = tl
        try:
            return super().forward(imgs, K, T_cam_ego, v0, prev_bev,
                                   warp_theta, lidar=lidar,
                                   lidar_bev=lidar_bev, kin=kin,
                                   intent=intent, sdmap=sdmap)
        finally:
            self._tl = None


class DepthSegIPMNetV48(DepthSegIPMNetV47):
    """v48 (r47): PSEUDO-LiDAR -- predict the LiDAR BEV raster from cameras
    and feed it back through the SAME optional-LiDAR stem.

    The head reads the CAMERA-ONLY BEV (before any LiDAR/SD-map residual),
    so it cannot cheat by copying an injected real sweep, and predicts the
    4-channel raster extract_lidar_bev.py produces: log-count, max z,
    mean z, occupancy (no intensity -- cameras cannot infer reflectance).
    Supervision is the real raster where a sweep exists; the prediction is
    DETACHED before it is fed back, so the head is shaped only by that
    distillation loss and never by "whatever helps the other heads".

    Feeding reuses v32's zero-init `lidar_stem`, per sample:
        real sweep  > pseudo raster > nothing (zero residual)
    Training drops real LiDAR (--lidar-drop) and feeds the pseudo raster on
    a fraction of the rest (--pl-feed-p), so ONE checkpoint serves all three
    modes and `pl_feed=False` stays bit-equal to the camera-only network:
    inference-time ON/OFF, no re-training."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.pl_head = nn.Sequential(
            nn.Conv2d(BEV_CH, BEV_CH, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(BEV_CH), nn.ReLU(inplace=True),
            ConvBlock(BEV_CH, BEV_CH),
            nn.Conv2d(BEV_CH, 4, 1))
        self.pl_feed = False       # inference switch (or pl_feed= per call)
        self.pl_feed_p = 1.0       # train: fraction of eligible samples fed
        self._pl_want = False
        self._pl_raw = None

    @staticmethod
    def pl_activate(raw, hard=False):
        """logits -> a raster in the real one's units/sparsity.
        hard=True gates by occupancy>0.5 so the fed raster looks like a
        real sweep (empty cells exactly 0), which is what lidar_stem saw."""
        occ = torch.sigmoid(raw[:, 3:4].float())
        cnt = F.softplus(raw[:, 0:1].float())
        zmax = -1.0 + 5.0 * torch.sigmoid(raw[:, 1:2].float())
        zmean = -1.0 + 5.0 * torch.sigmoid(raw[:, 2:3].float())
        g = (occ > 0.5).to(occ.dtype) if hard else occ
        return torch.cat([cnt * g, zmax * g, zmean * g,
                          g if hard else occ], 1)

    def bev_extra(self, bev):
        if self._pl_want:
            self._pl_want = False           # current frame only
            raw = self.pl_head(bev)         # camera-only BEV -> raster
            self._pl_raw = raw
            if self.pl_feed:
                pl = self.pl_activate(raw, hard=True).detach().to(bev.dtype)
                if self.training and self.pl_feed_p < 1.0:
                    keep = (torch.rand(pl.shape[0], 1, 1, 1,
                                       device=pl.device)
                            < self.pl_feed_p).to(pl.dtype)
                    pl = pl * keep
                if self._lidar_bev is None:
                    self._lidar_bev = pl
                else:                        # real sweep wins per sample
                    real = crop_rows(self._lidar_bev, pl.shape[-2]).to(pl.dtype)
                    has = (real.abs().sum((1, 2, 3), keepdim=True) > 0
                           ).to(pl.dtype)
                    self._lidar_bev = real * has + pl * (1 - has)
        return super().bev_extra(bev)

    def pseudo_lidar_loss(self, raw, lb):
        """BCE on occupancy + L1 on log-count/heights inside GT-occupied
        cells. Frames without a sweep contribute 0 but stay in the graph."""
        if raw is None:
            return None
        if lb is None:
            return raw.float().sum() * 0.0
        lb = lb.to(raw.device).float()
        if lb.shape[-2:] != raw.shape[-2:]:
            lb = F.interpolate(lb, raw.shape[-2:], mode="nearest")
        valid = (lb.abs().sum((1, 2, 3), keepdim=True) > 0).float()
        occ_gt = (lb[:, 3:4] > 0.5).float()
        bce = F.binary_cross_entropy_with_logits(
            raw[:, 3:4].float().clamp(-15, 15), occ_gt, reduction="none")
        bce = (bce * valid).sum() / valid.expand_as(bce).sum().clamp(min=1)
        act = self.pl_activate(raw, hard=False)
        m = occ_gt * valid
        den = m.sum().clamp(min=1)
        l1 = ((act[:, 0:1] - lb[:, 0:1]).abs() * m).sum() / den
        l1 = l1 + 0.5 * ((act[:, 1:2] - lb[:, 1:2]).abs() * m).sum() / den
        l1 = l1 + 0.5 * ((act[:, 2:3] - lb[:, 2:3]).abs() * m).sum() / den
        return bce + 0.2 * l1

    def forward(self, imgs, K, T_cam_ego, v0=None, prev_bev=None,
                warp_theta=None, lidar=None, lidar_bev=None, kin=None,
                intent=None, sdmap=None, tl=None, pl_feed=None):
        self._pl_want = True
        self._pl_raw = None
        prev = self.pl_feed
        if pl_feed is not None:
            self.pl_feed = bool(pl_feed)
        try:
            out = super().forward(imgs, K, T_cam_ego, v0, prev_bev,
                                  warp_theta, lidar=lidar,
                                  lidar_bev=lidar_bev, kin=kin,
                                  intent=intent, sdmap=sdmap, tl=tl)
        finally:
            self._pl_want = False
            self.pl_feed = prev
        return tuple(out) + (self._pl_raw,)      # out[18] = pseudo-LiDAR


SIG_FLOOR = 0.8        # metres; must match RegDepthHead's own lower bound


class RegDepthHead(nn.Module):
    """Regression depth head: 2 channels for the existing seam + a per-pixel
    kernel width stashed on the side.

    The base forward does `dlog.softmax(1)` and hands the result to the lift, so
    the head has to keep emitting something softmax-able. Channel 0 after the
    softmax is read as a normalised depth p in (0,1). The width cannot ride in
    the same tensor without softmax coupling it to the depth (a confident close
    range would be forced to a wide kernel), so it is produced by its own 1x1
    and stashed for project_bev to pick up. One global sigma made the lift smear
    -- the depth panel of the distilled model reads visibly softer than the
    64-bin one -- so it is per pixel here.
    """

    def __init__(self, cin, width=128):
        super().__init__()
        self.body = ConvBlock(cin, width)
        self.mu = nn.Conv2d(width, 2, 1)
        self.sig = nn.Conv2d(width, 1, 1)
        nn.init.zeros_(self.sig.weight)
        nn.init.constant_(self.sig.bias, 0.0)      # sigmoid(0) -> mid range
        self.sigma_map = None

    def forward(self, x):
        h = self.body(x)
        # 0.8 .. 6.8 m, differentiable, always positive. The lower bound was
        # 0.3 m and r51 spent 2.0 % of its steps skipping a non-finite loss with
        # NaNs surfacing in the RL reward and the pseudo-LiDAR metric: at 0.3 m
        # the lift weight exp(-(d-mu)^2 / 2 sigma^2) has a 1/0.18 factor in the
        # exponent, so a single badly-placed cell overflows fp16 and the whole
        # step is thrown away. 0.8 m still resolves a lane line at 0.2 m cells.
        self.sigma_map = SIG_FLOOR + 6.0 * torch.sigmoid(
            self.sig(h).float())
        return self.mu(h)


class DepthSegIPMNetV50(DepthSegIPMNetV48):
    """v50: depth as a 0-1 REGRESSION instead of a 64-bin distribution.

    Measured motivation: the depth head is 3.29M parameters and 10.80 ms of the
    80.6 ms forward (and the top six individual layers of the TensorRT engine),
    while the lift then samples all 64 channels only to gather two of them.
    A regression head is 0.33M / 1.20 ms (9x) and the lift samples ONE channel.

    The head keeps the existing seam: it emits 2 channels, the base forward
    softmaxes them, and channel 0 is read as a normalised depth p in (0,1) ->
    mu = D_MIN + p * span. The lift weight becomes a Gaussian around mu with a
    LEARNED width, so the head can still express "far and uncertain" the way the
    64-bin histogram could.

    It cannot be trained on depth GT alone: the BEV feature it produces feeds
    all twelve heads, and matching depth error does not mean matching features.
    bevlane/distill_depth.py trains it to reproduce the frozen model's BEV
    feature, which is the thing the other heads actually consume.
    """

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        cin = self.depth_head[0][0].in_channels
        self.depth_head = RegDepthHead(cin, 128)
        # Fallback width for checkpoints distilled before the per-pixel sigma
        # existed. A BUFFER, not a Parameter: once sigma_map is in use nothing
        # reads it, and a Parameter with no gradient is exactly what DDP refuses
        # to tolerate (it killed r51 before step 1).
        self.register_buffer("log_sigma", torch.tensor(0.4055))   # 1.5 m
        self.depth_regr = True

    def project_bev(self, dprob, ctx, K, T_cam_ego, B, N, H, W):
        if not getattr(self, "depth_regr", False):
            return super().project_bev(dprob, ctx, K, T_cam_ego, B, N, H, W)
        if getattr(self, "frustum_lift", False) and not self.training:
            # This override used to skip the frustum path entirely, so the v50
            # engine ran a DENSE lift and came out at 58.2 ms against the
            # v48+frustum engine's 40.8 ms -- the regression head's saving was
            # swamped by giving back the 21 ms the frustum restriction had won.
            return self._project_bev_frustum_regr(dprob, ctx, K, T_cam_ego,
                                                  B, N, H, W)
        Cc = ctx.shape[1]
        pts = self.bev_pts
        G2 = pts.shape[0]
        pc = torch.matmul(T_cam_ego.reshape(B * N, 4, 4),
                          pts.t().unsqueeze(0).expand(B * N, 4, G2))
        x, y, z = pc[:, 0], pc[:, 1], pc[:, 2]
        Kf = K.reshape(B * N, 3, 3)
        zc = z.clamp(min=0.5)
        u = Kf[:, 0, 0].unsqueeze(-1) * x / zc + Kf[:, 0, 2].unsqueeze(-1)
        v = Kf[:, 1, 1].unsqueeze(-1) * y / zc + Kf[:, 1, 2].unsqueeze(-1)
        dist = torch.sqrt(x * x + y * y + z * z)
        valid = ((z > 0.5) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
                 & (dist < 90.0))
        gu = (u / (W - 1) * 2 - 1).clamp(-2, 2)
        gv = (v / (H - 1) * 2 - 1).clamp(-2, 2)
        grid = torch.stack([gu, gv], -1).unsqueeze(2)
        ctx_s = F.grid_sample(ctx, grid, align_corners=False).squeeze(-1)
        # ONE channel instead of 64
        p = F.grid_sample(dprob[:, :1], grid,
                          align_corners=False).squeeze(-1).squeeze(1)
        span = (self.D - 1) * self.D_STEP
        mu = self.D_MIN + p * span
        sm = getattr(self.depth_head, "sigma_map", None)
        if sm is None:
            sig = self.log_sigma.exp().clamp(0.3, 20.0).to(mu.dtype)
        else:                       # per-pixel width, sampled like the depth
            # grid_sample pads with ZEROS, so every BEV cell that projects
            # outside its camera reads sigma = 0, and 2*sigma*sigma = 0 turns
            # the exponent into 0/0 -> NaN whenever dist happens to equal mu to
            # fp16 resolution. Those cells are masked out a line later, but
            # 0 * NaN is NaN, so the mask cannot clean it: one such cell NaNs
            # the whole BEV feature through the encoder's convolutions and BN,
            # and the step is discarded. That was 2 % of every step of r51 and
            # r52 (measured: "lift: wgt 2/6400000 non-finite" with ctx, mu and
            # sig all clean). Raising the head's own floor did nothing because
            # the zero comes from the padding, not from the head.
            sig = F.grid_sample(sm.to(ctx.dtype), grid,
                                align_corners=False).squeeze(-1).squeeze(1)
            sig = sig.clamp(min=SIG_FLOOR)
        _nanchk("lift: ctx", ctx)
        _nanchk("lift: ctx_s (sampled)", ctx_s)
        _nanchk("lift: mu", mu)
        _nanchk("lift: sig", sig)
        wgt = (torch.exp(-((dist - mu) ** 2) / (2 * sig * sig))
               + 0.05).unsqueeze(1)
        wgt = wgt * valid.unsqueeze(1).to(wgt.dtype)
        _nanchk("lift: wgt", wgt)
        num = (ctx_s.view(B, N, Cc, G2) * wgt.view(B, N, 1, G2)).sum(1)
        den = wgt.view(B, N, 1, G2).sum(1).clamp(min=1e-4)
        _nanchk("lift: num", num)
        _nanchk("lift: den", den)
        return _nanchk("lift: out",
                       (num / den).view(B, Cc, BEV_H, BEV_W))

    def _project_bev_frustum_regr(self, dprob, ctx, K, T_cam_ego, B, N, H, W):
        """Frustum-restricted lift with the regression depth weight."""
        Cc = ctx.shape[1]
        pts = self.bev_pts
        G2 = pts.shape[0]
        pc = torch.matmul(T_cam_ego.reshape(B * N, 4, 4),
                          pts.t().unsqueeze(0).expand(B * N, 4, G2))
        x, y, z = pc[:, 0], pc[:, 1], pc[:, 2]
        Kf = K.reshape(B * N, 3, 3)
        zc = z.clamp(min=0.5)
        u = Kf[:, 0, 0].unsqueeze(-1) * x / zc + Kf[:, 0, 2].unsqueeze(-1)
        v = Kf[:, 1, 1].unsqueeze(-1) * y / zc + Kf[:, 1, 2].unsqueeze(-1)
        dist = torch.sqrt(x * x + y * y + z * z)
        valid = ((z > 0.5) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
                 & (dist < 90.0))
        gu = (u / (W - 1) * 2 - 1).clamp(-2, 2)
        gv = (v / (H - 1) * 2 - 1).clamp(-2, 2)
        grid = torch.stack([gu, gv], -1)
        key = (int(K.data_ptr()), int(T_cam_ego.data_ptr()), B * N, H, W,
               float(K.reshape(-1)[0]), float(T_cam_ego.reshape(-1)[3]))
        idx = self._frustum_idx(valid, key)
        span = (self.D - 1) * self.D_STEP
        sm = getattr(self.depth_head, "sigma_map", None)
        num = torch.zeros(B, Cc, G2, device=ctx.device, dtype=ctx.dtype)
        den = torch.zeros(B, 1, G2, device=ctx.device, dtype=ctx.dtype)
        for i in range(B * N):
            ii = idx[i]
            if ii.numel() == 0:
                continue
            g = grid[i].index_select(0, ii).view(1, -1, 1, 2).to(ctx.dtype)
            cs = F.grid_sample(ctx[i:i + 1], g,
                               align_corners=False).squeeze(-1)
            p = F.grid_sample(dprob[i:i + 1, :1], g,
                              align_corners=False).view(1, 1, -1)
            mu = self.D_MIN + p * span
            if sm is None:
                sig = self.log_sigma.exp().clamp(0.3, 20.0).to(mu.dtype)
            else:
                # same zero-padding trap as the dense path: sigma = 0 outside
                # the image makes the exponent 0/0
                sig = F.grid_sample(sm[i:i + 1].to(ctx.dtype), g,
                                    align_corners=False).view(1, 1, -1)
                sig = sig.clamp(min=SIG_FLOOR)
            d_ = dist[i].index_select(0, ii).view(1, 1, -1)
            w = torch.exp(-((d_ - mu) ** 2) / (2 * sig * sig)) + 0.05
            num[i // N].index_add_(1, ii, (cs * w)[0].to(num.dtype))
            den[i // N].index_add_(1, ii, w[0].to(den.dtype))
        return (num / den.clamp(min=1e-4)).view(B, Cc, BEV_H, BEV_W)

    def bev_extra(self, bev):
        return _nanchk("after bev_extra", super().bev_extra(
            _nanchk("into bev_extra", bev)))

    def temporal_fuse(self, bev):
        return _nanchk("after temporal_fuse", super().temporal_fuse(
            _nanchk("into temporal_fuse", bev)))

    def depth_metres(self, dlog):
        """Predicted distance in metres, for the depth GT loss / demos."""
        p = dlog.softmax(1)[:, :1]
        return self.D_MIN + p * ((self.D - 1) * self.D_STEP)

    def depth_loss(self, dlog, depth_gt):
        """L1 in metres, plus a gradient-matching term.

        The inherited loss is a cross-entropy over 64 bins and does not apply to
        a regression head. Plain L1 alone gives the smooth, low-contrast depth
        the distilled model produces, so the image-gradient term is added: it
        penalises a prediction that is flatter than the GT across edges, which
        is exactly the sharpness the bin histogram used to provide."""
        d = self.depth_metres(dlog.flatten(0, 1)).squeeze(1)
        g = depth_gt.flatten(0, 1).to(d.dtype)
        m = (g > 0.5) & (g < 90.0)
        if not m.any():
            return d.sum() * 0.0
        l1 = ((d - g).abs() * m).sum() / m.sum()
        # First differences along both axes, on cells where both ends are valid.
        # MEASURED FAILURE: at weight 0.5 against an L1 that was itself divided
        # by 10, this term took over and depth MAE went 3.10 -> 31.37 m while the
        # predicted distance drifted +30 m as a block (r51). The term is scale
        # invariant -- it constrains contrast, not distance -- so it must stay a
        # small correction. It is now normalised by the GT gradient so a big step
        # edge cannot dominate, and weighted 0.05 against an unscaled L1.
        def grad(t):
            return (t[:, :, 1:] - t[:, :, :-1], t[:, 1:] - t[:, :-1])
        dx, dy = grad(d)
        gx, gy = grad(g)
        mx = m[:, :, 1:] & m[:, :, :-1]
        my = m[:, 1:] & m[:, :-1]
        rx = ((dx - gx).abs() / (gx.abs() + 1.0) * mx).sum() \
            / mx.sum().clamp(min=1)
        ry = ((dy - gy).abs() / (gy.abs() + 1.0) * my).sum() \
            / my.sum().clamp(min=1)
        return l1 / 10.0 + 0.05 * (rx + ry)

    def forward(self, *a, **k):
        """The base forward reshapes the depth output to [B,N,D,h,w] with
        D = 64. This head emits 2 channels, so out[1] is re-formed here as the
        per-pixel distance in metres -- which is what a regression head means
        and what the demo's depth panel wants anyway."""
        D0 = self.D
        try:
            self.D = 2                      # so the base view() matches
            out = list(super().forward(*a, **k))
        finally:
            self.D = D0
        d = out[1]                          # [B,N,2,h,w] softmax-able logits
        B, N = d.shape[:2]
        out[1] = self.depth_metres(d.reshape(B * N, 2, *d.shape[-2:])) \
            .reshape(B, N, 1, *d.shape[-2:])
        return tuple(out)


class DepthSegIPMNetV49(DepthSegIPMNetV48):
    """v49 (r50): BEV seg decoder at HALF the internal width.

    Measured on the deployed graph: the decoder is 11.51 ms of the 31.8 ms INT8
    engine, and 9 output classes did not need 64/192/320 channels -- halving
    them is 4.10M -> 1.06M parameters and 7.72 -> 3.28 ms at 800x500 (2.35x).
    Nothing else changes: the SHARED 96-channel BEV feature that the other
    eleven tasks read is untouched, so this round isolates the seg-only cost.
    The decoder is shape-mismatched against v48 checkpoints and is therefore
    re-initialised on warm start, which is the point of the experiment: how
    many epochs it takes to recover.
    """

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        n_cls = self.dec.out[-1].out_channels
        self.dec = LaneDecED(BEV_CH, n_cls, w=0.5)



_NAN_PROBE = bool(os.environ.get("METEOR_NAN_PROBE"))


def _nanchk(tag, t):
    """Name the first tensor in the BEV path that goes non-finite.

    r51 and r52 both threw away ~2 % of their steps to a non-finite loss, and
    the loss is accumulated over 26 sites so its value identifies nothing. The
    output scan added to train.py narrowed it down: every BEV-space output is
    100 % non-finite while every image-space output is clean, so the corruption
    is between ctx and the fused BEV. This walks that stretch. Costs a device
    sync per call, so it is off unless METEOR_NAN_PROBE is set.
    """
    if not _NAN_PROBE or not torch.is_tensor(t):
        return t
    d = t.detach()
    fin = torch.isfinite(d)
    if not bool(fin.all()):
        ok = d[fin]
        mx = float(ok.abs().max()) if ok.numel() else float("nan")
        print(f"[nan] {tag}: {int((~fin).sum())}/{d.numel()} non-finite, "
              f"largest finite magnitude {mx:.4g}", flush=True)
    return t


class DepthSegIPMNetV51(DepthSegIPMNetV50):
    """v50 with the BEV lift run on a HALF-resolution grid, output unchanged.

    The lift is the single biggest cost in the engine. Measured on the INT8
    engine, the fused block that holds the depth head and the lift is 12.11 of
    30.7 ms, and a PyTorch breakdown puts 3.67 ms of that on the lift against
    1.24 ms on the depth head -- so the lift alone is roughly 9 ms, 29 % of the
    whole network. Its cost is set by the number of (camera, BEV cell) pairs,
    which falls 4x when the grid goes from 0.2 m to 0.4 m: measured 3.67 -> 1.45
    ms, 2.53x.

    Nothing downstream changes. The lifted feature is bilinearly resampled back
    to the full grid immediately, so every head sees exactly the tensor shape it
    saw before and no other module needs touching. What is actually lost is
    lift-time spatial precision: two 0.2 m cells that used to sample the image
    separately now share one sample. Thin classes (laneline is 0.12 IoU already)
    are where that will show, and it has to be measured, not assumed.
    """

    # Overridable: the Orin profile puts ~51 % of the engine in the lift's
    # Myelin cluster, and its cost scales with the (camera, cell) pair count.
    # DIV 2 = 0.4 m (default); METEOR_LIFT_DIV=4 lifts at 0.8 m -> quarter the
    # pairs. The 0.2 -> 0.4 move cost -0.003 mIoU without retraining, so 4 is
    # the same bet one step further -- priced before any round is spent on it.
    LIFT_DIV = int(os.environ.get("METEOR_LIFT_DIV", "2"))

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        d = self.LIFT_DIV
        self.lift_h, self.lift_w = BEV_H // d, BEV_W // d
        xs = torch.linspace(BEV_XH - BEV_RES * d / 2,
                            -BEV_XH + BEV_RES * d / 2, self.lift_h)
        ys = torch.linspace(BEV_YH - BEV_RES * d / 2,
                            -BEV_YH + BEV_RES * d / 2, self.lift_w)
        gx, gy = torch.meshgrid(xs, ys, indexing="ij")
        n = self.lift_h * self.lift_w
        self.register_buffer("bev_pts",
                             torch.stack([gx.reshape(-1), gy.reshape(-1),
                                          torch.zeros(n), torch.ones(n)], 1),
                             persistent=False)

    def project_bev(self, dprob, ctx, K, T_cam_ego, B, N, H, W):
        import bevlane.model as _M
        oh, ow = _M.BEV_H, _M.BEV_W
        _M.BEV_H, _M.BEV_W = self.lift_h, self.lift_w
        try:
            bev = super().project_bev(dprob, ctx, K, T_cam_ego, B, N, H, W)
        finally:
            _M.BEV_H, _M.BEV_W = oh, ow
        return F.interpolate(bev, size=(oh, ow), mode="bilinear",
                             align_corners=False)



class DepthSegIPMNetV52(DepthSegIPMNetV48):
    """v48's 64-bin classification depth, lifted on the 0.4 m grid.

    The regression depth head (v50) was adopted to make the lift cheap: it
    samples ONE channel per (camera, cell) pair instead of the 64-bin
    histogram, worth 11.5 ms of PyTorch forward. But it has cost two things.
    It drifts -- a uniform shift of all depths is very nearly a flat direction
    of the BEV objective, so every round so far has walked depth MAE from
    3.1 m out to 13-28 m and needed the head swapped back afterwards. And it is
    blurrier: image-gradient magnitude 0.424 against the 64-bin head's 0.547,
    where the LiDAR GT is 0.644.

    Once the lift runs on the 0.4 m grid there are 4x fewer pairs, so the
    64-bin sampling costs 4x less than it used to. This variant exists to
    measure whether that makes classification affordable again.
    """

    LIFT_DIV = int(os.environ.get("METEOR_LIFT_DIV", "2"))

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        d = self.LIFT_DIV
        self.lift_h, self.lift_w = BEV_H // d, BEV_W // d
        xs = torch.linspace(BEV_XH - BEV_RES * d / 2,
                            -BEV_XH + BEV_RES * d / 2, self.lift_h)
        ys = torch.linspace(BEV_YH - BEV_RES * d / 2,
                            -BEV_YH + BEV_RES * d / 2, self.lift_w)
        gx, gy = torch.meshgrid(xs, ys, indexing="ij")
        n = self.lift_h * self.lift_w
        self.register_buffer("bev_pts",
                             torch.stack([gx.reshape(-1), gy.reshape(-1),
                                          torch.zeros(n), torch.ones(n)], 1),
                             persistent=False)

    def project_bev(self, dprob, ctx, K, T_cam_ego, B, N, H, W):
        import bevlane.model as _M
        oh, ow = _M.BEV_H, _M.BEV_W
        _M.BEV_H, _M.BEV_W = self.lift_h, self.lift_w
        try:
            bev = super().project_bev(dprob, ctx, K, T_cam_ego, B, N, H, W)
        finally:
            _M.BEV_H, _M.BEV_W = oh, ow
        return F.interpolate(bev, size=(oh, ow), mode="bilinear",
                             align_corners=False)



class DepthSegIPMNetV53(DepthSegIPMNetV52):
    """v52 with the lift grid at 0.3 m instead of 0.4 m.

    Thin-class width tracks the lift resolution: laneline covered 2.71x the GT
    area with the 0.2 m grid (r53) and 2.96-3.72x with 0.4 m (r54 onward). The
    training objective is NOT what widens it -- backprop through each seg term
    puts a POSITIVE gradient on the ring of cells just outside a true line
    (+5.3e-6 summed, i.e. every term pushes thinner), so the width is a limit of
    how sharp a boundary the feature can express, and that is set by how finely
    the lift samples. 0.3 m sits between the two measured points.

    The grid no longer has to divide the output grid evenly -- the lifted
    feature is resampled to 800x500 either way -- so this is expressed in metres
    rather than as an integer divisor.
    """

    LIFT_RES = 0.3

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        r = self.LIFT_RES
        self.lift_h = int(round(2 * BEV_XH / r))
        self.lift_w = int(round(2 * BEV_YH / r))
        xs = torch.linspace(BEV_XH - r / 2, -BEV_XH + r / 2, self.lift_h)
        ys = torch.linspace(BEV_YH - r / 2, -BEV_YH + r / 2, self.lift_w)
        gx, gy = torch.meshgrid(xs, ys, indexing="ij")
        n = self.lift_h * self.lift_w
        self.register_buffer("bev_pts",
                             torch.stack([gx.reshape(-1), gy.reshape(-1),
                                          torch.zeros(n), torch.ones(n)], 1),
                             persistent=False)



class DepthSegIPMNetV54(DepthSegIPMNetV52):
    """v52 with the two aggressive width cuts, both dialable from outside.

    This exists because the latency target cannot be reached by building the
    engine better. A static FLOP pass over the exported v52 graph gives 3,405
    GFLOP per frame, which at the measured 19.24 ms is 177 TOPS -- 48 % of the
    workstation-GPU INT8 peak. 10 ms would need 93 % of peak, and real graphs do not
    reach that. FLOPs have to come down, and there are exactly two places worth
    cutting:

      depth_head  1091 GFLOP, 32.0 % of the graph. Four stages of 3x3 convs at
                  256/256/192/128, run over 8 cameras at 108x192. DEPTH_MULT
                  0.5 removes about 800 GFLOP -- 24 % of the whole graph -- from
                  this module alone.
      backbone    layer3+layer4 are 37 % of the parameters. resnet34 -> resnet18
                  keeps every channel count identical (64/128/256/512) and only
                  drops block counts (layer3 6->2, layer4 3->2), so nothing
                  downstream needs to change.

    The reason to expect the depth head to survive the cut: `bev_pts` has a
    single unique z = 0.0, so the lift is flat-ground and depth never moves a
    sample point -- it only re-weights which camera wins a BEV cell. That is
    also why the v50 regression head and the v48/v52 classification head
    measured neutral on BEV (mIoU delta -0.0001 over three measurements). A
    four-stage 256-channel tower to produce a per-pixel camera weight is very
    likely more than the job needs. "Likely" is not "measured": BEV Seg,
    3D BBox and E2E all get measured before and after, one head at a time.

    Subclass and set the two attributes to go further or less far:

        class MyTiny(DepthSegIPMNetV54):
            DEPTH_MULT = 0.25
            BACKBONE = "resnet18"
    """

    DEPTH_MULT = 0.5             # width multiplier on the depth tower
    BACKBONE = "resnet18"        # None keeps whatever the parent built

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        if self.BACKBONE:
            rn = getattr(torchvision.models, self.BACKBONE)(
                weights="IMAGENET1K_V1")
            # channel counts are identical across resnet18/34, so the lateral
            # convs and everything after them are untouched
            self.stem = nn.Sequential(rn.conv1, rn.bn1, rn.relu, rn.maxpool)
            self.layer1, self.layer2 = rn.layer1, rn.layer2
            self.layer3, self.layer4 = rn.layer3, rn.layer4
        m = self.DEPTH_MULT
        if m and m != 1.0:
            w = [max(16, int(round(c * m))) for c in (256, 256, 192, 128)]
            self.depth_head = nn.Sequential(
                ConvBlock(160, w[0]), ConvBlock(w[0], w[1]),
                ConvBlock(w[1], w[2]), ConvBlock(w[2], w[3]),
                nn.Conv2d(w[3], self.D, 1))


class DepthSegIPMNetV55(DepthSegIPMNetV54):
    """v54's depth-tower cut ONLY -- resnet34 backbone kept.

    v54 changed two things at once and lost 36 % of veh R50 and 56 % of turning
    ADEc for -10.8 % of latency. Which of the two cuts did that is not knowable
    from that run, and it decides everything: the depth tower is 787 of the 984
    GFLOP v54 removed (80 %), so if the depth cut is the expensive one there is
    very little left worth taking."""
    DEPTH_MULT = 0.5
    BACKBONE = None


class DepthSegIPMNetV56(DepthSegIPMNetV54):
    """v54's backbone cut ONLY -- full-width depth tower kept.

    resnet34 -> resnet18 is 197 GFLOP, a fifth of what v54 removed. Cheap if it
    holds the priority tasks; pointless if it does not."""
    DEPTH_MULT = 1.0
    BACKBONE = "resnet18"


class DepthSegIPMNetV64r50(DepthSegIPMNetV55):
    """R7 phase 1: ResNet-50 backbone, everything else = v55 (light line).

    The far-range plan (R7) has two axes -- backbone arch and input
    resolution -- and the v55/v56 lesson demands one axis per run. This class
    is the ARCH axis alone: bottleneck features at the same 432x768 input and
    the same stride-4 fusion, so any accuracy move is attributable to the
    representation, not to extra pixels. r50 laterals differ (256/512/1024/
    2048 vs 64/128/256/512), so the four lat convs rebuild; everything from
    `fuse` on is untouched and carries over from a v55-line checkpoint.
    +14 % backbone FLOP at equal resolution.
    """
    BACKBONE = None            # v54's swap logic does not fit r50; done here

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        rn = torchvision.models.resnet50(weights="IMAGENET1K_V1")
        self.stem = nn.Sequential(rn.conv1, rn.bn1, rn.relu, rn.maxpool)
        self.layer1, self.layer2 = rn.layer1, rn.layer2
        self.layer3, self.layer4 = rn.layer3, rn.layer4
        fc = self.lat1.out_channels
        self.lat1 = nn.Conv2d(256, fc, 1)
        self.lat2 = nn.Conv2d(512, fc, 1)
        self.lat3 = nn.Conv2d(1024, fc, 1)
        self.lat4 = nn.Conv2d(2048, fc, 1)


class DepthSegIPMNetV52s8(DepthSegIPMNetV52):
    """R7 解像度軸 (2026-08-14, r34 のまま): x2 入力 (dataset --img-scale 2) を
    stride-8 で融合。特徴グリッドは 108x192 のままなので depth/seg2d GT・
    リフト・BEV 以降は無変更、lat2/3/4 の入力チャンネルも一致するため
    成熟した r34 系列の重みを 100% 引き継げる (lat1 のみ未使用)。
    RepVGG 実験 (r68) は seg -0.05 / veh R50 -0.12 で却下 — 遠方精度の
    本命レバーは backbone アーキではなく解像度、という判断。"""
    FUSE_STRIDE = 8

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        # lat1 (stride-4 タップ) は FUSE_STRIDE=8 では使わない。モジュールに
        # 残すと DDP が未使用パラメータとして落ちる (find_unused_parameters は
        # 履歴の多重 forward と両立しない) ので、ここで外す。
        del self.lat1


class DepthSegIPMNetV52r50(DepthSegIPMNetV52):
    """R7 phase 1, LOCAL (accuracy) line: ResNet-50 backbone on the full v52
    recipe. Per the 2026-08-12 decision the light line stays inside the Orin
    budget, so the backbone axis lands here first. Same 432x768 input, same
    stride-4 fusion -- arch axis only; laterals rebuild for the bottleneck
    channel widths, everything from `fuse` on carries over from a v52-line
    checkpoint (r67 best)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        rn = torchvision.models.resnet50(weights="IMAGENET1K_V1")
        self.stem = nn.Sequential(rn.conv1, rn.bn1, rn.relu, rn.maxpool)
        self.layer1, self.layer2 = rn.layer1, rn.layer2
        self.layer3, self.layer4 = rn.layer3, rn.layer4
        fc = self.lat1.out_channels
        self.lat1 = nn.Conv2d(256, fc, 1)
        self.lat2 = nn.Conv2d(512, fc, 1)
        self.lat3 = nn.Conv2d(1024, fc, 1)
        self.lat4 = nn.Conv2d(2048, fc, 1)


class DepthSegIPMNetV52r50s8(DepthSegIPMNetV52r50):
    """R7 resolution axis (upsample-first): expects x2 input (864x1536,
    dataset img_scale=2) and fuses at STRIDE 8 (layer2/3/4 laterals), so the
    feature grid lands on the same 108x192 as today and every 2D GT, the
    lift, and the BEV stack are untouched. With upsampled jpgs this measures
    the stride-ratio/compute effect only; the true-resolution re-ingest
    swaps the data later without touching the model again. lat1 is kept but
    unused (checkpoint compatibility)."""

    def image_feats(self, imgs):
        B, N, _, H, W = imgs.shape
        x0 = self.stem(imgs.reshape(B * N, 3, H, W))
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        sz = x2.shape[-2:]
        up = lambda t: F.interpolate(t, size=sz, mode="bilinear",
                                     align_corners=False)
        f = self.lat2(x2) + up(self.lat3(x3)) + up(self.lat4(x4))
        return self.fuse(f)


class _RepVGGStage(nn.Module):
    """Deploy-form RepVGG stage: N plain conv3x3+BN+ReLU blocks, first one
    strided. Latency-ladder用 (学習時は timm の分岐形から再パラメータ化する
    前提で、ここでは Orin での Δms 計測に使うデプロイ形のみ)。"""

    def __init__(self, cin, cout, n, stride=2):
        super().__init__()
        L = []
        for i in range(n):
            L += [nn.Conv2d(cin if i == 0 else cout, cout, 3,
                            stride=stride if i == 0 else 1, padding=1,
                            bias=False), nn.BatchNorm2d(cout),
                  nn.ReLU(inplace=True)]
        self.f = nn.Sequential(*L)

    def forward(self, x):
        return self.f(x)


class DepthSegIPMNetV64rv(DepthSegIPMNetV55):
    """Orin backbone efficiency ladder: deploy-form RepVGG-ish backbone
    (pure dense 3x3 stacks, no branches/1x1/DW) behind the same FPN taps.
    WIDTHS/BLOCKS are class attrs so A1/A2 variants subclass in two lines.
    Latency measurement first; training (with reparam + ImageNet init via
    timm) only if the ladder says the ms is worth it."""
    WIDTHS = (48, 48, 96, 192, 384)     # ~A1 相当
    BLOCKS = (1, 2, 4, 14, 1)

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        w, b = self.WIDTHS, self.BLOCKS
        self.stem = _RepVGGStage(3, w[0], b[0], stride=2)
        self.layer1 = _RepVGGStage(w[0], w[1], b[1], stride=2)   # s4
        self.layer2 = _RepVGGStage(w[1], w[2], b[2], stride=2)   # s8
        self.layer3 = _RepVGGStage(w[2], w[3], b[3], stride=2)   # s16
        self.layer4 = _RepVGGStage(w[3], w[4], b[4], stride=2)   # s32
        fc = self.lat1.out_channels
        self.lat1 = nn.Conv2d(w[1], fc, 1)
        self.lat2 = nn.Conv2d(w[2], fc, 1)
        self.lat3 = nn.Conv2d(w[3], fc, 1)
        self.lat4 = nn.Conv2d(w[4], fc, 1)


class DepthSegIPMNetV64rvA2(DepthSegIPMNetV64rv):
    WIDTHS = (64, 64, 128, 256, 512)
    BLOCKS = (1, 2, 4, 14, 1)


class DepthSegIPMNetV64rvB0(DepthSegIPMNetV64rv):
    """RepVGG-B0 相当 + 幅微増: r34 超の容量を密 3x3 で。"""
    WIDTHS = (64, 96, 192, 384, 768)
    BLOCKS = (1, 4, 6, 16, 1)


class DepthSegIPMNetV64rvB1(DepthSegIPMNetV64rv):
    """RepVGG-B1 相当: 明確に r34/r50 超の容量。"""
    WIDTHS = (64, 128, 256, 512, 1024)
    BLOCKS = (1, 4, 6, 16, 1)


class DepthSegIPMNetV64r34w(DepthSegIPMNetV55):
    """幅広 r34: basic block のまま幅 x1.25 (80/160/320/640)。torchvision の
    resnet34 を width スケールで自前構築 (ImageNet init なし — ladder 用)。"""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        from torchvision.models.resnet import BasicBlock, ResNet
        rn = ResNet(BasicBlock, [3, 4, 6, 3], width_per_group=64)
        # torchvision ResNet は width スケール引数を basic に持たないため手動:
        def make(cin, cout, n, stride):
            return _RepVGGStage(cin, cout, n, stride)  # 3x3 スタックで代替
        self.stem = nn.Sequential(nn.Conv2d(3, 80, 7, 2, 3, bias=False),
                                  nn.BatchNorm2d(80), nn.ReLU(inplace=True),
                                  nn.MaxPool2d(3, 2, 1))
        self.layer1 = _RepVGGStage(80, 80, 3, stride=1)
        self.layer2 = _RepVGGStage(80, 160, 4, stride=2)
        self.layer3 = _RepVGGStage(160, 320, 6, stride=2)
        self.layer4 = _RepVGGStage(320, 640, 3, stride=2)
        fc = self.lat1.out_channels
        self.lat1 = nn.Conv2d(80, fc, 1)
        self.lat2 = nn.Conv2d(160, fc, 1)
        self.lat3 = nn.Conv2d(320, fc, 1)
        self.lat4 = nn.Conv2d(640, fc, 1)


class DepthSegIPMNetV52rvgg(DepthSegIPMNetV52):
    """R7 arch axis, take 2 (user decision 2026-08-13): RepVGG-A2 backbone
    via timm — training form keeps the 3x3+1x1+identity branches (where
    RepVGG earns its accuracy); export reparameterizes to the dense-3x3
    stack the Orin ladder priced at ~+5 ms for 1.55x r34 backbone FLOPs
    (26.8M params). ImageNet pretrained.

    Integration: timm stages are MAPPED onto the existing stem/layer1..4
    interface (stem := timm stem+stage0 at s4, layer1 := Identity,
    layer2..4 := stages 1..3) so the base image_feats and every override
    stacked on it (_last_f for the 2D det head, the TL-stem residual)
    run untouched. Only the laterals change width."""
    TIMM_NAME = "repvgg_a2"

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        import timm
        rv = timm.create_model(self.TIMM_NAME, pretrained=True)
        self.stem = nn.Sequential(rv.stem, rv.stages[0])   # -> 96ch, s4
        self.layer1 = nn.Identity()                         # x1 = stem out
        self.layer2 = rv.stages[1]                          # 192ch, s8
        self.layer3 = rv.stages[2]                          # 384ch, s16
        self.layer4 = rv.stages[3]                          # 1408ch, s32
        fc = self.lat1.out_channels
        self.lat1 = nn.Conv2d(96, fc, 1)
        self.lat2 = nn.Conv2d(192, fc, 1)
        self.lat3 = nn.Conv2d(384, fc, 1)
        self.lat4 = nn.Conv2d(1408, fc, 1)


class DepthSegIPMNetV63b(DepthSegIPMNetV55):
    """R6 axis 2: depth bins 64 -> 32, everything else = v55.

    The bin geometry keeps the 1..80 m range (D_STEP 1.25 -> 2.5), so the
    lift's flat-ground camera-weighting job is unchanged in coverage and only
    halved in resolution. Every consumer reads self.D / D_STEP, so the change
    is these three attributes -- the depth loss bins GT with the same fields.
    v63a (ctx 96 -> 64) is the OTHER axis and must ride a separate run:
    change one axis per run, or the cost of the wrong cut is unattributable.
    """
    D = 32
    D_MIN, D_STEP = 1.0, 2.5


class DepthSegIPMNetV55rvgg(DepthSegIPMNetV55):
    """Light-line arch axis (v68 candidate, pre-registered 2026-08-13):
    RepVGG-A2 backbone on the v55 light recipe via the same stage-mapping
    trick as v52rvgg. Orin ladder priced the deploy form at ~+5 ms for
    1.55x r34 backbone FLOPs; adoption is conditional on the LOCAL r68
    round demonstrating the accuracy side (out/rvgg_light_verdict.txt)."""
    TIMM_NAME = "repvgg_a2"

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        import timm
        rv = timm.create_model(self.TIMM_NAME, pretrained=True)
        self.stem = nn.Sequential(rv.stem, rv.stages[0])
        self.layer1 = nn.Identity()
        self.layer2 = rv.stages[1]
        self.layer3 = rv.stages[2]
        self.layer4 = rv.stages[3]
        fc = self.lat1.out_channels
        self.lat1 = nn.Conv2d(96, fc, 1)
        self.lat2 = nn.Conv2d(192, fc, 1)
        self.lat3 = nn.Conv2d(384, fc, 1)
        self.lat4 = nn.Conv2d(1408, fc, 1)


MODELS = {"v1": IPMSegNet, "v2": IPMSegNetV2, "v3s": IPMSegNetV3,
          "lss": LSSDepthNet, "v8": DepthGatedIPMNet, "v13": DepthSegIPMNet,
          "v13d": DepthSegIPMNetS4, "v14d": DepthSegIPMNetV14,
          "v15": DepthSegIPMNetV15, "v16": DepthSegIPMNetV16,
          "v17": DepthSegIPMNetV17, "v18": DepthSegIPMNetV18,
          "v19": DepthSegIPMNetV19, "v20": DepthSegIPMNetV20,
          "v21": DepthSegIPMNetV21, "v22": DepthSegIPMNetV22,
          "v23": DepthSegIPMNetV23, "v24": DepthSegIPMNetV24,
          "v25": DepthSegIPMNetV25, "v26": DepthSegIPMNetV26, "v27": DepthSegIPMNetV27, "v28": DepthSegIPMNetV28, "v29": DepthSegIPMNetV29, "v30": DepthSegIPMNetV30, "v31": DepthSegIPMNetV31, "v32": DepthSegIPMNetV32, "v33": DepthSegIPMNetV33, "v34": DepthSegIPMNetV34, "v35": DepthSegIPMNetV35, "v36": DepthSegIPMNetV36, "v37": DepthSegIPMNetV37, "v38": DepthSegIPMNetV38, "v39": DepthSegIPMNetV39, "v40": DepthSegIPMNetV40, "v41": DepthSegIPMNetV41, "v42": DepthSegIPMNetV42, "v43": DepthSegIPMNetV43, "v44": DepthSegIPMNetV44, "v45": DepthSegIPMNetV45, "v46": DepthSegIPMNetV46, "v47": DepthSegIPMNetV47,
          "v48": DepthSegIPMNetV48, "v49": DepthSegIPMNetV49, "v50": DepthSegIPMNetV50,
          "v51": DepthSegIPMNetV51,
          "v52": DepthSegIPMNetV52,
          "v53": DepthSegIPMNetV53,
          "v54": DepthSegIPMNetV54,
          "v55": DepthSegIPMNetV55,
          "v56": DepthSegIPMNetV56,
          "v63b": DepthSegIPMNetV63b,
          "v52s8": DepthSegIPMNetV52s8,
          "v52r50": DepthSegIPMNetV52r50,
          "v64rv": DepthSegIPMNetV64rv,
          "v64rva2": DepthSegIPMNetV64rvA2,
          "v64rvb0": DepthSegIPMNetV64rvB0,
          "v64rvb1": DepthSegIPMNetV64rvB1,
          "v64r34w": DepthSegIPMNetV64r34w,
          "v52r50s8": DepthSegIPMNetV52r50s8,
          "v52rvgg": DepthSegIPMNetV52rvgg,
          "v55rvgg": DepthSegIPMNetV55rvgg,
          "v64r50": DepthSegIPMNetV64r50}

"""IPM-based multi-camera BEV segmentation network.

Camera features (ResNet18, stride-8) are sampled onto a BEV grid via
ground-plane (z=0) projection using known intrinsics/extrinsics, fused across
cameras, and decoded by a small BEV U-Net into semantic logits.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

N_CLASSES = 9
FEAT_GRID = 200        # legacy square feature grid (0.3 m)
BEV_SIZE = 400         # legacy square output grid (0.15 m)
BEV_HALF = 30.0
# rectangular long-range BEV (v3s/v8): +-80 m fwd, +-50 m lateral @ 0.2 m
BEV_XH, BEV_YH, BEV_RES = 80.0, 50.0, 0.2
BEV_H, BEV_W = int(2 * BEV_XH / BEV_RES), int(2 * BEV_YH / BEV_RES)  # 800x500


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
        tgt = ((depth_gt - self.D_MIN) / self.D_STEP).round().long()
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

    def __init__(self, n_cams=6, feat_ch=160, ctx_ch=96):
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
        b = ((dist - self.D_MIN) / self.D_STEP).clamp(0, self.D - 1 - 1e-4)
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
        tgt = ((depth_gt - self.D_MIN) / self.D_STEP).round().long()
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

    def __init__(self, n_cams=8, feat_ch=160, ctx_ch=96, n_seg=N_SEG):
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

    def image_feats(self, imgs):
        B, N, _, H, W = imgs.shape
        x1 = self.layer1(self.stem(imgs.reshape(B * N, 3, H, W)))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        sz = x1.shape[-2:]
        up = lambda t: F.interpolate(t, size=sz, mode="bilinear", align_corners=False)
        f = self.lat1(x1) + up(self.lat2(x2)) + up(self.lat3(x3)) + up(self.lat4(x4))
        return self.fuse(f)

    def forward(self, imgs, K, T_cam_ego):
        B, N, _, H, W = imgs.shape
        f = self.image_feats(imgs)                     # [BN,C,H/4,W/4]
        seg2d = self.seg_head(f)                        # [BN,n_seg,H/4,W/4]
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
        b = ((dist - self.D_MIN) / self.D_STEP).clamp(0, self.D - 1 - 1e-4)
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
        tgt = ((depth_gt - self.D_MIN) / self.D_STEP).round().long()
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

    def depth_loss(self, dlog, depth_gt):
        logits = dlog.flatten(0, 1)                       # [BN,D,h,w]
        gt = depth_gt.flatten(0, 1)                       # [BN,h,w]
        tgt = ((gt - self.D_MIN) / self.D_STEP).round().long()
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
        ce_px = F.cross_entropy(logits, tgt, ignore_index=-1,
                                label_smoothing=0.05, reduction="none")
        ce = (ce_px * wpx)[valid].mean() if valid.any() \
            else logits.sum() * 0.0
        # L1 on expected depth (metres) -> metric accuracy, sharper distributions
        prob = logits.softmax(1)
        bins = (torch.arange(self.D, device=logits.device, dtype=prob.dtype)
                * self.D_STEP + self.D_MIN).view(1, -1, 1, 1)
        exp_d = (prob * bins).sum(1)
        if valid.any():
            l1 = (exp_d - gt).abs()[valid].mean()
        else:
            l1 = exp_d.sum() * 0
        return ce + 0.1 * l1


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
        seg2d = self.seg_head(f)
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
        b = ((dist - self.D_MIN) / self.D_STEP).clamp(0, self.D - 1 - 1e-4)
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
            nn.Conv2d(96, 128, 3, stride=DET_S, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            ConvBlock(128, 128))
        self.hm_head = nn.Conv2d(128, 2, 1)
        self.reg_head = nn.Conv2d(128, 6, 1)
        nn.init.constant_(self.hm_head.bias, -2.19)   # focal init (p~0.1)

    def project_bev(self, dprob, ctx, K, T_cam_ego, B, N, H, W):
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
        b = ((dist - self.D_MIN) / self.D_STEP).clamp(0, self.D - 1 - 1e-4)
        b0 = b.floor().long()
        fr = (b - b0.float()).unsqueeze(1)
        w0 = torch.gather(prob_s, 1, b0.unsqueeze(1))
        w1 = torch.gather(prob_s, 1, (b0 + 1).clamp(max=self.D - 1).unsqueeze(1))
        wgt = (w0 * (1 - fr) + w1 * fr) + 0.05
        wgt = wgt * valid.unsqueeze(1).to(wgt.dtype)
        num = (ctx_s.view(B, N, Cc, G2) * wgt.view(B, N, 1, G2)).sum(1)
        den = wgt.view(B, N, 1, G2).sum(1).clamp(min=1e-4)
        return (num / den).view(B, Cc, BEV_H, BEV_W)

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
        seg2d = self.seg_head(f)
        dlog = self.depth_head(self.depth_up(f))
        dprob = self.sharpen_dprob(dlog.softmax(1))
        ctx = self.ctx(f)
        bev = self.bev_extra(
            self.project_bev(dprob, ctx, K, T_cam_ego, B, N, H, W))
        self._last_bev = bev
        bev = self.temporal_fuse(bev)
        self._fused_bev = bev          # consumed by ego / occ / traj heads
        det = self.det_stem(self.det_input())
        self._det_feat = det
        lane_bev = self.lane_input()
        fh2, fw2 = dlog.shape[-2:]
        sh, sw = seg2d.shape[-2:]
        rg_out = self.reg_head(det)
        self._det_reg = rg_out          # v33 traj head reads sin/cos yaw
        return (self.dec(lane_bev), dlog.view(B, N, self.D, fh2, fw2),
                seg2d.view(B, N, seg2d.shape[1], sh, sw),
                self.hm_head(det), rg_out)

    @staticmethod
    def build_det_targets(boxes, nbox, device, dtype=torch.float32):
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
                r = (80.0 - xe) / DET_RES
                c = (50.0 - ye) / DET_RES
                ri, ci = int(r), int(c)
                if not (0 <= ri < DET_H and 0 <= ci < DET_W):
                    continue
                rad = min(max(2.0, 0.7 * max(l, w) / DET_RES / 2), 4.0)
                g = torch.exp(-(((ys - r) ** 2).view(-1, 1)
                                + ((xs - c) ** 2).view(1, -1)) / (2 * rad ** 2))
                ch = 0 if cls < 1.5 else 1
                hm[bi, ch] = torch.maximum(hm[bi, ch], g)
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

    def boxdet_loss(self, hm, reg, boxes, nbox):
        hm_t, reg_t, m = self.build_det_targets(boxes, nbox, hm.device,
                                                torch.float32)
        p = hm.float().sigmoid().clamp(1e-4, 1 - 1e-4)
        pos = (hm_t > 0.99).float()
        neg_w = (1 - hm_t) ** 4
        # near-range VRU emphasis: VRU channel x2.5, positives within 20 m
        # of ego x2 (user: near bicycles/bikes/pedestrians are weak)
        if getattr(self, "_det_posw", None) is None \
                or self._det_posw.device != hm.device:
            rr = torch.arange(DET_H, device=hm.device).view(-1, 1)
            cc = torch.arange(DET_W, device=hm.device).view(1, -1)
            xe = 80.0 - rr * DET_RES
            ye = 50.0 - cc * DET_RES
            r = (xe ** 2 + ye ** 2).sqrt()
            # near-range recall boost, per class: the veh boost is softened
            # (x3 overfired -> near duplicate/phantom FPs); VRU keeps x3
            near_veh = 1.0 + 0.25 * (r < 20.0).float() + 0.25 * (r < 12.0).float()
            near_vru = 1.0 + (r < 20.0).float() + (r < 12.0).float()
            near = torch.stack([near_veh, near_vru])
            cw = torch.tensor([2.0, 5.0], device=hm.device).view(2, 1, 1)
            # far positives are unresolvable at 768x432 (a 60 m pedestrian is
            # ~10 px); full-weight unlearnable positives push the focal loss
            # to suppress confidence everywhere -> damp them instead
            damp = torch.stack([torch.where(r > 50.0, 0.3, 1.0),
                                torch.where(r > 40.0, 0.2, 1.0)])
            # laterally distant objects are out of scope -> nearly ignore
            damp = damp * torch.where(ye.abs() > 15.0, 0.2, 1.0)
            self._det_posw = (near * cw * damp).unsqueeze(0)
        floss = -(pos * self._det_posw * (1 - p) ** 2 * p.log()
                  + (1 - pos) * neg_w * p ** 2 * (1 - p).log()).sum() \
            / (pos * self._det_posw).sum().clamp(min=1)
        # yaw channels (sin/cos) x3: orientation error is the weakest output
        if getattr(self, "_reg_cw", None) is None \
                or self._reg_cw.device != hm.device:
            self._reg_cw = torch.tensor([1., 1., 1., 1., 3., 3.],
                                        device=hm.device).view(1, 6, 1, 1)
        rloss = (torch.abs(reg.float() - reg_t) * m * self._reg_cw).sum() \
            / m.sum().clamp(min=1) / 10
        return floss + rloss

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
                xe = 80.0 - float(r) * DET_RES
                ye = 50.0 - float(c) * DET_RES
                l = float(o[2].exp())
                w = float(o[3].exp())
                yaw = float(torch.atan2(o[4], o[5]))
                boxes.append((int(cls[j]), float(sc[j]), xe, ye, l, w, yaw))
            out.append(boxes)
        return out


DET2D_S = 4                    # 2D det grid stride on the cached image (108x192)
N_DET2D = 10                   # fastlabel 10-class instance taxonomy


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
            nn.Conv2d(96, 64, 3, stride=4, padding=1, bias=False),
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
            nn.Conv2d(96, 128, 3, stride=4, padding=1, bias=False),
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
        d4 = self.det2d_stem(f)
        d8 = self.det2d_d8(d4)
        d16 = self.det2d_d16(d8)
        hms, regs = [], []
        for d, hh, rr in ((d4, self.hm2d_head, self.reg2d_head),
                          (d8, self.hm2d_head8, self.reg2d_head8),
                          (d16, self.hm2d_head16, self.reg2d_head16)):
            hm = hh(d)
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
            nn.Conv2d(96, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            ConvBlock(128, 192))
        self.occ_head = nn.Conv2d(192, OCC_Z * OCC_C, 1)

    def forward(self, imgs, K, T_cam_ego, v0=None):
        out = super().forward(imgs, K, T_cam_ego, v0)
        crop = self.occ_input()[:, :, 200:600, 50:450]
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
                ri = int((80.0 - float(xe)) / DET_RES)
                ci = int((50.0 - float(ye)) / DET_RES)
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
            nn.Conv2d(192, 96, 1, bias=False), nn.BatchNorm2d(96),
            nn.ReLU(inplace=True), ConvBlock(96, 96))
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
    def __init__(self, cin, n_cls):
        super().__init__()
        self.skip = ConvBlock(cin, 64)
        self.d1 = nn.Sequential(
            nn.Conv2d(cin, 192, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(192), nn.ReLU(inplace=True), ConvBlock(192, 192))
        self.d2 = nn.Sequential(
            nn.Conv2d(192, 320, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(320), nn.ReLU(inplace=True), ConvBlock(320, 320))
        self.u1 = nn.Conv2d(320, 192, 1)
        self.m1 = ConvBlock(192, 192)
        self.u2 = nn.Conv2d(192, 64, 1)
        self.out = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, n_cls, 1))

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
        self.dec = LaneDecED(96, n_cls)
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
            nn.Conv2d(96, 128, 3, stride=2, padding=1, bias=False),
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
    def stat_loss(stat, boxes, nbox, traj, tvalid):
        """BCE at GT centres; label = stationary (|d3s| < 0.5 m)."""
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
                lbl = torch.tensor(float(d3 <= 0.35), device=stat.device)
                num = num + F.binary_cross_entropy_with_logits(
                    stat[b, 0, ri, ci].float().clamp(-15, 15), lbl)
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
            ConvBlock(96, 64), ConvBlock(64, 64), nn.Conv2d(64, 1, 1))
        nn.init.constant_(self.risk_head[-1].bias, -2.0)   # start near 0 risk

    def forward(self, imgs, K, T_cam_ego, v0=None, prev_bev=None,
                warp_theta=None):
        out = super().forward(imgs, K, T_cam_ego, v0, prev_bev, warp_theta)
        crop = self._fused_bev[:, :, 200:600, 125:375]     # +-40 x +-25 m
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
    flow. forward(imgs, K, T, v0, hist_bev [B,3,96,800,500],
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
            nn.Conv2d(96 * (1 + HIST_N), 96, 1, bias=False),
            nn.BatchNorm2d(96), nn.ReLU(inplace=True), ConvBlock(96, 96))
        last_bn = self.tfuse3[-1][-2]
        nn.init.zeros_(last_bn.weight)
        nn.init.zeros_(last_bn.bias)
        # 3. lane-graph slot decoder on the RAW BEV ROI (x -10..60, |y|<=25)
        self.lg_tower = nn.Sequential(
            nn.Conv2d(96, 128, 3, stride=2, padding=1, bias=False),
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
        roi = self._last_bev.detach()[:, :, 100:450, 125:375]
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

    def ego_loss(self, ego, gt):
        valid = gt[:, 16:17]
        n = valid.sum().clamp(min=1)
        Kn = EGO_K
        wps = ego[:, :12 * Kn].view(-1, Kn, 6, 2)
        err = torch.abs(wps - gt[:, :12].view(-1, 1, 6, 2))
        tw = getattr(self, "EGO_TW", None)
        if tw is not None:                      # v36: near horizons weighted
            err = err * tw.to(err.device).view(1, 1, 6, 1)
        lw = getattr(self, "EGO_LONG_W", 1.0)
        wp_ek = (lw * err[..., 0] + 4.0 * err[..., 1]).mean(2) / 2.5  # [B,K]
        best = wp_ek.detach().argmin(1)
        e = self.EPS_WTA
        wp_e = ((1.0 - e) * wp_ek.gather(1, best[:, None])
                + e * wp_ek.mean(1, keepdim=True))
        mlog = ego[:, 12 * Kn:12 * Kn + Kn]
        ce = F.cross_entropy(mlog, best, reduction="none")[:, None]
        cw = 1.0 + gt[:, 11:12].abs().clamp(max=6.0) / 1.5
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
        tgt = np.zeros((B, 2, 200, 200), np.float32)
        msk = np.zeros((B, 1, 200, 200), np.float32)
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
                mm = np.zeros((200, 200), np.uint8)
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
            ri, ci = linear_sum_assignment(cost.detach().cpu().numpy())
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
            nn.Conv2d(192, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True), ConvBlock(128, 128))

    def traj_feat(self):
        # zero residual when the slot is missing (all-zero warped feature),
        # otherwise "everything just appeared" reads as fake motion
        valid = (self._warped0.abs().sum(1, keepdim=True) > 0).to(
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
                r = (80.0 - xe) / DET_RES
                c = (50.0 - ye) / DET_RES
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
        m = ((d > self.D_MIN) & (d < self.D_MIN + self.D_STEP * (self.D - 2))
             ).to(dprob.dtype)
        bins = (self.D_MIN + torch.arange(
            self.D, device=dprob.device, dtype=dprob.dtype)
            * self.D_STEP).view(1, D, 1, 1)
        tri = (1.0 - (d - bins).abs() / self.D_STEP).clamp(min=0)
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
            nn.Conv2d(64, 96, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 1))
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
        self.tgate = nn.Conv2d(96 * 4, 4, 1)
        nn.init.zeros_(self.tgate.weight); nn.init.zeros_(self.tgate.bias)
        # B1
        self.lgq = nn.Embedding(LG_M, 256)
        dl = nn.TransformerDecoderLayer(256, 4, 512, batch_first=True,
                                        dropout=0.0)
        self.lgdec = nn.TransformerDecoder(dl, 2)
        self.lg_in = nn.Conv2d(96, 256, 1)
        self.lg_pts2 = nn.Linear(256, LG_P * 2)
        nn.init.normal_(self.lg_pts2.weight, std=1e-3)
        with torch.no_grad():
            b = torch.zeros(LG_P, 2)
            b[:, 0] = torch.linspace(-4.0, 4.0, LG_P)
            self.lg_pts2.bias.copy_((b / 30.0).reshape(-1))
        self.lg_meta2 = nn.Linear(256, 4)
        # B3
        self.ego_q = nn.Embedding(3, 96)
        self.ego_attn = nn.MultiheadAttention(96, 4, batch_first=True)
        self.ego_delta = nn.Linear(3 * 96, 12 * EGO_K + EGO_K + 3)
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
        # B4 lite: scene interaction token -> traj/stat features rerun
        dt = F.adaptive_avg_pool2d(self._det_feat.detach(), (25, 16))             .flatten(2).transpose(1, 2)                # [B,400,128]
        ag, _ = self.agent_attn(self.agent_q.weight.unsqueeze(0)
                                .expand(B, -1, -1), dt, dt)
        ctx = self.agent_delta(ag.mean(1)[:, :, None, None])
        tf = self._tf[:, :256] + ctx
        out[9] = self.traj_head(torch.cat(
            [tf, self._det_reg[:, 4:6].detach()], 1))
        out[10] = self.stat_head2(tf)
        # B1: query-decoder lane graph replaces out[14..16]
        roi = self._last_bev.detach()[:, :, 100:450, 125:375]
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
    EGO_LONG_W = 2.0

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.risk_gate = nn.Parameter(torch.zeros(1))
        self.vprof_head = nn.Linear(96, 6)
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
        self.dec_head = nn.Linear(96, 36)      # [3 modes x (6 phi + 6 v)]
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
        v = F.softplus(d[:, :, 6:])
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
                 * (2 * BEV_XH / h)) / BEV_XH
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
        return seg_logits + self.out(y0)


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
                 * (2 * BEV_XH / h)) / BEV_XH
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
        return hm + res[:, :2], reg + res[:, 2:]


class E2ERefiner(nn.Module):
    """Residual second-stage planner for the E2E head (roadmap 3f, multi-task).

    The E2E output is a low-dim vector (K hypotheses x 6 waypoints x 2 +
    confidences + controls), not a raster, so the refiner is an MLP rather
    than a U-Net. It sees the predicted plan, the current speed v0, and a
    pooled summary of the fused BEV (scene context), and predicts a residual
    correction on the waypoints. Zero-init last layer => identity at start,
    so the base planner's ADE is preserved and can only improve."""

    def __init__(self, ego_dim, k=EGO_K, ctx_ch=96, hidden=256, max_res=3.0):
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


class MultiTaskRefiner(nn.Module):
    """Post-hoc residual refiners for the three priority heads, sharing the
    single frozen-model forward. Each enabled head is an independent zero-init
    residual module (BEV seg U-Net, 3D-box U-Net, E2E MLP), so none shares
    weights with another or with the frozen base -- every task is preserved by
    construction and only added to. Heads can be enabled independently and
    deployed separately."""

    def __init__(self, do_seg=True, do_box=True, do_e2e=True,
                 n_cls=N_CLASSES, seg_width=48, box_width=32, seg_ctx=0,
                 ego_dim=None, ego_k=EGO_K):
        super().__init__()
        self.seg = BEVSegRefiner(n_cls, ctx_ch=seg_ctx,
                                 width=seg_width) if do_seg else None
        self.box = BEVBoxRefiner(width=box_width) if do_box else None
        self.e2e = (E2ERefiner(ego_dim, k=ego_k)
                    if (do_e2e and ego_dim) else None)

    def forward(self, seg=None, hm=None, reg=None, ego=None, v0=None,
                fused=None, seg_ctx=None):
        """Refine whichever frozen outputs are provided; returns a dict. Called
        through DDP so every enabled head's params are tracked each step."""
        out = {}
        if self.seg is not None and seg is not None:
            out["seg"] = self.seg(seg, seg_ctx)
        if self.box is not None and hm is not None:
            out["hm"], out["reg"] = self.box(hm, reg)
        if self.e2e is not None and ego is not None:
            out["ego"] = self.e2e(ego, v0, fused)
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
        v0r = v0 if v0 is not None else out[0].new_zeros(B)
        r = self.refiner(seg=out[0].float(), hm=out[3].float(),
                         reg=out[4].float(), ego=out[7].float(),
                         v0=v0r, fused=self._fused_bev.float(), seg_ctx=None)
        out[0] = r["seg"]
        out[3], out[4] = r["hm"], r["reg"]
        out[7] = r["ego"]
        return tuple(out)


MODELS = {"v1": IPMSegNet, "v2": IPMSegNetV2, "v3s": IPMSegNetV3,
          "lss": LSSDepthNet, "v8": DepthGatedIPMNet, "v13": DepthSegIPMNet,
          "v13d": DepthSegIPMNetS4, "v14d": DepthSegIPMNetV14,
          "v15": DepthSegIPMNetV15, "v16": DepthSegIPMNetV16,
          "v17": DepthSegIPMNetV17, "v18": DepthSegIPMNetV18,
          "v19": DepthSegIPMNetV19, "v20": DepthSegIPMNetV20,
          "v21": DepthSegIPMNetV21, "v22": DepthSegIPMNetV22,
          "v23": DepthSegIPMNetV23, "v24": DepthSegIPMNetV24,
          "v25": DepthSegIPMNetV25, "v26": DepthSegIPMNetV26, "v27": DepthSegIPMNetV27, "v28": DepthSegIPMNetV28, "v29": DepthSegIPMNetV29, "v30": DepthSegIPMNetV30, "v31": DepthSegIPMNetV31, "v32": DepthSegIPMNetV32, "v33": DepthSegIPMNetV33, "v34": DepthSegIPMNetV34, "v35": DepthSegIPMNetV35, "v36": DepthSegIPMNetV36, "v37": DepthSegIPMNetV37, "v38": DepthSegIPMNetV38, "v39": DepthSegIPMNetV39, "v40": DepthSegIPMNetV40}

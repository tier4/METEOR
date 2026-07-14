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
        ce = F.cross_entropy(logits, tgt, ignore_index=-1, label_smoothing=0.05)
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
        dprob = self.depth_head(self.depth_up(f)).softmax(1)
        return self.project_bev(dprob, self.ctx(f), K, T_cam_ego, B, N, H, W)

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
        seg2d = self.seg_head(f)
        dlog = self.depth_head(self.depth_up(f))
        dprob = dlog.softmax(1)
        ctx = self.ctx(f)
        bev = self.project_bev(dprob, ctx, K, T_cam_ego, B, N, H, W)
        self._last_bev = bev
        bev = self.temporal_fuse(bev)
        self._fused_bev = bev          # consumed by ego / occ / traj heads
        det = self.det_stem(self.det_input())
        self._det_feat = det
        lane_bev = self.lane_input()
        fh2, fw2 = dlog.shape[-2:]
        sh, sw = seg2d.shape[-2:]
        return (self.dec(lane_bev), dlog.view(B, N, self.D, fh2, fw2),
                seg2d.view(B, N, seg2d.shape[1], sh, sw),
                self.hm_head(det), self.reg_head(det))

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
                rad = max(2.0, 0.7 * max(l, w) / DET_RES / 2)
                g = torch.exp(-(((ys - r) ** 2).view(-1, 1)
                                + ((xs - c) ** 2).view(1, -1)) / (2 * rad ** 2))
                ch = 0 if cls < 1.5 else 1
                hm[bi, ch] = torch.maximum(hm[bi, ch], g)
                ll, lw = math.log(max(l, .1)), math.log(max(w, .1))
                sy, cy = math.sin(yaw), math.cos(yaw)
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        r2, c2 = ri + dr, ci + dc
                        if dr == dc == 0                                 or not (0 <= r2 < DET_H and 0 <= c2 < DET_W):
                            continue
                        reg[bi, :, r2, c2] = torch.tensor(
                            [r - r2, c - c2, ll, lw, sy, cy],
                            device=device, dtype=dtype)
                        msk[bi, 0, r2, c2] = 1
                centres.append((bi, ri, ci,
                                torch.tensor([r - ri, c - ci, ll, lw, sy, cy],
                                             device=device, dtype=dtype)))
        for bi, ri, ci, t in centres:
            reg[bi, :, ri, ci] = t
            msk[bi, 0, ri, ci] = 1
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
            # near range must not miss: x2 inside 20 m, x3 inside 12 m
            near = 1.0 + (r < 20.0).float() + (r < 12.0).float()
            cw = torch.tensor([2.0, 5.0], device=hm.device).view(2, 1, 1)
            # far positives are unresolvable at 768x432 (a 60 m pedestrian is
            # ~10 px); full-weight unlearnable positives push the focal loss
            # to suppress confidence everywhere -> damp them instead
            damp = torch.stack([torch.where(r > 50.0, 0.3, 1.0),
                                torch.where(r > 40.0, 0.2, 1.0)])
            # laterally distant objects are out of scope -> nearly ignore
            damp = damp * torch.where(ye.abs() > 15.0, 0.2, 1.0)
            self._det_posw = (near.unsqueeze(0) * cw * damp).unsqueeze(0)
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
        pmax = F.max_pool2d(p, 3, 1, 1)
        p = p * (pmax == p)                        # 3x3 NMS
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
        o = self.occ_head(self.occ_stem(crop))
        B = o.shape[0]
        return out + (o.view(B, OCC_C, OCC_Z, o.shape[-2], o.shape[-1]),)

    def occ_loss(self, occ, occ_gt):
        """occ [B,C,Z,H,W], occ_gt [B,Z,H,W] uint8 (255 = unknown)."""
        if (occ_gt != 255).sum() == 0:
            return occ.sum() * 0.0
        if getattr(self, "_occ_w", None) is None \
                or self._occ_w.device != occ.device:
            w = torch.ones(OCC_C, device=occ.device)
            w[0] = 0.2                     # free dominates the carved volume
            w[2] = 2.0                     # vehicle
            w[1] = 3.0                     # obstacle/unknown (cones etc.)
            w[[3, 4]] = 4.0                # 2-wheelers / pedestrians
            self._occ_w = w
        return F.cross_entropy(occ, occ_gt.long(), weight=self._occ_w,
                               ignore_index=255)


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
        """-> (t [B,12,h,w], m [B,12,h,w]) at box-centre cells."""
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
                t[bi, :, ri, ci] = traj[bi, k].reshape(-1)
                m[bi, :, ri, ci] = tvalid[bi, k].repeat_interleave(2)
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

    def det_input(self):
        return self._last_bev           # 3D det joins the BEV-geometry group

    def traj_feat(self):
        return self.traj_stem(self._fused_bev)


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
                lbl = (traj[b, k, 5].norm() < 0.5).float()
                num = num + F.binary_cross_entropy_with_logits(
                    stat[b, 0, ri, ci].float().clamp(-15, 15), lbl)
                den += 1
        return num / max(den, 1)


MODELS = {"v1": IPMSegNet, "v2": IPMSegNetV2, "v3s": IPMSegNetV3,
          "lss": LSSDepthNet, "v8": DepthGatedIPMNet, "v13": DepthSegIPMNet,
          "v13d": DepthSegIPMNetS4, "v14d": DepthSegIPMNetV14,
          "v15": DepthSegIPMNetV15, "v16": DepthSegIPMNetV16,
          "v17": DepthSegIPMNetV17, "v18": DepthSegIPMNetV18,
          "v19": DepthSegIPMNetV19, "v20": DepthSegIPMNetV20,
          "v21": DepthSegIPMNetV21, "v22": DepthSegIPMNetV22,
          "v23": DepthSegIPMNetV23, "v24": DepthSegIPMNetV24,
          "v25": DepthSegIPMNetV25, "v26": DepthSegIPMNetV26}

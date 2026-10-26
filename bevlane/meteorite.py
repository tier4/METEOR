#!/usr/bin/env python3
"""METEORite -- custom-operator-free METEOR for third-party NPUs.

A complete, static-shape network whose ONNX contains only allow-listed (standard) operators.

  imgs uint8 [1,8,3,352,640] + v0 [1]
    -> shared ResNet18 backbone + FPN-lite        (1/8 feature, 128 ch)
    -> image heads @1/8: depth (D=48, softmax), seg2d (21), hm2d (10), reg2d (4)
    -> Prime extraction: per column, depth-weighted vertical collapse = batched MatMul
         ctx[n,u,c,v] x dprob[n,u,v,d] -> P[n,u,c,d]
    -> baked view transform (rig constants): P -> BEV[c, 400x250]   (two interchangeable
         implementations with identical numerics: "gather" = one constant-index Gather,
         "matmul" = block-sparse constant MatMuls, allow-list only)
    -> BEV encoder (2 levels, 400x250 / 200x125) -> heads
         lane_logit [9,800,500] (x2 nearest + conv), hm/reg/stationary/risk/traj @400x250,
         ego [42] (K=3 candidates, v0 concatenated), tl [4]

K / T_cam_ego are NOT graph inputs: bake_tables() turns one rig's calibration into constants.
No GridSample / ScatterND / Shape / Where / CumSum / trig / LayerNorm / Div in the graph.
"""
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

IMG_H, IMG_W = 352, 640          # 16:9-ish, divisible by 32
N_CAM = 8
FS = 8                           # image feature stride (8 = default; 4 = depth/ctx/2D heads and lift columns at 1/4 resolution.
                                 #   At 1/8, one pixel at 20 m covers ~1.5 m of depth: crosswalks fragment in BEV and depth edges blur)
FH, FW = IMG_H // FS, IMG_W // FS            # 44 x 80
IN_HW = None                     # export only: engine input (h, w) when it differs from IMG_H x IMG_W (in-graph Resize)
D = 48                           # depth bins (log-spaced over [D_MIN, D_MAX] metres, camera z)
D_MIN, D_MAX = 1.5, 90.0
C_P = 64                         # prime / BEV lift channels
BC0, BC1, BCF = 64, 128, 96      # BEV encoder L0 / L1 / fuse widths (capacity levers)
NECK_C, HEAD_C = 128, 96         # image neck (lat/smooth) and depth/ctx head widths (a fixed 128 bottlenecks wider backbones such as r50)
BEV_XF, BEV_XR, BEV_YH = 80.0, 80.0, 50.0    # same extents as METEOR (model.py)
LIFT_RES = 0.4                   # lift grid 400 x 250 (0.4 m); lane head upsamples to 800 x 500
LH, LW = int((BEV_XF + BEV_XR) / LIFT_RES), int(2 * BEV_YH / LIFT_RES)
BLK = 10                         # block edge (cells) for the matmul view transform
# ground-IPM paint branch: 1/4 image features -> 0.2 m ground grid (= lane output
# 800 x 500) by a baked per-rig nearest-pixel table, added into the lane head. Thin paint (laneline/stopline/crosswalk)
# lives on the road plane, so the image row IS its distance; the column collapse of the lift throws that away.
GIPM = False
GIPM_C = 32
# Dedicated box head width (0 = none). Without it hm/reg are plain 1x1 convs on the shared BEV feature m
# (no dedicated layers, shared receptive field).
DET_HEAD = 0
# Confidence-weighted camera fusion: weighted average with a per-camera, per-(depth bin, column) confidence e = exp(q).
# A plain average dilutes sharp tele-camera evidence with blurry wide-angle evidence. The q 1x1 conv is zero-initialised
# (starts as the plain average).
CAMW = 0
CAMW_CLIP = 6.0
# Gradient balancing: most of the shared backbone gradient comes from the ego loss. Scale only the gradient flowing
# back from the ego path into the shared trunk by EGO_GSCALE (identity in the forward pass = zero inference cost;
# the ego head itself still gets the full gradient).
EGO_GSCALE = 1.0
OCC_GSCALE = 1.0   # gradient scale from the OCC head into the shared BEV feature (train --occ-gscale)
OCC_HEAD = 0       # 3D occupancy head (±40 m, 0.4 m, Z 16 x 10 classes) grafted onto a crop of the 0.4 m BEV feature (Conv only)
OCC_Z, OCC_C = 16, 10
YAW2 = 0           # ckpt trained with π-periodic box yaw (sin 2θ, cos 2θ) (graph unchanged; the inference adapter halves the angle)
BOX_GSCALE = 1.0   # gradient scale from the 3D box head into the shared BEV feature (raise box weight without starving path/road; identity at inference)
# PointPainting: seg2d probabilities (classes in PAINT_CLS) are added to ctx before the lift via a zero-initialised 1x1 conv.
# seg2d is detached (trained by its own 2D loss only). The class selection is a weight mask on a 1x1 conv over all 21
# channels (graph is Softmax + Conv + Add only, no Gather).
PAINT_CLS = ()
# HEAD_V2 = 1: function-preserving head upgrades (initial output identical to the baseline):
#   seg  : ResBlock(48) @0.4 m after lane[0] -> x2 -> new 0.2 m branch conv_bn(48->32)+ResBlock(32,d=2)+Conv3x3(32->9, zero-init) added to the baseline output
#   paint: a linear residual block after the painting add (BN γ zero-init = identity)
#   depth: ResBlock(hc) between conv_bn and the 1x1 (identity init)
#   2D   : a shared ResBlock(nc) before seg2d/hm2d/reg2d/ctx/depth (identity init)
#   det  : use together with the existing DET_HEAD tower (--det-head 96)
HEAD_V2 = 0
EGO_GAP = 0        # single-graph NPU export: ego/tl input is GlobalAveragePool (BC1) instead of a fixed 5x4 pool + Flatten (2560), so Gemm/Flatten become 1x1 Convs
# "Simple is better for any NPU": no geometric table lookup (Gather); view transform and ground projection are learned
# low-rank block maps (LIFT_LR=1). Operators: constant MatMul / Add / Concat / Reshape only. Only the camera assignment
# is derived from the reference rig geometry.
LIFT_LR = 0
LR_BLK, LR_RANK = 50, 32         # view transform: 400x250 as 40 blocks of 50x50; per (block, camera) pair a rank-32 U [D*FW, r] / V [r, 2500]
GLR_BLK, GLR_RANK = 100, 16      # ground projection: 800x500 as 40 blocks of 100x100, rank 16
LIFT_KD_W = 1.0                  # distillation weight with the geometric lift (grid_sample) BEV feature as teacher (training only)
LR_CONV1X1 = 0                   # export: express the low-rank layers as 1x1 Convs (channel-axis Slice/Concat only, one transpose in and one out)
# NPU_MODE (export only): "npu" = strict NPU compiler profile:
#   uint8 input [8,3,H,W] (normalisation folded into the stem), prime as 3D bmm, Softmax on dim=1, low-rank layers as Conv1x1
#   (reduction axis = channels), tile expansion as per-tile Conv1x1, BEV assembly with 4D Reshape/Transpose only,
#   auxiliary outputs 4D, v0 as [1,1].
#   "conv1x1" = same Conv1x1 representation + channel split <= 8192 and Resize, for fixed-function accelerators.
NPU_MODE = ""
# LR_RECT: restrict each low-rank layer input to the rectangle implied by the reference rig geometry.
#   U_bn rows drop from D*U (12288) / H4*W4 (20736) to the rectangle (hundreds to thousands): ~1/10 weights, fewer ops,
#   and a geometric prior.
LR_RECT = 0
LR_MARGIN = 0.15                 # grow the rectangle by this fraction on each side (rig variation / learning freedom)
# Rank truncation hurts: the geometric selection matrix S_bn is inherently high-rank for 50-cell blocks. Remedy: 25-cell
#   blocks + initialise U/V from a truncated SVD of S_bn (LR_GEO_INIT; distillation then only learns the residual).
#   Requires LR_RECT.
LR_GEO_INIT = 0
# Export only: LR_MERGE=1 folds the rectangle-restricted U·V into a single Conv1x1 (W = U@V [k, cells]) (for the exact
#   geometric variant with rank = cells); LR_ROWS_ONLY=1 skips the column Slice/Reshape and slices rows (depth / image
#   rows) only (2 nodes per pair; weights grow by zero-padding along columns).
LR_MERGE = 0
LR_ROWS_ONLY = 0     # also affects training (full-width column range, U held as [pr*cols, r]) -> export is Slice(channel)+Conv1x1 only
# LR_HCONV=1: input as [N, cols, rows, C] (one Transpose); per pair Slice(camera, cols, rows) -> Conv(kernel=(rows,1),
#   in=cols, out=cells) computes the rectangle's linear sum. No per-pair Reshape (avoids hundreds of channel<->spatial
#   Reshapes that some NPU compilers reject); weights equal the dense rectangle.
LR_HCONV = 0
LR_PRUNE = 1     # export (two-stage U/V): drop zero-contribution rank dims per pair and snap tiny weights to exact 0 (avoids always-constant activation channels that break per-channel quantisation)
LR_MM4D = 0      # export only: build the lift input [N, U, D, C] directly with a 4D MatMul (dprob^T @ ctx^T, batch (N,U)), removing an extra Reshape/Transpose
# GLR_HALF: build the ground projection at half resolution (0.4 m, 400x250) and upsample nearest x2
#   (training: interpolate / NPU: ConvTranspose) back to 800x500. Dense exact blocks at 0.2 m need 209M weights
#   (blk 25, 6,300 nodes) vs 140M (1,700 nodes) at 0.4 m.
GLR_HALF = 0
REF_RIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ref_rig_drs.npz")


class _GradScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, s):
        ctx.s = s
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return g * ctx.s, None
# BEV2: lift at 0.2 m (800x500) and a 3-level BEV FPN at 0.2/0.4/0.8 m. Per-task heads:
#   seg (0.2 m): dedicated conv + residual (+GIPM) -> 9 classes (no upsampling)
#   det/stationary/risk/traj (0.4 m = box grid): dedicated tower (residual d=1,2,4) + a 3x3 each for hm/reg/traj
#   ego/tl (0.8 m): same shape as the baseline L1 (BC1, 200x125) -> ego_stem/ego_mlp/tl can be inherited from older ckpts
BEV2 = False
BEV2_C02 = 48                    # width of the 0.2 m level
BEV2_DW = 96                     # width of the box tower
DET_HW = (400, 250)              # box / LiDAR raster grid (0.4 m), unchanged under BEV2


def set_bev_res(res):
    """Switch the lift grid resolution (0.4 = default, 0.2 = BEV2). Every function reading LH/LW sees it at run time."""
    global LIFT_RES, LH, LW
    LIFT_RES = res
    LH, LW = int(round((BEV_XF + BEV_XR) / res)), int(round(2 * BEV_YH / res))
# backbone stage widths (stem+layer1 = 1/4, layer2..4 = 1/8..1/32). ResNet50 = bottleneck x4
BB_CH = {"resnet18": (64, 128, 256, 512), "resnet34": (64, 128, 256, 512), "resnet50": (256, 512, 1024, 2048)}
GIPM_RES = 0.2
GH, GW = int((BEV_XF + BEV_XR) / GIPM_RES), int(2 * BEV_YH / GIPM_RES)

# Front-only sensing variant: the internal BEV grid covers only the front sector (x 0..80 m, y ±25 m).
#   Lift (Gather table), BEV network and heads cost 1/4. External outputs (forward_geo = training / eval / demo) are padded
#   back to the canonical grid (x +80..-80, y ±50) (y ±25.6 m = 128 columns at 0.4 m, divisible down to 1/8), so loss,
#   eval and drawing code is unchanged. The export forward keeps the cropped grid (faster).
C_XF, C_XR, C_YH = 80.0, 80.0, 50.0       # canonical extents (same as METEOR)
BEV_FRONT = 0


def set_bev_front(on, xf=80.0, yh=25.6):
    """Switch the internal BEV grid to the front sector (call before building the network). Rebuilds LH/LW/GH/GW."""
    global BEV_FRONT, BEV_XF, BEV_XR, BEV_YH, LH, LW, GH, GW
    BEV_FRONT = int(bool(on))
    BEV_XF, BEV_XR, BEV_YH = (float(xf), 0.0, float(yh)) if BEV_FRONT else (C_XF, C_XR, C_YH)
    LH, LW = int(round((BEV_XF + BEV_XR) / LIFT_RES)), int(round(2 * BEV_YH / LIFT_RES))
    GH, GW = int(round((BEV_XF + BEV_XR) / GIPM_RES)), int(round(2 * BEV_YH / GIPM_RES))


def _front_pad(t, fill=0.0):
    """Pad a BEV tensor [..., h, w] on the internal (front) grid into its place on the canonical grid."""
    h, w = t.shape[-2:]
    res = (BEV_XF + BEV_XR) / h
    Hc, Wc = int(round((C_XF + C_XR) / res)), int(round(2 * C_YH / res))
    r0, c0 = int(round((C_XF - BEV_XF) / res)), int(round((C_YH - BEV_YH) / res))
    out = t.new_full(t.shape[:-2] + (Hc, Wc), fill)
    out[..., r0:r0 + h, c0:c0 + w] = t
    return out


def _front_crop(t):
    """Crop the internal (front) grid region out of a canonical-grid raster (e.g. the LiDAR input)."""
    Hc, Wc = t.shape[-2:]
    res = (C_XF + C_XR) / Hc
    r0, c0 = int(round((C_XF - BEV_XF) / res)), int(round((C_YH - BEV_YH) / res))
    return t[..., r0:r0 + int(round((BEV_XF + BEV_XR) / res)), c0:c0 + int(round(2 * BEV_YH / res))]
IMG_MEAN = (0.485, 0.456, 0.406)
IMG_STD = (0.229, 0.224, 0.225)


def conv_bn(cin, cout, k=3, s=1, d=1):
    p = d * (k // 2)
    return nn.Sequential(nn.Conv2d(cin, cout, k, s, p, dilation=d, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class ResBlock(nn.Module):
    def __init__(self, c, d=1):
        super().__init__()
        self.a = conv_bn(c, c, d=d)
        self.b = nn.Sequential(nn.Conv2d(c, c, 3, 1, d, dilation=d, bias=False), nn.BatchNorm2d(c))

    def forward(self, x):
        return F.relu(x + self.b(self.a(x)))


class ResBlockLin(nn.Module):
    """Residual block without the outer ReLU (for features that can be negative). Identity when bn2 γ=0."""
    def __init__(self, c, d=1):
        super().__init__()
        self.a = conv_bn(c, c, d=d)
        self.b = nn.Sequential(nn.Conv2d(c, c, 3, 1, d, dilation=d, bias=False), nn.BatchNorm2d(c))

    def forward(self, x):
        return x + self.b(self.a(x))


def identity_init(block):
    """Initialise a ResBlock / ResBlockLin to identity (last BN γ = 0)."""
    nn.init.zeros_(block.b[1].weight)
    return block


# ----------------------------------------------------------------------------- geometry
def depth_bin_edges():
    return np.exp(np.linspace(math.log(D_MIN), math.log(D_MAX), D))


def lift_cell_centres():
    """Ego-frame ground points of the lift grid. Row 0 = front (+80 m), col 0 = left (+50 m)."""
    xs = BEV_XF - (np.arange(LH) + 0.5) * LIFT_RES
    ys = BEV_YH - (np.arange(LW) + 0.5) * LIFT_RES
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    return np.stack([gx, gy, np.zeros_like(gx), np.ones_like(gx)], -1).reshape(-1, 4)   # [G2,4]



def _ablate_depth(dlog):
    """Diagnostic: env METEOR_DEPTH_ABLATE=uniform flattens the depth distribution, =sharp sharpens it 4x, to measure
    how much depth quality affects BEV outputs (inference only, not used in training)."""
    m = os.environ.get("METEOR_DEPTH_ABLATE", "")
    if m == "uniform":
        return torch.zeros_like(dlog)
    if m == "sharp":
        return dlog * 4.0
    return dlog

def bake_taps(K, T_cam_ego, img_hw_of_K):
    """For one rig: every (cell, camera) pair -> up to 4 bilinear taps into the prime map (d, u).

    K [N,3,3] for an image of size img_hw_of_K (h, w); T_cam_ego [N,4,4] maps ego -> camera.
    Returns a list over cells of (flat_index, weight) lists; flat index = n*(D*FW) + d*FW + u.
    Weights are normalised per cell so that overlapping cameras AVERAGE (the constant
    "1 / visible cameras" of the design; no Div in the graph)."""
    N = K.shape[0]
    sx, sy = IMG_W / img_hw_of_K[1], IMG_H / img_hw_of_K[0]
    pts = lift_cell_centres()                                    # [G2,4]
    G2 = pts.shape[0]
    log_lo, step = math.log(D_MIN), (math.log(D_MAX) - math.log(D_MIN)) / (D - 1)
    taps = [[] for _ in range(G2)]
    for n in range(N):
        pc = pts @ T_cam_ego[n].T                                # [G2,4] camera coords
        X, Y, Z = pc[:, 0], pc[:, 1], pc[:, 2]
        fx, cx = K[n, 0, 0] * sx, K[n, 0, 2] * sx
        ok = Z > D_MIN
        Zs = np.where(ok, Z, 1.0)
        u_px = fx * X / Zs + cx                                  # pixel column in the 640-wide image
        u = (u_px + 0.5) / FS - 0.5                              # continuous feature column
        b = (np.log(Zs) - log_lo) / step                         # continuous depth-bin index
        ok &= (u >= 0) & (u <= FW - 1) & (b >= 0) & (b <= D - 1)
        u0 = np.floor(u).astype(int); fu = u - u0
        b0 = np.floor(b).astype(int); fb = b - b0
        for cell in np.nonzero(ok)[0]:
            for (dd, wb) in ((b0[cell], 1 - fb[cell]), (min(b0[cell] + 1, D - 1), fb[cell])):
                for (uu, wu) in ((u0[cell], 1 - fu[cell]), (min(u0[cell] + 1, FW - 1), fu[cell])):
                    w = float(wb * wu)
                    if w > 1e-6:
                        taps[cell].append((n * (D * FW) + dd * FW + uu, w))
    for t in taps:                                               # per-cell normalisation
        s = sum(w for _, w in t)
        if s > 0:
            t[:] = [(i, w / s) for i, w in t]
    return taps


def ground_cell_centres():
    xs = BEV_XF - (np.arange(GH) + 0.5) * GIPM_RES
    ys = BEV_YH - (np.arange(GW) + 0.5) * GIPM_RES
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    return np.stack([gx, gy, np.zeros_like(gx), np.ones_like(gx)], -1).reshape(-1, 4)


def ground_pix(K, T_cam_ego, img_hw_of_K, pts):
    """numpy: per camera, nearest 1/4-feature pixel (v4, u4) of each ground point and its validity."""
    sx, sy = IMG_W / img_hw_of_K[1], IMG_H / img_hw_of_K[0]
    H4, W4 = -(-IMG_H // 4), -(-IMG_W // 4)
    out = []
    for n in range(K.shape[0]):
        pc = pts @ T_cam_ego[n].T
        X, Y, Z = pc[:, 0], pc[:, 1], pc[:, 2]
        ok = Z > D_MIN
        Zs = np.where(ok, Z, 1.0)
        u = (K[n, 0, 0] * sx * X / Zs + K[n, 0, 2] * sx + 0.5) / 4 - 0.5
        v = (K[n, 1, 1] * sy * Y / Zs + K[n, 1, 2] * sy + 0.5) / 4 - 0.5
        ui, vi = np.rint(u).astype(np.int64), np.rint(v).astype(np.int64)
        ok &= (ui >= 0) & (ui < W4) & (vi >= 0) & (vi < H4)
        out.append((vi, ui, ok))
    return out, H4, W4


class GroundGather(nn.Module):
    """Baked ground IPM: one constant Gather over the flattened 1/4 maps, averaging the cameras that see a cell."""

    def __init__(self, K, T_cam_ego, img_hw_of_K):
        super().__init__()
        pix, H4, W4 = ground_pix(K, T_cam_ego, img_hw_of_K, ground_cell_centres())
        G = GH * GW
        cnt = np.zeros(G, np.int64)
        for _, _, ok in pix:
            cnt += ok
        kmax = max(1, int(cnt.max()))
        pad = N_CAM * H4 * W4
        idx = np.full((G, kmax), pad, np.int64)
        wgt = np.zeros((G, kmax), np.float32)
        fill = np.zeros(G, np.int64)
        for n, (vi, ui, ok) in enumerate(pix):
            c = np.nonzero(ok)[0]
            idx[c, fill[c]] = n * H4 * W4 + vi[c] * W4 + ui[c]
            wgt[c, fill[c]] = 1.0 / cnt[c]
            fill[c] += 1
        self.register_buffer("idx", torch.from_numpy(idx.reshape(-1)))
        self.register_buffer("wgt", torch.from_numpy(wgt)[None])
        self.kmax, self.hw4 = kmax, (H4, W4)

    def forward(self, f4):                                       # f4 [N, C, H4, W4]
        C = f4.shape[1]
        flat = f4.permute(1, 0, 2, 3).reshape(C, -1)
        flat = torch.cat([flat, flat.new_zeros(C, 1)], 1)
        g = flat.index_select(1, self.idx).view(C, GH * GW, self.kmax)
        return (g * self.wgt).sum(-1).view(1, C, GH, GW)


# ----------------------------------------------------------------------------- view transforms
class VTGather(nn.Module):
    """One constant-index Gather over the flattened prime maps (Tier B: constant Gather)."""

    def __init__(self, taps):
        super().__init__()
        G2 = len(taps)
        kmax = max(1, max(len(t) for t in taps))
        pad = N_CAM * D * FW                                     # the appended zero slot
        idx = np.full((G2, kmax), pad, np.int64)
        wgt = np.zeros((G2, kmax), np.float32)
        for c, t in enumerate(taps):
            for j, (i, w) in enumerate(t):
                idx[c, j], wgt[c, j] = i, w
        self.register_buffer("idx", torch.from_numpy(idx.reshape(-1)))
        self.register_buffer("wgt", torch.from_numpy(wgt)[None])          # [1,G2,kmax]
        self.kmax = kmax

    def forward(self, P):                                        # P [N, C, D*FW]
        C = P.shape[1]                                           # C_P (+1: CAMW confidence channel)
        flat = P.permute(1, 0, 2).reshape(C, N_CAM * D * FW)
        flat = torch.cat([flat, flat.new_zeros(C, 1)], 1)        # pad slot
        g = flat.index_select(1, self.idx).view(C, LH * LW, self.kmax)
        bev = (g * self.wgt).sum(-1)
        return bev.view(1, C, LH, LW)


class VTMatmul(nn.Module):
    """Block-sparse constant MatMuls: allow-list operators only (Slice / Reshape / MatMul / Add / Concat / Transpose)."""

    def __init__(self, taps):
        super().__init__()
        assert LH % BLK == 0 and LW % BLK == 0
        self.nby, self.nbx = LH // BLK, LW // BLK
        self.blocks = []                                         # per block: list of (cam, d0, d1, u0, u1, W[pd*pu, BLK*BLK])
        self.weights = nn.ParameterList()
        n_mac = 0
        for by in range(self.nby):
            for bx in range(self.nbx):
                per_cam = {}
                for iy in range(BLK):
                    for ix in range(BLK):
                        cell = (by * BLK + iy) * LW + bx * BLK + ix
                        for i, w in taps[cell]:
                            n, rem = divmod(i, D * FW)
                            dd, uu = divmod(rem, FW)
                            per_cam.setdefault(n, []).append((dd, uu, iy * BLK + ix, w))
                ent = []
                for n, lst in sorted(per_cam.items()):
                    d0 = min(t[0] for t in lst); d1 = max(t[0] for t in lst) + 1
                    u0 = min(t[1] for t in lst); u1 = max(t[1] for t in lst) + 1
                    W = np.zeros((d1 - d0, u1 - u0, BLK * BLK), np.float32)
                    for dd, uu, j, w in lst:
                        W[dd - d0, uu - u0, j] += w
                    p = nn.Parameter(torch.from_numpy(W.reshape(-1, BLK * BLK)), requires_grad=False)
                    self.weights.append(p)
                    ent.append((n, d0, d1, u0, u1, len(self.weights) - 1))
                    n_mac += C_P * W.shape[0] * W.shape[1] * BLK * BLK
                self.blocks.append(ent)
        self.n_mac = n_mac
        self.n_params = sum(p.numel() for p in self.weights)

    def forward(self, P):                                        # P [N, C, D*FW]
        P4 = P.view(N_CAM, C_P, D, FW)
        outs = []
        zero = P.new_zeros(C_P, BLK * BLK)
        for ent in self.blocks:
            acc = None
            for (n, d0, d1, u0, u1, k) in ent:
                patch = P4[n:n + 1, :, d0:d1, u0:u1].reshape(C_P, -1)   # static Slice + Reshape (no Gather)
                y = patch @ self.weights[k]                            # [C, BLK*BLK]
                acc = y if acc is None else acc + y
            outs.append(zero if acc is None else acc)
        bev = torch.stack(outs, 1)                                     # [C, nby*nbx, BLK*BLK]
        bev = bev.view(C_P, self.nby, self.nbx, BLK, BLK).permute(0, 1, 3, 2, 4)
        return bev.reshape(1, C_P, LH, LW)


class VTBlockConv(nn.Module):
    """Same table as VTMatmul but every block-camera MatMul is a 1x1 Conv with constant weights (for fixed-function
    accelerators without MatMul, where Conv is the native op). The prime map is first permuted to [N, D, U, C] so a
    block's (d,u) patch becomes the channel axis of a [1, pd*pu, 1, C] tensor; the 1x1 Conv maps it to [1, BLK*BLK, 1, C];
    blocks are concatenated on the channel axis (the most widely supported Concat axis) and one shuffle at the end
    restores [1, C, LH, LW]. Intended for a larger block (BLK=25: 361 convs, 29M weights)."""

    def __init__(self, taps):
        super().__init__()
        base = VTMatmul(taps)
        self.nby, self.nbx, self.blocks = base.nby, base.nbx, base.blocks
        self.convs = nn.ModuleList()
        for p in base.weights:                                     # p: [pd*pu, BLK*BLK]
            c = nn.Conv2d(p.shape[0], BLK * BLK, 1, bias=False)
            c.weight = nn.Parameter(p.t().reshape(BLK * BLK, p.shape[0], 1, 1).contiguous(), requires_grad=False)
            self.convs.append(c)
        self.n_params, self.n_mac = base.n_params, base.n_mac

    def forward(self, P):                                          # P [N, C, D*FW]
        Q = P.view(N_CAM, C_P, D, FW).permute(0, 2, 3, 1)          # [N, D, U, C]
        outs = []
        zero = P.new_zeros(1, BLK * BLK, 1, C_P)
        for ent in self.blocks:
            acc = None
            for (n, d0, d1, u0, u1, k) in ent:
                patch = Q[n:n + 1, d0:d1, u0:u1, :].reshape(1, (d1 - d0) * (u1 - u0), 1, C_P)
                y = self.convs[k](patch)                           # [1, BLK*BLK, 1, C]
                acc = y if acc is None else acc + y
            outs.append(zero if acc is None else acc)
        bev = torch.cat(outs, 1).reshape(-1, C_P)                  # [nblk*BLK*BLK, C]  (reshape, not indexing: no Gather)
        bev = bev.t().reshape(C_P, self.nby, self.nbx, BLK, BLK).permute(0, 1, 3, 2, 4)
        return bev.reshape(1, C_P, LH, LW)



def _block_cams(Gh, Gw, B, cell_cams, min_frac=0.05):
    """Per block (B x B cells, row-major), the cameras that see at least min_frac of its cells (structure only, not learned)."""
    nby, nbx = Gh // B, Gw // B
    out = []
    for by in range(nby):
        for bx in range(nbx):
            cnt = {}
            for y in range(by * B, (by + 1) * B):
                row = y * Gw
                for x in range(bx * B, (bx + 1) * B):
                    for n in cell_cams[row + x]:
                        cnt[n] = cnt.get(n, 0) + 1
            out.append([n for n, c in sorted(cnt.items()) if c >= min_frac * B * B])
    return out


def ref_structures():
    """From the reference rig (REF_RIG) geometry, return the set of visible cameras per lift-grid cell and per
    ground-grid cell (structure for the low-rank maps)."""
    z = np.load(REF_RIG)
    K, T, hw = z["K"], z["T_cam_ego"], tuple(int(v) for v in z["img_hw"])
    taps = bake_taps(K, T, hw)
    lift_cams = [set(i // (D * FW) for i, _ in tp) for tp in taps]
    lift_src = [[(i // (D * FW), (i % (D * FW)) // FW, i % FW, float(w)) for i, w in tp] for tp in taps]      # (cam, d, u, w)
    global GH, GW, GIPM_RES
    if GLR_HALF:                                                 # build the structure on the half-resolution ground grid (0.4 m)
        GH0, GW0, R0 = GH, GW, GIPM_RES; GH, GW, GIPM_RES = GH0 // 2, GW0 // 2, R0 * 2
    try:
        pix, H4, W4 = ground_pix(K, T, hw, ground_cell_centres())
    finally:
        if GLR_HALF:
            GH, GW, GIPM_RES = GH0, GW0, R0
    GHs, GWs = (GH // 2, GW // 2) if GLR_HALF else (GH, GW)
    g_cams = [set() for _ in range(GHs * GWs)]
    g_src = [[] for _ in range(GHs * GWs)]
    cnt = np.zeros(GHs * GWs, np.int64)
    for _, _, ok in pix:
        cnt += ok
    for n, (vi, ui, ok) in enumerate(pix):
        for c in np.nonzero(ok)[0]:
            g_cams[c].add(n); g_src[c].append((n, int(vi[c]), int(ui[c]), 1.0 / cnt[c]))      # (cam, v, u, w) same weights as the geometric GroundGather
    ref_structures.last = (lift_src, g_src, H4, W4)
    return lift_cams, g_cams


class BlockLowRank(nn.Module):
    """Learned low-rank block map. Input: flat per-camera features X [B, N, C, Kin]
    (lift: polar map P [C, D*FW]; ground projection: 1/4-image paint features [C, H4*W4]).
    The output grid (Gh x Gw) is split into blk x blk blocks; for each camera n that sees block b,
        Y_bn = (X_n @ U_bn) @ V_bn,   U_bn [Kin, r], V_bn [r, blk*blk]
    and the contributions are summed. No geometric table lookup (Gather); only which camera contributes to which block
    is taken from the reference rig.
    ONNX: per camera MatMul (all U fused) -> Reshape -> batched MatMul (V) -> Add -> Concat -> Reshape. ~100 nodes;
    ~28M weights for the lift (rank 32 / blk 50), ~30M for the ground (rank 16 / blk 100)."""

    def __init__(self, cell_cams, Gh, Gw, C, Kin, blk, rank, cell_src=None, rows=0, cols=0, margin=0.15):
        super().__init__()
        assert Gh % blk == 0 and Gw % blk == 0, (Gh, Gw, blk)
        self.Gh, self.Gw, self.blk, self.C, self.rank = Gh, Gw, blk, C, rank
        self.nby, self.nbx = Gh // blk, Gw // blk
        self.cams = _block_cams(Gh, Gw, blk, cell_cams)
        self.cam_blocks = {n: [b for b, cs in enumerate(self.cams) if n in cs] for n in range(N_CAM)}
        self.rect = cell_src is not None
        self.rows, self.cols = rows, cols
        self.rects = {}                                            # (n, b) -> (r0, r1, c0, c1) reference rectangle (feature grid)
        self.U, self.V = nn.ParameterDict(), nn.ParameterDict()
        for n, bl in self.cam_blocks.items():
            if not bl:
                continue
            kb = len(bl)
            if self.rect:
                geo = {}
                for j, b in enumerate(bl):
                    by, bx = divmod(b, self.nbx)
                    rr, cc, ent = [], [], []
                    for jj, y in enumerate(range(by * blk, (by + 1) * blk)):
                        for ii, x in enumerate(range(bx * blk, (bx + 1) * blk)):
                            for e in cell_src[y * Gw + x]:
                                if e[0] == n:
                                    rr.append(e[1]); cc.append(e[2]); ent.append((e[1], e[2], jj * blk + ii, e[3] if len(e) > 3 else 1.0))
                    r0, r1, c0, c1 = min(rr), max(rr) + 1, min(cc), max(cc) + 1
                    mr, mc = max(2, int(round(margin * (r1 - r0)))), max(2, int(round(margin * (c1 - c0))))
                    r0, r1, c0, c1 = max(0, r0 - mr), min(rows, r1 + mr), max(0, c0 - mc), min(cols, c1 + mc)
                    if LR_ROWS_ONLY:                                   # restrict rows only (full-width columns): export becomes Slice(channel)+Conv1x1 only
                        c0, c1 = 0, cols
                    self.rects[(n, b)] = (r0, r1, c0, c1)
                    k = (r1 - r0) * (c1 - c0)
                    self.U[f"{n}_{j}"] = nn.Parameter(torch.randn(k, rank) / math.sqrt(k))
                    if LR_GEO_INIT:
                        geo[j] = (ent, r0, c0, c1 - c0, k)
            else:
                self.U[str(n)] = nn.Parameter(torch.randn(Kin, rank * kb) / math.sqrt(Kin))
            self.V[str(n)] = nn.Parameter(torch.randn(kb, rank, blk * blk) / math.sqrt(rank))
            if self.rect and LR_GEO_INIT:
                # Truncated SVD of the geometric selection matrix S [k, blk*blk] (cell j <- (row, col) in the rectangle with
                # weight w): U = Us*sqrt(s), V = sqrt(s)*Vh (U @ V ≈ S; distillation learns the residual). CPU, deterministic.
                with torch.no_grad():
                    for j, (ent, r0, c0, pc, k) in geo.items():
                        S = torch.zeros(k, blk * blk, dtype=torch.float64)
                        for (rw, cl, cell, w) in ent:
                            S[(rw - r0) * pc + (cl - c0), cell] += w
                        Us, sv, Vh = torch.linalg.svd(S, full_matrices=False)
                        q = min(rank, sv.numel()); rt = sv[:q].sqrt()
                        self.U[f"{n}_{j}"].zero_(); self.U[f"{n}_{j}"][:, :q] = (Us[:, :q] * rt).float()
                        self.V[str(n)][j].zero_(); self.V[str(n)][j, :q] = (rt[:, None] * Vh[:q]).float()
                        self.geo_loss = getattr(self, "geo_loss", 0.0) + float((sv[q:] ** 2).sum()); self.geo_tot = getattr(self, "geo_tot", 0.0) + float((sv ** 2).sum())
        self.n_params = sum(p.numel() for p in self.parameters())
        if self.rect and LR_GEO_INIT and getattr(self, "geo_tot", 0) > 0:
            print(f"[lr] geometric SVD init: blk {blk} rank {rank} pairs {len(self.rects)} operator energy loss {self.geo_loss / self.geo_tot:.3f}", flush=True)

    @torch.no_grad()
    def prepare_npu(self, in_scale=1.0):
        """Call before export (LR_MERGE / LR_ROWS_ONLY): fold the rectangle-restricted U·V into one weight W [k, cells]
        stored as a buffer (folding inside forward would be traced as MatMul/Transpose and push weights past 2 GB into
        external data)."""
        if not self.rect:
            return
        rows, cols = self.rows, self.cols
        for n, bl in self.cam_blocks.items():
            for j, b in enumerate(bl):
                r0, r1, c0, c1 = self.rects[(n, b)]
                pr, pc = r1 - r0, c1 - c0
                Uj = self.U[f"{n}_{j}"] * in_scale
                if LR_ROWS_ONLY:
                    Uf = Uj.new_zeros(pr, cols, Uj.shape[1]); Uf[:, c0:c1] = Uj.view(pr, pc, -1); Uj = Uf.reshape(pr * cols, -1)
                if LR_HCONV and (LR_MERGE or self.rank >= self.blk * self.blk):   # dense (exact geometric, rank = cells): Conv(kernel=(pr,1)) weight [cells, pc, pr, 1] (input [1, pc, pr, C])
                    W = (Uj @ self.V[str(n)][j]).t().reshape(self.blk * self.blk, pr, Uj.shape[0] // pr).permute(0, 2, 1).unsqueeze(-1).contiguous()
                    self.register_buffer(f"Wh_{n}_{j}", W, persistent=False)
                elif LR_MERGE:                                                    # keep as Conv1x1 weight [cells, k, 1, 1] (no constant Transpose/Reshape copies during trace)
                    W = (Uj @ self.V[str(n)][j]).t().reshape(self.blk * self.blk, Uj.shape[0], 1, 1).contiguous()
                    self.register_buffer(f"Wm_{n}_{j}", W, persistent=False)
                else:                                                             # two-stage U/V (Conv1x1 x2): prune ranks per pair, snap tiny weights to exact 0
                    Vj = self.V[str(n)][j]                                        # [r, cells]
                    if LR_PRUNE:
                        cu = Uj.abs().amax(0) * Vj.abs().amax(1)                  # upper bound of the contribution per rank dim
                        keep = cu > 1e-6 * float(cu.max())
                        if int(keep.sum()) == 0:
                            keep[0] = True
                        Uj, Vj = Uj[:, keep], Vj[keep]
                        Uj = torch.where(Uj.abs() < 1e-7 * float(Uj.abs().max()), torch.zeros_like(Uj), Uj)
                        Vj = torch.where(Vj.abs() < 1e-7 * float(Vj.abs().max()), torch.zeros_like(Vj), Vj)
                        self._pruned_ranks = getattr(self, "_pruned_ranks", []) + [int(Uj.shape[1])]
                    if LR_HCONV:                                                  # first stage also stays spatial: Conv(kernel=(pr,1)) [r', pc, pr, 1] (avoids a ~98k-channel Slice)
                        self.register_buffer(f"Uh_{n}_{j}", Uj.t().reshape(-1, pr, Uj.shape[0] // pr).permute(0, 2, 1).unsqueeze(-1).contiguous(), persistent=False)
                    else:
                        self.register_buffer(f"Um_{n}_{j}", Uj.t().reshape(-1, Uj.shape[0], 1, 1).contiguous(), persistent=False)
                    self.register_buffer(f"Vm_{n}_{j}", Vj.t().reshape(self.blk * self.blk, -1, 1, 1).contiguous(), persistent=False)
        if getattr(self, "_pruned_ranks", None):
            pr_ = self._pruned_ranks; print(f"[lr] export rank pruning: pairs {len(pr_)} rank {self.rank} -> mean {sum(pr_)/len(pr_):.1f} (min {min(pr_)}, max {max(pr_)})", flush=True)
        self.npu_ready = True                                                     # U/V are kept (the training-path forward uses them for parity checks); call drop_uv() right before export

    @torch.no_grad()
    def drop_uv(self):
        """Right before export: keep only the folded buffers and empty U/V (torch.onnx.export switches to the external
        data format when the module's total parameters exceed 2 GB, even if unused)."""
        for k in list(self.U.keys()):
            self.U[k] = nn.Parameter(torch.zeros(1), requires_grad=False)
        for k in list(self.V.keys()):
            self.V[k] = nn.Parameter(torch.zeros(1), requires_grad=False)

    def _rect_patch(self, X, n, b):
        """Training path: crop the (n, b) reference rectangle of X [B, N, C, rows, cols] to [B, C, k]."""
        r0, r1, c0, c1 = self.rects[(n, b)]
        return X.narrow(1, n, 1).squeeze(1).narrow(2, r0, r1 - r0).narrow(3, c0, c1 - c0).reshape(X.shape[0], X.shape[2], -1)

    def forward_npu(self, Xf, S, C, ch_max=0, out_max=0, in_scale=1.0):
        """NPU export (batch 1, 4D only): Xf [1, N*S, 1, C] (S = reduction dim on the channel axis, C = horizontal spatial axis).
        Camera n's [1, S, 1, C] is channel-Sliced -> Conv1x1 (U^T: S -> r*kb) -> per-tile Conv1x1 (V^T: r -> blk*blk) ->
        Add across cameras -> channel Concat [1, nblk*blk*blk, 1, C] -> three 4D Transpose/Reshape to [1, C, Gh, Gw].
        ch_max / out_max > 0 split input / output channels at that limit (for accelerators with channel limits)."""
        blk, r, nblk = self.blk, self.rank, self.nby * self.nbx
        nby, nbx = self.nby, self.nbx
        blocks = [None] * nblk
        if self.rect and LR_HCONV:                                                        # Xf [N, cols, rows, C] -> Slice(camera, cols, rows) -> Conv(kernel=(pr,1)) (no per-pair Reshape)
            assert getattr(self, "npu_ready", False), "LR_HCONV requires prepare_npu() first"
            for n, bl in self.cam_blocks.items():
                for j, b in enumerate(bl):
                    r0, r1, c0, c1 = self.rects[(n, b)]
                    xs = Xf.narrow(0, n, 1).narrow(1, c0, c1 - c0).narrow(2, r0, r1 - r0)       # [1, pc, pr, C]
                    if hasattr(self, f"Wh_{n}_{j}"):
                        y = F.conv2d(xs, getattr(self, f"Wh_{n}_{j}").to(Xf.dtype))         # dense: [1, blk*blk, 1, C]
                    else:                                                                   # two-stage low rank: Conv(kernel=(pr,1)) [1, r', 1, C] -> Conv1x1 [1, blk*blk, 1, C]
                        y = F.conv2d(F.conv2d(xs, getattr(self, f"Uh_{n}_{j}").to(Xf.dtype)), getattr(self, f"Vm_{n}_{j}").to(Xf.dtype))
                    blocks[b] = y if blocks[b] is None else blocks[b] + y
            zero = Xf.new_zeros(1, blk * blk, 1, C)
            full = torch.cat([z if z is not None else zero for z in blocks], 1)
            g = full.permute(0, 3, 2, 1).reshape(1, C * nby, nbx, blk * blk)
            g = g.permute(0, 1, 3, 2).reshape(1, C * nby * blk, blk, nbx)
            return g.permute(0, 1, 3, 2).reshape(1, C, nby * blk, nbx * blk)
        if self.rect:                                                                     # rectangle-restricted: Slice(rows) -> Reshape -> Slice(cols) -> Reshape -> Conv1x1
            rows, cols = self.rows, self.cols
            for n, bl in self.cam_blocks.items():
                V = self.V[str(n)].to(Xf.dtype)
                for j, b in enumerate(bl):
                    r0, r1, c0, c1 = self.rects[(n, b)]
                    pr, pc = r1 - r0, c1 - c0
                    if LR_ROWS_ONLY:                                                      # Slice rows only (zero-padded in prepare_npu)
                        xs = Xf.narrow(1, n * rows * cols + r0 * cols, pr * cols)        # [1, pr*cols, 1, C]
                    else:
                        xs = Xf.narrow(1, n * rows * cols + r0 * cols, pr * cols).reshape(1, pr, cols, C).narrow(2, c0, pc).reshape(1, pr * pc, 1, C)
                    if LR_MERGE:                                                          # W = U@V as one Conv1x1 (prepared by prepare_npu, Conv-shaped weight)
                        y = F.conv2d(xs, getattr(self, f"Wm_{n}_{j}").to(Xf.dtype))       # [1, blk*blk, 1, C]
                    else:
                        if getattr(self, "npu_ready", False):
                            Wu = getattr(self, f"Um_{n}_{j}").to(Xf.dtype)
                        else:
                            Uj = (self.U[f"{n}_{j}"] * in_scale).to(Xf.dtype); Wu = Uj.t().reshape(r, Uj.shape[0], 1, 1)
                        h = F.conv2d(xs, Wu)                                              # [1, r', 1, C]
                        Wv = getattr(self, f"Vm_{n}_{j}").to(Xf.dtype) if getattr(self, "npu_ready", False) else V[j].t().reshape(blk * blk, r, 1, 1)
                        y = F.conv2d(h, Wv)                                               # [1, blk*blk, 1, C]
                    blocks[b] = y if blocks[b] is None else blocks[b] + y
            zero = Xf.new_zeros(1, blk * blk, 1, C)
            full = torch.cat([z if z is not None else zero for z in blocks], 1)
            g = full.permute(0, 3, 2, 1).reshape(1, C * nby, nbx, blk * blk)
            g = g.permute(0, 1, 3, 2).reshape(1, C * nby * blk, blk, nbx)
            return g.permute(0, 1, 3, 2).reshape(1, C, nby * blk, nbx * blk)
        for n, bl in self.cam_blocks.items():
            if not bl:
                continue
            kb = len(bl)
            Wu = (self.U[str(n)] * in_scale).t().reshape(r * kb, S, 1, 1).to(Xf.dtype)
            xs = Xf.narrow(1, n * S, S)                                                    # [1, S, 1, C]
            if ch_max and S > ch_max:
                h = None
                for k0 in range(0, S, ch_max):
                    k1 = min(S, k0 + ch_max)
                    hk = F.conv2d(xs.narrow(1, k0, k1 - k0), Wu[:, k0:k1].contiguous())
                    h = hk if h is None else h + hk
            else:
                h = F.conv2d(xs, Wu)                                                       # [1, r*kb, 1, C]
            V = self.V[str(n)].to(Xf.dtype)                                                # [kb, r, blk*blk]
            for j, b in enumerate(bl):
                hj = h.narrow(1, j * r, r)                                                 # [1, r, 1, C]
                Wv = V[j].t().reshape(blk * blk, r, 1, 1)
                if out_max and blk * blk > out_max:
                    ys = [F.conv2d(hj, Wv[o0:min(blk * blk, o0 + out_max)].contiguous()) for o0 in range(0, blk * blk, out_max)]
                    y = torch.cat(ys, 1)
                else:
                    y = F.conv2d(hj, Wv)                                                   # [1, blk*blk, 1, C]
                blocks[b] = y if blocks[b] is None else blocks[b] + y
        zero = Xf.new_zeros(1, blk * blk, 1, C)
        full = torch.cat([z if z is not None else zero for z in blocks], 1)                 # [1, nblk*blk*blk, 1, C]  (by, bx, iy, ix)
        g = full.permute(0, 3, 2, 1).reshape(1, C * nby, nbx, blk * blk)                   # [1, C*by, bx, (iy,ix)]
        g = g.permute(0, 1, 3, 2).reshape(1, C * nby * blk, blk, nbx)                      # [1, C*by*iy, ix, bx]
        g = g.permute(0, 1, 3, 2).reshape(1, C, nby * blk, nbx * blk)                      # [1, C, Gh, Gw]
        return g

    def forward_conv1x1(self, X, ch_max=8192, out_max=2500):
        """Conv1x1 mode (export only, batch 1): U and V as 1x1 Convs. Input [1, N, C, Kin] is transposed once to
        [1, N*Kin, C, 1] (camera and Kin on channels, C on the vertical spatial axis); channel Slice (<= ch_max) -> Conv(U)
        -> Add -> per-block Conv(V) (output split at <= out_max) -> Add across cameras -> channel Concat -> one transpose
        to [1, C, Gh, Gw]. No Gather, no batch-axis Slice."""
        B, N, C, Kin = X.shape
        assert B == 1, "forward_conv1x1 is export-only (batch 1)"
        blk, r, nblk = self.blk, self.rank, self.nby * self.nbx
        Xt = X.reshape(N, C, Kin).permute(0, 2, 1).reshape(1, N * Kin, C, 1)
        blocks = [None] * nblk
        for n, bl in self.cam_blocks.items():
            if not bl:
                continue
            kb = len(bl)
            Wu = self.U[str(n)].t().reshape(r * kb, Kin, 1, 1).to(X.dtype)
            h = None
            for k0 in range(0, Kin, ch_max):
                k1 = min(Kin, k0 + ch_max)
                hk = F.conv2d(Xt.narrow(1, n * Kin + k0, k1 - k0), Wu[:, k0:k1].contiguous())   # [1, r*kb, C, 1]
                h = hk if h is None else h + hk
            V = self.V[str(n)].to(X.dtype)                                                   # [kb, r, blk*blk]
            for j, b in enumerate(bl):
                hj = h.narrow(1, j * r, r)                                                   # [1, r, C, 1]
                Wv = V[j].t().reshape(blk * blk, r, 1, 1)
                ys = [F.conv2d(hj, Wv[o0:min(blk * blk, o0 + out_max)].contiguous()) for o0 in range(0, blk * blk, out_max)]
                y = torch.cat(ys, 1) if len(ys) > 1 else ys[0]                               # [1, blk*blk, C, 1]
                blocks[b] = y if blocks[b] is None else blocks[b] + y
        zero = X.new_zeros(1, blk * blk, C, 1)
        full = torch.cat([z if z is not None else zero for z in blocks], 1)                   # [1, nblk*blk*blk, C, 1]
        out = full.view(1, self.nby, self.nbx, blk, blk, C).permute(0, 5, 1, 3, 2, 4)
        return out.reshape(1, C, self.Gh, self.Gw)

    def forward(self, X):                                      # X [B, N, C, Kin]
        """No Gather / Unsqueeze in the ONNX: indexing uses narrow (Slice) and stacking uses cat (Concat) only.
        Blocks are assembled as [B, nblk, C, blk*blk] and camera contributions are combined with Add."""
        B, N, C = X.shape[:3]
        blk, r, nblk = self.blk, self.rank, self.nby * self.nbx
        if self.rect:                                                               # rectangle-restricted (training path; X is 5D [B,N,C,rows,cols])
            blocks = [None] * nblk
            for n, bl in self.cam_blocks.items():
                V = self.V[str(n)].to(X.dtype)
                for j, b in enumerate(bl):
                    h = torch.matmul(self._rect_patch(X, n, b), self.U[f"{n}_{j}"].to(X.dtype))   # [B, C, r]
                    y = torch.matmul(h, V[j])                                                      # [B, C, blk*blk]
                    blocks[b] = y if blocks[b] is None else blocks[b] + y
            zero = X.new_zeros(B, C, blk * blk)
            total = torch.stack([z if z is not None else zero for z in blocks], 1)                 # [B, nblk, C, blk*blk]
            out = total.view(B, self.nby, self.nbx, C, blk, blk).permute(0, 3, 1, 4, 2, 5)
            return out.reshape(B, C, self.Gh, self.Gw)
        ys = {}
        for n, bl in self.cam_blocks.items():
            if not bl:
                continue
            kb = len(bl)
            Xn = X.narrow(1, n, 1).reshape(B, C, -1)                               # Slice + Reshape
            h = torch.matmul(Xn, self.U[str(n)].to(X.dtype))                        # [B, C, r*kb]
            h = h.view(B, C, kb, r).permute(0, 2, 1, 3)                             # [B, kb, C, r]
            ys[n] = torch.matmul(h, self.V[str(n)].to(X.dtype).unsqueeze(0))        # [B, kb, C, blk*blk]
        parts = {n: torch.split(y, 1, dim=1) for n, y in ys.items()}               # one Split per camera (kb outputs)
        blocks = []
        for b in range(nblk):                                                       # in block order, Add the contributions of the cameras that see it (no zero constants)
            acc = None
            for n in self.cams[b]:
                piece = parts[n][self.cam_blocks[n].index(b)]                        # [B, 1, C, blk*blk]
                acc = piece if acc is None else acc + piece
            blocks.append(acc if acc is not None else X.new_zeros(B, 1, C, blk * blk))
        total = torch.cat(blocks, 1)                                                # [B, nblk, C, blk*blk]
        out = total.view(B, self.nby, self.nbx, C, blk, blk).permute(0, 3, 1, 4, 2, 5)
        return out.reshape(B, C, self.Gh, self.Gw)


# ----------------------------------------------------------------------------- the network
class METEORite(nn.Module):
    def __init__(self, taps, vt="gather", pretrained=True, conv1x1=False, prime="matmul", backbone="resnet18"):
        """conv1x1=True: fit fixed-function accelerator layer limits (depth softmax on the channel axis, planner pool as two
        stages with kernel <= 8). Numerics identical to conv1x1=False (softmax axis is equivalent before/after the
        transpose; pool 10x8 = (5x4)(2x2)).
        prime: implementation of the per-column depth-weighted vertical collapse
          "matmul"      batched MatMul (dynamic x dynamic; unsupported on some accelerators)
          "conv_exact"  same numerics with basic ops only: constant 1x1 Conv expands to (c,d) -> Mul -> AvgPool(4,1) -> depthwise Conv(11,1) sums over v
          "conv_learned" learned, PON-style: [ctx; depth logits] -> AvgPool(2,1) -> Conv kernel (22,1) straight to [C*D, 1, U] (no product)"""
        super().__init__()
        self.conv1x1 = conv1x1
        self.prime = prime
        if prime == "conv_exact":
            # constant expansion matrices: tile_c[c*D+d, c] = 1, tile_d[c*D+d, d] = 1
            tc = torch.zeros(C_P * D, C_P, 1, 1); td = torch.zeros(C_P * D, D, 1, 1)
            for c in range(C_P):
                for d in range(D):
                    tc[c * D + d, c] = 1.0; td[c * D + d, d] = 1.0
            self.tile_c = nn.Conv2d(C_P, C_P * D, 1, bias=False); self.tile_c.weight = nn.Parameter(tc, requires_grad=False)
            self.tile_d = nn.Conv2d(D, C_P * D, 1, bias=False); self.tile_d.weight = nn.Parameter(td, requires_grad=False)
            assert FH == 44, "vertical reduction is baked for FH=44 = 4 x 11"
            self.vsum = nn.Conv2d(C_P * D, C_P * D, (11, 1), groups=C_P * D, bias=False)        # sum over the 11 pooled rows
            self.vsum.weight = nn.Parameter(torch.full((C_P * D, 1, 11, 1), 4.0), requires_grad=False)  # x4 undoes AvgPool(4,1)
        elif prime == "conv_learned":
            self.prime_conv = nn.Sequential(nn.AvgPool2d((2, 1), (2, 1)), nn.Conv2d(C_P + D, C_P * D, (FH // 2, 1)))
        self.backbone = backbone
        rn = getattr(torchvision.models, backbone)(weights="IMAGENET1K_V1" if pretrained else None)
        self.stem = nn.Sequential(rn.conv1, rn.bn1, rn.relu, rn.maxpool, rn.layer1)   # 1/4, 64 (r50: 256)
        self.l2, self.l3, self.l4 = rn.layer2, rn.layer3, rn.layer4                    # 1/8 128, 1/16 256, 1/32 512
        ch = BB_CH[backbone]                                                           # (1/4, 1/8, 1/16, 1/32) widths
        nc, hc = NECK_C, HEAD_C
        self.lat2, self.lat3, self.lat4 = nn.Conv2d(ch[1], nc, 1), nn.Conv2d(ch[2], nc, 1), nn.Conv2d(ch[3], nc, 1)
        self.smooth = conv_bn(nc, nc)
        self.fs = FS
        if FS == 4:
            # 1/4 level: smoothed 1/8 feature x2 nearest + 1x1 of the 1/4 stem feature (zero-init -> starts as the
            # upsampled 1/8 model). The ctx/depth 3x3 heads mix at 1/4. The column sum doubles with twice the rows,
            # so prime is scaled by FS/8 to match
            self.lat1 = nn.Conv2d(ch[0], nc, 1)
            nn.init.zeros_(self.lat1.weight); nn.init.zeros_(self.lat1.bias)
        self.ctx = nn.Sequential(conv_bn(nc, hc), nn.Conv2d(hc, C_P, 1))
        self.depth = nn.Sequential(conv_bn(nc, hc), nn.Conv2d(hc, D, 1))
        self.seg2d = nn.Conv2d(nc, 21, 1)
        self.hm2d = nn.Conv2d(nc, 10, 1)
        self.reg2d = nn.Conv2d(nc, 4, 1)
        self.head_v2 = HEAD_V2
        if HEAD_V2:
            self.f_x = identity_init(ResBlock(nc))          # shared block before 2D/ctx/depth (identity, f is post-ReLU)
            self.depth_x = identity_init(ResBlock(hc))
        self.vt = ({"gather": VTGather, "matmul": VTMatmul, "blockconv": VTBlockConv}[vt](taps)
                   if taps is not None else None)                  # None: training (per-sample K/T, forward_geo)
        self.float_input = False                                   # True: imgs arrive as float (no Cast in the graph)
        self.lift_lr = LIFT_LR
        self.rowband = None                   # row-band lift (bevlane/rowband_lift); set by attach()
        self.polar = None                     # polar BEV (bevlane/polar_lift.PolarLift, single-graph NPU export); set by attach()
        if LIFT_LR:
            lift_cams, g_cams = ref_structures()
            lift_src, g_src, H4, W4 = ref_structures.last
            if LR_RECT:
                self.vt_lr = BlockLowRank(lift_cams, LH, LW, C_P, D * FW, LR_BLK, LR_RANK, cell_src=lift_src, rows=D, cols=FW, margin=LR_MARGIN)
            else:
                self.vt_lr = BlockLowRank(lift_cams, LH, LW, C_P, D * FW, LR_BLK, LR_RANK)
            if GIPM:
                if LR_RECT:
                    self.gvt_lr = BlockLowRank(g_cams, GH // (2 if GLR_HALF else 1), GW // (2 if GLR_HALF else 1), GIPM_C, H4 * W4, GLR_BLK, GLR_RANK, cell_src=g_src, rows=H4, cols=W4, margin=LR_MARGIN)
                else:
                    assert not GLR_HALF, "GLR_HALF requires LR_RECT"
                    self.gvt_lr = BlockLowRank(g_cams, GH, GW, GIPM_C, H4 * W4, GLR_BLK, GLR_RANK)
            print(f"[meteorite] low-rank lift: vt {self.vt_lr.n_params/1e6:.1f}M params"
                  + (f", ground {self.gvt_lr.n_params/1e6:.1f}M" if GIPM else ""), flush=True)
        # optional LiDAR input: pillar raster [B,4,400,250] from the dataset's lidar_bev.
        # One bias-free conv -> Add after the view transform: zeros in = zeros out, so the host feeds a zero
        # raster when there is no LiDAR and the path is exactly off (no flag scalar in the graph; accelerator-safe).
        self.lidar_stem = nn.Conv2d(4, C_P, 3, 1, 1, bias=False)
        self.cam_w = CAMW
        self.paint_cls = tuple(PAINT_CLS)
        if self.paint_cls:
            self.pp_seg = nn.Conv2d(21, C_P, 1)
            nn.init.zeros_(self.pp_seg.weight); nn.init.zeros_(self.pp_seg.bias)
            pm = torch.zeros(1, 21, 1, 1); pm[:, list(self.paint_cls)] = 1.0
            self.register_buffer("pp_mask", pm, persistent=False)
            if HEAD_V2:
                self.pp_mix = identity_init(ResBlockLin(C_P))
        if CAMW:
            assert vt == "gather" or taps is None, "CAMW supports only the gather view transform"
            self.camw = nn.Conv2d(C_P, 1, 1)                     # prime [C,D,U] -> confidence logit [1,D,U]
            nn.init.zeros_(self.camw.weight); nn.init.zeros_(self.camw.bias)
        self.bev2 = BEV2
        if BEV2:
            c2, dw = BEV2_C02, BEV2_DW
            self.e0 = nn.Sequential(conv_bn(C_P, c2), ResBlock(c2))                                   # 0.2 m
            self.e1 = nn.Sequential(conv_bn(c2, BC0, s=2), ResBlock(BC0), ResBlock(BC0, d=2))        # 0.4 m
            self.e2 = nn.Sequential(conv_bn(BC0, BC1, s=2), ResBlock(BC1), ResBlock(BC1, d=2), ResBlock(BC1, d=4))  # 0.8 m
            self.lat21 = nn.Conv2d(BC1, BC0, 1)
            self.f4 = nn.Sequential(conv_bn(BC0, BCF), ResBlock(BCF))                                # 0.4 m fused
            self.lat42 = nn.Conv2d(BCF, c2, 1)
            self.f2 = nn.Sequential(conv_bn(c2, c2), ResBlock(c2, d=2))                              # 0.2 m fused
            self.seg_a = nn.Sequential(conv_bn(c2, 48), ResBlock(48, d=2), ResBlock(48, d=4))
            self.seg_out = nn.Conv2d(48, 9, 3, 1, 1)
            self.dtow = nn.Sequential(conv_bn(BCF, dw), ResBlock(dw), ResBlock(dw, d=2), ResBlock(dw, d=4))
            self.hm2 = nn.Sequential(conv_bn(dw, 64), nn.Conv2d(64, 2, 1))
            self.reg2 = nn.Sequential(conv_bn(dw, 64), nn.Conv2d(64, 6, 1))
            self.traj2 = nn.Sequential(conv_bn(dw, 64), nn.Conv2d(64, 39, 1))
            self.stat2, self.risk2 = nn.Conv2d(dw, 1, 1), nn.Conv2d(dw, 1, 1)
            nn.init.constant_(self.hm2[1].bias, -2.19)                                               # focal init (p~0.1)
            self.gipm = GIPM
            if GIPM:
                assert (GH, GW) == (LH, LW), "the BEV2 seg level must match the GIPM grid (0.2 m)"
                self.paint = conv_bn(BB_CH[backbone][0], GIPM_C, k=1)
                self.gipm_proj = nn.Sequential(conv_bn(GIPM_C, 48), nn.Conv2d(48, 48, 1))
                nn.init.zeros_(self.gipm_proj[1].weight); nn.init.zeros_(self.gipm_proj[1].bias)
                self.gvt = None
            self.det_head = 0
        else:
            # BEV encoder: L0 400x250 (64) / L1 200x125 (128), nearest x2 + lateral
            self.b0 = nn.Sequential(conv_bn(C_P, BC0), ResBlock(BC0))
            self.b1 = nn.Sequential(conv_bn(BC0, BC1, s=2), ResBlock(BC1), ResBlock(BC1, d=2), ResBlock(BC1, d=4))
            self.up_lat = nn.Conv2d(BC1, BC0, 1)
            self.fuse = nn.Sequential(conv_bn(BC0, BCF), ResBlock(BCF))
            self.lane = nn.Sequential(conv_bn(BCF, 48), nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(48, 9, 3, 1, 1))
            if HEAD_V2:
                self.seg_x0 = identity_init(ResBlock(48))                                         # 0.4 m
                self.seg_up = conv_bn(48, 32)                                                     # 0.2 m branch
                self.seg_x1 = identity_init(ResBlock(32, d=2))
                self.seg_o = nn.Conv2d(32, 9, 3, 1, 1)
                nn.init.zeros_(self.seg_o.weight); nn.init.zeros_(self.seg_o.bias)
            self.gipm = GIPM
            if GIPM:
                self.paint = conv_bn(BB_CH[backbone][0], GIPM_C, k=1)                       # 1/4 stem features -> paint channels
                self.gipm_proj = nn.Sequential(conv_bn(GIPM_C, 48), nn.Conv2d(48, 48, 1))
                nn.init.zeros_(self.gipm_proj[1].weight); nn.init.zeros_(self.gipm_proj[1].bias)   # starts == no branch
                self.gvt = None                                            # baked GroundGather (export) / None (training)
            self.hm, self.reg = nn.Conv2d(BCF, 2, 1), nn.Conv2d(BCF, 6, 1)
            self.det_head = DET_HEAD
            if DET_HEAD:
                # 3x3 + two residual blocks (d=1,2; ~±3 m extra receptive field on the 0.4 m grid). The output 1x1 is
                # zero-initialised and added to the existing hm/reg -> function-preserving (can be grafted onto a trained ckpt)
                self.det = nn.Sequential(conv_bn(BCF, DET_HEAD), ResBlock(DET_HEAD), ResBlock(DET_HEAD, d=2))
                self.hm_x, self.reg_x = nn.Conv2d(DET_HEAD, 2, 1), nn.Conv2d(DET_HEAD, 6, 1)
                for c in (self.hm_x, self.reg_x):
                    nn.init.zeros_(c.weight); nn.init.zeros_(c.bias)
            self.stationary, self.risk = nn.Conv2d(BCF, 1, 1), nn.Conv2d(BCF, 1, 1)
            self.traj = nn.Sequential(conv_bn(BCF, 64), nn.Conv2d(64, 39, 1))
        self.occ_on = bool(OCC_HEAD)
        if OCC_HEAD:
            # crop ±40 m from the 0.4 m BEV feature (400x250: rows x +80..-80, cols y +50..-50 -> rows 100:300, cols 25:225 = 200x200)
            # -> 3x3 conv-bn-relu -> 1x1 to Z*C channels. Output is 4D [B, OCC_C*OCC_Z, 200, 200] (NPU-friendly; viewed as
            # [B, C, Z, H, W] in training / demo). Final 1x1 is zero-initialised (uniform logits when grafted)
            self.occ_stem = conv_bn(BCF, 128)
            self.occ_head = nn.Conv2d(128, OCC_Z * OCC_C, 1)
            nn.init.zeros_(self.occ_head.weight); nn.init.zeros_(self.occ_head.bias)
        # planner: from L1, two stride-2 convs, fixed-kernel average pool -> 5x4 -> MLP
        pool = (nn.Sequential(nn.AvgPool2d((5, 4), (5, 4)), nn.AvgPool2d(2, 2)) if (conv1x1 or LIFT_LR) else nn.AvgPool2d((10, 8), (10, 8)))
        if BEV_FRONT:                                    # front grid 200x128 -> L1 100x64 -> /4 = 25x16 -> 5x4 (MLP input size unchanged)
            pool = nn.AvgPool2d((5, 4), (5, 4))
        self.ego_stem = nn.Sequential(conv_bn(BC1, BC1, s=2), conv_bn(BC1, BC1, s=2), pool)
        self.ego_mlp = nn.Sequential(nn.Linear(BC1 * 5 * 4 + 1, 256), nn.ReLU(inplace=True), nn.Linear(256, 42))
        self.ego_acc = False                 # current acceleration (CAN-like, /3) as an extra ego_mlp input (enable_ego_acc)
        self.tl = nn.Linear(BC1 * 5 * 4, 4)
        self.ego_gap = bool(EGO_GAP)
        if EGO_GAP:                                      # single-graph NPU export: GAP -> MLP (input BC1+1); exported as 1x1 Convs (no Flatten/Gemm)
            self.ego_stem = nn.Sequential(conv_bn(BC1, BC1, s=2), conv_bn(BC1, BC1, s=2), nn.AdaptiveAvgPool2d(1))
            self.ego_mlp = nn.Sequential(nn.Linear(BC1 + 1, 256), nn.ReLU(inplace=True), nn.Linear(256, 42))
            self.tl = nn.Linear(BC1, 4)
        self.register_buffer("mean", torch.tensor(IMG_MEAN).view(1, 3, 1, 1) / torch.tensor(IMG_STD).view(1, 3, 1, 1))
        self.register_buffer("scale", 1.0 / (255.0 * torch.tensor(IMG_STD).view(1, 3, 1, 1)))

    def _up2(self, x):
        """Nearest x2 upsampling. In the NPU-simplified mode (LIFT_LR) the same values come from a weight-1 transposed conv
        (ConvTranspose, groups=C), so no Resize appears in the graph."""
        if not self.lift_lr or LR_CONV1X1:                      # conv1x1 mode: some accelerators accept Resize(nearest) but not ConvTranspose(groups=C)
            return F.interpolate(x, scale_factor=2, mode="nearest")
        C = int(x.shape[1])
        name = f"_up2w_{C}"
        if not hasattr(self, name) or getattr(self, name).device != x.device:   # constant weight (buffer) -> ONNX initializer (shape survives tracing)
            self.register_buffer(name, torch.ones(C, 1, 2, 2, device=x.device), persistent=False)
        w = getattr(self, name).to(x.dtype)
        return F.conv_transpose2d(x, w, stride=2, groups=C)

    def enable_ego_acc(self):
        """Widen the ego_mlp[0] input by one column (new column zero = function-preserving). No-op if already widened."""
        if self.ego_acc:
            return
        old = self.ego_mlp[0]
        new = nn.Linear(old.in_features + 1, old.out_features).to(old.weight.device)
        with torch.no_grad():
            new.weight.zero_(); new.weight[:, :old.in_features] = old.weight; new.bias.copy_(old.bias)
        self.ego_mlp[0] = new
        self.ego_acc = True

    def enable_ae2e(self):
        """Optional extra inputs: ego-motion history [32] and a map / route raster [10,200,125] (dataset.ae2e_inputs).
        Both are zero-initialised additive branches (function-preserving): the raster enters at the BEV network input
        (same place as the LiDAR raster), the history at the ego_mlp hidden layer (before the ReLU).
        Set self._ae_hist / self._ae_ras before forward_geo (like acc; if absent the plain path is used)."""
        if getattr(self, "ae2e", False):
            return
        dev = self.ego_mlp[0].weight.device
        cp = self.lidar_stem.out_channels
        self.ae_ras_stem = nn.Sequential(nn.Conv2d(10, 32, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv2d(32, cp, 1)).to(dev)
        self.ae_hist_mlp = nn.Sequential(nn.Linear(32, 128), nn.ReLU(inplace=True), nn.Linear(128, self.ego_mlp[0].out_features)).to(dev)
        for m_ in (self.ae_ras_stem[2], self.ae_hist_mlp[2]):
            nn.init.zeros_(m_.weight); nn.init.zeros_(m_.bias)
        self.ae2e = True

    def _ego_out(self, g, v0, acc, B):
        x = self._ego_in(g, v0, acc, B)
        h_ = getattr(self, "_ae_hist", None) if getattr(self, "ae2e", False) else None
        if h_ is None:
            return self.ego_mlp(x)
        h = self.ego_mlp[0](x) + self.ae_hist_mlp(h_.to(x.dtype))
        return self.ego_mlp[2](self.ego_mlp[1](h))

    def _ego_in(self, g, v0, acc, B):
        parts = [g, v0.view(B, 1).to(g.dtype)]
        if self.ego_acc:
            a = acc if acc is not None else getattr(self, "_acc", None)
            a = (a.view(B, 1).to(g.dtype) / 3.0) if a is not None else g.new_zeros(B, 1)
            parts.append(a)
        return torch.cat(parts, 1)

    @staticmethod
    def _lin(x, layer):
        """nn.Linear as MatMul + Add (no Gemm in the graph)."""
        return torch.matmul(x, layer.weight.t()) + layer.bias

    def _heads(self, bev, lidar=None):
        """Everything after the view transform (shared between export forward and forward_geo)."""
        r_ = getattr(self, "_ae_ras", None) if getattr(self, "ae2e", False) else None
        if r_ is not None:                                       # map / route raster (zero-initialised additive branch)
            bev = bev + F.interpolate(self.ae_ras_stem(r_.to(bev.dtype)), size=bev.shape[-2:], mode="bilinear", align_corners=False)
        if self.bev2:
            if lidar is not None:                                # LiDAR raster stays at 0.4 m -> x2 after the stem
                bev = bev + self._up2(self.lidar_stem(lidar))
            l0 = self.e0(bev); l1 = self.e1(l0); l2 = self.e2(l1)
            m4 = self.f4(l1 + self._up2(self.lat21(l2)))
            m2 = self.f2(l0 + self._up2(self.lat42(m4)))
            return (m2, m4), self.ego_stem(l2).flatten(1)
        if lidar is not None:
            bev = bev + self.lidar_stem(lidar)
        l0 = self.b0(bev)
        l1 = self.b1(l0)
        m = self.fuse(l0 + self._up2(self.up_lat(l1)))
        self._fused_bev = m          # F3 refiner: expose the fused BEV (BCF=96 ch) (no effect on training or export)
        self._l1 = l1                # polar export (EGO_GAP): use the 4D ego_stem output without a Flatten
        g = self.ego_stem(l1).flatten(1)
        return m, g

    def _paint(self, ctx, s2, last_axis=False):
        """PointPainting: ctx + Conv1x1(selected classes of softmax(seg2d)). s2 = seg2d logits (pass detached).
        last_axis: for export. NPUs often restrict Softmax to the last axis (npu_lint) -> transpose and take it on the
        last axis (identical numerics)."""
        if not self.paint_cls:
            return ctx
        s2f = s2 if s2.dtype == torch.float32 else s2.float()                  # no Cast when already float32 (simpler NPU graph)
        if last_axis:
            pr = s2f.permute(0, 2, 3, 1).softmax(-1).permute(0, 3, 1, 2)
        else:
            pr = s2f.softmax(1)
        if pr.dtype != ctx.dtype:
            pr = pr.to(ctx.dtype)
        w_ = self.pp_seg.weight * self.pp_mask
        out = ctx + F.conv2d(pr, w_ if w_.dtype == ctx.dtype else w_.to(ctx.dtype), self.pp_seg.bias if self.pp_seg.bias.dtype == ctx.dtype else self.pp_seg.bias.to(ctx.dtype))
        return self.pp_mix(out) if self.head_v2 else out

    def _depth(self, f):
        if not self.head_v2:
            return self.depth(f)
        return self.depth[1](self.depth_x(self.depth[0](f)))

    def _outs(self, m, gfeat=None):
        """(lane, hm, reg, stationary, risk, traj); m is the first return value of _heads."""
        if self.bev2:
            m2, m4 = m
            h = self.seg_a(m2)
            if gfeat is not None:
                h = h + self.gipm_proj(gfeat)
            t = self.dtow(m4)
            return self.seg_out(h), self.hm2(t), self.reg2(t), self.stat2(t), self.risk2(t), self.traj2(t)
        hm, reg = self._det(m)
        return self._lane(m, gfeat), hm, reg, self.stationary(m), self.risk(m), self.traj(m)

    def _occ(self, m):
        """OCC head (OCC_HEAD=1): m is the 0.4 m fused BEV feature [B, BCF, 400, 250] -> [B, OCC_C*OCC_Z, 200, 200]. Not supported for bev2 (m is 0.2 m)."""
        if self.bev2:
            m = m[1] if isinstance(m, (tuple, list)) else m
        r0 = int(round((BEV_XF - 40.0) / 0.4)); c0 = int(round((BEV_YH - 40.0) / 0.4))
        x = m[:, :, r0:r0 + 200, c0:c0 + 200]
        if self.training and OCC_GSCALE != 1.0:      # gradient scale from OCC into the shared BEV (0 = train the head only, other tasks unaffected)
            x = _GradScale.apply(x, OCC_GSCALE)
        return self.occ_head(self.occ_stem(x))

    def _det(self, m):
        if self.training and BOX_GSCALE != 1.0:
            m = _GradScale.apply(m, BOX_GSCALE)
        if not self.det_head:
            return self.hm(m), self.reg(m)
        d = self.det(m)
        return self.hm(m) + self.hm_x(d), self.reg(m) + self.reg_x(d)

    def _up_lane(self, x):
        """Lane head upsampling. For polar BEV (self.polar) only the row (range) axis is x2 (320 image columns are enough
        laterally, and 640 is too wide for some NPU compilers)."""
        if self.polar is not None:
            return F.interpolate(x, scale_factor=(2, 1), mode="nearest")
        return self._up2(x)

    def _lane(self, m, gfeat=None):
        if not self.head_v2:
            if gfeat is None:
                return self.lane[2](self._up_lane(self.lane[0](m)))
            h = self._up_lane(self.lane[0](m))
            return self.lane[2](h + self.gipm_proj(gfeat))
        h = self._up_lane(self.seg_x0(self.lane[0](m)))                   # mix at 0.4 m, then x2 (rows only for polar)
        if gfeat is not None:
            h = h + self.gipm_proj(gfeat)
        return self.lane[2](h) + self.seg_o(self.seg_x1(self.seg_up(h)))      # baseline output + new 0.2 m branch (zero-init)

    def _ground_geo(self, s4, K, T_cam_ego, h0, w0):
        """training path of the ground IPM: per-sample K/T, nearest sampling (== the baked table)."""
        BN, C, H4, W4 = s4.shape
        B = K.shape[0]; N = BN // B
        if not hasattr(self, "_gpts_t") or self._gpts_t.device != s4.device:
            self._gpts_t = torch.from_numpy(ground_cell_centres().astype("float32")).to(s4.device)
        G = self._gpts_t.shape[0]
        pc = torch.matmul(T_cam_ego.reshape(BN, 4, 4).float(), self._gpts_t.t().unsqueeze(0).expand(BN, 4, G))
        X, Y, Z = pc[:, 0], pc[:, 1], pc[:, 2]
        Kf = K.reshape(BN, 3, 3).float()
        sx, sy = IMG_W / w0, IMG_H / h0
        Zs = torch.where(Z > D_MIN, Z, torch.ones_like(Z))
        u = ((Kf[:, 0, 0:1] * sx) * X / Zs + Kf[:, 0, 2:3] * sx + 0.5) / 4 - 0.5
        v = ((Kf[:, 1, 1:2] * sy) * Y / Zs + Kf[:, 1, 2:3] * sy + 0.5) / 4 - 0.5
        ui, vi = torch.round(u), torch.round(v)
        ok = ((Z > D_MIN) & (ui >= 0) & (ui < W4) & (vi >= 0) & (vi < H4)).float()
        grid = torch.stack([ui / (W4 - 1) * 2 - 1, vi / (H4 - 1) * 2 - 1], -1).view(BN, G, 1, 2)
        vals = F.grid_sample(s4.float(), grid, mode="nearest", align_corners=True)[..., 0] * ok.unsqueeze(1)
        num = vals.view(B, N, C, G).sum(1)
        den = ok.view(B, N, 1, G).sum(1).clamp(min=1.0)
        return (num / den).view(B, C, GH, GW).to(s4.dtype)

    def forward_geo(self, imgs, K, T_cam_ego, v0, lidar=None, acc=None):
        """Training / eval forward with per-sample geometry (the exact dense path; the baked
        table is export-time only, like METEOR's frustum lift). imgs [B,N,3,h,w] ALREADY
        normalised floats (dataset output, any h,w), K for that image size, T [B,N,4,4].
        Images are resized in-graph to IMG_H x IMG_W and K is rescaled to match."""
        B, N = imgs.shape[:2]
        if BEV_FRONT and lidar is not None:
            lidar = _front_crop(lidar)
        h0, w0 = imgs.shape[-2:]
        x = imgs.view(B * N, 3, h0, w0)
        if (h0, w0) != (IMG_H, IMG_W):
            x = F.interpolate(x, (IMG_H, IMG_W), mode="bilinear", align_corners=False)
        s4 = self.stem(x)
        c2 = self.l2(s4); c3 = self.l3(c2); c4 = self.l4(c3)
        p4 = self.lat4(c4)
        # size-matched (not x2): with H=432 the 1/32 level is 14 rows but 1/16 is 27 (ceil vs floor).
        # The export graph keeps integer x2 Resize when the size divides by 32 (352x640 does).
        p3 = self.lat3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        p2 = self.lat2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="nearest")
        f = self.smooth(p2)
        if self.fs == 4:
            f = F.interpolate(f, size=s4.shape[-2:], mode="nearest") + self.lat1(s4)   # [BN,nc,H/4,W/4]
        if self.head_v2:
            f = self.f_x(f)
        s2 = self.seg2d(f)
        ctx = self._paint(self.ctx(f), s2.detach())
        dlog = _ablate_depth(self._depth(f))                                      # [BN,D,V,U]
        dprob = dlog.float().softmax(1)
        # prime map: per-column depth-weighted vertical collapse (geometry-free)
        prime = torch.matmul(ctx.float().permute(0, 3, 1, 2), dprob.permute(0, 3, 2, 1))  # [BN,U,C,V]x[BN,U,V,D]
        if self.fs != 8:
            prime = prime * (self.fs / 8.0)                        # sum over 8/FS times more rows -> same scale as the 1/8 model
        P4 = prime.permute(0, 2, 3, 1).contiguous()                # [BN, C, D, U]
        if self.polar is not None:                                  # polar BEV: no view transform; learn a Conv-only vertical collapse with P4 (MatMul prime) as teacher
            assert N == 1 and not self.cam_w and not self.lift_lr and self.rowband is None, "polar BEV requires a single camera (N_CAM=1) and no CAMW/LIFT_LR/row-band lift"
            bev_geo = P4.view(B, C_P, D, FW).to(f.dtype)
            bev = self.polar.lift(ctx, dlog, self.paint(s4) if self.gipm else None).to(f.dtype)
            extra = {"bev_lr": bev, "bev_geo": bev_geo} if self.polar.mode == 1 else {}   # distillation (liftkd) only for the depth-bin variant (same shape as P4)
            return self._finish_geo(bev, extra, v0, acc, B, N, f, s2, dlog, None, gfeat=None)   # the LiDAR raster (Cartesian grid) cannot be added
        # per-sample (cell -> (depth bin, column)) map; bilinear == the baked taps
        pts = lift_cell_centres()
        if not hasattr(self, "_pts_t") or self._pts_t.device != x.device:
            self._pts_t = torch.from_numpy(pts.astype("float32")).to(x.device)
        G2 = self._pts_t.shape[0]
        pc = torch.matmul(T_cam_ego.view(B * N, 4, 4).float(),
                          self._pts_t.t().unsqueeze(0).expand(B * N, 4, G2))
        X, Z = pc[:, 0], pc[:, 2]
        sx = IMG_W / w0
        fx = (K.view(B * N, 3, 3)[:, 0, 0] * sx).unsqueeze(1)
        cx = (K.view(B * N, 3, 3)[:, 0, 2] * sx).unsqueeze(1)
        Zs = Z.clamp(min=D_MIN)
        u = ((fx * X / Zs + cx) + 0.5) / FS - 0.5                  # feature column
        import math as _m
        b = (Zs.log() - _m.log(D_MIN)) / ((_m.log(D_MAX) - _m.log(D_MIN)) / (D - 1))
        valid = ((Z > D_MIN) & (u >= 0) & (u <= FW - 1) & (b >= 0) & (b <= D - 1)).float()
        grid = torch.stack([u / (FW - 1) * 2 - 1, b / (D - 1) * 2 - 1], -1).view(B * N, G2, 1, 2)
        # at 0.2 m (BEV2) BN*C*G2 exceeds 2^31 and cuDNN returns NOT_SUPPORTED -> split per sample
        if self.cam_w:
            E = self.camw(P4).clamp(-CAMW_CLIP, CAMW_CLIP).exp()                   # [BN,1,D,U]
            P4 = torch.cat([P4 * E, E], 1)
        num = torch.stack([
            (F.grid_sample(P4[i * N:(i + 1) * N], grid[i * N:(i + 1) * N], mode="bilinear",
                           align_corners=True, padding_mode="zeros")[..., 0]
             * valid[i * N:(i + 1) * N].unsqueeze(1)).sum(0) for i in range(B)])   # [B, C(+1), G2]
        if self.cam_w:
            bev = (num[:, :C_P] / num[:, C_P:].clamp(min=1e-4)).view(B, C_P, LH, LW).to(f.dtype)
        else:
            den = valid.view(B, N, 1, G2).sum(1).clamp(min=1.0)
            bev = (num / den).view(B, C_P, LH, LW).to(f.dtype)
        extra = {}
        if self.lift_lr:                                            # learned low-rank map (the geometric version stays as the distillation teacher)
            assert not self.cam_w, "LIFT_LR does not support CAMW"
            bev_geo = bev
            bev = self.vt_lr((P4.view(B, N, C_P, D, FW) if self.vt_lr.rect else P4.view(B, N, C_P, D * FW)).to(f.dtype))
            extra = {"bev_lr": bev, "bev_geo": bev_geo}
        if self.rowband is not None:                                # row-band lift (no Gather / MatMul); the geometric version is the distillation teacher
            assert N == 1 and not self.cam_w, "row-band lift requires a single camera (N_CAM=1) and no CAMW"
            bev_geo = bev
            bev = self.rowband.lift(ctx.float(), dlog.float()).to(f.dtype)
            extra = {"bev_lr": bev, "bev_geo": bev_geo}
        gfeat = self._ground_geo(self.paint(s4), K, T_cam_ego, h0, w0) if self.gipm else None
        if self.gipm and self.rowband is not None and getattr(self.rowband, "use_ground", True):   # --rowband-ground 0: ground keeps the original Gather (row band for vt only)
            g_h = self.rowband.ground(self.paint(s4).float()).to(f.dtype)
            extra.update({"g_lr": g_h, "g_geo": gfeat})
            gfeat = g_h
        if self.gipm and self.lift_lr:
            pf = self.paint(s4)
            g_lr = self.gvt_lr(pf.view(B, N, GIPM_C, *pf.shape[-2:]) if self.gvt_lr.rect else pf.view(B, N, GIPM_C, -1))
            if GLR_HALF:
                g_lr = F.interpolate(g_lr, scale_factor=2, mode="nearest")
            extra.update({"g_lr": g_lr, "g_geo": gfeat})
            gfeat = g_lr
        return self._finish_geo(bev, extra, v0, acc, B, N, f, s2, dlog, lidar, gfeat)

    def _finish_geo(self, bev, extra, v0, acc, B, N, f, s2, dlog, lidar, gfeat):
        """Second half of forward_geo (BEV network -> heads -> canonical grid). Polar BEV (self.polar) also goes through
        here: head outputs are resampled to the Cartesian grid with grid_sample."""
        m, g = self._heads(bev, lidar=lidar)
        ge = _GradScale.apply(g, EGO_GSCALE) if (self.training and EGO_GSCALE != 1.0) else g
        ego = self._ego_out(ge, v0, acc, B)
        lane, hm, reg, st, rk, tj = self._outs(m, gfeat)
        if self.polar is not None:                                  # polar -> Cartesian (lane on the 0.2 m grid, others 0.4 m); lane outside the FOV = background
            bg = torch.full((lane.shape[1],), -20.0, device=lane.device); bg[0] = 20.0
            lane = self.polar.to_cart(lane, "lane", fill=bg)
            neg = torch.full((1,), -10.0, device=lane.device)              # logit outputs (stationary / risk) outside the FOV: strongly negative (0 would give p=0.5 and blow up focal loss)
            st = self.polar.to_cart(st, "lift", fill=neg); rk = self.polar.to_cart(rk, "lift", fill=neg)
            tj = self.polar.to_cart(tj, "lift")
            extra = dict(extra, hm_polar=hm, reg_polar=reg)                 # boxes are trained on the polar grid (the box loss uses polar.det_loss)
            if self.training:
                hm = self.polar.to_cart(hm, "lift", fill=neg.expand(hm.shape[1])); reg = self.polar.to_cart(reg, "lift")   # shape matching only (not used by the loss)
            else:                                                            # eval: decode in polar, then redraw on the Cartesian map (compatible with eval_unified decode_boxes)
                bl = self.polar.decode(hm, reg)
                hm, reg = self.polar.render_cart(bl, hm.shape[0], hm.device, hm.dtype)
        if getattr(self, "occ_on", False):
            extra["occ"] = self._occ(m)
        if BEV_FRONT:                                               # external outputs on the canonical grid (loss / eval / drawing unchanged)
            lane, hm, reg = _front_pad(lane, -20.0), _front_pad(hm, -20.0), _front_pad(reg, 0.0)
            lane[:, 0] = torch.where(lane[:, 0] == -20.0, torch.full_like(lane[:, 0], 20.0), lane[:, 0])   # outside the grid = background (no stripe artefacts in demos; loss / eval ignore it via the front mask)
            st, rk, tj = _front_pad(st, -20.0), _front_pad(rk, -20.0), _front_pad(tj, 0.0)   # risk is a logit (avoid a p=0.5 band)
            if self.rowband is None and self.polar is None:         # row-band / polar lift distillation (bev_lr/bev_geo, g_lr/g_geo) uses the internal grid in the loss
                extra = {k: v for k, v in extra.items() if k not in ("bev_lr", "bev_geo", "g_lr", "g_geo", "occ")} if extra else extra
        return {**extra, "lane": lane, "hm": hm, "reg": reg,
                "stationary": st, "risk": rk, "traj": tj,
                "ego": ego, "tl": self.tl(g),
                "dlog": dlog.view(B, N, D, FH, FW),
                "seg2d": s2.view(B, N, 21, FH, FW),
                "hm2d": self.hm2d(f).view(B, N, 10, FH, FW),
                "reg2d": self.reg2d(f).view(B, N, 4, FH, FW)}

    @torch.no_grad()
    def prepare_npu(self):
        """Pre-processing for NPU export (exactly equivalent; no Pad/Mul/Sub/spatial bias map):
           x_norm = (x - 255*mean_img) * (1/(255 std)) -> the mean subtraction becomes a per-channel constant Add on the
           input (in_bias_npu); the scale and the stem BN are folded into the conv1 weight (stem_w_npu) and a per-channel
           constant bias (stem_b_npu)."""
        c0, bn = self.stem[0], self.stem[1]
        W = c0.weight.detach().float()
        w = W * self.scale.view(1, 3, 1, 1).float()
        b = c0.bias.detach().float() if c0.bias is not None else torch.zeros(W.shape[0], device=W.device)
        g = bn.weight.detach().float() / torch.sqrt(bn.running_var.float() + bn.eps)
        w = w * g.view(-1, 1, 1, 1)
        b = (b - bn.running_mean.float()) * g + bn.bias.detach().float()
        self.register_buffer("in_bias_npu", (-255.0 * torch.tensor(IMG_MEAN, device=W.device)).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("stem_w_npu", w.contiguous(), persistent=False)
        self.register_buffer("stem_b_npu", b.contiguous(), persistent=False)

    def _forward_npu(self, imgs, v0, acc=None):                                   # acc [1,1]: causal acceleration m/s^2 (only when the F3 --extra acc variant is bundled)
        """Strict-NPU export (NPU_MODE="npu"). imgs uint8 [N,3,H,W] (normalisation folded into the stem), v0 [1,1].
        Tensors of rank <= 4 only; MatMul only as 3D bmm (prime) and 2D FC (ego/tl); low-rank layers as Conv1x1."""
        assert self.lift_lr and not self.cam_w and self.prime == "matmul"
        N = N_CAM
        x = imgs.float() if imgs.dtype != torch.float32 else imgs
        if not hasattr(self, "stem_w_npu"):
            self.prepare_npu()
        c0 = self.stem[0]
        # normalisation: mean subtraction is a per-channel constant Add on the input (exact with zero padding: padding 0 =
        # raw pixel 255*mean = 0 after normalisation); 1/(255 std) and BN are folded into the conv1 weight and a per-channel
        # constant bias (a spatial bias map does not quantise well)
        x = x + self.in_bias_npu
        s4 = F.relu(F.conv2d(x, self.stem_w_npu, self.stem_b_npu, stride=c0.stride, padding=c0.padding))
        for mod in list(self.stem)[3:]:                                            # maxpool, layer1
            s4 = mod(s4)
        c2 = self.l2(s4); c3 = self.l3(c2); c4 = self.l4(c3)
        p4 = self.lat4(c4)
        s16 = (-(-IMG_H // 16), -(-IMG_W // 16)); s8 = (-(-IMG_H // 8), -(-IMG_W // 8))
        p3 = self.lat3(c3) + self._up2(p4)[..., :s16[0], :s16[1]]
        p2 = self.lat2(c2) + self._up2(p3)[..., :s8[0], :s8[1]]
        f = self.smooth(p2)
        if self.fs == 4:
            s4hw = (-(-IMG_H // 4), -(-IMG_W // 4))
            f = self._up2(f)[..., :s4hw[0], :s4hw[1]] + self.lat1(s4)
        if self.head_v2:
            f = self.f_x(f)
        s2 = self.seg2d(f)
        ctx = self._paint(self.ctx(f), s2, last_axis=False)                     # Softmax dim=1
        dlog = _ablate_depth(self._depth(f))                                                    # [N, D, V, U]
        dprob = dlog.softmax(1)                                                  # dim=1
        Vh, U = FH, FW
        if LR_HCONV and LR_MM4D:                                                 # 4D MatMul: [N, U, D, V] @ [N, U, V, C] -> [N, U, D, C] (directly the Slice input)
            polar = torch.matmul(dprob.permute(0, 3, 1, 2), ctx.permute(0, 3, 2, 1))
        else:
            a_ = ctx.permute(0, 3, 1, 2).reshape(N * U, C_P, Vh)                 # [N*U, C, V]
            b_p = dprob.permute(0, 3, 2, 1).reshape(N * U, Vh, D)                # [N*U, V, D]
            polar = torch.bmm(a_, b_p)                                           # [N*U, C, D]  (3D bmm)
        # (the fs/8 constant factor is folded into the U weights: a Mul right after the bmm quantises poorly)
        # low-rank layer input: [1, N*S, 1, C], S = D*U (index d*U + u), matching the U_bn row order
        if LR_HCONV and LR_MM4D:
            Xl = polar                                                           # [N, U, D, C] (MatMul output as is)
        elif LR_HCONV:                                                           # [N, U(cols), D(rows), C]: one Transpose; per pair Slice + Conv(kernel=(rows,1))
            Xl = polar.reshape(N, U, C_P, D).permute(0, 1, 3, 2)
        else:
            Xl = polar.reshape(N, U, C_P, D).permute(0, 3, 1, 2).reshape(N, D * U, 1, C_P).reshape(1, N * D * U, 1, C_P)
        bev = self.vt_lr.forward_npu(Xl, D * U, C_P, in_scale=(self.fs / 8.0))    # [1, C, LH, LW]
        m, g = self._heads(bev)
        e_in = self._ego_in(g, v0, None, 1)
        ego = self._lin(F.relu(self._lin(e_in, self.ego_mlp[0])), self.ego_mlp[2])
        tl_out = self._lin(g, self.tl)
        gfeat = None
        if self.gipm:
            pf = self.paint(s4)                                                  # [N, GC, H4, W4]
            H4, W4 = pf.shape[-2:]
            if LR_HCONV:                                                         # [N, W4(cols), H4(rows), GC]
                Xg = pf.permute(0, 3, 2, 1)
            else:
                Xg = pf.reshape(N, GIPM_C, H4 * W4).permute(0, 2, 1).reshape(1, N * H4 * W4, 1, GIPM_C)
            gfeat = self.gvt_lr.forward_npu(Xg, H4 * W4, GIPM_C)                 # [1, GC, GH, GW] (GLR_HALF: half resolution -> x2)
            if GLR_HALF:
                if True:                                                         # insert an identity Conv1x1 between the BEV-assembly Reshape and the ConvTranspose
                    gc_ = int(gfeat.shape[1])                                    #   (some NPU compilers reject that pair; numpy constant because torch.eye traces to EyeLike)
                    eye = torch.from_numpy(np.eye(gc_, dtype=np.float32)).view(gc_, gc_, 1, 1).to(gfeat.device, gfeat.dtype)
                    gfeat = F.conv2d(gfeat, eye)
                gfeat = self._up2(gfeat)
        outs = (*self._outs(m, gfeat), ego, tl_out, dprob, s2, self.hm2d(f), self.reg2d(f))   # auxiliary outputs are 4D [N, C, V, U]
        if getattr(self, "occ_on", False):
            outs = outs + (self._occ(m),)                                      # "occ" [1, 160, 200, 200] (OCC-head ckpts only)
        f3 = getattr(self, "f3_npu", None)
        if f3 is not None:                                                     # F3 residual head (NPU-friendly form) in the same graph: refines all 3 candidates into "ego_f3" [1, 36].
            #   Candidate selection (argmax / acceleration rule) is post-processing on the CPU, so the graph has no data-dependent ops.
            #   acc-input variant (train_f3 --extra acc): graph input "acc" [1,1] (m/s^2) is divided by 3 into ext (same as train_f3.ext_of).
            if f3.extra:
                assert f3.extra in (1, 2), "F3 extra input must be acc (1) or acc,brk (2)"
                if acc is None:                                                # verification forwards (e.g. pre-export parity) run with 0; export() passes graph input "acc" [1,1] / "accbrk" [1,2]
                    acc = torch.zeros(1, f3.extra, device=v0.device, dtype=torch.float32)
                # raw inputs (acc m/s^2, brk 0/1): apply the train_f3.ext_of normalisation [1/3, 1] as a constant Mul
                esc = torch.from_numpy(np.array([1.0 / 3.0, 1.0][:f3.extra], np.float32)).view(1, f3.extra, 1, 1).to(v0.device)
            gf = f3.trunk(F.avg_pool2d(m, self._f3_poolk))                     # AveragePool(k) -> Conv x3 (BN) -> GlobalAveragePool [1, 256, 1, 1]
            cand = ego[:, :36].reshape(3, 12, 1, 1)                            # Slice + Reshape (candidates as batch 3)
            g3 = torch.cat([gf, gf, gf], 0)                                    # avoids Expand
            v3 = torch.cat([v0.reshape(1, 1, 1, 1)] * 3, 0) * 0.1
            if f3.extra:
                e3 = torch.cat([acc.reshape(1, f3.extra, 1, 1)] * 3, 0) * esc
                outs = outs + (f3.mlp_res(g3, cand, v3, e3).reshape(1, 36),)
            else:
                outs = outs + (f3.mlp_res(g3, cand, v3).reshape(1, 36),)
        return outs

    def forward(self, imgs, v0, lidar=None, acc=None):
        # Cast FIRST: TensorRT rejects uint8 intermediate tensors (a Reshape on the raw uint8 input fails to parse)
        ih, iw = IN_HW or (IMG_H, IMG_W)
        if NPU_MODE == "npu":
            return self._forward_npu(imgs, v0, acc)
        if LR_CONV1X1 and self.lift_lr and (ih, iw) == (IMG_H, IMG_W):
            # conv1x1 mode: fold normalisation (x*scale - mean) into the first conv (no broadcast Sub needed).
            # Only the zero-padded border differs slightly (padding pixels become 0 instead of the -mean equivalent).
            x = (imgs if self.float_input else imgs.float()).view(N_CAM, 3, ih, iw)
            c0 = self.stem[0]
            w_ = c0.weight * self.scale.view(1, 3, 1, 1)
            b_ = -(c0.weight * self.mean.view(1, 3, 1, 1)).sum((1, 2, 3))
            s4 = F.conv2d(x, w_, b_, stride=c0.stride, padding=c0.padding)
            for mod in list(self.stem)[1:]:
                s4 = mod(s4)
        else:
            x = (imgs if self.float_input else imgs.float()).view(N_CAM, 3, ih, iw) * self.scale - self.mean   # uint8 -> normalised
            if (ih, iw) != (IMG_H, IMG_W):
                # run a network trained at IMG_H x IMG_W on a different engine input size: same bilinear resize as
                # forward_geo (normalisation is a per-pixel affine map, so it commutes with resizing). Set IN_HW=None to disable
                x = F.interpolate(x, (IMG_H, IMG_W), mode="bilinear", align_corners=False)
            s4 = self.stem(x)
        c2 = self.l2(s4); c3 = self.l3(c2); c4 = self.l4(c3)
        p4 = self.lat4(c4)
        # constant sizes (python ints, no Shape op in the graph): conv stride-2 chain = ceil division
        s16 = (-(-IMG_H // 16), -(-IMG_W // 16)); s8 = (-(-IMG_H // 8), -(-IMG_W // 8))
        # x2 nearest + constant crop == size-matched nearest for these ceil sizes (floor(i*n/(2n-1)) == floor(i/2)
        # for i < 2n-1), so the graph stays Resize(scales)+Slice with no Shape op and matches forward_geo exactly
        p3 = self.lat3(c3) + self._up2(p4)[..., :s16[0], :s16[1]]
        p2 = self.lat2(c2) + self._up2(p3)[..., :s8[0], :s8[1]]
        f = self.smooth(p2)                                                      # [N,128,44,80]
        if self.fs == 4:
            s4hw = (-(-IMG_H // 4), -(-IMG_W // 4))
            f = self._up2(f)[..., :s4hw[0], :s4hw[1]] + self.lat1(s4)
        if self.head_v2:
            f = self.f_x(f)
        s2 = self.seg2d(f)
        ctx = self._paint(self.ctx(f), s2, last_axis=True)                       # [N,C,V,U]
        dlog = _ablate_depth(self._depth(f))                                                    # [N,D,V,U]
        if self.polar is not None:                                               # polar BEV: no view transform; outputs on the polar grid (lane [9,2D,2U], others [.,D,U]); the fixed remap to Cartesian runs on the host
            assert N_CAM == 1 and not self.cam_w and not self.lift_lr
            bev = self.polar.lift(ctx, dlog, self.paint(s4) if self.gipm else None)   # polar.export_unpack=True expands with ConvTranspose (no Reshape)
            m, g = self._heads(bev)
            if self.ego_gap:                                                 # no Flatten/Gemm: GAP [1,BC1,1,1] + v0 [1,1,1,1] -> Concat -> 1x1 Conv x2 (weights are reshaped Linear weights)
                g4 = self.ego_stem(self._l1); v4 = v0.reshape(1, 1, 1, 1).to(g4.dtype)   # [1,BC1,1,1] (no Flatten; the flattened g is unused and pruned)
                e = torch.cat([g4, v4], 1)
                e = F.relu(F.conv2d(e, self.ego_mlp[0].weight.view(256, -1, 1, 1), self.ego_mlp[0].bias))
                ego = F.conv2d(e, self.ego_mlp[2].weight.view(42, 256, 1, 1), self.ego_mlp[2].bias).view(1, 42)
                tl_out = F.conv2d(g4, self.tl.weight.view(4, -1, 1, 1), self.tl.bias).view(1, 4)
            else:
                ego = self.ego_mlp(self._ego_in(g, v0, None, 1)); tl_out = self.tl(g)
            dprob = dlog.softmax(1)
            return (*self._outs(m, None), ego, tl_out, dprob.view(1, N_CAM, D, FH, FW),
                    s2.view(1, N_CAM, 21, FH, FW), self.hm2d(f).view(1, N_CAM, 10, FH, FW), self.reg2d(f).view(1, N_CAM, 4, FH, FW))
        if self.prime == "conv_exact":
            dp = dlog.softmax(1)                                                 # [N,D,V,U] channel-axis softmax
            prod = self.tile_c(ctx) * self.tile_d(dp)                            # [N,C*D,V,U] = ctx[c] * dprob[d]  (same shape, no broadcast)
            P = self.vsum(F.avg_pool2d(prod, (4, 1), (4, 1)))                    # sum over v -> [N,C*D,1,U]
            P = P.view(N_CAM, C_P, D * FW)                                       # [N,C,D*U]
            dprob = dp.permute(0, 3, 2, 1)
        elif self.prime == "conv_learned":
            dp = dlog.softmax(1)
            P = self.prime_conv(torch.cat([ctx, dlog], 1)).view(N_CAM, C_P, D * FW)   # learned column->(c,d) map, no product
            dprob = dp.permute(0, 3, 2, 1)
        else:
            if self.conv1x1:
                dprob = dlog.softmax(1).permute(0, 3, 2, 1)                      # conv1x1 mode: softmax on the channel axis, then [N,U,V,D]
            else:
                dprob = dlog.permute(0, 3, 2, 1).softmax(-1)                     # [N,U,V,D]  softmax on the last axis
            prime = torch.matmul(ctx.permute(0, 3, 1, 2), dprob)                 # [N,U,C,V] x [N,U,V,D] -> [N,U,C,D]
            if self.fs != 8:
                prime = prime * (self.fs / 8.0)
            P = prime.permute(0, 2, 3, 1).reshape(N_CAM, C_P, D * FW)            # [N,C,D*U]
        if self.cam_w:
            E = self.camw(P.view(N_CAM, C_P, D, FW)).clamp(-CAMW_CLIP, CAMW_CLIP).exp().view(N_CAM, 1, D * FW)
            s = self.vt(torch.cat([P * E, E], 1))                                # [1,C+1,LH,LW]
            bev = s[:, :C_P] / (s[:, C_P:] + 1e-6)                               # cells no camera sees: 0/1e-6 = 0
        elif self.lift_lr:
            Xl = P.view(1, N_CAM, C_P, D * FW)
            bev = self.vt_lr.forward_conv1x1(Xl) if LR_CONV1X1 else self.vt_lr(Xl)      # learned low-rank map (no Gather)
        else:
            bev = self.vt(P)                                                     # [1,C,400,250]
        m, g = self._heads(bev, lidar=lidar)
        if self.lift_lr:
            e_in = self._ego_in(g, v0, None, 1)
            ego = self._lin(F.relu(self._lin(e_in, self.ego_mlp[0])), self.ego_mlp[2])
            tl_out = self._lin(g, self.tl)
        else:
            ego = self.ego_mlp(self._ego_in(g, v0, None, 1)); tl_out = self.tl(g)
        if self.gipm and self.lift_lr:
            Xg = self.paint(s4).view(1, N_CAM, GIPM_C, -1)
            gfeat = self.gvt_lr.forward_conv1x1(Xg) if LR_CONV1X1 else self.gvt_lr(Xg)
            if GLR_HALF:
                gfeat = self._up2(gfeat)
        else:
            gfeat = self.gvt(self.paint(s4)) if self.gipm else None
        outs = (*self._outs(m, gfeat),
                ego, tl_out,
                dprob.permute(0, 3, 2, 1).reshape(1, N_CAM, D, FH, FW),
                s2.view(1, N_CAM, 21, FH, FW), self.hm2d(f).view(1, N_CAM, 10, FH, FW),
                self.reg2d(f).view(1, N_CAM, 4, FH, FW))
        if getattr(self, "occ_on", False):
            outs = outs + (self._occ(m),)                                      # "occ" [1, 160, 200, 200]
        return outs


OUTPUT_NAMES = ["lane_logit", "hm", "reg", "stationary", "risk", "traj", "ego", "tl",
                "depth", "seg2d", "hm2d", "reg2d"]


def export(net, path, opset=17, with_lidar=False):
    net.eval()
    if NPU_MODE == "npu":
        imgs = torch.zeros(N_CAM, 3, IMG_H, IMG_W, dtype=torch.uint8); v0 = torch.zeros(1, 1)
    else:
        imgs = torch.zeros(1, N_CAM, 3, *(IN_HW or (IMG_H, IMG_W)), dtype=torch.float32 if net.float_input else torch.uint8)
        v0 = torch.zeros(1)
    args = (imgs, v0) + ((torch.zeros(1, 4, *DET_HW),) if with_lidar else ())
    names = ["imgs", "v0"] + (["lidar_bev"] if with_lidar else [])
    f3 = getattr(net, "f3_npu", None)
    if f3 is not None and getattr(f3, "extra", 0):                              # F3 acc variant: third input "acc" [1,1] (causal acceleration m/s^2) / acc,brk variant "accbrk" [1,2] (acc, brk 0/1)
        E = int(f3.extra); nm = "acc" if E == 1 else "accbrk"
        args = args + ({"acc": torch.zeros(1, E)},); names = names + [nm]
    torch.onnx.export(net, args, path, opset_version=opset, input_names=names,
                      output_names=OUTPUT_NAMES + (["occ"] if OCC_HEAD else []) + (["ego_f3"] if getattr(net, "f3_npu", None) is not None else []),
                      do_constant_folding=True, dynamo=False)
    return path



def reg_half_angle(reg):
    """YAW2 ckpt: convert the reg yaw channels (sin 2θ, cos 2θ) back to (sin θ', cos θ') (θ' in [-90°, 90°); the tracker
    resolves front/back from motion). Identity when YAW2=0. Used only by inference adapters (MiteEval / demos), never
    in the exported graph."""
    if not YAW2:
        return reg
    th = torch.atan2(reg[:, 4:5].float(), reg[:, 5:6].float()) * 0.5
    return torch.cat([reg[:, :4], torch.sin(th).to(reg.dtype), torch.cos(th).to(reg.dtype), reg[:, 6:]], 1)

# ----------------------------------------------------------------------------- evaluation adapter
class MiteEval(nn.Module):
    """closed_loop_eval / probe_e2e_tail_stop adapter: METEORite answers with a v52-like
    tuple (len 8: lane, dlog, seg2d, hm, reg, stationary, risk, ego). occ/traj-map indices
    (8/9/12) are absent, so the guardrail and stop probe degrade exactly as documented."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, imgs, K, T_cam_ego, v0, **kw):
        o = self.net.forward_geo(imgs, K, T_cam_ego, v0.view(-1), acc=kw.get("acc", getattr(self, "_acc", None)))
        self._fused_bev = getattr(self.net, "_fused_bev", None)     # read by F3 (dump_f3_cache / f3_refine)
        return (o["lane"], o["dlog"], o["seg2d"], o["hm"], reg_half_angle(o["reg"]),
                o["stationary"], o["risk"], o["ego"])


_DEF = dict(BEV_FRONT=0, OCC_HEAD=OCC_HEAD, YAW2=YAW2, N_CAM=N_CAM, FS=FS, NECK_C=NECK_C, HEAD_C=HEAD_C, IMG_H=IMG_H, IMG_W=IMG_W, C_P=C_P, D=D, BC0=BC0, BC1=BC1, BCF=BCF, BEV2_C02=BEV2_C02, BEV2_DW=BEV2_DW)


def load_mite_eval(path, device="cuda", verbose=True):
    import bevlane.model as _mm
    ck = torch.load(path, map_location="cpu", weights_only=False)
    # rounds can train at another input size (e.g. 432x768, same as METEOR). weights are
    # size-agnostic (conv heads + BEV-side MLP) but forward_geo reads the module constants,
    # so apply the trained geometry from the ckpt args before running.
    global IMG_H, IMG_W, FS, FH, FW, C_P, D, BC0, BC1, BCF, N_CAM
    a = ck.get("args", {})
    # rebuild from defaults (otherwise loading several ckpts in one process carries over the previous config, e.g. r50 -> r18 size mismatch)
    IMG_H, IMG_W = a.get("img_h") or _DEF["IMG_H"], a.get("img_w") or _DEF["IMG_W"]
    N_CAM = int(a.get("n_cams") or _DEF["N_CAM"])   # single-camera ckpts
    set_bev_front(int(a.get("bev_front") or 0))      # front-only internal BEV grid
    global YAW2
    YAW2 = int(a.get("yaw2") or 0)
    if YAW2:
        # MiteEval already converts reg back to (sin θ', cos θ') via reg_half_angle. With env METEOR_YAW2=1,
        # bevlane.model.decode_boxes would apply atan2/2 a second time and halve the angle (crossing vehicles at 90°
        # flip to ±45°). Always disable decode-side halving on the METEORite eval path.
        import bevlane.model as _MD
        if getattr(_MD, "_YAW2", 0):
            if verbose:
                print("[meteorite] YAW2 ckpt: setting bevlane.model._YAW2 = 0 (MiteEval already halves the angle; avoids double conversion)", flush=True)
            _MD._YAW2 = 0
    global OCC_HEAD
    OCC_HEAD = 1 if float(a.get("occ_w") or 0) > 0 else 0
    FS = int(a.get("feat_stride") or _DEF["FS"])
    FH, FW = IMG_H // FS, IMG_W // FS
    print(f"[meteorite] ckpt geometry: input {IMG_H}x{IMG_W} (feature {FH}x{FW}), cams {N_CAM}", flush=True)
    C_P = a.get("cp") or _DEF["C_P"]
    D = a.get("depth_bins") or _DEF["D"]
    BC0 = a.get("bev_c0") or _DEF["BC0"]
    BC1 = a.get("bev_c1") or _DEF["BC1"]
    BCF = a.get("fuse_c") or _DEF["BCF"]
    global NECK_C, HEAD_C
    NECK_C = a.get("neck_c") or _DEF["NECK_C"]
    HEAD_C = a.get("head_c") or _DEF["HEAD_C"]
    global GIPM, DET_HEAD, BEV2, BEV2_C02, BEV2_DW, CAMW, PAINT_CLS, HEAD_V2, LIFT_LR, LR_BLK, LR_RANK, GLR_BLK, GLR_RANK, EGO_GAP
    GIPM = bool(a.get("gipm", False))
    LIFT_LR = int(a.get("lift_lr", 0) or 0)
    LR_BLK, LR_RANK = int(a.get("lr_blk") or 50), int(a.get("lr_rank") or 32)
    GLR_BLK, GLR_RANK = int(a.get("glr_blk") or 100), int(a.get("glr_rank") or 16)
    global LR_RECT, LR_MARGIN
    LR_RECT = int(a.get("lr_rect", 0) or 0); LR_MARGIN = float(a.get("lr_margin") or 0.15)
    global GLR_HALF, LR_ROWS_ONLY
    LR_ROWS_ONLY = int(a.get("lr_rows_only", 0) or 0); GLR_HALF = int(a.get("glr_half", 0) or 0)
    HEAD_V2 = int(a.get("head_v2", 0) or 0)
    EGO_GAP = int(a.get("ego_gap", 0) or 0)                   # GAP ego/tl for single-graph NPU export
    DET_HEAD = int(a.get("det_head", 0) or 0)
    CAMW = int(a.get("cam_w", 0) or 0)
    PAINT_CLS = tuple(int(c) for c in str(a.get("paint_seg", "") or "").split(",") if c.strip())
    BEV2 = bool(a.get("bev2", 0))
    BEV2_C02 = a.get("bev2_c02") or _DEF["BEV2_C02"]
    BEV2_DW = a.get("bev2_dw") or _DEF["BEV2_DW"]
    set_bev_res(0.2 if BEV2 else 0.4)
    net = METEORite(taps=None, pretrained=False, backbone=a.get("backbone", "resnet18"))
    if int(a.get("ego_acc", 0) or 0):
        net.enable_ego_acc()                                   # ckpt with the acc input
    if int(a.get("ae2e", 0) or 0):
        net.enable_ae2e()                                      # ckpt with the extra inputs (ego history + map / route raster)
    if int(a.get("rowband_lift", 0) or 0):                     # ckpt with the row-band lift: build the modules, then load weights
        from bevlane.rowband_lift import attach as _hl_attach
        _hl_attach(net, a)
    if int(a.get("polar", 0) or 0):                            # polar BEV ckpt: build modules and load weights; if no grid is stored, build geometry from the reference scene
        from bevlane.polar_lift import attach as _pl_attach, ensure_geometry as _pl_geo
        _pl_attach(net, a)
    sd = {k.replace("module.", ""): v for k, v in ck["sd"].items()}
    net.load_state_dict(sd, strict=net.polar is None)         # polar: older ckpts have no grid buffers -> strict=False
    if net.polar is not None:
        _pl_geo(net, scene=a.get("rowband_ref_scene"))
    w = MiteEval(net).to(device).eval()
    w.decode_boxes = _mm.DepthSegIPMNetV52.decode_boxes
    if verbose:
        print(f"[meteorite] eval adapter: {path} (ep {ck.get('ep')}, ema={ck.get('ema', False)}, "
              f"{sum(p.numel() for p in net.parameters())/1e6:.1f}M params, strict load OK)", flush=True)
    return w


SP24_DENSE = ("stem.", "ego_stem", "ego_mlp", "traj", "tl.", "stationary", "risk", "hm.", "reg.", "hm_x.", "reg_x.", "hm2.1", "reg2.1", "traj2", "stat2", "risk2", "seg_out", "camw", "pp_seg")


def prune_24(net, exclude=SP24_DENSE, zeros=2):                                   # zeros = zeros per group of 4 (1/2/3 = 25/50/75%)
    """Post-hoc zeroing for Orin 2:4 structured sparsity (for speed measurement; accuracy recovery fine-tuning is separate).
    Same rule as train.py --sparse-24: in each group of 4 along the input channels, zero the 2 smallest |w|. Applies to
    convs with Cin%4==0, Cin>=16, Cout>=32. The stem stays dense (sparse INT8 layer1 breaks on Orin TensorRT 8.6), as do
    the planner and output layers."""
    n_sp = n_dn = 0
    with torch.no_grad():
        for nm, p in net.named_parameters():
            if p.dim() != 4 or p.shape[1] % 4 or p.shape[1] < 16 or p.shape[0] < 32 or nm.startswith(exclude):
                n_dn += 1
                continue
            o, c, kh, kw = p.shape
            f = p.abs().reshape(o, c // 4, 4, kh * kw)
            m = torch.ones_like(f)
            m.scatter_(2, f.argsort(2)[:, :, :int(zeros)], 0.0)
            p.mul_(m.reshape(o, c, kh, kw))
            n_sp += 1
    return n_sp, n_dn

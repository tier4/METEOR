#!/usr/bin/env python3
"""Dump the METEOR lift's constant tables + one real frame's tensors for the
standalone CUDA lift benchmark (liftbench).

Geometry: v55 model, METEOR_BEV_XR=40.0 METEOR_LIFT_DIV=4 -> output BEV
600x500, lift grid 150x125 (G2=18750), 7 cameras, image 432x768, feature
grid 108x192 (stride 4), D=64 depth bins (D_MIN=1.0, D_STEP=1.25).

Exact lift math replicated (DepthSegIPMNetV16.project_bev, the path the
TRT engine implements via bake_frustum/bake_gather):
  for camera i, lift cell p (ego ground point X_p, z=0):
    pc   = T_cam_ego[i] @ X_p            (camera coords x,y,z)
    u    = fx*x/max(z,0.5) + cx ; v = fy*y/max(z,0.5) + cy   (IMAGE pixels)
    dist = |pc|
    valid = z>0.5 and 0<=u<768 and 0<=v<432 and dist<90
    gu   = u/(W-1)*2-1 ; gv = v/(H-1)*2-1          (W,H = 768,432)
    (ix,iy) = grid_sample unnormalize, align_corners=False, on 192x108:
              ix = ((gu+1)*192-1)/2 ; iy = ((gv+1)*108-1)/2
    ctx_s[c] = bilinear(ctx[i,c], ix, iy)          (zeros padding)
    b  = clamp((dist-1.0)/1.25, 0, 63-1e-4); b0=floor(b); fr=b-b0
    ps = bilinear(dprob[i,b0]) * (1-fr) + bilinear(dprob[i,min(b0+1,63)]) * fr
    w  = ps + 0.05
    num[c, cell] += ctx_s[c]*w ; den[cell] += w    (over valid pairs only)
  bev = num / max(den, 1e-4)                        -> [96, 150, 125]

Outputs (all little-endian, flat):
  pair_cam.bin  int32 [P]     camera index of pair
  pair_cell.bin int32 [P]     lift-cell index of pair (scatter target)
  pair_ix.bin   f32   [P]     feature-grid x sample coord (continuous)
  pair_iy.bin   f32   [P]     feature-grid y sample coord
  pair_b0.bin   int32 [P]     lower depth bin
  pair_fr.bin   f32   [P]     depth-bin lerp fraction
  csr_rowptr.bin int32 [G2+1] CSR: per-cell pair list (gather form)
  csr_col.bin   int32 [P]     pair ids, cell-major
  dprob.bin     f16 [7,64,108,192]   real frame, softmaxed depth
  ctx.bin       f16 [7,96,108,192]   real frame, ctx features
  ref_lift.bin  f32 [96,18750]       model's own lift output (pre-interp),
                                     computed from the fp16-quantized inputs
  meta.json     shapes + constants + pair-count distribution
"""
import argparse
import json
import os
import sys

os.environ.setdefault("METEOR_BEV_XR", "40.0")
os.environ.setdefault("METEOR_LIFT_DIV", "4")

import numpy as np
import torch

sys.path.insert(0, ".")
from bevlane.model import MODELS, DepthSegIPMNetV16  # noqa: E402
import bevlane.model as _M                            # noqa: E402
from bevlane.dataset import BevLaneDataset            # noqa: E402

_ap = argparse.ArgumentParser()
_ap.add_argument("--ckpt",
                 default="./out/v62_best_e2e.pt")
_ap.add_argument("--out",
                 default=os.path.dirname(os.path.abspath(__file__))
                 + "/tables")
_ARGS = _ap.parse_args()
OUT = _ARGS.out
CKPT = _ARGS.ckpt
ROOT = "./out/bevlane"
NC = int(os.environ.get("METEOR_NCAMS", "7"))
IMG_H, IMG_W = 432, 768


def build(ckpt):
    ck = torch.load(ckpt, map_location="cpu")
    mv = (ck.get("args") or {}).get("model")
    assert mv in ("v55", "v63b", "v52"), mv          # all share the V52 lift
    net = MODELS[mv](n_seg=21)
    sd = {k.replace("module.", ""): v for k, v in ck["model"].items()}
    cur = net.state_dict()
    sd = {k: v for k, v in sd.items() if k in cur and cur[k].shape == v.shape}
    miss = net.load_state_dict(sd, strict=False)
    print(f"[build] {mv}: missing {len(miss.missing_keys)} "
          f"unexpected {len(miss.unexpected_keys)}")
    net.eval()
    return net


def main():
    os.makedirs(OUT, exist_ok=True)
    scene = open("./val.lst").readline().strip()
    print("[scene]", scene)
    net = build(CKPT)
    lift_h, lift_w = net.lift_h, net.lift_w
    G2 = lift_h * lift_w
    D, D_MIN, D_STEP = net.D, net.D_MIN, net.D_STEP
    print(f"[geom] lift {lift_h}x{lift_w} G2={G2} D={D} "
          f"BEV {_M.BEV_H}x{_M.BEV_W}")

    ds = BevLaneDataset(ROOT, [scene], gt_key="gt_vec")
    b = ds[0]
    imgs = b[0][None][:, :NC]
    K = b[1][None][:, :NC].float()
    Tc = b[2][None][:, :NC].float()
    print("[frame]", imgs.shape, "K", K.shape)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = net.to(dev)
    imgs, K, Tc = imgs.to(dev), K.to(dev), Tc.to(dev)

    # ---- real dprob / ctx exactly as the lift consumes them --------------
    with torch.no_grad():
        f = net.image_feats(imgs)
        dlog = net.depth_head(net.depth_up(f))         # RAW logits [N,64,...]
        dprob = net.sharpen_dprob(dlog.softmax(1))     # [N,64,108,192]
        ctx = net.ctx(f)                               # [N,96,108,192]
    Hf, Wf = ctx.shape[-2:]
    Cc = ctx.shape[1]
    assert (Hf, Wf) == (108, 192) and dprob.shape[1] == D, (ctx.shape,
                                                            dprob.shape)
    # fp16-quantize: this is what the engine sees and what the kernel reads
    dprob_h = dprob.half()
    ctx_h = ctx.half()

    # ---- reference: the model's own lift on the quantized inputs ---------
    # V52.project_bev swaps module globals to the lift grid then runs the
    # V16 dense lift; replicate that swap and call the V16 impl directly so
    # the reference is the pre-interpolate lift output.
    oh, ow = _M.BEV_H, _M.BEV_W
    _M.BEV_H, _M.BEV_W = lift_h, lift_w
    try:
        with torch.no_grad():
            ref = DepthSegIPMNetV16.project_bev(
                net, dprob_h.float(), ctx_h.float(), K, Tc,
                1, NC, IMG_H, IMG_W)                   # [1,Cc,150,125]
    finally:
        _M.BEV_H, _M.BEV_W = oh, ow
    ref = ref[0].reshape(Cc, G2).float().cpu().numpy()

    # ---- fused-plugin reference: fp16 LOGITS in, fp32 softmax inside, ----
    # ---- lift, then bilinear resize to the full BEV (Resize_3 fold) ------
    dlog_h = dlog.half()
    dprob_fold = dlog_h.float().softmax(1)
    oh2, ow2 = _M.BEV_H, _M.BEV_W
    _M.BEV_H, _M.BEV_W = lift_h, lift_w
    try:
        with torch.no_grad():
            ref_f = DepthSegIPMNetV16.project_bev(
                net, dprob_fold, ctx_h.float(), K, Tc, 1, NC, IMG_H, IMG_W)
    finally:
        _M.BEV_H, _M.BEV_W = oh2, ow2
    with torch.no_grad():
        ref_up = torch.nn.functional.interpolate(
            ref_f, size=(oh2, ow2), mode="bilinear", align_corners=False)
    ref_up = ref_up[0].float().cpu().numpy()           # [Cc, 600, 500]

    # ---- constant tables (same math as bake_frustum, kept in fp32) -------
    pts = net.bev_pts.to(dev)                          # [G2,4] ego, z=0
    pc = torch.matmul(Tc.reshape(NC, 4, 4),
                      pts.t().unsqueeze(0).expand(NC, 4, G2))
    x, y, z = pc[:, 0], pc[:, 1], pc[:, 2]
    Kf = K.reshape(NC, 3, 3)
    zc = z.clamp(min=0.5)
    u = Kf[:, 0, 0].unsqueeze(-1) * x / zc + Kf[:, 0, 2].unsqueeze(-1)
    v = Kf[:, 1, 1].unsqueeze(-1) * y / zc + Kf[:, 1, 2].unsqueeze(-1)
    dist = torch.sqrt(x * x + y * y + z * z)
    valid = ((z > 0.5) & (u >= 0) & (u < IMG_W) & (v >= 0) & (v < IMG_H)
             & (dist < 90.0))
    gu = (u / (IMG_W - 1) * 2 - 1).clamp(-2, 2)
    gv = (v / (IMG_H - 1) * 2 - 1).clamp(-2, 2)
    ix = ((gu + 1) * Wf - 1) / 2                       # feature-space coords
    iy = ((gv + 1) * Hf - 1) / 2
    bb = ((dist - D_MIN) / D_STEP).clamp(0, D - 1 - 1e-4)
    b0 = bb.floor()
    fr = bb - b0

    cam_l, cell_l = [], []
    ix_l, iy_l, b0_l, fr_l = [], [], [], []
    per_cam = []
    for i in range(NC):
        ii = valid[i].nonzero(as_tuple=True)[0]
        per_cam.append(int(ii.numel()))
        cam_l.append(torch.full_like(ii, i, dtype=torch.int32))
        cell_l.append(ii.to(torch.int32))
        ix_l.append(ix[i, ii]); iy_l.append(iy[i, ii])
        b0_l.append(b0[i, ii].to(torch.int32)); fr_l.append(fr[i, ii])
    cam = torch.cat(cam_l).cpu().numpy()
    cell = torch.cat(cell_l).cpu().numpy()
    P = len(cam)
    print(f"[pairs] P={P} of {NC*G2} ({100*P/(NC*G2):.1f}%) per-cam {per_cam}")

    # CSR: per-cell pair list, ordered like the flat pair array (camera-major)
    order = np.argsort(cell, kind="stable")
    col = order.astype(np.int32)
    counts = np.bincount(cell, minlength=G2)
    rowptr = np.zeros(G2 + 1, np.int32)
    np.cumsum(counts, out=rowptr[1:])
    kdist = np.bincount(counts)
    print("[K per cell] hist:", {i: int(n) for i, n in enumerate(kdist)},
          "mean", counts.mean(), "max", counts.max())

    def wr(name, arr, dt):
        a = np.ascontiguousarray(arr, dtype=dt)
        a.tofile(os.path.join(OUT, name))
        print(f"  {name:16s} {a.shape} {a.dtype} {a.nbytes/1e6:.2f} MB")

    wr("pair_cam.bin", cam, np.int32)
    wr("pair_cell.bin", cell, np.int32)
    wr("pair_ix.bin", torch.cat(ix_l).cpu().numpy(), np.float32)
    wr("pair_iy.bin", torch.cat(iy_l).cpu().numpy(), np.float32)
    wr("pair_b0.bin", torch.cat(b0_l).cpu().numpy(), np.int32)
    wr("pair_fr.bin", torch.cat(fr_l).cpu().numpy(), np.float32)
    wr("csr_rowptr.bin", rowptr, np.int32)
    wr("csr_col.bin", col, np.int32)
    wr("dprob.bin", dprob_h.cpu().numpy(), np.float16)
    wr("dlog.bin", dlog_h.cpu().numpy(), np.float16)
    wr("ctx.bin", ctx_h.cpu().numpy(), np.float16)
    wr("ref_lift.bin", ref, np.float32)
    wr("ref_lift_up.bin", ref_up, np.float32)

    meta = dict(N=NC, Cc=Cc, D=D, Hf=Hf, Wf=Wf, G2=G2,
                lift_h=lift_h, lift_w=lift_w, P=P,
                out_h=_M.BEV_H, out_w=_M.BEV_W,
                D_MIN=D_MIN, D_STEP=D_STEP, eps=0.05, den_clamp=1e-4,
                img_h=IMG_H, img_w=IMG_W, scene=scene,
                per_cam_pairs=per_cam,
                k_hist={int(i): int(n) for i, n in enumerate(kdist) if n},
                k_mean=float(counts.mean()), k_max=int(counts.max()))
    json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=1)
    print("[done]", OUT)


if __name__ == "__main__":
    main()

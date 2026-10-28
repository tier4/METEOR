#!/usr/bin/env python3
"""F3 trajectory refinement head at inference.

The F3 "single" arm (train_f3.py) learns a bounded residual on the SELECTED candidate
from the pooled fused BEV (64x40, or an integer AveragePool for the NPU variant) + the
candidate + v0, with the 0.5 s point frozen. This module attaches that head to a live
network so closed_loop_eval / the demo measure it on the real network output, not on
the training cache.

  closed_loop_eval : METEOR_F3_HEAD=<f3 run>/best.pt  (forward hook, enable_f3_refine)
  demo_rgbd_bev    : METEOR_F3_HEAD=<f3 run>/best.pt  (explicit f3_apply after the forward)
  NPU export       : deploy/export_mite_ckpt.py --npu npu --f3 <best.pt> bundles the head into the graph
"""
import os
import torch
import torch.nn.functional as F

from bevlane.model import EGO_K

POOL = (64, 40)                                     # dump_f3_cache.POOL


def f3_apply(net, e, v, ext=None):
    """Refine the ego vector e [B,42]: (1) optionally re-select the candidate by an acceleration rule
    (METEOR_F3_ACC_SEL > 0, ext[:,0]*3 = current acceleration), then (2) refine the 1-3 s points of the
    selected candidate with the F3 residual head (net.f3_head, features from net._fused_bev).
    net is the network that carries f3_head (METEORite or MiteEval). Returns the new e."""
    B = e.shape[0]
    fb = net._fused_bev.float()
    feat = F.avg_pool2d(fb, net._f3_poolk) if getattr(net, "_f3_poolk", 0) else F.adaptive_avg_pool2d(fb, POOL)   # the NPU head uses integer pooling
    wp = e[:, :12 * EGO_K].float().view(B, EGO_K, 12)
    k = e[:, 12 * EGO_K:12 * EGO_K + EGO_K].float().argmax(1)
    ar = torch.arange(B, device=e.device)
    _th = float(os.environ.get("METEOR_F3_ACC_SEL", "0") or 0)        # re-select by current acceleration (braking -> slowest candidate, accelerating -> fastest)
    if ext is None:
        ext = getattr(net, "_f3_ext", None)                             # the evaluator may set [B, n] (acc/3, brk)
    if _th > 0 and ext is not None:
        acc = ext.to(e.device)[:, 0].float() * 3.0
        x3 = wp[:, :, 10]                                               # x of each candidate at 3 s
        k = torch.where(acc < -_th, x3.argmin(1), torch.where(acc > _th, x3.argmax(1), k))
        e = e.clone(); lg = torch.full((B, EGO_K), -10.0, device=e.device, dtype=e.dtype); lg[ar, k] = 10.0
        e[:, 12 * EGO_K:12 * EGO_K + EGO_K] = lg                       # downstream sel_wp picks the same candidate
    sel = wp[ar, k]
    if os.environ.get("METEOR_F3_NOREFINE"):                            # measure the selection rule alone
        return e
    with torch.autocast("cuda", enabled=False):
        new = net.f3_head(feat, sel, v.float(), ext.to(e.device) if ext is not None else None)
    wp2 = wp.clone()
    wp2[ar, k] = new
    e2 = e.clone()
    e2[:, :12 * EGO_K] = wp2.reshape(B, -1).to(e.dtype)
    return e2


def _f3_hook(net, args, kwargs, output):
    out = list(output)
    e = out[7]
    v0 = kwargs.get("v0", args[3] if len(args) > 3 else None)
    v = v0.view(-1).float() if v0 is not None else torch.zeros(e.shape[0], device=e.device)
    out[7] = f3_apply(net, e, v)
    return tuple(out)


def enable_f3_refine(net, path, hook=True):
    """Attach the F3 head stored in path (best.pt of train_f3.py). hook=False: the caller runs f3_apply itself (demo)."""
    from bevlane.train_f3 import F3Head, F3HeadNPU
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("arm", "single") != "single":
        raise ValueError("f3_refine: only the single arm runs at inference (fuse / stack need the previous frame)")
    dev = next(net.parameters()).device
    if ck.get("npu"):                                   # NPU head: integer pooling k = 400 / feat_h
        head = F3HeadNPU(96, float(ck.get("bound", 2.0)), int(ck.get("extra", 0))).to(dev)
        net._f3_poolk = int(os.environ.get("METEOR_F3_POOLK", "0") or round(400 / int(ck["feat"][1])))
    else:
        head = F3Head(96, float(ck.get("bound", 2.0)), extra=int(ck.get("extra", 0))).to(dev)
    head.load_state_dict(ck["head"])
    head.eval()
    net.f3_head = head
    if hook:
        net.register_forward_hook(_f3_hook, with_kwargs=True)
    print(f"[f3_refine] residual head attached from {path} (cache val new={ck.get('val', {}).get('new', '?')})",
          flush=True)
    return net

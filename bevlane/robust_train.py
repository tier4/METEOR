"""Quantization- and 2:4-sparsity-robust training (METEORite / Nano).

On Orin Nano, full INT8 (per-tensor activations) degrades the BEV branch, and post-hoc
2:4 zeroing of weights loses accuracy. This module
  (a) applies fake-quant (symmetric int8, per-tensor, moving-average min/max) to the
      input activations of Conv / Linear layers and to the product tensors of the
      Gather lift during training; deployment still uses calibrated PTQ (the graph is
      unchanged, also for the NPU compiler);
  (b) re-applies the same 2:4 mask as train.py --sparse-24 at every step
      (layers in SP24_DENSE stay dense).
Fake-quant keys are stripped from the saved state_dict."""
import torch
import torch.nn as nn
from torch.ao.quantization import FakeQuantize, MovingAverageMinMaxObserver

from bevlane import meteorite as M


def _fq():
    return FakeQuantize(observer=MovingAverageMinMaxObserver, quant_min=-128, quant_max=127, dtype=torch.qint8,
                        qscheme=torch.per_tensor_symmetric, averaging_constant=0.01)


def attach_qat(net, include_linear=True):
    """Attach fake-quant to Conv2d / Linear inputs and to VTGather / GroundGather inputs (product tensors) and outputs. Returns the number attached."""
    net.qat_fq = nn.ModuleDict()
    n = 0
    for name, mod in list(net.named_modules()):
        if name == "" or name.startswith("qat_fq"):
            continue
        key = name.replace(".", "_")
        if isinstance(mod, nn.Conv2d) or (include_linear and isinstance(mod, nn.Linear)):
            net.qat_fq[key] = _fq().to(next(mod.parameters()).device)
            mod.register_forward_pre_hook(_pre_hook(net.qat_fq[key]))
            n += 1
        elif isinstance(mod, (M.VTGather, M.GroundGather)):
            net.qat_fq[key + "_in"] = _fq().to(mod.idx.device)
            net.qat_fq[key + "_out"] = _fq().to(mod.idx.device)
            mod.register_forward_pre_hook(_pre_hook(net.qat_fq[key + "_in"]))
            mod.register_forward_hook(_post_hook(net.qat_fq[key + "_out"]))
            n += 2
    set_fake_quant(net, False)          # observe only at first (warm-up); the training loop enables it after qat_warm
    return n


def _pre_hook(fq):
    def hook(mod, inputs):
        x = inputs[0]
        if not torch.is_tensor(x) or not x.is_floating_point():
            return None
        y = fq(x.float()).to(x.dtype)
        return (y,) + tuple(inputs[1:])
    return hook


def _post_hook(fq):
    def hook(mod, inputs, out):
        return fq(out.float()).to(out.dtype)
    return hook


def set_fake_quant(net, on):
    for m in getattr(net, "qat_fq", {}).values():
        (m.enable_fake_quant if on else m.disable_fake_quant)()
        m.enable_observer()


def strip_qat(sd):
    return {k: v for k, v in sd.items() if not k.startswith("qat_fq.")}


def build_24_masks(net, ramp=False, exclude=M.SP24_DENSE, zeros=2):
    """Build masks with the same rule as prune_24 and apply them to the weights.
    zeros = number of zeros per group of 4 input channels (1 = 25%, 2 = 50% (2:4), 3 = 75%).
    With ramp=True, start at zeros-1 (min 1) and keep the final `zeros` mask in `finals`.
    Returns (pairs, finals, n_dense)."""
    pairs, finals, n_dn = [], [], 0
    zeros = int(zeros); assert 1 <= zeros <= 3, zeros
    z0 = max(1, zeros - 1)
    with torch.no_grad():
        for nm, p in net.named_parameters():
            if p.dim() != 4 or p.shape[1] % 4 or p.shape[1] < 16 or p.shape[0] < 32 or nm.startswith(exclude):
                n_dn += 1
                continue
            o, c, kh, kw = p.shape
            f = p.detach().abs().reshape(o, c // 4, 4, kh * kw)
            srt = f.argsort(2)
            m2 = torch.ones_like(f); m2.scatter_(2, srt[:, :, :zeros], 0.0); m2 = m2.reshape(o, c, kh, kw)
            if ramp and z0 < zeros:
                m1 = torch.ones_like(f); m1.scatter_(2, srt[:, :, :z0], 0.0); m1 = m1.reshape(o, c, kh, kw)
                p.mul_(m1); pairs.append((p, m1)); finals.append(m2)
            else:
                p.mul_(m2); pairs.append((p, m2))
    return pairs, finals, n_dn


def apply_24(pairs):
    with torch.no_grad():
        for p, m in pairs:
            p.mul_(m)

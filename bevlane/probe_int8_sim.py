"""Emulate INT8 rounding inside PyTorch and locate the layer that breaks E2E (ego).

2026-08-22: the Orin INT8 engine freezes the ego output (frame-to-frame diff 0.06 vs
2.05 in fp16) and halves the stationary logit range. To tell TensorRT-specific from
model-inherent, insert fake-quantization hooks that round Conv outputs to symmetric
8 bit and check whether the same symptom reproduces.
--scope narrows the target so the breaking layer can be bisected.
"""
import argparse, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bevlane.dataset import BevLaneDataset
from bevlane.model import MODELS, EGO_K

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--list", default="val.lst")
ap.add_argument("--root", default="out/bevlane")
ap.add_argument("--frames", type=int, default=4)
ap.add_argument("--scope", default="all",
                help="all / none / backbone / ego / head / not-ego")
ap.add_argument("--bits", type=int, default=8)
ap.add_argument("--calib", default="dynamic",
                help="dynamic (per-tensor max) / static (fixed by calibration) ")
ap.add_argument("--pct", type=float, default=100.0,
                help="scale percentile for static mode. TensorRT entropy "
                     "calibration clips outliers, so try e.g. 99.9")
ap.add_argument("--calib-frames", type=int, default=16)
ap.add_argument("--quant-weight", action="store_true",
                help="also round weights to per-channel INT8 (as TensorRT does)")
a = ap.parse_args()

scenes = [l.strip() for l in open(a.list) if l.strip()][:2]
ds = BevLaneDataset(a.root, scenes, gt_key="gt_cons", with_ego=True,
                    max_per_scene=8, n_cams=8, trim_start=3, trim_end=10)
m = MODELS["v52"](n_seg=21).cuda().eval()
sd = torch.load(a.ckpt, map_location="cpu"); sd = sd.get("model", sd)
sd = {k.replace("module.", ""): v for k, v in sd.items()}
cur = m.state_dict()
m.load_state_dict({k: v for k, v in sd.items()
                   if k in cur and cur[k].shape == v.shape}, strict=False)

QMAX = 2 ** (a.bits - 1) - 1
SCALES = {}          # per-layer scales fixed by static calibration
COLLECT = {"on": False}

def fq(mod, inp, out):
    if not torch.is_tensor(out) or not out.is_floating_point():
        return out
    key = id(mod)
    if COLLECT["on"]:
        v = out.detach().abs().flatten().float()
        if a.pct >= 100.0:
            mx = float(v.amax())
        else:
            k = max(1, int(v.numel() * a.pct / 100.0))
            mx = float(v.kthvalue(k).values)
        SCALES[key] = max(SCALES.get(key, 0.0), mx)
        return out
    if a.calib == "static":
        mx = SCALES.get(key, 0.0)
        if mx <= 0:
            return out
        s = mx / QMAX
    else:
        mxt = out.detach().abs().amax()
        if mxt <= 0:
            return out
        s = mxt / QMAX
    return torch.round(out / s).clamp(-QMAX, QMAX) * s

def want(name):
    n = name.lower()
    ego_like = any(k in n for k in ("ego", "intent", "kin", "vprof", "stat"))
    if a.scope == "all":      return True
    if a.scope == "none":     return False
    if a.scope == "ego":      return ego_like
    if a.scope == "not-ego":  return not ego_like
    if a.scope == "backbone": return any(k in n for k in
                                         ("layer", "stem", "conv1", "fpn", "lat"))
    if a.scope == "head":     return any(k in n for k in
                                         ("dec", "head", "out"))
    return False

# per-channel weight quantization (same scheme as the TensorRT default)
n_wq = 0
if a.quant_weight:
    with torch.no_grad():
        for name, mod in m.named_modules():
            if not isinstance(mod, (torch.nn.Conv2d, torch.nn.Linear)):
                continue
            if not want(name):
                continue
            w = mod.weight.data
            dims = tuple(range(1, w.dim()))
            mx = w.abs().amax(dim=dims, keepdim=True)
            sc = (mx / QMAX).clamp_min(1e-12)
            mod.weight.data = torch.round(w / sc).clamp(-QMAX, QMAX) * sc
            n_wq += 1
    print(f"[weight] {n_wq} layers quantized to per-channel INT8")

hooks, n_hooked = [], 0
for name, mod in m.named_modules():
    if isinstance(mod, (torch.nn.Conv2d, torch.nn.Linear)) and want(name):
        hooks.append(mod.register_forward_hook(fq))
        n_hooked += 1

if a.calib == "static":
    COLLECT["on"] = True
    for i in range(a.calib_frames):
        b = ds[i]
        if b is None: continue
        eg = b[4]
        v0 = torch.tensor([float(eg[12])]).cuda()
        with torch.no_grad(), torch.autocast("cuda", torch.float16):
            m(b[0][None].cuda(), b[1][None].cuda(), b[2][None].cuda(), v0=v0)
    COLLECT["on"] = False
    print(f"[calib] scales for {len(SCALES)} layers fixed from {a.calib_frames} frames"
          f" at percentile {a.pct}")

ys, egos = [], []
for i in range(a.frames):
    b = ds[i]
    if b is None: continue
    eg = b[4]
    v0 = torch.tensor([float(eg[12])]).cuda()
    with torch.no_grad(), torch.autocast("cuda", torch.float16):
        out = m(b[0][None].cuda(), b[1][None].cuda(), b[2][None].cuda(), v0=v0)
    e = out[7].float()[0]
    wp = e[:12 * EGO_K].view(EGO_K, 6, 2)
    k = int(e[12 * EGO_K:12 * EGO_K + EGO_K].argmax())
    ys.append(float(wp[k, -1, 1]))
    egos.append(e.cpu().numpy())
for h in hooks:
    h.remove()
E = np.array(egos)
diffs = [float(np.abs(E[i] - E[i-1]).mean()) for i in range(1, len(E))]
print(f"\n=== scope={a.scope} {a.calib} pct={a.pct} ({a.bits}bit, {n_hooked} layers) ===")
print(f"  ego std={E.std():.3f}  mean frame-to-frame diff={np.mean(diffs) if diffs else 0:.4f}")
print(f"  final-point y: {[round(v,2) for v in ys]}")
print("INT8_SIM_DONE")

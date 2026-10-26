#!/usr/bin/env python3
"""Export a TRAINED METEORite checkpoint for one rig (e.g. for deployment and benchmarking on Orin).

Geometry/widths come from the ckpt args (img 432x768 for mite4). The rig table is baked from one real
sample; parity is checked against the training path (forward_geo, per-sample K/T) on that sample.
usage: python deploy/export_mite_ckpt.py --ckpt X.pt --out out/meteorite/mite4 [--vt gather] [--with-lidar]
"""
import argparse, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bevlane.meteorite as M

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--vt", default="gather")
ap.add_argument("--blk", type=int, default=0, help="block size of the matmul/blockconv view transform (0 = default 10; 25 for the conv1x1 target)")
ap.add_argument("--with-lidar", action="store_true")
ap.add_argument("--root", default=os.environ.get("METEOR_BEV_ROOT", "out/bevlane"))
ap.add_argument("--scene", default=os.environ.get("METEOR_REF_SCENE", ""), help="scene used for the export parity check (default: $METEOR_REF_SCENE)")
ap.add_argument("--sparse24", action="store_true", help="post-hoc 2:4 zeroing (for speed measurement only)")
ap.add_argument("--conv1x1", action="store_true", help="conv1x1 mode: low-rank layers as 1x1 Convs, depth softmax on the channel axis, normalisation folded into the stem")
ap.add_argument("--npu", default="", choices=["", "npu", "conv1x1"], help="NPU export mode: npu = strict NPU compiler (Conv-only graph, standard ONNX operators only, static shapes, <=4-D tensors); conv1x1 = same as --conv1x1")
ap.add_argument("--lr-merge", type=int, default=0, help="fold U*V of the rectangle-restricted low-rank layer into a single Conv1x1 (geometry-exact)")
ap.add_argument("--lr-rows-only", type=int, default=0, help="omit the column Slice/Reshape (2 nodes per pair, more weights)")
ap.add_argument("--lr-mm4d", type=int, default=0, help="build the hconv lift input directly with a 4-D MatMul (no Reshape/Transpose)")
ap.add_argument("--lr-hconv", type=int, default=0, help="input [N, cols, rows, C] + per-pair Slice -> Conv(kernel=(rows,1)) (no per-pair Reshape; dense rectangular weights)")
ap.add_argument("--img-hw", default="", help="override img_h/img_w from the ckpt args (e.g. 720,1280) for structural checks; the lift changes shape, so rebuild it with --geo-init 1")
ap.add_argument("--img-scale", type=float, default=1.0, help="data-side rescale (without hr_root the 768 images are upscaled; K follows). 1.6667 for 720p")
ap.add_argument("--feat-stride", type=int, default=0, help="override feat_stride from the ckpt args (e.g. 1/8 lift columns at 720p, for checks)")
ap.add_argument("--geo-init", type=int, default=0, help="1: initialise the low-rank lift by geometric SVD (when the ckpt has no lift weights or they differ in shape)")
ap.add_argument("--f3", default="", help="bundle an F3 residual head (best.pt from train_f3.py --npu 1) into the npu graph and add an ego_f3 [1,36] output")
ap.add_argument("--in-native", type=int, default=1, help="1: keep the engine input at the stored resolution (432x768) and upscale to IMG inside the graph")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
ar = ck.get("args", {})
if a.img_hw:
    ar = dict(ar); ar["img_h"], ar["img_w"] = (int(v) for v in a.img_hw.split(","))
    print(f"[cfg] img override {ar['img_h']}x{ar['img_w']} (--img-hw)", flush=True)
if ar.get("img_h"):
    M.IMG_H, M.IMG_W = ar["img_h"], ar["img_w"]; M.FS = int(a.feat_stride or ar.get("feat_stride") or 8); M.FH, M.FW = M.IMG_H // M.FS, M.IMG_W // M.FS
for k, attr in (("cp", "C_P"), ("depth_bins", "D"), ("bev_c0", "BC0"), ("bev_c1", "BC1"), ("fuse_c", "BCF"),
                ("neck_c", "NECK_C"), ("head_c", "HEAD_C")):
    if ar.get(k):
        setattr(M, attr, ar[k])
M.GIPM = bool(ar.get("gipm", False))
# later configuration flags also come from the ckpt args (same as load_mite_eval)
M.DET_HEAD = int(ar.get("det_head", 0) or 0)
M.HEAD_V2 = int(ar.get("head_v2", 0) or 0)
M.EGO_GAP = int(ar.get("ego_gap", 0) or 0)   # ego/tl heads via GlobalAveragePool (single compiled graph)
M.LIFT_LR = int(ar.get("lift_lr", 0) or 0); M.LR_BLK, M.LR_RANK = int(ar.get("lr_blk") or 50), int(ar.get("lr_rank") or 32); M.GLR_BLK, M.GLR_RANK = int(ar.get("glr_blk") or 100), int(ar.get("glr_rank") or 16)
M.LR_RECT = int(ar.get("lr_rect", 0) or 0); M.LR_MARGIN = float(ar.get("lr_margin") or 0.15); M.GLR_HALF = int(ar.get("glr_half", 0) or 0); M.LR_ROWS_ONLY = int(ar.get("lr_rows_only", 0) or 0)
M.CAMW = int(ar.get("cam_w", 0) or 0)
M.OCC_HEAD = 1 if float(ar.get("occ_w", 0) or 0) > 0 else 0   # occupancy head
M.YAW2 = int(ar.get("yaw2", 0) or 0)
M.PAINT_CLS = tuple(int(c) for c in str(ar.get("paint_seg", "") or "").split(",") if c.strip())
M.BEV2 = bool(ar.get("bev2", 0))
M.BEV2_C02 = ar.get("bev2_c02") or M.BEV2_C02
M.BEV2_DW = ar.get("bev2_dw") or M.BEV2_DW
M.set_bev_res(0.2 if M.BEV2 else 0.4)
M.set_bev_front(int(ar.get("bev_front") or 0))   # front-only internal BEV grid (exported graph stays cropped)
print(f"[cfg] bev2={M.BEV2} det_head={M.DET_HEAD} cam_w={M.CAMW} paint={M.PAINT_CLS}", flush=True)
print(f"[geom] input {M.IMG_H}x{M.IMG_W} feat {M.FH}x{M.FW} D={M.D} C_P={M.C_P}", flush=True)

from bevlane.dataset import BevLaneDataset
M.N_CAM = int(ar.get("n_cams") or 8)   # single-camera ckpts set n_cams=1
ds = BevLaneDataset(a.root, [a.scene], gt_key="gt_cons", with_ego=True, n_cams=M.N_CAM, max_per_scene=1,
                    trim_start=3, trim_end=10, gt_optional=True, img_scale=a.img_scale)
b = ds[0]
K, T, hw = b[1].numpy(), b[2].numpy(), tuple(b[0].shape[-2:])
taps = M.bake_taps(K, T, hw)
if a.in_native and hw != (M.IMG_H, M.IMG_W):
    M.IN_HW = hw
    print(f"[geom] engine input {hw} -> in-graph bilinear -> {M.IMG_H}x{M.IMG_W}", flush=True)
np.save(os.path.join(a.out, "rig_K.npy"), K); np.save(os.path.join(a.out, "rig_T_cam_ego.npy"), T)
if a.blk: M.BLK = a.blk   # change the block size (must divide LH/LW)
M.LR_CONV1X1 = int(a.conv1x1 or a.npu == "conv1x1"); M.NPU_MODE = a.npu; M.LR_MERGE = int(a.lr_merge); M.LR_ROWS_ONLY = int(a.lr_rows_only) or M.LR_ROWS_ONLY; M.LR_HCONV = int(a.lr_hconv); M.LR_MM4D = int(a.lr_mm4d)
M.LR_GEO_INIT = int(a.geo_init) or M.LR_GEO_INIT
net = M.METEORite(taps, vt=a.vt, pretrained=False, backbone=ar.get("backbone", "resnet18"), conv1x1=a.conv1x1).eval()
if int(ar.get("polar", 0) or 0):                   # polar BEV ckpt: no lift, outputs on the polar grid; unpack via one-hot ConvTranspose (no Reshape)
    from bevlane.polar_lift import attach as _pl_attach
    _K, _T = np.asarray(K), np.asarray(T)
    _pl_attach(net, ar, geometry=(_K[0] if _K.ndim == 3 else _K, _T[0] if _T.ndim == 3 else _T, hw)); net.polar.export_unpack = True   # geometry is only used by forward_geo for the parity check
    net.eval()                                     # attach runs after eval(), so put PolarLift's BN in eval mode (train-mode BN breaks the ORT comparison)
    print(f"[polar] {net.polar.extra_repr()} (export_unpack)", flush=True)
if M.GIPM:
    net.gvt = M.GroundGather(K, T, hw)
    print(f"[gipm] ground table {M.GH}x{M.GW}, kmax {net.gvt.kmax}", flush=True)
sd = {k.replace("module.", ""): v for k, v in ck["sd"].items()}
if a.img_hw or a.feat_stride:   # resolution change (structural check): the resolution-dependent low-rank lift (U/V of vt_lr/gvt_lr) changes shape, so drop it and keep the --geo-init 1 values
    cur = net.state_dict(); drop = [k for k, v in sd.items() if k in cur and tuple(cur[k].shape) != tuple(v.shape)]
    gone = [k for k in sd if k not in cur]            # feat_stride 8 has no stride-4 lat1 (dropped; structural check only)
    if gone: print(f"[load] structure change: dropping {len(gone)} weights absent from the current model {gone[:6]}", flush=True)
    for k in gone: sd.pop(k)
    assert all(k.startswith(("vt_lr.", "gvt_lr.", "vt.", "gvt.")) for k in drop), [k for k in drop if not k.startswith(("vt_lr.", "gvt_lr."))][:10]
    print(f"[load] resolution change: dropping {len(drop)} lift weights with mismatched shapes, keeping the geometric init (e.g. {drop[:2]})", flush=True)
    for k in drop: sd.pop(k)
miss, unexp = net.load_state_dict(sd, strict=False)
if a.img_hw or a.feat_stride:
    miss = [k for k in miss if not k.startswith(("vt_lr.", "gvt_lr."))]
if (a.lr_merge or a.lr_rows_only or a.lr_hconv) and M.LIFT_LR and M.LR_RECT:          # store the folded rectangle-restricted U*V weights as buffers (after loading weights, before tracing)
    net.vt_lr.prepare_npu(in_scale=net.fs / 8.0)
    if M.GIPM:
        net.gvt_lr.prepare_npu()

assert not unexp and all(k.startswith(("vt.", "gvt.", "polar.grid_", "polar.valid_")) for k in miss), (miss, unexp)   # polar grids are not needed for export (absent in older ckpts)
if a.sparse24:
    print("[sparse24] pruned / dense convs:", M.prune_24(net), flush=True)
print(f"[load] {a.ckpt} ep {ck.get('ep')} ema={ck.get('ema')} (missing only baked vt.* = {len(miss)})", flush=True)
if a.f3:
    assert a.npu == "npu", "--f3 is only supported with --npu npu"
    from bevlane.train_f3 import F3HeadNPU
    ckf = torch.load(a.f3, map_location="cpu", weights_only=False)
    assert ckf.get("npu") and ckf.get("arm", "single") == "single", "--f3 needs a head trained with train_f3.py --npu 1 --arm single"
    h = F3HeadNPU(96, float(ckf.get("bound", 2.0)), int(ckf.get("extra", 0))); h.load_state_dict(ckf["head"]); h.eval()
    net.f3_npu = h; net._f3_poolk = int(round(400 / int(ckf["feat"][1])))
    print(f"[f3] bundled NPU residual head: {a.f3} (pool k={net._f3_poolk}, cache val new={ckf.get('val', {}).get('new', '?')})", flush=True)

# parity: baked export forward vs training forward_geo on the same real frame
img_n = b[0][None]                                                  # normalised floats [1,8,3,h,w]
mean = torch.tensor(M.IMG_MEAN).view(1, 1, 3, 1, 1); std = torch.tensor(M.IMG_STD).view(1, 1, 3, 1, 1)
u8 = ((img_n * std + mean) * 255).round().clamp(0, 255)
if hw != (M.IMG_H, M.IMG_W) and not M.IN_HW:
    u8 = torch.nn.functional.interpolate(u8[0], (M.IMG_H, M.IMG_W), mode="bilinear", align_corners=False)[None].round()
v0 = b[4][12].view(1) if len(b) > 4 else torch.tensor([10.0])
lid = torch.zeros(1, 4, M.LH, M.LW) if a.with_lidar else None
with torch.no_grad():
    if a.npu == "npu":
        ob = net(u8[0].to(torch.uint8), v0.view(1, 1))
    else:
        ob = net(u8.to(torch.uint8), v0, lid)
    og = net.forward_geo((u8 / 255 - mean) / std, torch.from_numpy(K)[None], torch.from_numpy(T)[None], v0, lidar=lid)
    if M.BEV_FRONT:   # forward_geo pads to the canonical grid while the exported graph keeps the front grid -> crop for comparison
        og = {k: (M._front_crop(v) if k in ("lane", "hm", "reg", "stationary", "risk", "traj") else v) for k, v in og.items()}
ego_b, ego_g = ob[6][0, :36].view(3, 6, 2), og["ego"][0, :36].view(3, 6, 2)
k = int(og["ego"][0, 36:39].argmax())
if net.polar is not None:                                            # polar: dense outputs of the exported graph are on the polar grid, so compare ego directly and lane after polar.to_cart
    lane_c = net.polar.to_cart(ob[0], "lane")
    print(f"[parity] polar export vs forward_geo: selected path max|d| {float((ego_b[k] - ego_g[k]).abs().max()):.3f} m, "
          f"lane(polar->cart) argmax agree {float((lane_c.argmax(1) == og['lane'].argmax(1)).float().mean()):.4f}; polar lane {tuple(ob[0].shape)} hm {tuple(ob[1].shape)}", flush=True)
else:
    print(f"[parity] baked vs forward_geo: selected path max|d| {float((ego_b[k] - ego_g[k]).abs().max()):.3f} m, "
          f"lane argmax agree {float((ob[0].argmax(1) == og['lane'].argmax(1)).float().mean()):.4f}, "
          f"hm max|d| {float((ob[1] - og['hm']).abs().max()):.3f}", flush=True)
path = os.path.join(a.out, f"mite_{a.vt}{'_' + a.npu if a.npu else ('_conv1x1' if a.conv1x1 else '')}{'_f3' if a.f3 else ''}{'_L' if a.with_lidar else ''}.onnx")
if (a.lr_merge or a.lr_hconv) and M.LIFT_LR and M.LR_RECT:                      # drop U/V after the parity check and before export (avoids the 2 GB protobuf limit)
    net.vt_lr.drop_uv()
    if M.GIPM:
        net.gvt_lr.drop_uv()
if a.lr_rows_only and not (a.lr_merge or a.lr_hconv) and M.LIFT_LR and M.LR_RECT:   # rows-only two-stage U/V form: U/V were copied into prepare_npu's Um/Vm buffers (rank-pruned), so drop them
    net.vt_lr.drop_uv()
    if M.GIPM:
        net.gvt_lr.drop_uv()
t0 = time.time(); M.export(net, path, with_lidar=a.with_lidar)
try:
    import onnxruntime as ort
except ImportError:                 # onnxruntime not installed -> skip the parity check
    ort = None
_so = ort and ort.SessionOptions()
if _so is not None:
    _so.intra_op_num_threads = int(os.environ.get("ORT_THREADS", "8")); _so.inter_op_num_threads = 1   # may share the machine with training: limit CPU usage
s = ort and ort.InferenceSession(path, _so, providers=["CPUExecutionProvider"])
feed = {"imgs": u8[0].to(torch.uint8).numpy(), "v0": v0.view(1, 1).numpy()} if a.npu == "npu" else {"imgs": u8.to(torch.uint8).numpy(), "v0": v0.numpy()}
if a.with_lidar: feed["lidar_bev"] = lid.numpy()
if s:                                                                                             # F3 variants with acc / acc,brk inputs (fed zeros, as in the PyTorch forward)
    for i in s.get_inputs():
        if i.name in ("acc", "accbrk"): feed[i.name] = np.zeros((1, int(i.shape[1])), np.float32)
_ds = [float(np.abs(x - y.numpy()).max()) for x, y in zip(s.run(None, feed), ob)] if s else [float("nan")]
d = max(_ds)
print(f"[export] {path} {os.path.getsize(path)/1e6:.1f} MB [{time.time()-t0:.0f}s]; ORT vs PyTorch max|d| {d:.2e}"
      + (f"  per-output {[(n, round(v, 4)) for n, v in zip(M.OUTPUT_NAMES, _ds)]}" if d > 1e-2 else ""), flush=True)

if a.npu == "npu":
    import onnx, collections
    mm = onnx.load(path); cnt = collections.Counter(n.op_type for n in mm.graph.node)
    nd = [n.name for n in mm.graph.node]
    print("[npu-check] non-const nodes", len(mm.graph.node) - cnt["Constant"], sorted(cnt.items(), key=lambda x: -x[1]), flush=True)
    bad = {"Gather", "Resize", "Gemm", "Unsqueeze", "ScatterND", "GridSample", "Einsum"} & set(cnt)
    assert not bad, bad
    from onnx import shape_inference
    mi = shape_inference.infer_shapes(mm)
    hi = [(vi.name, len(vi.type.tensor_type.shape.dim)) for vi in list(mi.graph.value_info) + list(mi.graph.input) + list(mi.graph.output) if len(vi.type.tensor_type.shape.dim) > 4]
    print("[npu-check] tensors with >4 dims:", hi[:5], flush=True)
    assert not hi, hi
    raise SystemExit(0)
# ---- Orin demo variant: METEOR runtime/renderer naming (lane/depth/seg2d as uint8 argmax + lane_logit) ----
class _OrinDemo(torch.nn.Module):
    def __init__(self, n):
        super().__init__(); self.n = n
    def forward(self, imgs, v0, lidar=None):
        o = self.n(imgs, v0, lidar)
        lane_l, hm, reg, st, rk, tj, ego, tl, dep, s2 = o[:10]
        return (lane_l.argmax(1).to(torch.uint8), dep.argmax(2).to(torch.uint8), s2.argmax(2).to(torch.uint8),
                hm, reg, ego, tj, st, tl, rk, lane_l)
dp = os.path.join(a.out, f"mite_{a.vt}{'_L' if a.with_lidar else ''}_orindemo.onnx")
names_in = ["imgs", "v0"] + (["lidar_bev"] if a.with_lidar else [])
args_in = (u8.to(torch.uint8), v0) + ((lid,) if a.with_lidar else ())
torch.onnx.export(_OrinDemo(net).eval(), args_in, dp, opset_version=17, input_names=names_in,
                  output_names=["lane", "depth", "seg2d", "hm", "reg", "ego", "traj", "stationary", "tl", "risk", "lane_logit"],
                  do_constant_folding=True, dynamo=False)
print(f"[export] orin-demo variant {dp} {os.path.getsize(dp)/1e6:.1f} MB", flush=True)

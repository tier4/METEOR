#!/usr/bin/env python3
"""METEOR TensorRT streaming runtime.

Wraps a serialized engine (build_engine.md / trtexec) behind a simple
per-frame API that owns the temporal recurrence:

    rt = MeteorRT("meteor_v26_fp16.engine")
    for frame in stream:
        out = rt.infer(imgs, K, T_cam_ego, v0, ego_pose_xyyaw)
        boxes = decode_boxes(out["hm"], out["reg"])       # 3D detection
        ...

The engine graph is static: prev_bev / warp_theta are ordinary inputs and
raw_bev is an ordinary output. This class feeds raw_bev back as the next
frame's prev_bev and derives warp_theta from consecutive ego poses, i.e.
the recurrence lives on the host, not in the graph.
"""
import math

import numpy as np

BEV_H, BEV_W = 800, 500
DET_RES = 0.4

try:                                     # imported lazily so decode utils
    import pycuda.autoinit               # remain usable without a GPU
    import pycuda.driver as cuda
    import tensorrt as trt
    _TRT = True
except Exception:                        # pragma: no cover
    _TRT = False

INPUTS = ["imgs", "K", "T_cam_ego", "v0", "hist_bev", "hist_theta"]
OUTPUTS = ["lane", "depth", "seg2d", "hm", "reg", "hm2d", "reg2d",
           "ego", "occ", "traj", "stationary", "tl", "risk", "flow",
           "lg_pts", "lg_meta", "lg_adj", "unk", "raw_bev"]
HIST_OFFS = (2, 6, 14)          # slots at t-0.4 / -1.2 / -2.8 s (5 Hz frames)
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def preprocess_images(imgs_bgr):
    """list of 8 HxWx3 BGR uint8 (432x768) -> [1,8,3,432,768] float32."""
    out = np.stack([((im[:, :, ::-1] / 255.0 - MEAN) / STD).transpose(2, 0, 1)
                    for im in imgs_bgr]).astype(np.float32)
    return out[None]


def make_warp_theta(pose_prev, pose_cur):
    """global (x, y, yaw) of the previous and current frame -> theta [1,2,3].

    Same math as training (bevlane.model.make_warp_theta) with the relative
    pose computed here: p_prev = R(dyaw) p_cur + t.
    """
    cp, sp = math.cos(pose_prev[2]), math.sin(pose_prev[2])
    dx, dy = pose_cur[0] - pose_prev[0], pose_cur[1] - pose_prev[1]
    tx = cp * dx + sp * dy
    ty = -sp * dx + cp * dy
    dyaw = pose_cur[2] - pose_prev[2]
    a = 0.1 * (BEV_H - 1)
    b = 0.1 * (BEV_W - 1)
    cd, sd = math.cos(dyaw), math.sin(dyaw)
    Cx = cd * (80 - a) - sd * (50 - b) + tx
    Cy = sd * (80 - a) + cd * (50 - b) + ty
    th = np.zeros((1, 2, 3), np.float32)
    th[0, 0] = (cd, a * sd / b, ((50 - b) - Cy) / b)
    th[0, 1] = (-(b / a) * sd, cd, ((80 - a) - Cx) / a)
    return th


def _identity_theta():
    return np.array([[[1, 0, 0], [0, 1, 0]]], np.float32)


class MeteorRT:
    def __init__(self, engine_path):
        assert _TRT, "tensorrt / pycuda not available"
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            rt_ = trt.Runtime(logger)
            try:                    # accept version-compatible engines
                rt_.engine_host_code_allowed = True
            except AttributeError:
                pass
            self.engine = rt_.deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        self.host, self.dev, self.shapes = {}, {}, {}
        self._zeroed = False
        for i in range(self.engine.num_io_tensors):
            nm = self.engine.get_tensor_name(i)
            shp = tuple(self.engine.get_tensor_shape(nm))
            dt = trt.nptype(self.engine.get_tensor_dtype(nm))
            self.shapes[nm] = shp
            self.host[nm] = cuda.pagelocked_empty(
                int(np.prod(shp)), dtype=dt)
            self.dev[nm] = cuda.mem_alloc(self.host[nm].nbytes)
            self.ctx.set_tensor_address(nm, int(self.dev[nm]))
        self.stream = cuda.Stream()
        # DEVICE-RESIDENT temporal ring: raw_bev is 153 MB/frame and the
        # 3-slot hist_bev is 460 MB -- staging them through the host cost
        # >1 GB of copies per frame (~350 ms). Keep the last max(HIST_OFFS)
        # raw_bev tensors on the GPU and splice hist_bev with D2D copies.
        self._ring_n = max(HIST_OFFS)
        self._slot_bytes = None
        self._ring = None
        self._ring_pose = [None] * self._ring_n
        self._ring_t = [-1] * self._ring_n
        if "raw_bev" in self.shapes and "hist_bev" in self.shapes:
            self._slot_bytes = self.host["raw_bev"].nbytes
            self._ring = [cuda.mem_alloc(self._slot_bytes)
                          for _ in range(self._ring_n)]
        self._hist = {}                 # legacy host path (no raw_bev I/O)
        self._t = 0

    def reset(self):
        """call at scene boundaries: drops the temporal state."""
        self._hist.clear()
        self._ring_pose = [None] * self._ring_n
        self._ring_t = [-1] * self._ring_n
        self._t = 0

    def infer(self, imgs, K, T_cam_ego, v0, pose=None):
        """imgs [1,8,3,432,768] f32 (preprocess_images), K [1,8,3,3],
        T_cam_ego [1,8,4,4], v0 scalar, pose (x, y, yaw) or None.

        Returns {name: ndarray}. The temporal memory is maintained here:
        each frame's raw_bev is stored with its pose and the three history
        slots (t-0.4 / -1.2 / -2.8 s) are fed back on the next call. Slots
        with no history yet are zero-filled with an identity warp, exactly
        as training does for scene starts."""
        ht = np.zeros(self.shapes["hist_theta"], np.float32)
        if not self._zeroed:
            for nm_ in list(self.host):
                self.host[nm_][:] = 0
                cuda.memcpy_htod_async(self.dev[nm_], self.host[nm_],
                                       self.stream)
            self.stream.synchronize()
            self._zeroed = True
        # splice hist_bev on-device: D2D from the raw_bev ring (~1 ms)
        # instead of a 460 MB host round-trip (~350 ms)
        hb_base = int(self.dev["hist_bev"])
        for i, off in enumerate(HIST_OFFS):
            ti = self._t - off
            slot = ti % self._ring_n if ti >= 0 else -1
            valid = (self._ring is not None and ti >= 0
                     and self._ring_t[slot] == ti and pose is not None
                     and self._ring_pose[slot] is not None)
            dst = hb_base + i * self._slot_bytes
            if valid:
                cuda.memcpy_dtod_async(dst, int(self._ring[slot]),
                                       self._slot_bytes, self.stream)
                ht[0, i] = make_warp_theta(self._ring_pose[slot], pose)[0]
            else:
                cuda.memset_d8_async(dst, 0, self._slot_bytes, self.stream)
                ht[0, i] = _identity_theta()[0]
        feed = {"imgs": imgs, "K": K, "T_cam_ego": T_cam_ego,
                "v0": np.array([v0], np.float32), "hist_theta": ht}
        for nm, v in feed.items():
            np.copyto(self.host[nm],
                      np.ascontiguousarray(v, np.float32).ravel())
            cuda.memcpy_htod_async(self.dev[nm], self.host[nm], self.stream)
        self.ctx.execute_async_v3(self.stream.handle)
        # store this frame's raw_bev in the device ring (D2D, no host copy)
        if self._ring is not None:
            slot = self._t % self._ring_n
            cuda.memcpy_dtod_async(int(self._ring[slot]),
                                   int(self.dev["raw_bev"]),
                                   self._slot_bytes, self.stream)
            self._ring_pose[slot] = pose
            self._ring_t[slot] = self._t
        out = {}
        outs = [nm for nm in OUTPUTS if nm in self.shapes
                and nm != "raw_bev"]
        for nm in outs:
            cuda.memcpy_dtoh_async(self.host[nm], self.dev[nm], self.stream)
        self.stream.synchronize()
        for nm in outs:
            out[nm] = self.host[nm].reshape(self.shapes[nm]).copy()
        self._t += 1
        return out


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def decode_boxes(hm, reg, stationary=None, thresh=0.3, topk=64):
    """hm [1,2,h,w], reg [1,6,h,w] -> list of dicts (one per box)."""
    p = _sigmoid(hm[0])
    C, H, W = p.shape
    # 3x3 NMS
    pad = np.pad(p, ((0, 0), (1, 1), (1, 1)), constant_values=-1)
    mx = np.max([pad[:, 1 + dr:H + 1 + dr, 1 + dc:W + 1 + dc]
                 for dr in (-1, 0, 1) for dc in (-1, 0, 1)], axis=0)
    p = np.where(p == mx, p, 0)
    flat = p.reshape(-1)
    idx = np.argpartition(flat, -topk)[-topk:]
    boxes = []
    for i in idx[np.argsort(-flat[idx])]:
        sc = float(flat[i])
        if sc <= thresh:
            break
        cls, rc = divmod(int(i), H * W)
        ri, ci = divmod(rc, W)
        o = reg[0, :, ri, ci]
        boxes.append({
            "cls": "vehicle" if cls == 0 else "vru", "score": sc,
            "x": 80.0 - (ri + float(o[0])) * DET_RES,
            "y": 50.0 - (ci + float(o[1])) * DET_RES,
            "l": float(np.exp(o[2])), "w": float(np.exp(o[3])),
            "yaw": float(np.arctan2(o[4], o[5])),
            "stationary": (bool(stationary[0, 0, ri, ci] > 0)
                           if stationary is not None else None)})
    return boxes


def decode_agent_traj(traj, boxes):
    """attach 6x0.5s future waypoints (ego frame, metres) to decoded boxes."""
    for b in boxes:
        ri = int((80.0 - b["x"]) / DET_RES)
        ci = int((50.0 - b["y"]) / DET_RES)
        if 0 <= ri < traj.shape[-2] and 0 <= ci < traj.shape[-1]:
            wp = traj[0, :, ri, ci].reshape(6, 2)
            b["future"] = [(b["x"] + float(dx), b["y"] + float(dy))
                           for dx, dy in wp]
    return boxes

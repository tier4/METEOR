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

INPUTS = ["imgs", "K", "T_cam_ego", "v0", "prev_bev", "warp_theta"]
OUTPUTS = ["lane", "depth", "seg2d", "hm", "reg", "hm2d", "reg2d",
           "ego", "occ", "traj", "stationary", "raw_bev"]
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
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        self.host, self.dev, self.shapes = {}, {}, {}
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
        self._prev_bev = np.zeros(self.shapes["prev_bev"], np.float32)
        self._prev_pose = None

    def reset(self):
        """call at scene boundaries: drops the temporal state."""
        self._prev_bev[:] = 0
        self._prev_pose = None

    def infer(self, imgs, K, T_cam_ego, v0, pose=None):
        """imgs [1,8,3,432,768] f32 (preprocess_images), K [1,8,3,3],
        T_cam_ego [1,8,4,4], v0 scalar, pose (x, y, yaw) or None.
        Returns {name: ndarray}; feeds raw_bev/theta forward automatically."""
        if pose is None or self._prev_pose is None:
            theta = _identity_theta()
            if pose is None:
                self._prev_bev[:] = 0        # no odometry -> no history
        else:
            theta = make_warp_theta(self._prev_pose, pose)
        feed = {"imgs": imgs, "K": K, "T_cam_ego": T_cam_ego,
                "v0": np.array([v0], np.float32),
                "prev_bev": self._prev_bev, "warp_theta": theta}
        for nm, v in feed.items():
            np.copyto(self.host[nm],
                      np.ascontiguousarray(v, np.float32).ravel())
            cuda.memcpy_htod_async(self.dev[nm], self.host[nm], self.stream)
        self.ctx.execute_async_v3(self.stream.handle)
        out = {}
        for nm in OUTPUTS:
            cuda.memcpy_dtoh_async(self.host[nm], self.dev[nm], self.stream)
        self.stream.synchronize()
        for nm in OUTPUTS:
            out[nm] = self.host[nm].reshape(self.shapes[nm]).copy()
        self._prev_bev = out["raw_bev"]
        self._prev_pose = pose
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

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
import os
import time as _time

import numpy as np

BEV_H, BEV_W = 800, 500
DET_RES = 0.4
STAT_LOGIT_THRESH = float(os.environ.get("METEOR_STAT_LOGIT_THRESH", "0"))

try:                                     # imported lazily so decode utils
    # pycuda.autoinit は使わない (2026-08-25): autoinit は make_context() で
    # **新規**コンテキストを作るが、TensorRT (cudart) は**プライマリ**
    # コンテキストで動く。別コンテキストのストリームを execute_async_v3 に
    # 渡すと "Cuda Runtime (invalid resource handle)" が全リフォーマット層で
    # 出る (ノート PC の GPU で実害)。Orin では pycuda が無く cudart
    # シムに落ちるため発症しなかった。プライマリを retain して共有する。
    import pycuda.driver as cuda
    cuda.init()
    _pyc_ctx = cuda.Device(0).retain_primary_context()
    _pyc_ctx.push()
    import tensorrt as trt
    _TRT = True
except Exception:                        # pragma: no cover
    # JetPack 7 / CUDA 13: apt python3-pycuda pins CUDA 12 deps and the pip
    # wheel does not build, but NVIDIA's own cuda-python installs fine. This
    # shim exposes exactly the seven pycuda calls this file makes, backed by
    # cudart, so MeteorRT runs unmodified on the Orin.
    try:
        from cuda.bindings import runtime as _rt
        import tensorrt as trt

        def _ck(res):
            err = res[0]
            assert int(err) == 0, f"cudart error {err}"
            return res[1] if len(res) > 1 else None

        class _Stream:
            def __init__(self):
                self.handle = _ck(_rt.cudaStreamCreate())
            def synchronize(self):
                _ck(_rt.cudaStreamSynchronize(self.handle))

        class _CudaShim:
            Stream = _Stream
            @staticmethod
            def pagelocked_empty(shape, dtype):
                import numpy as _np
                n = int(_np.prod(shape)) * _np.dtype(dtype).itemsize
                ptr = _ck(_rt.cudaHostAlloc(n, 0))
                import ctypes
                buf = (ctypes.c_byte * n).from_address(ptr)
                return _np.frombuffer(buf, dtype=dtype).reshape(shape)
            @staticmethod
            def mem_alloc(nbytes):
                return _ck(_rt.cudaMalloc(int(nbytes)))
            @staticmethod
            def memcpy_htod_async(dst, src, stream):
                _ck(_rt.cudaMemcpyAsync(int(dst), src.ctypes.data,
                    src.nbytes, _rt.cudaMemcpyKind.cudaMemcpyHostToDevice,
                    stream.handle))
            @staticmethod
            def memcpy_dtoh_async(dst, src, stream):
                _ck(_rt.cudaMemcpyAsync(dst.ctypes.data, int(src),
                    dst.nbytes, _rt.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                    stream.handle))
            @staticmethod
            def memcpy_dtod_async(dst, src, nbytes, stream):
                _ck(_rt.cudaMemcpyAsync(int(dst), int(src), int(nbytes),
                    _rt.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
                    stream.handle))
            @staticmethod
            def memset_d8_async(dst, val, nbytes, stream):
                _ck(_rt.cudaMemsetAsync(int(dst), int(val), int(nbytes),
                                         stream.handle))

        cuda = _CudaShim()
        _TRT = True
    except Exception:
        _TRT = False

INPUTS = ["imgs", "K", "T_cam_ego", "v0", "hist_bev", "hist_theta", "lidar_bev", "lidar_flag"]
_ZERO_HIST = os.environ.get("METEOR_ZERO_HIST", "0") == "1"
if _ZERO_HIST:
    print("[rt] METEOR_ZERO_HIST=1: 履歴スロットを零固定 (学習条件に一致)", flush=True)
OUTPUTS = ["lane", "depth", "seg2d", "hm", "reg",
           "hm2d_s0", "hm2d_s1", "hm2d_s2",
           "reg2d_s0", "reg2d_s1", "reg2d_s2",
           "ego", "occ", "traj", "stationary", "tl", "risk", "flow",
           "lg_pts", "lg_meta", "lg_adj", "unk", "lane_logit", "depth_mean", "raw_bev"]
HIST_OFFS = (2, 6, 14)          # slots at t-0.4 / -1.2 / -2.8 s (5 Hz frames)
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def preprocess_images(imgs_bgr):
    """list of 8 HxWx3 BGR uint8 (432x768) -> [1,8,3,432,768] float32."""
    out = np.stack([((im[:, :, ::-1] / 255.0 - MEAN) / STD).transpose(2, 0, 1)
                    for im in imgs_bgr]).astype(np.float32)
    return out[None]


def make_warp_theta(pose_prev, pose_cur, bev_h=BEV_H, bev_w=BEV_W):
    """global (x, y, yaw) of the previous and current frame -> theta [1,2,3].

    Same math as training (bevlane.model.make_warp_theta) with the relative
    pose computed here: p_prev = R(dyaw) p_cur + t.
    bev_h/bev_w MUST match the engine's hist_bev grid: the 800-row constant
    silently mis-warped history for rear-40 (600-row) engines (2026-08-13).
    """
    cp, sp = math.cos(pose_prev[2]), math.sin(pose_prev[2])
    dx, dy = pose_cur[0] - pose_prev[0], pose_cur[1] - pose_prev[1]
    tx = cp * dx + sp * dy
    ty = -sp * dx + cp * dy
    dyaw = pose_cur[2] - pose_prev[2]
    a = 0.1 * (bev_h - 1)
    b = 0.1 * (bev_w - 1)
    cd, sd = math.cos(dyaw), math.sin(dyaw)
    Cx = cd * (80 - a) - sd * (50 - b) + tx
    Cy = sd * (80 - a) + cd * (50 - b) + ty
    th = np.zeros((1, 2, 3), np.float32)
    th[0, 0] = (cd, a * sd / b, ((50 - b) - Cy) / b)
    th[0, 1] = (-(b / a) * sd, cd, ((80 - a) - Cx) / a)
    return th


def _identity_theta():
    return np.array([[[1, 0, 0], [0, 1, 0]]], np.float32)



_RT_PROF = os.environ.get("METEOR_RT_PROFILE") == "1"
_RT_T = {}


def _rt_mark(rt, key, t0):
    """METEOR_RT_PROFILE=1 のときだけ段階時間を積む。

    90ms 目標に対し GPU 計算 92.2ms / 実ランタイム 109.2ms と 17ms の開きが
    あり、その内訳を取るために入れた。同期を挟むので重なりは潰れるが、
    「どの段階が何 ms 使うか」はこれでしか分からない。
    """
    if not _RT_PROF:
        return t0
    rt.stream.synchronize()
    t1 = _time.perf_counter()
    d = _RT_T.setdefault(key, [0.0, 0])
    d[0] += (t1 - t0) * 1000.0
    d[1] += 1
    return t1


def rt_profile_report(reset=True):
    if not _RT_T:
        return "(プロファイル無効)"
    tot = sum(v[0] / max(v[1], 1) for v in _RT_T.values())
    lines = [f"{'段階':<22}{'平均 ms':>10}{'割合':>8}"]
    for k, v in sorted(_RT_T.items(), key=lambda kv: -kv[1][0] / max(kv[1][1], 1)):
        m = v[0] / max(v[1], 1)
        lines.append(f"{k:<22}{m:>10.2f}{m / max(tot, 1e-9) * 100:>7.1f}%")
    lines.append(f"{'合計':<22}{tot:>10.2f}")
    if reset:
        _RT_T.clear()
    return "\n".join(lines)


class MeteorRT:
    def __init__(self, engine_path, skip_outputs=(), n_out_slots=1):
        # skip_outputs: output names to leave on the device (no D2H). The
        # realtime renderer reads 8 of the 21 outputs; the other 13 cost
        # ~40 MB of D2H per frame for nothing.
        # n_out_slots: host-side output buffers are multi-buffered so a
        # consumer can hold frame t's outputs while frame t+1 infers. With
        # one slot the caller must COPY every output before the next infer
        # (~25 MB, ~10 ms on the producer's critical path); with N slots the
        # consumer reads the pinned buffers directly and releases the slot.
        self.skip_outputs = set(skip_outputs)
        self.n_out_slots = max(1, int(n_out_slots))
        assert _TRT, "tensorrt / pycuda not available"
        logger = trt.Logger(trt.Logger.WARNING)
        # custom-plugin engines (MeteorLift): load the .so so its creator is
        # registered before deserialization. METEOR_PLUGIN_SO=/path/lib.so
        import os as _os
        _so = _os.environ.get("METEOR_PLUGIN_SO")
        if _so:
            import ctypes
            ctypes.CDLL(_so, mode=ctypes.RTLD_GLOBAL)
            trt.init_libnvinfer_plugins(logger, "")
        with open(engine_path, "rb") as f:
            rt_ = trt.Runtime(logger)
            try:                    # accept version-compatible engines
                rt_.engine_host_code_allowed = True
            except AttributeError:
                pass
            self.engine = rt_.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(
                "エンジンのデシリアライズに失敗 (直前の TRT エラーを参照)。"
                "典型: TensorRT の版違い (パッチ版まで一致が必要)。"
                "同梱の ONNX + 較正キャッシュから現地ビルドすること")
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
        # extra host slots for the outputs the caller reads (slot 0 = host)
        # --split-hist で焼いたエンジンは hist_bev0/1/2 の 3 入力を持つ。
        # このときリングのスロットをそのままバインドできるので、毎フレーム
        # 230MB (fp16) の D2D コピー (実測 6.2 ms) が丸ごと不要になる。
        self._split_hist = all(f"hist_bev{i}" in self.shapes for i in range(3))
        # --no-hist エンジン (2026-09-05): 履歴入力を持たない。splice も
        # hist_theta も raw_bev リングも不要 (学習条件 = 履歴零に一致)。
        self._no_hist = ("hist_bev" not in self.shapes
                         and "hist_bev0" not in self.shapes)
        if self._no_hist:
            print("[rt] 履歴入力なしエンジンを検出 (no-hist)", flush=True)
        # 履歴が無いスロットに束ねる 0 バッファ。hist_bev0 用に確保済みの
        # 領域をそのまま使う (初回 infer の一括ゼロ化で 0 になる。以後は
        # アドレスを差し替えるだけなので中身は書き換わらない)。
        self._zero_slot = self.dev["hist_bev0"] if self._split_hist else None
        if self._split_hist:
            print("[rt] hist_bev 分割入力を検出 -> D2D コピーを省く", flush=True)
        self.host_slots = [self.host]
        out_names = [nm for nm in OUTPUTS if nm in self.shapes
                     and nm != "raw_bev" and nm not in self.skip_outputs]
        for _ in range(self.n_out_slots - 1):
            sl = dict(self.host)
            for nm in out_names:
                sl[nm] = cuda.pagelocked_empty(
                    int(np.prod(self.shapes[nm])),
                    dtype=self.host[nm].dtype)
            self.host_slots.append(sl)
        self.stream = cuda.Stream()
        # DEVICE-RESIDENT temporal ring: raw_bev is 153 MB/frame and the
        # 3-slot hist_bev is 460 MB -- staging them through the host cost
        # >1 GB of copies per frame (~350 ms). Keep the last max(HIST_OFFS)
        # raw_bev tensors on the GPU and splice hist_bev with D2D copies.
        self._ring_n = max(HIST_OFFS)
        self._slot_bytes = None
        self._ring = None
        self._ring_disabled = False
        self._ring_pose = [None] * self._ring_n
        self._ring_t = [-1] * self._ring_n
        # 分割版エンジンは hist_bev ではなく hist_bev0/1/2 を持つ。
        # スロット長はどちらから測っても同じ (1 スロット分)。
        _hb_name = "hist_bev" if "hist_bev" in self.shapes else "hist_bev0"
        if "raw_bev" in self.shapes and _hb_name in self.shapes:
            # スロット長は hist_bev 側から決める。fp16 IO を hist_bev だけに
            # 指定したエンジンでは raw_bev が fp32 のまま残り、raw_bev 基準だと
            # 2 倍の長さで memset して cudaErrorInvalidValue になる
            # (2026-08-14, 8cam エンジンで実害)。
            _hb = self.host[_hb_name].nbytes
            if _hb_name == "hist_bev":          # 3 スロットが 1 本に入っている
                _hb //= max(len(HIST_OFFS), 1)
            self._slot_bytes = _hb
            if self.host["raw_bev"].nbytes != _hb:
                print(f"[rt] raw_bev({self.host['raw_bev'].dtype}) と "
                      f"{_hb_name}({self.host[_hb_name].dtype}) の型が不一致 "
                      f"-> 時系列リングを無効化 (履歴ゼロで動作)", flush=True)
                self._ring_disabled = True
            if not self._ring_disabled:
                self._ring = [cuda.mem_alloc(self._slot_bytes)
                              for _ in range(self._ring_n)]
        self._graph = None
        self._graph_warm = 0
        self._hist = {}                 # legacy host path (no raw_bev I/O)
        self._t = 0

    def pinned_input_slots(self, n):
        """imgs 用の pinned スロットを n 個返す ([1,N,3,H,W] のエンジン dtype)。
        デコード側がここへ直接書けば infer() の CPU コピーが消える。"""
        shp = tuple(self.shapes["imgs"]); dt = self.host["imgs"].dtype
        slots = []
        for _ in range(n):
            flat = cuda.pagelocked_empty(int(np.prod(shp)), dtype=dt)
            slots.append(flat.reshape(shp))
        self._in_slot_addrs = getattr(self, "_in_slot_addrs", set())
        self._in_slot_addrs.update(int(a.ctypes.data) for a in slots)
        return slots

    def _is_input_slot(self, v):
        try:
            return int(v.ctypes.data) in getattr(self, "_in_slot_addrs", ()) \
                and v.dtype == self.host["imgs"].dtype and v.flags["C_CONTIGUOUS"]
        except Exception:
            return False

    def reset(self):
        """call at scene boundaries: drops the temporal state."""
        self._hist.clear()
        self._ring_pose = [None] * self._ring_n
        self._ring_t = [-1] * self._ring_n
        self._t = 0

    def infer(self, imgs, K, T_cam_ego, v0, pose=None, out_slot=0, lidar_bev=None):
        """imgs [1,8,3,432,768] f32 (preprocess_images), K [1,8,3,3],
        T_cam_ego [1,8,4,4], v0 scalar, pose (x, y, yaw) or None.

        Returns {name: ndarray}. The temporal memory is maintained here:
        each frame's raw_bev is stored with its pose and the three history
        slots (t-0.4 / -1.2 / -2.8 s) are fed back on the next call. Slots
        with no history yet are zero-filled with an identity warp, exactly
        as training does for scene starts."""
        _t0 = _time.perf_counter() if _RT_PROF else None
        ht = (np.zeros(self.shapes["hist_theta"], np.float32)
              if "hist_theta" in self.shapes else None)
        if not self._zeroed:
            for nm_ in list(self.host):
                self.host[nm_][:] = 0
                cuda.memcpy_htod_async(self.dev[nm_], self.host[nm_],
                                       self.stream)
            self.stream.synchronize()
            self._zeroed = True
        # splice hist_bev on-device: D2D from the raw_bev ring (~1 ms)
        # instead of a 460 MB host round-trip (~350 ms)
        hb_base = (int(self.dev["hist_bev"])
                   if (not self._split_hist and not self._no_hist) else 0)
        for i, off in (enumerate(HIST_OFFS) if not self._no_hist else ()):
            ti = self._t - off
            slot = ti % self._ring_n if ti >= 0 else -1
            valid = (self._ring is not None and ti >= 0
                     and self._ring_t[slot] == ti and pose is not None
                     and self._ring_pose[slot] is not None)
            # METEOR_ZERO_HIST=1: 履歴スロットを常に零 (学習条件に一致)。
            # 2026-09-04 に判明: dataset.py の ego キャッシュ衝突で E2E 全ラウンドの
            # 学習・検証は履歴零だった。実履歴を流すと同一フレームで E2E ADE が
            # 約 35% 悪化 (v132 0.611→0.806)。実履歴で学習した世代 (v144〜) までは
            # 零固定が学習条件の再現になる。
            if _ZERO_HIST:
                valid = False
            dst = hb_base + i * self._slot_bytes
            _hb = self.shapes["hist_bev0" if self._split_hist else "hist_bev"]
            if self._split_hist:
                # コピーせず、そのスロット (無効なら 0 バッファ) を束ねる
                src = int(self._ring[slot]) if valid else int(self._zero_slot)
                self.ctx.set_tensor_address(f"hist_bev{i}", src)
                ht[0, i] = (make_warp_theta(self._ring_pose[slot], pose,
                                            bev_h=_hb[-2], bev_w=_hb[-1])[0]
                            if valid else _identity_theta()[0])
            elif valid:
                cuda.memcpy_dtod_async(dst, int(self._ring[slot]),
                                       self._slot_bytes, self.stream)
                ht[0, i] = make_warp_theta(self._ring_pose[slot], pose,
                                           bev_h=_hb[-2], bev_w=_hb[-1])[0]
            else:
                cuda.memset_d8_async(dst, 0, self._slot_bytes, self.stream)
                ht[0, i] = _identity_theta()[0]
        _t0 = _rt_mark(self, "B_hist_splice_D2D", _t0)
        feed = {"imgs": imgs, "K": K, "T_cam_ego": T_cam_ego,
                "v0": np.array([v0], np.float32)}
        if ht is not None:
            feed["hist_theta"] = ht
        if "lidar_bev" in self.shapes:          # --with-lidar エンジン: 無ければ零 (= カメラのみ)
            feed["lidar_bev"] = (lidar_bev if lidar_bev is not None
                                 else np.zeros(self.shapes["lidar_bev"], np.float32))
        if "lidar_flag" in self.shapes:         # ホスト側フラグ (グラフ内縮約の 14.8 ms を回避)
            feed["lidar_flag"] = np.array([1.0 if lidar_bev is not None else 0.0], np.float32)
        for nm, v in feed.items():
            # ゼロコピー入力 (2026-09-05): imgs が pinned_input_slots() の
            # スロットそのものなら CPU コピー (実測 4.6 ms) を省き、その
            # スロットから直接 H2D する (0.4 ms)。呼び出し側はデコード時に
            # スロットへ書き、infer() が返るまでスロットを再利用しない。
            if nm == "imgs" and self._is_input_slot(v):
                cuda.memcpy_htod_async(self.dev[nm], v.reshape(-1), self.stream)
                _t0 = _rt_mark(self, "C1_imgs_zerocopy", _t0)
                continue
            # cast to the ENGINE'S input dtype, not blanket float32 -- the
            # uint8-in graphs take images as uint8 and copyto refuses a
            # float32 -> uint8 downcast (rightly)
            np.copyto(self.host[nm],
                      np.ascontiguousarray(v, self.host[nm].dtype).ravel())
            if nm == "imgs":
                _t0 = _rt_mark(self, "C1_imgs_copyto", _t0)
            cuda.memcpy_htod_async(self.dev[nm], self.host[nm], self.stream)
        _t0 = _rt_mark(self, "C_input_H2D", _t0)
        # CUDA Graph (2026-08-14 実測: 65.71 -> 62.55 ms, -3.2 ms):
        # refiner だけで 522 レイヤある本エンジンでは、Orin の弱い CPU での
        # カーネル起動が実測できる量を占める。形状は静的なので 1 回だけ
        # キャプチャして以後は再生する。入出力アドレスは固定 (self.dev) の
        # ままなので、グラフは毎フレーム同じバッファを読み書きする。
        if self._graph is not None:
            _ck(_rt.cudaGraphLaunch(self._graph, self.stream.handle))
        elif (self._graph_warm >= 2
              and os.environ.get("METEOR_CUDAGRAPH") == "1"):
            # 既定 OFF (2026-08-15): 本エンジンは補助ストリームを 6 本使うため、
            # enqueueV3 を単純にストリームキャプチャすると一部の処理がグラフに
            # 入らず、出力が更新されない (デモ映像が固定される実害を確認)。
            # trtexec 側の --useCudaGraph は独自にストリームを整えており
            # 同じ問題は出ない。ランタイムで使うには補助ストリームを 1 本に
            # した専用エンジンでの数値検証が必要。
            try:
                _ck(_rt.cudaStreamBeginCapture(
                    self.stream.handle,
                    _rt.cudaStreamCaptureMode.cudaStreamCaptureModeThreadLocal))
                self.ctx.execute_async_v3(self.stream.handle)
                g = _ck(_rt.cudaStreamEndCapture(self.stream.handle))
                self._graph = _ck(_rt.cudaGraphInstantiate(g, 0))
                _ck(_rt.cudaGraphLaunch(self._graph, self.stream.handle))
                print("[rt] CUDA Graph captured", flush=True)
            except Exception as e:                    # 失敗したら通常実行へ
                print(f"[rt] CUDA Graph 不可 ({e}); 通常実行", flush=True)
                self._graph = None
                self._graph_warm = -10**9
                self.ctx.execute_async_v3(self.stream.handle)
        else:
            self._graph_warm += 1
            self.ctx.execute_async_v3(self.stream.handle)
        _t0 = _rt_mark(self, "D_execute", _t0)
        # store this frame's raw_bev in the device ring (D2D, no host copy)
        if self._ring is not None:
            slot = self._t % self._ring_n
            cuda.memcpy_dtod_async(int(self._ring[slot]),
                                   int(self.dev["raw_bev"]),
                                   self._slot_bytes, self.stream)
            self._ring_pose[slot] = pose
            self._ring_t[slot] = self._t
        _t0 = _rt_mark(self, "E_ring_store_D2D", _t0)
        out = {}
        outs = [nm for nm in OUTPUTS if nm in self.shapes
                and nm != "raw_bev" and nm not in self.skip_outputs]
        hs = self.host_slots[out_slot % len(self.host_slots)]
        for nm in outs:
            cuda.memcpy_dtoh_async(hs[nm], self.dev[nm], self.stream)
        self.stream.synchronize()
        _t0 = _rt_mark(self, "F_output_D2H", _t0)
        copy_out = len(self.host_slots) == 1
        for nm in outs:
            v = hs[nm].reshape(self.shapes[nm])
            # single-slot callers get a copy (legacy safety); multi-slot
            # callers own the slot until they release it and read in place
            out[nm] = v.copy() if copy_out else v
        _rt_mark(self, "G_host_copy", _t0)
        self._t += 1
        return out


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# topk 64 -> 128 (2026-08-17): 混雑シーンで弱いピークが枠から溢れ、
# しきい値 0.10 で前後とも約 5pt の未検出を作っていた (実測)。学習側の
# 評価デコードは既存ラウンドとの比較可能性のため 64 のまま変えない。
def stationary_head_healthy(stationary, min_std=0.05, min_range=0.20):
    """Return False when an INT8 stationary output has collapsed.

    A healthy FP16 v115 map has std ~= 1.49, while the broken full-INT8
    engine is exactly constant.  Requiring both a little variance and range
    also catches near-constant quantised maps without second-guessing normal
    logits.
    """
    if stationary is None:
        return False
    s = np.asarray(stationary, dtype=np.float32)
    finite = s[np.isfinite(s)]
    if finite.size == 0:
        return False
    return (float(finite.std()) >= min_std and
            float(finite.max() - finite.min()) >= min_range)


def stationary_at(stationary, traj, ri, ci, stat_healthy=None,
                  displacement_thresh=0.5, logit_thresh=None):
    """Decode stationary state, using the 3 s trajectory as a safety path.

    ``displacement_thresh`` lies inside the training dead-band (stationary
    <=0.35 m, moving >=0.8 m), so it never contradicts a supervised example.
    The trajectory fallback is selected only if the explicit head is absent
    or demonstrably collapsed, so healthy-engine behaviour is unchanged.
    """
    if stat_healthy is None:
        stat_healthy = stationary_head_healthy(stationary)
    if logit_thresh is None:
        logit_thresh = STAT_LOGIT_THRESH
    if stat_healthy and (0 <= ri < stationary.shape[-2] and
                         0 <= ci < stationary.shape[-1]):
        return bool(stationary[0, 0, ri, ci] > logit_thresh), "head"
    if traj is not None and 0 <= ri < traj.shape[-2] and 0 <= ci < traj.shape[-1]:
        v = np.asarray(traj[0, :, ri, ci], dtype=np.float32)
        if v.size >= 39:
            k = int(np.argmax(v[36:39]))
            wp = v[k * 12:(k + 1) * 12]
        else:
            wp = v[:12]
        if wp.size == 12 and np.isfinite(wp).all():
            return bool(np.linalg.norm(wp.reshape(6, 2)[-1]) <
                        displacement_thresh), "trajectory"
    return None, "unavailable"


def decode_boxes(hm, reg, stationary=None, thresh=0.3, topk=128, traj=None):
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
    stat_ok = stationary_head_healthy(stationary)
    for i in idx[np.argsort(-flat[idx])]:
        sc = float(flat[i])
        if sc <= thresh:
            break
        cls, rc = divmod(int(i), H * W)
        ri, ci = divmod(rc, W)
        o = reg[0, :, ri, ci]
        is_stationary, stat_source = stationary_at(
            stationary, traj, ri, ci, stat_healthy=stat_ok)
        boxes.append({
            "cls": "vehicle" if cls == 0 else "vru", "score": sc,
            "x": 80.0 - (ri + float(o[0])) * DET_RES,
            "y": 50.0 - (ci + float(o[1])) * DET_RES,
            "l": float(np.exp(o[2])), "w": float(np.exp(o[3])),
            "yaw": float(np.arctan2(o[4], o[5])),
            "stationary": is_stationary,
            "stationary_source": stat_source})
    if os.environ.get("METEOR_BOX_NMS", "1") != "0":
        boxes = _box_nms(boxes)
    return boxes


def _box_nms(boxes, shrink=0.8, margin=0.2):
    """箱レベル NMS (2026-08-27)。3x3 ピーク NMS はセル間 1.2m までしか
    抑制できず、大型車で複数ピーク→多重箱が出る (ユーザー指摘)。
    スコア降順を前提に、既採用の同クラス箱の内側 (shrink 倍 + margin[m])
    に中心が入る箱を落とす。回転は既採用箱の yaw 座標系で判定。"""
    kept = []
    for b in boxes:
        dup = False
        for k in kept:
            if k["cls"] != b["cls"]:
                continue
            dx, dy = b["x"] - k["x"], b["y"] - k["y"]
            c, s = np.cos(k["yaw"]), np.sin(k["yaw"])
            lx = c * dx + s * dy
            ly = -s * dx + c * dy
            if abs(lx) < k["l"] / 2 * shrink + margin \
                    and abs(ly) < k["w"] / 2 * shrink + margin:
                dup = True
                break
        if not dup:
            kept.append(b)
    return kept


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

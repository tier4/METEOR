#!/usr/bin/env python3
"""Real-time METEOR demo on the Orin: pipelined inference + rendering.

orin_render.py runs the stages serially, so a frame costs infer + render.
This version overlaps them -- a producer thread owns the TRT engine and a
bounded queue hands outputs to the render loop -- so the effective rate is
max(infer, render), not their sum. Same drawing path as the offline renderer
(deploy/viz_np.py = the PyTorch demo's own functions), same layout.

The workstation realtime harness taught two lessons that shape this file: a renderer
that holds the GIL for hundreds of ms starves the GPU thread (so the queue is
bounded at 2 and the producer never waits on drawing), and a dead stage must
kill the whole pipeline loudly instead of deadlocking it (the sentinel).

    python3 deploy/orin_realtime.py --engine eng/v59_demo_fp16.engine \
        --root fast --stride 1 [--display] [--out out/rt.mp4] [--loop]

--display shows a live window on the Orin's desktop (q quits, space pauses);
without a display, pass --out to write the same frames to a video.
"""
import argparse
import json
import os
import queue
import sys
import threading
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deploy.runtime import MeteorRT, decode_boxes                # noqa: E402
from deploy import orin_render as R                              # noqa: E402
from deploy.viz_np import decode_boxes2d_ms_np                   # noqa: E402

CAMS = R.CAMS


_LIDAR = os.environ.get("METEOR_LIDAR", "0") == "1"
# METEOR_VLA_JSONL=<file>: overlay a VLA run (scene / hazards / rationale / command / waypoints per
# (scene, frame)) on the demo canvas (2026-09-10). Records keyed by (scene name, frame index).
_VLA = {}
if os.environ.get("METEOR_VLA_JSONL"):
    import json as _json
    for _l in open(os.environ["METEOR_VLA_JSONL"]):
        _r = _json.loads(_l)
        _VLA[(_r["scene"], int(_r["frame"]))] = _r
    print(f"[rt] VLA overlay: {len(_VLA)} records from {os.environ['METEOR_VLA_JSONL']}", flush=True)
# METEOR_VLA_LIVE=<ckpt_last.pt>: run the METEOR-VLA in-process on the engine's bev_tok output (the 13th
# head) every METEOR_VLA_EVERY frames (2026-09-10). One METEOR forward feeds both the E2E plan and the VLA.
_VLA_LIVE = None
if os.environ.get("METEOR_VLA_LIVE") or os.environ.get("METEOR_VLA_SERVER"):
    from deploy.vla_live import derive_cmd as _vla_derive_cmd
    if os.environ.get("METEOR_VLA_SERVER"):        # VLA worker(s) in their own process / environment
        from deploy.vla_live import VLAClient, VLAPool
        _addrs = [x for x in os.environ["METEOR_VLA_SERVER"].split(",") if x]
        _VLA_LIVE = VLAClient(_addrs[0]) if len(_addrs) == 1 else VLAPool(_addrs)
    else:
        from deploy.vla_live import VLALive
        _VLA_LIVE = VLALive(os.environ["METEOR_VLA_LIVE"])
    _VLA_EVERY = int(os.environ.get("METEOR_VLA_EVERY", "10"))
import json
_VLA_DUMP = open(os.environ["METEOR_VLA_DUMP"], "a") if os.environ.get("METEOR_VLA_DUMP") else None


def loader(scenes, root, stride, q_raw, stop, loop, in_slots=None, in_free=None):
    """Disk -> raw frames. Separate from inference so JPEG decode (~60 ms for
    seven cameras) overlaps the GPU instead of serialising with it -- that
    alone was the difference between 4.4 and ~6 FPS on the fp16 engine."""
    while not stop.is_set():
        for s in scenes:
            from deploy.t4input import is_t4_scene, load_t4_scene
            _sd = os.path.join(root, s)
            if is_t4_scene(_sd):
                # Read raw t4dataset without conversion (2026-08-25): interpret
                # the annotation on the fly and downscale images to 768x432 on load.
                m, _t4v0, _t4pose = load_t4_scene(_sd, CAMS)
                _reduced = True     # a full 2880x1860 decode takes ~1s/frame;
                                    # the JPEG 1/4 reduced decode (720x465)
                                    # is 4-8x faster, and the slight upscale
                                    # from there to 768x432 barely hurts quality
            else:
                m = json.load(open(os.path.join(_sd, "manifest.json")))
                _t4v0 = _t4pose = None
                _reduced = False
            K = np.stack([np.array(m["cams"][c]["K"], np.float32)
                          for c in CAMS])[None]
            Tc = np.stack([np.linalg.inv(np.array(
                m["cams"][c]["T_ego_cam"], np.float32))
                for c in CAMS])[None]
            if _t4v0 is not None:
                v0s, poses = _t4v0, _t4pose
            else:
                try:
                    z = np.load(os.path.join(root, s, "ego_motion.npz"))
                    v0s = z["v0"]
                    poses = z["pose"] if "pose" in z else None
                    wps_nav = z["wp"] if "wp" in z else None      # route -> navigation command for the VLA
                except Exception:
                    v0s = poses = wps_nav = None
            _last_cmd = "keep_lane"
            for f in m["frames"][::stride]:
                if stop.is_set():
                    return
                raw = {}
                ok = True
                for c in CAMS:
                    _fl = cv2.IMREAD_REDUCED_COLOR_4 if _reduced \
                        else cv2.IMREAD_COLOR
                    im = cv2.imread(os.path.join(root, s,
                                                 f["imgs"].get(c, "_")), _fl)
                    if im is None:
                        ok = False
                        break
                    if im.shape[1] != 768:     # raw t4 -> training resolution
                        im = cv2.resize(im, (768, 432),
                                        interpolation=cv2.INTER_AREA)
                    raw[c] = im
                if not ok:
                    continue
                if _VLA:      # METEOR_VLA_JSONL: attach the VLA record (this frame, else the previous one)
                    raw["_vla"] = _VLA.get((s, int(f["frame"]))) or _VLA.get((s, int(f["frame"]) - 1))
                if _VLA_LIVE is not None:
                    raw["_scene"], raw["_frame"] = s, int(f["frame"])
                    _fi = int(f["frame"])
                    if wps_nav is not None and _fi < len(wps_nav) and np.abs(wps_nav[_fi]).sum() > 0:
                        _last_cmd = _vla_derive_cmd(np.asarray(wps_nav[_fi], np.float32),
                                                    float(v0s[_fi]) if v0s is not None and _fi < len(v0s) else 8.0)
                    raw["_cmd"] = _last_cmd
                # stack here, off the producer's critical path (~8 ms)
                if in_slots is not None:
                    # Zero-copy (2026-09-05): write CHW directly into a pinned slot.
                    # The producer hands the slot straight to rt.infer(), skipping the
                    # CPU copy (4.6 ms). The slot goes back to in_free after infer().
                    si = in_free.get()
                    imgs = in_slots[si]
                    for ci, c in enumerate(CAMS):
                        imgs[0, ci] = raw[c][:, :, ::-1].transpose(2, 0, 1)
                else:
                    si = None
                    imgs = np.ascontiguousarray(np.stack(
                        [raw[c][:, :, ::-1].transpose(2, 0, 1)
                         for c in CAMS])[None])
                v0 = float(v0s[f["frame"]]) if v0s is not None \
                    and f["frame"] < len(v0s) else 8.0
                po = tuple(float(x) for x in poses[f["frame"]]) \
                    if poses is not None and f["frame"] < len(poses) else None
                lb = None
                if _LIDAR:                       # METEOR_LIDAR=1: feed the pillar raster [4,400,250]
                    _lp = f.get("lidar_bev") or f"lidar_bev/{int(f['frame']):04d}.npz"
                    try:
                        lb = np.load(os.path.join(_sd, _lp))["lb"].astype(np.float32)[None]
                    except Exception:
                        lb = None
                try:
                    q_raw.put((raw, imgs, K, Tc, v0, po, si, lb), timeout=5)
                except queue.Full:
                    if stop.is_set():
                        return
        if not loop:
            break
    q_raw.put(None)


def producer(rt, q_raw, q, free_slots, stop, in_free=None):
    u8 = rt.host["imgs"].dtype == np.uint8
    while not stop.is_set():
        item = q_raw.get()
        if item is None:
            break
        raw, imgs, K, Tc, v0, pose, si, lb = item
        if not u8:
            imgs = imgs.astype(np.float32) / 255.0
        slot = free_slots.get()          # blocks until a consumer released
        t0 = time.time()
        out = rt.infer(imgs, K[0][None], Tc[0][None], v0=v0, pose=pose,
                       out_slot=slot, lidar_bev=lb)
        if lb is not None:
            out = dict(out); out["lidar_bev_in"] = lb    # so rendering can overlay the point cloud
        if si is not None:               # H2D already completed inside infer()
            in_free.put(si)
        dt = (time.time() - t0) * 1000
        # out holds VIEWS into host slot `slot`; the renderer releases it
        try:
            q.put((raw, K, Tc, v0, out, dt, pose, slot), timeout=5)
        except queue.Full:
            if stop.is_set():
                return
    q.put(None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--root", default="fast")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--display", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N frames (0 = all)")
    a = ap.parse_args()

    N_SLOTS = 4
    # occ is used for rendering, so copy it (2026-08-16). The rest is unused,
    # so skip the host transfer (the compute itself stays in the graph).
    # risk is rendered since 2026-08-28 (v128cR-line engines emit it);
    # its transfer is [1,1,400,250] fp32 = 400KB/frame, negligible.
    rt = MeteorRT(a.engine, skip_outputs=(
        "flow", "unk", "pl", "tl", "lg_pts", "lg_meta",
        "lg_adj"), n_out_slots=N_SLOTS)
    # 8-camera engines (r64 line): the model's camera order is the dataset's
    # CAMS, which appends CAM_BACK_NARROW after the 7 the renderer knows.
    # 2026-08-15: pass the same list to the renderer (R.CAMS). Without this
    # update the rendering side stays at 7 cameras and the 8th camera, which
    # is actually used as input, is displayed as "(blank)".
    global CAMS
    if rt.shapes.get("imgs", (1, 7))[1] == 8 and len(CAMS) == 7:
        CAMS = CAMS + ["CAM_BACK_NARROW"]
        R.CAMS = CAMS
        # 2026-08-25: tile rendering reads CAM_DRAW (introduced by the 8/24
        # rendering fix). Updating only R.CAMS leaves the 8th tile "(blank)"
        # even though 8 images are fed in (bit us in practice on the laptop).
        R.CAM_DRAW = list(CAMS)
        print("[rt] 8-camera engine: loading CAM_BACK_NARROW as camera 8")
    # Derive the BEV fore/aft extent from the engine's output rows (light
    # 600 rows = 40 m rear / baseline 800 rows = 80 m rear). With a fixed
    # value the 8-camera build is also displayed cut off at 40 m rear.
    _lane_shape = rt.shapes.get("lane")
    if _lane_shape is not None:
        R.set_bev_extent(_lane_shape[-2])
    free_slots = queue.Queue()
    for i in range(N_SLOTS):
        free_slots.put(i)
    from deploy.t4input import is_t4_scene
    _root = a.root.rstrip("/")
    if os.path.isfile(os.path.join(_root, "manifest.json")) or is_t4_scene(_root):
        # --root is the scene itself (for t4, annotation/sample.json sits directly
        # under it). Unless this check comes first, subfolders inside the scene
        # such as tmp/ are mistaken for the scene list and we crash (bit us on the laptop).
        a.root = os.path.dirname(_root) or "."
        scenes = [os.path.basename(_root)]
    else:
        scenes = sorted(
            s for s in os.listdir(a.root)
            if os.path.isfile(os.path.join(a.root, s, "manifest.json"))
            or is_t4_scene(os.path.join(a.root, s)))
    q = queue.Queue(maxsize=2)
    stop = threading.Event()
    q_raw = queue.Queue(maxsize=3)
    # zero-copy input slots: q_raw 3 + 1 in inference + 1 being written = 5
    _u8 = rt.host["imgs"].dtype == np.uint8
    in_slots = rt.pinned_input_slots(5) if _u8 else None
    in_free = queue.Queue()
    if in_slots is not None:
        for i in range(len(in_slots)):
            in_free.put(i)
        print("[rt] zero-copy input slots x5 (pinned)", flush=True)
    th_l = threading.Thread(target=loader,
                            args=(scenes, a.root, a.stride, q_raw, stop,
                                  a.loop, in_slots, in_free), daemon=True)
    th_i = threading.Thread(target=producer,
                            args=(rt, q_raw, q, free_slots, stop, in_free),
                            daemon=True)
    th_l.start()
    th_i.start()

    vw = None
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        # 2026-09-06: the output fps was fixed at 8, so the video played back at
        # 8 fps regardless of the run rate (user report). METEOR_REC_FPS (default 10 = ~effective rate).
        _rec_fps = float(os.environ.get("METEOR_REC_FPS", "10"))
        vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"),
                             _rec_fps, (1920, 1080))
    n = 0
    t_start = time.time()
    infer_ms, render_ms = [], []
    # Render pool: compose_frame costs ~85-90 ms and became the pipeline
    # bound once inference dropped to ~84 ms. numpy/cv2 release the GIL, so
    # two workers overlap to ~45 ms effective and inference is the limit
    # again. Frames are dispatched round-robin with sequence numbers and
    # re-ordered before the writer/display. compose_frame's mode-hysteresis
    # state (_prev_mode) is shared across workers: a worker can read a
    # one-frame-stale value, which weakens the smoothing for that frame but
    # cannot corrupt anything -- accepted for the throughput.
    done_q = queue.Queue()

    # producer puts un-numbered items; wrap with sequence numbers here
    q_seq = queue.Queue(maxsize=4)

    _vla_state = {"rec": None}

    def sequencer():
        i = 0
        while True:
            item2 = q.get()
            if item2 is None:
                q_seq.put(None)
                return
            if _VLA_LIVE is not None:
                raw2, out2, v02 = item2[0], item2[4], item2[3]
                if "bev_tok" in out2 and i % _VLA_EVERY == 0:
                    if hasattr(_VLA_LIVE, "submit"):      # worker pool: run ahead, the renderer waits per frame
                        raw2["_vla_future"] = _VLA_LIVE.submit(np.asarray(out2["bev_tok"])[0].copy(), float(v02),
                                                               raw2.get("_cmd", "keep_lane"))
                    else:
                        _t = time.time()
                        _vla_state["rec"] = _VLA_LIVE.infer(np.asarray(out2["bev_tok"])[0], float(v02),
                                                            raw2.get("_cmd", "keep_lane"))
                        if i % (_VLA_EVERY * 10) == 0:
                            print(f"[vla] frame {i}: {time.time() - _t:.1f}s  cmd={raw2.get('_cmd')}  "
                                  f"decision={(_vla_state['rec'].get('json') or {}).get('command')}", flush=True)
                if "_vla_future" not in raw2:
                    raw2["_vla"] = _vla_state["rec"]
            q_seq.put((i,) + item2)
            i += 1

    # rewire: sequencer reads q, workers read q_seq
    th_s = threading.Thread(target=sequencer, daemon=True)
    th_s.start()
    def render_worker_seq():
        while True:
            item2 = q_seq.get()
            if item2 is None:
                q_seq.put(None)
                done_q.put(None)
                return
            seq, raw, K, Tc, v0, out, dt, pose, slot = item2
            if "_vla_future" in raw:                     # pool result for this frame (ordered by construction)
                raw["_vla"] = raw.pop("_vla_future").result()
                _vla_state["rec"] = raw["_vla"]
                if _VLA_DUMP is not None and raw["_vla"] is not None:   # METEOR_VLA_DUMP: keep the live records (re-render without re-inference)
                    _VLA_DUMP.write(json.dumps({"scene": raw.get("_scene"), "frame": raw.get("_frame"), **raw["_vla"]}) + "\n"); _VLA_DUMP.flush()
                if seq % 100 == 0:
                    print(f"[vla] frame {seq}: decision={(raw['_vla'].get('json') or {}).get('command')}", flush=True)
            t0 = time.time()
            canvas = R.compose_frame(raw, K, Tc, v0, out, dt,
                                     fps_now=(seq / max(time.time()
                                                        - t_start, 1e-3)),
                                     pose=pose)
            free_slots.put(slot)         # slot data fully consumed
            done_q.put((seq, canvas, dt, (time.time() - t0) * 1000))

    workers = [threading.Thread(target=render_worker_seq, daemon=True)
               for _ in range(2)]
    for w in workers:
        w.start()

    pending = {}
    next_seq = 0
    ended = 0
    try:
        while ended < len(workers):
            item2 = done_q.get()
            if item2 is None:
                ended += 1
                continue
            seq, canvas, dt, rms = item2
            pending[seq] = (canvas, dt, rms)
            while next_seq in pending:
                canvas, dt, rms = pending.pop(next_seq)
                render_ms.append(rms)
                infer_ms.append(dt)
                n += 1
                next_seq += 1
                if vw is not None:
                    vw.write(canvas)
                if a.display:
                    # Fullscreen display (2026-08-28). To avoid the backend's
                    # letterboxing, resize the canvas itself to the screen
                    # resolution before showing it (borderless). The resolution is
                    # auto-detected, falling back to METEOR_SCREEN (e.g. 2560x1600).
                    # METEOR_FULLSCREEN=0 gives the classic windowed display.
                    if not hasattr(a, "_win"):
                        a._win = True
                        a._scr = None
                        cv2.namedWindow("METEOR Orin realtime",
                                        cv2.WINDOW_NORMAL
                                        | cv2.WINDOW_FREERATIO)
                        if os.environ.get("METEOR_FULLSCREEN", "1") != "0":
                            cv2.setWindowProperty(
                                "METEOR Orin realtime",
                                cv2.WND_PROP_FULLSCREEN,
                                cv2.WINDOW_FULLSCREEN)
                            _scr = os.environ.get("METEOR_SCREEN", "")
                            if "x" in _scr:
                                _w, _h = _scr.split("x")
                                a._scr = (int(_w), int(_h))
                            else:
                                try:
                                    import tkinter as _tk
                                    _r = _tk.Tk(); _r.withdraw()
                                    a._scr = (_r.winfo_screenwidth(),
                                              _r.winfo_screenheight())
                                    _r.destroy()
                                except Exception:
                                    a._scr = (2560, 1600)
                            # GNOME ignores WND_PROP_FULLSCREEN from an unfocused
                            # process (demo started over ssh): size and place the
                            # window explicitly so it fills the display anyway.
                            try:
                                cv2.resizeWindow("METEOR Orin realtime",
                                                 a._scr[0], a._scr[1])
                                cv2.moveWindow("METEOR Orin realtime", 0, 0)
                            except Exception:
                                pass
                    if a._scr is not None and                             (canvas.shape[1], canvas.shape[0]) != a._scr:
                        canvas = cv2.resize(canvas, a._scr,
                                            interpolation=cv2.INTER_LINEAR)
                    cv2.imshow("METEOR Orin realtime", canvas)
                    k = cv2.waitKey(1) & 0xFF
                    if k == ord("q"):
                        ended = len(workers)
                        break
                    if k == ord(" "):
                        cv2.waitKey(0)
                if a.limit and n >= a.limit:
                    ended = len(workers)
                    break
            if a.limit and n >= a.limit:
                break
    finally:
        stop.set()
        for qq in (q, q_seq):
            try:
                qq.get_nowait()
            except Exception:
                pass
        if vw is not None:
            vw.release()
    el = time.time() - t_start
    im = np.array(infer_ms[3:] if len(infer_ms) > 6 else infer_ms)
    rm = np.array(render_ms[3:] if len(render_ms) > 6 else render_ms)
    print(f"frames={n} wall={el:.1f}s -> {n / el:.1f} FPS  "
          f"(infer {im.mean():.0f} ms, render {rm.mean():.0f} ms, "
          f"pipelined)")


if __name__ == "__main__":
    main()

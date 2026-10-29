#!/usr/bin/env python3
"""METEOR-VLA worker process (2026-09-10). Runs the VLA (deploy/vla_live.VLALive) in its own Python
environment and answers JSON-lines requests over TCP: {"tok": <base64 fp16 [96,25,16]>, "v0": f, "cmd": s}
-> the overlay record. The demo runner (deploy/orin_realtime.py, METEOR_VLA_SERVER=host:port) sends the
engine's bev_tok output, so one METEOR forward feeds both the E2E plan and the VLA.

  HF_HOME=~/hf_cache python3 deploy/vla_server.py --ckpt out/vla_v17/ckpt_last.pt --port 5771
"""
import argparse, base64, json, os, socket, sys, threading
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deploy.vla_live import VLALive

ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--port", type=int, default=5771)
ap.add_argument("--reg-scale", type=float, default=2.0); ap.add_argument("--max-new", type=int, default=192)
a = ap.parse_args()
vla = VLALive(a.ckpt, reg_scale=a.reg_scale, max_new=a.max_new)
lock = threading.Lock()
srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind(("127.0.0.1", a.port)); srv.listen(4); print(f"[vla-server] listening on 127.0.0.1:{a.port}", flush=True)
while True:
    conn, _ = srv.accept()
    f = conn.makefile("rwb")
    try:
        for line in f:
            req = json.loads(line)
            tok = np.frombuffer(base64.b64decode(req["tok"]), np.float16).reshape(96, 25, 16)
            with lock:
                rec = vla.infer(tok, float(req["v0"]), req.get("cmd", "keep_lane"))
            f.write((json.dumps(rec) + "\n").encode()); f.flush()
    except Exception as e:
        print("[vla-server] connection ended:", e, flush=True)
    finally:
        conn.close()

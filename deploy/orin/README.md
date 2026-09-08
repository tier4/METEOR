# Orin-side operational scripts (mirror)

Where the canonical copies live:
- C++ runtime: `deploy/cpp/` (local copy is canonical; `rsync deploy/cpp/ <orin-user>@<orin-host>:~/meteor/cpp/`, then `cmake/make` on the Orin).
- Python runtime/rendering: `deploy/runtime.py`, `deploy/orin_realtime.py`, `deploy/orin_render.py`, `deploy/viz_np.py`, `deploy/orin_build_int8.py` (local copy is canonical, rsync'd to `~/meteor/deploy/` on the Orin).
- This directory: mirror of helper scripts written directly on the Orin (imported 2026-09-08).
  - `demo.sh` / `demo_cpp.sh`: demo launchers (default engine and rendering env).
  - `bench_rt.py` / `bench_rt_zc.py`: end-to-end latency including the runtime (zc = zero-copy input).
  - `bev_frozen_test.py` / `stat_probe.py`: INT8 checks (frozen ego, stationary-verdict distribution).
  - `v142c3Z_orin_job.sh`: job template the deployment chain derives via sed (fp16 -> real-calibration INT8 -> checks -> bench). `v157c3Z_orin_job.sh` / `v157Lc3Z_orin_job.sh` are derived examples (the latter has 6 inputs incl. LiDAR).
  - `prof_diff.py` / `sparse_profile_job.sh` / `record_lidar.sh`: per-layer profile diff and recording jobs.

# Orin 側の運用スクリプト (ミラー)

正本の所在:
- C++ ランタイム: `deploy/cpp/` (ローカルが正本。`rsync deploy/cpp/ <orin-user>@<orin-host>:~/meteor/cpp/` → Orin で `cmake/make`)。
- Python ランタイム/描画: `deploy/runtime.py`, `deploy/orin_realtime.py`, `deploy/orin_render.py`, `deploy/viz_np.py`, `deploy/orin_build_int8.py` (ローカルが正本、Orin `~/meteor/deploy/` へ rsync)。
- このディレクトリ: Orin 上で直接書いた補助スクリプトのミラー (2026-09-08 取り込み)。
  - `demo.sh` / `demo_cpp.sh`: デモ起動 (既定エンジン・描画 env)。
  - `bench_rt.py` / `bench_rt_zc.py`: ランタイム込みレイテンシ計測 (zc = ゼロコピー入力)。
  - `bev_frozen_test.py` / `stat_probe.py`: INT8 判定 (ego 凍結・停止判定分布)。
  - `v142c3Z_orin_job.sh`: 配備チェーンが sed で派生させるジョブ雛形 (fp16 → 実較正 INT8 → 判定 → bench)。`v157c3Z_orin_job.sh` / `v157Lc3Z_orin_job.sh` は派生例 (後者は LiDAR 6 入力)。
  - `prof_diff.py` / `sparse_profile_job.sh` / `record_lidar.sh`: 層別プロファイル差分・録画ジョブ。

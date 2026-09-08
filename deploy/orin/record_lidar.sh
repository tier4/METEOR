#!/bin/bash
# v157L (LiDAR 入力付き) の valday を LiDAR ON / OFF で録画 (C++)。Orin ジョブ完了を待つ。
cd ~/meteor; L=out/record_lidar.log; : > $L
until grep -q V142C3Z_DONE out/auto_v157Lc3Z.log 2>/dev/null; do sleep 60; done
export METEOR_PLUGIN_SO=/home/nvidia/meteor/liftbench/plugin/build/libmeteor_lift.so METEOR_CUDAGRAPH=1 METEOR_TH2D=0.30 METEOR_2D_HIDE=7 METEOR_SEG2D_OVERLAY=0 METEOR_DEPTH_PANEL=1 METEOR_REC_FPS=10
for m in 1 0; do
  echo "=== LiDAR=$m valday" >> $L
  METEOR_LIDAR=$m ./cpp/build/meteor_realtime --engine eng/v157Lc3Zg_int8.engine --root valday --out out/demo_cpp_v157L_valday_lidar${m}.mp4 2>&1 | grep -E "^frames=|LiDAR|Error|missing" >> $L
done
echo "=== LiDAR=1 valcurve" >> $L
METEOR_LIDAR=1 ./cpp/build/meteor_realtime --engine eng/v157Lc3Zg_int8.engine --root valcurve --out out/demo_cpp_v157L_valcurve_lidar1.mp4 2>&1 | grep -E "^frames=|Error" >> $L
echo RECORD_LIDAR_DONE >> $L

#!/bin/bash
# v157Lc3Z = v142 best_e2e + v3 較正 + --no-hist (履歴経路の焼き込み除去, 2026-09-05)。Orin 実装ジョブ。
cd ~/meteor; export METEOR_LIDAR=1
export METEOR_PLUGIN_SO=/home/nvidia/meteor/liftbench/plugin/build/libmeteor_lift.so
SO=$METEOR_PLUGIN_SO; T=/usr/src/tensorrt/bin/trtexec
L=out/auto_v157Lc3Z.log; : > $L
IFM="uint8:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw"   # no-hist: imgs,K,T,v0
# 18出力 (no-hist: raw_bev なし; risk が tl の後、lane_logit が末尾)
OFMR="uint8:chw,uint8:chw,uint8:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp16:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp16:chw,fp16:chw"   # no-hist: raw_bev なし + depth_mean (19 出力, 2026-09-07)
echo "=== $(date) v129cR fp16" >> $L
$T --onnx=out/meteor_v157Lc3Z_final.onnx --saveEngine=eng/v157Lc3Zf_fp16.engine \
   --staticPlugins=$SO --fp16 --sparsity=enable --memPoolSize=workspace:8192 \
   --inputIOFormats=$IFM --outputIOFormats=$OFMR >> $L 2>&1
echo "=== $(date) v129cR 素の INT8 (aux0)" >> $L
rm -f eng/v157Lc3Zg_int8.engine.calib
python3 deploy/orin_build_int8.py --onnx out/meteor_v157Lc3Z_final.onnx \
  --companion eng/v157Lc3Zf_fp16.engine --roots calib fast \
  --out eng/v157Lc3Zg_int8.engine --sparse --calib 96 --stride 4 --cams8 \
  --max-aux-streams 0 >> $L 2>&1
echo "=== $(date) 判定" >> $L
METEOR_CUDAGRAPH=1 python3 bev_frozen_test.py eng/v157Lc3Zg_int8.engine,eng/v157Lc3Zf_fp16.engine >> $L 2>&1
METEOR_CUDAGRAPH=1 python3 stat_probe.py eng/v157Lc3Zg_int8.engine >> $L 2>&1
echo "--- bench graph-ON" >> $L
METEOR_CUDAGRAPH=1 METEOR_OUT_SLOTS=2 python3 bench_rt.py eng/v157Lc3Zg_int8.engine 2>&1 | grep 中央値 >> $L
echo V142C3Z_DONE >> $L

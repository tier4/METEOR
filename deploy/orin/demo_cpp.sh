#!/bin/bash
# C++ runtime demo (2026-09-06). Same defaults as demo.sh (valday, TH2D 0.30, panel omitted, 10 fps recording).
#   ./demo_cpp.sh [ROOT=valday] [LIMIT=0] [OUT=out/demo_cpp_<root>.mp4]
cd ~/meteor
ROOT=${1:-valday}; LIMIT=${2:-0}; OUT=${3:-out/demo_cpp_${ROOT}.mp4}
export METEOR_PLUGIN_SO=/home/nvidia/meteor/liftbench/plugin/build/libmeteor_lift.so
export METEOR_CUDAGRAPH=1 METEOR_TH2D=${METEOR_TH2D:-0.30} METEOR_2D_HIDE=${METEOR_2D_HIDE:-7}
export METEOR_SEG2D_OVERLAY=${METEOR_SEG2D_OVERLAY:-0} METEOR_DEPTH_PANEL=${METEOR_DEPTH_PANEL:-1} METEOR_REC_FPS=${METEOR_REC_FPS:-10}
for cand in eng/v151c3Zg_int8.engine eng/v147c3Zg_int8.engine eng/v145c3Zg_int8.engine eng/v142c3Zg_int8.engine; do [ -s "$cand" ] && { ENGINE="$cand"; break; }; done
echo "[demo_cpp] engine=$ENGINE root=$ROOT limit=$LIMIT out=$OUT"
exec ./cpp/build/meteor_realtime --engine "$ENGINE" --root "$ROOT" --limit "$LIMIT" --out "$OUT"

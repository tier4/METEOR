#!/bin/bash
# METEOR — Orin on-device demo
#
#   ./demo.sh                        real-time display in a GUI (recommended)
#   ./demo.sh --record               write to a video file (no on-screen display)
#   ./demo.sh --root calib           use a different dataset
#   ./demo.sh --engine eng/xxx.engine   use a different engine
#   ./demo.sh --list                 list available engines and data
#
# Press q to quit while displaying.
#
# Why orin_realtime.py: orin_render.py runs inference and rendering sequentially,
# so one frame = inference + rendering, which drops to ~3.5 FPS effective in the GUI.
# orin_realtime.py runs inference in a producer thread overlapped with rendering,
# so it runs at max(inference, rendering). Both draw via deploy/viz_np.py (the same
# functions as the PyTorch demo), so the look matches the local visualization.
set -u
cd "$(dirname "$0")"

ENGINE=""
# 2026-08-31: for calibrated engines the 2D threshold should be 0.42 (0.5 costs recall -17%/-20%)
export METEOR_TH2D=${METEOR_TH2D:-0.30}   # 2026-09-06: GT (h>=24px) sweep peaks F1 at 0.25; default 0.30 keeps P>=0.85 (0.42 gives recall 0.44, too low)
export METEOR_ZERO_HIST=${METEOR_ZERO_HIST:-1}   # 2026-09-04: history forced to zero (matches training; lift for the v144 real-history generation)
export METEOR_SEG2D_OVERLAY=${METEOR_SEG2D_OVERLAY:-0}   # 2026-09-05: no seg overlay on the 2D tiles (user request)
export METEOR_OCC_DS=${METEOR_OCC_DS:-2}   # 2026-09-05: OCC voxels subsampled 1/2 (rendering 335ms -> ~110ms)
export METEOR_OCC_EVERY=${METEOR_OCC_EVERY:-6}   # OCC panel refreshed once every 6 frames
export METEOR_OCC_PANEL=${METEOR_OCC_PANEL:-0}   # 2026-09-05: OCC voxel panel omitted (10+ FPS requested; =1 shows it)
export METEOR_2D_HIDE=${METEOR_2D_HIDE:-7}   # 2026-09-06: do not draw boxes for 2D class 7 (road paint)
ROOT=valday   # 2026-09-06: default is the 5 daytime curve scenes from val (valcurve is the night-time expressway; Okinawa is not used)
MODE=display
STRIDE=1
LOOP="--loop"
LIMIT=""

while [ $# -gt 0 ]; do
  case "$1" in
    --record)  MODE=record; LOOP="";      shift ;;
    --engine)  ENGINE="$2";               shift 2 ;;
    --root)    ROOT="$2";                 shift 2 ;;
    --stride)  STRIDE="$2";               shift 2 ;;
    --limit)   LIMIT="--limit $2";        shift 2 ;;
    --no-loop) LOOP="";                   shift ;;
    --list)    MODE=list;                 shift ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1  (--help for usage)"; exit 1 ;;
  esac
done

export METEOR_PLUGIN_SO=/home/nvidia/meteor/liftbench/plugin/build/libmeteor_lift.so
[ -f "$METEOR_PLUGIN_SO" ] || { echo "lift plugin missing: $METEOR_PLUGIN_SO"; exit 1; }

if [ "$MODE" = list ]; then
  echo "== available engines (0-byte files are failed builds, excluded) =="
  for e in eng/*.engine; do
    [ -s "$e" ] || continue
    printf "  %-34s %5s MB\n" "$e" "$(( $(stat -c %s "$e") / 1048576 ))"
  done
  echo
  echo "== datasets =="
  for r in sample calib fast; do
    [ -d "$r" ] && printf "  %-8s %d scenes\n" "$r" "$(ls "$r" | wc -l)"
  done
  echo
  echo "  note: fast is a 7-camera rig. With an 8-camera engine the 8th input is"
  echo "        zero-filled, so image quality drops. sample / calib are 8-camera."
  exit 0
fi

# If no engine is given, look for ones verified healthy, newest first.
# v106sf = v106 + healthy ego + stationary-verdict fix; v106fk = stationary verdict unfixed.
if [ -z "$ENGINE" ]; then
  for cand in eng/v151c3Zg_int8.engine eng/v147c3Zg_int8.engine eng/v145c3Zg_int8.engine eng/v142c3Zg_int8.engine eng/v142c3Rg_int8.engine eng/v141c3Rg_int8.engine eng/v132c3Rg_int8.engine eng/v135c3Rg_int8.engine eng/v130c3Rg_int8.engine eng/v129c3Rg_int8.engine \
              eng/v128g_int8.engine; do
    [ -s "$cand" ] && { ENGINE="$cand"; break; }
  done
fi
[ -n "$ENGINE" ] && [ -s "$ENGINE" ] || { echo "engine not found (check with --list)"; exit 1; }
[ -d "$ROOT" ] || { echo "data missing: $ROOT  (check with --list)"; exit 1; }

# CUDA Graph is only safe with engines built with aux-streams 0 (*g_int8)
case "$ENGINE" in *g_int8*) export METEOR_CUDAGRAPH=1 ;; esac
echo "engine  : $ENGINE"
echo "data    : $ROOT ($(ls "$ROOT" | wc -l) scenes, stride $STRIDE)"

if [ "$MODE" = display ]; then
  # Pick the X display. When run directly from a GUI session DISPLAY is
  # already set. Over ssh, pick it up from /tmp/.X11-unix.
  if [ -z "${DISPLAY:-}" ]; then
    for x in /tmp/.X11-unix/X*; do
      [ -e "$x" ] && { export DISPLAY=":${x##*/X}"; break; }
    done
  fi
  [ -n "${DISPLAY:-}" ] || { echo "no X display found. Run from a GUI session or set DISPLAY=:0"; exit 1; }
  echo "display : DISPLAY=$DISPLAY   (press q to quit)"
  echo
  exec python3 deploy/orin_realtime.py --engine "$ENGINE" --root "$ROOT" \
       --stride "$STRIDE" --display $LOOP $LIMIT
else
  OUT="out/demo_$(basename "${ENGINE%.engine}")_${ROOT}.mp4"
  echo "output  : $OUT"
  echo
  python3 deploy/orin_realtime.py --engine "$ENGINE" --root "$ROOT" \
       --stride "$STRIDE" --out "$OUT" $LIMIT
  ls -lh "$OUT" 2>/dev/null
fi

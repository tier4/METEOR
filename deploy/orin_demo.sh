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
ROOT=sample
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
  for cand in eng/v115_int8.engine eng/v106sf_int8.engine \
              eng/v106fk_int8.engine eng/v103fk_int8.engine; do
    [ -s "$cand" ] && { ENGINE="$cand"; break; }
  done
fi
[ -n "$ENGINE" ] && [ -s "$ENGINE" ] || { echo "engine not found (check with --list)"; exit 1; }
[ -d "$ROOT" ] || { echo "data missing: $ROOT  (check with --list)"; exit 1; }

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

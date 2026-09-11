#!/bin/bash
# Endless on-device demo over the six published scenes (the Hugging Face demo set):
# highway -> mountain -> arterial -> city -> night expressway -> dusk expressway, then repeat.
#
#   ./demo_public6.sh                 show on the Orin display, loop forever, auto-restart on error
#   ./demo_public6.sh --record        write one pass to out/demo_public6_<engine>.mp4 instead
#   ./demo_public6.sh --engine ENG    use another engine (default: newest sparse INT8 engine)
#   ./demo_public6.sh --stop          stop a running demo
#   ./demo_public6.sh --autostart     install a systemd user service that starts the demo at login
#
# Data: ~/meteor/public6/<scene>/{manifest.json,img/,ego_motion.npz,lidar_bev/} + scenes.txt
#       (anonymised; identical to AutowareFoundation/meteor-demo-scenes). Rendering: Python
#       runtime deploy/orin_realtime.py, same picture as the C++ runtime.
cd ~/meteor
ROOT=public6
MODE=display
ENGINE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --record) MODE=record; shift ;;
    --engine) ENGINE="$2"; shift 2 ;;
    --root)   ROOT="$2"; shift 2 ;;
    --stop)   # kill the wrapper loops first (not this invocation), then the runtime
              for p in $(pgrep -f "demo_public6.s[h]"); do [ "$p" != "$$" ] && kill "$p" 2>/dev/null; done
              pkill -f "orin_realtime.py --engine .* --displa[y] --loop"; echo "stopped"; exit 0 ;;
    --autostart)
      mkdir -p ~/.config/systemd/user
      cat > ~/.config/systemd/user/meteor-demo.service <<EOF
[Unit]
Description=METEOR endless demo on the six public scenes
After=graphical-session.target
[Service]
Environment=DISPLAY=:1
ExecStart=/bin/bash /home/nvidia/meteor/demo_public6.sh --run
Restart=always
RestartSec=5
[Install]
WantedBy=default.target
EOF
      systemctl --user daemon-reload && systemctl --user enable --now meteor-demo.service
      echo "installed: systemctl --user status meteor-demo.service   (disable: systemctl --user disable --now meteor-demo.service)"
      exit 0 ;;
    --run)    MODE=run; shift ;;
    *) echo "unknown option $1"; exit 1 ;;
  esac
done

[ -d "$ROOT" ] && [ -s "$ROOT/scenes.txt" ] || { echo "data missing: ~/meteor/$ROOT (rsync the public scenes first)"; exit 1; }
if [ -z "$ENGINE" ]; then
  for cand in eng/v157c3Zg_int8.engine eng/v151c3Zg_int8.engine eng/v147c3Zg_int8.engine; do
    [ -s "$cand" ] && { ENGINE="$cand"; break; }
  done
fi
[ -s "$ENGINE" ] || { echo "engine missing: $ENGINE"; exit 1; }

# Same rendering defaults as demo.sh: no 2D seg overlay, no OCC panel, road-paint boxes hidden,
# zero history for engines that still carry history inputs, CUDA Graph on, lift plugin.
export METEOR_PLUGIN_SO=/home/nvidia/meteor/liftbench/plugin/build/libmeteor_lift.so
export METEOR_CUDAGRAPH=1 METEOR_ZERO_HIST=${METEOR_ZERO_HIST:-1}
export METEOR_TH2D=${METEOR_TH2D:-0.30} METEOR_2D_HIDE=${METEOR_2D_HIDE:-7}
export METEOR_SEG2D_OVERLAY=${METEOR_SEG2D_OVERLAY:-0} METEOR_OCC_PANEL=${METEOR_OCC_PANEL:-0}
export METEOR_GPU_NAME=${METEOR_GPU_NAME:-"AGX Orin"}

echo "[demo_public6] engine=$ENGINE root=$ROOT ($(wc -l < $ROOT/scenes.txt) scenes) mode=$MODE"
if [ "$MODE" = record ]; then
  OUT="out/demo_public6_$(basename "${ENGINE%.engine}").mp4"
  exec python3 deploy/orin_realtime.py --engine "$ENGINE" --root "$ROOT" --out "$OUT"
fi

# display: find the X display when started over ssh
if [ -z "${DISPLAY:-}" ]; then
  for x in /tmp/.X11-unix/X*; do [ -e "$x" ] && { export DISPLAY=":${x##*/X}"; break; }; done
fi
[ -n "${DISPLAY:-}" ] || { echo "no X display; run from the GUI session or set DISPLAY=:1"; exit 1; }
echo "[demo_public6] DISPLAY=$DISPLAY  (q in the window stops one pass; this script restarts it — use --stop to end)"
mkdir -p log
while true; do
  python3 deploy/orin_realtime.py --engine "$ENGINE" --root "$ROOT" --display --loop \
      >> log/demo_public6.log 2>&1
  echo "[demo_public6] runtime exited ($?) $(date) — restarting in 3 s" | tee -a log/demo_public6.log
  sleep 3
done

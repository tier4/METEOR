#!/bin/bash
# METEOR — Orin 実機デモ
#
#   ./demo.sh                        GUI にリアルタイム表示 (推奨)
#   ./demo.sh --record               動画に書き出す (画面表示なし)
#   ./demo.sh --root calib           別のデータセットを使う
#   ./demo.sh --engine eng/xxx.engine   別のエンジンを使う
#   ./demo.sh --list                 使えるエンジンとデータを一覧表示
#
# 表示中は q キーで終了。
#
# なぜ orin_realtime.py なのか: orin_render.py は推論と描画を逐次実行する
# ので 1 フレーム = 推論 + 描画 になり、GUI では実効 3.5 FPS 程度まで落ちる。
# orin_realtime.py は生産者スレッドが推論を回して描画と重ねるので
# max(推論, 描画) で回る。描画コードは両者とも deploy/viz_np.py (PyTorch の
# デモと同一の関数) を使っており、見た目はローカルの可視化と揃えてある。
set -u
cd "$(dirname "$0")"

ENGINE=""
# 2026-08-31: 較正エンジンの 2D 閾値は 0.42 が正 (0.5 は recall -17%/-20%)
export METEOR_TH2D=${METEOR_TH2D:-0.30}   # 2026-09-06: GT(h>=24px) 掃引で F1 最大は 0.25、P>=0.85 を保つ 0.30 を既定 (0.42 は recall 0.44 と低すぎ)
export METEOR_ZERO_HIST=${METEOR_ZERO_HIST:-1}   # 2026-09-04: 履歴零固定 (学習条件に一致; v144 実履歴世代で解除)
export METEOR_SEG2D_OVERLAY=${METEOR_SEG2D_OVERLAY:-0}   # 2026-09-05: 2D タイルへの seg 重畳なし (ユーザー指示)
export METEOR_OCC_DS=${METEOR_OCC_DS:-2}   # 2026-09-05: OCC ボクセル 1/2 間引き (描画 335ms→約110ms)
export METEOR_OCC_EVERY=${METEOR_OCC_EVERY:-6}   # OCC パネル更新は 6 フレームに 1 回
export METEOR_OCC_PANEL=${METEOR_OCC_PANEL:-0}   # 2026-09-05: OCC ボクセルパネル省略 (10 FPS 以上の指示; =1 で表示)
export METEOR_2D_HIDE=${METEOR_2D_HIDE:-7}   # 2026-09-06: 2D クラス 7 (路面ペイント) の枠は描かない
ROOT=valday   # 2026-09-06: 既定は val の昼間カーブ 5 シーン (valcurve は夜の首都高; 沖縄は使わない)
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
    *) echo "不明な引数: $1  (--help で使い方)"; exit 1 ;;
  esac
done

export METEOR_PLUGIN_SO=/home/nvidia/meteor/liftbench/plugin/build/libmeteor_lift.so
[ -f "$METEOR_PLUGIN_SO" ] || { echo "リフトプラグインが無い: $METEOR_PLUGIN_SO"; exit 1; }

if [ "$MODE" = list ]; then
  echo "== 使えるエンジン (0 バイトはビルド失敗なので除外) =="
  for e in eng/*.engine; do
    [ -s "$e" ] || continue
    printf "  %-34s %5s MB\n" "$e" "$(( $(stat -c %s "$e") / 1048576 ))"
  done
  echo
  echo "== データセット =="
  for r in sample calib fast; do
    [ -d "$r" ] && printf "  %-8s %d シーン\n" "$r" "$(ls "$r" | wc -l)"
  done
  echo
  echo "  注: fast は 7 カメラのリグ。8 カメラのエンジンでは 8 本目を"
  echo "      ゼロ埋めするため画質が落ちる。sample / calib は 8 カメラ。"
  exit 0
fi

# エンジン未指定なら、健全性を確認済みのものを新しい順に探す。
# v106sf = v106 + ego 健全 + 停止判定の修正、v106fk = 停止判定が未修正。
if [ -z "$ENGINE" ]; then
  for cand in eng/v151c3Zg_int8.engine eng/v147c3Zg_int8.engine eng/v145c3Zg_int8.engine eng/v142c3Zg_int8.engine eng/v142c3Rg_int8.engine eng/v141c3Rg_int8.engine eng/v132c3Rg_int8.engine eng/v135c3Rg_int8.engine eng/v130c3Rg_int8.engine eng/v129c3Rg_int8.engine \
              eng/v128g_int8.engine; do
    [ -s "$cand" ] && { ENGINE="$cand"; break; }
  done
fi
[ -n "$ENGINE" ] && [ -s "$ENGINE" ] || { echo "エンジンが見つからない (--list で確認)"; exit 1; }
[ -d "$ROOT" ] || { echo "データが無い: $ROOT  (--list で確認)"; exit 1; }

# CUDA Graph は aux-streams 0 でビルドしたエンジン (*g_int8) のみ安全
case "$ENGINE" in *g_int8*) export METEOR_CUDAGRAPH=1 ;; esac
echo "エンジン : $ENGINE"
echo "データ   : $ROOT ($(ls "$ROOT" | wc -l) シーン, stride $STRIDE)"

if [ "$MODE" = display ]; then
  # X の表示先を決める。GUI セッションから直接実行するなら DISPLAY は
  # 既に入っている。ssh 経由なら /tmp/.X11-unix から拾う。
  if [ -z "${DISPLAY:-}" ]; then
    for x in /tmp/.X11-unix/X*; do
      [ -e "$x" ] && { export DISPLAY=":${x##*/X}"; break; }
    done
  fi
  [ -n "${DISPLAY:-}" ] || { echo "X ディスプレイが見つからない。GUI セッションから実行するか DISPLAY=:0 を指定"; exit 1; }
  echo "表示先   : DISPLAY=$DISPLAY   (q キーで終了)"
  echo
  exec python3 deploy/orin_realtime.py --engine "$ENGINE" --root "$ROOT" \
       --stride "$STRIDE" --display $LOOP $LIMIT
else
  OUT="out/demo_$(basename "${ENGINE%.engine}")_${ROOT}.mp4"
  echo "書き出し : $OUT"
  echo
  python3 deploy/orin_realtime.py --engine "$ENGINE" --root "$ROOT" \
       --stride "$STRIDE" --out "$OUT" $LIMIT
  ls -lh "$OUT" 2>/dev/null
fi

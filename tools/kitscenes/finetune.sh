#!/bin/bash
# Fine-tune a METEOR2 (8-camera) checkpoint on the KITScenes train split. See docs/KITSCENES.md.
#
#   usage: INIT=<METEOR2 checkpoint> VARIANT=full|nomap|camonly [GPUS=0,1,2,3] [EP=12] [BOXES=1] \
#          HR=<hr image root> bash tools/kitscenes/finetune.sh
#
#   VARIANT=full     images + ego speed + ego history (past 6.4 s) + HD map + lane-level route  (--route-map 1)
#   VARIANT=nomap    as full, but the HD-map channels are empty: route + ego history only      (METEOR_RMAP_NOMAP=1)
#   VARIANT=camonly  images + ego speed only (the plain METEOR2 interface)
#   BOXES=1          also train 3D boxes on the KITScenes auto-labels (tools/kitscenes/kits_box_gt.py, METEOR_KITS_BOX=1)
#   ROUTE_DENY=<f>   scenes treated as having no valid route (default tools/kitscenes/route_unmatched_scenes.txt;
#                    ROUTE_DENY= disables it)
#   KD_BOX_CKPT=<f>  optional far-box distillation teacher (METEOR v52-line checkpoint)
#
# Training uses KITScenes train only; validation during training uses the first 60 val scenes. Six cameras: the
# converter writes the 6 ring cameras plus a 30 deg crop of the front-centre camera as CAM_FRONT_NARROW, and
# METEOR_ABSENT_CAMS=6 feeds a zero image in that slot, so the network sees the 6 ring cameras.
# Report a fixed epoch decided in advance (the final epoch), not the best one on val.
set -u
cd "$(dirname "$0")/../.."
INIT=${INIT:?METEOR2 checkpoint to start from}; VARIANT=${VARIANT:-full}; GPUS=${GPUS:-0,1,2,3}
NP=$(echo "$GPUS" | tr ',' '\n' | grep -c .); EP=${EP:-12}; ROUND=${ROUND:-kits_$VARIANT}
ROOT=${ROOT:-out/bevlane}; HR=${HR:?root of the 1152x648 images written by the converter}; PY=${PYTHON:-python3}
K=out/kitscenes; OUTD=${OUTD:-out/kits_ft}; mkdir -p "$OUTD"
sort -u $K/scenes_train.txt > "$OUTD/train.txt"
head -60 $K/scenes_val.txt > "$OUTD/val60.txt"

export CUDA_VISIBLE_DEVICES=$GPUS METEOR_ABSENT_CAMS=6 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
COND=""
case "$VARIANT" in
  full)    export METEOR_RMAP=1 METEOR_RMAP_ROUTE=lanelet METEOR_RMAP_HIST=past64; COND="--route-map 1 --route-map-lr-mult 20" ;;
  nomap)   export METEOR_RMAP=1 METEOR_RMAP_ROUTE=lanelet METEOR_RMAP_HIST=past64 METEOR_RMAP_NOMAP=1; COND="--route-map 1 --route-map-lr-mult 20" ;;
  camonly) ;;
  *) echo "unknown VARIANT=$VARIANT (full | nomap | camonly)"; exit 1 ;;
esac
ROUTE_DENY=${ROUTE_DENY-tools/kitscenes/route_unmatched_scenes.txt}
[ -n "$ROUTE_DENY" ] && export METEOR_RMAP_ROUTE_DENY=$ROUTE_DENY
[ "${BOXES:-0}" = 1 ] && export METEOR_KITS_BOX=1
KD=""; [ -n "${KD_BOX_CKPT:-}" ] && KD="--kd-box-ckpt $KD_BOX_CKPT --kd-box-w 2.0 --kd-box-rmin 0"

# the METEOR2 648p recipe (must match the architecture of INIT) + the KITScenes fine-tune levers
ARCH="--depth-bins 64 --feat-stride 4 --gipm --head-v2 1 --det-head 96 --img-h 648 --img-w 1152 --paint-seg 2,3,4,5,6,7,8,13 \
  --lift-lr 1 --lr-rect 1 --lr-rows-only 1 --lr-blk-m 10 --lr-rank 128 --glr-blk 25 --glr-rank 128 --glr-half 1 --lr-geo-init 1 --yaw2 1"
LOSS="--ema 0.999 --box-far 1 --det2d-w 0 --seg2d-w 0 --ego-gscale 0.1 --seg2d-hr 1 --seg-w 4 --depth-w 3 --ego-w 3.0 --ego-ce-tau 0.3 \
  --lovasz-w 0.5 --tversky-w 0.5 --tversky-ab 0.6,0.4 --seg-far-w 2 --seg-far-x0 15 --seg-far-x1 40 --ego-ce-mult 2.5 --eps-wta 0.05"
DATA="--hr-root $HR --img-scale 1.5 --depth-key depth_hr --workers 8 --lidar-drop 0.3 --cam-drop 0.25 --yaw-fix-deg 0.7 --limit-train 60000"

echo "[kits-ft] $ROUND variant=$VARIANT init=$INIT gpus=$GPUS ep=$EP train=$(wc -l < "$OUTD/train.txt") val=60 $(date)"
"$PY" -m torch.distributed.run --nproc_per_node="$NP" --master_port="${PORT:-29890}" bevlane/train_meteorite.py \
  --root "$ROOT" --train-list "$OUTD/train.txt" --val-list "$OUTD/val60.txt" --batch "${BATCH:-2}" --epochs "$EP" --lr "${LR:-1.5e-5}" \
  $ARCH $LOSS $DATA $KD $COND --init-ckpt "$INIT" --out "$OUTD/ckpt_$ROUND"

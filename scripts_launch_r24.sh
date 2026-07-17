#!/bin/bash
# r24: v31 = v30 + optional LiDAR depth input (roadmap C6a).
# One set of weights serves camera-only AND LiDAR-assisted inference:
# --lidar-drop 0.5 hides the LiDAR input on half the train samples, the
# probe prints both modes ("mIoU" = camera-only, "+lidar mIoU").
# Waits for r23 to finish, inits from its last.pt (v31 only adds lid_alpha).
B=/home/umedan/work/BevLane
PY=/home/umedan/comet_venv/bin/python3
TR=/home/umedan/comet_venv/bin/torchrun
until [ -f $B/out/bevlane_ckpt_r23/last.pt ]; do sleep 300; done
while pgrep -f "out/bevlane_ckpt_r23" > /dev/null; do sleep 120; done
sleep 60
cd $B
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup $TR --nproc_per_node=8 \
  $B/bevlane/train.py --model v31 --batch 2 --epochs 8 --workers 0 \
  --gt-key gt_vec --root $B/out/bevlane \
  --train-list $B/out/round22_scenes.txt --limit-train 46000 \
  --train-bg --aug --lr 5e-5 --val-every 500 --seed-subset 24 \
  --turn-oversample 3.0 --lidar-drop 0.5 \
  --init-ckpt $B/out/bevlane_ckpt_r23/last.pt \
  --seg-w 1.0 --dice-w .5 --lovasz-w .5 --boundary-w 3 --tversky-w .6 \
  --far-w 1 --depth-w 0.6 --seg2d-w 0.35 --box-w 1.2 --bbox2d-w 0.25 \
  --ego-w 0.8 --occ-w 0.4 --traj-w 0.5 --tl-w 0.6 --risk-w 0.3 \
  --lanegraph-w 0.5 --flow-w 0.3 --unk-w 0.5 \
  --seg2d-key seg2d21 --n-seg2d 21 \
  --out $B/out/bevlane_ckpt_r24 > $B/out/train_r24_log.txt 2>&1 &
echo "r24 launched $(date)" >> $B/out/r24_launcher.log

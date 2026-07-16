#!/bin/bash
# Auto-launch r21 when r20 finishes: keeps r20's record BEV weights, but
# re-diversifies the collapsed K=3 modes and trains them with eps-WTA.
# Also refreshes the scene list with whatever the GT catch-up has landed.
B=/home/umedan/work/BevLane
PY=/home/umedan/comet_venv/bin/python3
TR=/home/umedan/comet_venv/bin/torchrun
while pgrep -f "train.py --model v29" > /dev/null; do sleep 120; done
sleep 90

$PY $B/deploy/expand_v29_init.py $B/out/bevlane_ckpt_r20/last.pt \
    $B/out/bevlane_ckpt_r20/last_r21init.pt

$PY - <<'PYEOF'
import json, os
B="/home/umedan/work/BevLane"; root=f"{B}/out/bevlane"
bad=set(open(f"{B}/out/indoor_scenes.txt").read().split())
keep=[]
for s in sorted(os.listdir(root)):
    if "2026-01-23T15-26-01" in s or s in bad: continue
    mf=f"{root}/{s}/manifest.json"
    if not os.path.exists(mf): continue
    try: m=json.load(open(mf))
    except Exception: continue
    fs=m.get("frames",[])
    if fs and any("agent_traj" in f for f in fs): keep.append(s)
open(f"{B}/out/round21_scenes.txt","w").write("\n".join(keep))
print(len(keep), "scenes for r21")
PYEOF

cd $B
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup $TR --nproc_per_node=8 \
  $B/bevlane/train.py --model v29 --batch 2 --epochs 8 --workers 0 \
  --gt-key gt_vec --root $B/out/bevlane \
  --train-list $B/out/round21_scenes.txt --limit-train 46000 \
  --train-bg --aug --lr 5e-5 --val-every 500 --seed-subset 21 \
  --init-ckpt $B/out/bevlane_ckpt_r20/last_r21init.pt \
  --seg-w 1.0 --dice-w .5 --lovasz-w .5 --boundary-w 3 --tversky-w .6 \
  --far-w 1 --depth-w 0.6 --seg2d-w 0.35 --box-w 1.2 --bbox2d-w 0.25 \
  --ego-w 0.8 --occ-w 0.4 --traj-w 0.5 --tl-w 0.3 --risk-w 0.3 \
  --lanegraph-w 0.5 --flow-w 0.3 \
  --seg2d-key seg2d21 --n-seg2d 21 \
  --out $B/out/bevlane_ckpt_r21 > $B/out/train_r21_log.txt 2>&1 &
echo "r21 launched $(date)" >> $B/out/r21_launcher.log

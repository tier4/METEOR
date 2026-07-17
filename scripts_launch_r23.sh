#!/bin/bash
# r23: turn oversampling x3, mode-CE 0.6, shadow-filtered OCC GT.
# Waits for r22, then waits for the occ shadow filter to finish.
B=/home/umedan/work/BevLane
PY=/home/umedan/comet_venv/bin/python3
TR=/home/umedan/comet_venv/bin/torchrun
while pgrep -f "ckpt_r22 " > /dev/null || pgrep -f "bevlane_ckpt_r22$" > /dev/null || pgrep -f "out/bevlane_ckpt_r22" > /dev/null; do sleep 120; done
until grep -q 'UNKNOWN GT DONE' $B/out/unknown_extract_log.txt 2>/dev/null; do sleep 300; done
sleep 60
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
open(f"{B}/out/round22_scenes.txt","w").write("\n".join(keep))
print(len(keep),"scenes for r23")
PYEOF
# graft r22 -> v30: traj_stem now takes [fused(96) | motion residual(96)];
# copy the trained 96 input channels, zero the residual ones so the initial
# forward is bit-equal to r22 (same trick as the r20 tfuse graft)
$PY - <<'PYEOF'
import torch
B="/home/umedan/work/BevLane"
ck=torch.load(f"{B}/out/bevlane_ckpt_r22/last.pt",map_location="cpu")
sd=ck["model"]
k="traj_stem.0.weight"
k=k if k in sd else "module."+k
w=sd[k]
assert w.shape[1]==96, w.shape
w2=torch.zeros(w.shape[0],192,*w.shape[2:],dtype=w.dtype)
w2[:,:96]=w
sd[k]=w2
torch.save({"model":sd},f"{B}/out/bevlane_ckpt_r22/last_v30init.pt")
print("grafted traj_stem 96->192",w2.shape)
PYEOF
cd $B
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup $TR --nproc_per_node=8 \
  $B/bevlane/train.py --model v30 --batch 2 --epochs 8 --workers 0 \
  --gt-key gt_vec --root $B/out/bevlane \
  --train-list $B/out/round22_scenes.txt --limit-train 46000 \
  --train-bg --aug --lr 5e-5 --val-every 500 --seed-subset 23 \
  --turn-oversample 3.0 \
  --init-ckpt $B/out/bevlane_ckpt_r22/last_v30init.pt \
  --seg-w 1.0 --dice-w .5 --lovasz-w .5 --boundary-w 3 --tversky-w .6 \
  --far-w 1 --depth-w 0.6 --seg2d-w 0.35 --box-w 1.2 --bbox2d-w 0.25 \
  --ego-w 0.8 --occ-w 0.4 --traj-w 0.5 --tl-w 0.6 --risk-w 0.3 \
  --lanegraph-w 0.5 --flow-w 0.3 --unk-w 0.5 \
  --seg2d-key seg2d21 --n-seg2d 21 \
  --out $B/out/bevlane_ckpt_r23 > $B/out/train_r23_log.txt 2>&1 &
echo "r23 launched $(date)" >> $B/out/r23_launcher.log

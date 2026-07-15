#!/bin/bash
# Catch-up: run the missing GT stages on already-converted scenes, in
# dependency order, with high parallelism. Guards /home free space.
B=/home/umedan/work/BevLane
PY=/home/umedan/comet_venv/bin/python3
export BEVLANE_ROOT=$B/out/allroot
cd $B
W=${1:-40}
MIN_FREE_GB=25

guard () {
  free=$(df -BG --output=avail /home | tail -1 | tr -dc '0-9')
  if [ "$free" -lt "$MIN_FREE_GB" ]; then
    echo "[STOP] /home free ${free}G < ${MIN_FREE_GB}G — halting catch-up" >&2
    exit 1
  fi
}

stage () {           # stage <name> <script> <list-file>
  guard
  n=$(wc -l < "$3")
  [ "$n" -eq 0 ] && { echo "== $1: nothing to do"; return; }
  echo "== $1: $n scenes, $W workers, $(date)"
  $PY bevlane/$2 --workers $W --scenes "$3" >> out/catchup_$1.log 2>&1
  echo "== $1 done $(date)"
}

# dependency order: seg2d21 -> occ; boxes -> agent_traj -> risk
$PY - <<'PYEOF'
import json, os
root='out/bevlane'
need={k:[] for k in ('seg2d','bbox2d','occ','agenttraj','tl','risk','lanegraph')}
for s in sorted(os.listdir(root)):
    mf=f"{root}/{s}/manifest.json"
    if not os.path.exists(mf): continue
    try: m=json.load(open(mf))
    except Exception: continue
    fs=m.get('frames',[])
    if not fs: continue
    f0=fs[len(fs)//2]
    if 'seg2d21' not in f0: need['seg2d'].append(s)
    if 'bbox2d' not in f0: need['bbox2d'].append(s)
    if 'occ' not in f0 and 'seg2d21' in f0: need['occ'].append(s)
    if 'agent_traj' not in f0: need['agenttraj'].append(s)
    if 'tl_state' not in m: need['tl'].append(s)
    if 'risk_map' not in m: need['risk'].append(s)
    if 'lanegraph' not in m: need['lanegraph'].append(s)
for k,v in need.items():
    open(f'/tmp/need_{k}.txt','w').write("\n".join(v))
    print(f"{k}: {len(v)}")
PYEOF

stage seg2d     extract_seg2d.py      /tmp/need_seg2d.txt
stage bbox2d    extract_bbox2d.py     /tmp/need_bbox2d.txt
# occ list must be rebuilt: seg2d just landed for ~1700 scenes
$PY - <<'PYEOF'
import json, os
root='out/bevlane'; need=[]
for s in sorted(os.listdir(root)):
    mf=f"{root}/{s}/manifest.json"
    if not os.path.exists(mf): continue
    try: m=json.load(open(mf))
    except Exception: continue
    fs=m.get('frames',[])
    if fs and 'seg2d21' in fs[len(fs)//2] and 'occ' not in fs[len(fs)//2]:
        need.append(s)
open('/tmp/need_occ.txt','w').write("\n".join(need)); print("occ:", len(need))
PYEOF
stage occ       extract_occ.py        /tmp/need_occ.txt
stage agenttraj extract_agent_traj.py /tmp/need_agenttraj.txt
stage tl        extract_tl.py         /tmp/need_tl.txt
stage risk      extract_risk.py       /tmp/need_risk.txt
stage lanegraph extract_lanegraph.py  /tmp/need_lanegraph.txt
$PY bevlane/annotate_indoor.py --workers $W >> out/catchup_indoor.log 2>&1
echo "ALL CATCH-UP DONE $(date)"

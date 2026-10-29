#!/bin/bash
# Convert every downloaded and verified KITScenes archive (<KITS_DL>/done/*.ok) into the bevlane layout, in parallel.
# Converted scenes are appended to out/kitscenes/scenes_<split>.txt and to out/nobox_scenes.txt (KITScenes has no 3D
# boxes: the box loss is disabled for these scenes unless auto-labels are added with kits_box_gt.py + METEOR_KITS_BOX=1).
# Re-running skips scenes that already have a manifest.json.
#
#   usage: KITS_DL=/path/to/kitscenes_download OUT=out/bevlane HR=/path/to/hr_images [WORKERS=6] bash tools/kitscenes/convert_all.sh
set -u
cd "$(dirname "$0")/../.."
KITS_DL=${KITS_DL:?download root of download_kitscenes.py}; OUT=${OUT:-out/bevlane}; HR=${HR:?root for the 1152x648 images}
W=${WORKERS:-6}; PY=${PYTHON:-python3}
mkdir -p "$OUT" out/kitscenes
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
one() {   # $1 = archive path relative to KITS_DL (data/<split>/<uuid>.tar)
  local tar=$1 id sp; id=$(basename "$tar" .tar); sp=$(basename "$(dirname "$tar")")
  [ -f "$OUT/KITS_$id/manifest.json" ] && return 0
  if "$PY" tools/kitscenes/kitscenes_to_bevlane.py --tar "$KITS_DL/$tar" --out "$OUT" --hr "$HR" --stride 2 --check >> out/kitscenes/convert.log 2>&1; then
    echo "KITS_$id" >> "out/kitscenes/scenes_$sp.txt"; echo "KITS_$id" >> out/nobox_scenes.txt
  else
    echo "FAIL $tar" >> out/kitscenes/convert.log
  fi
}
export -f one; export KITS_DL OUT HR PY
for f in "$KITS_DL"/done/*.ok; do head -1 "$f"; done | xargs -P "$W" -I{} bash -c 'one {}'
echo "converted: $(ls -d "$OUT"/KITS_*/manifest.json 2>/dev/null | wc -l) scenes (log: out/kitscenes/convert.log)"

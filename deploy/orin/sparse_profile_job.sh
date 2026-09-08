#!/bin/bash
# 密 vs 事後2:4 INT8 の層別プロファイル差分 (2026-09-06)。CPPBENCH_DONE を待って GPU 単独で。
cd ~/meteor; T=/usr/src/tensorrt/bin/trtexec; SO=/home/nvidia/meteor/liftbench/plugin/build/libmeteor_lift.so
L=out/sparse_profile.log; : > $L
while ! grep -q CPPBENCH_DONE out/cpp_vs_py_bench.log 2>/dev/null; do sleep 60; done
for e in v147c3Zg v147s24g; do
  $T --loadEngine=eng/${e}_int8.engine --staticPlugins=$SO --iterations=30 --warmUp=500 --dumpProfile --separateProfileRun --exportProfile=out/prof_${e}.json > out/prof_${e}.txt 2>&1
  grep -E "GPU Compute Time: min" out/prof_${e}.txt | cut -c1-120 >> $L
done
python3 - >> $L <<EOF
import json
a = {r["name"]: r["averageMs"] for r in json.load(open("out/prof_v147c3Zg.json")) if "name" in r}
b = {r["name"]: r["averageMs"] for r in json.load(open("out/prof_v147s24g.json")) if "name" in r}
print(f"layers dense {len(a)} sparse {len(b)} total dense {sum(a.values()):.1f} sparse {sum(b.values()):.1f} ms")
common = [(b[k]-a[k], k, a[k], b[k]) for k in a if k in b]
common.sort(reverse=True)
print("--- slower with sparse (top 15)")
for d, k, x, y in common[:15]: print(f"{d:+.2f} ms  {x:.2f}->{y:.2f}  {k[:90]}")
print("--- faster with sparse (top 10)")
for d, k, x, y in common[-10:]: print(f"{d:+.2f} ms  {x:.2f}->{y:.2f}  {k[:90]}")
onlyb = [k for k in b if k not in a]; onlya = [k for k in a if k not in b]
print(f"layers only in sparse: {len(onlyb)} ({sum(b[k] for k in onlyb):.1f} ms); only in dense: {len(onlya)} ({sum(a[k] for k in onlya):.1f} ms)")
for k in onlyb[:8]: print("  +", f"{b[k]:.2f}", k[:100])
EOF
echo SPARSEPROF_DONE >> $L

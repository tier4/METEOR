import json, re
from collections import defaultdict
a = {r["name"]: r["averageMs"] for r in json.load(open("out/prof_v147c3Zg.json")) if "name" in r}
b = {r["name"]: r["averageMs"] for r in json.load(open("out/prof_v147s24Fg.json")) if "name" in r}
def grp(k):
    m = re.match(r"/net/([a-z_0-9]+)", k)
    if m: return m.group(1)
    if k.startswith("__myl"): return "myelin"
    return "lift" if "MeteorLift" in k else "other"
ga = defaultdict(float); gb = defaultdict(float)
for k, v in a.items(): ga[grp(k)] += v
for k, v in b.items(): gb[grp(k)] += v
has = any("MeteorLift" in k for k in b)
print("total dense %.1f sparse+plugin %.1f ms; lift plugin present: %s" % (sum(a.values()), sum(b.values()), has))
for g in sorted(set(ga) | set(gb), key=lambda g: -(gb.get(g, 0) - ga.get(g, 0))):
    d = gb.get(g, 0) - ga.get(g, 0)
    if abs(d) > 0.15: print("  %-14s %6.2f -> %6.2f  (%+.2f)" % (g, ga.get(g, 0), gb.get(g, 0), d))

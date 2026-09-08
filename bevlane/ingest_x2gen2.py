#!/usr/bin/env python3
"""Ingest the x2gen2 (J6Gen2 / ePalette) 7-camera corpus.

x2gen2 differs from every earlier source in two ways that matter:

1. **7 cameras.** There is no CAM_BACK_NARROW: neither in the manifest `cams`
   dict nor in the per-frame `imgs`. Everything else is already written in the
   8-slot layout with slot 7 empty (depth4n ch1 zeros, seg2d ch7 = 255,
   bbox2d counts[7] = 0), so BevLaneDataset synthesises the missing camera as
   a zero image with a borrowed pose (see dataset.py `_CAM_FALLBACK`) instead
   of discarding the recording.
2. **No gt_cons.** Only the single-pass `gt` is present; the dataset falls
   back to it per frame.

30 % of the corpus is held out for test. The split is by RECORDING, not by
scene: consecutive chunks of one drive share the same road, so a scene-level
split would leak the test geography into training. It is additionally
stratified by location so both sides cover all five sites.

Idempotent: re-running picks up newly converted scenes and keeps the existing
train/test assignment for everything already assigned.
"""
import argparse
import collections
import json
import os
import re

SRC = "data/x2gen2/bevlane"
DST = "out/bevlane"
MIN_FRAMES = 20
MIN_CAMS = 7


def recording(s):
    """Drive identity: the scene name minus the trailing chunk index."""
    return re.sub(r"_(\d+)(_\d+)?$", "", s.replace(".db3.zst", ""))


def location(s):
    p = s.split("_")
    if s.startswith("DB_"):
        # DB_<vehicle>_<ver>_<Site>[_Sub]_<uuid>_<timestamp>_<chunk>
        return p[3]
    return p[0]


def survey(src):
    """Scene -> frame count for every scene that is complete enough to use."""
    ok, rej = {}, collections.Counter()
    for s in sorted(os.listdir(src)):
        mf = os.path.join(src, s, "manifest.json")
        if not os.path.exists(mf):
            rej["no manifest"] += 1
            continue
        try:
            m = json.load(open(mf))
        except Exception:
            rej["unreadable manifest"] += 1
            continue
        fr = m.get("frames", [])
        if len(fr) < MIN_FRAMES:
            rej[f"<{MIN_FRAMES} frames"] += 1
            continue
        if len(m.get("cams", {})) < MIN_CAMS:
            rej["too few cameras"] += 1
            continue
        # the GT the dataset will actually read must exist on disk
        f0 = fr[0]
        g = f0.get("gt_cons") or f0.get("gt")
        if not g or not os.path.exists(os.path.join(src, s, g)):
            rej["gt missing on disk"] += 1
            continue
        ok[s] = len(fr)
    return ok, rej


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--dst", default=DST)
    ap.add_argument("--test-frac", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    scenes, rej = survey(a.src)
    print(f"[survey] usable {len(scenes)} scenes, "
          f"{sum(scenes.values())} frames; rejected {dict(rej)}")

    # ---- split by recording, stratified by location, deterministic ----
    by_loc = collections.defaultdict(list)
    for r in sorted({recording(s) for s in scenes}):
        by_loc[location(r)].append(r)
    import random
    rng = random.Random(a.seed)
    test_recs = set()
    for loc, recs in sorted(by_loc.items()):
        recs = sorted(recs)
        rng.shuffle(recs)
        # frames per recording, so the 30 % is 30 % of DATA not of drives
        fr = {r: sum(n for s, n in scenes.items() if recording(s) == r)
              for r in recs}
        want = a.test_frac * sum(fr.values())
        got = 0
        for r in recs:
            if got >= want:
                break
            test_recs.add(r)
            got += fr[r]
        print(f"  {loc:14s} {len(recs):3d} drives  test "
              f"{sum(1 for r in recs if r in test_recs):3d} drives "
              f"{got}/{sum(fr.values())} frames "
              f"({100 * got / max(sum(fr.values()), 1):.0f}%)")

    tr = sorted(s for s in scenes if recording(s) not in test_recs)
    te = sorted(s for s in scenes if recording(s) in test_recs)
    ftr = sum(scenes[s] for s in tr)
    fte = sum(scenes[s] for s in te)
    print(f"[split] train {len(tr)} scenes / {ftr} frames | "
          f"test {len(te)} scenes / {fte} frames "
          f"({100 * fte / max(ftr + fte, 1):.1f}% held out)")
    assert not (set(tr) & set(te))

    if a.dry_run:
        return

    # ---- symlink into the training root ----
    os.makedirs(a.dst, exist_ok=True)
    made = kept = clash = 0
    for s in scenes:
        d = os.path.join(a.dst, s)
        src = os.path.join(a.src, s)
        if os.path.islink(d):
            if os.path.realpath(d) == os.path.realpath(src):
                kept += 1
            else:
                clash += 1
                print(f"[clash] {s} already points elsewhere; left alone")
            continue
        if os.path.exists(d):
            clash += 1
            print(f"[clash] {s} exists as a real directory; left alone")
            continue
        os.symlink(src, d)
        made += 1
    print(f"[link] new {made}  already correct {kept}  conflicts {clash}")

    open("out/x2gen2_train.txt", "w").write("\n".join(tr) + "\n")
    open("out/x2gen2_test.txt", "w").write("\n".join(te) + "\n")
    print("[write] out/x2gen2_train.txt out/x2gen2_test.txt")

    # ---- leak audit against every list already in use ----
    for name in ("out/round48_scenes.txt", "val.lst", "test.lst"):
        if not os.path.exists(name):
            continue
        have = set(l.strip() for l in open(name) if l.strip())
        print(f"[audit] {name}: overlap with x2gen2 train "
              f"{len(have & set(tr))}, with x2gen2 test {len(have & set(te))}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Fold the July 2026 recordings into the training root with an 80/20 split.

Three drops land in different shapes -- 2026-07-15T20-21 keeps its scenes one
level down under bevlane/, and 2026-07-23 has stray mp4s and a text file beside
the scenes -- so a scene is defined here as any directory holding a
manifest.json, found at either depth.

The split is BY RECORDING, never by scene. Scenes named <recording>_0, _1, _2
are consecutive slices of the same drive: splitting those individually would put
frames seconds apart on both sides of the train/holdout line and the holdout
would report memorisation as generalisation. Grouping by the name with the
trailing _N removed keeps a whole drive on one side.

Scenes are symlinked into the existing root rather than copied: 91 GB, and the
dataset only ever reads them.

    python3 bevlane/ingest_newdata.py --commit
"""
import argparse
import hashlib
import json
import os

ROOTS = ["data/2026-07-15T16-20",
         "data/2026-07-15T20-21",
         "data/2026-07-23",
         "data/2026-07-16",
         "data/3017_1",
         "data/3017_3"]


def scenes_under(root, depth=0, _max=3):
    """-> {scene name: absolute path}, at whatever depth the drop used.

    The drops are not consistent: 2026-07-16 puts scenes at the top, 2026-07-15
    T20-21 hides them under bevlane/, and 3017_1 goes two deeper still
    (3017_1/out/bevlane/<scene>). A fixed depth silently found zero scenes in
    3017_1 and would have dropped 157 of them.
    """
    out = {}
    if depth > _max:
        return out
    try:
        items = sorted(os.listdir(root))
    except Exception:
        return out
    for a in items:
        pa = os.path.join(root, a)
        if not os.path.isdir(pa):
            continue
        if os.path.exists(os.path.join(pa, "manifest.json")):
            out[a] = pa
        else:
            out.update(scenes_under(pa, depth + 1, _max))
    return out


def already_trained(lists):
    """Every scene any past round trained on. Those can never become holdout."""
    seen = set()
    for f in lists:
        if os.path.exists(f):
            seen |= set(open(f).read().split())
    return seen


def why_bad(path):
    """-> reason string if this scene must not be ingested, else None.

    Failed conversions are left on disk looking like scenes: us_1 has 41
    directories whose manifest parses but carries no frames, no cameras and an
    empty img/. Feeding those to the loader costs a stat storm per epoch and
    silently shrinks the effective batch. Everything here is a thing the
    training loop actually needs.
    """
    mf = os.path.join(path, "manifest.json")
    try:
        m = json.load(open(mf))
    except Exception as e:
        return f"manifest unreadable ({type(e).__name__})"
    if not m.get("frames"):
        return "no frames"
    if not m.get("cams"):
        return "no cameras"
    img = os.path.join(path, "img")
    if not os.path.isdir(img) or not os.listdir(img):
        return "img/ empty"
    if not os.path.exists(os.path.join(path, "ego_motion.npz")):
        return "no ego_motion.npz"
    f0 = m["frames"][0]
    if not f0.get("imgs"):
        return "frame 0 has no images"
    first = next(iter(f0["imgs"].values()))
    if not os.path.exists(os.path.join(path, first)):
        return "frame 0 image missing on disk"
    if not (f0.get("gt_cons") or f0.get("gt")):
        return "no BEV GT key"
    return None


def recording(scene):
    """scene name minus the trailing _<index>."""
    return scene.rsplit("_", 1)[0] if "_" in scene else scene


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="out/bevlane")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--train-out", default="out/newdata_train.txt")
    ap.add_argument("--test-out", default="out/newdata_test.txt")
    ap.add_argument("--trained-lists",
                    default="out/round54_scenes.txt,out/round49_scenes.txt,"
                            "out/round48x_scenes.txt,out/newdata_train.txt")
    ap.add_argument("--commit", action="store_true",
                    help="actually create the symlinks")
    a = ap.parse_args()

    found, rejected = {}, {}
    for r in ROOTS:
        if not os.path.isdir(r):
            print(f"  {r}: 存在しません")
            continue
        s = scenes_under(r)
        bad = {k: why_bad(v) for k, v in s.items()}
        bad = {k: b for k, b in bad.items() if b}
        ok = {k: v for k, v in s.items() if k not in bad}
        print(f"  {os.path.basename(r):24s} {len(ok):4d} 有効 / "
              f"{len(bad):3d} 除外 / {len(set(map(recording, ok))):3d} 録画")
        rejected.update({k: (os.path.basename(r), b) for k, b in bad.items()})
        for k, v in ok.items():
            if k in found:
                # later roots win: a re-export of the same recording is newer
                pass
            found[k] = v
    if rejected:
        from collections import Counter
        print("\n  除外の内訳:")
        for reason, n in Counter(b for _, b in rejected.values()).most_common():
            print(f"    {reason:34s} {n:4d}")

    recs = sorted(set(map(recording, found)))
    # deterministic, name-based: rerunning must reproduce the same split even if
    # more recordings arrive later, so a hash of the name decides, not an index
    def h(r):
        return int(hashlib.sha1(r.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    # A scene that any past round trained on cannot go to holdout: the model
    # has already seen it, so measuring on it would report memorisation as
    # generalisation. 205 of the 264 scenes in 2026-07-15T16-20 are in
    # round54_scenes.txt already, and a naive re-split would have moved some of
    # them across the line.
    trained = already_trained(a.trained_lists.split(","))
    protected = {recording(s) for s in found if s in trained}
    test_recs = {r for r in recs if h(r) < a.test_frac and r not in protected}
    if protected & {r for r in recs if h(r) < a.test_frac}:
        n = len(protected & {r for r in recs if h(r) < a.test_frac})
        print(f"  ({n} 録画は学習済みのため holdout から除外)")
    tr = sorted(s for s in found if recording(s) not in test_recs)
    te = sorted(s for s in found if recording(s) in test_recs)
    print(f"\n合計 {len(found)} シーン / {len(recs)} 録画")
    print(f"  train    {len(tr):4d} シーン / {len(recs) - len(test_recs):3d} 録画"
          f"  ({100 * len(tr) / max(len(found), 1):.1f}%)")
    print(f"  holdout  {len(te):4d} シーン / {len(test_recs):3d} 録画"
          f"  ({100 * len(te) / max(len(found), 1):.1f}%)")
    assert not (set(tr) & set(te))
    assert not ({recording(s) for s in tr} & {recording(s) for s in te}), \
        "録画が train と holdout の両方に現れています"

    if not a.commit:
        print("\n--commit を付けると symlink を張ります（今は何もしていません）")
        return
    os.makedirs(a.dest, exist_ok=True)
    n_new = n_skip = 0
    for k, v in found.items():
        d = os.path.join(a.dest, k)
        if os.path.lexists(d):
            n_skip += 1
            continue
        os.symlink(v, d)
        n_new += 1
    open(a.train_out, "w").write("\n".join(tr) + "\n")
    open(a.test_out, "w").write("\n".join(te) + "\n")
    print(f"\nsymlink: 新規 {n_new} / 既存 {n_skip}  -> {a.dest}")
    print(f"書き出し: {a.train_out} ({len(tr)})  {a.test_out} ({len(te)})")


if __name__ == "__main__":
    main()

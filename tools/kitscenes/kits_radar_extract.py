#!/usr/bin/env python3
"""Extract the files needed for the radar conversion from KITScenes sequence archives.

From each <uuid>.tar only the radar parquet files (radar_front / radar_left / radar_right), calibration/calib.json,
poses.txt and timestamp.reference.txt are written to OUT_DIR/<uuid>/ (a few tens of MB per sequence). The archive is
read once as a stream. Sequences that are already extracted (OUT_DIR/<uuid>/.ok) are skipped.
The result is the --raw input of tools/kitscenes/kits_radar_to_bevlane.py.

usage: kits_radar_extract.py OUT_DIR TAR [TAR ...]
"""
import os
import sys
import tarfile

KEEP_DIRS = {"radar_front", "radar_left", "radar_right", "calibration"}
KEEP_FILES = {"poses.txt", "timestamp.reference.txt"}


def one(tar, out):
    uid = os.path.basename(tar)[:-4]
    ok = os.path.join(out, uid, ".ok")
    if os.path.exists(ok):
        return f"[skip] {uid}"
    n = 0
    with tarfile.open(tar, "r|") as tf:
        for m in tf:
            parts = m.name.split("/")
            if len(parts) < 2 or not m.isfile() or ".." in parts:
                continue
            if parts[1] in KEEP_DIRS or (len(parts) == 2 and parts[1] in KEEP_FILES):
                dst = os.path.join(out, uid, *parts[1:])
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                with open(dst + ".tmp", "wb") as f:
                    f.write(tf.extractfile(m).read())
                os.replace(dst + ".tmp", dst); n += 1
    with open(ok, "w") as f:
        f.write(str(n))
    return f"[ok] {uid} files={n}"


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    out = sys.argv[1]
    for t in sys.argv[2:]:
        try:
            print(one(t, out), flush=True)
        except Exception as e:
            print(f"[fail] {t}: {e}", flush=True)

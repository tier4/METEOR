#!/usr/bin/env python3
"""Download KITScenes-Multimodal sequence archives from the Hugging Face Hub, split by split, resumable.

KITScenes-Multimodal (KIT-MRT) is a gated dataset under CC BY-NC: accept its terms on the Hub and log in
(`hf auth login`) first. Each archive is verified against the size and sha256 of the dataset's archive index
(CSV with the columns sequence_id, split, archive_path, archive_sha256, archive_size_bytes); a verified archive
leaves <root>/done/<sequence_id>.ok and is skipped on the next run.

  usage: download_kitscenes.py --root <download dir> --index <archive index csv> val [train ...] [--workers 4] [--max N]
"""
import argparse, csv, hashlib, os, time
from concurrent.futures import ThreadPoolExecutor

from huggingface_hub import hf_hub_download

REPO = "KIT-MRT/KITScenes-Multimodal"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("splits", nargs="+", help="e.g. val train")
    ap.add_argument("--root", required=True, help="download directory (archives land in <root>/data/<split>/)")
    ap.add_argument("--index", required=True, help="archive index CSV of the dataset (sequence_id, split, archive_path, archive_sha256, archive_size_bytes)")
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max", type=int, default=0, help="download at most N archives (smallest first; 1 = quick look)")
    a = ap.parse_args()
    os.makedirs(f"{a.root}/done", exist_ok=True)

    def log(m):
        print(time.strftime("%F %T ") + m, flush=True)

    def one(r):
        sid, path, sha, size = r["sequence_id"], r["archive_path"], r["archive_sha256"], int(r["archive_size_bytes"])
        ok = f"{a.root}/done/{sid}.ok"; dst = f"{a.root}/{path}"
        if os.path.exists(ok):
            return "skip"
        for att in range(3):
            try:
                hf_hub_download(a.repo, path, repo_type="dataset", local_dir=a.root)
                if os.path.getsize(dst) != size:
                    raise RuntimeError(f"size {os.path.getsize(dst)} != {size}")
                if sha256(dst) != sha:
                    raise RuntimeError("sha256 mismatch")
                open(ok, "w").write(path + "\n"); log(f"ok {r['split']} {sid} {size / 1e9:.2f} GB"); return "ok"
            except Exception as e:
                log(f"retry {att} {sid}: {str(e)[:120]}"); time.sleep(30 * (att + 1))
                try:
                    os.remove(dst)
                except OSError:
                    pass
        log(f"FAILED {sid}"); return "fail"

    rows = [r for r in csv.DictReader(open(a.index)) if r["split"] in a.splits]
    rows.sort(key=lambda r: (a.splits.index(r["split"]), int(r["archive_size_bytes"])))
    if a.max:
        rows = rows[:a.max]
    log(f"start splits={a.splits} n={len(rows)} total={sum(int(r['archive_size_bytes']) for r in rows) / 1e12:.2f} TB workers={a.workers}")
    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(one, rows))
    log(f"end ok={res.count('ok')} skip={res.count('skip')} fail={res.count('fail')}")


if __name__ == "__main__":
    main()

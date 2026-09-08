#!/usr/bin/env python3
"""Publish the METEOR release artefacts to the Hugging Face Hub.

Two repositories are created (or updated) under one namespace:

    <namespace>/meteor               model   checkpoint + plain ONNX + lift tables (AutowareFoundation layout)
    <namespace>/meteor-demo-scenes   dataset anonymised demo scene roots

The artefacts themselves are never in git (see .gitignore); they are staged on disk:

    <stage>/model/   README.md (copied from hf/MODEL_CARD.md), meteor_v157.pt, meteor_v157c3Z.onnx, meteor_v157.param.yaml, deploy_metadata.yaml, lift_plugin_tables_r64/, SHA256SUMS
    <stage>/data/    README.md (copied from hf/DATASET_CARD.md), one scene per road-type root, SHA256SUMS

Prerequisites (one-off):
    pip install -U huggingface_hub
    hf auth login            # a WRITE token of the account / org you publish under

Usage:
    python3 hf/publish_to_hf.py --namespace <hf-user-or-org> [--stage out/hf_stage] [--private]
                                [--only model|data] [--dry-run]
"""
import argparse
import hashlib
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_REPO = "meteor"
DATA_REPO = "meteor-demo-scenes"


def sha256sums(root, out_name="SHA256SUMS", skip=("README.md", ".gitattributes", ".gitignore")):
    """Write <root>/SHA256SUMS over every regular file below root (relative paths)."""
    lines = []
    for d, dirs, files in os.walk(root):
        dirs[:] = [x for x in dirs if x != ".cache"]        # upload_large_folder's local bookkeeping
        for f in sorted(files):
            rel = os.path.relpath(os.path.join(d, f), root)
            if rel in skip or rel == out_name:
                continue
            h = hashlib.sha256()
            with open(os.path.join(d, f), "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 22), b""):
                    h.update(chunk)
            lines.append(f"{h.hexdigest()}  {rel}")
    with open(os.path.join(root, out_name), "w") as fh:
        fh.write("\n".join(sorted(lines, key=lambda s: s.split("  ", 1)[1])) + "\n")
    return len(lines)


def du(root):
    return sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(root) for f in fs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--namespace", required=True, help="Hugging Face user or organisation")
    ap.add_argument("--stage", default="out/hf_stage", help="staging dir with model/ and data/")
    ap.add_argument("--private", action="store_true", help="create the repos as private")
    ap.add_argument("--only", choices=["model", "data"], default=None)
    ap.add_argument("--model-repo", default=MODEL_REPO)
    ap.add_argument("--data-repo", default=DATA_REPO)
    ap.add_argument("--dry-run", action="store_true", help="stage cards + checksums, print the plan, upload nothing")
    ap.add_argument("--sums-only", action="store_true", help="only (re)write SHA256SUMS in the staging dirs")
    ap.add_argument("--tag", default="v1.0", help="git tag to create on both repos after upload ('' to skip)")
    a = ap.parse_args()

    jobs = []
    if a.only in (None, "model"):
        jobs.append(("model", os.path.join(a.stage, "model"), f"{a.namespace}/{a.model_repo}", "MODEL_CARD.md"))
    if a.only in (None, "data"):
        jobs.append(("dataset", os.path.join(a.stage, "data"), f"{a.namespace}/{a.data_repo}", "DATASET_CARD.md"))

    for repo_type, folder, repo_id, card in jobs:
        if not os.path.isdir(folder):
            sys.exit(f"staging dir missing: {folder}")
        shutil.copyfile(os.path.join(HERE, card), os.path.join(folder, "README.md"))
        n = sha256sums(folder)
        print(f"[{repo_type}] {repo_id}: {n} files, {du(folder) / 2**30:.2f} GiB staged in {folder}")
    if a.sums_only or a.dry_run:
        print("dry run: nothing uploaded")
        return

    from huggingface_hub import HfApi
    api = HfApi()
    who = api.whoami()
    print(f"logged in as {who.get('name')}  (orgs: {[o['name'] for o in who.get('orgs', [])]})")

    for repo_type, folder, repo_id, _ in jobs:
        url = api.create_repo(repo_id, repo_type=repo_type, private=a.private, exist_ok=True)
        print(f"[{repo_type}] {url}")
        if repo_type == "dataset":
            # thousands of small jpgs: the resumable multi-worker path
            api.upload_large_folder(repo_id=repo_id, repo_type=repo_type, folder_path=folder,
                                    print_report_every=30)
        else:
            api.upload_folder(repo_id=repo_id, repo_type=repo_type, folder_path=folder,
                              commit_message="METEOR v157 release: checkpoint, plain ONNX, parameters")
        if a.tag:
            api.create_tag(repo_id, tag=a.tag, repo_type=repo_type, exist_ok=True)
            print(f"[{repo_type}] tagged {a.tag}")
        print(f"[{repo_type}] uploaded -> https://huggingface.co/"
              f"{'datasets/' if repo_type == 'dataset' else ''}{repo_id}")


if __name__ == "__main__":
    main()

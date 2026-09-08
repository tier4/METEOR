#!/usr/bin/env python3
"""Register Cosmos3 Transfer scenes in an existing BevLane dataset root.

Cosmos stores the same source scene under several appearance conditions, so a
plain UUID is ambiguous.  This tool creates stable, condition-qualified
symlinks in the normal dataset root and writes the corresponding scene list.
It never overwrites an existing path.
"""
import argparse
import json
import os
from pathlib import Path


DEFAULT_CONDITIONS = (
    "backlit",
    "heavy_rain",
    "heavy_snow",
    "night",
    "night_heavy_rain",
    "night_heavy_snow",
)
REQUIRED_FRAME_KEYS = {
    "gt", "gt_vec", "imgs", "depth4", "depth4n", "bev_box_p",
    "seg2d21", "bbox2d", "occ", "agent_traj", "lidar_bev",
}


def read_words(path):
    return [x for x in Path(path).read_text().split() if x]


def validate_scene(scene_dir):
    manifest_path = scene_dir / "manifest.json"
    if not manifest_path.is_file():
        return False, "manifest missing"
    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception as exc:
        return False, f"manifest unreadable: {exc}"
    if len(manifest.get("cams", {})) != 8:
        return False, f"camera count={len(manifest.get('cams', {}))}"
    frames = manifest.get("frames", [])
    if not frames:
        return False, "no frames"
    for index in {0, len(frames) // 2, len(frames) - 1}:
        frame = frames[index]
        missing = REQUIRED_FRAME_KEYS - set(frame)
        if missing:
            return False, f"frame {index} missing {sorted(missing)}"
        for rel in frame["imgs"].values():
            if not (scene_dir / rel).is_file():
                return False, f"frame {index} image missing: {rel}"
        for key in REQUIRED_FRAME_KEYS - {"imgs"}:
            if not (scene_dir / frame[key]).is_file():
                return False, f"frame {index} {key} missing: {frame[key]}"
    return True, f"frames={len(frames)}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True,
                        help="cosmos3_transfer directory")
    parser.add_argument("--dataset-root", required=True,
                        help="normal flat BevLane dataset directory")
    parser.add_argument("--out-list", required=True)
    parser.add_argument("--base-list", default="",
                        help="optional existing scene list to extend")
    parser.add_argument("--combined-list", default="",
                        help="write base-list plus Cosmos aliases")
    parser.add_argument("--list-kind", choices=("clean", "train"),
                        default="clean")
    parser.add_argument("--conditions", default=",".join(DEFAULT_CONDITIONS))
    parser.add_argument("--prefix", default="cosmos3")
    parser.add_argument("--exclude-list", action="append", default=[],
                        help="base UUIDs that must not enter training")
    parser.add_argument("--apply", action="store_true",
                        help="create symlinks and the output list")
    args = parser.parse_args()

    source_root = Path(args.source_root).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    conditions = tuple(x for x in args.conditions.split(",") if x)
    excluded = set()
    for path in args.exclude_list:
        excluded.update(read_words(path))

    accepted = []
    failures = []
    for condition in conditions:
        list_path = source_root / f"{condition}_scenes_{args.list_kind}.txt"
        if not list_path.is_file():
            failures.append((condition, "condition list missing"))
            continue
        for scene in read_words(list_path):
            if scene in excluded:
                failures.append((f"{condition}/{scene}", "excluded UUID"))
                continue
            scene_dir = source_root / condition / scene
            ok, detail = validate_scene(scene_dir)
            if not ok:
                failures.append((f"{condition}/{scene}", detail))
                continue
            alias = f"{args.prefix}_{condition}_{scene}"
            accepted.append((alias, scene_dir, condition, detail))

    aliases = [x[0] for x in accepted]
    if len(aliases) != len(set(aliases)):
        raise RuntimeError("condition-qualified aliases are not unique")

    if args.apply:
        dataset_root.mkdir(parents=True, exist_ok=True)
        for alias, source, _, _ in accepted:
            target = dataset_root / alias
            if target.is_symlink() and target.resolve() == source.resolve():
                continue
            if target.exists() or target.is_symlink():
                raise FileExistsError(f"refusing to replace {target}")
            target.symlink_to(source, target_is_directory=True)
        out_path = Path(args.out_list)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        tmp.write_text("".join(f"{alias}\n" for alias in aliases))
        tmp.replace(out_path)
        if args.combined_list:
            if not args.base_list:
                raise ValueError("--combined-list requires --base-list")
            combined = []
            seen = set()
            for scene in read_words(args.base_list) + aliases:
                if scene not in seen:
                    combined.append(scene)
                    seen.add(scene)
            combined_path = Path(args.combined_list)
            combined_path.parent.mkdir(parents=True, exist_ok=True)
            combined_tmp = combined_path.with_suffix(combined_path.suffix + ".tmp")
            combined_tmp.write_text("".join(f"{scene}\n" for scene in combined))
            combined_tmp.replace(combined_path)

    print(f"accepted={len(accepted)} failed_or_excluded={len(failures)} "
          f"apply={args.apply}")
    for name, reason in failures[:20]:
        print(f"[reject] {name}: {reason}")
    if len(failures) > 20:
        print(f"... {len(failures) - 20} more rejections")
    for condition in conditions:
        count = sum(c == condition for _, _, c, _ in accepted)
        print(f"{condition}: {count}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""生の t4dataset シーンを、変換ファイル無しでデモに読ませるアダプタ。

manifest.json の代わりに annotation/*.json をその場で解釈し、デモ側が
期待するのと同じ形 (cams: K/T_ego_cam、frames: imgs、v0/pose) を返す。
キット同梱のため bevlane パッケージには依存しない (json/numpy のみ)。

- K は 768x432 へのリサイズ分をここでスケールする (画像のリサイズは
  読み込み側 load_image が行う)
- t4 の罠: ego_pose.json は時刻順とは限らない -> timestamp で扱う
"""
import json
import os

import numpy as np

IMG_W, IMG_H = 768, 432


def is_t4_scene(d):
    # annotation ディレクトリの存在だけでは足りない (シーン内の tmp/ 等が
    # annotation もどきを持つことがあり、誤ってシーン扱いされた実害あり)。
    # 実際に読む sample.json の実在まで確認する。
    return os.path.isfile(os.path.join(d, "annotation", "sample.json")) and \
        not os.path.isfile(os.path.join(d, "manifest.json"))


def _quat_to_rot(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def load_t4_scene(sdir, cams):
    ann = os.path.join(sdir, "annotation")
    load = lambda n: json.load(open(os.path.join(ann, n + ".json")))
    samples = load("sample")
    by_tok = {s["token"]: s for s in samples}
    cur = [s for s in samples if not s["prev"]][0]
    ordered = []
    while True:
        ordered.append(cur)
        if not cur["next"]:
            break
        cur = by_tok[cur["next"]]
    per_sample = {}
    for d in load("sample_data"):
        if d.get("is_key_frame"):
            ch = d["filename"].split("/")[1]
            per_sample.setdefault(d["sample_token"], {})[ch] = d
    calib = {c["token"]: c for c in load("calibrated_sensor")}
    egop = {e["token"]: e for e in load("ego_pose")}

    cam_cache, frames, poses, stamps = {}, [], [], []
    for s in ordered:
        fr = per_sample.get(s["token"], {})
        imgs = {}
        for ch in cams:
            sd = fr.get(ch)
            if sd is None:
                imgs = None
                break
            if ch not in cam_cache:
                cal = calib[sd["calibrated_sensor_token"]]
                K = np.array(cal["camera_intrinsic"], dtype=np.float64)
                w0 = sd.get("width", 2880) or 2880
                h0 = sd.get("height", 1860) or 1860
                K[0] *= IMG_W / w0
                K[1] *= IMG_H / h0
                T = np.eye(4)
                T[:3, :3] = _quat_to_rot(cal["rotation"])
                T[:3, 3] = cal["translation"]
                cam_cache[ch] = {"K": K.tolist(), "T_ego_cam": T.tolist()}
            imgs[ch] = sd["filename"]          # 元解像度のパスをそのまま
        if not imgs:
            continue
        ep = egop[fr[cams[0]]["ego_pose_token"]]
        R = _quat_to_rot(ep["rotation"])
        poses.append([float(ep["translation"][0]),
                      float(ep["translation"][1]),
                      float(np.arctan2(R[1, 0], R[0, 0]))])
        stamps.append(float(fr[cams[0]]["timestamp"]) * 1e-6)
        frames.append({"frame": len(frames), "imgs": imgs})

    pose = np.array(poses, np.float32)
    ts = np.array(stamps, np.float64)
    v0 = np.zeros(len(pose), np.float32)
    if len(pose) > 1:
        d = np.linalg.norm(np.diff(pose[:, :2], axis=0), axis=1)
        v0[1:] = d / np.clip(np.diff(ts), 1e-3, None)
        v0[0] = v0[1]
    man = {"scene": os.path.basename(sdir.rstrip("/")),
           "img_hw": [IMG_H, IMG_W], "cams": cam_cache, "frames": frames}
    return man, v0, pose

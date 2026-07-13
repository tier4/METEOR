"""Dataset over extracted bevlane samples (6 cams + ego-centric BEV GT)."""
import json
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

cv2.setNumThreads(0)

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
CAMS = ["CAM_FRONT_WIDE", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
        "CAM_BACK_WIDE", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
        "CAM_FRONT_NARROW", "CAM_BACK_NARROW"]


class BevLaneDataset(Dataset):
    def __init__(self, root, scenes, max_per_scene=None, gt_key="gt",
                 dontcare_sidewalk=False, with_depth=False, augment=False,
                 with_seg2d=False, depth_hw=None, with_box=False,
                 with_boxdet=False, trim_start=0, trim_end=0,
                 min_cov_core=0.0, min_cov_fwd=0.0, seg2d_key="seg2d",
                 with_bbox2d=False, with_ego=False, with_occ=False):
        self.root = root
        self.gt_key = gt_key
        self.dontcare_sidewalk = dontcare_sidewalk
        self.with_depth = with_depth
        self.with_seg2d = with_seg2d
        self.seg2d_key = seg2d_key   # "seg2d" (12cls) or "seg2d21" (csv 21cls)
        self.depth_hw = depth_hw     # (H,W): resize all depth to this (mixed-res safe)
        self.with_box = with_box
        self.with_boxdet = with_boxdet
        self.with_bbox2d = with_bbox2d
        self.with_ego = with_ego
        self.with_occ = with_occ
        self._ego_cache = {}
        self.augment = augment
        self.items = []
        self.calib = {}
        for s in scenes:
            mf = os.path.join(root, s, "manifest.json")
            if not os.path.exists(mf):
                continue
            try:                       # robust to concurrent manifest writes
                m = json.load(open(mf))
            except Exception:
                continue
            if set(CAMS) - set(m["cams"]):
                continue
            K = np.stack([np.array(m["cams"][c]["K"], np.float32) for c in CAMS])
            Tc = np.stack([np.linalg.inv(np.array(m["cams"][c]["T_ego_cam"],
                                                  np.float32)) for c in CAMS])
            self.calib[s] = (K, Tc)
            frames = [f for f in m["frames"] if gt_key in f
                      and (not with_depth or "depth4" in f or "depth" in f)

                      and (not with_box or "bev_box" in f)
                      and (not with_boxdet or "bev_box_p" in f)
                      and (min_cov_core <= 0 or "gtcov" not in f
                           or (f["gtcov"][0] >= min_cov_core
                               and f["gtcov"][1] >= min_cov_fwd))]
            # scene ends lack accumulated LiDAR ahead/behind -> weak GT there
            if trim_end and len(frames) > trim_start + trim_end + 10:
                frames = frames[trim_start:len(frames) - trim_end]
            if max_per_scene:
                frames = frames[:max_per_scene]
            for f in frames:
                self.items.append((s, f))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        s, f = self.items[i]
        K, Tc = self.calib[s]
        imgs = []
        drop = np.random.randint(6) if (self.augment and np.random.rand() < 0.15) else -1
        for ci, c in enumerate(CAMS):
            img = cv2.imread(os.path.join(self.root, s, f["imgs"][c]))
            if img is None:      # unreadable sample: fall back to a neighbor
                return self.__getitem__((i + 1) % len(self.items))
            img = img[:, :, ::-1].astype(np.float32) / 255.0
            if self.augment:
                if ci == drop:
                    img[:] = 0.0
                else:
                    img *= np.random.uniform(0.7, 1.3)                # brightness
                    img += np.random.uniform(-0.08, 0.08)             # offset
                    mu = img.mean()
                    img = (img - mu) * np.random.uniform(0.8, 1.25) + mu  # contrast
                    img += np.random.normal(0, 0.012, img.shape).astype(np.float32)
                    img = np.clip(img, 0, 1)
            img = (img - MEAN) / STD
            imgs.append(img.transpose(2, 0, 1))
        gt = cv2.imread(os.path.join(self.root, s, f[self.gt_key]), 0)
        if gt is None:
            return self.__getitem__((i + 1) % len(self.items))
        if self.dontcare_sidewalk:
            gt = gt.copy()
            gt[gt == 2] = 0    # sidewalk -> don't care (ignored like bg)
        # copy K/Tc: they are cached/shared per scene -> from_numpy on shared
        # storage breaks the DataLoader shared-memory collate ("not resizable")
        out = [torch.from_numpy(np.ascontiguousarray(np.stack(imgs))),
               torch.from_numpy(np.ascontiguousarray(K)),
               torch.from_numpy(np.ascontiguousarray(Tc)),
               torch.from_numpy(np.ascontiguousarray(gt.astype(np.int64)))]
        if self.with_depth:
            try:
                d = np.load(os.path.join(self.root, s, f["depth4"]))["depth"]
                d = d.astype(np.float32)
            except Exception:
                d = np.zeros((6, 72, 128), np.float32)
            if self.depth_hw and d.shape[1:] != tuple(self.depth_hw):
                H2, W2 = self.depth_hw
                d = np.stack([cv2.resize(c, (W2, H2), interpolation=cv2.INTER_NEAREST)
                              for c in d])
            if len(CAMS) == 8:
                try:
                    dn = np.load(os.path.join(self.root, s, f["depth4n"]))["depth"]
                    dn = dn.astype(np.float32)
                    if self.depth_hw and dn.shape[1:] != tuple(self.depth_hw):
                        H2, W2 = self.depth_hw
                        dn = np.stack([cv2.resize(c, (W2, H2),
                                       interpolation=cv2.INTER_NEAREST) for c in dn])
                    if dn.shape[1:] != d.shape[1:]:   # stale/mismatched narrow
                        dn = np.zeros((2,) + d.shape[1:], np.float32)
                except Exception:
                    dn = np.zeros((2,) + d.shape[1:], np.float32)
                d = np.concatenate([d, dn], 0)
            out.append(torch.from_numpy(np.ascontiguousarray(d)))
        if self.with_seg2d:
            try:
                sg = np.load(os.path.join(self.root, s, f[self.seg2d_key]))["seg"]
            except Exception:
                sg = np.full((len(CAMS), 108, 192), 255, np.uint8)
            out.append(torch.from_numpy(sg.astype(np.int64)))
        if self.with_box:
            bx = cv2.imread(os.path.join(self.root, s, f.get("bev_box", "_")), 0)
            if bx is None:
                bx = np.zeros((800, 500), np.uint8)
            out.append(torch.from_numpy(bx.astype(np.int64)))
        if self.with_boxdet:
            try:
                bp = np.load(os.path.join(self.root, s,
                                          f["bev_box_p"]))["boxes"]
            except Exception:
                bp = np.zeros((0, 6), np.float32)
            KMAX = 64
            pad = np.zeros((KMAX, 6), np.float32)
            nb = min(len(bp), KMAX)
            pad[:nb] = bp[:nb]
            out.append(torch.from_numpy(pad))
            out.append(torch.tensor(nb, dtype=torch.int64))
        if self.with_bbox2d:
            K2 = 96
            try:
                z = np.load(os.path.join(self.root, s, f["bbox2d"]))
                b2, c2 = z["boxes"].astype(np.float32), z["counts"]
            except Exception:                 # not extracted yet -> no boxes
                b2 = np.zeros((len(CAMS), K2, 5), np.float32)
                c2 = np.zeros(len(CAMS), np.uint8)
            if b2.shape[1] != K2:             # normalize old KMAX npz
                p = np.zeros((len(CAMS), K2, 5), np.float32)
                k = min(K2, b2.shape[1])
                p[:, :k] = b2[:, :k]
                b2 = p
                c2 = np.minimum(c2, k)
            out.append(torch.from_numpy(b2))
            out.append(torch.from_numpy(c2.astype(np.int64)))
        if self.with_ego:
            # [17] = wp(12), v0, acc, steer, brake, valid (0 when missing)
            e = np.zeros(17, np.float32)
            if s not in self._ego_cache:
                p = os.path.join(self.root, s, "ego_motion.npz")
                try:
                    z = np.load(p)
                    self._ego_cache[s] = {k: z[k] for k in
                                          ("wp", "v0", "acc", "steer",
                                           "brake", "valid")}
                except Exception:
                    self._ego_cache[s] = None
            z = self._ego_cache[s]
            fi = f["frame"]
            if z is not None and fi < len(z["v0"]):
                e[:12] = z["wp"][fi].reshape(-1)
                e[12], e[13] = z["v0"][fi], z["acc"][fi]
                e[14], e[15] = z["steer"][fi], z["brake"][fi]
                e[16] = z["valid"][fi]
            out.append(torch.from_numpy(e))
        if self.with_occ:
            try:
                oc = np.load(os.path.join(self.root, s, f["occ"]))["occ"]
            except Exception:                 # not extracted yet -> all ignore
                oc = np.full((16, 200, 200), 255, np.uint8)
            out.append(torch.from_numpy(oc.astype(np.int64)))
        # clone -> each tensor owns fresh, resizable storage (from_numpy storage
        # is not resizable, which breaks the shared-memory DataLoader collate)
        return tuple(t.clone() for t in out)

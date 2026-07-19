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
                 with_bbox2d=False, with_ego=False, with_occ=False,
                 with_agenttraj=False, with_temporal=False, with_tl=False,
                 with_risk=False, with_lanegraph=False, temporal_hist=0,
                 with_unknown=False, with_lidarbev=False):
        self.root = root
        self.with_lidarbev = with_lidarbev
        self.gt_key = gt_key
        self.dontcare_sidewalk = dontcare_sidewalk
        self.with_depth = with_depth
        self.with_seg2d = with_seg2d
        self.seg2d_key = seg2d_key   # "seg2d" (12cls) or "seg2d21" (csv 21cls)
        self.depth_hw = depth_hw     # (H,W): resize all depth to this (mixed-res safe)
        self.with_box = with_box
        self.with_boxdet = with_boxdet
        self.with_agenttraj = with_agenttraj
        self.with_temporal = with_temporal
        self._byfi = {}
        self.with_bbox2d = with_bbox2d
        self.with_ego = with_ego
        self.with_occ = with_occ
        self.with_tl = with_tl
        self.with_risk = with_risk
        self.with_lanegraph = with_lanegraph
        self.with_unknown = with_unknown
        self._unk_cache = {}
        self.temporal_hist = temporal_hist   # v29: N history slots
        self._tl_cache = {}
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
            if with_temporal:
                self._byfi[s] = {fr["frame"]: fr for fr in m["frames"]}

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
        if self.gt_key == "gt_cons":
            gt = gt.copy()
            gt[gt == 255] = 0     # consensus-ignore -> this codebase's 0

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
        if self.with_agenttraj:
            # boxes for the 3D det loss + per-instance future offsets
            try:
                z = np.load(os.path.join(self.root, s, f["agent_traj"]))
                ab, an = z["boxes"], int(z["count"])
                at, av = z["traj"], z["tvalid"]
            except Exception:                 # not extracted yet
                ab = np.zeros((64, 6), np.float32); an = 0
                at = np.zeros((64, 6, 2), np.float32)
                av = np.zeros((64, 6), np.float32)
            out.append(torch.from_numpy(ab.astype(np.float32)))
            out.append(torch.tensor(an, dtype=torch.int64))
            out.append(torch.from_numpy(at.astype(np.float32)))
            out.append(torch.from_numpy(av.astype(np.float32)))
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
        if self.with_tl:
            # [1] int64 state: 0 none / 1 green / 2 yellow / 3 red; 255 = no GT
            t = 255
            if s not in self._tl_cache:
                try:
                    self._tl_cache[s] = np.load(
                        os.path.join(self.root, s, "tl_state.npz"))["label"]
                except Exception:
                    self._tl_cache[s] = None
            z = self._tl_cache[s]
            fi = f["frame"]
            if z is not None and fi < len(z):
                t = int(z[fi])
            out.append(torch.tensor(t, dtype=torch.int64))
        if self.with_risk:
            # [400,250] float32 risk in [0,1]; all -1 when not extracted yet
            r = np.full((400, 250), -1.0, np.float32)
            try:
                z = np.load(os.path.join(self.root, s, "risk_map.npz"))
                arr = z["risk"]
                fi = f["frame"]
                if fi < len(arr):
                    r = arr[fi].astype(np.float32) / 255.0
            except Exception:
                pass
            out.append(torch.from_numpy(r))
        if self.with_lanegraph:
            # pts [24,12,2], cls [24] (255 empty), n, adj [24,24]
            lp = np.zeros((24, 12, 2), np.float32)
            lc = np.full(24, 255, np.int64)
            ln = 0
            la = np.zeros((24, 24), np.float32)
            try:
                z = np.load(os.path.join(self.root, s, "lanegraph.npz"))
                fi = f["frame"]
                if fi < len(z["n"]):
                    lp = z["pts"][fi].astype(np.float32)
                    lc = z["cls"][fi].astype(np.int64)
                    ln = int(z["n"][fi])
                    la = z["adj"][fi].astype(np.float32)
            except Exception:
                pass
            out.append(torch.from_numpy(lp))
            out.append(torch.from_numpy(lc))
            out.append(torch.tensor(ln, dtype=torch.int64))
            out.append(torch.from_numpy(la))
        if self.with_unknown:
            UKM = 64
            uc = np.zeros((UKM, 2), np.float32)
            un = 0
            if s not in self._unk_cache:
                try:
                    z = np.load(os.path.join(self.root, s, "unknown_obj.npz"))
                    self._unk_cache[s] = (z["centers"], z["n"],
                                          z["n_ign"] if "n_ign" in z
                                          else None)
                except Exception:
                    self._unk_cache[s] = None
            z = self._unk_cache[s]
            fi = f["frame"]
            if z is not None and fi < len(z[1]):
                c = z[0][fi].astype(np.float32)     # KMAX 32 (old) or 64
                k = min(len(c), UKM)
                uc[:k] = c[:k]
                nv = min(int(z[1][fi]), k)
                ni = min(int(z[2][fi]), k - nv) if z[2] is not None else 0
                # packed: low byte = positives, high byte = ignore entries
                # that follow them in uc (occluded blobs, v3 GT)
                un = nv + (ni << 8)
            out.append(torch.from_numpy(uc))
            out.append(torch.tensor(un, dtype=torch.int64))
        if self.with_lidarbev:
            lb = np.zeros((4, 400, 250), np.float32)
            p_ = f.get("lidar_bev")
            if p_:
                try:
                    lb = np.load(os.path.join(self.root, s, p_)
                                 )["lb"].astype(np.float32)
                except Exception:
                    pass
            out.append(torch.from_numpy(lb))
        if self.with_temporal and self.temporal_hist > 0:
            # v29 memory queue: N history frames at fi-2, fi-6, fi-14
            HN = self.temporal_hist
            OFFS = (2, 6, 14)[:HN]
            himgs = np.zeros((HN, len(CAMS), 3, 432, 768), np.float32)
            hrel = np.zeros((HN, 3), np.float32)
            hval = np.zeros(HN, np.float32)
            if s not in self._ego_cache:
                try:
                    z = np.load(os.path.join(self.root, s, "ego_motion.npz"))
                    self._ego_cache[s] = {k: z[k] for k in z.files}
                except Exception:
                    self._ego_cache[s] = None
            z = self._ego_cache[s]
            if z is not None and "pose" in z and f["frame"] < len(z["pose"]):
                pc_ = z["pose"][f["frame"]]
                for hi, off in enumerate(OFFS):
                    fp = self._byfi.get(s, {}).get(f["frame"] - off)
                    if fp is None or fp["frame"] >= len(z["pose"]):
                        continue
                    pp_ = z["pose"][fp["frame"]]
                    if abs(pc_).sum() == 0 or abs(pp_).sum() == 0:
                        continue
                    ok = True
                    tmp = []
                    for c in CAMS:
                        im = cv2.imread(os.path.join(self.root, s,
                                                     fp["imgs"].get(c, "_")))
                        if im is None:
                            ok = False
                            break
                        im = im[:, :, ::-1].astype(np.float32) / 255.0
                        tmp.append(((im - MEAN) / STD).transpose(2, 0, 1))
                    if not ok:
                        continue
                    himgs[hi] = np.stack(tmp)
                    dy = float(pc_[2] - pp_[2])
                    cp, sp = np.cos(pp_[2]), np.sin(pp_[2])
                    dx0, dy0 = pc_[0] - pp_[0], pc_[1] - pp_[1]
                    hrel[hi] = (cp * dx0 + sp * dy0,
                                -sp * dx0 + cp * dy0, dy)
                    hval[hi] = 1.0
            out.append(torch.from_numpy(np.ascontiguousarray(himgs)))
            out.append(torch.from_numpy(hrel))
            out.append(torch.from_numpy(hval))
        elif self.with_temporal:
            # previous frame (0.4 s back): images + relative 2D pose
            pimgs = np.zeros((len(CAMS), 3, 432, 768), np.float32)
            rel = np.zeros(3, np.float32)
            pv = np.zeros(1, np.float32)
            fp = self._byfi.get(s, {}).get(f["frame"] - 2)
            if s not in self._ego_cache:
                try:
                    z = np.load(os.path.join(self.root, s, "ego_motion.npz"))
                    self._ego_cache[s] = {k: z[k] for k in z.files}
                except Exception:
                    self._ego_cache[s] = None
            z = self._ego_cache[s]
            if fp is not None and z is not None and "pose" in z                     and f["frame"] < len(z["pose"]):
                ok = True
                tmp = []
                for c in CAMS:
                    im = cv2.imread(os.path.join(self.root, s,
                                                 fp["imgs"].get(c, "_")))
                    if im is None:
                        ok = False
                        break
                    im = im[:, :, ::-1].astype(np.float32) / 255.0
                    tmp.append(((im - MEAN) / STD).transpose(2, 0, 1))
                pc_, pp_ = z["pose"][f["frame"]], z["pose"][fp["frame"]]
                if ok and (abs(pc_).sum() > 0) and (abs(pp_).sum() > 0):
                    pimgs = np.stack(tmp)
                    dy = float(pc_[2] - pp_[2])
                    cp, sp = np.cos(pp_[2]), np.sin(pp_[2])
                    dx0, dy0 = pc_[0] - pp_[0], pc_[1] - pp_[1]
                    rel[:] = (cp * dx0 + sp * dy0,
                              -sp * dx0 + cp * dy0, dy)
                    pv[0] = 1.0
            out.append(torch.from_numpy(np.ascontiguousarray(pimgs)))
            out.append(torch.from_numpy(rel))
            out.append(torch.from_numpy(pv))
        # clone -> each tensor owns fresh, resizable storage (from_numpy storage
        # is not resizable, which breaks the shared-memory DataLoader collate)
        return tuple(t.clone() for t in out)

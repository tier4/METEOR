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
# pose donor for a camera a recording does not carry
_CAM_FALLBACK = {"CAM_BACK_NARROW": "CAM_BACK_WIDE",
                 "CAM_FRONT_NARROW": "CAM_FRONT_WIDE",
                 "CAM_BACK_WIDE": "CAM_BACK_LEFT",
                 "CAM_FRONT_WIDE": "CAM_FRONT_NARROW"}


class BevLaneDataset(Dataset):
    def __init__(self, root, scenes, max_per_scene=None, gt_key="gt",
                 dontcare_sidewalk=False, with_depth=False, augment=False,
                 with_seg2d=False, depth_hw=None, with_box=False,
                 with_boxdet=False, trim_start=0, trim_end=0,
                 min_cov_core=0.0, min_cov_fwd=0.0, seg2d_key="seg2d",
                 with_bbox2d=False, with_ego=False, with_occ=False,
                 with_agenttraj=False, with_temporal=False, with_tl=False,
                 with_risk=False, with_lanegraph=False, temporal_hist=0,
                 with_unknown=False, with_lidarbev=False,
                 with_unknown_v2=False, unk2_key="unknown_v2",
                 with_sdmap=False, with_tlin=False, cam_drop=0.0,
                 n_cams=len(CAMS), img_scale=1, yaw_fix_deg=0.0,
                 use_gt_valid=False, ego_mask_prefix=None):
        # img_scale=2: R7 resolution axis, phase "upsample-first" -- the
        # stored 432x768 jpgs are bicubic-upsampled on load and K is scaled
        # to match, so the x2/stride-8 model trains BEFORE the true-res
        # re-ingest exists. 2D GT (depth/seg2d at 108x192) needs no change:
        # the s8 fuse lands on the same feature grid. bbox2d pixel coords
        # are scaled here.
        self.img_scale = int(img_scale)
        self.root = root
        self.with_lidarbev = with_lidarbev
        self.gt_key = gt_key
        self.use_gt_valid = bool(use_gt_valid)
        # v131 design (2026-08-27): synthetic domain (cosmos) trains perception GT only
        # and is excluded from the ego (E2E) loss. Just zero the valid flag for prefix-
        # matched scenes; the existing valid gate in ego_loss does the rest.
        self.ego_mask_prefix = ego_mask_prefix
        # Inter-layer GT rotation fix (2026-08-19). Pose-derived GT (BEV rasters, ego
        # trajectory wp) is rotated ~0.72 deg clockwise relative to calib-derived GT
        # (3D boxes, depth) (self-calibrated via box-to-lane-center offset vs distance, out/yawfix_plan.md).
        # A positive angle rotates back CCW about the ego origin (row400,col250).
        # occ/risk/agent are handled in the next stage (after the A/B gate passes).
        self.yaw_fix_deg = float(yaw_fix_deg)
        self._yaw_M = (cv2.getRotationMatrix2D((250.0, 400.0),
                                               self.yaw_fix_deg, 1.0)
                       if self.yaw_fix_deg else None)
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
        self.with_unknown_v2 = with_unknown_v2
        self.unk2_key = unk2_key
        self.with_sdmap = with_sdmap
        self.n_cams = int(n_cams)
        self.with_tlin = with_tlin
        self._unk_cache = {}
        self.temporal_hist = temporal_hist   # v29: N history slots
        self._tl_cache = {}
        self._ego_cache = {}
        self.augment = augment
        self.items = []
        self.calib = {}
        # Cameras a recording simply does not have (x2gen2 is a 7-camera
        # J6Gen2 set: no CAM_BACK_NARROW). Its per-camera arrays are already
        # written in the 8-slot layout with slot 7 empty (depth4n ch1 zeros,
        # seg2d ch7 255, bbox2d counts[7] 0), so only the manifest `cams`
        # entry and the jpg are missing -> synthesise those instead of
        # discarding the recording.
        self.absent = {}
        self.hw = {}
        # Scenes whose 3D-box conversion NEVER RAN (out/nobox_scenes.txt =
        # the x2gen2 corpus). This is a provenance list, deliberately NOT a
        # measured "is it empty" test: 1,404 Japanese scenes are genuinely
        # object-free in all 147 frames (quiet rural roads) and that IS valid
        # negative supervision the detector needs; only x2gen2's boxes were
        # never produced (2D boxes are present there, the BEV/3D stage is
        # missing). out/empty_box_scenes_diag.txt holds the measured scan for
        # reference.
        # Empty files there taught the detector to stay silent on that rig
        # (heatmap 0.919 on the Japanese rig vs 0.080 on x2gen2, zero boxes over
        # the demo threshold). Read from a list, never probed per scene --
        # opening ~10 npz per scene made the dataset scan take hours across
        # 8 ranks and hung r49 twice.
        self.nobox = set()
        for _p in ("out/nobox_scenes.txt",
                   os.path.join(os.path.dirname(os.path.abspath(
                       root.rstrip("/"))), "nobox_scenes.txt")):
            if os.path.exists(_p):
                self.nobox = {l.strip() for l in open(_p) if l.strip()}
                break
        self.cam_drop = cam_drop
        for s in scenes:
            mf = os.path.join(root, s, "manifest.json")
            if not os.path.exists(mf):
                continue
            try:                       # robust to concurrent manifest writes
                m = json.load(open(mf))
            except Exception:
                continue
            miss = [c for c in CAMS if c not in m["cams"]]
            if len(miss) > 3:                   # too little of the rig left
                continue
            # Geometry for an absent camera: borrow the analogous present one.
            # The image fed there is all zeros, so only the pose has to be
            # sane -- a bogus K would splat the zero features anywhere.
            sub = {}
            for c in miss:
                for cand in (_CAM_FALLBACK.get(c), *CAMS):
                    if cand in m["cams"]:
                        sub[c] = cand
                        break
            K = np.stack([np.array(m["cams"][sub.get(c, c)]["K"], np.float32)
                          for c in CAMS])
            if self.img_scale != 1:
                K = K.copy()
                K[:, :2] *= self.img_scale      # fx fy cx cy follow the pixels
            Tc = np.stack([np.linalg.inv(
                np.array(m["cams"][sub.get(c, c)]["T_ego_cam"], np.float32))
                for c in CAMS])
            self.calib[s] = (K, Tc)
            self.hw[s] = tuple(m.get("img_hw", (432, 768)))
            # Scenes whose 3D-box annotation was never produced. x2gen2 ships
            # bev_box_p / agent_traj files that are EMPTY in every frame (2D
            # boxes are present, only the BEV/3D conversion never ran), and an
            # empty file is indistinguishable from "no objects here" -- it
            # trains the detector to stay silent on that whole domain. Measured
            # on r48: heatmap score 0.919 on the Japanese rig vs 0.080 on
            # x2gen2, zero boxes over the 0.25 demo threshold in 0 % of frames.
            # Probe up to 5 frames spread over the scene; the Japanese corpus
            # has boxes in 93 % of frames, so 5 empty probes is not chance.
            if miss:
                self.absent[s] = tuple(CAMS.index(c) for c in miss)
            have = [c for c in CAMS if c in m["cams"]]
            # A frame that falls back to `gt` must actually HAVE that file:
            # scene 6yb9g3aj_..._475 lists gt for frames 74-146 whose PNGs were
            # never written (74 of 147), and 73 consecutive unreadable samples
            # defeated the 64-try loop and killed the refiner six times.
            # os.path.exists only runs on fallback frames, so scenes with
            # gt_cons pay nothing.
            def _gt_ok(f):
                if gt_key in f:
                    return True
                p = f.get("gt")
                return bool(p) and os.path.exists(os.path.join(root, s, p))

            frames = [f for f in m["frames"]
                      if _gt_ok(f)
                      # one US scene ships a frame whose imgs dict lacks
                      # CAM_BACK_NARROW: a KeyError deep in the dataloader
                      # killed the whole 8-GPU round (r47, 2026-08-01).
                      # Judged against the scene's OWN camera set so that a
                      # 7-camera recording is not thrown away wholesale.
                      and not (set(have) - set(f.get("imgs", {})))
                      and (not with_depth or "depth4" in f or "depth" in f)

                      and (not with_box or "bev_box" in f)
                      and (not with_boxdet or "bev_box_p" in f)
                      and (min_cov_core <= 0 or "gtcov" not in f
                           or (f["gtcov"][0] >= min_cov_core
                               and f["gtcov"][1] >= min_cov_fwd))]
            # scene ends lack accumulated LiDAR ahead/behind -> weak GT there
            if (trim_start or trim_end) and \
                    len(frames) > trim_start + trim_end + 10:
                stop = len(frames) - trim_end if trim_end else len(frames)
                frames = frames[trim_start:stop]
            if max_per_scene:
                frames = frames[:max_per_scene]
            for f in frames:
                self.items.append((s, f))
            if with_temporal:
                self._byfi[s] = {fr["frame"]: fr for fr in m["frames"]}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        n = len(self.items)
        # stride the retries: consecutive indices are consecutive FRAMES of one
        # scene, so a scene with a long broken run defeated a +1 walk. 1009 is
        # coprime with any plausible n, so 64 tries sample the whole set.
        for k in range(64):
            r = self._get_one((i + k * 1009) % n)
            if r is not None:
                return r
        raise RuntimeError(
            f"64 consecutive unreadable samples from index {i} "
            f"(scene {self.items[i % n][0]}) -- data is broken, not transient")

    def _get_one(self, i):
        s, f = self.items[i]
        K, Tc = self.calib[s]
        imgs = []
        drop = np.random.randint(6) if (self.augment and np.random.rand() < 0.15) else -1
        # Cameras that carry NO image for this sample: the ones the recording
        # lacks, plus (with prob cam_drop) CAM_BACK_NARROW on 8-camera samples
        # so the 7-camera configuration stays calibrated in the same weights.
        # An absent camera is a ZERO TENSOR, matching model.zero_cams and the
        # 7-camera demo, so train and inference see the identical thing.
        gone = set(self.absent.get(s, ()))
        if self.cam_drop and 7 not in gone and np.random.rand() < self.cam_drop:
            gone.add(7)
        for ci, c in enumerate(CAMS):
            if ci in gone:
                imgs.append(np.zeros(
                    (3, self.hw[s][0] * self.img_scale,
                     self.hw[s][1] * self.img_scale), np.float32))
                continue
            img = cv2.imread(os.path.join(self.root, s, f["imgs"][c]))
            if img is None:      # unreadable sample: fall back to a neighbor
                return None            # caller advances
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
            if self.img_scale != 1:
                # Target is base resolution (manifest img_hw) x scale. Full-res roots
                # (R program: stores 1536x864) are already at target size, so 1x;
                # legacy 768 roots are upsampled x2 as before.
                th = self.hw[s][0] * self.img_scale
                tw = self.hw[s][1] * self.img_scale
                if img.shape[0] != th or img.shape[1] != tw:
                    img = cv2.resize(img, (tw, th),
                                     interpolation=cv2.INTER_CUBIC)
            imgs.append(img.transpose(2, 0, 1))
        gk = self.gt_key if self.gt_key in f else "gt"
        gt = cv2.imread(os.path.join(self.root, s, f[gk]), 0)
        if gt is None:
            return None                # caller advances
        if self._yaw_M is not None and gt.shape == (800, 500):
            # Border is 255 (ignore in gt_cons; raw gt maps 255->0 = background below).
            gt = cv2.warpAffine(gt, self._yaw_M, (500, 800),
                                flags=cv2.INTER_NEAREST,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=255)
            if gk != "gt_cons":
                gt[gt == 255] = 0      # rotation border -> ignore(0)
        if gk == "gt_cons":
            gt = gt.copy()
            # Preserve consensus 255 as true don't-care. Mapping it to class 0
            # is wrong when --train-bg is enabled: unobserved/conflicting rear
            # cells then become hard background negatives for lane/road.
        if self.use_gt_valid and f.get("gt_valid"):
            valid = cv2.imread(os.path.join(self.root, s, f["gt_valid"]), 0)
            if valid is None or valid.shape != gt.shape:
                return None
            if self._yaw_M is not None and valid.shape == (800, 500):
                valid = cv2.warpAffine(
                    valid, self._yaw_M, (500, 800),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            gt = gt.copy()
            gt[valid == 0] = 255
        # US-derived paint labels (marking=7) are treated as road (2026-08-18,
        # user decision). The JP GT policy maps zebra/channelizing areas to road;
        # only the US corpus teaches 7, and that inconsistency made marking (yellow)
        # fire on channelizing areas in the new domain. JP GT has no 7, so this remap
        # only affects US labels. The 2D seg (21-class) taxonomy is untouched.
        if (gt == 7).any():
            gt = gt.copy()
            gt[gt == 7] = 1

        if self.dontcare_sidewalk:
            gt = gt.copy()
            gt[gt == 2] = 0    # sidewalk -> don't care (ignored like bg)
        # copy K/Tc: they are cached/shared per scene -> from_numpy on shared
        # storage breaks the DataLoader shared-memory collate ("not resizable")
        # n_cams < len(CAMS) DROPS the tail of the camera list rather than
        # zeroing it. CAMS ends with CAM_BACK_NARROW, so n_cams=7 removes
        # exactly that one and the backbone and depth tower run 7/8 as often --
        # zeroing (cam_drop above) keeps the compute and only removes the
        # information, which is what you want for rig robustness and not what
        # you want for latency.
        nc = self.n_cams
        out = [torch.from_numpy(np.ascontiguousarray(np.stack(imgs[:nc]))),
               torch.from_numpy(np.ascontiguousarray(K[:nc])),
               torch.from_numpy(np.ascontiguousarray(Tc[:nc])),
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
            if gone:                      # no image there -> no depth target
                d = d.copy()
                for ci in gone:
                    if ci < d.shape[0]:
                        d[ci] = 0.0
            out.append(torch.from_numpy(np.ascontiguousarray(d[:nc])))
        if self.with_seg2d:
            try:
                sg = np.load(os.path.join(self.root, s, f[self.seg2d_key]))["seg"]
            except Exception:
                sg = np.full((len(CAMS), 108, 192), 255, np.uint8)
            if gone:
                sg = sg.copy()
                for ci in gone:
                    if ci < sg.shape[0]:
                        sg[ci] = 255           # ignore index
            out.append(torch.from_numpy(sg[:nc].astype(np.int64)))
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
            # -1 = "this scene has NO box annotation", so the detector losses
            # can skip it instead of learning "no objects in this domain".
            out.append(torch.tensor(-1 if s in self.nobox else nb,
                                    dtype=torch.int64))
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
            out.append(torch.tensor(-1 if s in self.nobox else an,
                                    dtype=torch.int64))   # -1 = unannotated
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
            if self.img_scale != 1:
                # Layout is (cls, cx, cy, w, h) -- coords are 1:5. [:4] would double
                # the class column and point at channel 18 (real bug, 2026-08-14).
                b2 = b2.copy()
                b2[:, :, 1:5] *= self.img_scale
            if gone:
                c2 = np.array(c2).copy()
                for ci in gone:
                    if ci < len(c2):
                        c2[ci] = 0         # no image -> no 2D boxes to find
            out.append(torch.from_numpy(b2[:nc]))
            out.append(torch.from_numpy(np.asarray(c2[:nc]).astype(np.int64)))
        if self.with_ego:
            # [17] = wp(12), v0, acc, steer, brake, valid (0 when missing)
            e = np.zeros(17, np.float32)
            if s not in self._ego_cache:
                p = os.path.join(self.root, s, "ego_motion.npz")
                try:
                    z = np.load(p)
                    # Keep all keys (2026-09-04). Previously only 6 keys were kept, so
                    # the temporal branch downstream could not find "pose" in the same
                    # cache and, with with_ego, the 3 history slots were always invalid
                    # (all zeros) = temporal fusion got zero input in every E2E round.
                    self._ego_cache[s] = {k: z[k] for k in z.files}
                except Exception:
                    self._ego_cache[s] = None
            z = self._ego_cache[s]
            fi = f["frame"]
            if z is not None and fi < len(z["v0"]):
                wp_ = z["wp"][fi].reshape(6, 2)
                if self.yaw_fix_deg:
                    # Same corrective rotation as the rasters (+deg in the ego plane, same
                    # convention as apply_yaw_fix; straight-wp drift measured -0.16 -> +0.06 m).
                    th_ = np.radians(self.yaw_fix_deg)
                    c_, s_ = np.cos(th_), np.sin(th_)
                    wp_ = np.stack([c_ * wp_[:, 0] - s_ * wp_[:, 1],
                                    s_ * wp_[:, 0] + c_ * wp_[:, 1]], 1)
                e[:12] = wp_.reshape(-1)
                e[12], e[13] = z["v0"][fi], z["acc"][fi]
                e[14], e[15] = z["steer"][fi], z["brake"][fi]
                e[16] = z["valid"][fi]
                if self.ego_mask_prefix and s.startswith(self.ego_mask_prefix):
                    e[16] = 0.0        # perception only (excluded from E2E loss)
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
        if self.with_unknown_v2:
            # dense small-obstacle occupancy mask [400,250] (LiDAR-accumulated
            # unknown - known-box residual); -1 sentinel = no GT for this
            # frame. unknown_v3 adds per-cell code 2 = camera-occluded ->
            # mapped to -1 (don't-care) here so loss/eval skip those cells.
            um = np.full((400, 250), -1.0, np.float32)
            p_ = f.get(self.unk2_key) or f.get("unknown_v2")
            if p_:
                try:
                    um = np.load(os.path.join(self.root, s, p_)
                                 )["mask"].astype(np.float32)
                    um[um == 2.0] = -1.0
                except Exception:
                    pass
            out.append(torch.from_numpy(um))
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
        if self.with_sdmap:
            # OSM SD-map prior [4,400,250] (road/centerline/inters/cross);
            # zeros when the scene has no GNSS/OSM raster (= prior off)
            sd = np.zeros((4, 400, 250), np.float32)
            p_ = f.get("sdmap")
            if p_:
                try:
                    sd = np.load(os.path.join(self.root, s, p_)
                                 )["sd"].astype(np.float32)
                except Exception:
                    pass
            out.append(torch.from_numpy(sd))
        if self.with_tlin:
            # per-camera BOX-LEVEL traffic-light raster [8,7,27,48]:
            # [red,yel,grn,is_ped,is_arrow,sin,cos] painted in each bbox.
            # zeros = recognizer off (bit-equal to no-input in v47).
            tlr = np.zeros((self.n_cams, 7, 27, 48), np.float32)
            p_ = f.get("tl")
            if p_:
                try:
                    bx = np.load(os.path.join(self.root, s, p_))["boxes"]
                    for b in bx:
                        ci = int(b[0])
                        if not 0 <= ci < 8:
                            continue
                        x1 = int(np.clip(b[1] * 48, 0, 47))
                        x2 = int(np.clip(b[3] * 48, 0, 47)) + 1
                        y1 = int(np.clip(b[2] * 27, 0, 26))
                        y2 = int(np.clip(b[4] * 27, 0, 26)) + 1
                        for ch in range(7):
                            v = float(b[5 + ch])
                            if v != 0.0:
                                tlr[ci, ch, y1:y2, x1:x2] = v
                except Exception:
                    pass
            out.append(torch.from_numpy(tlr))
        if self.with_temporal and self.temporal_hist > 0:
            # v29 memory queue: N history frames at fi-2, fi-6, fi-14
            HN = self.temporal_hist
            OFFS = (2, 6, 14)[:HN]
            himgs = np.zeros((HN, self.n_cams, 3, 432, 768), np.float32)
            hrel = np.zeros((HN, 3), np.float32)
            hval = np.zeros(HN, np.float32)
            if s not in self._ego_cache:
                try:
                    z = np.load(os.path.join(self.root, s, "ego_motion.npz"))
                    self._ego_cache[s] = {k: z[k] for k in z.files}
                except Exception:
                    self._ego_cache[s] = None
            z = self._ego_cache[s]
            if os.environ.get("METEOR_ZERO_HIST", "0") == "1":
                # Pin history to zero for comparable evaluation (2026-09-04). All rounds
                # before the fix trained/validated with zero history, so evaluations compared
                # against old numbers (Okinawa holdout / dummy gating runs) set this.
                z = None
            if z is not None and "pose" not in z:      # legacy cache fallback
                try:
                    _z = np.load(os.path.join(self.root, s, "ego_motion.npz"))
                    z = self._ego_cache[s] = {k: _z[k] for k in _z.files}
                except Exception:
                    pass
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
                    _gone = set(self.absent.get(s, ()))
                    for ci, c in enumerate(CAMS[:self.n_cams]):
                        # Missing camera on 7-camera vehicles -> zero image, as in the main path
                        # (2026-09-04: previously imread("_") failed -> whole history
                        # slot invalid + a flood of OpenCV WARNs).
                        if ci in _gone or c not in fp["imgs"]:
                            tmp.append(np.zeros((3, 432, 768), np.float32))
                            continue
                        im = cv2.imread(os.path.join(self.root, s,
                                                     fp["imgs"][c]))
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
            pimgs = np.zeros((self.n_cams, 3, 432, 768), np.float32)
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

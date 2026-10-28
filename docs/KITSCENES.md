# METEOR2 on KITScenes

[KITScenes-Multimodal](https://huggingface.co/datasets/KIT-MRT/KITScenes-Multimodal) (KIT-MRT) is a public
driving dataset with a 6-camera ring, a high-resolution front camera, 7 LiDARs, 10 Hz ego poses and a Lanelet2 HD
map per sequence. METEOR2 is fine-tuned on its **train split only** and evaluated on its **val split**; the
pretraining is the in-house METEOR2 checkpoint (Co-MLOps data).

> **License.** KITScenes is released under **CC BY-NC**. The tools in this repository (Apache-2.0) convert and use
> the data, but **checkpoints fine-tuned on KITScenes are research-only and are not released.**

## 1. Download

The dataset is gated: accept its terms on the Hugging Face Hub and `hf auth login` first. The downloader verifies
every archive against the dataset's archive index (size + sha256) and is resumable.

```bash
python3 tools/kitscenes/download_kitscenes.py --root /data/kitscenes --index <archive index csv> val train
```

## 2. Convert to the bevlane layout

```bash
KITS_DL=/data/kitscenes OUT=out/bevlane HR=/data/kitscenes_hr WORKERS=6 bash tools/kitscenes/convert_all.sh
# one sequence: python3 tools/kitscenes/kitscenes_to_bevlane.py --tar <seq>.tar --out out/bevlane --hr /data/kitscenes_hr --check
```

Each sequence becomes a scene `KITS_<uuid>` (50 frames at 5 Hz) with 768×432 images (and 1152×648 copies under
`HR`), BEV lane / road GT rendered from the Lanelet2 map, LiDAR rasters, sparse depth targets, ego-motion targets,
the raw 10 Hz poses (`poses10.txt`) and the map itself (`maps/`). Camera mapping: the 6 ring cameras map onto the
6 wide slots of the Co-MLOps rig; a centred 30° crop of the front-centre camera fills `CAM_FRONT_NARROW`. Scene
lists are written to `out/kitscenes/scenes_<split>.txt`; `--check` writes `check_overlay.png` (map GT projected into
the front camera) per scene. Scenes whose name starts with `KITS_` are treated as HD-map scenes
(`METEOR_HDMAP_PREFIX`): no yaw-calibration correction, and map / route rasters are available.

## 3. (Optional) 3D boxes from auto-labels

KITScenes has no 3D box annotations. To train the box head, run an offline LiDAR 3D detector over the converted
scenes (we use the CoMET auto-labeler) and convert its output:

```bash
python3 tools/kitscenes/kits_box_gt.py --labels <label root> --root out/bevlane    # format in the script's docstring
```

and fine-tune with `BOXES=1` (`METEOR_KITS_BOX=1`). Detection scores on KITScenes are then measured **against
these auto-labels**, so they are reference numbers, not scores against human annotations.

## 4. Fine-tune

Three input configurations, all from the same METEOR2 checkpoint, all with the 6 ring cameras
(`METEOR_ABSENT_CAMS=6` feeds a zero image in the narrow slot):

| `VARIANT` | Inputs |
|---|---|
| `full` | images + ego speed + ego history (64 past steps at 10 Hz, strictly past samples) + HD map + lane-level route |
| `nomap` | images + ego speed + ego history + route (HD-map channels empty) |
| `camonly` | images + ego speed (the plain METEOR2 interface) |

```bash
INIT=<METEOR2 checkpoint> VARIANT=full GPUS=0,1,2,3 EP=12 HR=/data/kitscenes_hr bash tools/kitscenes/finetune.sh
```

The route is a **lane-level corridor**: the scene's driven trajectory is matched to Lanelet2 lanelets with an HMM
(`bevlane/ae2e_route.py`), the visited lanelets' centrelines are drawn 3.5 m wide, and the destination is the last
pose of the scene. Scenes whose match fails the quality gate get no route. The map / route raster and the history
enter the network through zero-initialised adapters (`--ae2e 1`), so the fine-tune starts from the camera-only
function. Report a **fixed epoch decided before training (the final epoch)**, not the best epoch on val.

## 5. Evaluate (trajectory)

```bash
METEOR_AE2E_ROUTE=lanelet METEOR_AE2E_HIST=past64 METEOR_ABSENT_CAMS=6 \
python3 bevlane/kitscenes_eval.py full=out/kits_ft/ckpt_kits_full/last.pt \
    --val-list out/kitscenes/scenes_val.txt --hr-root /data/kitscenes_hr --set kf202
# no-HD-map checkpoint: add METEOR_AE2E_NOMAP=1; camera-only checkpoint: the conditioning env is ignored
```

| Set | Definition |
|---|---|
| `kf202` | per val sequence, 10 Hz keyframes a = 40 + 90k with a + 30 ≤ U − 1 (4 s of past, 3 s of future) → **202 keyframes** |
| `dense` | every even 10 Hz frame a = 40 … U − 51 (the converted images are the even frames) |

Ground truth is the raw 10 Hz future (`poses10.txt`) in the ego frame of keyframe a. The prediction is the selected
candidate of the ego head (6 points at 0.5 s), linearly interpolated to 30 points at 0.1 s. ADE@1s / 2s / 3s = mean
L2 error over the first 10 / 20 / 30 points; FDE@3s = error at 3 s. `--dump DIR` saves per-keyframe errors.

## 6. Demo

```bash
METEOR_AE2E=1 METEOR_AE2E_ROUTE=lanelet METEOR_AE2E_HIST=past64 METEOR_ABSENT_CAMS=6 \
METEOR_HR_ROOT=/data/kitscenes_hr METEOR_IMG_SCALE=1.5 \
python3 bevlane/demo_rgbd_bev.py --ckpt models/meteor_v157.pt --model v52 --n-seg2d 21 \
    --mite-ckpt out/kits_ft/ckpt_kits_full/last.pt --root out/bevlane --scenes <KITS_uuid ...> \
    --seg-fuse --show-ae-map --bev-proj --no-occ --fps 10 --out out/kits_demo.mp4
```

With `--mite-ckpt`, every forward pass is answered by the METEOR2 checkpoint; the v52 checkpoint (`--ckpt`, e.g.
the released `meteor_v157.pt`) only hosts the decoding / rendering. `--show-ae-map` draws the input HD map, route
corridor and destination next to the predicted BEV; `--bev-proj` projects the predicted BEV classes into the camera
images. `METEOR_F3_HEAD=<best.pt>` adds the F3 trajectory refinement head.

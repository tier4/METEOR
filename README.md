<div align="center">

# ☄️ METEOR

### **M**ulti-task **E**stimation of **T**raffic **E**lements, **O**bjects & **R**oads

*Eight cameras (Co-MLOps DRS). One network. Seven driving tasks. No LiDAR at inference.*

**Zero human labels. Zero human-written code.**

[![Built on CoMET](https://img.shields.io/badge/built_on-CoMET_(Co--MLOps)-orange)](https://co-mlops.tier4.jp/)
[![GTC 2026](https://img.shields.io/badge/NVIDIA_GTC_2026-session_S81897-76B900)](https://www.nvidia.com/ja-jp/gtc/session-catalog/sessions/gtc26-s81897/)
[![Tasks](https://img.shields.io/badge/tasks-7-blueviolet)]()
[![Params](https://img.shields.io/badge/params-38.3M-blue)]()
[![Compute](https://img.shields.io/badge/compute-3.1_TFLOPs-informational)]()
[![TensorRT](https://img.shields.io/badge/TensorRT-ready-76B900)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C)]()
[![Labels](https://img.shields.io/badge/human_labels-0-success)]()
[![Code](https://img.shields.io/badge/human_written_code-0-success)]()
[![Author](https://img.shields.io/badge/authored_by-Claude_Fable_5-cc785c)]()

<img src="docs/media/inference.gif" width="880" alt="METEOR multi-task inference"/>

*Live inference — 8 DRS cameras in, everything out: 2D segmentation, 2D & 3D detection,
metric depth, BEV lane map, 3D occupancy, and an end-to-end driving path (green ribbon).*

</div>

---

## What is METEOR?

METEOR is a **surround-view multi-task network** for autonomous driving: it consumes
the **8 cameras of the [Co-MLOps](https://co-mlops.tier4.jp/) Data Recording System (DRS)** — no LiDAR at
inference — and is trained **entirely on auto-generated ground truth, zero human
annotation** (LiDAR is used offline, by the label factory only).

It is built on **CoMET — the autolabeling foundation of the [Co-MLOps](https://co-mlops.tier4.jp/) project**:
CoMET's LiDAR × panoptic × ego-pose autolabels are distilled into every one of
METEOR's seven supervision signals. The pipeline in this repository extends the
CoMET foundation from BEV lane maps to depth, 2D/3D detection, occupancy and
end-to-end driving targets — turning raw t4dataset-format recordings into a
complete multi-task training set with no labeling cost. CoMET made the labels;
METEOR is what the labels can train.

From **8 cameras (768×432) + calibration + current speed**, a single forward pass predicts:

| # | Task | Output | Head cost |
|---|------|--------|-----------|
| 1 | **BEV lane segmentation** | 9 classes, 160 × 100 m @ 0.2 m | 1.16 M / 933 G |
| 2 | **Metric depth** | 64 bins × 8 cams @ stride 4 | 3.29 M / 1093 G |
| 3 | **3D oriented boxes** | CenterPoint-style, vehicles + VRU | 0.41 M / 82 G |
| 4 | **2D semantic segmentation** | 21 classes (Cityscapes-like palette) | 4.18 M / 219 G |
| 5 | **2D detection** | 10 classes, YOLO-style 3-scale | 2.85 M / 219 G |
| 6 | **E2E driving** | 3 s trajectory + steering / accel / brake | 4.04 M / 37 G |
| 7 | **3D semantic occupancy** | 10 classes, 16 × 200 × 200 voxels @ 0.4 m | 0.70 M / 55 G |

All heads share one ResNet-34 + FPN image backbone and one **depth-gated IPM** BEV
representation — adding a task costs **< 5 %** of total compute.

> ### 🚀 Have Co-MLOps data? You can build this.
> Everything here needs nothing but **raw [Co-MLOps](https://co-mlops.tier4.jp/) DRS recordings** (t4dataset
> format). No annotation team, no labeling budget, no hand-written code:
> point the autolabel factory at your scenes, run the trainer, and you get a
> 7-task driving model — **from raw logs to a trained network in three
> commands** (see [Quickstart](#quickstart)).

## Fully automated, end to end

This project is an experiment in **full automation of an ML system** — both of
its ingredients are machine-made:

- **No human labels.** Every supervision signal for all seven tasks is
  distilled automatically from raw recordings by the CoMET-based autolabel
  factory. Nobody drew a box, a mask, or a trajectory.
- **No human-written code.** Every line in this repository — the models, the
  GT extractors, the trainer, the demo renderers, the docs, the architecture
  diagram, and this README — was written by **Claude (Fable 5, Anthropic)**
  operating autonomously: humans set the goals and reviewed the results; the
  agent designed, implemented, debugged, measured, and iterated. The rolling
  training rounds themselves (data refresh, relaunch, metric-driven loss
  fixes) run under the same agent loop.

## Architecture

<div align="center">
<img src="docs/media/architecture.png" width="880" alt="METEOR architecture"/>
</div>

The signature block is the **depth-gated IPM**: BEV grid points are projected into every
camera (K/T), context features are `grid_sample`d, and each sample is **gated by the
predicted depth probability at its true range** — depth acts as a learned visibility valve
for the geometric projection. No transformers, no deformable attention:
`conv / grid_sample / gather / maxpool / MLP` only, so the whole network exports to
**TensorRT** as-is. Details in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## The autolabel factory (powered by CoMET / [Co-MLOps](https://co-mlops.tier4.jp/))

Every supervision signal is distilled offline from raw recordings — LiDAR, ego-pose and
2D panoptic masks — by a 10-stage, per-scene resumable pipeline built on the
**CoMET autolabeling foundation from the Co-MLOps project**:

<div align="center">
<img src="docs/media/groundtruth.gif" width="880" alt="auto-generated ground truth"/>

*Auto-generated GT: 21-class 2D seg + 10-class 2D boxes + BEV lanes + oriented
3D boxes + E2E trajectory (green) with speed / steering / accel / brake.*

<img src="docs/media/occ_gt.gif" width="880" alt="occupancy ground truth"/>

*3D occupancy GT: accumulated labeled LiDAR, voxelized with ray-carved free space —
top-down and isometric views.*

</div>

Highlights (full recipe in [docs/DATA_PIPELINE.md](docs/DATA_PIPELINE.md)):

- **BEV lanes** — LiDAR × panoptic accumulation in the map frame, vectorized into
  connected polylines, re-rendered with hole-filling and an ego-connected drivable filter.
- **Depth** — LiDAR splats + road-merged linear interpolation + geometric ground fill.
- **3D boxes** — LiDAR annotations kept only when geometrically confirmed by a camera.
- **E2E** — future ego-pose → trajectory; bicycle-model steering; accel / brake from
  smoothed speed (no CAN required).
- **Occupancy** — multi-sweep labeled LiDAR voxelization, dynamic-object de-smearing,
  0.4 m ray-stepped free-space carving.
- **Quality gates** — scene-end trimming, stationary-spot exclusion, GT-coverage filters,
  intersection guards.

## Results (validation, unseen recording)

| Metric | Value |
|---|---|
| BEV lane mIoU | **0.292** |
| 2D seg mIoU (21 cls) | **0.451** |
| Depth MAE | **1.60 m** (0–80 m) |
| E2E trajectory ADE / FDE (3 s) | **0.76 m / 1.65 m** |
| E2E curve ADE (\|lat\| > 2 m) | **0.72 m** |
| Brake accuracy | **0.81** |

Trained on **~2,000 scenes / two vehicle platforms**, list growing continuously as the
autolabel factory converts more recordings (rolling training rounds).

## Quickstart

From raw Co-MLOps recordings to a trained 7-task model — three commands:

```bash
# 1) Convert raw scenes into training GT (10 stages, resumable, scene-parallel)
python3 bevlane/convert_dtset.py --scenes scene_list.txt --workers 12

# 2) Train the 7-task model (7 GPUs)
torchrun --nproc_per_node=7 bevlane/train.py \
  --model v20 --batch 2 --epochs 24 --workers 0 \
  --gt-key gt_vec --train-list scenes.txt --train-bg --aug \
  --seg-w 1.0 --dice-w .5 --lovasz-w .5 --boundary-w 3 --tversky-w .6 --far-w 1 \
  --depth-w 0.6 --box-w 0.8 --seg2d-w 0.4 --bbox2d-w 0.3 --ego-w 0.5 --occ-w 0.5 \
  --seg2d-key seg2d21 --n-seg2d 21 --out out/ckpt

# 3) Render the full multi-task demo video
python3 bevlane/demo_rgbd_bev.py --model v20 --n-seg2d 21 \
  --ckpt out/ckpt/best.pt --show-seg2d --scenes <SCENE ...> --out out/demo.mp4
```

More recipes: [docs/TRAINING.md](docs/TRAINING.md) · [docs/DEMO.md](docs/DEMO.md)

## Repository layout

```
bevlane/
  model.py             # model zoo v1..v20 (v20 = 7-task METEOR)
  train.py             # DDP multi-task trainer + per-task val metrics
  dataset.py           # multi-task dataset (all GT modalities, ignore-safe)
  convert_dtset.py     # 10-stage per-scene GT factory driver
  extract_gt.py        # image cache + BEV lane GT crops
  render_vector_gt.py  # hybrid raster+vector lane GT (hole fill, guards)
  extract_depth_*.py   # dense metric depth GT (8 cams)
  extract_seg2d.py     # 21-class 2D seg GT (csv-driven taxonomy)
  extract_bbox2d.py    # 10-class 2D box GT (csv-driven taxonomy)
  extract_bev_box.py   # camera-confirmed 3D box GT
  extract_ego.py       # E2E GT from ego-pose (trajectory/steer/accel/brake)
  extract_occ.py       # 3D semantic occupancy GT
  demo_rgbd_bev.py     # 7-task inference demo renderer
  demo_gt_full.py      # all-GT visualisation
  demo_occ_gt.py       # occupancy GT visualisation (top-down + isometric)
autolabel_bev.py       # LiDAR x panoptic BEV accumulation (map frame)
vectorize_bev.py       # raster -> connected polyline vector maps
run_batch.py           # scene-parallel autolabel production
comlops-21cls-autolabel-2504.csv   # 2D seg taxonomy (id, name, colour)
fastlabel_2510_instance.csv        # 2D det taxonomy (id, name, colour)
docs/                  # architecture / data / training / demo docs
```

## Documentation

| Doc | Contents |
|---|---|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | stage-by-stage tensor shapes, params & FLOPs, depth-gated IPM |
| [DATA_PIPELINE.md](docs/DATA_PIPELINE.md) | the 10-stage autolabel factory, taxonomies, quality gates |
| [TRAINING.md](docs/TRAINING.md) | losses, curricula, rolling rounds, operational notes |
| [DEMO.md](docs/DEMO.md) | demo tooling and video layouts |

---

<div align="center">
<sub>METEOR — because it came from CoMET. ☄️<br/>
Built on the CoMET autolabeling foundation of the <a href="https://co-mlops.tier4.jp/"><b>Co-MLOps</b></a> project
· presented at <a href="https://www.nvidia.com/ja-jp/gtc/session-catalog/sessions/gtc26-s81897/">NVIDIA GTC 2026 (S81897)</a>.<br/>
Labels by machines. Code by <b>Claude Fable 5</b>. Direction by humans.</sub>
</div>

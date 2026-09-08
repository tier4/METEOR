# Reproducing METEOR in another environment

This is the operational guide: what the machine needs, how to get the data in,
how a training round is actually run and kept alive, how to build a **lightweight
variant**, and — the part that costs the most time if skipped — how to measure so
the numbers mean something.

`README.md` says what METEOR is. `docs/QUICKSTART.md` is the five-command tour.
This file is what you need when the repository is on a different machine.

---

## 1. What the machine needs

Everything below is what this project actually runs on, not a minimum spec
guess.

| | measured here |
|---|---|
| GPUs | 8 × NVIDIA data-center GPUs, 44 GiB each (Ada, SM 8.9) |
| driver | 580.65.06 |
| Python | 3.10.19 |
| PyTorch | 2.1.1+cu121 (CUDA 12.1, cuDNN 8902) |
| TensorRT | 8.6.0 |
| numpy / opencv / onnx / scipy / shapely | 1.26.4 / 3.4.18 / 1.17.0 / 1.15.3 / 1.8.5 |
| dataset on disk | 953 GB |
| converted training root | 12,595 scene directories (symlinks) |

**Scaling down.** A round uses 7 GPUs at `--batch 2` with `--sync-bn` and leaves
one free for measurement — that separation matters (see §6). On fewer or smaller
GPUs, drop `--batch` to 1 and raise `--val-batch`; the watchdog already does this
automatically after four OOMs. Peak allocation per rank is about 26 GiB at batch
2 and about 6 GiB for a demo render, so 24 GiB cards work at batch 1.

**One GPU is enough to run inference and the demos.** Only training needs the
fleet.

---

## 2. Install

```bash
python3.10 -m venv ~/meteor_venv
~/meteor_venv/bin/pip install torch==2.1.1 torchvision --index-url https://download.pytorch.org/whl/cu121
~/meteor_venv/bin/pip install numpy==1.26.4 opencv-python onnx scipy shapely \
                              pyquaternion pycocotools tensorrt==8.6.0
git clone git@github.com:tier4/METEOR.git && cd METEOR
```

Every command in this repository uses the interpreter path explicitly
(`python3` here). Substitute yours; nothing depends
on the venv being active.

---

## 3. Data

METEOR trains on **t4dataset-format recordings** converted by the label factory
in `bevlane/`. No human annotation is involved at any stage — LiDAR is used
*offline* by the factory to make the labels, and *optionally online* as an extra
input on the same weights.

```bash
# raw recordings -> training GT (11 stages, resumable, scene-parallel)
python3 bevlane/run_factory.py --src /data/t4dataset --out out/bevlane
```

The result is one directory per scene under `out/bevlane/`, each with a
`manifest.json`, an `img/` directory, `ego_motion.npz`, and the per-frame GT
npz files. Scenes may be symlinks — the loader only ever reads them, and 953 GB
is not worth copying.

**Ingesting a new drop** later goes through `bevlane/ingest_newdata.py`, which
validates each candidate scene against seven conditions before accepting it
(manifest parses, has frames, has cameras, `img/` non-empty, `ego_motion.npz`
present, frame 0's image exists on disk, a BEV GT key exists). One drop
contained 41 directories whose manifests parsed but carried no frames at all;
feeding those to the loader costs a stat storm per epoch and silently shrinks the
effective batch.

### The holdout rule is enforced in code, not by convention

`bevlane/train.py` reads `HOLDOUT_FILES = ("test.lst", "out/x2gen2_test.txt",
"out/newdata_test.txt")` and removes every scene in them from the training list,
printing the count at startup:

```
[holdout] N scenes excluded from training (your test list files)
```

Splits are **by recording, never by scene**. Scenes named `<recording>_0`, `_1`,
`_2` are consecutive slices of one drive; splitting them individually puts frames
seconds apart on both sides of the line and the holdout then reports memorisation
as generalisation.

Frames are stored at **5 Hz** (0.200 s apart), collected at 10 Hz with stride 2.
Driving time = frames × 0.2 s. Getting this wrong produced a 46 h estimate for a
92 h corpus.

---

## 4. Smoke test before anything long

```bash
# does the model build, run, and produce the expected 19 outputs?
CUDA_VISIBLE_DEVICES=0 python3 -c "
import torch, sys; sys.path.insert(0,'.')
from bevlane.model import MODELS
m = MODELS['v52'](n_seg=21).cuda().eval()
o = m(torch.randn(1,8,3,432,768).cuda(),
      torch.eye(3)[None,None].repeat(1,8,1,1).cuda(),
      torch.eye(4)[None,None].repeat(1,8,1,1).cuda())
print(len(o), o[0].shape, o[1].shape)"
```

Expect `19 torch.Size([1, 9, 800, 500]) torch.Size([1, 8, 64, 108, 192])`.

Then render one scene to confirm the data path:

```bash
CUDA_VISIBLE_DEVICES=0 python3 bevlane/demo_rgbd_bev.py \
  --model v52 --ckpt <checkpoint>.pt --n-seg2d 21 \
  --root out/bevlane --scenes-file val.lst --frame-stride 8 \
  --show-seg2d --guard --seg-fuse --pseudo-lidar --show-pl \
  --out /tmp/smoke.mp4 --fps 10
```

---

## 5. Running a round

A round is one invocation of `bevlane/train.py` under three layers of
supervision. Copy the r61 set and rename:

```bash
for f in relaunch supervise watchdog; do
  sed 's/r61/r62/g; s/R61/R62/g; s/29561/29562/g' out/${f}_r61.sh > out/${f}_r62.sh
  chmod +x out/${f}_r62.sh
done
bash out/relaunch_r62.sh
crontab -l | grep -v supervise_r6 > /tmp/ct
echo "*/5 * * * * bash $PWD/out/supervise_r62.sh >> $PWD/out/supervise_r62_log.txt 2>&1" >> /tmp/ct
crontab /tmp/ct
```

| layer | what it catches |
|---|---|
| `relaunch_rNN.sh` | flock + pgrep guarded launch; idempotent, safe to run twice |
| `watchdog_rNN.sh` | process death (relaunch) and hang (log stale > 30 min → kill, relaunch) |
| `supervise_rNN.sh` | cron every 5 min; restarts the watchdog itself, and **catches the silent failure below** |

### Uptime is not progress

The failure that cost a whole round: the process was alive, the log was moving,
the watchdog was satisfied — and **23,533 of 34,280 steps (69 %) were being
discarded as non-finite**. 0 % up to step 8k, then 98–100 % for the rest. The
weights simply stopped moving while validation kept printing plausible numbers.
A tell-tale sign is a metric that sits too still: a mIoU that moves by less than
0.003 across six epochs is not training, it is the same model measured six times.

So always check the discard rate over the **whole** log, never an early sample:

```bash
grep -c "SKIP non-finite" out/train_rNN_log.txt
grep -cE "^ep[0-9]+ step"   out/train_rNN_log.txt
```

`supervise_rNN.sh` now trips when more than half of the last 300 step lines were
discarded, repairs the checkpoint, and relaunches.

### The repair, and why it is not a recipe change

The cause is numerical. **A convolution feeding a BatchNorm has a free scale** —
BN divides it straight back out — and nothing in the loss keeps that scale near
1, so it drifts. `seg_head.out.1`'s `running_var` went 4.3e6 → 2.6e7 across two
rounds until the conv's fp16 output passed 65504 and became `inf`. A `clamp` on
the logits cannot help: the overflow is upstream, and `clamp(NaN)` is `NaN`.

```bash
# localise: names the first module whose output goes non-finite on finite input
METEOR_MODPROBE=1 <the training command>
# repair: rescales conv->BN pairs, function-preserving
python3 bevlane/renorm_convbn.py --ckpt out/.../last.pt --out out/.../last.pt
```

Verified function-preserving: BEV mIoU unchanged to four decimals, logit differences at the
fp16 rounding floor. That is why the supervisor is allowed to apply it
automatically — it repairs a numerical failure without changing what is being
optimised.

**Automatic recipe changes are not allowed.** An earlier watchdog dropped the
lane-graph loss after three failed starts; that was removed. A round that quietly
trains a different objective than the one on record is worse than a round that
stays down until the bug is fixed.

---

## 6. Training a lightweight variant

`v54` is `v52` with the two aggressive cuts, both dialable:

```python
class DepthSegIPMNetV54(DepthSegIPMNetV52):
    DEPTH_MULT = 0.5          # width multiplier on the depth tower
    BACKBONE = "resnet18"     # resnet34 -> resnet18
```

Measured against v52:

| | v52 | v54 | |
|---|---|---|---|
| parameters | 54.2 M | **41.7 M** | −23 % |
| depth_head | 3.29 M | 0.92 M | −72 % |
| backbone | 21.28 M | 11.18 M | −47 % |
| depth_head compute | 1088 GFLOP | **303 GFLOP** | −785 GFLOP = **23 % of the whole graph** |

Train it exactly like any other model — only `--model` changes:

```bash
torchrun --nproc_per_node=7 bevlane/train.py --model v54 --batch 2 --sync-bn \
  --val-batch 1 --epochs 8 --workers 4 --gt-key gt_cons --root out/bevlane \
  --train-list out/round59_scenes.txt --limit-train 60000 ...   # rest identical
```

### Why these two, and why it may hold

`depth_head` is **32.0 % of the graph's 3,405 GFLOP** — four stages of 3×3 convs
at 256/256/192/128 run over 8 cameras at 108×192. It is by far the largest single
target, and the per-layer *time* profile hides this: the profile on file made it
look like 5 %, because it was taken with the lane-graph decoder still in the
graph consuming 10.07 ms of 34.

The reason to expect it survives the cut: **`bev_pts` has a single unique
z = 0.0**. The lift is flat-ground; depth never moves a sample point, it only
re-weights which camera wins a BEV cell. That is also why swapping the v50
regression head for the v48/v52 classification head measured neutral on BEV
(mIoU delta within measurement noise over three measurements). A four-stage 256-channel tower to
produce a per-pixel camera weight is very likely more than the job needs.

`resnet34 → resnet18` keeps every channel count identical (64/128/256/512) and
only drops block counts (layer3 6→2, layer4 3→2), so no lateral conv and nothing
downstream changes. layer3+layer4 are 37 % of the parameters.

### Go further

```python
class MyTiny(DepthSegIPMNetV54):
    DEPTH_MULT = 0.25
```

Then register it in `MODELS` and pass `--model mytiny`.

### Two ways to train it

1. **From scratch on the same recipe.** Simplest, and the honest baseline.
2. **Initialise from the full model.** `--init-ckpt` loads with `strict=False`,
   so every layer whose shape still matches (the whole BEV decoder, every head,
   the lateral convs) starts trained; only the depth tower and the backbone
   blocks that changed shape start fresh. This converges far faster than (1) and
   is what the rounds here do.

**Whichever you pick, the acceptance gate is the same.** A latency win that costs
a priority task is not a win — BEV Seg, 3D BBox and E2E must not degrade. Measure
them before and after, in one process, with only `--model` toggled.

---

## 7. Measuring so the numbers mean something

This section is the one that saves the most time. Every rule here was bought
with a wrong conclusion that had to be retracted.

### An A/B is one process, one frame list, one toggled variable

Running each condition as its own process moves the frame set, the class count
(7 vs 8; `marking` as `nan` vs `0.0`), and the temporal state along with the
thing you meant to change — and each of those moves the metric more than the
effect does. Measured the wrong way, feeding the SD-map "cost 19 % of mIoU" and
LiDAR "did nothing". Measured in one process toggling only the kwarg, on the same
map-present val frames: SD-map **+0.3 %**, LiDAR **+7.0 %**. Both original
numbers were wrong, in opposite directions.

```python
for kw in ({}, {"sdmap": S}, {"lidar_bev": L}):     # same frame, back to back
    out = model(imgs, K, T, **kw)
```

Restrict to frames where the optional raster is actually non-zero: the flag gate
in `bev_extra` makes a zero input bit-identical to no input, so absent-map frames
only dilute the effect. And accumulate intersection/union **globally** — per-frame
averaged mIoU scores absent classes 0/0 and is far noisier than the effect.

### Reproduce the training metric before trusting your own probe

An offline probe that "could not reproduce" the training ADE — 47 % higher than
the logged value — differed by one dataset argument: `trim_end=10`. Scene ends lack
accumulated LiDAR ahead and behind, so the GT there is weak and training drops
them. With it, the same function on the same weights reproduces **the logged value to
three decimals**.

If you write a probe, build the dataset with the *same* flags the training run
used (they are saved in the checkpoint):

```python
torch.load(ckpt, map_location="cpu")["args"]   # trim_end, val_batch, gt_key, ...
```

### Report the decomposition, not one number

`[valE2Ed]` prints, every epoch:

```
oracle=<best-of-K> (選択ロス +<gap>) 走行=<moving> 停車=<stopped> 等速直進=<CV baseline>
```

- **oracle** — best of the K=3 hypotheses. The gap to ADE is what mode
  *selection* costs; on one round that was about a fifth of the error.
- **走行 / 停車** — moving (v0 > 2 m/s) and stopped. A quarter of val frames are
  stationary and they are a different problem.
- **等速直進** — constant velocity, the do-nothing baseline. On one round the model beat
  it by less than a quarter. Without this column an ADE looks better than it is.

### Time an engine on an idle GPU

A TensorRT engine timed 34.7 ms while a renderer shared the card; alone it was
19.24. Queue benchmarks behind renders, never beside them.

---

## 8. Deployment — the verified release path

Verified 2026-09-08 on a workstation (one data-center NVIDIA GPU shared with a training job, driver
580.65, CUDA 12.6, `tensorrt` 8.6.0 from pip, `onnxruntime` 1.23.2, `pycuda`) with the
release artefacts exactly as published:

```bash
pip install tensorrt==8.6.* pycuda onnxruntime opencv-python numpy huggingface_hub
hf download AutowareFoundation/meteor meteor_v157c3Z.onnx --local-dir meteor
hf download AutowareFoundation/meteor-demo-scenes --repo-type dataset --include "valday/*" --local-dir demo

# 1. does the ONNX run at all?  (CPU, ~4 s)
python3 hf/onnx_smoke_test.py --onnx meteor/meteor_v157c3Z.onnx --root demo/valday --frame 40

# 2. plain fp16 TensorRT engine: no plugin, no calibration (~10 min, 194 MB)
python3 deploy/build_engine_fp16.py meteor/meteor_v157c3Z.onnx out/meteor_v157c3Z_fp16.engine 8

# 3. the same real-time demo that runs on the Orin, on the workstation
METEOR_TH2D=0.30 METEOR_SEG2D_OVERLAY=0 METEOR_OCC_PANEL=0 METEOR_2D_HIDE=7 PYTHONPATH=. \
python3 deploy/orin_realtime.py --engine out/meteor_v157c3Z_fp16.engine \
    --root demo/valday --limit 120 --out out/demo_valday.mp4
```

Measured on that run: 120 frames, inference 30 ms (fp16, GPU shared), rendering 149 ms
on the CPU, 12 FPS pipelined; the video shows all heads (2D boxes, depth, BEV lanes,
3D boxes, ego path, risk field). `pycuda` prints a "context stack was not empty" message
at process exit — harmless, the video is complete.

What the released ONNX is: `deploy/export_onnx.py` on `meteor_v157.pt`
with the flags listed in `deploy/README.md` §1 (`--uint8-in --argmax-out --lane-logits
--seg-bias … --no-hist --depth-mean`, retired heads dropped). Re-exporting from the
checkpoint reproduces it weight-for-weight.

For the Orin INT8 engine follow `deploy/README.md` §6–§7: insert the lift plugin
(`make_plugin_onnx.py`), build **on the device** with `orin_build_int8.py` (real-frame
calibration), run the health checks. Engine plans are specific to the device and the
TensorRT version; ship ONNX, not plans, to anyone on other hardware.

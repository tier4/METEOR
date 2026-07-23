#!/usr/bin/env python3
"""METEOR v26 model-architecture deck -> out/METEOR_model_architecture.pptx.

Structure-focused: every module, tensor shape, and design decision of the
deployed network, with measured parameter budgets."""
import os
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_AUTO_SIZE, PP_ALIGN
from pptx.util import Emu, Inches, Pt

SW, SH = Inches(13.333), Inches(7.5)
DARK = RGBColor(0x20, 0x28, 0x30)
ACC = RGBColor(0x0E, 0x6E, 0xB8)
GRAY = RGBColor(0x60, 0x68, 0x70)
GREEN = RGBColor(0x1B, 0x78, 0x37)
RED = RGBColor(0xC0, 0x39, 0x2B)
AMBER = RGBColor(0xB9, 0x77, 0x0E)
PURPLE = RGBColor(0x6A, 0x3D, 0x9A)
BG1 = RGBColor(0xF2, 0xF4, 0xF6)
prs = Presentation()
prs.slide_width, prs.slide_height = SW, SH
BLANK = prs.slide_layouts[6]


def slide(title=None, sub=None):
    s = prs.slides.add_slide(BLANK)
    if title:
        tb = s.shapes.add_textbox(Inches(0.5), Inches(0.22), Inches(12.3), Inches(0.7))
        p = tb.text_frame.paragraphs[0]
        p.text = title
        p.font.size = Pt(27)
        p.font.bold = True
        p.font.color.rgb = DARK
        ln = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.5), Inches(0.95),
                                Inches(12.3), Emu(1))
        ln.fill.solid()
        ln.fill.fore_color.rgb = ACC
        ln.line.fill.background()
        if sub:
            tb2 = s.shapes.add_textbox(Inches(0.5), Inches(1.0), Inches(12.3), Inches(0.4))
            p2 = tb2.text_frame.paragraphs[0]
            p2.text = sub
            p2.font.size = Pt(14)
            p2.font.italic = True
            p2.font.color.rgb = GRAY
    return s


def bullets(s, items, x=0.6, y=1.5, w=12.1, h=5.6, size=17):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.NONE
    for i, it in enumerate(items):
        lvl, txt, col, sz = 0, "", None, None
        stack = list(it) if isinstance(it, tuple) else [it]
        while stack:
            e = stack.pop(0)
            if isinstance(e, RGBColor):   # RGBColor subclasses tuple: match first
                col = e
            elif isinstance(e, str):
                txt = e
            elif isinstance(e, tuple):
                stack = list(e) + stack
            elif isinstance(e, int):
                if e in (0, 1):
                    lvl = e
                else:
                    sz = e
        base = size if lvl == 0 else size - 3
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = ("• " if lvl == 0 else "   – ") + txt
        p.font.size = Pt(sz if sz else base)
        p.font.color.rgb = col if col else DARK
        p.font.bold = sz is not None and lvl == 0
        p.space_after = Pt(6)


def big(s, lines, y=2.4):
    tb = s.shapes.add_textbox(Inches(0.8), Inches(y), Inches(11.7), Inches(3.2))
    tf = tb.text_frame
    tf.word_wrap = True
    for i, (txt, sz, bold, col) in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = txt
        p.font.size = Pt(sz)
        p.font.bold = bold
        p.font.color.rgb = col
        p.alignment = PP_ALIGN.CENTER
        p.space_after = Pt(10)


def table(s, rows, x, y, w, col_w, fs=13, hdr_fill=RGBColor(0xC9, 0xD9, 0xEA)):
    from pptx.util import Inches as In
    nr, nc = len(rows), len(rows[0])
    gt = s.shapes.add_table(nr, nc, In(x), In(y), In(w), In(0.34 * nr)).table
    for ci, cw in enumerate(col_w):
        gt.columns[ci].width = In(cw)
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row):
            c = gt.cell(ri, ci)
            c.text = str(val)
            pr = c.text_frame.paragraphs[0]
            pr.font.size = Pt(fs)
            pr.font.bold = ri == 0
            pr.font.color.rgb = DARK
            if ri == 0:
                c.fill.solid()
                c.fill.fore_color.rgb = hdr_fill
    return gt


def box(s, x, y, w, h, head, headcol, lines, fill=None, fs_head=13, fs=11):
    b = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y),
                           Inches(w), Inches(h))
    b.fill.solid()
    b.fill.fore_color.rgb = fill if fill else BG1
    b.line.color.rgb = headcol
    b.line.width = Pt(1.5)
    tf = b.text_frame
    tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.NONE
    tf.margin_left = tf.margin_right = Inches(0.08)
    tf.margin_top = Inches(0.04)
    p = tf.paragraphs[0]
    p.text = head
    p.font.size = Pt(fs_head)
    p.font.bold = True
    p.font.color.rgb = headcol
    p.alignment = PP_ALIGN.CENTER
    for ln in lines:
        pp = tf.add_paragraph()
        pp.text = ln
        pp.font.size = Pt(fs)
        pp.font.color.rgb = DARK
        pp.alignment = PP_ALIGN.CENTER
        pp.space_before = Pt(1)
    return b


def arrow(s, x1, y1, x2, y2, col=GRAY, wpt=2.2):
    c = s.shapes.add_connector(2, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = col
    c.line.width = Pt(wpt)
    return c


def note(s, txt, y=6.85, col=GRAY, size=12):
    tb = s.shapes.add_textbox(Inches(0.55), Inches(y), Inches(12.2), Inches(0.4))
    p = tb.text_frame.paragraphs[0]
    p.text = txt
    p.font.size = Pt(size)
    p.font.italic = True
    p.font.color.rgb = col


# ---------- 1 title ----------
s = slide()
big(s, [
    ("METEOR", 60, True, ACC),
    ("Model Architecture (v40)", 32, True, DARK),
    ("Multi-task Estimation of Traffic Elements, Objects & Roads", 18, False, GRAY),
    ("One network - 8 cameras - 10+ tasks - 53M params (base 48M + grafted "
     "Refiner 5M) - TensorRT-safe ops only", 15, False, DARK),
], y=2.0)
note(s, "All architecture, code and labels produced autonomously by Claude Fable 5 "
        "on the CoMET / Co-MLOps autolabeling foundation", y=6.6)

# ---------- 2 overview diagram (image) ----------
s = slide("End-to-end overview", "8 surround cameras -> shared BEV -> task-routed heads")
s.shapes.add_picture("docs/media/architecture.png", Inches(0.55), Inches(1.55),
                     width=Inches(12.2))

# ---------- 3 dataflow block diagram (shapes) ----------
s = slide("Dataflow and task routing", "geometry tasks read the RAW BEV; motion tasks read the temporally FUSED BEV")
box(s, 0.5, 1.7, 1.75, 1.5, "8 cameras", DARK,
    ["768x432 RGB", "surround +", "2x narrow"])
box(s, 2.65, 1.7, 2.1, 1.5, "Image encoder", ACC,
    ["ResNet-34 + FPN", "shared feature", "s4, 160ch"])
box(s, 5.15, 1.45, 2.0, 0.62, "Depth head", PURPLE, ["64 bins @ s2"], fs=10)
box(s, 5.15, 2.16, 2.0, 0.62, "Context 1x1", PURPLE, ["96ch"], fs=10)
box(s, 5.15, 2.87, 2.0, 0.62, "2D seg / 2D det", PURPLE, ["aux heads on s4"], fs=10)
box(s, 7.55, 1.7, 2.15, 1.5, "RAW BEV (frame t)", GREEN,
    ["depth-weighted splat", "96 x 800 x 500", "0.2 m cells, 90 m"])
box(s, 10.2, 1.7, 2.55, 1.5, "FUSED BEV (t + t-0.4s)", AMBER,
    ["prev-frame BEV warped", "by ego-motion +", "residual tfuse"])
arrow(s, 2.25, 2.45, 2.65, 2.45)
arrow(s, 4.75, 2.45, 5.15, 2.45)
arrow(s, 7.15, 2.45, 7.55, 2.45)
arrow(s, 9.7, 2.45, 10.2, 2.45)
# geometry group: directly under RAW BEV
box(s, 2.6, 4.5, 1.9, 1.15, "BEV lanes", GREEN, ["LaneDecED", "9cls 800x500"], fs=10)
box(s, 4.6, 4.5, 1.9, 1.15, "3D detection", GREEN, ["CenterPoint hm+reg", "2cls 400x250"], fs=10)
box(s, 6.6, 4.5, 1.9, 1.15, "Occupancy", GREEN, ["10cls x 16z", "200x200 @0.4m"], fs=10)
arrow(s, 8.3, 3.2, 3.55, 4.5, GREEN)
arrow(s, 8.3, 3.2, 5.55, 4.5, GREEN)
arrow(s, 8.3, 3.2, 7.55, 4.5, GREEN)
# motion group: directly under FUSED BEV (temporal tasks)
box(s, 8.9, 4.5, 1.7, 1.15, "E2E ego", AMBER, ["6 waypoints", "steer/acc/brake"], fs=10)
box(s, 10.7, 4.5, 2.1, 1.15, "Agent traj + stationary", AMBER,
    ["12ch traj map", "+ stat logit (v26)"], fs=10)
arrow(s, 11.45, 3.2, 9.75, 4.5, AMBER)
arrow(s, 11.45, 3.2, 11.75, 4.5, AMBER)
# group captions
cap = s.shapes.add_textbox(Inches(2.6), Inches(5.7), Inches(5.9), Inches(0.35))
p = cap.text_frame.paragraphs[0]
p.text = "geometry group - single frame (RAW BEV)"
p.font.size = Pt(12); p.font.bold = True; p.font.color.rgb = GREEN
p.alignment = PP_ALIGN.CENTER
cap2 = s.shapes.add_textbox(Inches(8.9), Inches(5.7), Inches(3.9), Inches(0.35))
p2 = cap2.text_frame.paragraphs[0]
p2.text = "motion group - temporal (FUSED BEV)"
p2.font.size = Pt(12); p2.font.bold = True; p2.font.color.rgb = AMBER
p2.alignment = PP_ALIGN.CENTER
tbx = s.shapes.add_textbox(Inches(1.0), Inches(6.15), Inches(11.5), Inches(0.9))
p = tbx.text_frame.paragraphs[0]
p.text = ("E2E and agent forecasting are TEMPORAL: they read the fused BEV (current + previous frame), "
          "which carries velocity. Lanes / 3D boxes / occupancy are single-frame: temporal warp smears "
          "static geometry with moving-object ghosts, so they read the RAW BEV.")
p.font.size = Pt(13)
p.font.color.rgb = DARK
tbx.text_frame.word_wrap = True

# ---------- 4 image encoder ----------
s = slide("Image encoder + auxiliary 2D heads", "one shared backbone shapes all views; 21.7M + 10.3M aux")
bullets(s, [
    ("Backbone: ResNet-34 (ImageNet init) over 8 views as a batch - stem + layer1..4", 0),
    ("FPN-style fusion: 1x1 laterals from s4/s8/s16/s32, all upsampled to s4, 160ch (21.7M)", 1),
    ("Depth head (3.3M): 64 linear bins from 1.0m step 1.25m, predicted at stride 2 - "
     "supervised by accumulated-LiDAR dense depth GT", 0),
    ("2D semantic segmentation head (4.2M): SegHeadED encoder-decoder, 21 classes at s4 - "
     "auxiliary supervision that shapes the shared feature", 0),
    ("2D detection: YOLO-like 3-scale pyramid (2.9M) on the shared s4 feature", 0),
    ("s4: small objects (cones, traffic lights) / s8: mid / s16: large; GT routed by max(w,h) @ (40px, 120px)", 1),
    ("Context 1x1 -> 96ch: the only image feature lifted into BEV", 0),
], size=16)
note(s, "All aux heads cost nothing at BEV inference time if pruned; kept for deployment-side 2D outputs")

# ---------- 5 BEV projection ----------
s = slide("Depth-weighted BEV projection", "geometrically exact splat, TensorRT-safe (grid_sample + gather only)")
bullets(s, [
    ("BEV grid: 800 x 500 cells @ 0.2 m -> +-80 m forward, +-50 m lateral", 0),
    ("Each BEV cell projects into every camera (K, T_cam_ego); features are sampled "
     "by grid_sample at the projected pixel", 0),
    ("Sampled context is weighted by the predicted depth probability at the cell's "
     "true camera distance (linear interpolation between the two nearest depth bins) + 0.05 floor", 0),
    ("Multi-camera fusion: depth-probability-weighted average across the 8 views "
     "(overlap regions resolve automatically)", 0),
    ("Validity: z > 0.5 m, in-image, distance < 90 m", 1),
    ("No voxel pooling / no scatter ops -> static graph, exports to TensorRT directly", 0),
], size=16)

# ---------- 6 temporal ----------
s = slide("Streaming temporal BEV (v22/v23)", "one previous frame, warped and fused - recurrent at deploy time, static graph")
bullets(s, [
    ("Previous frame's RAW BEV is warped into the current ego frame by an affine "
     "theta built from relative pose (tx, ty, dyaw): affine_grid + grid_sample", 0),
    ("Residual fusion: bev + tfuse(concat(bev, warped)) with 1x1 192->96 + ConvBlock (0.18M)", 0),
    ("Last BN of tfuse zero-initialised -> fusion starts as identity and cannot perturb "
     "the warm-started BEV (fixed the v22 mIoU dip)", 0),
    ("Deployment: prev_bev and warp_theta are graph INPUTS, raw current BEV is an extra "
     "OUTPUT -> feed the engine its own output (streaming), no RNN in the graph", 0),
    ("Training: prev-frame BEV computed in a separate no_grad + autocast region "
     "(inside the main autocast region it caches detached fp16 weights and kills backbone grads)", 0),
], size=16)

# ---------- 7 lane decoder ----------
s = slide("BEV lane decoder - LaneDecED", "4.1M params at ~0.8x the FLOPs of the flat stack it replaced")
bullets(s, [
    ("Encoder-decoder over the RAW BEV (96 x 800 x 500):", 0),
    ("full-res skip: ConvBlock 96->64 (thin structures: lane lines, stop lines)", 1),
    ("down s2 -> 192ch ConvBlock; down s4 -> 320ch ConvBlock (context: road topology)", 1),
    ("up path with additive skips; head 64 -> 9 classes at full 800x500", 1),
    ("Design rule: parameters at low resolution, compute at high resolution", 0),
    ("Losses: weighted CE + dice (crosswalk) + Tversky beta=0.8 (line classes, precision) "
     "+ Lovasz-softmax (IoU) + boundary weighting + far-range weighting", 0),
], size=16)

# ---------- 8 3D det ----------
s = slide("BEV 3D detection head", "CenterPoint-style, near-range-first supervision (r15)")
bullets(s, [
    ("Det stem on RAW BEV: s2 96->128 + s4 tower 256ch fused back at s2 (_DetStemED, 2.1M)", 0),
    ("hm [2 x 400 x 250]: vehicle / VRU center heatmaps - penalty-reduced focal, bias init -2.19", 0),
    ("reg [6 x 400 x 250]: (off_r, off_c, log l, log w, sin yaw, cos yaw) - L1 at GT centers", 0),
    ("Near-range-first positive weighting (this round):", 0, RED),
    ("distance boost: x3 inside 12 m, x2 inside 20 m; class boost: veh x2, VRU x5", 1),
    ("laterally distant (|y| > 15 m): x0.2 - out of scope by requirement", 1),
    ("far damp: veh > 50 m x0.3, VRU > 40 m x0.2 (unresolvable at 768x432 - "
     "full-weight unlearnable positives suppress confidence everywhere)", 1),
    ("Gaussian target radius floor raised 1.5 -> 2.0 cells; VRU camera-confirmation "
     "threshold relaxed 0.3 -> 0.15 (GT recall)", 1),
    ("Decode: 3x3 max-pool NMS + topk, offsets refine centers", 0),
], size=15)
note(s, "Metrics: [val3D] P/R + R50 (<50 m) + Rn (<30 m, |lat|<12 m) - Rn is the no-miss corridor target")

# ---------- 9 traj + stationary ----------
s = slide("Agent forecasting + stationary flag (v26)", "one-shot, constant cost in the number of agents")
bullets(s, [
    ("traj_stem (0.4M): s2 conv 96->128 + ConvBlock on the FUSED BEV "
     "(motion needs the temporal feature; det stem stays raw)", 0),
    ("traj_head 1x1 -> 12ch: at every det cell, 6 future offsets (0.5..3.0 s, metres, ego frame)", 0),
    ("masked L1 at GT box centers; read out at detected centers -> per-agent forecast "
     "with zero per-agent cost", 1),
    ("stat_head 1x1 -> 1ch (v26, +0.0003M): explicit stationary logit", 0, RED),
    ("label derived on-the-fly from existing trajectory GT: |displacement @3s| < 0.5 m "
     "(32% of valid boxes) - no new extraction stage", 1),
    ("BCE at GT centers; replaces the forecast-threshold heuristic for parked/stopped "
     "coloring in the demo; measured as statAcc in [valTraj]", 1),
], size=16)

# ---------- 10 E2E + OCC ----------
s = slide("E2E ego head + occupancy head")
bullets(s, [
    ("E2E ego (4.0M) - reads the FUSED BEV:", 0, AMBER),
    ("conv pyramid s4->s8->s16->s32 (128->256ch) + global average pool", 1),
    ("MLP [256+1 -> 512 -> 512 -> 256 -> out] conditioned on current speed v0", 1),
    ("outputs: 6 waypoints @0.5 s + steer + accel + brake logit", 1),
    ("curvature-weighted loss: lateral L1 x4, per-sample weight 1 + min(|lat@3s|, 6)/1.5", 1),
    ("Occupancy (0.7M) - reads the RAW BEV:", 0, GREEN),
    ("crop +-40 x +-40 m -> s2 stem 128->192 -> 1x1 to 160ch -> reshape [10cls x 16z x 200 x 200] @0.4 m voxels", 1),
    ("class-weighted CE, ignore 255: free x0.2, vehicle x2, obstacle x3, 2-wheeler/pedestrian x4", 1),
], size=16)

# ---------- 11 parameter budget ----------
s = slide("Parameter budget (base 48M; v40 with grafted Refiner = 53M)")
table(s, [
    ["Component", "Params", "Input", "Notes"],
    ["Image encoder (ResNet-34 + FPN)", "21.67M", "8 x 768x432", "shared by all tasks"],
    ["Depth head (64 bins)", "3.29M", "s4 feature", "aux + splat weights"],
    ["2D seg head (21cls SegHeadED)", "4.18M", "s4 feature", "aux"],
    ["2D det 3-scale pyramid", "2.85M", "s4 feature", "aux"],
    ["Context 1x1 -> BEV", "0.02M", "s4 feature", "96ch lifted"],
    ["Temporal fuse (tfuse)", "0.18M", "raw+warped BEV", "zero-init BN"],
    ["BEV lane decoder (LaneDecED)", "4.10M", "RAW BEV", "9cls @800x500"],
    ["3D det stem + heads", "2.06M", "RAW BEV", "2cls @400x250"],
    ["Occupancy head", "0.70M", "RAW BEV", "10cls x 16z @0.4m"],
    ["E2E ego head", "4.04M", "FUSED BEV", "wp + steer/acc/brake"],
    ["Agent traj stem + head", "0.41M", "FUSED BEV", "12ch forecast map"],
    ["Stationary head (v26)", "129 params", "det feature", "1ch BCE"],
], 1.2, 1.45, 10.9, [4.3, 1.3, 2.2, 3.1], fs=12)

# ---------- 12 outputs ----------
s = slide("Forward signature and outputs", "static 11-output graph; every op TensorRT-compatible")
bullets(s, [
    ("forward(imgs [B,8,3,432,768], K, T_cam_ego, v0, prev_bev, warp_theta) ->", 0, ACC),
], size=16, y=1.5, h=0.6)
table(s, [
    ["#", "Output", "Shape", "Task"],
    ["0", "lane logits", "[B, 9, 800, 500]", "BEV lane segmentation"],
    ["1", "depth logits", "[B*8, 64, 216, 384]", "aux depth"],
    ["2", "seg2d logits", "[B, 8, 21, 108, 192]", "2D segmentation"],
    ["3/4", "hm / reg", "[B,2,400,250] / [B,6,400,250]", "BEV 3D detection"],
    ["5/6", "hm2d / reg2d", "3 scales (s4/s8/s16)", "2D detection"],
    ["7", "ego", "[B, 15]", "E2E waypoints + controls"],
    ["8", "occupancy", "[B, 10, 16, 200, 200]", "3D semantic occupancy"],
    ["9", "agent traj", "[B, 12, 400, 250]", "agent forecasting"],
    ["10", "stationary", "[B, 1, 400, 250]", "parked/stopped flag (v26)"],
], 0.9, 2.2, 11.5, [0.7, 2.4, 4.3, 4.1], fs=13)
note(s, "conv / grid_sample / affine_grid / gather / max_pool only - no NMS, no scatter, no dynamic shapes in the graph")

# ---------- 13 lineage ----------
s = slide("Version lineage (how the structure grew)")
table(s, [
    ["Ver", "Added structure", "Params"],
    ["v13-v15", "depth-gated splat, 2D seg aux, s4 backbone fusion", "~29M"],
    ["v16", "CenterPoint BEV 3D det head (hm + reg on s2 stem)", "+0.4M"],
    ["v17", "per-camera 2D det head", "+0.5M"],
    ["v18", "E2E ego head (v0-conditioned)", "+0.4M"],
    ["v19", "capacity re-balance: SegHeadED, 3-scale 2D det, big ego head", "+11M"],
    ["v20", "3D semantic occupancy head", "+0.7M"],
    ["v21", "one-shot agent trajectory head", "+0.002M"],
    ["v22", "streaming temporal BEV (warp + residual tfuse)", "+0.19M"],
    ["v23", "LaneDecED lane decoder, s4 det tower, tfuse zero-init", "+4.9M"],
    ["v24/25", "task routing: geometry on RAW BEV, motion on FUSED BEV", "+0.4M"],
    ["v26/27", "stationary-flag head; traffic-light head", "+0.3M"],
    ["v28-30", "risk field; 3-slot temporal queue; motion-residual forecast", "+1.5M"],
    ["v32-34", "optional LiDAR pillar branch; unknown-obstacle head", "+1.0M"],
    ["v36-39", "E2E: intent, in-training risk selection, decoupled head", "+2M"],
    ["v40", "multi-task Refiner grafted on as trainable post-heads", "+5M"],
], 1.6, 1.5, 10.1, [1.1, 7.4, 1.6], fs=12)
note(s, "Every version validated against the previous on the same held-out scenes before adoption")

# ---------- 14 refiner (v40, method A) ----------
s = slide("Post-hoc Refiner (v40) — residual heads on the outputs",
          "seg / 3D box / E2E / agent-traj / risk-map; each zero-init residual")
if os.path.exists("docs/media/refiner_arch.png"):
    s.shapes.add_picture("docs/media/refiner_arch.png", Inches(0.7),
                         Inches(1.5), width=Inches(8.0))
bullets(s, [
    ("residual + zero-init", 0, GREEN),
    (1, "= identity at start; can only improve the base, never degrade it."),
    ("5 heads, +5M params", 0, ACC),
    (1, "seg U-Net (far-range fill), box U-Net (peak/size), E2E MLP, and NEW"),
    (1, "agent-traj U-Net + risk-map U-Net."),
    ("Method A graft", 0, ACC),
    (1, "grafted into v40 (53M) and fine-tuned end-to-end from base + refiner"),
    (1, "weights; deployment needs no separate engine."),
], x=9.0, y=1.6, w=3.9, size=13)

# ---------- 15 data source ----------
s = slide("Data & auto-label foundation")
bullets(s, [
    ("Data: Co-MLOps driving data.", 0, ACC),
    ("Auto-label platform: CoMET.", 0, ACC),
    (1, "LiDAR-accumulated, geometry-consistent, consensus ground truth."),
    (1, "Zero human annotation — 7,147 scenes / ~1.05M frames / ~60 h so far."),
    ("Roadmap:", 0, ACC),
    (1, "NVIDIA Cosmos-generated data for robustness."),
    (1, "Feature-focused optimisation -> NVIDIA Orin SoC & Renesas R-Car Gen5."),
    (1, "Release as a Reference AI (open source)."),
], size=16)

os.makedirs("out", exist_ok=True)
prs.save("out/METEOR_model_architecture.pptx")
print("saved out/METEOR_model_architecture.pptx,", len(prs.slides.__iter__.__self__._sldIdLst), "slides")

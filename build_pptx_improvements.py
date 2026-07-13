#!/usr/bin/env python3
"""METEOR improvements deck (English, ~12 slides) -> out/METEOR_improvements.pptx.

Documents each measurement-driven optimization: the observed failure, the
fix, and the quantified effect."""
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
        # type-driven parse (mixed conventions): str=text, RGBColor=colour,
        # int 0/1 = level, other int = explicit size; nested tuples flatten
        lvl, txt, col, sz = 0, "", None, None
        stack = list(it) if isinstance(it, tuple) else [it]
        while stack:
            e = stack.pop(0)
            if isinstance(e, RGBColor):     # RGBColor subclasses tuple/bytes:
                col = e                     # must be matched before tuple below
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
    tb = s.shapes.add_textbox(Inches(0.8), Inches(y), Inches(11.7), Inches(3.0))
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


def card(s, x, y, w, h, head, headcol, lines):
    b = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y),
                           Inches(w), Inches(h))
    b.fill.solid()
    b.fill.fore_color.rgb = BG1
    b.line.color.rgb = headcol
    b.line.width = Pt(1.5)
    tf = b.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.12)
    p = tf.paragraphs[0]
    p.text = head
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = headcol
    for ln, cc in lines:
        pp = tf.add_paragraph()
        pp.text = ln
        pp.font.size = Pt(11)
        pp.font.color.rgb = cc
        pp.space_before = Pt(2)


# ---------- 1 title ----------
s = slide()
big(s, [
    ("☄ METEOR — Optimization Log", 40, True, DARK),
    ("Every improvement started as a measured failure", 22, False, GRAY),
], y=2.3)
big(s, [
    ("12 rounds · 4 new task heads · temporal fusion — in one week", 18, True, ACC),
    ("Dan Umeda — TIER IV, Inc.   ·   built on CoMET / Co-MLOps", 14, False, GRAY),
], y=4.6)

# ---------- 2 overview timeline ----------
s = slide("The improvement journey", "each round diagnosed one weakness and fixed it, warm-starting the next")
table(s, [
    ["Round", "Model", "Improvement", "Key metric", "Result"],
    ["r5", "v17", "+ 10-class 2D detection", "BEV mIoU", "0.286"],
    ["r6", "v16", "lane / ego-vehicle GT fix", "2D-seg lane IoU", "0.371"],
    ["r7", "v18", "+ E2E driving head", "traj ADE", "0.95 m"],
    ["r8", "v19", "head capacity re-balance + KMAX 96", "BEV mIoU / 2D-seg", "0.293 / 0.450"],
    ["r9", "v20", "+ occupancy, E2E curvature weighting", "curve ADE", "2.31 → 0.72 m"],
    ["r10", "v20", "small-object / near-VRU / side-cam weights", "2D-seg lane", "0.548"],
    ["r11", "v21", "+ one-shot agent trajectory forecast", "agent ADE", "2.17 m"],
    ["r12", "v22", "temporal BEV fusion (streaming, TRT-safe)", "curve ADE (ep0)", "2.28 → 1.91 m"],
], 0.55, 1.55, 12.2, [0.9, 0.9, 5.0, 2.6, 2.2], fs=13)
note = s.shapes.add_textbox(Inches(0.55), Inches(6.55), Inches(12.2), Inches(0.4))
np_ = note.text_frame.paragraphs[0]
np_.text = "warm-started throughout — a class-count change retrains only the affected head; nothing restarts from scratch"
np_.font.size = Pt(12)
np_.font.italic = True
np_.font.color.rgb = GRAY

# ---------- 3 capacity re-balance ----------
s = slide("① Head capacity: parameters at low resolution, compute at high",
          "symptom — 2D heads were 'high FLOPs, low params': the worst trade-off")
card(s, 0.55, 1.6, 3.9, 2.0, "PROBLEM", RED, [
    ("3×3 convs at stride-4 (full res)", DARK),
    ("weak 2D seg (mIoU 0.362) and", DARK),
    ("small classes never detected", DARK)])
card(s, 4.7, 1.6, 4.0, 2.0, "FIX", ACC, [
    ("encoder–decoder seg head", DARK),
    ("(heavy channels at s8/s16)", GRAY),
    ("3-scale YOLO-style detection", DARK),
    ("1×1 laterals only at s4", GRAY)])
card(s, 8.9, 1.6, 3.9, 2.0, "EFFECT", GREEN, [
    ("2D-seg mIoU 0.362 → 0.450", GREEN),
    ("pole IoU 0.251 → 0.449", GREEN),
    ("+18× params for +2× FLOPs", DARK),
    ("model +37% params / +10% FLOPs", GRAY)])
bullets(s, [
    "Design rule: put channels where pixels are cheap (s8/s16), touch full resolution only through 1×1",
    "Traffic lights, distant signs and cones — previously silent classes — began firing",
], y=4.0, size=16)

# ---------- 4 small-object GT ----------
s = slide("② Small objects were silently deleted by the GT pipeline",
          "symptom — cones / traffic lights / distant unknowns almost never detected")
bullets(s, [
    ("Root cause 1 — box budget truncation", 18),
    (1, "per-camera cap of 32, area-sorted → dropped 29% of annotations in crowded scenes (all small)"),
    (1, ("fix: KMAX 32 → 96  +  per-class positive weights (obstacle ×3, two-wheeler/light ×2)", GREEN)),
    ("Root cause 2 — thin classes vanish on downsample", 18),
    (1, "a lane line is ~0.3 px wide at stride-4 GT; nearest-neighbour resize deletes it"),
    (1, ("fix: coverage-based rasterisation (cell = thin class if >12% covered)  →  +49% lane pixels", GREEN)),
    ("Effect: 2D-seg lane IoU 0.371 → 0.548; obstacle / light / sign classes detected in demos", 17),
], y=1.55, size=16)

# ---------- 5 E2E curvature ----------
s = slide("③ The E2E head learned to 'always go straight'",
          "symptom — planning looked fine on average, failed on every curve")
card(s, 0.55, 1.6, 4.0, 2.3, "DIAGNOSIS", AMBER, [
    ("83% of frames are near-straight", DARK),
    ("longitudinal targets ~18× larger", DARK),
    ("than lateral → plain L1 ignores", DARK),
    ("steering signal", DARK),
    ("aggregate ADE 0.86 m HID it", RED)])
card(s, 4.75, 1.6, 4.0, 2.3, "FIX", ACC, [
    ("lateral error weight ×4", DARK),
    ("per-sample curve weight", DARK),
    ("1 + |lat@3s| / 1.5  (up to ×5)", GRAY),
    ("new metric: curve-only ADEc", DARK),
    ("reported every epoch", GRAY)])
card(s, 8.95, 1.6, 3.85, 2.3, "EFFECT", GREEN, [
    ("curve ADE 2.31 m (start)", DARK),
    ("→ 1.07 m (1 round)", GREEN),
    ("→ 0.72 m (2 rounds)", GREEN),
    ("straight frames unchanged", DARK),
    ("(0.62 m)", GRAY)])
bullets(s, [
    "Lesson: measure per-condition. The failure was invisible until we conditioned the metric on curvature.",
], y=4.3, size=17)

# ---------- 6 GT bug fixes ----------
s = slide("④ Ground-truth bugs found by watching the demos")
bullets(s, [
    ("Scene-tail speed bug", 18),
    (1, "instantaneous v0 left at 0 past the 3 s-future cutoff → model, conditioned on a bogus 0, predicted"),
    (1, ("'HOLD' at 80 km/h. Fix: write v0/steer/accel for every frame; only trajectory carries a validity mask", GREEN)),
    ("Ego-vehicle unsupervised", 18),
    (1, ("the hood was 'ignore' → free-running noise. Fix: supervise ego as background → clean hood region", GREEN)),
    ("Best-checkpoint criterion", 18),
    (1, ("'best' was BEV-mIoU only → a pre-curve-fix epoch got inherited. Fix: composite score (mIoU − 0.01·ADEc)", GREEN)),
    ("Occupancy hallucination", 18),
    (1, ("coarse 1.7 m ray-carving left above-road voxels 'unknown' → confident phantom structure.", DARK)),
    (1, ("Fix: 0.4 m dense carving + per-class confidence gates in visualisation", GREEN)),
], y=1.5, size=15)

# ---------- 7 new tasks: occupancy + agents ----------
s = slide("⑤ Two new tasks added at negligible cost")
card(s, 0.6, 1.7, 5.9, 3.4, "3D semantic occupancy  (v20)", ACC, [
    ("10-class voxels, 16 × 200 × 200 @ 0.4 m", DARK),
    ("GT: multi-sweep labeled LiDAR (sampled from", GRAY),
    ("cached 2D-seg, no RLE decode — 690→45 s/scene)", GRAY),
    ("dynamic-object de-smearing (±1 frame only)", GRAY),
    ("0.4 m ray-carved free space", GRAY),
    ("cost: +0.70 M params / +55 GFLOPs (2%)", GREEN)])
card(s, 6.8, 1.7, 5.9, 3.4, "One-shot agent forecasting  (v21)", ACC, [
    ("per-detection future offsets, 6 × 0.5 s", DARK),
    ("constant cost in the number of agents", GRAY),
    ("(dense regression on the detection grid)", GRAY),
    ("GT: instance_token tracks → current ego frame", GRAY),
    ("agent ADE 2.17 m (single frame)", DARK),
    ("cost: +0.002 M params (a 1×1 conv)", GREEN)])
bullets(s, [
    "Both share the existing BEV feature and detection stem — 'adding a task' became a head, not a network.",
], y=5.4, size=16)

# ---------- 8 temporal ----------
s = slide("⑥ Temporal fusion — velocity becomes observable",
          "single-frame perception cannot see motion; a streaming BEV can — without breaking TensorRT")
bullets(s, [
    "Previous frame's raw BEV is ego-motion-warped into the current frame (affine_grid + grid_sample) and",
    (1, "residual-fused with the current BEV — TensorRT-safe operators only, no RNN"),
    "Deployment: prev-BEV + warp matrix are graph INPUTS, current BEV an extra OUTPUT → static feed-forward",
    (1, ("cost: +0.19 M params, +5% step time (measured); ONNX/TensorRT-exportable as-is", GREEN)),
    "Early effect (1 epoch): curve ADE 2.28 → 1.91 m; agent forecasting expected to gain most from motion cues",
    (1, "Debugging note: a no_grad forward inside an autocast region cached detached fp16 weights and", GRAY),
    (1, "all backbone gradients under DDP — diagnosed with TORCH_DISTRIBUTED_DEBUG on a 2-GPU mini-run", GRAY),
], y=1.55, size=16)

# ---------- 9 quantitative summary ----------
s = slide("Quantitative summary", "held-out recording · every gain without a single human label")
table(s, [
    ["Metric", "Before", "After", "Driver"],
    ["BEV lane mIoU", "0.283", "0.293", "capacity + data"],
    ["2D-seg mIoU (21 cls)", "0.362", "0.458", "ED head + KMAX 96"],
    ["2D-seg lane IoU", "0.371", "0.548", "coverage rasterise + weights"],
    ["2D-seg pole IoU", "0.251", "0.449", "3-scale detection"],
    ["E2E curve ADE", "2.31 m", "0.72 m", "curvature weighting"],
    ["E2E curve ADE (temporal, ep0)", "2.28 m", "1.91 m", "temporal fusion"],
    ["Agent forecast ADE", "—", "2.15 m", "new task (v21)"],
    ["Model size", "26.8 M", "38.5 M", "8 tasks + temporal"],
], 1.1, 1.55, 11.1, [4.2, 1.9, 1.9, 3.1], fs=14)

# ---------- 10 principles ----------
s = slide("Principles that generalised")
bullets(s, [
    "Measure per-condition — aggregates hide the failures that matter (curves, ranges, rare classes)",
    "Add the diagnostic metric with the fix — ADEc, per-range occupancy IoU, per-class detection counts",
    "Deployment constraints are design inputs — TensorRT-safe ops shaped the temporal fusion, not the reverse",
    "Capacity has a shape — parameters at low resolution, compute at high resolution",
    "Watch the output, not just the loss — the scene-tail and ego-vehicle bugs were found in demo videos",
    "Warm-start everything — 12 rounds, 4 new heads, temporal fusion, and never a cold restart",
], y=1.6, size=18)

# ---------- 11 end ----------
s = slide()
big(s, [
    ("From one task to eight — plus time", 30, True, DARK),
    ("", 10, False, GRAY),
    ("a new task went from 'a labeling campaign' to 'a head and a GPU-day'", 20, False, ACC),
    ("", 10, False, GRAY),
    ("github.com/tier4/METEOR", 20, True, DARK),
    ("Labels by machines. Code by Claude Fable 5. Direction by humans.", 15, True, GREEN),
], y=2.0)

prs.save("out/METEOR_improvements.pptx")
print("saved out/METEOR_improvements.pptx,",
      len(prs.slides.__iter__.__self__._sldIdLst), "slides")

#!/usr/bin/env python3
"""METEOR conference talk deck (English, 15 slides) -> paper/METEOR_talk.pptx."""
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

SW, SH = Inches(13.333), Inches(7.5)
DARK = RGBColor(0x20, 0x28, 0x30)
ACC = RGBColor(0x0E, 0x6E, 0xB8)
GRAY = RGBColor(0x60, 0x68, 0x70)
GREEN = RGBColor(0x1B, 0x78, 0x37)
AI = RGBColor(0xF7, 0xCE, 0x9C)
OP = RGBColor(0xCF, 0xE2, 0xF3)
E2E = RGBColor(0xE8, 0xD5, 0xF2)
prs = Presentation()
prs.slide_width, prs.slide_height = SW, SH
BLANK = prs.slide_layouts[6]


def slide(title=None, sub=None):
    s = prs.slides.add_slide(BLANK)
    if title:
        tb = s.shapes.add_textbox(Inches(0.5), Inches(0.22), Inches(12.3), Inches(0.7))
        p = tb.text_frame.paragraphs[0]
        p.text = title
        p.font.size = Pt(28)
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
            p2.font.color.rgb = GRAY
    return s


def bullets(s, items, x=0.6, y=1.5, w=12.1, h=5.6, size=17):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    for i, it in enumerate(items):
        lvl, txt = (it if isinstance(it, tuple) else (0, it))
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = ("• " if lvl == 0 else "  – ") + txt
        p.font.size = Pt(size if lvl == 0 else size - 3)
        p.font.color.rgb = DARK if lvl == 0 else GRAY
        p.space_after = Pt(6)


def big_center(s, lines, y=2.4, size=30, color=DARK):
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


def pic(s, path, x, y, w=None, h=None):
    kw = {}
    if w:
        kw["width"] = Inches(w)
    if h:
        kw["height"] = Inches(h)
    return s.shapes.add_picture(path, Inches(x), Inches(y), **kw)


def table(s, rows, x, y, w, col_w, fs=13, hdr=True):
    from pptx.util import Inches as In
    nr, nc = len(rows), len(rows[0])
    gt = s.shapes.add_table(nr, nc, In(x), In(y), In(w), In(0.32 * nr)).table
    for ci, cw in enumerate(col_w):
        gt.columns[ci].width = In(cw)
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row):
            c = gt.cell(ri, ci)
            c.text = str(val)
            pr = c.text_frame.paragraphs[0]
            pr.font.size = Pt(fs)
            pr.font.bold = hdr and ri == 0
            pr.font.color.rgb = DARK
    return gt


# ---------- 1 title ----------
s = slide()
big_center(s, [
    ("☄ METEOR", 60, True, DARK),
    ("Multi-task Estimation of Traffic Elements, Objects & Roads", 22, False, GRAY),
    ("From Raw Surround-View Recordings to a Seven-Task Driving Network", 24, True, ACC),
    ("without Human Labels or Human-Written Code", 24, True, ACC),
], y=1.5)
big_center(s, [
    ("Dan Umeda — TIER IV, Inc.", 18, False, DARK),
    ("Built on CoMET, the autolabeling foundation of the Co-MLOps project", 15, False, GRAY),
    ("github.com/tier4/METEOR   ·   NVIDIA GTC 2026 (S81897)", 14, False, GRAY),
], y=5.3)

# ---------- 2 problem ----------
s = slide("The problem: perception is built one labeling campaign at a time")
bullets(s, [
    "Every task needs its own annotation effort — boxes, masks, lanes, occupancy…",
    "Labeling is the dominant cost, and every new task restarts the cycle",
    "Meanwhile, fleets already record everything: 8 cameras + LiDAR + cm-level ego-motion",
    (1, "these raw recordings implicitly contain almost everything the labels would say"),
])
big_center(s, [("What if we never labeled — and never even wrote the code?", 26, True, ACC)], y=5.4)

# ---------- 3 idea ----------
s = slide("The recipe", "Record everything  →  autolabel everything  →  train one network")
bullets(s, [
    "Input: raw Co-MLOps DRS recordings (8 synchronized cameras, LiDAR, ego-pose,",
    (1, "machine-generated 2D panoptic labels) — https://co-mlops.tier4.jp/"),
    "CoMET autolabel foundation → factory distills SEVEN supervision signals, zero humans",
    "One camera-only network learns all seven tasks jointly (no LiDAR at inference)",
    "Hard constraint: TensorRT-safe operators only — deployable as-is",
    "Second constraint (an experiment): all code written by an LLM agent, Claude Fable 5",
])

# ---------- 4 seven tasks ----------
s = slide("Seven tasks, one forward pass")
table(s, [
    ["#", "Task", "Output", "Head cost"],
    ["1", "BEV lane segmentation", "9 cls · 160×100 m @0.2 m", "1.2 M / 933 G"],
    ["2", "Metric depth", "64 bins × 8 cams", "3.3 M / 1093 G"],
    ["3", "3D oriented boxes", "vehicles + VRU", "0.4 M / 82 G"],
    ["4", "2D segmentation", "21 classes × 8 cams", "4.2 M / 219 G"],
    ["5", "2D detection", "10 classes · 3 scales", "2.9 M / 219 G"],
    ["6", "E2E driving", "3 s path + steer/accel/brake", "4.0 M / 37 G"],
    ["7", "3D occupancy", "10 cls · 16×200×200 @0.4 m", "0.7 M / 55 G"],
], 0.7, 1.5, 8.2, [0.5, 3.0, 3.2, 1.5], fs=14)
tb = s.shapes.add_textbox(Inches(9.3), Inches(2.3), Inches(3.6), Inches(3))
tf = tb.text_frame
tf.word_wrap = True
for t, sz, b, c in [("38.3 M params", 24, True, DARK), ("3.1 TFLOPs @ 8×768×432", 18, False, GRAY),
                    ("adding a task costs < 5%", 18, True, GREEN)]:
    p = tf.add_paragraph()
    p.text = t
    p.font.size = Pt(sz)
    p.font.bold = b
    p.font.color.rgb = c

# ---------- 5 factory overview ----------
s = slide("The autolabel factory", "10 stages per scene · resumable · runs concurrently with training")
bullets(s, [
    "BEV lanes — LiDAR × panoptic accumulated in the map frame → vectorized polylines",
    "Metric depth — LiDAR splats + interpolation inside merged road+paint segments",
    "3D boxes — LiDAR annotations kept only when geometrically camera-confirmed",
    "2D seg (21 cls) & 2D boxes (10 cls) — CSV-driven taxonomies from panoptic masks",
    "E2E targets — trajectory / steering / accel / brake from ego-pose alone (no CAN)",
    "3D occupancy — multi-sweep labeled LiDAR + ray-carved free space",
    (1, "quality gates: scene-end trim · stationary-spot filter · indoor rejection · intersection guard"),
], size=16)

# ---------- 6 GT visual ----------
s = slide("Ground truth, entirely machine-made")
pic(s, "paper/figs/qual_gt.png", 0.6, 1.45, w=8.6)
pic(s, "paper/figs/qual_occgt.png", 0.6, 4.55, w=8.6)
tb = s.shapes.add_textbox(Inches(9.5), Inches(2.2), Inches(3.4), Inches(4))
tf = tb.text_frame
tf.word_wrap = True
for t in ["21-class 2D seg", "10-class 2D boxes", "oriented 3D boxes",
          "BEV lanes", "E2E trajectory + controls", "3D occupancy (top + iso)"]:
    p = tf.add_paragraph()
    p.text = "✓ " + t
    p.font.size = Pt(16)
    p.font.color.rgb = GREEN
p = tf.add_paragraph()
p.text = "0 boxes drawn by humans"
p.font.size = Pt(18)
p.font.bold = True
p.font.color.rgb = DARK

# ---------- 7 architecture ----------
s = slide("Architecture")
pic(s, "docs/media/architecture.png", 0.55, 1.35, w=12.2)

# ---------- 8 depth-gated IPM ----------
s = slide("Depth-gated IPM", "depth as a visibility valve for geometric projection")
bullets(s, [
    "Project BEV ground points into every camera with calibrated K/T",
    "grid_sample context features and the predicted depth distribution",
    "gather the depth probability at each point's true range:",
    (1, "w = P(depth = range) + 0.05   →   features flow only where depth agrees"),
    "Weighted average over 8 cameras → 96-ch BEV feature (800×500 @ 0.2 m)",
    "Parameter-free · pull-based (LSS inverted: gather, not scatter) · no attention",
    (1, "suppresses occlusion bleed-through that plagues plain IPM; exports to TensorRT unchanged"),
], size=17)

# ---------- 9 capacity rule ----------
s = slide("Design rule: parameters at low resolution, compute at high resolution")
bullets(s, [
    "Naive stride-4 conv heads = worst FLOPs/params corner",
    "2D heads redesigned: heavy channels at s8/s16, 1×1 laterals touch s4",
    (1, "18× more parameters for ~2× FLOPs"),
    "Result (one round): 2D-seg mIoU 0.375 → 0.451, pole IoU 0.244 → 0.449",
    "3-scale detection (YOLO-style size assignment) turned on small classes:",
    (1, "traffic lights, distant signs, cones — previously never fired"),
    "Model total: +37% params for +10% FLOPs",
])

# ---------- 10 training ----------
s = slide("Rolling rounds", "conversion, GT refinement and training run concurrently")
table(s, [
    ["Round", "Date", "Change", "BEV mIoU"],
    ["r2", "07-11", "4-task baseline", "0.283"],
    ["r4", "07-12", "+ new-vehicle data", "0.286"],
    ["r5", "07-12", "+ 2D detection (v17)", "0.286"],
    ["r7", "07-12", "+ E2E head (v18)", "0.287"],
    ["r8", "07-12→13", "capacity re-balance (v19)", "0.292"],
    ["r9", "07-13", "+ occupancy, curve-weighted E2E (v20)", "0.290"],
    ["r10", "07-13", "small-object / near-VRU / side-cam weights", "running"],
], 0.7, 1.6, 9.0, [0.9, 1.3, 5.2, 1.6], fs=14)
tb = s.shapes.add_textbox(Inches(10.1), Inches(2.6), Inches(2.9), Inches(3))
tf = tb.text_frame
tf.word_wrap = True
for t, c in [("9 rounds", DARK), ("4 new heads", DARK), ("3 days", ACC)]:
    p = tf.add_paragraph()
    p.text = t
    p.font.size = Pt(26)
    p.font.bold = True
    p.font.color.rgb = c

# ---------- 11 measurement-driven fixes ----------
s = slide("Fix what you can measure", "every loss correction started as a quantified failure")
bullets(s, [
    "E2E learned to go straight: 83% of frames near-straight, lon/lat scale 18×",
    (1, "curve ADE 2.29 m vs 0.62 m straight — invisible in aggregate ADE"),
    (1, "fix: lateral L1 ×4 + curvature sample weight  →  curve ADE 2.31 → 1.07 → 0.72 m"),
    "Thin classes vanished from GT: lane line ≈ 0.3 px after 4× downsample",
    (1, "fix: coverage-based rasterisation (+49% lane pixels, connected dashes)"),
    "Small objects silently truncated: box budget 32 dropped 29% of annotations",
    (1, "fix: budget 96 + per-class positive weights → cones / lights / unknowns detected"),
    "NaN-safe aux losses: all-ignore batches return a graph-preserving zero",
])

# ---------- 12 results ----------
s = slide("Results", "held-out recording · zero human labels")
table(s, [
    ["Metric", "Value"],
    ["BEV lane mIoU", "0.292"],
    ["2D segmentation mIoU (21 cls)", "0.451"],
    ["Depth MAE (0–80 m)", "1.60 m"],
    ["E2E ADE / FDE (3 s)", "0.76 / 1.65 m"],
    ["E2E curve ADE (|lat|>2 m)", "0.72 m"],
    ["Steering MAE / brake acc.", "0.027 rad / 0.81"],
    ["Occupancy road IoU (0–40 m)", "0.63–0.64"],
], 0.7, 1.6, 5.6, [3.7, 1.9], fs=15)
pic(s, "paper/figs/qual_inference.png", 6.6, 1.7, w=6.3)

# ---------- 13 agent ----------
s = slide("Zero human-written code", "the second automation axis")
bullets(s, [
    "Every artefact — 10 GT extractors, 20 model variants, trainer, demos, docs,",
    (1, "the paper, this deck — written by Claude Fable 5 (Anthropic), autonomously"),
    "Humans: ~70 natural-language instructions + review of videos & metrics",
    "Agent: design, implementation, launches, monitoring, diagnosis, iteration",
    (1, "several fixes originated from the agent's own measurements (e.g. box-budget analysis)"),
    "Division of labour: humans specify WHAT · the agent implements and explains HOW",
    (1, "cadence: nine training rounds and four new task heads in three days"),
])

# ---------- 14 takeaways ----------
s = slide("Takeaways")
bullets(s, [
    "Fleet recordings already contain the labels — a factory can extract all of them",
    "One shared BEV representation carries seven tasks; new tasks cost <5%",
    "Deployment constraints (TensorRT ops) and capacity rules are design inputs, not afterthoughts",
    "Measure per-condition (curves, ranges, classes) — aggregates hide the failures that matter",
    "LLM agents can own the full engineering loop when the metrics are automated",
], size=18)
big_center(s, [("a task used to cost a labeling campaign — now it costs a prompt and a GPU-day", 22, True, ACC)], y=5.6)

# ---------- 15 end ----------
s = slide()
big_center(s, [
    ("☄ METEOR", 48, True, DARK),
    ("github.com/tier4/METEOR", 24, False, ACC),
    ("paper · code · docs · demo videos — all in the repository", 16, False, GRAY),
    ("", 8, False, GRAY),
    ("Co-MLOps: https://co-mlops.tier4.jp/", 16, False, GRAY),
    ("NVIDIA GTC 2026 session S81897", 16, False, GRAY),
    ("", 8, False, GRAY),
    ("Labels by machines. Code by Claude Fable 5. Direction by humans.", 18, True, GREEN),
], y=1.8)

prs.save("paper/METEOR_talk.pptx")
print("saved paper/METEOR_talk.pptx,", len(prs.slides.__iter__.__self__._sldIdLst), "slides")

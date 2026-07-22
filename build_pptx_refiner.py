#!/usr/bin/env python3
"""METEOR Refiner concept deck -> out/METEOR_refiner.pptx.

Explains the post-hoc residual refiner (BEV seg + 3D box + E2E), its
guarantees, measured effects, and the method-A graft into v40/r35.
Also renders the architecture diagram docs/media/refiner_arch.png.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_AUTO_SIZE, PP_ALIGN
from pptx.util import Emu, Inches, Pt

# ---------------------------------------------------------------- diagram
AI = "#F7CE9C"; OP = "#CFE2F3"; E2 = "#E8D5F2"; MEM = "#FDE7B5"
INb = "#EEEEEE"; OUT = "#D9EAD3"; EDGE = "#606870"; DK = "#202830"
GRN = "#1B7837"; AMB = "#B9770E"

fig, ax = plt.subplots(figsize=(15.5, 7.4), dpi=115)
ax.set_xlim(0, 160); ax.set_ylim(0, 76); ax.axis("off")
ax.set_title("Post-hoc Refiner — residual heads on the FROZEN model's outputs",
             fontsize=16, fontweight="bold", color=DK)


def box(x, y, w, h, t, s="", fc=AI, fs=11, sfs=8.6, out=""):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5",
                                fc=fc, ec=EDGE, lw=1.2))
    lines = 1 + bool(s) + bool(out)
    top = y + h - h / (lines + 1)
    step = h / (lines + 1) * 1.12
    ax.text(x + w / 2, top, t, ha="center", va="center", fontsize=fs,
            fontweight="bold", color=DK)
    if s:
        ax.text(x + w / 2, top - step, s, ha="center", va="center",
                fontsize=sfs, color="#555")
    if out:
        ax.text(x + w / 2, top - step * (1 + bool(s)), out, ha="center",
                va="center", fontsize=sfs, color=GRN, fontweight="bold")


def arr(x1, y1, x2, y2, col=DK, lw=1.6, dash=False):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=13, lw=lw, color=col,
                                 linestyle="--" if dash else "-"))


box(2, 30, 30, 18, "FROZEN main model", "v39 / r34  (48 M params)\n"
    "requires_grad = False", fc=OP,
    out="its outputs, not its weights")
# three frozen outputs
box(40, 56, 26, 13, "BEV seg logits", "9 x 800 x 500", fc=MEM)
box(40, 32, 26, 13, "3D box hm+reg", "8 x 400 x 250", fc=MEM)
box(40, 8, 26, 13, "E2E plan vector", "K x 6 wp + conf + ctrl", fc=MEM)
# three residual heads
box(78, 56, 30, 13, "seg refiner (U-Net)", "3.93 M  far-range completion",
    fc=AI, out="+ residual")
box(78, 32, 30, 13, "box refiner (U-Net)", "0.99 M  peak/size sharpen",
    fc=AI, out="+ residual")
box(78, 8, 30, 13, "E2E refiner (MLP)", "0.11 M  waypoint correction",
    fc=AI, out="+ residual")
# outputs
box(118, 32, 30, 20, "Refined outputs", "sharper far seg, tighter boxes,\n"
    "better ADE", fc=OUT,
    out="zero-init => identity at start\n(never degrades the base)")

arr(32, 42, 40, 62); arr(32, 39, 40, 38); arr(32, 36, 40, 14)
arr(66, 62, 78, 62); arr(66, 38, 78, 38); arr(66, 14, 78, 14)
arr(108, 62, 118, 48); arr(108, 38, 133, 42); arr(108, 14, 118, 36)
ax.text(80, 2, "Base weights are shared with nothing (0 key overlap); the "
        "refiner is a separate 5 M network on the outputs.",
        fontsize=9.5, style="italic", color=AMB, ha="center")
fig.savefig("docs/media/refiner_arch.png", bbox_inches="tight",
            facecolor="white")
print("saved docs/media/refiner_arch.png")

# ---------------------------------------------------------------- deck
SW, SH = Inches(13.333), Inches(7.5)
DARK = RGBColor(0x20, 0x28, 0x30); ACC = RGBColor(0x0E, 0x6E, 0xB8)
GRAY = RGBColor(0x60, 0x68, 0x70); GREEN = RGBColor(0x1B, 0x78, 0x37)
RED = RGBColor(0xC0, 0x39, 0x2B); AMBER = RGBColor(0xB9, 0x77, 0x0E)
prs = Presentation(); prs.slide_width, prs.slide_height = SW, SH
BLANK = prs.slide_layouts[6]


def slide(title=None, sub=None):
    s = prs.slides.add_slide(BLANK)
    if title:
        tb = s.shapes.add_textbox(Inches(0.5), Inches(0.22), Inches(12.3),
                                  Inches(0.7))
        p = tb.text_frame.paragraphs[0]
        p.text = title; p.font.size = Pt(27); p.font.bold = True
        p.font.color.rgb = DARK
        ln = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.5), Inches(0.95),
                                Inches(12.3), Emu(1))
        ln.fill.solid(); ln.fill.fore_color.rgb = ACC
        ln.line.fill.background()
        if sub:
            tb2 = s.shapes.add_textbox(Inches(0.5), Inches(1.0), Inches(12.3),
                                       Inches(0.4))
            p2 = tb2.text_frame.paragraphs[0]
            p2.text = sub; p2.font.size = Pt(14); p2.font.italic = True
            p2.font.color.rgb = GRAY
    return s


def bullets(s, items, x=0.6, y=1.5, w=12.1, h=5.6, size=17):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.NONE
    for i, it in enumerate(items):
        lvl, txt, col = 0, it, None
        if isinstance(it, tuple):
            lvl, txt = it[0], it[1]
            col = it[2] if len(it) > 2 else None
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = ("• " if lvl == 0 else "   – ") + txt
        p.level = lvl
        p.font.size = Pt(size if lvl == 0 else size - 3)
        p.font.color.rgb = col or DARK
        p.space_after = Pt(6)


# 1 title
s = slide()
tb = s.shapes.add_textbox(Inches(0.8), Inches(2.6), Inches(11.7), Inches(2))
p = tb.text_frame.paragraphs[0]
p.text = "The Refiner"; p.font.size = Pt(48); p.font.bold = True
p.font.color.rgb = DARK; p.alignment = PP_ALIGN.CENTER
tb2 = s.shapes.add_textbox(Inches(0.8), Inches(3.9), Inches(11.7), Inches(1))
p2 = tb2.text_frame.paragraphs[0]
p2.text = ("A post-hoc residual network that improves the frozen model's "
           "outputs — without ever degrading them")
p2.font.size = Pt(18); p2.font.italic = True; p2.font.color.rgb = GRAY
p2.alignment = PP_ALIGN.CENTER

# 2 problem
s = slide("Why a refiner?", "Some errors are hard to fix inside the base model")
bullets(s, [
    "The camera-only base model has structural limits that more training "
    "cannot easily remove:",
    (1, "BEV seg drops out past ~50 m: wide cameras give < 2 px per 0.2 m "
        "cell, so a single frame has almost no evidence there (recall "
        "collapse, not blur).", RED),
    (1, "3D-box centre peaks are soft at range; box size / heading noisy.",
        RED),
    (1, "E2E plan has a residual error the scene encoder cannot shave off "
        "alone.", RED),
    "A refiner attacks these AFTER the base model, using extra receptive "
    "field, temporal accumulation, and priors (lanes are continuous, "
    "obstacles are static) that the per-frame base cannot apply.",
    (0, "Key idea: don't retrain the whole model — add a small corrector on "
        "its outputs.", GREEN),
])

# 3 concept
s = slide("The concept: residual correction of frozen outputs")
bullets(s, [
    ("The base model is FROZEN (eval, requires_grad=False). The refiner reads "
     "its OUTPUTS (logits / vectors), not its weights."),
    ("It predicts a RESIDUAL that is added to each output; the last layer is "
     "zero-initialised, so at the start the refiner is exactly identity."),
    (1, "=> the refined output starts equal to the base output and can only "
        "improve it. The priority tasks (Seg / 3D / E2E) are preserved by "
        "construction — never degraded.", GREEN),
    ("The base and the refiner share ZERO weights (different architecture, "
     "different tensors) — it is a separate 5 M network, not a fine-tune."),
    ("Trainable without touching the running base model, and deployable as "
     "its own engine chained after the main graph."),
])
s.shapes.add_picture("docs/media/refiner_arch.png", Inches(4.6), Inches(4.4),
                     height=Inches(2.9))

# 4 architecture / three heads
s = slide("Three heads, one frozen forward",
          "shared base forward feeds three independent residual heads")
s.shapes.add_picture("docs/media/refiner_arch.png", Inches(0.7), Inches(1.4),
                     width=Inches(8.0))
bullets(s, [
    "seg — U-Net",
    (1, "3.93 M; wide receptive field (to s16) propagates near structure "
        "into the >50 m field."),
    "box — U-Net",
    (1, "0.99 M; on the [hm+reg] det grid; sharpens peaks, corrects size."),
    "E2E — MLP",
    (1, "0.11 M; waypoint residual conditioned on v0 + pooled BEV."),
    "Each head zero-init residual; DDP-safe; heads deployable separately.",
], x=9.0, y=1.6, w=3.9, size=13)

# 5 guarantees
s = slide("Guarantees — why it is safe")
bullets(s, [
    (0, "Never degrades the priority tasks.", GREEN),
    (1, "Residual + zero-init => identity at start; it can only add a learned "
        "correction."),
    (0, "Base fully frozen during refiner training.", GREEN),
    (1, "eval + requires_grad=False + forward under no_grad; optimizer sees "
        "only the refiner's params."),
    (0, "Numerically robust.", GREEN),
    (1, "E2E head: LayerNorm + tanh-bounded residual + fp32 + grad clip; "
        "DDP-safe non-finite guard (all ranks agree)."),
    (0, "Deployment-flexible.", GREEN),
    (1, "Run as a separate engine, OR graft it into the model (method A)."),
])

# 6 measured
s = slide("Measured effect (val, raw -> refined)",
          "held-out val day; larger gains at long range where the base drops out")
rows = [
    ("Task / band", "raw", "refined", ""),
    ("BEV road IoU  40-80 m", "0.43", "0.57", "+0.14 (far-range fill)"),
    ("BEV laneline IoU  40-80 m", "0.02", "0.04", "2x (was near-empty)"),
    ("BEV laneline IoU  20-40 m", "0.10", "0.13", "recovered"),
    ("E2E ADE (min-of-K)", "0.95 m", "0.88 m", "second-stage planner"),
    ("temporal stability (seg-fuse)", "0.83", "0.93", "flicker down"),
]
tb = s.shapes.add_table(len(rows), 4, Inches(0.9), Inches(1.7),
                        Inches(11.5), Inches(3.6)).table
for c, wdt in enumerate((4.6, 2.0, 2.2, 2.7)):
    tb.columns[c].width = Inches(wdt)
for r, row in enumerate(rows):
    for c, val in enumerate(row):
        cell = tb.cell(r, c); cell.text = val
        pr = cell.text_frame.paragraphs[0]
        pr.font.size = Pt(14); pr.font.bold = (r == 0)
        pr.font.color.rgb = DARK if r == 0 else (
            GREEN if c == 3 else DARK)
bullets(s, [
    "Black (unobserved) trained as a real class so road does not bleed into "
    "the background; road-completion in the far field is preserved.",
], y=5.6, size=14)

# 7 method A graft
s = slide("Method A — graft into the model (v40, r35)",
          "the refiner's learned weights carry forward into continued training")
bullets(s, [
    ("v40 = v39 base + the refiner grafted on as trainable post-heads "
     "(53 M = 48 M base + 5 M refiner)."),
    ("r35 is initialised from a combined checkpoint: r34 base weights + the "
     "trained refiner weights (verified: missing=0, unexpected=0)."),
    (1, "=> the refiner's learned weights are NOT thrown away — they continue "
        "training end-to-end with the base.", GREEN),
    ("Because each head is residual, at the graft point the combined model "
     "reproduces base+refiner behaviour; fine-tuning only adapts it."),
    ("Deployment simplifies: one 53 M model, no separate refiner engine."),
    (0, "Alternative (not chosen): distillation — bake the effect into the "
        "48 M base and drop the refiner weights.", GRAY),
])

out = "out/METEOR_refiner.pptx"
prs.save(out)
print("saved", out, "-", len(prs.slides._sldIdLst), "slides")

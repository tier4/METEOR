#!/usr/bin/env python3
"""Render the English METEOR architecture diagram (docs/media/architecture.png)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

AI = "#F7CE9C"      # learned
OP = "#CFE2F3"      # geometry / fixed
E2E = "#E8D5F2"     # planning
IN = "#EEEEEE"
EDGE = "#606870"
DARK = "#202830"
GREEN = "#1B7837"

fig, ax = plt.subplots(figsize=(17.6, 8.6), dpi=110)
ax.set_xlim(0, 176)
ax.set_ylim(0, 86)
ax.axis("off")


def box(x, y, w, h, title, sub="", out="", fc=AI, fs=12, sfs=9.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6",
                                fc=fc, ec=EDGE, lw=1.2))
    lines = 1 + bool(sub) + bool(out)
    top = y + h - h / (lines + 1)
    step = h / (lines + 1) * 1.15
    ax.text(x + w / 2, top, title, ha="center", va="center",
            fontsize=fs, fontweight="bold", color=DARK)
    if sub:
        ax.text(x + w / 2, top - step, sub, ha="center", va="center",
                fontsize=sfs, color="#555c63")
    if out:
        ax.text(x + w / 2, top - step * (1 + bool(sub)), out, ha="center",
                va="center", fontsize=sfs, color=GREEN, fontweight="bold")


def arrow(x1, y1, x2, y2, lw=1.6):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=14, lw=lw, color=DARK))


# ---- inputs (left column) ----
box(2, 56, 17, 12, "8 cameras", "3ch 432×768\nWIDE/L/R/NARROW ×F/B", fc=IN, fs=11)
box(2, 36, 17, 9, "Calibration K/T", "used by IPM only", fc=IN, fs=10.5)
box(2, 20, 17, 9, "Speed v0", "used by E2E only", fc=IN, fs=10.5)

# ---- image branch ----
box(25, 56, 21, 12, "ResNet-34 + FPN",
    "shared feature 160ch\n@108×192 (s4) · 21.7M · 475G", fs=11.5)
box(54, 72, 30, 11.5, "2D Seg head", "enc-dec s8/s16 · 4.2M · 219G",
    out="→ 21-class semantics ×8 cams")
box(54, 57, 30, 11.5, "2D Det head", "3-scale CenterNet · 2.9M · 219G",
    out="→ 10-class boxes ×8 cams")
box(54, 42, 30, 11.5, "Depth decoder", "64 bins @s4 · 3.3M · 1093G",
    out="→ metric depth 0–80 m ×8")
box(54, 29, 30, 9.5, "Context 1×1", "96ch · 0.02M · 5G", fs=11)

# ---- IPM + BEV ----
box(92, 36, 25, 13, "Depth-gated IPM",
    "project (K/T) · grid_sample\n· gather — parameter-free", fc=OP, fs=11.5)
box(123, 38, 15, 9.5, "BEV feature", "96ch 800×500 @0.2 m", fc=OP, fs=10.5)

# ---- BEV heads ----
box(144, 68, 30, 11.5, "BEV lane decoder", "@800×500 · 1.2M · 933G",
    out="→ 9-class lane map 160×100 m")
box(144, 53, 30, 11.5, "3D Box head", "CenterPoint @s2 · 0.4M · 82G",
    out="→ oriented boxes: veh + VRU")
box(144, 38, 30, 11.5, "Occupancy head", "16z × 200×200 · 0.7M · 55G",
    out="→ 10-class voxels ±40 m")
box(144, 23, 30, 11.5, "E2E head", "pyramid+MLP(·, v0) · 4.0M · 37G",
    out="→ 3 s path · steer · accel · brake", fc=E2E)

# ---- arrows ----
arrow(19, 62, 25, 62)
for hy in (77.5, 62.5, 47.5, 33.5):
    arrow(46, 62, 54, hy)
arrow(84, 45, 92, 44)            # depth -> IPM
arrow(84, 33.5, 92, 40)          # ctx  -> IPM
arrow(19, 40.5, 92, 41.5)        # K/T  -> IPM
arrow(117, 42.5, 123, 42.5)
for hy in (73.5, 58.5, 43.5, 28.5):
    arrow(138, 43, 144, hy)
arrow(19, 24.5, 144, 26)         # v0 -> E2E

# ---- legend (bottom-left, clear zone) ----
lx = 25
for c, t in ((AI, "learned"), (OP, "geometry (no params)"),
             (E2E, "planning"), (IN, "inputs")):
    ax.add_patch(FancyBboxPatch((lx, 8), 3.4, 3.4,
                                boxstyle="round,pad=0.3", fc=c, ec=EDGE, lw=1))
    ax.text(lx + 4.6, 9.7, t, fontsize=10.5, va="center", color=DARK)
    lx += 7.5 + len(t) * 1.2
ax.text(25, 3.2, "green = task outputs (one forward pass produces all seven)",
        fontsize=10.5, color=GREEN, fontweight="bold")

ax.text(88, 83.5,
        "METEOR v20 — 7 tasks · 38.3M params · 3.1 TFLOPs @ 8×768×432 · TensorRT-safe ops only",
        ha="center", fontsize=14, fontweight="bold", color=DARK)
ax.text(104, 31.5,
        "depth gates the IPM:\nfeatures enter the BEV only where\npredicted depth matches the ray range",
        ha="center", fontsize=9.2, style="italic", color="#555c63")

plt.tight_layout()
plt.savefig("docs/media/architecture.png", bbox_inches="tight", facecolor="white")
print("saved")

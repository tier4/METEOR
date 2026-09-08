#!/usr/bin/env python3
"""Render the English METEOR architecture diagram (docs/media/architecture.png).

v29: 12 tasks, streaming 3-slot temporal memory, task-routed BEV
(geometry heads on the raw single-frame BEV, motion heads on the fused BEV).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

AI = "#F7CE9C"      # learned
OP = "#CFE2F3"      # geometry / fixed
E2E = "#E8D5F2"     # planning / motion
MEM = "#FDE7B5"     # temporal memory
IN = "#EEEEEE"
EDGE = "#606870"
DARK = "#202830"
GREEN = "#1B7837"
AMBER = "#B9770E"

fig, ax = plt.subplots(figsize=(19.0, 9.6), dpi=110)
ax.set_xlim(0, 190)
ax.set_ylim(0, 96)
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


def arrow(x1, y1, x2, y2, lw=1.6, col=DARK):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=14, lw=lw, color=col))


# ---- inputs (left column) ----
box(1, 62, 17, 12, "8 cameras", "3ch 432×768\nWIDE/L/R/NARROW ×F/B", fc=IN, fs=11)
box(1, 46, 17, 8.5, "Calibration K/T", "used by IPM only", fc=IN, fs=10.5)
box(1, 33, 17, 8.5, "Speed v0", "used by E2E only", fc=IN, fs=10.5)
box(1, 18, 17, 9.5, "Ego pose", "→ warp θ ×3 slots\n+ kin history (v36)", fc=IN, fs=10.5)
box(1, 4, 17, 11, "OPTIONAL inputs", "LiDAR depth / pillar raster\nroute intent (nav)\nzeros = camera-only,\nbit-equal", fc=IN, fs=9.5, sfs=8.0)

# ---- image branch ----
box(23, 62, 20, 12, "ResNet-34 + FPN",
    "shared feature 160ch\n@108×192 (s4) · 21.7M", fs=11.5)
box(50, 82, 28, 10.5, "2D Seg head", "enc-dec s8/s16 · 4.2M",
    out="→ 21-class semantics ×8", sfs=8.8)
box(50, 69.5, 28, 10.5, "2D Det head", "3-scale CenterNet · 2.9M",
    out="→ 10-class boxes ×8", sfs=8.8)
box(50, 57, 28, 10.5, "Depth decoder", "64 bins @s4 · 3.3M",
    out="→ metric depth 0–80 m ×8", sfs=8.8)
box(50, 45.5, 28, 9.5, "TL head", "front WIDE+NARROW · 1.1M",
    out="→ ego traffic-light state", sfs=8.8)
box(50, 34.5, 28, 9, "Context 1×1", "96ch · 0.02M", fs=11)

# ---- IPM + BEV ----
box(84, 40, 22, 12.5, "Depth-gated IPM",
    "project (K/T) · grid_sample\n· gather — parameter-free", fc=OP, fs=11.5)
box(111, 52, 17, 9.5, "RAW BEV", "96ch 800×500 @0.2 m\nsingle frame", fc=OP,
    fs=10.5, sfs=8.6)
box(111, 33, 17, 11.5, "Temporal fuse", "tfuse3 + per-cell slot\ngate (B2)",
    fc=MEM, fs=10.5, sfs=8.6)
box(111, 16, 17, 10.5, "Memory queue", "t−0.4 / 1.2 / 2.8 s\nBEVs, ego-warped",
    fc=MEM, fs=10, sfs=8.4)
box(133, 33, 14, 9.5, "FUSED BEV", "96ch · velocity", fc=MEM, fs=10.5, sfs=8.6)

# ---- geometry heads (from RAW BEV) ----
box(152, 82, 36, 10.5, "BEV lane seg head", "LaneDecED @800×500 · 4.1M",
    out="→ 9-class lane map 160×100 m", sfs=8.8)
box(152, 70, 36, 10.5, "3D Box head", "CenterPoint @s2 · 2.1M",
    out="→ oriented boxes: veh + VRU", sfs=8.8)
box(152, 58, 36, 10.5, "Occupancy + flow", "16z×200×200 · 0.75M",
    out="→ 10-class voxels + velocity", sfs=8.8)

# ---- motion heads (from FUSED BEV) ----
box(152, 32, 36, 10.5, "E2E head (K=3)", "attn-pool + intent/kin +\nrisk-integral selection (v38)",
    out="→ 3 paths + conf · steer/accel/brake", fc=E2E, sfs=8.4)
box(152, 20, 36, 10.5, "Agent forecast (K=3)", "+ class feature · 0.5M",
    out="→ per-agent 3 s ×3 + parked flag", fc=E2E, sfs=8.6)
box(152, 8, 36, 10.5, "Risk field", "ConvBlocks on fused BEV · 0.2M",
    out="→ area risk ±40×±25 m", fc=E2E, sfs=8.8)

# ---- arrows ----
arrow(18, 68, 23, 68)
for hy in (87, 74.5, 62, 50, 39):
    arrow(43, 68, 50, hy)
arrow(78, 61, 84, 50)                     # depth -> IPM
arrow(78, 39, 84, 45)                     # ctx  -> IPM
arrow(18, 50, 84, 47, lw=1.2)             # K/T  -> IPM
arrow(106, 47, 111, 55)                   # IPM -> RAW BEV
arrow(119, 52, 119, 44.5, lw=1.4)         # RAW -> fuse
arrow(119, 26.5, 119, 33, lw=1.4)         # queue -> fuse
arrow(18, 22, 111, 21, lw=1.2)            # pose -> queue
arrow(128, 38, 133, 38)                   # fuse -> FUSED BEV
for hy in (87, 75, 63):                   # raw BEV -> geometry heads
    arrow(128, 57, 152, hy, col=GREEN, lw=1.3)
for hy in (37, 25, 13):                   # fused BEV -> motion heads
    arrow(147, 38, 152, hy, col=AMBER, lw=1.3)
arrow(18, 37, 152, 34, lw=1.2)            # v0 -> E2E
arrow(18, 9, 84, 42, lw=1.2, col=AMBER)   # optional LiDAR -> IPM/BEV

# streaming feedback: the raw BEV returns as next frame's history.
# Routed through the clear gap between the IPM box (ends x=106.6) and the
# BEV column (starts x=111) so it crosses nothing.
for seg, head in ((((111, 54), (108.6, 54)), False),
                  (((108.6, 54), (108.6, 19)), False),
                  (((108.6, 19), (111, 19)), True)):
    ax.add_patch(FancyArrowPatch(seg[0], seg[1],
                                 arrowstyle="-|>" if head else "-",
                                 mutation_scale=13, lw=1.5, color=AMBER))
ax.text(119.5, 12.5, "raw BEV feeds back as next frame's history",
        ha="center", fontsize=8.4, style="italic", color=AMBER,
        fontweight="bold")

# ---- legend ----
lx = 23
for c, t in ((AI, "learned"), (OP, "geometry (no params)"),
             (MEM, "temporal memory"), (E2E, "motion / planning"),
             (IN, "inputs")):
    ax.add_patch(FancyBboxPatch((lx, 8), 3.0, 3.0,
                                boxstyle="round,pad=0.3", fc=c, ec=EDGE, lw=1))
    ax.text(lx + 4.2, 9.5, t, fontsize=10, va="center", color=DARK)
    lx += 7.0 + len(t) * 1.15
ax.text(23, 3.6, "green arrows = geometry tasks read the RAW single-frame BEV   ·   "
        "amber = motion tasks read the FUSED temporal BEV", fontsize=10,
        color=DARK)
ax.text(23, 0.8, "green text = task outputs — one forward pass produces all twelve",
        fontsize=10.5, color=GREEN, fontweight="bold")

ax.text(95, 94,
        "METEOR v52 — 12 tasks · 54M params (refiner incl.) · one static TensorRT engine "
        "(AGX Orin INT8: 69.6 ms 2:4 sparse / 78.7 ms dense) · zero human labels, zero human code",
        ha="center", fontsize=13.5, fontweight="bold", color=DARK)
ax.text(95, 29.5,
        "depth gates the IPM: features enter the BEV only where\n"
        "predicted depth matches the ray range",
        ha="center", fontsize=9, style="italic", color="#555c63")

plt.tight_layout()
plt.savefig("docs/media/architecture.png", bbox_inches="tight", facecolor="white")
print("saved docs/media/architecture.png")

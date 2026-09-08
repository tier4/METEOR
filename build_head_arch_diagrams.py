#!/usr/bin/env python3
"""Layer-level architecture diagrams for every head (v38/v39).

Two figures: geometry heads and motion/planning heads. Each head is a
vertical pipeline of layer boxes annotated with channels / resolution,
matching bevlane/model.py.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

AI = "#F7CE9C"; OP = "#CFE2F3"; E2E = "#E8D5F2"; MEM = "#FDE7B5"
IN = "#EEEEEE"; OUT = "#D9EAD3"
EDGE = "#606870"; DARK = "#202830"; GREEN = "#1B7837"


def pipeline(ax, x, w, title, layers, y_top=88, fc=AI):
    """Vertical chain of layer boxes. layers = [(text, color|None), ...]"""
    ax.text(x + w / 2, y_top + 4, title, ha="center", fontsize=11.5,
            fontweight="bold", color=DARK)
    h = 7.2
    y = y_top - h
    for i, (txt, c) in enumerate(layers):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4",
                                    fc=c or fc, ec=EDGE, lw=1.0))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center",
                fontsize=8.0, color=DARK)
        if i < len(layers) - 1:
            ax.add_patch(FancyArrowPatch((x + w / 2, y),
                                         (x + w / 2, y - 2.3),
                                         arrowstyle="-|>",
                                         mutation_scale=9, lw=1.1,
                                         color=DARK))
        y -= h + 2.3
    return y


def new_fig(title):
    fig, ax = plt.subplots(figsize=(19, 10.2), dpi=110)
    ax.set_xlim(0, 190); ax.set_ylim(0, 100); ax.axis("off")
    ax.set_title(title, fontsize=16, fontweight="bold", color=DARK, pad=14)
    return fig, ax


# ------------------------------------------------- fig 1: geometry heads
fig, ax = new_fig("Head architectures I — geometry (all consume the RAW "
                  "single-frame BEV or image features)")
pipeline(ax, 2, 26, "BEV lane seg head", [
    ("RAW BEV 96×800×500", IN),
    ("enc: Conv s2 → 128 @400×250", None),
    ("enc: Conv s2 → 192 @200×125", None),
    ("dec: up ×2 + skip → 128", None),
    ("dec: up ×2 + skip → 96", None),
    ("1×1 → 11 classes @800×500", None),
    ("lane map 160×100 m @0.2 m", OUT)])
pipeline(ax, 33, 26, "Depth decoder", [
    ("FPN feat 160×108×192 (s4)", IN),
    ("depth_up: ConvBlock ×2", None),
    ("1×1 → 64 bins ×8 cams", None),
    ("softmax = P(depth)", None),
    ("(+ C6a LiDAR sharpen)", MEM),
    ("metric depth 0–80 m", OUT)])
pipeline(ax, 64, 26, "2D seg / 2D det", [
    ("FPN feat s4/s8/s16", IN),
    ("seg: enc-dec s8/s16 → 21c", None),
    ("det: 3-scale CenterNet", None),
    ("hm 10c + wh/off per scale", None),
    ("21-class masks + 10-class boxes", OUT)])
pipeline(ax, 95, 26, "3D box + unknown", [
    ("RAW BEV 96×800×500", IN),
    ("det_stem: Conv s2 → 128 @400×250", None),
    ("hm 1×1→2c | reg 1×1→6c", None),
    ("unknown (v34): temporal feat 256\n→ Conv3×3 64 → ConvBlock 64", MEM),
    ("unk hm 1×1 → 1c @400×250", None),
    ("veh/VRU boxes + 0.4 m unknowns", OUT)])
pipeline(ax, 126, 27, "Occupancy + flow", [
    ("RAW BEV crop 96×400×400", IN),
    ("occ_stem: Conv s2 → 128", None),
    ("ConvBlock → 192 @200×200", None),
    ("occ 1×1 → 16z×10c", None),
    ("flow 1×1 → 2ch (from occ feat)", None),
    ("voxels + velocity field", OUT)])
ax.set_xlim(0, 158)   # lane-graph column retired (2026-09-08)
fig.savefig("docs/media/headarch_geometry.png", bbox_inches="tight",
            facecolor="white")

# --------------------------------------------- fig 2: motion / planning
fig, ax = new_fig("Head architectures II — motion & planning (consume the "
                  "FUSED temporal BEV)")
pipeline(ax, 2, 30, "Forecast + stationary", [
    ("FUSED BEV + motion residual\n(RAW − warped slot0) = 192ch", MEM),
    ("traj_stem: Conv s2 128\n+ ConvBlock @400×250", None),
    ("cat det_feat.detach (128)\n+ det yaw sin/cos (2) = 258", None),
    ("traj 1×1 → K3×12 + 3 logits", None),
    ("stat (v33): 1×1 on 256 → 1", None),
    ("per-agent 3 s ×3 + parked flag", OUT)])
pipeline(ax, 37, 30, "E2E planner (v36–v39)", [
    ("FUSED BEV 96", MEM),
    ("ego_stem → pool 96\n+ B3: 3 queries × 400 tokens MHA", None),
    ("MLP → K3×12 wp + 3 conf\n+ steer/accel/brake", None),
    ("+ kin Δ (7) · intent Δ (3)\n· v(t) aux (96→6)", None),
    ("v39: φ(t)×v(t) cumsum blend\n(tanh gate)", E2E),
    ("v38: conf − gate·∫risk → pick", E2E),
    ("chosen plan + controls", OUT)])
pipeline(ax, 72, 26, "Traffic light", [
    ("front WIDE + NARROW\nfeats @s4, concat", IN),
    ("tl_head convs → pool", None),
    ("fc → 4 states", None),
    ("none/green/yellow/red", OUT)])
pipeline(ax, 103, 26, "Risk field", [
    ("FUSED BEV crop\n96×400×250", MEM),
    ("ConvBlocks ×2 → 64", None),
    ("1×1 → 1ch logit", None),
    ("sigmoid risk ±40×±25 m", OUT)])
pipeline(ax, 134, 27, "B2 / B4 modules", [
    ("B2: cat 4 BEV slots 384ch", MEM),
    ("1×1 → 4 · softmax per cell\n= slot gate (zero-init)", None),
    ("B4: det tokens 400 ×\n4 queries MHA 128", None),
    ("1×1 → 256 residual into\nforecast features", None),
    ("ghost-robust fusion +\nscene interaction", OUT)])
pipeline(ax, 164, 24, "Optional LiDAR (v32)", [
    ("pillar raster 4×400×250", IN),
    ("Conv3×3 64 → ConvBlock 96", None),
    ("up ×2 → 96×800×500", None),
    ("flag-gated add to RAW BEV\n(zeros = bit-equal off)", MEM),
    ("sensor-optional BEV", OUT)])
fig.savefig("docs/media/headarch_motion.png", bbox_inches="tight",
            facecolor="white")
print("saved 2 head-architecture diagrams")

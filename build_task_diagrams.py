#!/usr/bin/env python3
"""Per-task detail diagrams -> docs/media/detail_*.png (v38/v39 era).

Four figures: BEV generation, temporal memory & routing, E2E planning
stack (with guardrails), perception heads (det / forecast / unknown /
occupancy).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

AI = "#F7CE9C"; OP = "#CFE2F3"; E2E = "#E8D5F2"; MEM = "#FDE7B5"
IN = "#EEEEEE"; SAFE = "#D9EAD3"
EDGE = "#606870"; DARK = "#202830"; GREEN = "#1B7837"; AMBER = "#B9770E"
RED = "#C0392B"


def new_fig(w=15.5, h=7.6):
    fig, ax = plt.subplots(figsize=(w, h), dpi=115)
    ax.set_xlim(0, 155); ax.set_ylim(0, 76); ax.axis("off")
    return fig, ax


def mk(ax):
    def box(x, y, w, h, title, sub="", fc=AI, fs=11.5, sfs=8.8, out=""):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6",
                                    fc=fc, ec=EDGE, lw=1.2))
        lines = 1 + bool(sub) + bool(out)
        top = y + h - h / (lines + 1)
        step = h / (lines + 1) * 1.15
        ax.text(x + w / 2, top, title, ha="center", va="center", fontsize=fs,
                fontweight="bold", color=DARK)
        if sub:
            ax.text(x + w / 2, top - step, sub, ha="center", va="center",
                    fontsize=sfs, color="#555c63")
        if out:
            ax.text(x + w / 2, top - step * (1 + bool(sub)), out,
                    ha="center", va="center", fontsize=sfs, color=GREEN,
                    fontweight="bold")

    def arrow(x1, y1, x2, y2, lw=1.6, col=DARK, dash=False):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=13, lw=lw, color=col,
                                     linestyle="--" if dash else "-"))
    return box, arrow


# ---------------------------------------------------------------- 1. BEV
fig, ax = new_fig(); box, arrow = mk(ax)
ax.set_title("BEV generation — depth-gated IPM with optional LiDAR "
             "(v31/v32)", fontsize=15, fontweight="bold", color=DARK)
box(2, 52, 22, 14, "8 cameras", "432×768 ×8", fc=IN)
box(2, 30, 22, 12, "LiDAR points", "OPTIONAL — zeros =\ncamera-only bit-equal",
    fc=IN)
box(30, 52, 26, 14, "Backbone + Depth", "per-pixel depth softmax\n64 bins, 0–80 m")
box(30, 30, 26, 12, "C6a sharpen", "p' = (1−αm)p + αm·tri(d)\nelementwise only",
    fc=MEM)
box(62, 52, 26, 14, "Depth-gated IPM", "project by K/T, weight by\ndepth prob at true range",
    fc=OP)
box(62, 30, 26, 12, "C6b pillar branch", "raster 400×250×4 → conv\nflag-gated residual",
    fc=MEM)
box(94, 44, 24, 16, "RAW BEV 96ch", "800×500 @0.2 m\nexact geometry", fc=OP)
box(124, 44, 28, 16, "All 12 heads", "+ temporal queue", fc=E2E,
    out="zero optional inputs →\nidentical camera-only net")
arrow(24, 59, 30, 59); arrow(24, 36, 30, 36); arrow(56, 59, 62, 59)
arrow(43, 52, 43, 42); arrow(56, 36, 66, 42, col=AMBER)
arrow(88, 59, 94, 54); arrow(75, 42, 98, 44, col=AMBER)
arrow(118, 52, 124, 52)
fig.savefig("docs/media/detail_bev.png", bbox_inches="tight",
            facecolor="white")

# ----------------------------------------------------- 2. temporal memory
fig, ax = new_fig(); box, arrow = mk(ax)
ax.set_title("Temporal memory & task routing (v29 + B2)", fontsize=15,
             fontweight="bold", color=DARK)
box(2, 46, 22, 14, "RAW BEV (t)", fc=OP)
box(2, 12, 22, 20, "Memory queue", "t−0.4 / −1.2 / −2.8 s\nego-warped by pose θ",
    fc=MEM)
box(32, 34, 30, 16, "B2 slot gate + tfuse3", "per-cell softmax over 4 slots\nzero-init = today's fusion",
    fc=MEM)
box(70, 34, 22, 14, "FUSED BEV", "motion context", fc=MEM)
box(102, 52, 48, 12, "GEOMETRY heads ← RAW", "lanes · 3D boxes · occupancy\n(no moving-object ghosts)",
    fc=OP)
box(102, 30, 48, 12, "MOTION heads ← FUSED", "E2E · forecasting · flow · risk",
    fc=E2E)
box(102, 8, 48, 12, "Motion residual (v30+)", "RAW − warped slot0 → forecast\noncoming reads as signed dipole",
    fc=E2E)
arrow(13, 46, 13, 32); arrow(24, 22, 32, 38); arrow(24, 53, 36, 50)
arrow(62, 42, 70, 41); arrow(24, 56, 102, 58, col=GREEN)
arrow(92, 41, 102, 36, col=AMBER); arrow(24, 50, 102, 14, col=AMBER, dash=True)
ax.text(60, 5, "raw BEV rings back as next frame's history (streaming, TRT-safe)",
        fontsize=9.5, style="italic", color=AMBER, fontweight="bold")
fig.savefig("docs/media/detail_temporal.png", bbox_inches="tight",
            facecolor="white")

# ------------------------------------------------------------- 3. E2E
fig, ax = new_fig(); box, arrow = mk(ax)
ax.set_title("E2E planning stack (v36–v39) + deterministic guardrails (C7)",
             fontsize=15, fontweight="bold", color=DARK)
box(2, 56, 20, 12, "FUSED BEV", fc=MEM)
box(2, 40, 20, 12, "v0 + kin history", "accel / yaw-rate\nfrom memory poses", fc=IN)
box(2, 24, 20, 12, "Intent (E6)", "straight/left/right\nzeros = no nav", fc=IN)
box(28, 48, 26, 16, "B3 attention pool", "K=3 queries × 400\nBEV tokens", fc=E2E)
box(28, 26, 26, 14, "v39 decoupled", "heading φ(t) × speed v(t)\ncumsum composition", fc=E2E)
box(60, 40, 26, 20, "K=3 hypotheses", "ε-WTA · diverse init\n+ conf logits", fc=E2E)
box(92, 40, 26, 20, "v38 selection", "conf − gate·∫risk\nalong each path", fc=E2E,
    out="chosen plan")
box(60, 8, 58, 18, "GUARDRAIL (deterministic)", "spacetime collision · red-light ·\nfeasibility · drivable  →  VETO ⇒ MRM stop",
    fc=SAFE)
box(124, 40, 28, 20, "Output", "path + steer/accel/brake", fc=E2E,
    out="ADE 0.78→0.67 (intent)\nADEc 0.41 record")
arrow(22, 62, 28, 58); arrow(22, 46, 60, 48); arrow(22, 30, 60, 44)
arrow(54, 56, 60, 52); arrow(54, 33, 60, 44); arrow(86, 50, 92, 50)
arrow(118, 50, 124, 50); arrow(105, 40, 95, 26, col=RED)
ax.text(105, 3, "checker runs on RAW-BEV heads = partially independent input path",
        fontsize=9.5, style="italic", color=DARK)
fig.savefig("docs/media/detail_e2e.png", bbox_inches="tight",
            facecolor="white")

# ---------------------------------------------- 4. perception head detail
fig, ax = new_fig(); box, arrow = mk(ax)
ax.set_title("Perception heads — detection, forecasting, unknown, "
             "occupancy", fontsize=15, fontweight="bold",
             color=DARK)
box(2, 46, 20, 14, "RAW BEV", fc=OP)
box(2, 24, 20, 14, "FUSED BEV\n+ motion residual", fc=MEM, fs=10.5)
box(28, 56, 34, 12, "3D box head", "CenterPoint + crossing-yaw\nweight (yaw 4.7°, flip 8%)")
box(28, 42, 34, 12, "Unknown head (v34)", "temporal stem → cones/posts\n0.4 m fixed, occ-blob GT")
box(28, 28, 34, 12, "Forecast + stationary", "det-yaw feature + heading loss\nvehHead 23°, vruHead 63°", fc=E2E)
box(28, 14, 34, 12, "Occupancy + flow", "16z voxels + velocity\nnear-ego FP penalty (v33)")
box(68, 22, 36, 14, "B4 interaction (lite)", "scene token → forecast\ncontext residual", fc=E2E)
box(110, 34, 42, 20, "Outputs", "boxes+unknown / futures /\nvoxels+flow",
    fc=E2E, out="all from ONE BEV;\neach head < 5% compute")
arrow(22, 55, 28, 61); arrow(22, 52, 28, 48); arrow(22, 33, 28, 34)
arrow(22, 50, 28, 20, dash=True)
arrow(22, 30, 68, 28, col=AMBER); arrow(62, 61, 110, 46)
arrow(62, 48, 110, 44); arrow(62, 34, 110, 42); arrow(62, 20, 110, 38)
arrow(104, 29, 110, 40)
fig.savefig("docs/media/detail_heads.png", bbox_inches="tight",
            facecolor="white")
print("saved 4 detail diagrams")

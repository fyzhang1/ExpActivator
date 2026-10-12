"""Figure for task-level activation threshold sensitivity."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from paper_figure_style import apply_paper_style, finish_typography

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "expactivator/results/leaf_proactive_sensitivity"
PAPER = REPO / "manuscript/paper/iclr2027"

PRIMARY = "#176CCB"
SECOND = "#12A9BE"
SOFT = "#CFE3F4"
GRID = "#E7E9ED"
SPINE = "#D9DCE2"
QUIET = "#BCC1CA"
TEXT = "#5F636B"
FIG_W = 5.5

RAMP = LinearSegmentedColormap.from_list("leaf", ["#F2F7FC", SOFT, PRIMARY])

apply_paper_style()


def main(method_name: str = "LEAF") -> None:
    report = json.loads((RESULTS / "sensitivity.json").read_text())
    config = report["protocol"]["config"]
    eta0 = config["semantic_threshold"]
    n_min0 = config["minimum_support"]

    etas = sorted({row["eta"] for row in report["eta_by_n_min"]})
    n_mins = sorted({row["n_min"] for row in report["eta_by_n_min"]})
    auroc = np.full((len(etas), len(n_mins)), np.nan)
    empty = {}
    for row in report["eta_by_n_min"]:
        auroc[etas.index(row["eta"]), n_mins.index(row["n_min"])] = row["auroc"]
        if row["eta"] == eta0:
            empty[row["n_min"]] = row["no_eligible_routine_rate"]

    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, 2.65),
                             gridspec_kw={"width_ratios": [1.45, 1]},
                             layout="constrained")

    ax = axes[0]
    image = ax.imshow(auroc, cmap=RAMP, aspect="auto", origin="lower",
                      vmin=55, vmax=82)
    for i in range(len(etas)):
        for j in range(len(n_mins)):
            value = auroc[i, j]
            ax.text(j, i, f"{value:.1f}", ha="center", va="center",
                    fontsize=4.8,
                    color="white" if value >= 76 else TEXT)
    ax.add_patch(Rectangle((n_mins.index(n_min0) - 0.5, etas.index(eta0) - 0.5),
                           1, 1, fill=False, edgecolor=TEXT, linewidth=1.3,
                           zorder=5))
    ax.set_xticks(range(len(n_mins)), [str(n) for n in n_mins])
    ax.set_yticks(range(len(etas)), [f"{e:.2f}" for e in etas])
    ax.set(xlabel="Minimum total support $n_{\\min}$",
           ylabel="Routine threshold $\\eta$")
    ax.set_title("Activation AUROC (%)", fontsize=8, fontweight="bold",
                 color=TEXT, pad=5)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.02)
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=0, labelsize=6)

    ax = axes[1]
    values = [empty[n] for n in n_mins]
    ax.plot(n_mins, values, color=PRIMARY, linewidth=1.6, marker="o",
            markersize=3.4, markeredgecolor="white", markeredgewidth=0.8,
            zorder=4)
    ax.axvline(n_min0, color=QUIET, linewidth=1.0, linestyle=(0, (4, 3)),
               zorder=1)
    ax.annotate(f"selected $n_{{\\min}}={n_min0}$", (n_min0, 74),
                textcoords="offset points", xytext=(-4, 0), ha="right",
                va="top", fontsize=6.2, color=TEXT, rotation=90)
    ax.annotate(f"{empty[n_min0]:.0f}%", (n_min0, empty[n_min0]),
                textcoords="offset points", xytext=(-5, 5), ha="right",
                fontsize=6.4, color=PRIMARY, fontweight="bold")
    ax.set(xlim=(2, 26), ylim=(0, 80),
           xlabel="Minimum total support $n_{\\min}$",
           ylabel="States with no eligible routine (%)")
    ax.set_title("Why the surface falls away", fontsize=8, fontweight="bold",
                 color=TEXT, pad=5)
    ax.grid(color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)

    finish_typography(fig)
    # Resolve the original layout first, then scale both plot areas uniformly.
    # Keep outer edges fixed so the released width becomes an internal gutter.
    # Treat the heatmap and its colorbar as one group; retain readable font sizes.
    fig.canvas.draw()
    fig.set_layout_engine("none")
    scale = .94
    original = {a: a.get_position().frozen() for a in (axes[0], bar.ax, axes[1])}
    center_y = (original[axes[0]].y0 + original[axes[0]].y1) / 2
    for group, anchor_x in [((axes[0], bar.ax), original[axes[0]].x0),
                            ((axes[1],), original[axes[1]].x1)]:
        for panel in group:
            bounds = original[panel]
            panel.set_position([
                anchor_x + scale * (bounds.x0 - anchor_x),
                center_y + scale * (bounds.y0 - center_y),
                scale * bounds.width, scale * bounds.height,
            ])
    for stem in (PAPER / "figure/leaf_proactive_sensitivity.pdf",
                 RESULTS / "leaf_proactive_sensitivity.png"):
        stem.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(stem, dpi=220, bbox_inches="tight", pad_inches=.01)
    plt.close(fig)
    print("wrote leaf_proactive_sensitivity (pdf in the manuscript figure directory, "
          "png under results)")


if __name__ == "__main__":
    main()

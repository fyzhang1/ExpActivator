"""Figures for context cost and history-size stratification."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from paper_figure_style import apply_paper_style, finish_typography

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "expactivator/results/leaf_context_cost"
PAPER = REPO / "manuscript/paper/iclr2027"

# Figure palette.
PRIMARY = "#176CCB"   # the method
SECOND = "#12A9BE"    # the no-history reference
SOFT = "#CFE3F4"      # dominated region and supporting fills
GRID = "#E7E9ED"      # separators
SPINE = "#D9DCE2"
QUIET = "#BCC1CA"     # categories that are not emphasized
TEXT = "#5F636B"      # all text

ORDER = ["Base", "Recent", "Retrieve", "LLM-UM", "HIM-Agent", "LEAF"]

# The manuscript text width is 5.5 in; figures are authored at that width so
# that \includegraphics[width=\linewidth] does not rescale the type.
FIG_W = 5.5

apply_paper_style()


def save(fig, stem: str, *, compact_border: bool = False) -> None:
    finish_typography(fig)
    for destination in (PAPER / "figure" / f"{stem}.pdf", RESULTS / f"{stem}.png"):
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, dpi=220, bbox_inches="tight", pad_inches=.01 if compact_border else .1)
    plt.close(fig)


def dominated_region(ax, x: float, y: float, xlim, ylim) -> None:
    """Shade the quadrant that costs more than (x) and scores less than (y)."""
    ax.add_patch(Rectangle((x, ylim[0]), xlim[1] - x, y - ylim[0],
                           facecolor=SOFT, alpha=0.55, edgecolor="none",
                           zorder=0))


def pareto(method_name: str = "LEAF") -> None:
    """Accuracy against history-token cost.  Exact values live in the table,
    so the points carry names only and the figure carries the shape."""
    tokens = json.loads((RESULTS / "token_cost.json").read_text())["conditions"]
    history = json.loads((RESULTS / "history_size.json").read_text())
    decomposition = history["entry_decomposition"]
    points = {
        name: {
            "tokens": tokens[name]["mean_history_tokens"],
            "all": decomposition["all_steps"]["ssr"][name],
            "nonentry": decomposition["non_entry_steps"]["ssr"][name],
        }
        for name in ORDER
    }

    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, 2.45), layout="constrained")
    xlim = (-90, 1180)
    panels = [
        ("all", (16, 34), f"All {decomposition['all_steps']['n_steps']:,} steps",
         {"Base": (6, -7), "Recent": (-6, 0), "Retrieve": (0, 7),
          "LLM-UM": (6, 0), "HIM-Agent": (0, 7), "LEAF": (9, 0)}),
        ("nonentry", (14, 30),
         f"{decomposition['non_entry_steps']['n_steps']:,} "
         "within-trajectory actions",
         {"Base": (6, -7), "LEAF": (9, 0)}),
    ]
    for ax, (key, ylim, title, offsets) in zip(axes, panels):
        leaf = points["LEAF"]
        dominated_region(ax, leaf["tokens"], leaf[key], xlim, ylim)
        ax.annotate(f"costs more and scores\nless than {method_name}",
                    (1150, leaf[key] - 0.4), ha="right", va="top", fontsize=5.9,
                    color=TEXT, alpha=0.9, linespacing=1.3)
        base = points["Base"][key]
        ax.axhline(base, color=SECOND, linewidth=1.0, linestyle=(0, (4, 3)),
                   zorder=2)
        ax.annotate("no history", (1150, base), textcoords="offset points",
                    xytext=(0, 3), ha="right", fontsize=6.2, color=SECOND)
        for name in ORDER:
            point = points[name]
            method = name == "LEAF"
            anchor = name == "Base"
            ax.scatter(point["tokens"], point[key], s=44 if method else 22,
                       color=PRIMARY if method else (SECOND if anchor else QUIET),
                       edgecolors="white", linewidths=0.8,
                       zorder=6 if method else 4,
                       marker="D" if method else "o")
            if name not in offsets:
                continue
            dx, dy = offsets[name]
            horizontal = "center" if dx == 0 else ("left" if dx > 0 else "right")
            vertical = "center" if dy == 0 else ("bottom" if dy > 0 else "top")
            ax.annotate(method_name if method else name,
                        (point["tokens"], point[key]),
                        textcoords="offset points", xytext=(dx, dy),
                        ha=horizontal, va=vertical,
                        fontsize=6.6, color=TEXT,
                        fontweight="bold" if method else "normal")
        if key == "nonentry":
            low = min(points[n][key] for n in ORDER[1:-1])
            high = max(points[n][key] for n in ORDER[1:-1])
            ax.annotate("Recent, Retrieve, LLM-UM, HIM-Agent:\n"
                        "no gain over no history",
                        (640, low - 0.4), textcoords="offset points",
                        xytext=(0, -6), ha="center", va="top", fontsize=6.2,
                        color=TEXT, linespacing=1.3)
        ax.set(xlim=xlim, ylim=ylim, xlabel="History tokens per step",
               ylabel="Step success rate (%)")
        ax.set_title(title, fontsize=8, fontweight="bold", color=TEXT, pad=5)
        ax.grid(color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
    save(fig, "leaf_context_cost_pareto", compact_border=True)


def stratification(method_name: str = "LEAF", only_all: bool = False) -> None:
    report = json.loads((RESULTS / "history_size.json").read_text())

    # Base is the no-history reference, HIM-Agent is the benchmark's own agent,
    # and the remaining record-based baselines share an ordered neutral ramp.
    style = {
        "Base": {"facecolor": "white", "edgecolor": QUIET, "linewidth": 0.8},
        "Recent": {"facecolor": "#E7E9ED", "edgecolor": "white", "linewidth": 0.5},
        "Retrieve": {"facecolor": "#D9DCE2", "edgecolor": "white", "linewidth": 0.5},
        "LLM-UM": {"facecolor": QUIET, "edgecolor": "white", "linewidth": 0.5},
        "HIM-Agent": {"facecolor": SECOND, "edgecolor": "white", "linewidth": 0.5},
        "LEAF": {"facecolor": PRIMARY, "edgecolor": "white", "linewidth": 0.5},
    }
    panels = [
        ("all_steps", "All steps", 45.0, 41.5),
        ("non_entry_steps", "Within-trajectory execution", 41.0, 38.0),
    ]
    if only_all:
        panels = panels[:1]

    fig, axes = plt.subplots(len(panels), 1, figsize=(FIG_W, 2.35 if only_all else 4.25),
                             layout="constrained", sharex=True, squeeze=False)
    axes = axes[:, 0]
    width = 0.135
    for ax, (slice_name, title, top, annotation_y) in zip(axes, panels):
        buckets = [b for b in report["quartiles"][slice_name]
                   if b["label"] != "All"]
        x = np.arange(len(buckets))
        for position, name in enumerate(ORDER):
            values = [b["ssr"][name] for b in buckets]
            offset = (position - (len(ORDER) - 1) / 2) * width
            label = method_name if name == "LEAF" else name
            ax.bar(x + offset, values, width * 0.9,
                   label=label if ax is axes[0] else None, zorder=3,
                   **style[name])
            for i, value in enumerate(values):
                ax.annotate(f"{value:.1f}", (x[i] + offset, value),
                            textcoords="offset points", xytext=(0, 1.8),
                            ha="center", va="bottom", rotation=90,
                            fontsize=5.6, color=TEXT,
                            fontweight="bold" if name == "LEAF" else "normal")
        for i, bucket in enumerate(buckets):
            gain = bucket["gain_vs_strongest"]
            ax.annotate(
                f"{gain['delta']:+.1f} [{gain['ci95'][0]:.1f}, "
                f"{gain['ci95'][1]:.1f}]\nvs {bucket['strongest_baseline']}",
                (x[i], annotation_y), ha="center", va="center", fontsize=5.9,
                color=PRIMARY, zorder=4, linespacing=1.35)
        ax.set(ylim=(0, top), xlim=(-0.55, len(buckets) - 0.45),
               ylabel="Step success rate (%)")
        ax.set_yticks(np.arange(0, top, 10))
        if not only_all:
            ax.set_title(title, fontsize=8, fontweight="bold", color=TEXT, pad=4)
        ax.grid(axis="y", color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
    buckets = [b for b in report["quartiles"]["all_steps"] if b["label"] != "All"]
    axes[-1].set_xticks(np.arange(len(buckets)))
    axes[-1].set_xticklabels([
        f"{b['label']}\n{b['history_records_min']}\u2013"
        f"{b['history_records_max']} records"
        for b in buckets
    ], fontsize=6.8, linespacing=1.4)
    axes[-1].tick_params(axis="x", length=0)
    axes[-1].set_xlabel("Users grouped by history size (quartiles)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=len(ORDER),
               frameon=False, fontsize=7, handlelength=1.2, handleheight=1.0,
               columnspacing=1.1, handletextpad=0.5)
    save(fig, "leaf_history_size_all" if only_all else "leaf_history_size")


if __name__ == "__main__":
    pareto()
    stratification()
    stratification(only_all=True)
    print("wrote leaf_context_cost_pareto and leaf_history_size "
          "(pdf in the manuscript directory, png under results)")

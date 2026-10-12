"""How often step-level activation moves between past trajectories.

For every episode, count the distinct historical trajectories that supply its
within-trajectory references, and the share of consecutive steps whose
reference comes from a different trajectory than the previous one.  Computed
from the sealed evidence files of all four backbones; no model is run.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paper_figure_style import apply_paper_style, finish_typography

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "expactivator/results"
OUT = RESULTS / "leaf_reference_switching"
PAPER = REPO / "manuscript/paper/iclr2027/figure"
BACKBONES = {"qwen3_vl_8b": "Qwen3-VL-8B", "mai_ui_8b": "MAI-UI-8B",
             "qwen35_9b": "Qwen3.5-9B", "gui_owl_1_5_8b": "GUI-Owl-1.5-8B"}
COUNTS = [1, 2, 3, 4, 5]                      # 5 means five or more
POSITIONS = [(2, 2, "2"), (3, 4, "3–4"), (5, 7, "5–7"),
             (8, 11, "8–11"), (12, 99, "12+")]

PRIMARY, SECOND, SOFT = "#176CCB", "#12A9BE", "#CFE3F4"
GRID, SPINE, TEXT = "#E7E9ED", "#D9DCE2", "#2E3238"


def statistics() -> dict:
    report = {}
    for slug, label in BACKBONES.items():
        sources = json.loads((RESULTS / f"grounded/{slug}_contexts.json").read_text())["sources"]
        episodes = defaultdict(dict)
        for step_id, source in sources.items():
            episode, position = step_id.rsplit("_", 1)
            if int(position) > 0:
                episodes[episode][int(position)] = source.get("time")
        distinct = [len(set(steps.values())) for steps in episodes.values()]
        histogram = {c: sum(min(n, 5) == c for n in distinct) / len(distinct) * 100
                     for c in COUNTS}
        switches = total = 0
        by_position = {}
        for lo, hi, name in POSITIONS:
            moved = seen = 0
            for steps in episodes.values():
                for t, source in steps.items():
                    if lo <= t <= hi and (t - 1) in steps:
                        seen += 1
                        moved += source != steps[t - 1]
            by_position[name] = moved / seen * 100
            switches += moved
            total += seen
        report[label] = {
            "episodes": len(distinct),
            "mean_distinct_trajectories": float(np.mean(distinct)),
            "at_least_two": sum(n >= 2 for n in distinct) / len(distinct) * 100,
            "switch_rate": switches / total * 100,
            "histogram": histogram,
            "switch_by_position": by_position,
        }
    return report


def _gradient(ax, extent, clip, top, bottom, alpha=(1.0, 1.0), zorder=3):
    """Paint a vertical colour gradient inside ``clip`` (a patch in data coords)."""
    from matplotlib.colors import to_rgba
    rows = 256
    t = np.linspace(1, 0, rows)[:, None]          # 1 at the top row, 0 at the bottom
    c0, c1 = np.array(to_rgba(bottom)), np.array(to_rgba(top))
    image = c0 + (c1 - c0) * t[..., None]
    image[..., 3] = alpha[1] + (alpha[0] - alpha[1]) * t
    im = ax.imshow(image.reshape(rows, 1, 4), extent=extent, aspect="auto",
                   origin="upper", interpolation="bicubic", zorder=zorder)
    im.set_clip_path(clip)
    return im


def plot(report: dict) -> None:
    from matplotlib.patches import FancyBboxPatch, Polygon
    apply_paper_style()
    labels = list(report)
    W, H = 5.5, 2.15
    fig = plt.figure(figsize=(W, H))
    left = fig.add_axes([0.085, 0.2, 0.465, 0.62])
    right = fig.add_axes([0.655, 0.2, 0.335, 0.62])
    DEEP, LIGHT = "#0F4F9C", "#6FA8E6"

    # ---- left: how many past trajectories one episode draws on ----
    hist = np.array([[report[b]["histogram"][c] for c in COUNTS] for b in labels])
    mean, lo, hi = hist.mean(0), hist.min(0), hist.max(0)
    x = np.arange(len(COUNTS))
    width, ylim = 0.56, (0, 46)
    left.set(xlim=(-0.55, len(COUNTS) - 0.45), ylim=ylim)
    ax_w, ax_h = 0.465 * W, 0.62 * H
    aspect = ((ylim[1] - ylim[0]) / ax_h) / ((len(COUNTS) - 0.1) / ax_w)

    # a pale panel behind the episodes that use two or more trajectories
    left.add_patch(FancyBboxPatch((0.6, 0.0), 3.8, 43.5,
                                  boxstyle="round,pad=0,rounding_size=0.18",
                                  mutation_aspect=aspect, facecolor="#EEF4FB",
                                  edgecolor="none", zorder=0))
    at_least_two = np.mean([report[b]["at_least_two"] for b in labels])
    left.text(2.5, 41.8, f"{at_least_two:.0f}% of episodes draw on two or more",
              ha="center", va="top", fontsize=6.6, fontweight="bold", zorder=5)

    for i, value in enumerate(mean):
        color = SOFT if i == 0 else PRIMARY
        left.add_patch(plt.Rectangle((x[i] - width / 2, 0), width, value,
                                     facecolor=color, edgecolor="none", zorder=3))
    left.errorbar(x, mean, yerr=[mean - lo, hi - mean], fmt="none", ecolor="#9AA0A8",
                  elinewidth=0.6, capsize=1.6, capthick=0.6, zorder=4)
    for i, value in enumerate(mean):
        left.annotate(f"{value:.0f}%", (x[i], hi[i]), textcoords="offset points",
                      xytext=(0, 2.5), ha="center", va="bottom", fontsize=6.4,
                      fontweight="bold" if i else "normal")
    left.set_xticks(x, ["1", "2", "3", "4", "5+"])
    left.set_yticks([0, 10, 20, 30, 40])
    left.set_ylabel("Episodes (%)")
    left.set_xlabel("Past trajectories supplying one episode")
    left.tick_params(length=0)
    left.grid(axis="y", color=GRID, linewidth=0.6)
    left.set_axisbelow(True)
    left.spines["left"].set_visible(False)
    left.set_title("Each episode draws on several past trajectories", fontsize=7.4,
                   fontweight="bold", pad=6, loc="left")

    # ---- right: how often the reference switches along the episode ----
    names = [p[2] for p in POSITIONS]
    rate = np.array([[report[b]["switch_by_position"][n] for n in names] for b in labels])
    avg = rate.mean(0)
    xr = np.arange(len(names))
    floor = 35
    right.set(xlim=(-0.35, len(names) - 0.65), ylim=(floor, 72))
    area = Polygon(np.r_[np.c_[xr, avg], [[xr[-1], floor], [xr[0], floor]]],
                   closed=True, facecolor="none", edgecolor="none")
    right.add_patch(area)
    _gradient(right, (xr[0], xr[-1], floor, avg.max()), area, top=PRIMARY, bottom=PRIMARY,
              alpha=(0.28, 0.0), zorder=1)
    right.fill_between(xr, rate.min(0), rate.max(0), color=SOFT, linewidth=0, zorder=2)
    right.plot(xr, avg, color=PRIMARY, linewidth=1.8, zorder=3, solid_capstyle="round")
    right.scatter(xr, avg, s=18, color="white", edgecolor=PRIMARY, linewidth=1.2, zorder=4)
    for i, value in enumerate(avg):
        right.annotate(f"{value:.0f}%", (xr[i], rate.max(0)[i]), textcoords="offset points",
                       xytext=(0, 3.5), ha="center", va="bottom", fontsize=6.4,
                       fontweight="bold")
    overall = np.mean([report[b]["switch_rate"] for b in labels])
    right.axhline(overall, color=SECOND, linewidth=0.8, linestyle=(0, (3, 2)), zorder=2)
    right.text(-0.22, overall - 1.2, f"all steps {overall:.0f}%", ha="left",
               va="top", fontsize=6.2, zorder=5)
    right.set_xticks(xr, names)
    right.set_yticks([40, 50, 60, 70])
    right.set_ylabel("Steps that switch (%)")
    right.set_xlabel("Step position within the episode")
    right.tick_params(length=0)
    right.grid(axis="y", color=GRID, linewidth=0.6)
    right.set_axisbelow(True)
    right.spines["left"].set_visible(False)
    right.set_title("The reference switches throughout the episode", fontsize=7.4,
                    fontweight="bold", pad=6, loc="left")

    finish_typography(fig)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(PAPER / "leaf_reference_switching.pdf", bbox_inches="tight", pad_inches=0.01)
    fig.savefig(OUT / "leaf_reference_switching.png", dpi=250, bbox_inches="tight",
                pad_inches=0.01)
    plt.close(fig)


def main() -> None:
    report = statistics()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "statistics.json").write_text(json.dumps(report, indent=1))
    for label, entry in report.items():
        print(f"{label:16s} >=2: {entry['at_least_two']:.1f}%  mean {entry['mean_distinct_trajectories']:.2f}"
              f"  switch {entry['switch_rate']:.1f}%")
    plot(report)
    print("wrote leaf_reference_switching.pdf / .png")


if __name__ == "__main__":
    main()

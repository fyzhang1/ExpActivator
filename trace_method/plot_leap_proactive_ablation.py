"""Plot complete activation curves and paired discrimination contrasts."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from paper_figure_style import apply_paper_style, finish_typography, FIG_W

ROOT = Path(__file__).resolve().parents[1] / "trace_method/results/leap_ablations"

# Paper palette: deep blue key series, cyan emphasis, light blue weak data, grays for guides
COLORS = {"full": "#176CCB", "no_time": "#12A9BE", "no_scenario": "#CFE3F4"}
FILL = "#CFE3F4"
INK = "#1F2937"
MUTED = "#D9DCE2"
GRID = "#E7E9ED"


def main(method_name="LEAF", output_stem=None):
    output_stem = Path(output_stem) if output_stem is not None else ROOT / "proactive_activation_curves"
    data = json.loads((ROOT / "proactive/analysis.json").read_text())
    names = {"full": method_name, "no_time": "No time weighting", "no_scenario": "No scenario condition"}
    apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, 2.65), gridspec_kw={"width_ratios": [1.2, 1]}, layout="constrained")

    ax = axes[0]
    ax.plot([0, 100], [0, 100], color=MUTED, linewidth=1.1, linestyle=(0, (5, 4)), zorder=1)
    for variant in names:
        curve = json.loads((ROOT / f"proactive/{variant}_roc_curve.json").read_text())
        m = data["metrics"][variant]
        xs = [100 * r["false_alarm_rate"] for r in curve]
        ys = [100 * r["recall"] for r in curve]
        if variant == "full":
            ax.fill_between(xs, ys, color=FILL, alpha=0.35, zorder=1, linewidth=0)
        ax.plot(xs, ys, color=COLORS[variant],
                linewidth={"full": 2.4, "no_time": 2.0, "no_scenario": 2.4}[variant],
                solid_capstyle="round", zorder=3 if variant == "full" else 2,
                label=f"{names[variant]} (AUC {100*m['auroc']:.1f})")
        ax.scatter(100 * m["false_alarm_rate"], 100 * m["recall"], color=COLORS[variant], s=42,
                   edgecolors="white", linewidths=1.0, zorder=5)
    ax.set(xlim=(0, 100), ylim=(0, 102), xlabel="False-alarm rate (%)", ylabel="Recall (%)")
    ax.set_title("Task-level activation ROC", fontsize=8, fontweight="bold", pad=5)
    ax.legend(loc="lower right", frameon=False, fontsize=5.8, handlelength=1.6, borderaxespad=0.3)
    ax.grid(color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)

    ax = axes[1]
    for position, variant in enumerate(("no_scenario", "no_time")):
        c = data["full_minus_ablation"][variant]["auroc"]
        center = 100 * c["delta"]
        low, high = np.array(c["delta_ci95"]) * 100
        ax.errorbar(center, position, xerr=[[center - low], [high - center]], fmt="o",
                    color=COLORS[variant], ecolor=COLORS[variant], elinewidth=2.0,
                    capsize=4.5, capthick=1.8, markersize=7.5,
                    markeredgecolor="white", markeredgewidth=1.1)
        ax.annotate(f"{center:+.2f} [{low:+.2f}, {high:+.2f}]", (center, position),
                    xytext=(0, 12), textcoords="offset points", ha="center",
                    fontsize=6.1)
    ax.axvline(0, color=MUTED, linestyle=(0, (4, 3)), linewidth=1.1)
    ax.set(yticks=[0, 1], yticklabels=["No scenario\ncondition", "No time\nweighting"],
           ylim=(-.65, 1.65), xlim=(-4.5, 14), xlabel=f"{method_name} minus control (AUROC pp)")
    ax.set_title("Effect on activation AUROC", fontsize=8, fontweight="bold", pad=5)
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)

    finish_typography(fig)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=.01)
    fig.savefig(output_stem.with_suffix(".png"), dpi=220, bbox_inches="tight", pad_inches=.01)
    plt.close(fig)


if __name__ == "__main__":
    main()

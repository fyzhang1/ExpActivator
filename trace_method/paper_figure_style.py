"""Shared typography for manuscript figures; require real Arial font files."""
from pathlib import Path

import matplotlib as mpl
from matplotlib import font_manager
from matplotlib.colors import to_rgba
from matplotlib.text import Text

FIG_W = 5.5
INK = "#202020"
TEXT = "#454545"
GRID = "#E7E9ED"
SPINE = "#D9DCE2"


def apply_paper_style():
    # User-installed Microsoft core fonts; fonts are not bundled in the repo.
    directory = Path.home() / ".local/share/fonts/arial"
    for path in directory.glob("*.[tT][tT][fF]"):
        font_manager.fontManager.addfont(str(path))
    for style, weight in [("normal", "normal"), ("normal", "bold"),
                          ("italic", "normal"), ("italic", "bold")]:
        prop = font_manager.FontProperties(family="Arial", style=style, weight=weight)
        path = font_manager.findfont(prop, fallback_to_default=False)
        if font_manager.FontProperties(fname=path).get_name() != "Arial":
            raise RuntimeError("Install Arial regular, bold, italic and bold italic before plotting.")
    mpl.rcParams.update({
        "font.family": "Arial", "font.sans-serif": ["Arial"],
        "font.serif": ["Arial"], "font.monospace": ["Arial"],
        "font.cursive": ["Arial"], "font.size": 7.0, "font.weight": "normal",
        "text.usetex": False, "text.color": TEXT,
        "axes.labelcolor": TEXT, "axes.labelsize": 7.5,
        "axes.titlesize": 8.0, "axes.titleweight": "bold",
        "axes.edgecolor": SPINE, "axes.linewidth": .8,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": TEXT, "ytick.color": TEXT,
        "xtick.labelsize": 7.0, "ytick.labelsize": 7.0,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.major.width": .8, "ytick.major.width": .8,
        "legend.fontsize": 7.0, "legend.labelcolor": TEXT,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "mathtext.fontset": "custom", "mathtext.rm": "Arial",
        "mathtext.it": "Arial:style=italic", "mathtext.bf": "Arial:weight=bold",
        "mathtext.sf": "Arial", "mathtext.tt": "Arial", "mathtext.cal": "Arial",
        "mathtext.fallback": None,
    })
    if "mathtext.bfit" in mpl.rcParams:
        mpl.rcParams["mathtext.bfit"] = "Arial:style=italic:weight=bold"


def finish_typography(fig):
    """Normalize text only; retain data colors and white text on dark heatmaps."""
    for item in fig.findobj(match=Text):
        item.set_fontfamily("Arial")
        if to_rgba(item.get_color())[:3] == (1., 1., 1.):
            continue
        weight = item.get_fontweight()
        bold = weight in ("bold", "semibold", "demibold", "heavy", "black") or (
            isinstance(weight, (float, int)) and weight >= 600)
        item.set_color(INK if bold else TEXT)


def align_panel_titles(fig, axes, y=.94):
    """Give fixed-layout panels one shared title baseline in PDF and PNG."""
    for ax in axes:
        title = ax.get_title()
        ax.set_title("")
        bounds = ax.get_position()
        fig.text(bounds.x0 + bounds.width / 2, y, title, ha="center", va="center",
                 fontsize=8, fontweight="bold", color=INK)

"""Regenerate the paper's activation figure using its global LaTeX method name."""
from pathlib import Path
import re
import runpy
import shutil

REPO = Path(__file__).resolve().parents[1]
PAPER = REPO / "manuscript/paper/iclr2027"
PREVIEWS = REPO / "expactivator/results/leap_ablations/paper_previews"


def main():
    source = (PAPER / "iclr2027_conference.tex").read_text()
    method_name = re.search(r"\\newcommand\{\\methodname\}\{([^{}]+)\}", source).group(1)
    plot = runpy.run_path(str(REPO / "expactivator/plot_leap_proactive_ablation.py"))
    PREVIEWS.mkdir(parents=True, exist_ok=True)
    plot["main"](method_name=method_name, output_stem=PREVIEWS / "leap_proactive_activation_curves")
    (PAPER / "figure").mkdir(exist_ok=True)
    shutil.copy2(PREVIEWS / "leap_proactive_activation_curves.pdf",
                 PAPER / "figure/leap_proactive_activation_curves.pdf")


if __name__ == "__main__":
    main()

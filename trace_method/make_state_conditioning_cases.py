"""Illustrative, outcome-selected cases using the unaltered benchmark screens.

Screens and action locations were visually inspected. These examples are not a
random sample and do not estimate how often a mechanism occurs.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "trace_method/results/state_conditioning"
MODEL = ROOT / "qwen3_vl_8b"
HISTORY = Path("/storage/zhangfuyao/datasets/fingertip20k")


def read(path):
    return json.loads(path.read_text())


def circle(ax, point, width, height, color, normalized=True, radius=.047):
    x, y = point
    if normalized:
        x, y = x / 999 * width, y / 999 * height
    ax.add_patch(Circle((x, y), width * radius, fill=False, edgecolor="white", linewidth=4.5))
    ax.add_patch(Circle((x, y), width * radius, fill=False, edgecolor=color, linewidth=2.4))


def action_text(action):
    if action.get("action") == "click":
        return f"click{tuple(action['coordinate'])}"
    return action.get("action", "unparsed")


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    targets = {r["id"]: r for r in read(REPO / "data/personal_execution/personal_execution.json")}
    contexts = {v: read(MODEL / f"{v}_contexts.json") for v in ("top1_ordinal", "top1_state")}
    outputs = {v: {r["episode_id"]: r for r in read(MODEL / f"{v}_policy_gpu0.json")["outputs"]}
               for v in contexts}
    relevance = {r["episode_id"]: r for r in read(MODEL / "task_relevance_inputs.json")["rows"]}
    cases = [
        ("31_20250424_132715_4", "fixed_history_case",
         "Query: Check the match results.    Historical goal: Open Hupu and check the match results.",
         "State matching selects the NBA-tab reference; the frozen policy matches the recorded click.",
         {"top1_ordinal": 0, "top1_state": 1}),
        ("13_20250421_210857_5", "evidence_policy_gap_case",
         "Query: Check the next 15 days of weather.    Historical goal: Check the next 15 days of weather.",
         "The matched reference selects the line view, but the frozen policy waits; the recorded action is a click.",
         {"top1_ordinal": 0, "top1_state": 0}),
    ]
    audit = {"selection": "Illustrative selection after inspecting evidence and policy outcomes; not a prevalence estimate.",
             "visual_review": "Assistant inspection of current and source screens, not an independent human annotation.",
             "cases": [],
             "rejected_illustration": {
                 "step_id": "13_20250413_230015_7",
                 "reason": "The target click label is (85,1445) on a 1080x2400 screen, but the visible like icon is near the bottom (approximately y=2340). The apparent metric win cannot be shown as a verified current-screen click success.",
                 "quantitative_treatment": "Retained unchanged in every benchmark aggregate; no target relabeling or sample deletion.",
             }}
    for key, filename, heading, conclusion, expected in cases:
        target = targets[key]
        episode = key.rsplit("_", 1)[0]
        a, b = (contexts[v]["sources"][key] for v in ("top1_ordinal", "top1_state"))
        assert a["train_row_index"] == b["train_row_index"]
        for variant, score in expected.items():
            assert outputs[variant][key]["sr"] == score
        paths = [HISTORY / target["img_path"].removeprefix("./data/Fingertip/"),
                 Path(a["image_path"]), Path(b["image_path"])]
        fig, axes = plt.subplots(1, 3, figsize=(10.5, 8.7))
        fig.subplots_adjust(left=.025, right=.975, top=.86, bottom=.20, wspace=.10)
        titles = [f"Current screen: non-entry step {key.rsplit('_', 1)[1]}",
                  f"Ordinal reference: history step {a['history_state_index'] + 1}",
                  f"State reference: history step {b['history_state_index'] + 1}"]
        sizes = []
        for ax, path, title in zip(axes, paths, titles):
            with Image.open(path) as source:
                im = source.convert("RGB")
                sizes.append(im.size)
                ax.imshow(im)
            ax.set_title(title, fontsize=10, pad=8)
            ax.axis("off")
        for i, (variant, color) in enumerate((("top1_ordinal", "#BD4B3C"), ("top1_state", "#126C8A")), 1):
            pred = outputs[variant][key]["output"]
            ref = contexts[variant]["contexts"][key]["retrieved_action"]
            if pred.get("action") == "click":
                circle(axes[0], pred["coordinate"], *sizes[0], color)
            if ref.get("action") == "click":
                circle(axes[i], ref["coordinate"], *sizes[i], color)
            axes[i].text(.5, -.025, f"Reference: {action_text(ref)}\nPolicy: {action_text(pred)}; benchmark SR={outputs[variant][key]['sr']}",
                         transform=axes[i].transAxes, ha="center", va="top", fontsize=9, color=color)
        label_point = tuple(map(int, re.search(r"coordinates=\((\d+),\s*(\d+)\)", target["answer"]).groups()))
        circle(axes[0], label_point, *sizes[0], "#31823B", normalized=False, radius=.07)
        axes[0].text(.5, -.025, "Green: recorded target\nRed / blue: policy click",
                     transform=axes[0].transAxes, ha="center", va="top", fontsize=9)
        fig.suptitle(heading, fontsize=11, x=.5, y=.95)
        fig.text(.5, .08, conclusion, ha="center", fontsize=10)
        fig.text(.5, .045, "Both references come from the same historical episode. Coordinates in labels use the policy's 0–999 frame.",
                 ha="center", fontsize=9, color="#444444")
        for extension in ("pdf", "png"):
            fig.savefig(ROOT / f"{filename}.{extension}", dpi=200, bbox_inches="tight")
        plt.close(fig)
        audit["cases"].append({"step_id": key, "task_descriptions": relevance[episode],
            "source_episode_id": f"{a['user_id']}_{a['time']}",
            "source_indices_zero_based": [a["history_state_index"], b["history_state_index"]],
            "policy_outputs": {v: outputs[v][key]["output"] for v in contexts},
            "policy_scores": expected, "target_action": target["answer"],
            "screens": [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths],
            "interpretation": conclusion})
    (ROOT / "qualitative_case_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print(ROOT / "qualitative_case_audit.json")


if __name__ == "__main__":
    main()

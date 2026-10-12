"""Render a source-backed LEAF case. Screens are recorded, not policy rollouts."""
from pathlib import Path
import hashlib
import json
import re
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "trace_method/results"
OUT = RESULTS / "english_case_candidates"
EPISODE = "43_20250416_181150"
DATA = Path("/storage/zhangfuyao/datasets/fingertip20k")
BLUE, INK, MUTED, RED = "#2166B5", "#172B44", "#59687B", "#AD433D"


def read(p):
    return json.loads(p.read_text())


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def action(a):
    return "click({}, {})".format(*a["coordinate"]) if a["action"] == "click" else a["action"]


def main():
    name = re.search(r"\\newcommand\{\\methodname\}\{([^}]+)\}",
                     (REPO / "manuscript/paper/iclr2027/iclr2027_conference.tex").read_text()).group(1)
    targets = {int(r["id"].rsplit("_", 1)[1]): r for r in read(
        REPO / "data/personal_execution/personal_execution.json")
        if r["id"].rsplit("_", 1)[0] == EPISODE}
    paths = {i: DATA / r["img_path"].removeprefix("./data/Fingertip/")
             for i, r in targets.items() if i > 0}
    files = {name: "grounded/qwen3_vl_8b_grounded_policy_gpu0.json",
             "Previous screen": "leap_ablations/refresh/delay1_policy_gpu0.json",
             "Update every four steps": "leap_ablations/refresh/refresh4_policy_gpu0.json",
             "HIM-Agent": "paper_baselines/qwen3_vl_8b_him_agent_gpu0.json"}
    results = {k: {r["episode_id"]: r for r in read(RESULTS / p)["outputs"]}
               for k, p in files.items()}
    context_path = RESULTS / "grounded/qwen3_vl_8b_contexts.json"
    context = read(context_path)
    assert context["protocol"]["trajectory_top_k"] == 8
    steps = (10, 11)
    selected = {i: context["sources"][targets[i]["id"]] for i in steps}
    assert sorted(paths) == list(range(1, 12))
    assert [int(paths[i].stem) for i in sorted(paths)] == sorted(int(p.stem) for p in paths.values())
    for i in steps:
        pred = results[name][targets[i]["id"]]
        assert pred["grounded_evidence"] == context["contexts"][targets[i]["id"]]
        assert pred["output"] == pred["grounded_evidence"]["retrieved_action"]
        assert pred["sr"] == 1
        assert all(results[k][targets[i]["id"]]["sr"] == 0 for k in files if k != name)
    excerpts = {10: "Click on the D2 route to view the nearest bus schedule.",
                11: "based on the historical evidence, the task has been completed."}
    for i, excerpt in excerpts.items():
        assert excerpt in results[name][targets[i]["id"]]["raw_output"]

    copies = OUT / "screenshots" / EPISODE
    copies.mkdir(parents=True, exist_ok=True)
    image_index = []
    for label, src in [(f"step_{i:02d}", paths[i]) for i in sorted(paths)] + [
            (f"selected_history_for_step_{i:02d}", Path(selected[i]["image_path"])) for i in steps]:
        dst = copies / f"{label}_{src.name}"
        shutil.copy2(src, dst)
        assert digest(src) == digest(dst)
        image_index.append({"role": label, "source": str(src), "copy": str(dst), "sha256": digest(src)})
    (copies / "index.json").write_text(json.dumps(image_index, indent=2) + "\n")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "pdf.fonttype": 42})
    fig = plt.figure(figsize=(18, 14), facecolor="white")

    def text(x, y, s, size=11, color=INK, **kw):
        return fig.text(x, y, s, fontsize=size, color=color, **kw)

    def box(x, y, w, h, color="#EBF5FF"):
        fig.add_artist(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.006",
                       transform=fig.transFigure, facecolor=color, edgecolor="#C7DCF2", linewidth=.8, zorder=0))

    def arrow(x1, y1, x2, y2):
        fig.add_artist(FancyArrowPatch((x1, y1), (x2, y2), transform=fig.transFigure,
                       arrowstyle="-|>", mutation_scale=15, linewidth=1.3, color=BLUE))

    def screen(path, bounds, point=None):
        ax = fig.add_axes(bounds)
        with Image.open(path) as im:
            w, h = im.size
            ax.imshow(im)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#B6C9DF")
        if point:
            for color, lw in [("white", 4), (BLUE, 2)]:
                ax.add_patch(Circle((point[0] / 999 * w, point[1] / 999 * h), w * .065,
                                    fill=False, edgecolor=color, linewidth=lw))
        return ax

    text(.035, .963, f"{name}: update the historical reference as the screen changes", 23, fontweight="bold")
    text(.035, .934, "Task (translated): Check the next bus departures from UTown to PGPR.", 13)
    text(.035, .908, "NUS NextBus  |  Qwen3-VL-8B  |  Three consecutive recorded observations", 11, MUTED)
    for j, i in enumerate((9, 10, 11)):
        x = .035 + j * .205
        title = {9: "Building search", 10: "Recommended routes", 11: "Route details"}[i]
        text(x, .879, f"Step {i}  |  {title}", 12, fontweight="bold")
        pred = results[name][targets[i]["id"]]
        ax = screen(paths[i], [x, .575, .173, .288], pred["output"].get("coordinate") if i == 10 else None)
        text(x, .553, f"{name}: {action(pred['output'])}", 11)
        text(x, .535, f"Benchmark SR = {pred['sr']}", 10, BLUE if pred["sr"] else RED)
        if j < 2:
            arrow(x + .177, .72, x + .199, .72)

    box(.666, .555, .294, .307)
    text(.681, .836, "Same screens, different references", 13, fontweight="bold")
    text(.681, .81, "Benchmark step score (SR)", 10, MUTED)
    text(.87, .782, "10", 11, fontweight="bold")
    text(.924, .782, "11", 11, fontweight="bold")
    for j, k in enumerate(files):
        y = .752 - j * .032
        text(.681, y, k, 10.4, BLUE if k == name else INK, fontweight="bold" if k == name else "normal")
        for i, x in [(10, .875), (11, .929)]:
            sr = results[k][targets[i]["id"]]["sr"]
            text(x, y, str(sr), 12, BLUE if sr else RED)
    text(.681, .599, "With stale references, both update ablations\nmisclick at step 10 and keep clicking at step 11.", 10, MUTED, linespacing=1.5)

    text(.035, .493, "Step-level activation: from selected history to policy output", 17, fontweight="bold")
    text(.035, .467, "Current-screen features select a historical state from eight task-relevant histories at each step.", 11, MUTED)
    text(.035, .435, "Selected historical screen", 12, fontweight="bold")
    text(.237, .435, "Visual similarity", 12, fontweight="bold")
    text(.431, .435, "Injected action reference", 12, fontweight="bold")
    text(.703, .435, "Actual model output", 12, fontweight="bold")

    for row, i in enumerate(steps):
        y = .234 if row == 0 else .052
        source = selected[i]
        pred = results[name][targets[i]["id"]]
        ref = pred["grounded_evidence"]["retrieved_action"]
        screen(Path(source["image_path"]), [.035, y + .003, .101, .155], ref.get("coordinate"))
        text(.146, y + .132, f"Step {i}", 12, BLUE, fontweight="bold")
        text(.146, y + .106, "Apr 8" if i == 10 else "Apr 14", 10, MUTED)
        text(.146, y + .074, "A1 route" if i == 10 else "D2 details", 10)
        text(.146, y + .052, "click" if i == 10 else "finished()", 10, BLUE)
        arrow(.207, y + .082, .229, y + .082)
        box(.239, y + .027, .144, .111)
        text(.311, y + .103, f"{source['visual_similarity']:.4f}", 22, BLUE, ha="center")
        text(.311, y + .072, "current vs. history", 10, MUTED, ha="center")
        text(.311, y + .047, "retrieval score", 10, MUTED, ha="center")
        arrow(.388, y + .082, .421, y + .082)
        box(.433, y + .027, .232, .111)
        code = ('"retrieved_action": {\n  "action": "click",\n  "coordinate": [283, 289]\n}' if i == 10 else
                '"retrieved_action": {\n  "action": "terminate"\n}')
        text(.445, y + .113, code, 10.8, va="top", family="DejaVu Sans Mono", linespacing=1.5)
        arrow(.674, y + .082, .695, y + .082)
        box(.706, y + .027, .254, .111)
        text(.719, y + .108, f"{action(pred['output'])}     SR = 1", 13, BLUE, fontweight="bold")
        text(.719, y + .082, "Model output excerpt (verbatim):", 9.6, MUTED)
        quote = ('"Click on the D2 route to view\nthe nearest bus schedule."' if i == 10 else
                 '"based on the historical evidence,\nthe task has been completed."')
        text(.719, y + .06, quote, 10.5, va="top", linespacing=1.4)
    text(.431, .402, "Policy also receives: current screenshot + task query + recorded action history.", 10, MUTED)
    text(.239, .224, "New screen, new reference: route selection (click) becomes task completion (terminate).", 11, BLUE)
    text(.035, .026, "Historical images and scores are shown for inspection; only the action evidence is injected. Coordinates use the 0–999 frame.", 9.5, MUTED)
    text(.035, .01, "Arrows above show recording order, not a policy rollout. SR is the recorded step score. Original screenshot content is preserved.", 9.5, MUTED)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"leaf_reference_case.{ext}", dpi=210)
    plt.close(fig)

    caption = (
        "Screen-conditioned reference updates in a NUS NextBus task. Three consecutive recorded observations are shown above. "
        "Below, LEAF selects a historical route-selection screen at step 10 (similarity 0.9391) and a route-details screen at step 11 "
        "(0.8450), changing the injected reference from a click to termination. Both predictions match the benchmark, whereas "
        "Previous screen and Update every four steps fail on both observations. The lower panels show the retrieved_action "
        "field and verbatim model-output excerpts. Historical screenshots and similarity scores belong to retrieval and are not "
        "policy inputs. Arrows above indicate recording order; predictions use recorded observations and action history, not a "
        "closed-loop rollout. The task query is translated; screenshot content is unchanged."
    )
    (OUT / "leaf_reference_case_caption.txt").write_text(caption + "\n")
    latex = caption.replace("LEAF", r"\methodname{}").replace("retrieved_action", r"\texttt{retrieved\_action}")
    (OUT / "leaf_reference_case_figure.tex").write_text(
        "\\begin{figure}[t]\n\\centering\n"
        "\\includegraphics[width=\\linewidth]{../../../trace_method/results/english_case_candidates/leaf_reference_case.pdf}\n"
        "\\caption{" + latex + "}\n\\label{fig:leaf-reference-case}\n\\end{figure}\n")
    evidence = {
        "episode_id": EPISODE, "method_name": name, "displayed_steps": [9, 10, 11],
        "selection": "Outcome-selected qualitative example of reference refresh; no population-level claim.",
        "source_files": {k: {"path": str(RESULTS / p), "sha256": digest(RESULTS / p)} for k, p in files.items()},
        "context_file": {"path": str(context_path), "sha256": digest(context_path)},
        "protocol": context["protocol"], "caption": caption, "screenshots": image_index,
        "steps": {i: {"dataset_row": r, "predictions": {k: results[k][r["id"]] for k in files},
                      "reference_source": context["sources"][r["id"]],
                      "injected_evidence": context["contexts"][r["id"]]} for i, r in targets.items()},
        "checks": {"all_copies_byte_identical": True, "quoted_excerpts_present_verbatim": True,
                   "reference_matches_logged_policy_input": True, "displayed_scores_verified": True},
        "limitations": ["Teacher-forced recorded-screen predictions, not closed-loop task success.",
                        "Step 9 is a failure and is displayed. All episode predictions retained here.",
                        "The thought excerpt acknowledges history; it does not establish causal reliance.",
                        "The route labels differ (historical A1, current D2); only coordinates and action are injected.",
                        "App controls are English; native map labels include Chinese and are preserved.",
                        "Lower code panels excerpt retrieved_action; full evidence dictionaries are in this file."]}
    (OUT / "leaf_reference_case_evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"figure": str(OUT / "leaf_reference_case.png"), "original_images_copied": len(image_index),
                      "scores": {k: [results[k][targets[i]["id"]]["sr"] for i in steps] for k in files}}, indent=2))


if __name__ == "__main__":
    main()

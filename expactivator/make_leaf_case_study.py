"""Case study of how LEAF works, built only from sealed outputs.

Stage ``prepare`` (needs torch) gathers every number shown in the figure:
(a) step-level activation on an English-interface NUS NextBus episode, and
(b) task-level activation for one user across a day, recomputed with the exact
activation rule over the sealed routine index.  Stage ``plot`` (needs
matplotlib) renders the figure from the saved evidence file.

    python make_leaf_case_study.py prepare   # hymem-serve environment
    python make_leaf_case_study.py plot      # agent environment
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "expactivator/results"
OUT = RESULTS / "leaf_case_study"
PAPER_FIG = REPO / "manuscript/paper/iclr2027/figure"
HISTORY_ROOT = Path("/storage/zhangfuyao/datasets/fingertip20k")

EPISODE = "43_20250415_131541"
STEPS = [1, 2, 6, 13, 14]
TASK = "Check the next bus from PGPR to Kent Ridge MRT"
TRIPS = {  # translated goals of the source episodes, looked up by (user, time)
    "20250408_173830": "Apr 8: LT9 → PGPR",
    "20250409_173029": "Apr 9: PGPR → AS3",
    "20250411_202219": "Apr 11: PGPR → Kent Ridge MRT",
}
LABELS = {  # English description of each recorded action
    1: ("tap NUS NextBus", "tap NUS NextBus"),
    2: ("wait", "wait"),
    5: ("tap Directions", "tap Directions"),
    6: ("tap End Location", "tap End Location"),
    13: ("tap route card", "tap route A1"),
    14: ("finish", "finish"),
}

USER = 94
SCENARIO = "住所"  # home
GROUPS = [  # routine families, matched on the Chinese intent, with translations
    ("睡眠", "Check last night's sleep"),
    ("天气", "Check the 24-hour weather"),
    ("消息", "Read WeChat messages"),
    ("植物", "Identify a plant with WeChat Scan"),
    ("运动数据", "Check today's exercise"),
    ("闹钟", "Set tomorrow morning's alarm"),
]


def prepare() -> None:
    import csv
    import numpy as np
    import pandas as pd
    import torch
    from PIL import Image

    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "src/eval_execution"))
    from evaluation_new_backbones import resolve_image
    from expactivator.evaluate_proactive import circular_hour_distance, hour
    from expactivator.run_leap_proactive_ablation import select

    test = {s["id"]: s for s in json.loads(
        (REPO / "data/personal_execution/personal_execution.json").read_text())}
    contexts = json.loads((RESULTS / "grounded/qwen3_vl_8b_contexts.json").read_text())
    policy = {r["episode_id"]: r for r in json.loads(
        (RESULTS / "grounded/qwen3_vl_8b_grounded_policy_gpu0.json").read_text())["outputs"]}
    goals = {(r["user_id"], r["time"]): r["intentDescription"]
             for r in csv.DictReader(open(REPO / "samples/total.csv", encoding="utf-8"))}

    steps = []
    for k in STEPS:
        sid = f"{EPISODE}_{k}"
        step, src, out = test[sid], contexts["sources"][sid], policy[sid]
        current = resolve_image(step["img_path"], HISTORY_ROOT)
        with Image.open(current) as im:
            width, height = im.size
        hist = re.search(r"\((\d+),\s*(\d+)\)", src["raw_historical_action"] or "")
        coordinate = out["output"].get("coordinate")
        steps.append({
            "step": k, "step_id": sid, "answer": step["answer"], "sr": out["sr"],
            "current_image": str(current), "image_size": [width, height],
            "output": out["output"],
            "output_px": ([coordinate[0] / 999 * width, coordinate[1] / 999 * height]
                          if coordinate else None),
            "reference": contexts["contexts"][sid]["retrieved_action"],
            "history_image": src["image_path"], "history_time": src["time"],
            "history_goal_zh": goals.get(("43", src["time"])),
            "history_trip": TRIPS[src["time"]],
            "history_action": src["raw_historical_action"],
            "history_px": [int(hist.group(1)), int(hist.group(2))] if hist else None,
            "similarity": src["visual_similarity"],
            "labels": LABELS[k],
            "thought": out["raw_output"].split("<tool_call>")[0].strip(),
        })

    production = json.loads((RESULTS / "proactive_predictions.json").read_text())
    config = production["config"]
    cache = torch.load(RESULTS / "proactive_history_index_gpu0.pt",
                       map_location="cpu", weights_only=False)
    total = pd.read_csv(REPO / "samples/total.csv").reset_index(drop=True)
    memory = (total.iloc[cache["memory_indices"]].reset_index()
              .rename(columns={"index": "source_index"}))
    keys = cache["keys"].float().numpy()
    local = memory[memory.user_id == USER]
    similarity = keys[local.index] @ keys[local.index].T
    hours = np.array([hour(t) for t in local.time])
    scenario = local.scenario.astype(str).to_numpy()
    neighbors = similarity >= config["semantic_threshold"]
    support = neighbors.sum(1)
    state = neighbors & (scenario == SCENARIO)[None, :]
    grid = np.linspace(0, 24, 481)
    weights = np.exp(-(circular_hour_distance(hours[None, :].repeat(len(grid), 0),
                                              grid[:, None]) ** 2)
                     / (2 * config["time_sigma_hours"] ** 2))
    mass = state.astype(float) @ weights.T
    stable = ((support >= config["minimum_support"])
              & (state.sum(1) >= config["minimum_scenario_support"]))
    kappa = mass / np.maximum(support, 1)[:, None] ** config["selection_power"]
    winners = []
    for g in range(len(grid)):
        ok = stable & (mass[:, g] >= config["trigger_mass"])
        winners.append(int(np.argmax(np.where(ok, kappa[:, g], -1))) if ok.any() else -1)
    intents = local.intentDescription.astype(str).tolist()

    def group_of(center: int) -> int | None:
        for index, (key, _) in enumerate(GROUPS):
            if key in intents[center]:
                return index
        return None

    groups = []
    for index, (key, label) in enumerate(GROUPS):
        centers = [c for c in range(len(intents)) if key in intents[c] and stable[c]]
        members = sorted({float(hours[m]) for c in centers for m in np.flatnonzero(state[c])})
        curve = mass[centers].max(0) if centers else np.zeros(len(grid))
        wins = [w >= 0 and group_of(w) == index for w in winners]
        groups.append({"key": key, "label": label, "centers": centers,
                       "member_hours": members, "support_curve": curve.tolist(),
                       "wins": wins})

    positives = json.loads((REPO / "data/proactive_suggestion/proactive_samples.json").read_text())
    queries = []
    for item in positives:
        if int(item["user_id"]) != USER or item["scenario"] != SCENARIO:
            continue
        query = {"user_id": USER, "time": str(item["time"]), "scenario": item["scenario"]}
        result = select(local.reset_index(drop=True), similarity, query, config, "full")
        predicted = result["predicted_intent"]
        center = int(np.flatnonzero(local.source_index.to_numpy()
                                    == result["routine_center_source_index"])[0]) \
            if result.get("routine_center_source_index") is not None else None
        queries.append({"time": item["time"], "hour": hour(item["time"]),
                        "ground_truth": item["originalInstruction"], "activated": predicted,
                        "group": group_of(center) if center is not None else None,
                        "support": result.get("routine_support"),
                        "mass": result.get("trigger_mass"),
                        "ground_truth_group": next((g for g, (k, _) in enumerate(GROUPS)
                                                    if k in item["originalInstruction"]), None)})

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "evidence.json").write_text(json.dumps({
        "episode": EPISODE, "task": TASK, "steps": steps,
        "user": USER, "scenario": SCENARIO, "config": config,
        "grid": grid.tolist(), "groups": groups, "queries": queries,
        "abstain": [w < 0 for w in winners],
    }, ensure_ascii=False, indent=1))
    for s in steps:
        print(f"step {s['step']}: sr={s['sr']} sim={s['similarity']:.3f} {s['history_trip']}")
    for q in queries:
        print(f"{q['time']} h={q['hour']:.2f} group={q['group']} gt_group={q['ground_truth_group']} "
              f"n={q['support']} M={q['mass'] and round(q['mass'], 2)} | {q['activated']}")


def plot(step_only: bool = False) -> None:
    import difflib
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Circle
    from PIL import Image

    PRIMARY, SECOND = "#176CCB", "#12A9BE"
    ORANGE, ORANGE_SOFT = PRIMARY, "#CFE3F4"  # task-level panel drawn in blue
    GRID, SPINE, TEXT = "#E7E9ED", "#D9DCE2", "#5F636B"
    plt.rcParams.update({"font.size": 6.5, "text.color": TEXT, "axes.edgecolor": SPINE,
                         "axes.labelcolor": TEXT, "xtick.color": TEXT, "ytick.color": TEXT,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    ev = json.loads((OUT / "evidence.json").read_text())
    W, H = 5.5, 6.53
    fig = plt.figure(figsize=(W, H))

    def box(x, y, w, h):
        return fig.add_axes([x / W, y / H, w / W, h / H])

    def put(x, y, text, **kw):
        fig.text(x / W, y / H, text, **kw)

    # ---------------- (a) step-level activation ----------------
    left, steps = 0.42, ev["steps"]
    ncol = len(steps)
    colw = (W - left - 0.04) / ncol
    imw = 0.7
    imh = imw * 2720 / 1260
    top = H - 0.06
    if not step_only:
        put(0.0, top - 0.1, "(a)", fontsize=8, fontweight="bold")
    put(0.0 if step_only else 0.24, top - 0.1, "Step-level activation", fontsize=7.5,
        fontweight="bold", color=PRIMARY)
    put(1.62, top - 0.1, f"Task: {ev['task']} (instruction translated)", fontsize=6.3)
    row1 = top - 0.36 - imh
    row2 = row1 - 0.2 - imh
    put(0.16, row1 + imh / 2, "Current screen", rotation=90, ha="center", va="center",
        fontsize=6.4, fontweight="bold")
    put(0.16, row2 + imh / 2, "Activated historical state", rotation=90, ha="center",
        va="center", fontsize=6.4, fontweight="bold", color=PRIMARY)

    def show(ax, path, size, point=None):
        with Image.open(path) as img:
            ax.imshow(img.convert("RGB"))
        ax.set_xlim(0, size[0]); ax.set_ylim(size[1], 0)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(SPINE); spine.set_linewidth(0.6)
        if point:
            ax.add_patch(Circle(point, 105, fill=False, edgecolor="white", linewidth=2.8))
            ax.add_patch(Circle(point, 105, fill=False, edgecolor=PRIMARY, linewidth=1.6))

    for i, s_ in enumerate(steps):
        cx = left + i * colw + (colw - imw) / 2
        mid = cx + imw / 2
        put(mid, row1 + imh + 0.06, f"Step {s_['step']}", ha="center", fontsize=6.5,
            fontweight="bold")
        show(box(cx, row1, imw, imh), s_["current_image"], s_["image_size"], s_["output_px"])
        put(mid, row2 + imh + 0.05, f"↓  sim {s_['similarity']:.2f}", ha="center",
            fontsize=6.0, color=PRIMARY, fontweight="bold")
        show(box(cx, row2, imw, imh), s_["history_image"], s_["image_size"])
        ref, out = s_["labels"]
        put(mid, row2 - 0.34, ref, ha="center", fontsize=6.0)
        put(mid, row2 - 0.49, out + " ✓", ha="center", fontsize=6.0, color=PRIMARY,
            fontweight="bold")
    put(0.16, row2 - 0.32, "Reference", ha="center", fontsize=6.0, fontweight="bold")
    put(0.16, row2 - 0.47, "Output", ha="center", fontsize=6.0, fontweight="bold",
        color=PRIMARY)

    runs, start = [], 0
    for i in range(1, ncol + 1):
        if i == ncol or steps[i]["history_trip"] != steps[start]["history_trip"]:
            runs.append((start, i - 1, steps[start]["history_trip"])); start = i
    by = row2 - 0.07
    for a_, b_, trip in runs:
        x0 = left + a_ * colw + 0.05; x1 = left + (b_ + 1) * colw - 0.05
        ax = box(x0, by - 0.05, x1 - x0, 0.05); ax.axis("off")
        ax.plot([0, 0, 1, 1], [1, 0, 0, 1], color=SECOND, linewidth=0.9,
                transform=ax.transAxes, clip_on=False)
        put((x0 + x1) / 2, by - 0.17, trip, ha="center", fontsize=6.0, color=SECOND,
            fontweight="bold")

    if step_only:
        for target, kw in ((PAPER_FIG / "leaf_case_step.pdf", {}),
                           (OUT / "leaf_case_step.png", {"dpi": 220})):
            fig.savefig(target, bbox_inches="tight", pad_inches=0.005, **kw)
        print("wrote leaf_case_step.pdf / .png")
        return

    # ---------------- (b) task-level activation ----------------
    btop = row2 - 0.66
    put(0.0, btop - 0.1, "(b)", fontsize=8, fontweight="bold")
    put(0.24, btop - 0.1, "Task-level activation", fontsize=7.5, fontweight="bold",
        color=ORANGE)
    put(1.62, btop - 0.1, "One user at home: contextual support of each routine across "
        "the day (intents translated)", fontsize=6.3)
    grid = np.array(ev["grid"])
    groups = ev["groups"]
    rowh, x0, x1 = 0.215, 1.62, W - 0.1
    peak = max(max(g["support_curve"]) for g in groups)
    first = btop - 0.78
    axes = []
    abstain = np.array(ev["abstain"])
    for r, g in enumerate(groups):
        y = first - r * rowh
        ax = box(x0, y - rowh + 0.02, x1 - x0, rowh - 0.03)
        ax.fill_between(grid, -0.25, 1.05, where=abstain, color=GRID, linewidth=0, zorder=0)
        curve = np.array(g["support_curve"]) / peak
        ax.fill_between(grid, 0, curve, color=ORANGE_SOFT, linewidth=0, zorder=1)
        ax.fill_between(grid, 0, curve, where=np.array(g["wins"]), color=ORANGE,
                        linewidth=0, zorder=2)
        ax.plot(g["member_hours"], [-0.13] * len(g["member_hours"]), "|", color=TEXT,
                markersize=2.6, markeredgewidth=0.5, zorder=3)
        ax.set_xlim(0, 24); ax.set_ylim(-0.25, 1.05)
        ax.set_yticks([]); ax.set_xticks(range(0, 25, 3))
        ax.tick_params(length=0, labelbottom=(r == len(groups) - 1), labelsize=5.8)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        put(x0 - 0.06, y - rowh / 2 + 0.01, g["label"], ha="right", va="center",
            fontsize=6.0)
        axes.append(ax)
    axes[-1].set_xlabel("Hour of day", fontsize=6.2, labelpad=1)

    edges = np.flatnonzero(np.diff(np.r_[0, abstain.astype(int), 0]))
    for a_, b_ in zip(edges[::2], edges[1::2]):
        hour_mid = (grid[a_] + grid[b_ - 1]) / 2
        put(x0 + hour_mid / 24 * (x1 - x0), first - 3 * rowh + 0.02, "abstain",
            ha="center", va="center", rotation=90, fontsize=5.8, style="italic",
            zorder=10)

    best = {}
    for q in ev["queries"]:
        if q["group"] is None or q["group"] != q["ground_truth_group"]:
            continue
        score = difflib.SequenceMatcher(None, q["activated"], q["ground_truth"]).ratio()
        if q["group"] not in best or score > best[q["group"]][0]:
            best[q["group"]] = (score, q)
    last_hour, level = -10.0, 0
    for _, q in sorted(best.values(), key=lambda item: item[1]["hour"]):
        level = level + 1 if q["hour"] - last_hour < 3.6 else 0
        last_hour = q["hour"]
        lift = 2.3 * level
        for ax in axes:
            ax.axvline(q["hour"], color=TEXT, linewidth=0.6, linestyle=(0, (2, 2)), zorder=4)
        axes[q["group"]].plot([q["hour"]], [0.45], marker="o", markersize=4.4,
                              color="white", markeredgecolor=ORANGE,
                              markeredgewidth=1.3, zorder=6, clip_on=False)
        clock = q["time"].split("_")[1]
        hh, mm = int(clock[:2]), int(clock[2:4])
        align = "right" if q["hour"] > 22.5 else "center"
        axes[0].text(q["hour"], 2.35 + lift, f"{hh:02d}:{mm:02d} ✓", ha=align, fontsize=6.0,
                     color=ORANGE, fontweight="bold", clip_on=False)
        axes[0].text(q["hour"], 1.3 + lift, f"$M$={q['mass']:.1f}", ha=align, fontsize=5.6,
                     clip_on=False)

    fig.savefig(PAPER_FIG / "leaf_case_study.pdf", bbox_inches="tight")
    fig.savefig(OUT / "leaf_case_study.png", dpi=220, bbox_inches="tight")
    print("wrote leaf_case_study.pdf / .png")


if __name__ == "__main__":
    {"prepare": prepare, "plot": plot,
     "plot_step": lambda: plot(step_only=True)}[sys.argv[1]]()

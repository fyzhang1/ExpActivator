"""Task-level activation case figures, one per user, from the sealed index.

Each case names a user, one or more scenarios, and the routine families to draw
(matched on the Chinese intent, with an English label).  ``prepare`` recomputes
the activation rule over the day and refuses to continue if any routine that
wins at some hour is not covered by a drawn family.  ``plot`` renders one figure
per case into ``results/leaf_case_study/task_level``.

    python make_leaf_task_cases.py prepare   # environment with torch
    python make_leaf_task_cases.py plot      # environment with matplotlib
"""
from __future__ import annotations

import difflib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "expactivator/results"
OUT = RESULTS / "leaf_case_study/task_level"
MATCH = 0.9  # a query is marked when the activated intent reproduces the ground truth

HOME, SCHOOL = "住所", "学校"
STREET, VEHICLE = "城市街道", "交通工具"
CASES = {
    "user94_home": {"user": 94, "title": "One user at home: four routines across the day",
        "blocks": [(HOME, "At home", [
            ("睡眠", "Check last night's sleep"),
            ("天气", "Check the 24-hour weather"),
            ("消息", "Read WeChat messages"),
            ("植物", "Identify a plant with WeChat Scan"),
            ("运动数据", "Check today's exercise"),
            ("闹钟", "Set tomorrow morning's alarm")])]},
    "user61_home": {"user": 61, "title": "Morning check-in and a bedtime timer",
        "blocks": [(HOME, "At home", [
            ("米游社", "Check in on Miyoushe and visit the Tavern"),
            ("QQ", "Reply to the pinned QQ friend"),
            ("私人雷达", "Play the Private Radar playlist"),
            ("bilibili", "Play a Watch Later video on Bilibili"),
            ("刷牙", "Start the tooth-brushing timer")])]},
    "user75_home": {"user": 75, "title": "Evening trends before a late-night video",
        "blocks": [(HOME, "At home", [
            ("闲鱼", "Refresh my listings on Xianyu"),
            ("超话", "Check in to a Weibo fan topic"),
            ("天气", "Check tomorrow's weather"),
            ("QQ音乐", "Daily check-in on QQ Music"),
            ("抖音", "Like the first recommended Douyin video"),
            ("热搜", "Open the top trending topic on Weibo"),
            ("b站", "Open the top trending video on Bilibili")])]},
    "user95_home": {"user": 95, "title": "Morning news and evening exercise",
        "blocks": [(HOME, "At home", [
            ("早报", "Read the daily morning brief"),
            ("第一条新闻", "Read today's top news in QQ Browser"),
            ("睡眠", "Check last night's sleep"),
            ("资治通鉴", "Resume the Zizhi Tongjian audiobook"),
            ("运动数据", "Check today's exercise")])]},
    "user77_home": {"user": 77, "title": "Daily check-ins in a fixed morning order",
        "blocks": [(HOME, "At home", [
            ("番茄畅听", "Check in on Fanqie Audiobooks"),
            ("快手", "Check in on Kuaishou Lite"),
            ("腾讯视频", "Check in on Tencent Video"),
            ("清理", "Clean junk files with Phone Manager")])]},
    "user86_commute": {"user": 86, "title": "Same morning, different scenario",
        "blocks": [(STREET, "On the street", [
            ("公交卡", "Show the Guiyang e-bus card in Alipay")]),
                   (VEHICLE, "On a vehicle", [
            ("筑红云", "Finish today's study on Zhuhongyun"),
            ("易捷加油", "Top up the Yijie fuel wallet"),
            ("拼多多", "Track a Pinduoduo delivery")])]},
}


def prepare() -> None:
    import numpy as np
    import pandas as pd
    import torch
    sys.path.insert(0, str(REPO))
    from expactivator.evaluate_proactive import circular_hour_distance, hour
    from expactivator.run_leap_proactive_ablation import select

    production = json.loads((RESULTS / "proactive_predictions.json").read_text())
    config = production["config"]
    cache = torch.load(RESULTS / "proactive_history_index_gpu0.pt", map_location="cpu",
                       weights_only=False)
    total = pd.read_csv(REPO / "samples/total.csv").reset_index(drop=True)
    memory = (total.iloc[cache["memory_indices"]].reset_index()
              .rename(columns={"index": "source_index"}))
    keys = cache["keys"].float().numpy()
    positives = json.loads((REPO / "data/proactive_suggestion/proactive_samples.json").read_text())
    grid = np.linspace(0, 24, 481)

    for name, case in CASES.items():
        local = memory[memory.user_id == case["user"]].reset_index(drop=True)
        vectors = keys[memory.index[memory.user_id == case["user"]]]
        similarity = vectors @ vectors.T
        hours = np.array([hour(t) for t in local.time])
        scenarios = local.scenario.astype(str).to_numpy()
        intents = local.intentDescription.astype(str).tolist()
        neighbors = similarity >= config["semantic_threshold"]
        support = neighbors.sum(1)
        weights = np.exp(-(circular_hour_distance(hours[None, :].repeat(len(grid), 0),
                                                  grid[:, None]) ** 2)
                         / (2 * config["time_sigma_hours"] ** 2))
        blocks = []
        for scenario, block_label, families in case["blocks"]:
            state = neighbors & (scenarios == scenario)[None, :]
            mass = state.astype(float) @ weights.T
            stable = ((support >= config["minimum_support"])
                      & (state.sum(1) >= config["minimum_scenario_support"]))
            score = mass / np.maximum(support, 1)[:, None] ** config["selection_power"]
            winners = []
            for g in range(len(grid)):
                ok = stable & (mass[:, g] >= config["trigger_mass"])
                winners.append(int(np.argmax(np.where(ok, score[:, g], -1))) if ok.any() else -1)

            def family(center):
                for index, (key, _) in enumerate(families):
                    if key in intents[center]:
                        return index
                return None

            uncovered = sorted({intents[w] for w in winners if w >= 0 and family(w) is None})
            if uncovered:
                raise SystemExit(f"{name}/{scenario}: winning routines not drawn: {uncovered}")
            rows = []
            for index, (key, label) in enumerate(families):
                centers = [c for c in range(len(intents)) if key in intents[c] and stable[c]]
                wins = [w >= 0 and family(w) == index for w in winners]
                if not any(wins):
                    raise SystemExit(f"{name}/{scenario}: family '{label}' never wins")
                members = sorted({float(hours[m]) for c in centers
                                  for m in np.flatnonzero(state[c])})
                rows.append({"key": key, "label": label, "member_hours": members,
                             "support_curve": mass[centers].max(0).tolist(), "wins": wins})
            queries = []
            for item in positives:
                if int(item["user_id"]) != case["user"] or item["scenario"] != scenario:
                    continue
                query = {"user_id": case["user"], "time": str(item["time"]),
                         "scenario": scenario}
                result = select(local, similarity, query, config, "full")
                activated = result["predicted_intent"]
                ratio = difflib.SequenceMatcher(None, activated, item["originalInstruction"]).ratio()
                queries.append({"time": item["time"], "hour": hour(item["time"]),
                                "activated": activated, "ground_truth": item["originalInstruction"],
                                "match": ratio, "row": next((i for i, (k, _) in enumerate(families)
                                                             if k in activated), None),
                                "support": result.get("routine_support"),
                                "mass": result.get("trigger_mass")})
            blocks.append({"scenario": scenario, "label": block_label, "rows": rows,
                           "abstain": [w < 0 for w in winners], "queries": queries})
        folder = OUT / name
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "evidence.json").write_text(json.dumps({
            "case": name, "user": case["user"], "title": case["title"], "config": config,
            "grid": grid.tolist(), "match_threshold": MATCH, "blocks": blocks},
            ensure_ascii=False, indent=1))
        marked = [(q["time"], round(q["match"], 2)) for b in blocks for q in b["queries"]
                  if q["match"] >= MATCH]
        print(f"{name}: rows={[len(b['rows']) for b in blocks]} marked={marked}")


FONT_DIRS = [Path.home() / ".local/share/fonts", Path("/usr/share/fonts")]


def use_comic_sans() -> None:
    """Register Comic Sans MS, falling back to DejaVu Sans for missing glyphs."""
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt
    found = [p for d in FONT_DIRS if d.exists()
             for p in d.rglob("comic*.ttf")]
    if not found:
        raise SystemExit(
            "Comic Sans MS not found. Extract comic.ttf and comicbd.ttf from "
            "https://downloads.sourceforge.net/corefonts/comic32.exe (e.g. with "
            "cabextract) into ~/.local/share/fonts.")
    for path in found:
        fm.fontManager.addfont(str(path))
    plt.rcParams["font.family"] = ["Comic Sans MS", "DejaVu Sans"]


def plot() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Rectangle

    PRIMARY, SOFT = "#176CCB", "#CFE3F4"
    GRID, SPINE, TEXT = "#E7E9ED", "#D9DCE2", "#2E3238"
    use_comic_sans()
    plt.rcParams.update({"font.size": 6.5, "text.color": TEXT, "axes.edgecolor": SPINE,
                         "axes.labelcolor": TEXT, "xtick.color": TEXT, "ytick.color": TEXT,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    W, ROWH, LABEL_FS = 5.5, 0.235, 6.3

    for name in CASES:
        folder = OUT / name
        ev = json.loads((folder / "evidence.json").read_text())
        grid = np.array(ev["grid"])
        blocks = ev["blocks"]
        multi = len(blocks) > 1
        peak = max(max(r["support_curve"]) for b in blocks for r in b["rows"])

        # Measure the widest row label so the plotting area starts right after it.
        probe = plt.figure(figsize=(W, 1))
        renderer = probe.canvas.get_renderer()
        widest = max(probe.text(0, 0, r["label"], fontsize=LABEL_FS)
                     .get_window_extent(renderer).width
                     for case in CASES
                     for b in json.loads((OUT / case / "evidence.json").read_text())["blocks"]
                     for r in b["rows"]) / probe.dpi  # shared so all figures align
        plt.close(probe)
        x0 = widest + 0.16
        x1 = W - 0.1

        def hour_x(h):
            return x0 + h / 24 * (x1 - x0)

        # Marked queries per block.  Callout widths are measured, then each
        # callout takes the lowest level at which it overlaps no other one.
        probe = plt.figure(figsize=(W, 1))
        renderer = probe.canvas.get_renderer()
        span = (x1 - x0) / 24  # inches per hour
        callouts = []
        for block in blocks:
            best = {}
            for q in block["queries"]:
                if q["match"] < ev["match_threshold"] or q["row"] is None:
                    continue
                if q["row"] not in best or q["match"] > best[q["row"]]["match"]:
                    best[q["row"]] = q
            placed = []
            for q in sorted(best.values(), key=lambda q: q["hour"]):
                clock = q["time"].split("_")[1]
                text = f"{clock[:2]}:{clock[2:4]} ✓\nM = {q['mass']:.1f}"
                width = probe.text(0, 0, text, fontsize=6.0, fontweight="bold",
                                   multialignment="center").get_window_extent(renderer).width
                width = width / probe.dpi + 0.12
                align = ("right" if q["hour"] > 21.5 else
                         "left" if q["hour"] < 2.5 else "center")
                centre = q["hour"] * span
                lo = {"right": centre - width, "left": centre}.get(align, centre - width / 2)
                hi = lo + width
                level = 0
                while any(p["level"] == level and lo < p["hi"] + 0.05 and p["lo"] < hi + 0.05
                          for p in placed):
                    level += 1
                placed.append({**q, "text": text, "align": align, "lo": lo, "hi": hi,
                               "level": level})
            callouts.append(placed)
        plt.close(probe)
        CALL = 0.33

        def strip(i):
            return CALL * (1 + max((c["level"] for c in callouts[i]), default=-1)) + 0.04

        head = 0.20 + 0.24                 # case title, legend
        body = sum(len(b["rows"]) * ROWH for b in blocks)
        headers = (0.2 * len(blocks) + 0.08 * (len(blocks) - 1)) if multi else 0
        H = head + sum(strip(i) for i in range(len(blocks))) + headers + body + 0.36
        fig = plt.figure(figsize=(W, H))

        def box(x, y, w, h):
            return fig.add_axes([x / W, y / H, w / W, h / H])

        def put(x, y, text, **kw):
            return fig.text(x / W, y / H, text, **kw)

        # Title, subtitle and legend.
        put(0.02, H - 0.16, ev["title"], fontsize=7.2, fontweight="bold")
        lg = box(0.02, H - 0.40, W - 0.04, 0.14)
        lg.set_xlim(0, W - 0.04); lg.set_ylim(0, 1); lg.axis("off")
        items = [("soft", "contextual support"), ("dark", "activated"),
                 ("tick", "past occurrences"), ("gray", "abstain")]
        cx = 0.0
        for kind, text in items:
            if kind == "tick":
                lg.plot([cx + 0.04 + i * 0.045 for i in range(4)], [0.5] * 4, "|",
                        color=TEXT, markersize=5, markeredgewidth=0.6)
            else:
                color = {"soft": SOFT, "dark": PRIMARY, "gray": GRID}[kind]
                lg.add_patch(Rectangle((cx, 0.15), 0.2, 0.7, color=color, linewidth=0))
            label = lg.text(cx + 0.27, 0.5, text, va="center", fontsize=6.0)
            fig.canvas.draw()
            cx += 0.27 + label.get_window_extent().width / fig.dpi + 0.28

        y = H - head
        first_axes = []
        for bi, block in enumerate(blocks):
            if multi:
                if bi > 0:
                    y -= 0.08
                put(0.02, y - 0.15, block["label"], fontsize=6.8, fontweight="bold",
                    color=PRIMARY)
                y -= 0.2
            y -= strip(bi)
            abstain = np.array(block["abstain"])
            axes = []
            for row in block["rows"]:
                ax = box(x0, y - ROWH + 0.025, x1 - x0, ROWH - 0.035)
                ax.fill_between(grid, -0.3, 1.08, where=abstain, color=GRID, linewidth=0,
                                zorder=0)
                curve = np.array(row["support_curve"]) / peak
                ax.fill_between(grid, 0, curve, color=SOFT, linewidth=0, zorder=1)
                ax.fill_between(grid, 0, curve, where=np.array(row["wins"]), color=PRIMARY,
                                linewidth=0, zorder=2)
                ax.plot(row["member_hours"], [-0.15] * len(row["member_hours"]), "|",
                        color=TEXT, markersize=2.8, markeredgewidth=0.5, zorder=3)
                ax.set_xlim(0, 24); ax.set_ylim(-0.3, 1.08)
                ax.set_yticks([]); ax.set_xticks(range(0, 25, 3))
                ax.tick_params(length=0, labelbottom=False, labelsize=6.0, pad=2)
                for sp in ("top", "right", "left"):
                    ax.spines[sp].set_visible(False)
                ax.spines["bottom"].set_color(SPINE)
                put(x0 - 0.1, y - ROWH / 2 + 0.01, row["label"], ha="right",
                    va="center", fontsize=LABEL_FS)
                axes.append(ax)
                y -= ROWH
            block_top = y + len(axes) * ROWH
            edges = np.flatnonzero(np.diff(np.r_[0, abstain.astype(int), 0]))
            for a_, b_ in zip(edges[::2], edges[1::2]):
                width = grid[b_ - 1] - grid[a_]
                if width < 1.0:
                    continue
                put(hour_x((grid[a_] + grid[b_ - 1]) / 2), y + len(axes) * ROWH / 2,
                    "abstain", ha="center", va="center", rotation=90 if width < 2.5 else 0,
                    fontsize=6.0, color=TEXT, zorder=10)
            for c in callouts[bi]:
                for ax in axes:
                    ax.axvline(c["hour"], color=TEXT, linewidth=0.6, linestyle=(0, (2, 2)),
                               zorder=4)
                axes[c["row"]].plot([c["hour"]], [0.45], marker="o", markersize=4.6,
                                    color="white", markeredgecolor=PRIMARY,
                                    markeredgewidth=1.3, zorder=6, clip_on=False)
                ly = block_top + 0.07 + CALL * c["level"]
                put(hour_x(c["hour"]), ly, c["text"], ha=c["align"], va="bottom",
                    multialignment="center", fontsize=6.0, color=PRIMARY,
                    fontweight="bold", linespacing=1.25,
                    bbox=dict(boxstyle="round,pad=0.25,rounding_size=0.35",
                              facecolor="white", edgecolor=PRIMARY, linewidth=0.6))
                lead = box(hour_x(c["hour"]) - 0.005, block_top - 0.005, 0.01,
                           ly - block_top + 0.01)
                lead.axis("off")
                lead.plot([0.5, 0.5], [0, 1], color=PRIMARY, linewidth=0.6,
                          transform=lead.transAxes, clip_on=False)
            first_axes += axes
        first_axes[-1].tick_params(labelbottom=True)
        first_axes[-1].set_xticklabels([f"{h}:00" for h in range(0, 25, 3)])
        first_axes[-1].set_xlabel("Hour of day", fontsize=6.4, labelpad=2)
        for suffix, kw in (("pdf", {}), ("png", {"dpi": 220})):
            fig.savefig(folder / f"{name}.{suffix}", bbox_inches="tight", pad_inches=0.005,
                        **kw)
        plt.close(fig)
        print(f"wrote {folder / name}.pdf/png")


COMPACT = {  # half-width panel for the main text: case, shown families, short labels
    "case": "user94_home",
    "rows": [("天气", "Check weather"), ("植物", "Identify plant"),
             ("运动数据", "Check exercise"), ("闹钟", "Set alarm")],
    "title": "One user's day at home",
    "size": (1.86, 1.46),
}


def plot_compact() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    PRIMARY, SOFT = "#176CCB", "#CFE3F4"
    GRID, SPINE, TEXT = "#E7E9ED", "#D9DCE2", "#2E3238"
    use_comic_sans()
    plt.rcParams.update({"font.size": 6.0, "text.color": TEXT, "axes.edgecolor": SPINE,
                         "xtick.color": TEXT, "pdf.fonttype": 42, "ps.fonttype": 42})
    name = COMPACT["case"]
    ev = json.loads((OUT / name / "evidence.json").read_text())
    block = ev["blocks"][0]
    grid = np.array(ev["grid"])
    by_key = {row["key"]: (i, row) for i, row in enumerate(block["rows"])}
    shown = [(by_key[key], label) for key, label in COMPACT["rows"]]
    peak = max(max(row["support_curve"]) for (_, row), _ in shown)
    marks = {}
    for q in block["queries"]:
        if q["match"] >= ev["match_threshold"] and q["row"] is not None:
            if q["row"] not in marks or q["match"] > marks[q["row"]]["match"]:
                marks[q["row"]] = q

    W, H = COMPACT["size"]
    fig = plt.figure(figsize=(W, H))
    label_w, top, bottom = 0.62, 0.2, 0.3
    rowh = (H - top - bottom) / len(shown)
    fig.text(0.0, 1 - 0.05 / H, COMPACT["title"], fontsize=7.0, fontweight="bold",
             va="top", color=TEXT)
    abstain = np.array(block["abstain"])
    axes = []
    for r, ((index, row), label) in enumerate(shown):
        y = H - top - (r + 1) * rowh
        ax = fig.add_axes([label_w / W, (y + 0.02) / H, (W - label_w - 0.03) / W,
                           (rowh - 0.035) / H])
        ax.fill_between(grid, -0.3, 1.1, where=abstain, color=GRID, linewidth=0, zorder=0)
        curve = np.array(row["support_curve"]) / peak
        ax.fill_between(grid, 0, curve, color=SOFT, linewidth=0, zorder=1)
        ax.fill_between(grid, 0, curve, where=np.array(row["wins"]), color=PRIMARY,
                        linewidth=0, zorder=2)
        ax.plot(row["member_hours"], [-0.17] * len(row["member_hours"]), "|", color=TEXT,
                markersize=2.2, markeredgewidth=0.45, zorder=3)
        ax.set_xlim(0, 24); ax.set_ylim(-0.3, 1.1)
        ax.set_yticks([]); ax.set_xticks([0, 6, 12, 18, 24])
        ax.tick_params(length=0, labelbottom=False, labelsize=5.8, pad=1.5)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        fig.text((label_w - 0.06) / W, (y + rowh / 2 + 0.005) / H, label, ha="right",
                 va="center", fontsize=6.0)
        q = marks.get(index)
        if q is not None:
            ax.plot([q["hour"]], [0.45], marker="o", markersize=3.8, color="white",
                    markeredgecolor=PRIMARY, markeredgewidth=1.1, zorder=6, clip_on=False)
            clock = q["time"].split("_")[1]
            right = q["hour"] < 17
            # walk away from the marker until the curve is low and not activated
            wins = np.array(row["wins"])
            k = int(np.argmin(np.abs(grid - q["hour"])))
            step = 1 if right else -1
            while 0 < k < len(grid) - 1 and curve[k] > 0.3:
                k += step
            anchor = grid[k] + 0.5 * step
            if abs(anchor - q["hour"]) < 1.4:  # keep clear of the marker ring
                anchor = q["hour"] + 1.4 * step
            ax.text(anchor, 0.45,
                    f"{clock[:2]}:{clock[2:4]} ✓", ha="left" if right else "right",
                    va="center", fontsize=5.8, color=PRIMARY, fontweight="bold", zorder=7,
                    bbox=dict(boxstyle="round,pad=0.15,rounding_size=0.3", facecolor="white",
                              edgecolor="none", alpha=0.9))
        axes.append(ax)
    axes[-1].tick_params(labelbottom=True)
    axes[-1].set_xticklabels(["0h", "6h", "12h", "18h", "24h"])
    axes[-1].set_xlabel("hour of day", fontsize=5.8, labelpad=1)
    for target, kw in ((OUT / name / f"{name}_compact.pdf", {}),
                       (OUT / name / f"{name}_compact.png", {"dpi": 300})):
        fig.savefig(target, bbox_inches="tight", pad_inches=0.005, **kw)
    plt.close(fig)
    print(f"wrote {name}_compact.pdf/png")


if __name__ == "__main__":
    {"prepare": prepare, "plot": plot, "compact": plot_compact}[sys.argv[1]]()

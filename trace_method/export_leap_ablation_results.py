"""Export complete declared ablations, including null and adverse findings."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "trace_method/results/leap_ablations"


def pct(x):
    return f"{100*x:.2f}"


def effect(c):
    return f"{pct(c['delta'])} [{pct(c['delta_ci95'][0])}, {pct(c['delta_ci95'][1])}]"


def table(caption, label, columns, header, rows):
    return "\n".join([
        r"\begin{table}[htbp]", r"\centering\small", r"\caption{"+caption+"}",
        r"\label{"+label+"}", r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{"+columns+"}",
        r"\toprule", " & ".join(header)+r" \\", r"\midrule",
        *[" & ".join(row)+r" \\" for row in rows], r"\bottomrule", r"\end{tabular}}", r"\end{table}", ""])


def main():
    refresh = json.loads((ROOT / "refresh/analysis.json").read_text())
    proactive = json.loads((ROOT / "proactive/analysis.json").read_text())
    identity = json.loads((ROOT / "user_history_partial/analysis.json").read_text())
    assert all(d["status"] == "complete" for d in (refresh, proactive, identity))
    delay = refresh["contrasts"]["delay1"]["policy"]["all_nonentry"]
    periodic = refresh["contrasts"]["refresh4"]["policy"]["all_nonentry"]
    time = proactive["full_minus_ablation"]["no_time"]["auroc"]
    scenario = proactive["full_minus_ablation"]["no_scenario"]["auroc"]
    norm = proactive["full_minus_ablation"]["raw_mass"]["semantic_conditional"]
    norm_row = proactive["normalization_row_bootstrap_sensitivity"]
    own = identity["policy_contrasts"]["partial_other"]["all_775"]
    strict = identity["policy_contrasts"]["partial_other"]["strict_complete_match"]
    exact = identity["policy_contrasts"]["partial_other"]["all_replaced_goals_exact"]
    interactive = refresh["contrasts"]["delay1"]["policy"]["interactive"]
    no_history = identity["policy_contrasts"]["base"]["all_775"]
    pm = proactive["metrics"]
    prose = (
        r"\paragraph{Updating the action reference.} "
        "We retain LEAP's fixed neighborhood of eight historical trajectories and vary when its "
        "reference is selected, while the frozen Qwen3-VL-8B policy always observes the current screen. "
        f"Across 7,140 non-entry steps, policy SSR is {pct(delay['method'])}\\% with current-state selection "
        f"and {pct(delay['reference'])}\\% when selection uses the preceding screen, a difference of "
        f"{effect(delay)} percentage points (95\\% user-cluster bootstrap interval). "
        f"Refreshing the reference every four steps yields {pct(periodic['reference'])}\\% SSR "
        f"(current minus periodic: {effect(periodic)} points). "
        f"On 5,151 click, long-click, scroll and text-entry steps, the current-versus-delayed "
        f"policy difference is {effect(interactive)} points. "
        "These comparisons measure reference refresh under teacher forcing; they do not establish "
        "closed-loop error recovery.\n\n"
        r"\paragraph{Activating a routine in context.} "
        "We evaluate routine selection before backbone decoding, with the historical intent index fixed. "
        f"Removing the time kernel changes activation AUROC from {pct(pm['full']['auroc'])}\\% to "
        f"{pct(pm['no_time']['auroc'])}\\% (full minus ablation: {effect(time)} points). "
        f"The corresponding scenario-removal contrast is {effect(scenario)} points. "
        f"At the original activation threshold, false alarms are {pct(pm['full']['false_alarm_rate'])}\\% "
        f"with time and scenario information and {pct(pm['no_scenario']['false_alarm_rate'])}\\% without the scenario condition; "
        "this operating-point comparison also reflects the changed score scale. "
        f"Frequency-normalized ranking changes conditional Semantic by {pct(norm['delta'])} points "
        f"with identical trigger decisions. Its user-cluster interval is "
        f"[{pct(norm['delta_ci95'][0])}, {pct(norm['delta_ci95'][1])}], whereas independent-query "
        f"resampling gives [{pct(norm_row['delta_ci95'][0])}, {pct(norm_row['delta_ci95'][1])}]; "
        "the uncertainty therefore depends on the resampling unit.\n\n"
        r"\paragraph{Application choices from user history.} "
        "We match historical residual goals across users and replace the matched application labels "
        "while preserving the original vote ranks and weights. Unmatched records remain unchanged. "
        f"On all 775 task entries, policy SSR is {pct(own['method'])}\\% with the user's own history and "
        f"{pct(own['reference'])}\\% with partial replacement (difference: {effect(own)} points). "
        f"The no-history baseline obtains {pct(no_history['reference'])}\\% SSR. "
        "Only 25 queries admit complete matching of all ten records; their difference is "
        f"{effect(strict)} points. "
        f"For {exact['n_steps']} queries whose replaced historical goals all match exactly after "
        f"normalization, the difference is {effect(exact)} points. "
        "This diagnostic concerns application selection with partial replacement of evidence from "
        "the user's history; it does not isolate recovery of later content preferences.\n"
    )
    (ROOT / "leap_ablation_results.tex").write_text(prose)
    refresh_rows = []
    for v, label in (("delay1", "Preceding screen"), ("refresh4", "Refresh every four steps")):
        p = refresh["contrasts"][v]["policy"]["all_nonentry"]
        e = refresh["contrasts"][v]["evidence"]["all_nonentry"]
        refresh_rows.append([label, pct(e['reference']), pct(p['reference']), effect(p)])
    e = refresh["contrasts"]["delay1"]["evidence"]["all_nonentry"]
    refresh_rows.append(["Current screen (LEAP)", pct(e['method']), pct(delay['method']), "Reference"])
    (ROOT / "leap_refresh_table.tex").write_text(table(
        "Reference refresh with the same eight-trajectory neighborhood. All conditions run the frozen "
        "Qwen3-VL-8B policy on the same current inputs and 7,140 non-entry steps. Evidence agreement "
        "scores the retrieved action separately from policy SSR. Rates are percentages; differences "
        "are current minus control in percentage points with 95\\% paired user-cluster intervals.",
        "tab:leap-refresh", "lrrl", ["Reference selection", "Evidence agreement", "Policy SSR", "Policy difference [95\\% CI]"], refresh_rows))
    rows = []
    for v, label in (("full", "LEAP"), ("no_time", "Without time"), ("no_scenario", "Without scenario"), ("raw_mass", "Unnormalized ranking")):
        rows.append([label]+[pct(pm[v][k]) for k in ("auroc", "balanced_accuracy", "recall", "false_alarm_rate", "semantic_conditional", "semantic_all_positive")])
    (ROOT / "leap_proactive_table.tex").write_text(table(
        "Ablations of routine selection before backbone decoding on 215 positive and 100 negative states. "
        "BA is balanced accuracy and FA is false-alarm rate. Conditional Semantic follows the official "
        "metric on emitted positive suggestions; all-positive Semantic assigns zero to abstentions and "
        "is an additional coverage diagnostic. Trigger rates use the original, unretuned threshold; "
        "removing context changes the score scale. All values are percentages.",
        "tab:leap-proactive-factors", "lrrrrrr", ["Condition", "AUROC", "BA", "Recall", "FA", "Cond. Semantic", "All-pos. Semantic"], rows))
    rows = []
    for key, label in (("all_775", "All task entries"), ("at_least_one_replacement", "At least one matched record"), ("all_replaced_goals_exact", "All replaced goals match exactly"), ("strict_complete_match", "All ten records matched")):
        c = identity["policy_contrasts"]["partial_other"][key]
        base = identity["policy_contrasts"]["base"][key]
        rows.append([label, str(c['n_steps']), pct(c['method']), pct(c['reference']), pct(base['reference']), effect(c)])
    (ROOT / "leap_user_history_table.tex").write_text(table(
        "Matched application-label replacement at task entry. Distinct cross-user donors must have "
        "historical residual-goal cosine at least 0.90 and query-similarity difference at most 0.02. "
        "Matched labels replace own-user labels at the original ranks and vote weights; unmatched "
        "records remain unchanged. All 775 entries are evaluated in every policy condition. Rates "
        "are policy SSR (\\%); differences are own minus replaced with paired 95\\% user-cluster intervals.",
        "tab:leap-user-history", "lrrrrl", ["Subset", "Entries", "Own history", "Replacement", "No history", "Difference [95\\% CI]"], rows))
    methods = r"""All new execution conditions use the same frozen Qwen3-VL-8B checkpoint,
200,704-pixel image budget, 256-token output cap, 8,192-token context limit,
256-request batches, and at most 64 concurrent sequences. The policy receives
the same user profile, ambiguous query, current screen and teacher-forced
action prefix. Reference refresh changes only the historical action evidence.
At non-entry position $t$, delayed retrieval uses screen $s_{\max(1,t-1)}$;
periodic retrieval uses $s_{1+4\lfloor(t-1)/4\rfloor}$. Both preserve the
original application-entry evidence. The fixed semantic neighborhood and
stateless selector permit exact reuse of sealed evidence from those positions.

Proactive ablations use the same frozen MiniLM index over 15,966 historical
intents and expose only user identity, time and scenario to selection. The
full condition reproduces every production suggestion and source record.
Time removal replaces the temporal kernel by one. Scenario removal drops the
scenario indicator, scenario-support gate and restriction on the representative
record. Unnormalized ranking uses contextual mass directly with the same
eligible set. Activation curves vary the threshold on the maximum contextual
mass among support-eligible routines, with a score of $-1$ for queries without
such a routine. These are ROC diagnostics of the declared score, including
standard endpoints; dots identify the actual selector's fixed-threshold
decisions. No test threshold is selected. The
official Semantic scorer uses Qwen3-Embedding-0.6B and fuzzy edit similarity.

The application experiment matches each of the ten production neighbors to
distinct other-user records using only residual-goal representations. Exact
normalized goal matches are preferred, followed by higher goal cosine and
closer query similarity, subject to the stated calipers. Application labels
and target answers do not enter matching. Complete matching covers 25 queries.
After observing this coverage and before generating policy predictions, we
add a partial-replacement diagnostic on all 775 queries. It replaces 2,101 of
7,750 record slots across 536 queries and changes 118 application references.
Donors can come from multiple users. The strict 25-query comparison reuses
the identical contexts and outputs within the 775-entry batches. The no-history
condition is evaluated separately with the original baseline prompt.
Before entry-policy inference, a small assistant review of 20 random matched
record pairs finds incomplete task equivalence despite high embedding scores.
We retain all pairs and additionally report queries for which every replaced
historical goal matches its donor exactly after text normalization. This subset
does not filter on application agreement or policy correctness. Matching scores
are dot products of the stored half-precision normalized embeddings and can
deviate slightly from exact cosine after quantization.

All primary intervals use 20,000 paired bootstrap draws over whole users;
execution comparisons also include episode-cluster sensitivity. Proactive
normalization includes independent-query resampling as a post-primary
sensitivity analysis to match the earlier report. These exploratory analyses
use an already inspected benchmark. All declared conditions and raw outputs
are retained; intervals are unadjusted for multiple comparisons and do not
estimate variation across inference runs. The entry diagnostic concerns
application choice, while the execution study uses teacher forcing.
"""
    (ROOT / "leap_ablation_methods.tex").write_text(methods)
    wrapper = r"""\documentclass[10pt]{article}
\usepackage[T1]{fontenc}
\usepackage{times,amsmath,amssymb,booktabs,graphicx,geometry}
\geometry{margin=0.85in}
\title{LEAP: Ablations of Experience Use}
\author{}
\date{}
\begin{document}
\maketitle
\input{leap_ablation_results}
\input{leap_refresh_table}
\clearpage
\input{leap_proactive_table}
\begin{figure}[htbp]
\centering
\includegraphics[width=\linewidth]{proactive_activation_curves.pdf}
\caption{ROC diagnostics of the activation score and paired AUROC differences. Dots mark the original
threshold. Error bars are 95\% user-cluster bootstrap intervals. Unnormalized
routine ranking has the same trigger decisions and curve as LEAP.}
\end{figure}
\clearpage
\input{leap_user_history_table}
\section*{Experimental protocols}
\input{leap_ablation_methods}
\end{document}
"""
    (ROOT / "leap_ablation_report.tex").write_text(wrapper)
    overview = ["# LEAP 新增消融与分析", "", "三组研究均已完成；所有有效条件及原始输出保留。", "",
                "| 问题 | 对照 | 完整/自身条件 | 对照条件 | 差值及用户聚类 95% CI (pp) |",
                "|---|---|---:|---:|---:|",
                f"| 逐步更新参考（策略 SSR） | 上一步状态 | {pct(delay['method'])} | {pct(delay['reference'])} | {effect(delay)} |",
                f"| 更新频率（策略 SSR） | 每四步更新 | {pct(periodic['method'])} | {pct(periodic['reference'])} | {effect(periodic)} |",
                f"| 时间条件（激活 AUROC） | 无时间核 | {pct(pm['full']['auroc'])} | {pct(pm['no_time']['auroc'])} | {effect(time)} |",
                f"| 场景条件（激活 AUROC） | 无场景条件 | {pct(pm['full']['auroc'])} | {pct(pm['no_scenario']['auroc'])} | {effect(scenario)} |",
                f"| 用户应用证据（入口策略 SSR） | 部分匹配替换 | {pct(own['method'])} | {pct(own['reference'])} | {effect(own)} |", "",
                "## 解释边界", "",
                "- 更新实验保持完整 Top-8 历史池不变，策略始终看当前屏幕；不证明闭环错误恢复。",
                "- 去掉场景条件后，原阈值下 FA 从 37% 变为 62%，但 AUROC 差值区间跨零；不能声称场景条件普遍提高区分能力。",
                f"- 归一化的条件 Semantic 差值为 {effect(norm)} pp；独立查询重采样区间为 [{pct(norm_row['delta_ci95'][0])}, {pct(norm_row['delta_ci95'][1])}]，结论对重采样单位敏感。",
                f"- 十条历史全部匹配仅 25 个任务，其策略差值为 {effect(strict)} pp；完整与部分替换都保留，且后者仍保有未匹配的自身历史。",
                f"- 语义匹配抽查发现相近 embedding 不保证完整任务等价；另行保留 {exact['n_steps']} 个完全同文本替换任务，其策略差值为 {effect(exact)} pp。此子集在入口推理前确定。",
                "- 用户历史实验只检验应用入口，不能推广为后续内容偏好恢复。",
                "- 历史固定轨迹实验中的等长子集负收益仍然有效，见 ../state_conditioning/RESULTS_zh.md；新实验没有推翻该边界。", "",
                "## 交付", "", "- 各 study 子目录：完整报告、输入、输出、统计、源文件哈希。",
                "- leap_ablation_results.tex：可供论文使用的英文结果段落。",
                "- leap_*_table.tex：三组完整对照表。",
                "- leap_ablation_methods.tex：协议与分析方法。",
                "- leap_ablation_report.pdf：独立编译的实验报告。",
                "- proactive_activation_curves.pdf：完整激活曲线与不确定性。", ""]
    (ROOT / "RESULTS_zh.md").write_text("\n".join(overview))
    print("Exported all three complete studies; no result-sign filter was applied.")


if __name__ == "__main__":
    main()

"""Export verified paired results without modifying the active manuscript."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "trace_method/results/state_conditioning"
NAMES = {"qwen3_vl_8b": "Qwen3-VL-8B", "qwen35_9b": "Qwen3.5-9B",
         "mai_ui_8b": "MAI-UI-8B", "gui_owl_1_5_8b": "GUI-Owl-1.5-8B"}


def p(value):
    return f"{100 * value:.2f}"


def effect(stats):
    low, high = stats["delta_ci95"]
    return f"{p(stats['delta'])} [{p(low)}, {p(high)}]"


def write(name, contents):
    (ROOT / name).write_text(contents.rstrip() + "\n")


def primary_paragraph(model):
    pair = "top1_state_minus_top1_ordinal"
    policy, evidence = model["policy"][pair], model["evidence"][pair]
    a, e = policy["all_nonentry"], evidence["all_nonentry"]
    strict, strict_e = policy["literal_and_app_match"], evidence["literal_and_app_match"]
    interaction = policy["unequal_minus_equal_gain"]
    return (
        r"\paragraph{Selecting a reference step within a retrieved task.} "
        "We hold the highest-ranked usable historical trajectory fixed throughout each episode and "
        "select its reference action either by ordinal position or by matching the current screen. "
        "The frozen Qwen3-VL-8B policy receives the same instruction, screen, teacher-forced action prefix, and "
        "evidence prompt template in both conditions. "
        f"Across {a['n_steps']:,} non-entry steps from {a['n_episodes']} episodes and {a['n_clusters']} users, the benchmark step "
        f"success of the retrieved historical actions increases from {p(e['reference'])}\\% to {p(e['method'])}\\%. "
        f"The frozen policy's step success changes from {p(a['reference'])}\\% to {p(a['method'])}\\%, "
        f"a difference of {p(a['delta'])} percentage points (95\\% user-cluster bootstrap CI "
        f"[{p(a['delta_ci95'][0])}, {p(a['delta_ci95'][1])}]). "
        f"For {strict['n_steps']} steps from {strict['n_episodes']} episodes with the same app and "
        "literally matching task goals after removal of the historical app-opening phrase, "
        f"evidence success increases from {p(strict_e['reference'])}\\% "
        f"to {p(strict_e['method'])}\\%, while the policy difference of {p(strict['delta'])} points "
        f"[{p(strict['delta_ci95'][0])}, {p(strict['delta_ci95'][1])}] remains uncertain. "
        "Task descriptions can therefore match while their reference actions remain misaligned; "
        "the observed improvement in evidence quality is larger than the improvement in policy outcomes.\n\n"
        f"State matching changes policy success by {effect(policy['same_step_count'])} points for equal historical "
        f"and target trajectory lengths, and {effect(policy['different_step_count'])} points for unequal lengths. "
        f"The difference between these gains is {p(interaction['unequal_minus_equal_gain'])} points "
        f"[{p(interaction['ci95'][0])}, {p(interaction['ci95'][1])}]. "
        "Length inequality is a structural proxy, not an annotation of task phase. "
        "We find no clear increase in benefit with action position. "
        "These exploratory results use teacher forcing and the benchmark's recorded actions; they do not "
        "establish closed-loop recovery or isolate personal preference recovery. "
        "The intervals reflect resampling users and do not include variation across inference runs."
    )


def main():
    analysis = json.loads((ROOT / "analysis.json").read_text())
    assert analysis["protocol"]["status"] == "complete", "Finish all policy runs before exporting"
    model = analysis["models"]["qwen3_vl_8b"]
    pair = "top1_state_minus_top1_ordinal"
    policy, evidence = model["policy"][pair], model["evidence"][pair]
    a, e = policy["all_nonentry"], evidence["all_nonentry"]
    labels = [("all_nonentry", "All non-entry steps"),
              ("task_match_consensus", "Goal-match consensus"),
              ("task_and_app_match", "Goal consensus + same app"),
              ("literal_and_app_match", "Literal goal match + same app"),
              ("interactive_actions", "Interactive actions"),
              ("without_exact_screenshot", "Nonidentical matched screens"),
              ("same_step_count", "Equal trajectory lengths"),
              ("different_step_count", "Unequal trajectory lengths")]
    table = [r"\begin{table*}[t]", r"\centering", r"\small",
             r"\caption{State conditioning with a fixed historical trajectory (Qwen3-VL-8B). "
             r"The same top-ranked trajectory is used for both conditions. Ordinal alignment selects "
             r"historical step $\min(t-1,T-1)$; state matching selects a step using the current screen. "
             r"Rates are non-entry step success (\%). Differences are percentage points with paired "
             r"95\% user-cluster bootstrap intervals (20,000 draws). Evidence rows score retrieved "
             r"actions; policy rows score the frozen model's final decisions. Goal-match consensus uses "
             r"two frozen text annotators, not human gold labels. Subgroups are exploratory.}",
             r"\label{tab:fixed-history-state}", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{llrrrr}", r"\toprule",
             r"Scored output & Subset & Steps & Ordinal & State & Difference [95\% CI] \\",
             r"\midrule"]
    for level, values in (("Policy", policy), ("Evidence", evidence)):
        for key, label in labels:
            s = values[key]
            table.append(f"{level} & {label} & {s['n_steps']:,} & {p(s['reference'])} & {p(s['method'])} & {effect(s)} " + r"\\")
        if level == "Policy":
            table.append(r"\midrule")
    table += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    write("fixed_history_table.tex", "\n".join(table))

    controls = [r"\begin{table}[t]", r"\centering", r"\small",
                r"\caption{Qwen3-VL-8B controls on 7,140 non-entry steps. Full-trajectory prompting "
                r"uses a different evidence format and prompt template. The preceding-step condition "
                r"is a deliberate perturbation of TRACE, not an optimized baseline.}",
                r"\label{tab:state-controls}", r"\begin{tabular}{lrr}", r"\toprule",
                r"Condition & Evidence SSR & Policy SSR \\", r"\midrule"]
    for key, label in [("top1_trajectory", "Fixed top-one: full record"),
                       ("top1_ordinal", "Fixed top-one: ordinal"),
                       ("top1_state", "Fixed top-one: current state"),
                       ("top8_previous", "TRACE: preceding step"),
                       ("top8_state", "TRACE: current state")]:
        direct = model["evidence"]["nonentry_rates"].get(key)
        controls.append(f"{label} & {p(direct) if direct is not None else '--'} & {p(model['policy']['nonentry_rates'][key])} " + r"\\")
    controls += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("state_controls_table.tex", "\n".join(controls))

    write("state_conditioning_paragraph.tex", primary_paragraph(model))

    methods = (
        r"\paragraph{Fixed-history diagnostic protocol.} "
        "We use the existing history split, retrieval scores, and history-calibrated visual representation "
        "without additional parameter tuning. For each test episode, we fix the highest-ranked usable "
        "trajectory retrieved from that user's historical pool. At non-entry step $t$, ordinal alignment "
        "selects historical action $\\min(t-1,T-1)$ using zero-based historical indices, where $T$ is the "
        "historical trajectory length. State matching selects an action from the same trajectory with "
        "TRACE's current-screen matching rule. Both actions are supplied through the same evidence "
        "template to a frozen Qwen3-VL-8B policy; the current screen and ground-truth preceding actions "
        "are visible in both conditions. Entry-step evidence is identical and entry steps are excluded "
        "from the primary analysis. We score retrieved reference actions separately from policy outputs "
        "using the benchmark's step-success rules. Neither reference selection nor task-relevance "
        "annotation uses target actions or policy predictions.\n\n"
        "The primary contrast uses all 7,140 non-entry steps. We estimate paired 95\\% confidence "
        "intervals by resampling 82 users with replacement 20,000 times, retaining each sampled user's "
        "complete set of episodes and steps; episode-cluster intervals provide a sensitivity check. "
        "Task-match consensus requires both frozen Qwen3-VL-8B and Qwen3.5-9B text annotators to label "
        "the current instruction and historical task description as matching. These are model judgments "
        "from the same model family. A separate literal-match subset compares normalized goals after "
        "removing the historical app-opening phrase and requires the same app. Subgroup analyses are "
        "exploratory and unadjusted for multiple comparisons. Inference uses greedy decoding with "
        "seed 0, a maximum context length of 8,192 tokens, and up to 256 output tokens. The intervals "
        "quantify data resampling uncertainty rather than variation across repeated inference runs."
    )
    write("state_conditioning_methods.tex", methods)
    captions = (
        r"\begin{figure*}[t]" "\n"
        r"\centering" "\n"
        r"\includegraphics[width=\textwidth]{../../../trace_method/results/state_conditioning/fixed_history_case.pdf}" "\n"
        r"\caption{Reference-step selection within the same historical task. "
        "The query and historical goal both ask to check match results in Hupu. Ordinal alignment "
        "selects a reference from the fourth historical screen, whereas current-state matching "
        "selects the first historical screen. The resulting policy action matches the recorded NBA-tab "
        "click only in the state-matched condition. The current NBA tab already appears selected, so "
        "the figure demonstrates agreement with the benchmark action, not the necessity of this click "
        "for successful execution. This example was selected after inspecting outcomes.}"
        "\n" r"\label{fig:fixed-history-case}" "\n" r"\end{figure*}" "\n\n"
        r"\begin{figure*}[t]" "\n"
        r"\centering" "\n"
        r"\includegraphics[width=\textwidth]{../../../trace_method/results/state_conditioning/evidence_policy_gap_case.pdf}" "\n"
        r"\caption{Improved reference evidence without improved policy output. "
        "For a 15-day weather query, state matching retrieves a historical click on the chart-view "
        "control that matches the recorded target, while ordinal alignment retrieves a termination "
        "action. The state-conditioned policy nevertheless waits, and both policy outputs score zero. "
        "This outcome-selected example illustrates a limit on converting useful historical evidence "
        "into the benchmark's target action; it is not a prevalence estimate.}"
        "\n" r"\label{fig:evidence-policy-gap}" "\n" r"\end{figure*}"
    )
    write("state_conditioning_case_figures.tex", captions)

    full = model["policy"]["top1_state_minus_top1_trajectory"]["all_nonentry"]
    phase = model["policy"]["top8_state_minus_top8_previous"]["all_nonentry"]
    controls_paragraph = (
        r"\paragraph{Additional evidence controls.} "
        f"Supplying the same top-ranked trajectory as a complete record yields {p(full['reference'])}\\% "
        f"non-entry policy step success, compared with {p(full['method'])}\\% for state-selected "
        f"action evidence (difference {effect(full)} percentage points). "
        "This comparison changes the prompt template and evidence format as well as the selection granularity. "
        f"Replacing TRACE's selected reference with the preceding historical action yields {p(phase['reference'])}\\%, "
        f"compared with {p(phase['method'])}\\% for the selected reference "
        f"(difference {effect(phase)} points). The preceding-action condition is a deliberate perturbation, "
        "not an optimized alternative. All intervals use paired user-cluster bootstrap resampling."
    )
    write("state_conditioning_controls_paragraph.tex", controls_paragraph)
    report = ["# 固定历史轨迹的状态条件化实验", "",
              "这组实验保持 TRACE 的现有方法不变，检验检索到历史轨迹之后，如何选择当前步骤的参考动作。", "",
              f"结果支持一个有边界的主张：任务描述匹配并不保证参考动作与当前步骤一致，状态匹配能缓解这一问题。固定同一条历史后，状态匹配使参考动作分数提升 {p(e['delta'])} 个百分点，最终策略分数提升 {p(a['delta'])} 个百分点（用户聚类 95% CI [{p(a['delta_ci95'][0])}, {p(a['delta_ci95'][1])}]）。", "",
              "任务文字与应用均相同的严格子集也显示参考动作改善，但其策略增益尚不确定；等长轨迹子集出现负收益。这组结果适合支持参考步骤选择的作用与局限，不能证明个人偏好恢复，也没有显示收益随执行位置持续增加。", "",
              "## 核心结果", "",
              f"主模型：Qwen3-VL-8B。完整运行 775 个任务、7,915 步；主分析排除 775 个应用入口步骤，覆盖 {a['n_clusters']} 位用户、7,140 个后续步骤。", "",
              f"任务总长度（包含入口）的中位数为 {model['audit']['episode_length_summary']['median']:g} 步，范围为 {model['audit']['episode_length_summary']['min']}–{model['audit']['episode_length_summary']['max']} 步；其中 {model['audit']['episode_length_summary']['episodes_with_at_least_13_steps']} 个任务至少有 13 步。", "",
              "| 输出 / 子集 | 顺序选择 (%) | 状态匹配 (%) | 差值及用户聚类 95% CI (pp) | 任务 / 步数 |",
              "|---|---:|---:|---|---:|"]
    for level, values in (("策略决策", policy), ("历史参考动作", evidence)):
        for key, label in labels:
            s = values[key]
            report.append(f"| {level} / {label} | {p(s['reference'])} | {p(s['method'])} | {effect(s)} | {s['n_episodes']} / {s['n_steps']} |")
    report += ["", f"主策略对比：状态匹配纠正 {a['wins']} 步，同时使 {a['losses']} 步从正确变为错误。逐任务聚类的敏感性区间为 [{p(a['episode_cluster_delta_ci95'][0])}, {p(a['episode_cluster_delta_ci95'][1])}] pp。",
               "", "## 额外对照", "",
               f"- 同一条完整历史直接输入策略：{p(full['reference'])}%；选取当前状态参考动作：{p(full['method'])}%，差值 {effect(full)} pp。这一比较使用不同提示模板与证据格式，不能解释为单一状态变量的因果效应。",
               f"- 将完整 TRACE 选中的历史动作替换为前一步：{p(phase['reference'])}%；完整 TRACE：{p(phase['method'])}%，差值 {effect(phase)} pp。这是人为扰动检查。",
               "", "## 四种视觉表征的参考动作诊断", "",
               "这里只复现参考动作选择，没有将其他模型的参考动作分数冒充完整策略结果。", "",
               "| 表征 | 顺序 (%) | 当前状态 (%) | 差值 [95% CI] (pp) |", "|---|---:|---:|---|"]
    for key, value in analysis["models"].items():
        s = value["evidence"][pair]["all_nonentry"]
        report.append(f"| {NAMES[key]} | {p(s['reference'])} | {p(s['method'])} | {effect(s)} |")
    report += ["", "## 点击阈值敏感性分析", "",
               "保留基准的其他评分规则，只将归一化点击距离阈值从 0.14 收紧。下表仍报告所有 7,140 个后续步骤，非点击步骤的分数不变。", "",
               "| 输出 | 点击阈值 | 顺序 (%) | 当前状态 (%) | 差值 [95% CI] (pp) |",
               "|---|---:|---:|---:|---|"]
    for level, scores in model["click_tolerance_sensitivity"].items():
        for threshold, s in scores.items():
            report.append(f"| {level} | {threshold} | {p(s['reference'])} | {p(s['method'])} | {effect(s)} |")
    report += ["", "## 输入内容恢复的额外诊断", "",
               "对 type 动作额外采用去空白、忽略大小写后的精确文本匹配，避免把基准的字符串相似度分数解释为精确内容恢复。", "",
               "| 输出 | 子集 | 步数 | 顺序精确命中 | 状态精确命中 |", "|---|---|---:|---:|---:|"]
    for level, values in model["typed_value_diagnostic"].items():
        for name, s in values.items():
            report.append(f"| {level} | {name} | {s['n_steps']} | {s['exact_counts']['top1_ordinal']} | {s['exact_counts']['top1_state']} |")
    report += ["", "输入值未字面出现在指令中，只是一个描述性检查。它可能涉及改写、拼音或部分输入，也可能已出现在当前屏幕或真实动作前缀中，不能直接当作个人偏好标签。此检查没有提供控制用户身份或个人偏好因果作用的证据。"]
    report += ["", "## 参考动作与策略结果之间的差距", "",
               "下面按参考动作的评分划分全部步骤，只用于描述错误变化；这些分组使用结果标签，不能作为因果中介分析。", "",
               "| 参考动作的变化 | 步数 | 顺序组策略 (%) | 状态组策略 (%) | 策略净增正确步数 |",
               "|---|---:|---:|---:|---:|"]
    for name, s in model["reference_to_policy_transition"]["groups"].items():
        report.append(f"| {name} | {s['n_steps']} | {p(s['reference'])} | {p(s['method'])} | {s['wins'] - s['losses']} |")
    interaction = policy["late_minus_early_gain"]
    length_interaction = policy["unequal_minus_equal_gain"]
    rerun = model["audit"]["full_trace_rerun"]
    rerun_note = ("解析动作和评分逐条一致。这项单条件复跑不能量化其他条件之间的运行方差。"
                  if rerun["action_changes"] == rerun["success_changes"] == 0 else
                  "旧运行未保存全部源代码哈希，不能将输出差异单独归因于数值非确定性。")
    report += ["", "## 可解释范围", "",
               "1. 主对照在每个任务中固定同一条同用户历史轨迹，提示模板和证据字段一致；变量是历史动作的选取方式。当前屏幕在两组策略输入中均可见。",
               "2. 任务相关性由 Qwen3-VL-8B 和 Qwen3.5-9B 分别根据当前模糊指令与历史任务描述标注，不读取目标动作、策略输出或目标完整指令；它们是自动标注，不能称作人工确认的检索正确率。两个模型属于同一模型家族，错误可能相关。",
               "3. 按原评测规范使用真实历史动作前缀（teacher forcing）。这组离线结果不能证明闭环中的错误恢复，也不能单独证明个人偏好恢复。",
               f"4. 不能仅凭长任务上的数值声称收益随步骤增加。策略的晚期减早期收益为 {p(interaction['late_minus_early_gain'])} pp，95% CI [{p(interaction['ci95'][0])}, {p(interaction['ci95'][1])}]。",
               "5. 使用已评测过的数据开展机制分析，属于探索性实验。主对照先于新策略预测确定；完整历史、任务与应用联合匹配是查看参考动作诊断后补充的探索性检查。未按实验结果调参。",
               "6. 各子集未作多重比较校正。轨迹长度是否相同不能直接等同于语义阶段是否匹配。",
               f"   直接比较两类长度子集的策略收益，不等长减等长为 {p(length_interaction['unequal_minus_equal_gain'])} pp，用户聚类 95% CI [{p(length_interaction['ci95'][0])}, {p(length_interaction['ci95'][1])}]。这是在主对照完成后补充的探索性交互检验。",
               "7. 定性检查发现一个点击标签与当前可见按钮位置不一致的样本，详见 `qualitative_case_audit.json`。它在统计中按原基准保留，但不作为经核实的操作成功案例。该非随机检查不能估计整个基准的标注噪声比例。",
               "", "## 完整性与复现", "",
               "- 历史池与 775 个测试任务无整轨迹重叠；选择过程不使用目标答案。上下文、预测、模型版本及文件哈希保留在原始 JSON 中。",
               f"- 固定轨迹中有 {model['audit']['exact_screenshot_pairs']} 个查询与所选历史屏幕像素完全相同的步骤；排除后另行报告。该检查不等于排除了所有近重复。",
               "- 主置信区间使用 20,000 次配对用户聚类 bootstrap；同次重采样保留用户的全部任务和步骤。另提供任务聚类敏感性分析。",
               f"- 两组有 {model['audit']['identical_nonentry_evidence']} 个后续步骤使用完全相同的证据上下文，其中 {model['audit']['policy_success_changes_with_identical_evidence']} 步的策略评分仍发生变化，净差值为 {p(model['audit']['success_delta_with_identical_evidence'])} pp。入口步骤有 {model['audit']['entry_policy_success_changes']} 个评分变化。使用贪心解码和相同 seed 不保证不同批次计算路径下逐步输出完全一致；所报区间刻画数据重采样，未单独估计推理运行方差。",
               f"- 完整 TRACE 此次复跑的后续步骤分数为 {p(rerun['current_nonentry_rate'])}%，旧结果为 {p(rerun['previous_nonentry_rate'])}%；全部 7,915 步中有 {rerun['action_changes']} 个解析动作变化、{rerun['success_changes']} 个评分变化。{rerun_note}额外对照均使用本次复跑结果。",
               "- 五组完整策略预测均保存于 `qwen3_vl_8b/*_policy_gpu0.json`；其他三个模型保存参考动作诊断。", "",
               "```bash", "# 仓库根目录；分别使用 hymem (CPU 分析) 与 hymem-serve (vLLM) 环境",
               "python trace_method/build_state_conditioning_experiment.py --model-key qwen3_vl_8b",
               "# 对其余三个 model-key 同样执行上条命令；标注器命令参数见 --help",
               "python trace_method/build_state_conditioning_full_context.py",
               "python trace_method/seal_state_conditioning_sources.py",
               "VARIANTS='top1_ordinal top1_state top8_previous top8_state top1_trajectory' bash scripts/execution/eval_state_conditioning.sh",
               "python trace_method/analyze_state_conditioning_experiment.py --require-policy",
               "python trace_method/export_state_conditioning_paper.py", "```", "",
               "## 论文材料", "",
               "- `fixed_history_table.tex`：主对照与边界子集，区分策略与参考动作。",
               "- `state_controls_table.tex`：五组策略对照。",
               "- `state_conditioning_paragraph.tex`：英文结果段落。",
               "- `state_conditioning_controls_paragraph.tex`：额外对照的英文说明。",
               "- `state_conditioning_methods.tex`：固定历史对照、任务匹配子集及统计方法的英文说明。",
               "- `state_conditioning_case_figures.tex`：两张案例图的 LaTeX 插图片段及有边界说明的图注。",
               "- `state_conditioning_forest.pdf` / `.png`：主效果及子集置信区间。",
               "- `state_conditioning_by_position.pdf` / `.png`：执行位置分层。",
               "- `fixed_history_case.pdf` / `.png`：同一任务历史中参考步骤错位的示例。",
               "- `evidence_policy_gap_case.pdf` / `.png`：正确参考动作未转化为正确策略动作的示例。",
               "- `qualitative_case_audit.json`：案例来源、图像哈希、助手目视核对记录及弃用案例理由。",
               "- `analysis.json`、`paired_results.csv`、`REPORT.md`：完整统计与审计。",
               "", "这些材料独立保存，尚未插入主稿。PDF 图已生成并检查；当前环境缺少可用的 LaTeX 编译器，TeX 片段尚未编译，合入主稿后仍需检查最终排版。"]
    write("RESULTS_zh.md", "\n".join(report))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(11.3, 4.4), sharex=True, sharey=True)
    y = np.arange(len(labels))[::-1]
    for ax, values, title, color in ((axes[0], evidence, "Historical evidence diagnostic", "#8296AB"),
                                    (axes[1], policy, "Frozen policy output", "#166A87")):
        means = np.array([values[key]["delta"] for key, _ in labels]) * 100
        ci = np.array([values[key]["delta_ci95"] for key, _ in labels]) * 100
        ax.errorbar(means, y, xerr=np.vstack([means - ci[:, 0], ci[:, 1] - means]),
                    fmt="o", capsize=3, color=color, elinewidth=1.5, markersize=6)
        ax.axvline(0, color="#777777", linestyle="--", linewidth=1)
        ax.set_yticks(y, [name for _, name in labels])
        ax.set_xlabel("State minus ordinal step success (pp)")
        ax.set_title(title)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", alpha=0.15)
    fig.tight_layout()
    for extension in ("pdf", "png"):
        fig.savefig(ROOT / f"state_conditioning_forest.{extension}", dpi=240, bbox_inches="tight")
    plt.close(fig)
    paths = [ROOT / name for name in ("analysis.json", "fixed_history_table.tex", "state_controls_table.tex",
                                     "state_conditioning_paragraph.tex", "state_conditioning_controls_paragraph.tex",
                                     "state_conditioning_methods.tex", "state_conditioning_case_figures.tex", "RESULTS_zh.md")]
    write("paper_export_hashes.json", json.dumps({str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest()
                                                for path in paths}, indent=2))
    print(ROOT / "RESULTS_zh.md")


if __name__ == "__main__":
    main()

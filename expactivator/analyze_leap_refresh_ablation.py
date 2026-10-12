"""Complete paired analysis; never discard a valid ablation by effect sign."""
from __future__ import annotations
import csv
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("USE_TF", "0")
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from expactivator.analyze_state_conditioning_experiment import paired_cluster, by_id, label_action_kind
from expactivator.build_leap_refresh_ablation import sha
from expactivator.evaluate_execution import image_size
from expactivator.execution_core import resolve_test_image
from src.eval_execution.utils import convert_action_qwen


def main():
    out = REPO / "expactivator/results/leap_ablations/refresh"
    sealed = json.loads((out / "source_hashes.json").read_text())
    for name, digest in sealed.items():
        assert sha(REPO / name) == digest, name
    metadata = json.loads((out / "step_metadata.json").read_text())["rows"]
    test = json.loads((REPO / "data/personal_execution/personal_execution.json").read_text())
    expected = [r["id"] for r in test]
    targets = {r["id"]: r for r in test}
    contexts, policies, evidence, protocols, metrics = {}, {}, {}, {}, {}
    for variant in ("current", "delay1", "refresh4"):
        context_path = out / f"{variant}_contexts.json"
        contexts[variant] = json.loads(context_path.read_text())["contexts"]
        result = json.loads((out / f"{variant}_policy_gpu0.json").read_text())
        assert [r["episode_id"] for r in result["outputs"]] == expected
        assert result["protocol"]["memory_contexts_sha256"] == sha(context_path)
        assert result["protocol"]["test_sha256"] == sha(REPO / "data/personal_execution/personal_execution.json")
        policies[variant] = by_id(result["outputs"])
        protocols[variant] = {k: v for k, v in result["protocol"].items()
                              if k not in ("memory_contexts", "memory_contexts_sha256")}
        metrics[variant] = result["metrics"]
        evidence[variant] = {}
        for row in test:
            key = row["id"]
            assert policies[variant][key]["grounded_evidence"] == contexts[variant][key]
            action = contexts[variant][key]["retrieved_action"]
            size = image_size(str(resolve_test_image(row["img_path"], Path("/storage/zhangfuyao/datasets/fingertip20k"))))
            kind, score = convert_action_qwen(action, row["answer"], *size, coord_map=True)
            evidence[variant][key] = {"sr": score, "type_acc": kind}
    assert protocols["current"] == protocols["delay1"] == protocols["refresh4"]
    nonentry = [r for r in metadata if r["position"] > 0]
    strata = {
        "all_nonentry": nonentry,
        "after_first_nonentry": [r for r in nonentry if r["position"] > 1],
        "interactive": [r for r in nonentry if label_action_kind(targets[r["step_id"]]["answer"])
                        in ("click", "long_click", "scroll", "type")],
    }
    contrasts = {}
    for comparator in ("delay1", "refresh4"):
        contrasts[comparator] = {}
        for level, values in (("policy", policies), ("evidence", evidence)):
            contrasts[comparator][level] = {
                name: paired_cluster(values[comparator], values["current"], subset)
                for name, subset in strata.items()}
        contrasts[comparator]["episode_cluster_sensitivity"] = paired_cluster(
            policies[comparator], policies["current"], nonentry, cluster="episode_id")
        identical = [r for r in nonentry if contexts[comparator][r["step_id"]] == contexts["current"][r["step_id"]]]
        changed = [r for r in nonentry if contexts[comparator][r["step_id"]] != contexts["current"][r["step_id"]]]
        contrasts[comparator]["identical_context_diagnostic"] = paired_cluster(policies[comparator], policies["current"], identical)
        contrasts[comparator]["changed_context_diagnostic"] = paired_cluster(policies[comparator], policies["current"], changed)
    old = by_id(json.loads((REPO / "expactivator/results/state_conditioning/qwen3_vl_8b/top8_state_policy_gpu0.json").read_text())["outputs"])
    repeat = {"parsed_action_changes": sum(old[k]["output"] != policies["current"][k]["output"] for k in expected),
              "score_changes": sum(old[k]["sr"] != policies["current"][k]["sr"] for k in expected)}
    result = {"status": "complete", "primary_contrast": "current minus delay1; policy all_nonentry",
              "secondary_contrast": "current minus refresh4", "inference_protocol": protocols["current"],
              "teacher_forced": True, "all_declared_conditions_retained": True,
              "metrics": metrics, "contrasts": contrasts, "current_reproduction": repeat,
              "source_hashes": sealed,
              "output_hashes": {f"{v}_policy_gpu0.json": sha(out / f"{v}_policy_gpu0.json") for v in policies}}
    (out / "analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    with (out / "paired_results.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step_id", "episode_id", "user_id", "position"] +
                                 [f"{v}_{level}" for v in policies for level in ("policy_sr", "evidence_sr")])
        writer.writeheader()
        for row in metadata:
            writer.writerow({**{k: row[k] for k in ("step_id", "episode_id", "user_id", "position")},
                             **{f"{v}_{level}_sr": values[v][row["step_id"]]["sr"]
                                for v in policies for level, values in (("policy", policies), ("evidence", evidence))}})
    def interval(value):
        return f"{100*value['delta']:.2f} [{100*value['delta_ci95'][0]:.2f}, {100*value['delta_ci95'][1]:.2f}]"
    report = ["# LEAP 参考更新消融", "", "三组均为完整 Qwen3-VL-8B 冻结策略运行，主比较覆盖 7,140 个非入口步骤。", "",
              "| 对照 | 对照策略 SSR (%) | 当前更新 SSR (%) | 当前减对照及用户聚类 95% CI (pp) |",
              "|---|---:|---:|---:|"]
    for v, label in (("delay1", "上一步状态选择"), ("refresh4", "每四步更新")):
        c = contrasts[v]["policy"]["all_nonentry"]
        report.append(f"| {label} | {100*c['reference']:.2f} | {100*c['method']:.2f} | {interval(c)} |")
    report += ["", "参考动作对照（非最终策略）："]
    for v in ("delay1", "refresh4"):
        report.append(f"- 当前更新减 {v}: {interval(contrasts[v]['evidence']['all_nonentry'])} pp。")
    report += ["", "全部预设条件及原始输出均保留。主分析为 current−delay1，refresh4 为次要比较。",
               "区间描述用户重采样，不包含推理运行方差；使用 teacher forcing，不代表闭环成功或错误恢复。",
               "相同上下文诊断、入口结果与重复运行差异见 analysis.json。", ""]
    (out / "RESULTS_zh.md").write_text("\n".join(report))
    print("\n".join(report))


if __name__ == "__main__":
    main()

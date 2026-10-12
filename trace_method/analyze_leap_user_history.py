"""Paired application-entry evidence intervention with complete coverage logs."""
from __future__ import annotations
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("USE_TF", "0")
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from trace_method.analyze_state_conditioning_experiment import paired_cluster, by_id
from trace_method.build_leap_refresh_ablation import sha
from trace_method.evaluate_execution import image_size
from trace_method.execution_core import resolve_test_image
from src.eval_execution.utils import convert_action_qwen


def main():
    out = REPO / "trace_method/results/leap_ablations/user_history_partial"
    strict_dir = REPO / "trace_method/results/leap_ablations/user_history"
    sealed = json.loads((out / "source_hashes.json").read_text())
    for name, digest in sealed.items():
        assert sha(REPO / name) == digest, name
    test_path = out / "entry_test.json"
    test = json.loads(test_path.read_text())
    expected = [r["id"] for r in test]
    assignment = json.loads((out / "assignments.json").read_text())["rows"]
    rows_by_id = {r["step_id"]: r for r in assignment}
    metadata = [{**rows_by_id[key], "position": 0} for key in expected]
    contexts = {v: json.loads((out / f"{v}_contexts.json").read_text())["contexts"]
                for v in ("own", "partial_other")}
    policies, protocols, metrics = {}, {}, {}
    for variant in ("own", "partial_other", "base"):
        data = json.loads((out / f"{variant}_policy_gpu0.json").read_text())
        assert [r["episode_id"] for r in data["outputs"]] == expected
        assert data["protocol"]["test_sha256"] == sha(test_path)
        policies[variant] = by_id(data["outputs"])
        protocols[variant] = {k:v for k,v in data["protocol"].items() if k not in
                              ("memory_contexts", "memory_contexts_sha256", "memory_context_scope", "baseline_name")}
        metrics[variant] = data["metrics"]
        if variant != "base":
            assert data["protocol"]["memory_contexts_sha256"] == sha(out / f"{variant}_contexts.json")
            assert all(policies[variant][key]["grounded_evidence"] == contexts[variant][key] for key in expected)
    assert protocols["own"] == protocols["partial_other"] == protocols["base"]
    strata = {"all_775": metadata,
              "at_least_one_replacement": [r for r in metadata if r["n_replaced"] > 0],
              "all_replaced_goals_exact": [r for r in metadata if r["n_replaced"] > 0 and all(r["exact_residual_matches"])],
              "strict_complete_match": [r for r in metadata if r["complete_match"]]}
    evidence = {v: {} for v in contexts}
    for row in test:
        size = image_size(str(resolve_test_image(row["img_path"], Path("/storage/zhangfuyao/datasets/fingertip20k"))))
        for variant in contexts:
            kind, score = convert_action_qwen(contexts[variant][row["id"]]["retrieved_action"], row["answer"], *size, coord_map=True)
            evidence[variant][row["id"]] = {"sr": score, "type_acc": kind}
    contrasts = {v: {name: paired_cluster(policies[v], policies["own"], subset)
                    for name, subset in strata.items()} for v in ("partial_other", "base")}
    evidence_contrasts = {name: paired_cluster(evidence["partial_other"], evidence["own"], subset)
                         for name, subset in strata.items()}
    episode_sensitivity = paired_cluster(policies["partial_other"], policies["own"], metadata, cluster="episode_id")
    unchanged = [r for r in metadata if contexts["own"][r["step_id"]] == contexts["partial_other"][r["step_id"]]]
    unchanged_diagnostic = paired_cluster(policies["partial_other"], policies["own"], unchanged)
    strict_contexts = json.loads((strict_dir / "matched_other_contexts.json").read_text())["contexts"]
    assert set(strict_contexts) == {r["step_id"] for r in strata["strict_complete_match"]}
    assert all(strict_contexts[k] == contexts["partial_other"][k] for k in strict_contexts)
    result = {"status": "complete", "scope": "application-entry evidence, not later content preferences",
              "primary_extension_contrast": "own minus partial_other on all_775",
              "strict_contrast_reused_from_identical_contexts": True,
              "matching_summary": json.loads((out / "matching_summary.json").read_text()),
              "policy_metrics": metrics, "policy_contrasts": contrasts, "evidence_contrasts": evidence_contrasts,
              "episode_cluster_sensitivity": episode_sensitivity, "identical_context_diagnostic": unchanged_diagnostic,
              "inference_protocol": protocols["own"], "source_hashes": sealed,
              "all_declared_conditions_retained": True,
              "matching_quality_note_sha256": sha(REPO / "trace_method/LEAP_HISTORY_MATCH_QUALITY_NOTE.md"),
              "output_hashes": {f"{v}_policy_gpu0.json": sha(out / f"{v}_policy_gpu0.json") for v in policies}}
    (out / "analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    strict_report = {"status": "complete", "n": len(strata["strict_complete_match"]),
                     "prediction_source": "../user_history_partial; identical own/donor contexts, evaluated in shared 775-entry batches",
                     "own_minus_other_policy": contrasts["partial_other"]["strict_complete_match"],
                     "own_minus_other_evidence": evidence_contrasts["strict_complete_match"],
                     "parent_analysis_sha256": sha(out / "analysis.json")}
    (strict_dir / "analysis.json").write_text(json.dumps(strict_report, ensure_ascii=False, indent=2)+"\n")
    lines = ["# LEAP 用户历史替换诊断", "", "仅评价应用入口，三个条件在同一 775 个入口步骤上独立运行冻结策略。", "",
             "| 子集 | n | 自身历史 SSR (%) | 匹配替换 SSR (%) | 自身减替换及用户聚类 95% CI (pp) |",
             "|---|---:|---:|---:|---:|"]
    for name in strata:
        c = contrasts["partial_other"][name]
        lines.append(f"| {name} | {c['n_steps']} | {100*c['method']:.2f} | {100*c['reference']:.2f} | "
                     f"{100*c['delta']:.2f} [{100*c['delta_ci95'][0]:.2f}, {100*c['delta_ci95'][1]:.2f}] |")
    base = contrasts["base"]["all_775"]
    lines += ["", f"无历史策略入口 SSR：{100*base['reference']:.2f}%；自身历史增益 {100*base['delta']:.2f} pp。",
              "严格十条匹配仅覆盖 25 个任务；部分替换覆盖 536 个任务，共替换 2,101/7,750 条记录，改变 118 个应用参考。",
              "未匹配记录保持自身历史；因此这是部分跨用户替换诊断，不是完全移除个性化。不能外推至后续内容偏好恢复。",
              "完全同文本子集要求所有被替换历史目标均与 donor 文本一致；该敏感性分析在入口推理前确定。余弦匹配不保证完整任务语义等价。",
              "完整匹配结果保留；扩展方案在查看策略结果前因匹配覆盖不足而固定。", ""]
    (out / "RESULTS_zh.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()

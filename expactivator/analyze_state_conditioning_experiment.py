"""Paired clustered analysis of fixed-trajectory ExpActivator controls.

All main comparisons exclude entry actions. User resampling is primary;
episode resampling is sensitivity analysis. Direct evidence is never reported
as a policy result. This module also audits pairing and protocol integrity.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter
from functools import lru_cache
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paired_cluster(reference, method, metadata, cluster="user_id", samples=20000, seed=20260912):
    """Ratio of pooled successes, resampling whole clusters with replacement."""
    if not metadata:
        return {"n_steps": 0, "n_clusters": 0}
    ids = [row["step_id"] for row in metadata]
    group_names = sorted({str(row[cluster]) for row in metadata})
    lookup = {name: i for i, name in enumerate(group_names)}
    values = np.zeros((len(group_names), 3), dtype=np.float64)
    a = np.array([reference[key]["sr"] for key in ids], dtype=float)
    b = np.array([method[key]["sr"] for key in ids], dtype=float)
    for row, first, second in zip(metadata, a, b):
        values[lookup[str(row[cluster])]] += [1, first, second]
    rng = np.random.default_rng(seed)
    estimates = []
    for offset in range(0, samples, 1000):
        weights = rng.multinomial(len(values), np.full(len(values), 1 / len(values)),
                                  size=min(1000, samples - offset))
        sums = weights @ values
        estimates.append(np.column_stack([sums[:, 1] / sums[:, 0], sums[:, 2] / sums[:, 0],
                                          (sums[:, 2] - sums[:, 1]) / sums[:, 0]]))
    boot = np.concatenate(estimates)
    intervals = np.quantile(boot, [0.025, 0.975], axis=0)
    return {
        "n_steps": len(ids), "n_clusters": len(group_names), "cluster": cluster,
        "n_episodes": len({row["episode_id"] for row in metadata}),
        "reference": float(a.mean()), "method": float(b.mean()),
        "delta": float((b - a).mean()),
        "reference_ci95": intervals[:, 0].tolist(), "method_ci95": intervals[:, 1].tolist(),
        "delta_ci95": intervals[:, 2].tolist(),
        "wins": int(((a == 0) & (b == 1)).sum()), "losses": int(((a == 1) & (b == 0)).sum()),
        "both_correct": int(((a == 1) & (b == 1)).sum()),
        "both_wrong": int(((a == 0) & (b == 0)).sum()),
        "bootstrap_samples": samples, "seed": seed,
    }


def clustered_gain_interaction(reference, method, metadata, memberships, samples=20000):
    """Second-minus-first gain using the same sampled users in both strata."""
    groups = sorted({row["user_id"] for row in metadata})
    index = {user: i for i, user in enumerate(groups)}
    values = np.zeros((len(groups), 4))
    for row, membership in zip(metadata, memberships):
        if membership is None:
            continue
        offset = 2 * membership
        k = index[row["user_id"]]
        values[k, offset] += 1
        values[k, offset + 1] += method[row["step_id"]]["sr"] - reference[row["step_id"]]["sr"]
    rng = np.random.default_rng(20260912)
    estimates = []
    for i in range(0, samples, 1000):
        w = rng.multinomial(len(groups), np.full(len(groups), 1 / len(groups)), size=min(1000, samples - i))
        v = w @ values
        valid = (v[:, 0] > 0) & (v[:, 2] > 0)
        v = v[valid]
        estimates.extend(v[:, 3] / v[:, 2] - v[:, 1] / v[:, 0])
    total = values.sum(0)
    return {"gain_difference": float(total[3] / total[2] - total[1] / total[0]),
            "ci95": np.quantile(estimates, [0.025, 0.975]).tolist(), "cluster": "user_id"}


def interaction_by_stage(reference, method, metadata, samples=20000):
    membership = [0 if r["position"] <= 3 else 1 if r["position"] >= 8 else None for r in metadata]
    result = clustered_gain_interaction(reference, method, metadata, membership, samples)
    result["late_minus_early_gain"] = result.pop("gain_difference")
    return result


def interaction_by_length(reference, method, metadata, samples=20000):
    membership = [int(r["source_steps"] != r["episode_steps"] - 1) for r in metadata]
    result = clustered_gain_interaction(reference, method, metadata, membership, samples)
    result["unequal_minus_equal_gain"] = result.pop("gain_difference")
    return result


def by_id(rows):
    result = {row["episode_id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate prediction IDs")
    return result


def pp(value):
    return f"{100 * value:.2f}"


def label_action_kind(value):
    match = re.match(r"([a-z_]+)\s*\(", value)
    return match.group(1) if match else "nonstandard_label"


def typed_target(value):
    if label_action_kind(value) != "type":
        return None
    try:
        expression = ast.parse(value, mode="eval").body
        result = ast.literal_eval(next(k.value for k in expression.keywords if k.arg == "text"))
        return result if isinstance(result, str) else None
    except (ValueError, StopIteration, SyntaxError, AttributeError):
        return None


def normalize_literal(value):
    return "".join(str(value).split()).lower()


@lru_cache(maxsize=None)
def screen_size(path):
    relative = path.removeprefix("./data/Fingertip/")
    with Image.open(Path("/storage/zhangfuyao/datasets/fingertip20k") / relative) as image:
        return image.size


def rescore_clicks(rows, targets, threshold):
    """Change only the benchmark click-radius tolerance; retain other scores."""
    result = {}
    for key, row in rows.items():
        result[key] = {"sr": row["sr"]}
        if label_action_kind(targets[key]["answer"]) not in ("click", "long_click"):
            continue
        if not row["sr"]:
            continue
        width, height = screen_size(targets[key]["img_path"])
        match = re.search(r"\((\d+),\s*(\d+)\)", targets[key]["answer"])
        x, y = map(int, match.groups())
        coordinates = row["output"].get("coordinate", [0, 0])
        px, py = round(coordinates[0] / 999 * width), round(coordinates[1] / 999 * height)
        distance = np.hypot(int(x / width * 1000) - int(px / width * 1000),
                            int(y / height * 1000) - int(py / height * 1000)) / 1000
        result[key]["sr"] = int(distance < threshold)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="expactivator/results/state_conditioning")
    parser.add_argument("--primary-model", default="qwen3_vl_8b")
    parser.add_argument("--require-policy", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    args = parser.parse_args()
    root = REPO / args.root
    test_path = REPO / "data/personal_execution/personal_execution.json"
    test = json.loads(test_path.read_text())
    expected = [row["id"] for row in test]
    targets = {row["id"]: row for row in test}
    primary_dir = root / args.primary_model
    relevance_inputs = json.loads((primary_dir / "task_relevance_inputs.json").read_text())["rows"]
    judges = {}
    for path in sorted(primary_dir.glob("task_relevance_qwen*.json")):
        value = json.loads(path.read_text())
        assert value["protocol"]["input_sha256"] == sha(primary_dir / "task_relevance_inputs.json")
        judges[path.stem] = {r["episode_id"]: r["label"] for r in value["rows"]}
        assert len(value["rows"]) == len(judges[path.stem]) == 775
        assert set(judges[path.stem]) == {r["episode_id"] for r in relevance_inputs}
    if args.require_policy:
        assert set(judges) == {"task_relevance_qwen3", "task_relevance_qwen35"}, "Both specified task annotators are required"
    # Literal normalization is a separate, deterministic descriptive stratum.
    def normalized(text):
        return "".join(c.lower() for c in text if c.isalnum())
    literal_matches = {r["episode_id"] for r in relevance_inputs
                       if normalized(r["query"]) == normalized(r["historical_residual_goal"])}
    consensus = {r["episode_id"] for r in relevance_inputs
                 if len(judges) >= 2 and all(j[r["episode_id"]] == "match" for j in judges.values())}
    # Additional conservative diagnostic, defined after the evidence-only
    # analysis: require exact canonical app agreement as well as goal agreement.
    label_rows = json.loads((REPO / "expactivator/results/grounded/execution_app_labels.json").read_text())["rows"]
    source_apps = {f"{r['user_id']}_{r['time']}": r["app_name"] for r in label_rows}
    target_episodes = json.loads((REPO / "data/personal_execution/personal_samples.json").read_text())
    target_apps = {f"{r['user_id']}_{r['time']}": r["app_name"] for r in target_episodes}
    app_matches = {r["episode_id"] for r in relevance_inputs
                   if source_apps[r["source_episode_id"]] == target_apps[r["episode_id"]]}
    result = {
        "protocol": {"status": "complete" if args.require_policy else "interim",
                     "primary_model": args.primary_model, "primary_outcome": "non-entry SSR",
                     "primary_contrast": "top1_state minus top1_ordinal with identical source trajectory",
                     "primary_cluster": "user_id", "sensitivity_cluster": "episode_id",
                     "test_sha256": sha(test_path), "protocol_sha256": sha(REPO / "expactivator/STATE_CONDITIONING_PROTOCOL.md"),
                     "exploratory_reanalysis": True, "teacher_forced": True,
                     "personal_preference_recovery_independently_identified": False},
        "task_relevance": {"annotators": {name: dict(Counter(v.values())) for name, v in judges.items()},
                           "consensus_match_episodes": len(consensus), "literal_match_episodes": len(literal_matches),
                           "goal_and_app_match_episodes": len(consensus & app_matches),
                           "literal_and_app_match_episodes": len(literal_matches & app_matches),
                           "gold_status": "model-annotated, not human gold"},
        "models": {}, "source_hashes": {},
    }
    initial_source_hashes = json.loads((primary_dir / "inference_source_hashes.json").read_text())
    result["inference_source_audit"] = {}
    for name, initial_hash in initial_source_hashes.items():
        current_hash = sha(REPO / name)
        amended_protocol = name.endswith("STATE_CONDITIONING_PROTOCOL.md")
        if not amended_protocol:
            assert current_hash == initial_hash, f"Inference source changed after sealing: {name}"
        else:
            current_protocol = (REPO / name).read_text()
            original_section = current_protocol if current_hash == initial_hash else (
                current_protocol.split("## Additional exploratory checks")[0].rstrip() + "\n")
            assert hashlib.sha256(original_section.encode()).hexdigest() == initial_hash, "Original protocol section changed"
            (root / "protocol_at_inference_start.md").write_text(original_section)
        result["inference_source_audit"][name] = {
            "at_inference_start": initial_hash, "current": current_hash,
            "identical": current_hash == initial_hash,
            "documented_exploratory_protocol_amendments": amended_protocol,
            "original_protocol_section_preserved": True if amended_protocol else None,
        }
    csv_rows = []
    plot_data = {}
    for directory in sorted(root.iterdir()):
        diagnostic_path = directory / "evidence_diagnostics.json"
        if not diagnostic_path.exists():
            continue
        model = directory.name
        metadata = json.loads((directory / "step_metadata.json").read_text())["rows"]
        assert len(metadata) == 7140
        for row in metadata:
            assert targets[row["step_id"]]["user_id"] == row["user_id"] or int(targets[row["step_id"]]["user_id"]) == row["user_id"]
        evidence = json.loads(diagnostic_path.read_text())["outputs"]
        for rows in evidence.values():
            assert [r["episode_id"] for r in rows] == expected
        evidence = {variant: by_id(rows) for variant, rows in evidence.items()}
        policy = {}
        policy_protocols = {}
        for path in sorted(directory.glob("*_policy_gpu0.json")):
            value = json.loads(path.read_text())
            if len(value["outputs"]) != 7915:
                continue
            assert [r["episode_id"] for r in value["outputs"]] == expected
            variant = path.name.removesuffix("_policy_gpu0.json")
            policy[variant] = by_id(value["outputs"])
            policy_protocols[variant] = value["protocol"]
            assert value["protocol"]["test_sha256"] == sha(test_path)
            assert value["protocol"]["teacher_forced_history"] is True
            assert value["protocol"]["memory_contexts_sha256"] == sha(Path(value["protocol"]["memory_contexts"]))
            result["source_hashes"][str(path.relative_to(REPO))] = sha(path)
        if args.require_policy and model == args.primary_model:
            assert {"top1_ordinal", "top1_state", "top8_previous", "top8_state", "top1_trajectory"} <= set(policy), "Primary policy runs incomplete"
        for protocol in policy_protocols.values():
            for key in ("model_revision", "backend", "prompt_style", "pixels", "max_new_tokens", "max_model_len",
                        "request_batch_size", "max_num_seqs", "seed", "vllm_version", "transformers_version", "torch_version",
                        "gpu_memory_utilization", "physical_gpu", "visible_cuda_devices", "gpu_name",
                        "coordinate_protocol", "deterministic_decoding", "stop_string"):
                assert protocol[key] == next(iter(policy_protocols.values()))[key], f"Policy mismatch: {key}"
        source_contexts = {v: json.loads((directory / f"{v}_contexts.json").read_text())
                           for v in ("top1_ordinal", "top1_state", "top8_previous")}
        original_contexts = json.loads((REPO / f"expactivator/results/grounded/{model}_contexts.json").read_text())
        starts = [key for key in expected if key.endswith("_0")]
        for key in starts:
            assert all(c["contexts"][key] == original_contexts["contexts"][key] for c in source_contexts.values())
        full_path = directory / "top1_trajectory_contexts.json"
        if full_path.exists():
            full_context = json.loads(full_path.read_text())
            assert all(full_context["contexts"][key] == original_contexts["contexts"][key] for key in starts)
            assert full_context["sources"] == source_contexts["top1_ordinal"]["sources"]
            assert full_context["protocol"]["source_context_sha256"] == sha(directory / "top1_ordinal_contexts.json")
        assert source_contexts["top1_ordinal"]["protocol"]["history_audit"]["test_episodes_in_memory"] == 0
        source_id_sets = {}
        for row in metadata:
            a = source_contexts["top1_ordinal"]["sources"][row["step_id"]]
            b = source_contexts["top1_state"]["sources"][row["step_id"]]
            assert a["train_row_index"] == b["train_row_index"]
            source_id_sets.setdefault(row["episode_id"], set()).add(a["train_row_index"])
        assert all(len(values) == 1 for values in source_id_sets.values())
        strata = {
            "all_nonentry": metadata,
            "task_match_consensus": [r for r in metadata if r["episode_id"] in consensus],
            "task_and_app_match": [r for r in metadata if r["episode_id"] in consensus & app_matches],
            "task_match_literal": [r for r in metadata if r["episode_id"] in literal_matches],
            "literal_and_app_match": [r for r in metadata if r["episode_id"] in literal_matches & app_matches],
            "without_exact_screenshot": [r for r in metadata if not r["top1_same_pixels"]],
            "task_match_no_exact_screenshot": [r for r in metadata if r["episode_id"] in consensus and not r["top1_same_pixels"]],
            "positions_1_3": [r for r in metadata if r["position"] <= 3],
            "positions_4_7": [r for r in metadata if 4 <= r["position"] <= 7],
            "positions_8_plus": [r for r in metadata if r["position"] >= 8],
            "length_le7": [r for r in metadata if r["episode_steps"] <= 7],
            "length_8_12": [r for r in metadata if 8 <= r["episode_steps"] <= 12],
            "length_ge13": [r for r in metadata if r["episode_steps"] >= 13],
            "same_step_count": [r for r in metadata if r["source_steps"] == r["episode_steps"] - 1],
            "different_step_count": [r for r in metadata if r["source_steps"] != r["episode_steps"] - 1],
            "ordinal_not_clamped": [r for r in metadata if not r["ordinal_was_clamped"]],
            "interactive_actions": [r for r in metadata if label_action_kind(targets[r["step_id"]]["answer"])
                                    in ("click", "long_click", "scroll", "type")],
        }
        action_types = sorted({label_action_kind(targets[r["step_id"]]["answer"]) for r in metadata})
        strata.update({f"action_{kind}": [r for r in metadata
                        if label_action_kind(targets[r["step_id"]]["answer"]) == kind]
                       for kind in action_types})
        type_values = {r["step_id"]: typed_target(targets[r["step_id"]]["answer"]) for r in metadata}
        typed_rows = [r for r in metadata if type_values[r["step_id"]] is not None]
        absent_type_rows = [r for r in typed_rows
                           if len(normalize_literal(type_values[r["step_id"]])) >= 2
                           and normalize_literal(type_values[r["step_id"]]) not in
                           normalize_literal(targets[r["step_id"]]["ambiguous_instruction"])]
        model_result = {"evidence": {}, "policy": {}, "audit": {
            "fixed_top1_source_per_episode": True, "paired_source_trajectory_identical": True,
            "entry_context_identical_across_variants": True,
            "n_nonentry": len(metadata), "exact_screenshot_pairs": sum(r["top1_same_pixels"] for r in metadata),
            "different_selected_step": sum(r["index_displacement"] > 0 for r in metadata),
            "ordinal_clamped_steps": sum(r["ordinal_was_clamped"] for r in metadata),
            "phase_control_unchanged_at_first_history_state": sum(r["top8_previous_is_same_state"] for r in metadata),
        }}
        episode_lengths = np.array(list({r["episode_id"]: r["episode_steps"] for r in metadata}.values()))
        model_result["audit"]["episode_length_summary"] = {
            "includes_entry": True, "min": int(episode_lengths.min()),
            "median": float(np.median(episode_lengths)), "max": int(episode_lengths.max()),
            "episodes_with_at_least_13_steps": int((episode_lengths >= 13).sum()),
        }
        for level, variants in (("evidence", evidence), ("policy", policy)):
            for left, right in (("top1_ordinal", "top1_state"), ("top8_previous", "top8_state"), ("top1_trajectory", "top1_state")):
                if left not in variants or right not in variants:
                    continue
                contrast = f"{right}_minus_{left}"
                analyses = {}
                selected_strata = strata if contrast.startswith("top1") else {"all_nonentry": metadata}
                for name, subset in selected_strata.items():
                    stats = paired_cluster(variants[left], variants[right], subset, samples=args.bootstrap_samples)
                    if stats["n_steps"]:
                        stats["episode_cluster_delta_ci95"] = paired_cluster(variants[left], variants[right], subset,
                            cluster="episode_id", samples=args.bootstrap_samples)["delta_ci95"]
                        csv_rows.append({"model": model, "level": level, "contrast": contrast, "stratum": name,
                            **{k: stats[k] for k in ("n_steps", "n_episodes", "n_clusters", "reference", "method", "delta", "wins", "losses")},
                            "ci_low": stats["delta_ci95"][0], "ci_high": stats["delta_ci95"][1]})
                    analyses[name] = stats
                if contrast.startswith("top1"):
                    analyses["late_minus_early_gain"] = interaction_by_stage(variants[left], variants[right], metadata, args.bootstrap_samples)
                    analyses["unequal_minus_equal_gain"] = interaction_by_length(variants[left], variants[right], metadata, args.bootstrap_samples)
                model_result[level][contrast] = analyses
            if variants:
                model_result[level]["nonentry_rates"] = {v: sum(rows[r["step_id"]]["sr"] for r in metadata) / len(metadata)
                                                         for v, rows in variants.items()}
            if "top1_ordinal" in variants and "top1_state" in variants:
                strict_scores = {}
                for threshold in (0.07, 0.05):
                    left = rescore_clicks(variants["top1_ordinal"], targets, threshold)
                    right = rescore_clicks(variants["top1_state"], targets, threshold)
                    strict_scores[str(threshold)] = paired_cluster(left, right, metadata,
                                                                   samples=args.bootstrap_samples)
                model_result.setdefault("click_tolerance_sensitivity", {})[level] = strict_scores
                exact = {v: {r["step_id"]: {"sr": int(rows[r["step_id"]]["output"].get("action") == "type"
                         and normalize_literal(rows[r["step_id"]]["output"].get("text", "")) ==
                         normalize_literal(type_values[r["step_id"]]))} for r in typed_rows}
                         for v, rows in variants.items()}
                typed_diagnostic = {}
                for name, subset in (("all_typed_values", typed_rows), ("literal_value_absent_from_query", absent_type_rows)):
                    typed_diagnostic[name] = {
                        "n_steps": len(subset), "n_episodes": len({r["episode_id"] for r in subset}),
                        "exact_counts": {v: sum(rows[r["step_id"]]["sr"] for r in subset) for v, rows in exact.items()},
                        "primary_pair": paired_cluster(exact["top1_ordinal"], exact["top1_state"], subset,
                                                        samples=args.bootstrap_samples),
                    }
                model_result.setdefault("typed_value_diagnostic", {})[level] = typed_diagnostic
        if policy and "top1_ordinal" in policy and "top1_state" in policy:
            reference_groups = {}
            for first, second, name in ((0, 0, "both_evidence_wrong"), (0, 1, "state_corrects_evidence"),
                                        (1, 0, "state_spoils_evidence"), (1, 1, "both_evidence_correct")):
                subset = [r for r in metadata if evidence["top1_ordinal"][r["step_id"]]["sr"] == first
                          and evidence["top1_state"][r["step_id"]]["sr"] == second]
                reference_groups[name] = paired_cluster(policy["top1_ordinal"], policy["top1_state"], subset,
                                                       samples=args.bootstrap_samples)
            assert sum(s["n_steps"] for s in reference_groups.values()) == len(metadata)
            model_result["reference_to_policy_transition"] = {
                "role": "Descriptive outcome-conditioned error decomposition; not a causal mediator analysis.",
                "groups": reference_groups,
            }
            identical_evidence_ids = [r["step_id"] for r in metadata if evidence["top1_ordinal"][r["step_id"]]["output"] == evidence["top1_state"][r["step_id"]]["output"]]
            model_result["audit"]["identical_nonentry_evidence"] = len(identical_evidence_ids)
            assert all(source_contexts["top1_ordinal"]["contexts"][key] == source_contexts["top1_state"]["contexts"][key]
                       for key in identical_evidence_ids)
            model_result["audit"]["identical_evidence_implies_identical_context"] = True
            model_result["audit"]["policy_action_changes_with_identical_evidence"] = sum(policy["top1_ordinal"][key]["output"] != policy["top1_state"][key]["output"] for key in identical_evidence_ids)
            model_result["audit"]["policy_success_changes_with_identical_evidence"] = sum(policy["top1_ordinal"][key]["sr"] != policy["top1_state"][key]["sr"] for key in identical_evidence_ids)
            model_result["audit"]["success_delta_with_identical_evidence"] = sum(policy["top1_state"][key]["sr"] - policy["top1_ordinal"][key]["sr"] for key in identical_evidence_ids) / len(identical_evidence_ids)
            starts = [key for key in expected if key.endswith("_0")]
            model_result["audit"]["entry_policy_action_changes"] = sum(policy["top1_ordinal"][key]["output"] != policy["top1_state"][key]["output"] for key in starts)
            model_result["audit"]["entry_policy_success_changes"] = sum(policy["top1_ordinal"][key]["sr"] != policy["top1_state"][key]["sr"] for key in starts)
            model_result["audit"]["entry_policy_rates"] = {v: sum(rows[k]["sr"] for k in starts) / len(starts)
                                                           for v, rows in policy.items()}
        if "top8_state" in policy:
            old_path = REPO / f"expactivator/results/grounded/{model}_grounded_policy_gpu0.json"
            if old_path.exists():
                previous = by_id(json.loads(old_path.read_text())["outputs"])
                assert set(previous) == set(expected)
                model_result["audit"]["full_expactivator_rerun"] = {
                    "previous_output_sha256": sha(old_path),
                    "action_changes": sum(previous[k]["output"] != policy["top8_state"][k]["output"] for k in expected),
                    "success_changes": sum(previous[k]["sr"] != policy["top8_state"][k]["sr"] for k in expected),
                    "previous_nonentry_rate": sum(previous[r["step_id"]]["sr"] for r in metadata) / len(metadata),
                    "current_nonentry_rate": sum(policy["top8_state"][r["step_id"]]["sr"] for r in metadata) / len(metadata),
                }
        result["models"][model] = model_result
        result["source_hashes"][str(diagnostic_path.relative_to(REPO))] = sha(diagnostic_path)
        plot_data[model] = model_result

    if args.require_policy:
        assert {"qwen3_vl_8b", "qwen35_9b", "mai_ui_8b", "gui_owl_1_5_8b"} <= set(result["models"]), "Four evidence diagnostics are required"
    (root / "analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    if csv_rows:
        with (root / "paired_results.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
            writer.writeheader(); writer.writerows(csv_rows)
    lines = ["# State-conditioned use of fixed personal trajectories", "",
             f"Status: {result['protocol']['status']}. All rates below exclude the 775 entry steps.", "",
             "Uncertainty: paired user-cluster bootstrap, 20,000 draws. The CSV/JSON includes episode-cluster sensitivity intervals.", "",
             "| Model | Output scored | Condition | Ordinal / previous (%) | State-matched (%) | Delta (pp), user 95% CI | N steps |",
             "|---|---|---|---:|---:|---|---:|"]
    for model, values in result["models"].items():
        for level in ("evidence", "policy"):
            for contrast, analyses in values[level].items():
                if contrast == "nonentry_rates": continue
                for stratum in ("all_nonentry", "task_match_consensus", "without_exact_screenshot", "task_match_no_exact_screenshot"):
                    stats = analyses.get(stratum, {})
                    if not stats.get("n_steps"): continue
                    ci = stats["delta_ci95"]
                    lines.append(f"| {model} | {level} | {contrast}; {stratum} | {pp(stats['reference'])} | {pp(stats['method'])} | {pp(stats['delta'])} [{pp(ci[0])}, {pp(ci[1])}] | {stats['n_steps']} |")
    lines.extend(["", "## Interpretation boundaries", "",
                  "- The top-one trajectory is selected by the same semantic ranking and remains fixed across the primary pair and all steps of each episode.",
                  "- The consensus goal-matched subgroup requires agreement between two outcome-blind frozen-model annotators; these labels are not human gold.",
                  "- Direct evidence scores measure historical candidate quality. Only policy rows measure frozen-model decisions.",
                  "- The phase control deliberately perturbs the full ExpActivator match; it is not an alternative optimized algorithm.",
                  "- Teacher forcing precludes claims about closed-loop recovery. These measurements alone cannot separate personal preference recovery from generic GUI state alignment.",
                  "- The experiment was designed after the public benchmark and prior ExpActivator results were known. It is an exploratory mechanism study."])
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")

    if args.primary_model in plot_data:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42})
        values = plot_data[args.primary_model]
        levels = [level for level in ("evidence", "policy") if "top1_state_minus_top1_ordinal" in values[level]]
        fig, axes = plt.subplots(1, len(levels), figsize=(5.1 * len(levels), 3.7), squeeze=False)
        for ax, level in zip(axes[0], levels):
            stats = values[level]["top1_state_minus_top1_ordinal"]
            names = ["positions_1_3", "positions_4_7", "positions_8_plus"]
            x = np.arange(3)
            for offset, key, label, color in ((-0.18, "reference", "Ordinal alignment", "#8296AB"), (0.18, "method", "Current-state matching", "#166A87")):
                rates = np.array([stats[n][key] for n in names]) * 100
                intervals = np.array([stats[n][key + "_ci95"] for n in names]) * 100
                yerr = np.vstack([np.maximum(0, rates - intervals[:, 0]), np.maximum(0, intervals[:, 1] - rates)])
                ax.bar(x + offset, rates, width=0.34, color=color, label=label)
                ax.errorbar(x + offset, rates, yerr=yerr, fmt="none", ecolor="#222222", capsize=3, linewidth=1)
            ax.set_xticks(x, ["1–3", "4–7", "8+"])
            ax.set_xlabel("Non-entry action position")
            ax.set_ylabel("Step success (%)")
            ax.set_title("Historical evidence diagnostic" if level == "evidence" else "Frozen policy output")
            ax.spines[["top", "right"]].set_visible(False)
        handles, legend_labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, legend_labels, loc="lower center", ncol=2, frameon=False, fontsize=9)
        fig.tight_layout(rect=(0, 0.10, 1, 1))
        for extension in ("pdf", "png"):
            fig.savefig(root / f"state_conditioning_by_position.{extension}", dpi=220, bbox_inches="tight")
        plt.close(fig)
    print(json.dumps({model: {level: values[level].get("top1_state_minus_top1_ordinal", {}).get("all_nonentry")
                             for level in ("evidence", "policy")} for model, values in result["models"].items()}, indent=2))


if __name__ == "__main__":
    main()

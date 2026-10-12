"""Validate and compare the reproduced PersonalAlign paper baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from expactivator.evaluation_stats import full_metric_bootstrap, metrics, paired_binary

MODELS = ("mai_ui_8b", "qwen35_9b", "gui_owl_1_5_8b")
METHODS = ("recent", "retrieve", "llm_um", "him_agent")
RAW_RESULTS = {
    "mai_ui_8b": "expactivator/results/backbones/mai_ui_8b_gpu0.json",
    "qwen35_9b": "expactivator/results/backbones/qwen35_9b_gpu0.json",
    "gui_owl_1_5_8b": "expactivator/results/backbones/gui_owl_1_5_8b_gpu0.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-root", default="expactivator/results/paper_baselines"
    )
    parser.add_argument(
        "--test-path", default="data/personal_execution/personal_execution.json"
    )
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument(
        "--output", default="expactivator/results/paper_baseline_analysis.json"
    )
    return parser.parse_args()


def load_result(path: Path) -> tuple[dict, list[dict]]:
    value = json.loads(path.read_text())
    return value["protocol"], value["outputs"]


def validate_order(name: str, rows: list[dict], expected: list[str]) -> None:
    ids = [row["episode_id"] for row in rows]
    if ids != expected:
        raise ValueError(f"{name} does not exactly match the official test order")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{name} contains duplicate step IDs")


def compare(
    reference: list[dict],
    method: list[dict],
    starts: list[int],
    nonfirst: list[int],
    rng: np.random.Generator,
    bootstrap_samples: int,
) -> dict:
    return {
        "full_metrics": full_metric_bootstrap(
            reference, method, rng, bootstrap_samples
        ),
        "first_step_success": paired_binary(
            np.asarray([reference[index]["sr"] for index in starts]),
            np.asarray([method[index]["sr"] for index in starts]),
            rng,
            bootstrap_samples,
        ),
        "nonfirst_success": paired_binary(
            np.asarray([reference[index]["sr"] for index in nonfirst]),
            np.asarray([method[index]["sr"] for index in nonfirst]),
            rng,
            bootstrap_samples,
        ),
    }


def main() -> None:
    args = parse_args()
    test_path = (REPO / args.test_path).resolve()
    test = json.loads(test_path.read_text())
    expected = [row["id"] for row in test]
    test_sha256 = hashlib.sha256(test_path.read_bytes()).hexdigest()
    starts = [index for index, row in enumerate(test) if row["id"].endswith("_0")]
    expected_episodes = {expected[index].rsplit("_", 1)[0] for index in starts}
    start_set = set(starts)
    nonfirst = [index for index in range(len(test)) if index not in start_set]
    result_root = (REPO / args.result_root).resolve()

    contexts = {}
    for method in METHODS:
        path = result_root / f"{method}_contexts.json"
        value = json.loads(path.read_text())
        context_protocol = value["protocol"]
        if context_protocol.get("method") != method:
            raise ValueError(f"Wrong method recorded in {path}")
        if context_protocol.get("test_sha256") != test_sha256:
            raise ValueError(f"Test hash mismatch in {path}")
        if set(value["contexts"]) != expected_episodes:
            raise ValueError(f"Contexts in {path} do not cover test episodes exactly")
        context_checks = {
            "history_split": context_protocol.get("history_split")
            == "chronological first 80% per user",
            "history_rows": context_protocol.get("history_rows") == 15_966,
            "n_episodes": context_protocol.get("n_episodes") == len(starts),
            "test_answers_not_accessed": context_protocol.get(
                "test_answers_accessed"
            )
            is False,
        }
        failed = [key for key, passed in context_checks.items() if not passed]
        if failed:
            raise ValueError(f"{path} failed context checks: {failed}")
        if method == "llm_um":
            profile_path = Path(context_protocol["llm_um_file"])
            profile_sha256 = hashlib.sha256(profile_path.read_bytes()).hexdigest()
            if profile_sha256 != context_protocol.get("llm_um_sha256"):
                raise ValueError("LLM-UM profile hash does not match its context")
            users = json.loads(profile_path.read_text())["users"]
            if len(users) != 83 or len({row["user_id"] for row in users}) != 83:
                raise ValueError("LLM-UM profiles do not cover 83 unique users")
            if any(not row.get("summaries") for row in users):
                raise ValueError("LLM-UM contains an empty user profile")
        contexts[method] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "protocol": context_protocol,
            "checks": context_checks,
        }

    rows_by_model: dict[str, dict[str, list[dict]]] = {}
    protocols = {}
    integrity = {}
    parse_quality = {}
    all_metrics = {}
    comparisons = {}
    rng = np.random.default_rng(args.seed)
    for model in MODELS:
        raw_protocol, raw_rows = load_result(REPO / RAW_RESULTS[model])
        validate_order(f"{model}/base", raw_rows, expected)
        rows_by_model[model] = {"base": raw_rows}
        protocols[f"{model}/base"] = raw_protocol
        all_metrics[model] = {"base": metrics(raw_rows)}
        comparisons[model] = {}

        for method in METHODS:
            name = f"{model}/{method}"
            path = result_root / f"{model}_{method}_gpu0.json"
            protocol, rows = load_result(path)
            validate_order(name, rows, expected)
            checks = {
                "test_sha256_matches": protocol.get("test_sha256") == test_sha256,
                "physical_gpu_zero": protocol.get("physical_gpu") == 0,
                "only_gpu_zero_visible": protocol.get("visible_cuda_devices") == "0",
                "teacher_forced_history": protocol.get("teacher_forced_history")
                is True,
                "ambiguous_instruction": protocol.get("ambiguous_instruction") is True,
                "baseline_name_matches": protocol.get("baseline_name") == method,
                "context_sha256_matches": protocol.get("memory_contexts_sha256")
                == contexts[method]["sha256"],
                "model_revision_matches_base": protocol.get("model_revision")
                == raw_protocol.get("model_revision"),
                "max_model_len_8192": protocol.get("max_model_len") == 8192,
            }
            failed = [key for key, value in checks.items() if not value]
            if failed:
                raise ValueError(f"{name} failed integrity checks: {failed}")
            rows_by_model[model][method] = rows
            protocols[name] = protocol
            integrity[name] = {"n_steps": len(rows), **checks}
            counts = Counter(row.get("parse_method", "unrecorded") for row in rows)
            parse_quality[name] = {
                "counts": dict(sorted(counts.items())),
                "parse_error_rate": counts["parse_error"] / len(rows),
            }
            all_metrics[model][method] = metrics(rows)
            comparisons[model][f"{method}_vs_base"] = compare(
                raw_rows, rows, starts, nonfirst, rng, args.bootstrap_samples
            )

        for reference_method, method in combinations(METHODS, 2):
            comparisons[model][f"{method}_vs_{reference_method}"] = compare(
                rows_by_model[model][reference_method],
                rows_by_model[model][method],
                starts,
                nonfirst,
                rng,
                args.bootstrap_samples,
            )

    rankings = {}
    metric_names = (
        "type_accuracy",
        "success_rate",
        "csr",
        "first_step_success_rate",
        "nonfirst_success_rate",
    )
    for model in MODELS:
        rankings[model] = {
            metric: sorted(
                (
                    {"method": method, "value": all_metrics[model][method][metric]}
                    for method in METHODS
                ),
                key=lambda item: item["value"],
                reverse=True,
            )
            for metric in metric_names
        }

    result = {
        "protocol": {
            "name": "PersonalAlign paper-baseline reconstruction",
            "n_steps": len(test),
            "n_episodes": len(starts),
            "models": list(MODELS),
            "methods": list(METHODS),
            "bootstrap_samples": args.bootstrap_samples,
            "seed": args.seed,
        },
        "metrics": all_metrics,
        "rankings": rankings,
        "integrity": integrity,
        "parse_quality": parse_quality,
        "comparisons": comparisons,
        "contexts": contexts,
        "source_protocols": protocols,
    }
    destination = (REPO / args.output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(all_metrics, ensure_ascii=False, indent=2))
    print(f"saved {destination}")


if __name__ == "__main__":
    main()

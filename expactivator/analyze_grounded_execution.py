"""Audit and compare ExpActivator grounded-policy execution runs.

The analysis keeps three quantities separate for every backbone: the raw
policy reference, the score obtained by copying the retrieved evidence, and
the score produced by the frozen policy after conditioning on that evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

os.environ.setdefault("USE_TF", "0")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.eval_execution.evaluation_new_backbones import resolve_image
from src.eval_execution.utils import convert_action_qwen
from expactivator.evaluation_stats import (
    action_breakdown,
    full_metric_bootstrap,
    metrics,
    paired_binary,
)


RUNS = {
    "qwen3_vl_8b": "expactivator/results/grounded/qwen3_vl_8b_grounded_policy_gpu0.json",
    "mai_ui_8b": "expactivator/results/grounded/mai_ui_8b_grounded_policy_gpu0.json",
    "qwen35_9b": "expactivator/results/grounded/qwen35_9b_grounded_policy_gpu0.json",
    "gui_owl_1_5_8b": "expactivator/results/grounded/gui_owl_1_5_8b_grounded_policy_gpu0.json",
}

REFERENCES = {
    "qwen3_vl_8b": "expactivator/results/paper_baselines/qwen3_vl_8b_base_gpu0.json",
    "mai_ui_8b": "expactivator/results/backbones/mai_ui_8b_gpu0.json",
    "qwen35_9b": "expactivator/results/backbones/qwen35_9b_gpu0.json",
    "gui_owl_1_5_8b": "expactivator/results/backbones/gui_owl_1_5_8b_gpu0.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-path", default="data/personal_execution/personal_execution.json"
    )
    parser.add_argument("--history-root", default="data/Fingertip")
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument(
        "--output", default="expactivator/results/grounded/analysis.json"
    )
    parser.add_argument(
        "--report", default="expactivator/results/grounded/REPORT.md"
    )
    for model, default in RUNS.items():
        option = model.replace("_", "-")
        parser.add_argument(f"--{option}-run", default=default)
        parser.add_argument(f"--{option}-reference", default=REFERENCES[model])
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_result(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    value = json.loads(path.read_text())
    return value.get("protocol", {}), value["outputs"]


def validate_order(name: str, rows: list[dict[str, Any]], expected: list[str]) -> None:
    ids = [row["episode_id"] for row in rows]
    if ids != expected or len(set(ids)) != len(ids):
        raise ValueError(f"{name} does not exactly match official test order")


def compare(
    reference: list[dict[str, Any]],
    method: list[dict[str, Any]],
    starts: list[int],
    nonfirst: list[int],
    rng: np.random.Generator,
    bootstrap_samples: int,
) -> dict[str, Any]:
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


def evidence_rows(
    test: list[dict[str, Any]],
    contexts: dict[str, dict[str, Any]],
    history_root: Path,
) -> list[dict[str, Any]]:
    rows = []
    for step in test:
        candidate = contexts[step["id"]]["retrieved_action"]
        image_path = resolve_image(step["img_path"], history_root)
        with Image.open(image_path) as image:
            width, height = image.size
        type_accuracy, success = convert_action_qwen(
            candidate, step["answer"], width, height, coord_map=True
        )
        rows.append(
            {
                "episode_id": step["id"],
                "answer": step["answer"],
                "output": candidate,
                "type_acc": type_accuracy,
                "sr": success,
            }
        )
    return rows


def policy_evidence_interaction(
    policy: list[dict[str, Any]], evidence: list[dict[str, Any]]
) -> dict[str, Any]:
    exact = 0
    action_type = 0
    coordinate_cases = 0
    changed_coordinates = 0
    for policy_row, evidence_row in zip(policy, evidence):
        policy_action = policy_row["output"]
        candidate = evidence_row["output"]
        exact += policy_action == candidate
        action_type += policy_action.get("action") == candidate.get("action")
        if "coordinate" in candidate:
            coordinate_cases += 1
            changed_coordinates += policy_action.get("coordinate") != candidate.get(
                "coordinate"
            )
    return {
        "exact_candidate_copy_rate": exact / len(policy),
        "action_type_agreement_rate": action_type / len(policy),
        "action_type_override_rate": 1 - action_type / len(policy),
        "coordinate_evidence_steps": coordinate_cases,
        "changed_coordinate_rate": (
            changed_coordinates / coordinate_cases if coordinate_cases else 0.0
        ),
    }


def pct(value: float) -> str:
    return f"{100 * value:.3f}"


def write_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# ExpActivator grounded-policy execution",
        "",
        "Retrieved actions are scored only as a direct-copy diagnostic. The ExpActivator",
        "result is the frozen policy output conditioned on that evidence.",
        "",
        "| Backbone / output | Type | SSR | CSR | First-step SR | Non-first SR |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model, values in result["models"].items():
        for label in ("reference", "retrieved_evidence_direct", "grounded_policy"):
            metric = values["metrics"][label]
            lines.append(
                f"| {model} / {label} | {pct(metric['type_accuracy'])} | "
                f"{pct(metric['success_rate'])} | {pct(metric['csr'])} | "
                f"{pct(metric['first_step_success_rate'])} | "
                f"{pct(metric['nonfirst_success_rate'])} |"
            )
    lines.extend(
        [
            "",
            "All runs cover 775 episodes and 7,915 teacher-forced steps. Integrity",
            "checks require a step-level evidence context for every test row, an",
            "exact context hash, no direct action emission by the retriever, and a",
            "frozen policy output for the final decision.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    test_path = (REPO / args.test_path).resolve()
    history_root = Path(args.history_root).resolve()
    test = json.loads(test_path.read_text())
    expected = [row["id"] for row in test]
    test_hash = sha256(test_path)
    starts = [index for index, row in enumerate(test) if row["id"].endswith("_0")]
    start_set = set(starts)
    nonfirst = [index for index in range(len(test)) if index not in start_set]
    rng = np.random.default_rng(args.seed)
    models: dict[str, Any] = {}

    for model in RUNS:
        option = model.replace("_", "-")
        run_path = (REPO / getattr(args, f"{model}_run")).resolve()
        reference_path = (REPO / getattr(args, f"{model}_reference")).resolve()
        protocol, policy = load_result(run_path)
        reference_protocol, reference = load_result(reference_path)
        validate_order(f"{model}/grounded_policy", policy, expected)
        validate_order(f"{model}/reference", reference, expected)

        if protocol.get("baseline_name") != "expactivator_grounded":
            raise ValueError(f"{model} is not a expactivator_grounded run")
        if protocol.get("memory_context_scope") != "step":
            raise ValueError(f"{model} did not use step-level evidence")
        if protocol.get("teacher_forced_history") is not True:
            raise ValueError(f"{model} did not use teacher-forced progress")
        if protocol.get("test_sha256") != test_hash:
            raise ValueError(f"{model} test hash mismatch")
        context_path = Path(protocol["memory_contexts"])
        if sha256(context_path) != protocol.get("memory_contexts_sha256"):
            raise ValueError(f"{model} evidence hash mismatch")
        context_file = json.loads(context_path.read_text())
        context_protocol = context_file["protocol"]
        contexts = context_file["contexts"]
        if set(contexts) != set(expected):
            raise ValueError(f"{model} evidence does not exactly cover the test")
        if context_protocol.get("direct_action_emission") is not False:
            raise ValueError(f"{model} retriever is recorded as final policy")
        if context_protocol.get("policy_may_override_evidence") is not True:
            raise ValueError(f"{model} policy override was disabled")
        if context_protocol.get("test_answers_used_for_retrieval") is not False:
            raise ValueError(f"{model} retrieval accessed test answers")
        mismatched_rows = sum(
            row.get("grounded_evidence") != contexts[row["episode_id"]]
            for row in policy
        )
        if mismatched_rows:
            raise ValueError(f"{model} has {mismatched_rows} evidence-row mismatches")

        direct = evidence_rows(test, contexts, history_root)
        reference_revision = reference_protocol.get("model_revision")
        policy_revision = protocol.get("model_revision")
        revision_comparable = (
            reference_revision is not None and reference_revision == policy_revision
        )
        source_user_mismatches = 0
        exact_episode_violations = 0
        for step in test:
            if step["id"].endswith("_0"):
                continue
            source = context_file["sources"][step["id"]]
            source_user_mismatches += int(source["user_id"] != int(step["user_id"]))
            exact_episode_violations += int(
                f"{source['user_id']}_{source['time']}"
                == step["id"].rsplit("_", 1)[0]
            )
        if source_user_mismatches or exact_episode_violations:
            raise ValueError(f"{model} failed historical-source isolation")

        models[model] = {
            "metrics": {
                "reference": metrics(reference),
                "retrieved_evidence_direct": metrics(direct),
                "grounded_policy": metrics(policy),
            },
            "grounded_policy_vs_direct_evidence": compare(
                direct,
                policy,
                starts,
                nonfirst,
                rng,
                args.bootstrap_samples,
            ),
            "grounded_policy_vs_reference": compare(
                reference,
                policy,
                starts,
                nonfirst,
                rng,
                args.bootstrap_samples,
            ),
            "action_type_breakdown_vs_direct_evidence": action_breakdown(
                test, direct, policy
            ),
            "policy_evidence_interaction": policy_evidence_interaction(policy, direct),
            "parse_methods": dict(
                sorted(
                    Counter(
                        row.get("parse_method", "unrecorded") for row in policy
                    ).items()
                )
            ),
            "integrity": {
                "n_steps": len(policy),
                "n_episodes": len(starts),
                "test_sha256_matches": True,
                "evidence_sha256_matches": True,
                "step_evidence_matches_output_rows": True,
                "direct_action_emission": False,
                "policy_may_override_evidence": True,
                "test_answers_used_for_retrieval": False,
                "source_user_mismatches": source_user_mismatches,
                "source_exact_episode_violations": exact_episode_violations,
                "reference_model_revision_comparable": revision_comparable,
            },
            "paths": {
                "run": str(run_path),
                "reference": str(reference_path),
                "evidence": str(context_path),
            },
            "protocol": protocol,
            "reference_protocol": reference_protocol,
            "evidence_protocol": context_protocol,
        }

    result = {
        "protocol": {
            "name": "ExpActivator grounded historical evidence plus frozen policy",
            "test_sha256": test_hash,
            "n_steps": len(test),
            "n_episodes": len(starts),
            "bootstrap_samples": args.bootstrap_samples,
            "seed": args.seed,
        },
        "models": models,
    }
    output = (REPO / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    report = (REPO / args.report).resolve()
    write_report(report, result)
    print(
        json.dumps(
            {name: value["metrics"] for name, value in models.items()}, indent=2
        )
    )
    print(f"saved {output}")
    print(f"saved {report}")


if __name__ == "__main__":
    main()

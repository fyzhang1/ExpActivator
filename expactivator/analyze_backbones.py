"""Compare complete GUI backbones with full backbone-specific ExpActivator runs."""

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

from src.eval_execution.utils import convert_action_qwen
from expactivator.evaluation_stats import (
    action_breakdown,
    full_metric_bootstrap,
    metrics,
    paired_binary,
)

DEFAULT_MODELS = {
    "mai_ui_8b": "expactivator/results/backbones/mai_ui_8b_gpu0.json",
    "qwen35_9b": "expactivator/results/backbones/qwen35_9b_gpu0.json",
    "gui_owl_1_5_8b": ("expactivator/results/backbones/gui_owl_1_5_8b_gpu0.json"),
}

DEFAULT_ExpActivatorS = {
    name: path.replace("_gpu0.json", "_expactivator_gpu0.json")
    for name, path in DEFAULT_MODELS.items()
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline", default="expactivator/results/execution_baseline_gpu0.json"
    )
    parser.add_argument(
        "--expactivator", default="expactivator/results/execution_predictions_gpu0.json"
    )
    parser.add_argument(
        "--app-index",
        default="expactivator/results/execution_app_predictions_gpu0.json",
    )
    parser.add_argument(
        "--test-path", default="data/personal_execution/personal_execution.json"
    )
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument(
        "--output", default="expactivator/results/backbone_analysis.json"
    )
    for name, default in DEFAULT_MODELS.items():
        parser.add_argument(f"--{name.replace('_', '-')}", default=default)
        parser.add_argument(
            f"--{name.replace('_', '-')}-expactivator", default=DEFAULT_ExpActivatorS[name]
        )
    return parser.parse_args()


def load_result(path: Path) -> tuple[dict, list[dict]]:
    value = json.loads(path.read_text())
    if isinstance(value, list):
        return {}, value
    return value.get("protocol", {}), value["outputs"]


def validate_order(name: str, rows: list[dict], expected: list[str]) -> None:
    ids = [row["episode_id"] for row in rows]
    if ids != expected:
        raise ValueError(f"{name} does not exactly match official test order")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{name} contains duplicate step IDs")


def add_expactivator_app(
    test: list[dict], rows: list[dict], app_names: dict[str, str]
) -> list[dict]:
    """Replace only episode-start actions; retain each model's later outputs."""
    result = []
    for step, source in zip(test, rows):
        row = dict(source)
        if step["id"].endswith("_0"):
            episode = step["id"].rsplit("_", 1)[0]
            output = {"action": "open_app", "app_name": app_names[episode]}
            type_acc, success = convert_action_qwen(
                output, step["answer"], 1, 1, coord_map=True
            )
            row.update(
                {
                    "output": output,
                    "type_acc": type_acc,
                    "sr": success,
                    "result": f"Type: {type_acc}, SR: {success}",
                }
            )
        result.append(row)
    return result


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
    test_path = REPO / args.test_path
    test = json.loads(test_path.read_text())
    test_sha256 = hashlib.sha256(test_path.read_bytes()).hexdigest()
    expected = [row["id"] for row in test]
    starts = [index for index, row in enumerate(test) if row["id"].endswith("_0")]
    start_set = set(starts)
    nonfirst = [index for index in range(len(test)) if index not in start_set]

    baseline_protocol, baseline = load_result(REPO / args.baseline)
    expactivator_protocol, expactivator = load_result(REPO / args.expactivator)
    validate_order("qwen3_vl_baseline", baseline, expected)
    validate_order("expactivator", expactivator, expected)

    app_index = json.loads((REPO / args.app_index).read_text())
    app_names = dict(zip(app_index["episode_ids"], app_index["factorized_app_names"]))
    if set(app_names) != {row["id"].rsplit("_", 1)[0] for row in test}:
        raise ValueError("ExpActivator app index does not cover the official episodes exactly")

    model_paths = {name: Path(getattr(args, name)) for name in DEFAULT_MODELS}
    raw_models: dict[str, list[dict]] = {}
    augmented_models: dict[str, list[dict]] = {}
    expactivator_models: dict[str, list[dict]] = {}
    protocols = {
        "qwen3_vl_baseline": baseline_protocol,
        "expactivator": expactivator_protocol,
    }
    parse_quality = {}
    integrity = {}
    for name, relative_path in model_paths.items():
        protocol, rows = load_result(REPO / relative_path)
        validate_order(name, rows, expected)
        if protocol.get("physical_gpu") != 0:
            raise ValueError(f"{name} was not recorded as a physical-GPU-0 run")
        if protocol.get("visible_cuda_devices") != "0":
            raise ValueError(f"{name} exposed a device other than physical GPU 0")
        if protocol.get("test_sha256") != test_sha256:
            raise ValueError(f"{name} used a different execution-test file")
        if protocol.get("teacher_forced_history") is not True:
            raise ValueError(f"{name} did not use teacher-forced history")
        if protocol.get("ambiguous_instruction") is not True:
            raise ValueError(f"{name} did not use ambiguous instructions")
        raw_models[name] = rows
        augmented_models[name] = add_expactivator_app(test, rows, app_names)
        protocols[name] = protocol
        counts = Counter(row.get("parse_method", "unrecorded") for row in rows)
        parse_quality[name] = {
            "counts": dict(sorted(counts.items())),
            "parse_error_rate": counts["parse_error"] / len(rows),
        }
        integrity[name] = {
            "n_steps": len(rows),
            "unique_step_ids": len({row["episode_id"] for row in rows}),
            "official_order": True,
            "test_sha256_matches": True,
            "physical_gpu": protocol["physical_gpu"],
            "visible_cuda_devices": protocol["visible_cuda_devices"],
        }
        expactivator_protocol, expactivator_rows = load_result(
            REPO / Path(getattr(args, f"{name}_expactivator"))
        )
        validate_order(f"{name}_expactivator", expactivator_rows, expected)
        if expactivator_protocol.get("test_answers_used_for_prediction") is not False:
            raise ValueError(f"{name} ExpActivator run used target answers for prediction")
        if expactivator_protocol.get("test_sha256") != test_sha256:
            raise ValueError(f"{name} ExpActivator run used a different execution test")
        if expactivator_protocol.get("exact_exclusion_violations") != 0:
            raise ValueError(f"{name} ExpActivator run violated exact-episode exclusion")
        if expactivator_protocol.get("physical_gpu_for_feature_extraction") != 0:
            raise ValueError(f"{name} ExpActivator features were not built on GPU 0")
        if expactivator_protocol.get("visible_cuda_devices_for_feature_extraction") != "0":
            raise ValueError(f"{name} ExpActivator feature extraction exposed another GPU")
        if expactivator_protocol.get("visual_model_revision") != protocol.get(
            "model_revision"
        ):
            raise ValueError(f"{name} raw and ExpActivator model revisions differ")
        expactivator_models[name] = expactivator_rows
        protocols[f"{name}_expactivator"] = expactivator_protocol
        integrity[f"{name}_expactivator"] = {
            "n_steps": len(expactivator_rows),
            "unique_step_ids": len({row["episode_id"] for row in expactivator_rows}),
            "official_order": True,
            "test_answers_used_for_prediction": False,
            "exact_exclusion_violations": 0,
            "physical_gpu_for_feature_extraction": 0,
            "visible_cuda_devices_for_feature_extraction": "0",
        }

    all_metrics = {
        "qwen3_vl_baseline": metrics(baseline),
        "expactivator": metrics(expactivator),
    }
    for name, rows in raw_models.items():
        all_metrics[name] = metrics(rows)
        all_metrics[f"{name}_plus_expactivator_app"] = metrics(augmented_models[name])
        all_metrics[f"{name}_plus_expactivator"] = metrics(expactivator_models[name])

    rng = np.random.default_rng(args.seed)
    comparisons = {}
    for name, rows in raw_models.items():
        augmented = augmented_models[name]
        comparisons[name] = {
            "raw_vs_qwen3_vl_baseline": compare(
                baseline,
                rows,
                starts,
                nonfirst,
                rng,
                args.bootstrap_samples,
            ),
            "expactivator_app_frontend_vs_raw": compare(
                rows,
                augmented,
                starts,
                nonfirst,
                rng,
                args.bootstrap_samples,
            ),
            "full_expactivator_vs_raw": compare(
                rows,
                expactivator_models[name],
                starts,
                nonfirst,
                rng,
                args.bootstrap_samples,
            ),
            "full_expactivator_vs_expactivator_app_only": compare(
                augmented,
                expactivator_models[name],
                starts,
                nonfirst,
                rng,
                args.bootstrap_samples,
            ),
            "expactivator_vs_raw": compare(
                rows,
                expactivator,
                starts,
                nonfirst,
                rng,
                args.bootstrap_samples,
            ),
            "expactivator_vs_model_plus_expactivator_app": compare(
                augmented,
                expactivator,
                starts,
                nonfirst,
                rng,
                args.bootstrap_samples,
            ),
            "fixed_qwen_expactivator_vs_backbone_expactivator": compare(
                expactivator_models[name],
                expactivator,
                starts,
                nonfirst,
                rng,
                args.bootstrap_samples,
            ),
            "expactivator_vs_raw_action_breakdown": action_breakdown(test, rows, expactivator),
            "backbone_expactivator_vs_raw_action_breakdown": action_breakdown(
                test, rows, expactivator_models[name]
            ),
        }

    expactivator_pairwise = {}
    for reference_name, method_name in combinations(expactivator_models, 2):
        expactivator_pairwise[f"{method_name}_vs_{reference_name}"] = compare(
            expactivator_models[reference_name],
            expactivator_models[method_name],
            starts,
            nonfirst,
            rng,
            args.bootstrap_samples,
        )

    metric_names = (
        "type_accuracy",
        "success_rate",
        "csr",
        "first_step_success_rate",
        "nonfirst_success_rate",
    )
    rankings = {
        metric: sorted(
            (
                {"name": name, "value": values[metric]}
                for name, values in all_metrics.items()
            ),
            key=lambda item: item["value"],
            reverse=True,
        )
        for metric in metric_names
    }
    result = {
        "protocol": {
            "name": "complete frozen-backbone comparison with ExpActivator controls",
            "n_steps": len(test),
            "n_episodes": len(starts),
            "teacher_forced_history": True,
            "ambiguous_instruction": True,
            "expactivator_app_control": (
                "replace only each model's first action with the fixed, "
                "history-derived ExpActivator app prediction"
            ),
            "full_backbone_expactivator": (
                "ExpActivator omission-aware app selection, goal-restricted trajectory "
                "memory, and state synchronization using each backbone's own "
                "frozen visual representation selected on histories only"
            ),
            "bootstrap_samples": args.bootstrap_samples,
            "seed": args.seed,
        },
        "metrics": all_metrics,
        "rankings": rankings,
        "integrity": integrity,
        "parse_quality": parse_quality,
        "comparisons": comparisons,
        "full_expactivator_pairwise": expactivator_pairwise,
        "source_protocols": protocols,
    }
    destination = REPO / args.output
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))
    print(f"saved {destination}")


if __name__ == "__main__":
    main()

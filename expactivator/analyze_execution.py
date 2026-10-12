"""Complete paired analysis and leakage audit for ExpActivator execution."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from expactivator.evaluation_stats import (
    action_breakdown,
    full_metric_bootstrap,
    metrics,
    paired_binary,
)
from expactivator.execution_core import (
    audit_memory_split,
    execution_memory_frame,
    history_key,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method", default="expactivator/results/execution_predictions_gpu0.json"
    )
    parser.add_argument(
        "--rgb", default="expactivator/results/execution_rgb_ablation_gpu0.json"
    )
    parser.add_argument(
        "--baseline", default="expactivator/results/execution_baseline_gpu0.json"
    )
    parser.add_argument(
        "--test-path", default="data/personal_execution/personal_execution.json"
    )
    parser.add_argument(
        "--legacy-train-path",
        default="samples/train_set.csv",
        help=(
            "Released non-chronological CSV, retained only to report how many "
            "test episodes the superseded memory would have exposed"
        ),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument(
        "--output", default="expactivator/results/execution_analysis.json"
    )
    return parser.parse_args()


def load_outputs(path: Path) -> list[dict]:
    value = json.loads(path.read_text())
    return value["outputs"] if isinstance(value, dict) else value


def subset(rows: list[dict], positions: list[int]) -> list[dict]:
    return [rows[index] for index in positions]


def main() -> None:
    args = parse_args()
    method_file = json.loads((REPO / args.method).read_text())
    method = method_file["outputs"]
    rgb_file = json.loads((REPO / args.rgb).read_text())
    rgb = rgb_file["outputs"]
    full_goal_app_only = rgb_file["ablation_outputs"]["full_goal_app_only"]
    factorized_app_only = rgb_file["ablation_outputs"]["factorized_app_only"]
    same_position = rgb_file["ablation_outputs"]["same_position"]
    baseline = load_outputs(REPO / args.baseline)
    test = json.loads((REPO / args.test_path).read_text())
    expected = [row["id"] for row in test]
    for name, rows in (
        ("baseline", baseline),
        ("full_goal_app_only", full_goal_app_only),
        ("factorized_app_only", factorized_app_only),
        ("same_position", same_position),
        ("rgb", rgb),
        ("method", method),
    ):
        if [row["episode_id"] for row in rows] != expected:
            raise ValueError(f"{name} does not exactly match official test order")

    rng = np.random.default_rng(args.seed)
    starts = [index for index, row in enumerate(test) if row["id"].endswith("_0")]
    nonfirst = [index for index, row in enumerate(test) if not row["id"].endswith("_0")]
    result = {
        "metrics": {
            "baseline": metrics(baseline),
            "full_goal_app_only": metrics(full_goal_app_only),
            "factorized_app_only": metrics(factorized_app_only),
            "same_position": metrics(same_position),
            "rgb_expactivator": metrics(rgb),
            "expactivator": metrics(method),
        },
        "expactivator_vs_rgb": {
            "full_metrics": full_metric_bootstrap(
                rgb, method, rng, args.bootstrap_samples
            ),
            "nonfirst_success": paired_binary(
                np.asarray([rgb[index]["sr"] for index in nonfirst]),
                np.asarray([method[index]["sr"] for index in nonfirst]),
                rng,
                args.bootstrap_samples,
            ),
            "action_type_breakdown": action_breakdown(test, rgb, method),
        },
        "innovation_1_goal_factorization": {
            "full_metrics": full_metric_bootstrap(
                full_goal_app_only,
                factorized_app_only,
                rng,
                args.bootstrap_samples,
            ),
            "first_step_success": paired_binary(
                np.asarray([full_goal_app_only[index]["sr"] for index in starts]),
                np.asarray([factorized_app_only[index]["sr"] for index in starts]),
                rng,
                args.bootstrap_samples,
            ),
        },
        "innovation_2_state_synchronization": {
            "app_only_to_expactivator_full_metrics": full_metric_bootstrap(
                factorized_app_only, method, rng, args.bootstrap_samples
            ),
            "same_position_to_expactivator_nonfirst": paired_binary(
                np.asarray([same_position[index]["sr"] for index in nonfirst]),
                np.asarray([method[index]["sr"] for index in nonfirst]),
                rng,
                args.bootstrap_samples,
            ),
        },
        "expactivator_vs_baseline": {
            "full_metrics": full_metric_bootstrap(
                baseline, method, rng, args.bootstrap_samples
            ),
            "nonfirst_success": paired_binary(
                np.asarray([baseline[index]["sr"] for index in nonfirst]),
                np.asarray([method[index]["sr"] for index in nonfirst]),
                rng,
                args.bootstrap_samples,
            ),
            "action_type_breakdown": action_breakdown(test, baseline, method),
        },
    }

    # The active memory is the chronological first-80% split, which contains
    # no test episode, so "overlap" is empty by construction.  The split is
    # kept as a regression check and reported against the superseded
    # train_set.csv memory, whose overlap is the leak this audit must catch.
    memory = execution_memory_frame(REPO)
    result["history_audit"] = audit_memory_split(memory, REPO)
    legacy = pd.read_csv(REPO / args.legacy_train_path)
    legacy_ids = {history_key(row.user_id, row.time) for row in legacy.itertuples()}
    memory_ids = {history_key(row.user_id, row.time) for row in memory.itertuples()}
    episode_ids = {row["id"].rsplit("_", 1)[0] for row in test}
    result["history_audit"]["superseded_train_set_overlap"] = len(
        legacy_ids & episode_ids
    )
    for split, require_overlap in (("overlap", True), ("nonoverlap", False)):
        positions = [
            index
            for index, row in enumerate(test)
            if (row["id"].rsplit("_", 1)[0] in memory_ids) == require_overlap
        ]
        if not positions:
            result[split] = {"n_steps": 0, "metrics": None, "note": "empty split"}
            continue
        rgb_split = subset(rgb, positions)
        method_split = subset(method, positions)
        baseline_split = subset(baseline, positions)
        result[split] = {
            "n_steps": len(positions),
            "metrics": {
                "baseline": metrics(baseline_split),
                "rgb_expactivator": metrics(rgb_split),
                "expactivator": metrics(method_split),
            },
            "expactivator_vs_rgb": full_metric_bootstrap(
                rgb_split, method_split, rng, args.bootstrap_samples
            ),
            "expactivator_vs_baseline": full_metric_bootstrap(
                baseline_split, method_split, rng, args.bootstrap_samples
            ),
        }

    method_metrics = result["metrics"]["expactivator"]
    rgb_metrics = result["metrics"]["rgb_expactivator"]
    result["adoption"] = {
        "predeclared_rule": "adopt only if full success rate and CSR both exceed RGB ExpActivator",
        "success_rate_improved": method_metrics["success_rate"]
        > rgb_metrics["success_rate"],
        "csr_improved": method_metrics["csr"] > rgb_metrics["csr"],
    }
    result["adoption"]["adopt"] = bool(
        result["adoption"]["success_rate_improved"]
        and result["adoption"]["csr_improved"]
    )
    result["audit"] = {
        "n_steps": len(test),
        "n_episode_starts": len(starts),
        "n_nonfirst_steps": len(nonfirst),
        "unique_ids": len(set(expected)),
        "first_step_output_differences_vs_rgb": sum(
            method[index]["output"] != rgb[index]["output"] for index in starts
        ),
        "method_protocol": method_file["protocol"],
        "memory_source_user_mismatches": sum(
            int((method[index].get("memory_source") or {}).get("user_id", -1))
            != int(test[index]["user_id"])
            for index in nonfirst
        ),
        "memory_source_exact_episode_violations": sum(
            (
                f"{(method[index].get('memory_source') or {}).get('user_id')}_"
                f"{(method[index].get('memory_source') or {}).get('time')}"
            )
            == test[index]["id"].rsplit("_", 1)[0]
            for index in nonfirst
        ),
        "unique_historical_source_states": len(
            {
                (method[index].get("memory_source") or {}).get("image_path")
                for index in nonfirst
            }
        ),
    }
    destination = REPO / args.output
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

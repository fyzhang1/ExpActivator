"""Run TRACE on the complete AndroidIntent proactive-suggestion benchmark.

The predictor receives only ``user_id``, ``time``, and ``scenario``.  It first
finds repeated semantic routines in the official chronological memory, then
binds the current state to a routine with a circular-time kernel.  Intent
frequency is used only as a stability requirement; routine selection uses a
frequency-normalized state likelihood so a very common but contextually wrong
routine cannot dominate.  A historical medoid is emitted, with no trained
decoder and no target instruction access.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from trace_method.shared_memory import task_frame, task_keys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=str(REPO))
    parser.add_argument(
        "--index", default="trace_method/results/proactive_history_index_gpu0.pt"
    )
    parser.add_argument(
        "--shared-memory",
        default=None,
        help="Read the proactive full-intent/time/scenario view from this artifact",
    )
    parser.add_argument(
        "--positive", default="data/proactive_suggestion/proactive_samples.json"
    )
    parser.add_argument(
        "--negative",
        default="data/proactive_suggestion/proactive_false_samples.json",
    )
    parser.add_argument(
        "--output", default="trace_method/results/proactive_predictions.json"
    )
    parser.add_argument(
        "--positive-csv", default="trace_method/results/proactive_positive.csv"
    )
    parser.add_argument(
        "--negative-csv", default="trace_method/results/proactive_negative.csv"
    )
    # Fixed development configuration.  No model parameters are fitted.
    parser.add_argument("--semantic-threshold", type=float, default=0.94)
    parser.add_argument("--minimum-support", type=int, default=12)
    parser.add_argument("--minimum-scenario-support", type=int, default=4)
    parser.add_argument("--time-sigma-hours", type=float, default=1.0)
    parser.add_argument("--trigger-mass", type=float, default=0.25)
    parser.add_argument("--selection-power", type=float, default=1.0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hour(time_value: str) -> float:
    clock = str(time_value).split("_")[1]
    return int(clock[:2]) + int(clock[2:4]) / 60.0 + int(clock[4:6]) / 3600.0


def circular_hour_distance(values: np.ndarray, query: float) -> np.ndarray:
    difference = np.abs(values - query)
    return np.minimum(difference, 24.0 - difference)


def predict_state(
    user_id: int,
    record_time: str,
    scenario: str,
    memory: pd.DataFrame,
    keys: np.ndarray,
    user_indices: dict[int, np.ndarray],
    config: dict[str, float | int],
) -> dict[str, Any]:
    indices = user_indices.get(int(user_id))
    if indices is None or len(indices) == 0:
        return {"predicted_intent": "False", "reason": "no_user_memory"}

    local = memory.iloc[indices]
    # Some official negative queries fall inside the first-80% memory split.
    # Excluding the exact state prevents direct answer replay.
    valid = local.time.astype(str).to_numpy() != str(record_time)
    local_keys = keys[indices]
    similarity = local_keys @ local_keys.T
    neighbors = (similarity >= float(config["semantic_threshold"])) & valid[None, :]
    support = neighbors.sum(1)

    same_scenario = local.scenario.astype(str).to_numpy() == str(scenario)
    state_neighbors = neighbors & same_scenario[None, :]
    scenario_support = state_neighbors.sum(1)
    hours = np.asarray([hour(value) for value in local.time], dtype=np.float32)
    distance = circular_hour_distance(hours, hour(record_time))
    sigma = float(config["time_sigma_hours"])
    weights = np.exp(-(distance**2) / (2.0 * sigma**2))
    trigger_mass = (state_neighbors * weights[None, :]).sum(1)

    eligible = (
        (support >= int(config["minimum_support"]))
        & (scenario_support >= int(config["minimum_scenario_support"]))
        & (trigger_mass >= float(config["trigger_mass"]))
    )
    if not bool(eligible.any()):
        return {
            "predicted_intent": "False",
            "reason": "no_stable_state_match",
            "excluded_exact_states": int((~valid).sum()),
        }

    # Estimate p(state | routine), rather than raw occurrence mass.  This is
    # the proactive analogue of TRACE's phase synchronization: it prevents a
    # globally frequent routine from drowning out a less frequent routine
    # whose temporal and physical state is a better match.
    state_likelihood = trigger_mass / np.maximum(support, 1) ** float(
        config["selection_power"]
    )
    state_likelihood[~eligible] = -1.0
    center = int(np.argmax(state_likelihood))
    members = np.flatnonzero(state_neighbors[center])
    all_members = np.flatnonzero(neighbors[center])

    # Emit the most central same-scenario history, not a target-conditioned
    # nearest neighbor.  Lexical/source-index tie breaking is deterministic.
    centrality = similarity[np.ix_(members, all_members)].mean(1)
    best_centrality = float(centrality.max())
    tied = members[np.isclose(centrality, best_centrality, rtol=0.0, atol=1e-7)]
    source_local = min(
        tied.tolist(), key=lambda index: int(local.iloc[index].source_index)
    )
    source = local.iloc[source_local]
    return {
        "predicted_intent": str(source.intentDescription),
        "reason": "stable_state_synchronized_routine",
        "source_index": int(source.source_index),
        "source_user_id": int(source.user_id),
        "source_time": str(source.time),
        "source_scenario": str(source.scenario),
        "routine_center_source_index": int(local.iloc[center].source_index),
        "routine_support": int(support[center]),
        "scenario_support": int(scenario_support[center]),
        "trigger_mass": float(trigger_mass[center]),
        "state_likelihood": float(state_likelihood[center]),
        "excluded_exact_states": int((~valid).sum()),
    }


def write_official_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["original_intent", "predicted_intent", "time", "token"])
        for row in rows:
            writer.writerow([row["original_intent"], row["predicted_intent"], 0.0, 0])


def main() -> None:
    args = parse_args()
    repo = Path(args.repo).resolve()
    index_path = repo / (args.shared_memory or args.index)
    positive_path = repo / args.positive
    negative_path = repo / args.negative
    cache = torch.load(index_path, map_location="cpu", weights_only=False)
    if args.shared_memory:
        memory = task_frame(cache, "proactive")
        keys = task_keys(cache, "full_intent").float().numpy()
    else:
        total = pd.read_csv(repo / "samples/total.csv").reset_index(drop=True)
        memory = (
            total.iloc[cache["memory_indices"]]
            .reset_index()
            .rename(columns={"index": "source_index"})
        )
        keys = cache["keys"].float().numpy()
    if len(memory) != len(keys):
        raise ValueError("Index rows and embedding keys are misaligned")
    user_indices = {
        int(user): group.index.to_numpy()
        for user, group in memory.groupby("user_id", sort=True)
    }
    positive = json.loads(positive_path.read_text())
    negative = json.loads(negative_path.read_text())
    config: dict[str, float | int] = {
        "semantic_threshold": args.semantic_threshold,
        "minimum_support": args.minimum_support,
        "minimum_scenario_support": args.minimum_scenario_support,
        "time_sigma_hours": args.time_sigma_hours,
        "trigger_mass": args.trigger_mass,
        "selection_power": args.selection_power,
    }

    predictions: list[dict[str, Any]] = []
    for label, collection in ((1, positive), (0, negative)):
        for item in collection:
            # Deliberately do not pass originalInstruction, app, or intentClass.
            result = predict_state(
                int(item["user_id"]),
                str(item["time"]),
                str(item["scenario"]),
                memory,
                keys,
                user_indices,
                config,
            )
            predictions.append(
                {
                    "split_label": label,
                    "user_id": int(item["user_id"]),
                    "time": str(item["time"]),
                    "scenario": str(item["scenario"]),
                    **result,
                }
            )

    positives = predictions[: len(positive)]
    negatives = predictions[len(positive) :]
    tp = sum(row["predicted_intent"] != "False" for row in positives)
    fp = sum(row["predicted_intent"] != "False" for row in negatives)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / len(positives)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    exact_exclusions = sum(
        int(row.get("excluded_exact_states", 0)) for row in predictions
    )
    source_user_violations = sum(
        row.get("source_user_id", row["user_id"]) != row["user_id"]
        for row in predictions
    )
    result = {
        "method": "TRACE training-free proactive state synchronization",
        "protocol": (
            "official 215 positive + 100 negative states; official chronological "
            "first-80% memory; query fields limited to user_id/time/scenario; exact "
            "query state excluded; no fitted or fine-tuned parameters"
        ),
        "parameter_training": False,
        "config": config,
        "counts": {
            "positive": len(positive),
            "negative": len(negative),
            "predicted_positive_on_positive": tp,
            "predicted_positive_on_negative": fp,
            "true_negative": len(negative) - fp,
            "false_negative": len(positive) - tp,
        },
        "identification": {
            "precision": precision,
            "recall": recall,
            "false_alarm_rate": fp / len(negative),
            "f1": f1,
        },
        "audit": {
            "memory_rows": len(memory),
            "exact_query_states_excluded": exact_exclusions,
            "source_user_violations": int(source_user_violations),
            "positive_sha256": sha256(positive_path),
            "negative_sha256": sha256(negative_path),
            "index_sha256": sha256(index_path),
            "target_instruction_passed_to_predictor": False,
            "target_app_passed_to_predictor": False,
            "target_intent_class_passed_to_predictor": False,
        },
        "predictions": predictions,
    }
    if args.shared_memory:
        result["audit"]["shared_memory"] = True
    output = repo / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    write_official_csv(
        repo / args.positive_csv,
        [
            {
                "original_intent": item["originalInstruction"],
                "predicted_intent": prediction["predicted_intent"],
            }
            for item, prediction in zip(positive, positives)
        ],
    )
    write_official_csv(
        repo / args.negative_csv,
        [
            {
                "original_intent": item["originalInstruction"],
                "predicted_intent": prediction["predicted_intent"],
            }
            for item, prediction in zip(negative, negatives)
        ],
    )
    print(
        json.dumps({"counts": result["counts"], **result["identification"]}, indent=2)
    )
    print(f"saved {output}")


if __name__ == "__main__":
    main()

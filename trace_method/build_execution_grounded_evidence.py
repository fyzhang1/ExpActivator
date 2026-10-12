"""Build per-step TRACE evidence for a frozen GUI policy model.

Retrieval stops at a structured candidate action.  The candidate is serialized
as grounded historical evidence and is never treated as the final prediction.
The downstream policy still observes the current screenshot, ambiguous goal,
and teacher-forced task progress and may adopt, adapt, or reject the evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from trace_method.evaluate_execution import image_size, load_selected_features
from trace_method.execution_core import (
    audit_memory_split,
    execution_memory_frame,
    historical_action_to_qwen,
    history_key,
    load_history_trajectories,
    resolve_test_image,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=str(REPO))
    parser.add_argument("--history-root", default="data/Fingertip")
    parser.add_argument(
        "--index", default="trace_method/results/execution_app_predictions_gpu0.json"
    )
    parser.add_argument(
        "--calibration",
        default="trace_method/results/execution_visual_calibration.json",
    )
    parser.add_argument(
        "--history-calibration",
        default="trace_method/results/grounded/execution_history_calibration.json",
    )
    parser.add_argument(
        "--visual-dir", default="trace_method/results/cache/visual_full"
    )
    parser.add_argument(
        "--test-path", default="data/personal_execution/personal_execution.json"
    )
    parser.add_argument(
        "--episodes", default="data/personal_execution/personal_samples.json"
    )
    parser.add_argument("--trajectory-top-k", type=int, default=None)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evidence(action: dict[str, Any], basis: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "evidence_type": "trace_grounded_action",
        "retrieved_action": action,
        "retrieval_basis": basis,
    }
    if "coordinate" in action or "coordinate2" in action:
        result["historical_coordinate_frame"] = "0-999 normalized"
    return result


def main() -> None:
    args = parse_args()
    repo = Path(args.repo).resolve()
    history_root = Path(args.history_root).resolve()
    index_path = repo / args.index
    calibration_path = repo / args.calibration
    history_calibration_path = repo / args.history_calibration
    visual_dir = repo / args.visual_dir
    test_path = repo / args.test_path

    train = execution_memory_frame(repo)
    history_audit = audit_memory_split(train, repo, args.episodes)
    trajectories = load_history_trajectories(train, history_root)
    index = json.loads(index_path.read_text())
    calibration = json.loads(calibration_path.read_text())
    history_calibration = json.loads(history_calibration_path.read_text())
    selected_app_k = int(history_calibration["selected_app_k"])
    if int(index.get("top_k", -1)) != selected_app_k:
        raise ValueError("App index and history calibration top-k differ")
    if index.get("history_audit", {}).get("test_data_accessed") is not False:
        raise ValueError("App index does not certify target isolation")
    trajectory_top_k = (
        args.trajectory_top_k
        if args.trajectory_top_k is not None
        else int(history_calibration["selected_trajectory_k"])
    )
    if calibration.get("test_data_accessed") is not False:
        raise ValueError("Representation calibration must be history-only")
    selected_mode = str(calibration["selected_mode"])
    features, visual_manifest = load_selected_features(
        visual_dir, selected_mode, calibration.get("selected_center")
    )
    if visual_manifest.get("scope") != "full":
        raise ValueError("Grounded execution requires a full-scope visual cache")
    if visual_manifest.get("test_answers_accessed") is not False:
        raise ValueError("Visual feature construction must not access test answers")
    if int(visual_manifest.get("trajectory_top_k", -1)) != trajectory_top_k:
        raise ValueError("Visual cache and evidence trajectory top-k differ")
    calibrated_model = calibration.get("visual_model")
    if calibrated_model:
        for key in ("model_path", "model_type", "feature_readout", "pixels"):
            if calibrated_model.get(key) != visual_manifest.get(key):
                raise ValueError(f"Calibration/full visual model mismatch: {key}")

    test = json.loads(test_path.read_text())
    episode_ids = index["episode_ids"]
    if len(episode_ids) != 775 or len(set(episode_ids)) != 775:
        raise ValueError("History index must contain all 775 unique episodes")
    index_row = {episode_id: row for row, episode_id in enumerate(episode_ids)}
    memory_keys = [history_key(row.user_id, row.time) for row in train.itertuples()]

    pools: dict[str, tuple[np.ndarray, list[tuple[str, int, str]]]] = {}
    for episode_id in episode_ids:
        query_index = index_row[episode_id]
        selected: list[int] = []
        for train_index_value in index["factorized_neighbors"][query_index]:
            train_index = int(train_index_value)
            if memory_keys[train_index] == episode_id:
                continue
            if not trajectories[train_index]:
                continue
            selected.append(train_index)
            if len(selected) == trajectory_top_k:
                break
        states: list[tuple[np.ndarray, str, int, str]] = []
        for train_index in selected:
            if memory_keys[train_index] == episode_id:
                raise ValueError(
                    f"Trajectory memory for {episode_id} retained its own episode"
                )
            for path, action in trajectories[train_index]:
                cache_key = str(path)
                if cache_key in features:
                    states.append((features[cache_key], action, train_index, cache_key))
        if not states:
            raise ValueError(f"No cached historical state for {episode_id}")
        pools[episode_id] = (
            np.stack([state[0] for state in states]),
            [(state[1], state[2], state[3]) for state in states],
        )

    contexts: dict[str, dict[str, Any]] = {}
    sources: dict[str, dict[str, Any]] = {}
    error_candidates = 0
    for step in test:
        step_id = str(step["id"])
        episode_id, position_text = step_id.rsplit("_", 1)
        position = int(position_text)
        query_index = index_row[episode_id]
        if position == 0:
            app_name = str(index["factorized_app_names"][query_index])
            candidate = {"action": "open_app", "app_name": app_name}
            contexts[step_id] = evidence(
                candidate,
                "same-user residual-goal retrieval and reciprocal-rank app vote",
            )
            sources[step_id] = {
                "source_type": "aggregate_history_vote",
                "app_name": app_name,
                "vote_scores": index["factorized_vote_scores"][query_index],
            }
            continue

        current_path = str(resolve_test_image(step["img_path"], history_root))
        query = features[current_path]
        candidate_matrix, states = pools[episode_id]
        source_index = int(np.argmax(candidate_matrix @ query))
        action, train_index, source_path = states[source_index]
        candidate = historical_action_to_qwen(action, image_size(source_path))
        error_candidates += candidate.get("action") == "error"
        contexts[step_id] = evidence(
            candidate,
            "same-user residual-goal trajectory retrieval and visual state matching",
        )
        source = train.iloc[train_index]
        sources[step_id] = {
            "source_type": "matched_historical_state_action",
            "train_row_index": train_index,
            "user_id": int(source.user_id),
            "time": str(source.time),
            "image_path": source_path,
            "raw_historical_action": action,
            "visual_similarity": float(candidate_matrix[source_index] @ query),
        }

    if set(contexts) != {str(step["id"]) for step in test}:
        raise ValueError("Grounded evidence does not cover every execution step")
    result = {
        "protocol": {
            "method": "trace_grounded",
            "context_scope": "step",
            "role": "retrieved historical action is evidence for the policy model",
            "final_action_source": "frozen GUI policy conditioned on current input and evidence",
            "direct_action_emission": False,
            "policy_may_override_evidence": True,
            "teacher_forced_history": True,
            "history_audit": history_audit,
            "selected_app_k": selected_app_k,
            "trajectory_top_k": trajectory_top_k,
            "selected_mode": selected_mode,
            "visual_model_name": visual_manifest.get("model_name", "Qwen3-VL"),
            "visual_model_path": visual_manifest.get("model_path"),
            "visual_model_revision": visual_manifest.get("model_revision"),
            "test_sha256": sha256(test_path),
            "index_sha256": sha256(index_path),
            "calibration_sha256": sha256(calibration_path),
            "history_calibration_sha256": sha256(history_calibration_path),
            "test_answers_used_for_retrieval": False,
            "parameter_training": False,
            "error_candidate_actions": int(error_candidates),
        },
        "contexts": contexts,
        "sources": sources,
    }
    output = (repo / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(
        json.dumps(
            {
                "steps": len(contexts),
                "visual_model": result["protocol"]["visual_model_name"],
                "selected_mode": selected_mode,
                "error_candidate_actions": error_candidates,
                "output": str(output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

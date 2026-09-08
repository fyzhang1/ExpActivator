"""Evaluate TRACE with a history-selected frozen visual state embedding.

The visual representation is chosen exclusively by ``calibrate_visual.py`` on
pseudo-held-out training trajectories.
This script fixes that choice, performs current-screen synchronization with
cosine similarity, and scores the resulting actions on the complete released
execution split.  No model parameter is fitted or updated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from trace_method.execution_core import (
    historical_action_to_qwen,
    load_history_trajectories,
    load_outputs,
    resolve_test_image,
    scored_row,
    summarize,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=str(REPO))
    parser.add_argument(
        "--index", default="trace_method/results/execution_app_predictions_gpu0.json"
    )
    parser.add_argument(
        "--calibration",
        default="trace_method/results/execution_visual_calibration.json",
    )
    parser.add_argument(
        "--visual-dir", default="trace_method/results/cache/visual_full"
    )
    parser.add_argument(
        "--test-path", default="data/personal_execution/personal_execution.json"
    )
    parser.add_argument(
        "--baseline", default="trace_method/results/execution_baseline_gpu0.json"
    )
    parser.add_argument("--history-root", default="data/Fingertip")
    parser.add_argument("--trajectory-top-k", type=int, default=5)
    parser.add_argument(
        "--output", default="trace_method/results/execution_predictions_gpu0.json"
    )
    return parser.parse_args()


def load_selected_features(
    directory: Path,
    selected_mode: str,
    selected_center: list[list[float]] | None,
) -> tuple[dict[str, np.ndarray], dict]:
    """Load the history-selected block-cosine descriptor from a full cache."""
    manifest = json.loads((directory / "manifest.json").read_text())
    base_mode, suffix = selected_mode.rsplit("_", 1)
    center = (
        np.asarray(selected_center, dtype=np.float32)
        if selected_center is not None
        else None
    )
    if (suffix == "centered") != (center is not None):
        raise ValueError("Selected visual mode and saved center are inconsistent")
    result: dict[str, np.ndarray] = {}
    for chunk_path in sorted(directory.glob("chunk_*.pt")):
        chunk = torch.load(chunk_path, map_location="cpu", weights_only=False)
        global_features = chunk["global_features"].float().numpy()
        spatial_features = chunk["spatial_features"].float().numpy()
        if base_mode == "deep8":
            values = global_features[:, 0:1]
        elif base_mode == "deep16":
            values = global_features[:, 1:2]
        elif base_mode == "deep24":
            values = global_features[:, 2:3]
        elif base_mode == "final":
            values = global_features[:, 3:4]
        elif base_mode == "deep_mean":
            values = global_features
        elif base_mode == "spatial":
            values = spatial_features
        elif base_mode == "pyramid":
            values = np.concatenate([global_features[:, 3:4], spatial_features], axis=1)
        elif base_mode == "deep_pyramid":
            values = np.concatenate([global_features, spatial_features], axis=1)
        else:
            raise ValueError(f"Unknown selected visual mode: {selected_mode}")
        if center is not None:
            values = values - center[None, ...]
            norms = np.linalg.norm(values, axis=-1, keepdims=True)
            values = values / np.maximum(norms, 1e-12)
        values = values.reshape(values.shape[0], -1) / np.sqrt(values.shape[1])
        for row, path in enumerate(chunk["paths"]):
            result[str(path)] = values[row]
    if len(result) != len(manifest["paths"]):
        raise ValueError(f"Incomplete cache: {len(result)} != {len(manifest['paths'])}")
    return result, manifest


@lru_cache(maxsize=100_000)
def image_size(path: str) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def main() -> None:
    args = parse_args()
    repo = Path(args.repo).resolve()
    history_root = Path(args.history_root).resolve()
    calibration = json.loads((repo / args.calibration).read_text())
    if calibration.get("test_data_accessed") is not False:
        raise ValueError("Representation calibration must be history-only")
    selected_mode = str(calibration["selected_mode"])
    features, visual_manifest = load_selected_features(
        repo / args.visual_dir,
        selected_mode,
        calibration.get("selected_center"),
    )
    if visual_manifest.get("scope") != "full":
        raise ValueError("TRACE execution requires a full-scope visual cache")
    if visual_manifest.get("test_answers_accessed") is not False:
        raise ValueError("Visual feature construction must not access test answers")
    calibrated_model = calibration.get("visual_model", {})
    if calibrated_model:
        for key in ("model_path", "model_type", "feature_readout", "pixels"):
            if calibrated_model.get(key) != visual_manifest.get(key):
                raise ValueError(f"Calibration/full visual model mismatch: {key}")

    train = pd.read_csv(repo / "samples/train_set.csv").reset_index(drop=True)
    test_path = repo / args.test_path
    test = json.loads(test_path.read_text())
    index = json.loads((repo / args.index).read_text())
    baseline = load_outputs(repo / args.baseline)
    if [row["episode_id"] for row in baseline] != [row["id"] for row in test]:
        raise ValueError("Baseline and official test order differ")

    episode_ids = index["episode_ids"]
    if len(episode_ids) != 775 or len(set(episode_ids)) != 775:
        raise ValueError("History index must contain all 775 unique episodes")
    index_row = {episode_id: row for row, episode_id in enumerate(episode_ids)}
    trajectories = load_history_trajectories(train, history_root)

    # Construct the same top-5 goal-restricted historical pools as RGB TRACE.
    # Target answers are not read during pool construction or state selection.
    pools: dict[str, tuple[np.ndarray, list[tuple[str, int, str]]]] = {}
    exact_exclusion_violations = 0
    for episode_id in episode_ids:
        query_index = index_row[episode_id]
        user, time = episode_id.split("_", 1)
        selected = []
        for train_index in index["factorized_neighbors"][query_index]:
            source = train.iloc[int(train_index)]
            if int(source.user_id) == int(user) and str(source.time) == time:
                continue
            if not trajectories[int(train_index)]:
                continue
            selected.append(int(train_index))
            if len(selected) == args.trajectory_top_k:
                break
        states = []
        for train_index in selected:
            source = train.iloc[train_index]
            if int(source.user_id) == int(user) and str(source.time) == time:
                exact_exclusion_violations += 1
            for path, action in trajectories[train_index]:
                if str(path) in features:
                    states.append((features[str(path)], action, train_index, str(path)))
        if not states:
            raise ValueError(f"No cached historical state for {episode_id}")
        pools[episode_id] = (
            np.stack([state[0] for state in states]),
            [(state[1], state[2], state[3]) for state in states],
        )

    outputs = []
    for step_number, step in enumerate(test, start=1):
        episode_id = step["id"].rsplit("_", 1)[0]
        position = int(step["id"].rsplit("_", 1)[1])
        query_index = index_row[episode_id]
        current_path = str(resolve_test_image(step["img_path"], history_root))
        width, height = image_size(current_path)

        if position == 0:
            output = {
                "action": "open_app",
                "app_name": index["factorized_app_names"][query_index],
            }
            source_metadata = None
        else:
            query = features[current_path]
            candidate_matrix, states = pools[episode_id]
            source_index = int(np.argmax(candidate_matrix @ query))
            action, train_index, source_path = states[source_index]
            historical_size = image_size(source_path)
            output = historical_action_to_qwen(action, historical_size)
            source = train.iloc[train_index]
            source_metadata = {
                "train_row_index": train_index,
                "user_id": int(source.user_id),
                "time": str(source.time),
                "image_path": source_path,
                "visual_similarity": float(candidate_matrix[source_index] @ query),
            }

        row = scored_row(step, output, width, height)
        row["memory_source"] = source_metadata
        outputs.append(row)
        if step_number % 500 == 0:
            print(f"predicted {step_number}/{len(test)}", flush=True)

    result = {
        "protocol": {
            "name": (
                "TRACE with frozen "
                f"{visual_manifest['model_name']} visual state synchronization"
            ),
            "training": "none; visual backbone and all other encoders frozen",
            "representation_selection": "history-only pseudo-held-out trajectories",
            "selected_mode": selected_mode,
            "calibration_delta_vs_rgb": calibration["selected_delta_vs_rgb"],
            "trajectory_top_k": args.trajectory_top_k,
            "state_similarity": (
                "mean block cosine similarity of the history-selected frozen "
                "visual representation"
            ),
            "visual_model_name": visual_manifest["model_name"],
            "visual_model_path": visual_manifest["model_path"],
            "visual_model_revision": visual_manifest["model_revision"],
            "visual_model_type": visual_manifest["model_type"],
            "feature_readout": visual_manifest["feature_readout"],
            "pixels": visual_manifest["pixels"],
            "n_steps": len(test),
            "n_episodes": len(episode_ids),
            "test_sha256": hashlib.sha256(test_path.read_bytes()).hexdigest(),
            "physical_gpu_for_feature_extraction": visual_manifest["physical_gpu"],
            "visible_cuda_devices_for_feature_extraction": visual_manifest[
                "visible_cuda_devices"
            ],
            "exact_train_episode_excluded_from_trajectory_memory": True,
            "exact_exclusion_violations": exact_exclusion_violations,
            "test_answers_used_for_prediction": False,
        },
        "metrics": {
            "paired_baseline": summarize(baseline),
            "trace": summarize(outputs),
        },
        "outputs": outputs,
    }
    destination = repo / args.output
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))
    print(f"saved {destination}")


if __name__ == "__main__":
    main()

"""Evaluate the RGB-MSE state matcher retained as a ExpActivator ablation.

The predictor has no fitted parameters.  Episode-start app actions come from
the frozen history index.  At later steps, ExpActivator first restricts memory to
same-user episodes with related residual goals, then synchronizes execution by
matching the current screenshot to historical pre-action screenshots.  The
action attached to the closest historical state is transferred in the
official structured action schema.

The exact same ``(user_id, time)`` record is excluded from trajectory memory,
even when it occurs in the released training table.  Test answers are accessed
only after prediction for metric computation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from expactivator.execution_core import (
    execution_memory_frame,
    historical_action_to_qwen,
    load_history_trajectories,
    load_outputs,
    make_image_reader,
    resolve_test_image,
    scored_row,
    summarize,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=str(REPO))
    parser.add_argument(
        "--index", default="expactivator/results/execution_app_predictions_gpu0.json"
    )
    parser.add_argument(
        "--test-path", default="data/personal_execution/personal_execution.json"
    )
    parser.add_argument(
        "--baseline", default="expactivator/results/execution_baseline_gpu0.json"
    )
    parser.add_argument("--history-root", default="data/Fingertip")
    parser.add_argument("--trajectory-top-k", type=int, default=5)
    parser.add_argument("--thumbnail-width", type=int, default=24)
    parser.add_argument("--thumbnail-height", type=int, default=52)
    parser.add_argument(
        "--output", default="expactivator/results/execution_rgb_ablation_gpu0.json"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = Path(args.repo).resolve()
    history_root = Path(args.history_root).resolve()
    train = execution_memory_frame(repo)
    test = json.loads((repo / args.test_path).read_text())
    index = json.loads((repo / args.index).read_text())
    baseline = load_outputs(repo / args.baseline)
    if [row["episode_id"] for row in baseline] != [row["id"] for row in test]:
        raise ValueError("Baseline and official test order differ")

    episode_ids = index["episode_ids"]
    if len(episode_ids) != len(set(episode_ids)) or len(episode_ids) != 775:
        raise ValueError("History index must contain all 775 unique episodes")
    index_row = {episode_id: i for i, episode_id in enumerate(episode_ids)}
    trajectories = load_history_trajectories(train, history_root)
    read_image = make_image_reader(args.thumbnail_width, args.thumbnail_height)

    # Build action-memory pools before opening target answers.  Exact released
    # train/test duplicate episodes are explicitly removed from these pools.
    pools: dict[str, list[tuple[np.ndarray, tuple[int, int], str, int, str]]] = {}
    nearest_trajectories: dict[str, list[tuple[Path, str]]] = {}
    for pool_number, episode_id in enumerate(episode_ids, start=1):
        query_index = index_row[episode_id]
        user, time = episode_id.split("_", 1)
        selected = []
        for train_index in index["factorized_neighbors"][query_index]:
            row = train.iloc[int(train_index)]
            if int(row.user_id) == int(user) and str(row.time) == time:
                continue
            if not trajectories[int(train_index)]:
                continue
            selected.append(int(train_index))
            if len(selected) == args.trajectory_top_k:
                break
        if not selected:
            raise ValueError(f"No non-duplicate history trajectory for {episode_id}")
        nearest_trajectories[episode_id] = trajectories[selected[0]]
        states = []
        for train_index in selected:
            for image_path, action in trajectories[train_index]:
                descriptor, size = read_image(str(image_path))
                if descriptor is not None:
                    states.append(
                        (descriptor, size, action, train_index, str(image_path))
                    )
        if not states:
            raise ValueError(f"No readable historical screenshot for {episode_id}")
        pools[episode_id] = states
        if pool_number % 50 == 0:
            print(
                f"indexed trajectory pools {pool_number}/{len(episode_ids)}; "
                f"image_cache={read_image.cache_info()}",
                flush=True,
            )

    variants: dict[str, list[dict]] = {
        "full_goal_app_only": [],
        "factorized_app_only": [],
        "same_position": [],
        "expactivator": [],
    }
    for step_number, (step, base_row) in enumerate(zip(test, baseline), start=1):
        episode_id = step["id"].rsplit("_", 1)[0]
        position = int(step["id"].rsplit("_", 1)[1])
        query_index = index_row[episode_id]
        current_path = resolve_test_image(step["img_path"], history_root)
        current, current_size = read_image(str(current_path))
        width, height = current_size

        if position == 0:
            full_action = {
                "action": "open_app",
                "app_name": index["full_goal_app_names"][query_index],
            }
            factorized_action = {
                "action": "open_app",
                "app_name": index["factorized_app_names"][query_index],
            }
            variants["full_goal_app_only"].append(
                scored_row(step, full_action, width, height)
            )
            factorized_row = scored_row(step, factorized_action, width, height)
            variants["factorized_app_only"].append(factorized_row)
            variants["same_position"].append(dict(factorized_row))
            variants["expactivator"].append(dict(factorized_row))
        else:
            # The app-only ablations retain the exact paired base prediction.
            variants["full_goal_app_only"].append(dict(base_row))
            variants["factorized_app_only"].append(dict(base_row))

            trajectory = nearest_trajectories[episode_id]
            if position - 1 < len(trajectory):
                same_path, same_action = trajectory[position - 1]
            else:
                same_path, same_action = trajectory[-1]
            _, historical_size = read_image(str(same_path))
            same_output = historical_action_to_qwen(same_action, historical_size)
            variants["same_position"].append(
                scored_row(step, same_output, width, height)
            )

            if current is None:
                expactivator_output = {"action": "error"}
                expactivator_source = None
            else:
                best = min(
                    pools[episode_id],
                    key=lambda state: float(np.mean((current - state[0]) ** 2)),
                )
                descriptor, historical_size, action, train_index, image_path = best
                expactivator_output = historical_action_to_qwen(action, historical_size)
                source_row = train.iloc[train_index]
                expactivator_source = {
                    "train_row_index": train_index,
                    "user_id": int(source_row.user_id),
                    "time": str(source_row.time),
                    "image_path": image_path,
                    "screen_distance": float(np.mean((current - descriptor) ** 2)),
                }
            expactivator_row = scored_row(step, expactivator_output, width, height)
            expactivator_row["memory_source"] = expactivator_source
            variants["expactivator"].append(expactivator_row)

        if step_number % 500 == 0:
            print(
                f"predicted {step_number}/{len(test)}; "
                f"image_cache={read_image.cache_info()}",
                flush=True,
            )

    metrics = {name: summarize(rows) for name, rows in variants.items()}
    metrics["paired_baseline"] = summarize(baseline)
    result = {
        "protocol": {
            "name": "ExpActivator: Training-free Record Alignment for Consistent Execution",
            "training": "none; all encoders frozen; no fitted prediction parameters",
            "target_inputs": "user id, ambiguous instruction, current screenshot",
            "trajectory_top_k": args.trajectory_top_k,
            "state_distance": (
                f"RGB MSE after deterministic {args.thumbnail_width}x"
                f"{args.thumbnail_height} resize"
            ),
            "exact_train_episode_excluded_from_trajectory_memory": True,
            "test_answers_used_for_prediction": False,
        },
        "metrics": metrics,
        "outputs": variants["expactivator"],
        "ablation_outputs": {
            name: rows for name, rows in variants.items() if name != "expactivator"
        },
    }
    destination = repo / args.output
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"saved {destination}")


if __name__ == "__main__":
    main()

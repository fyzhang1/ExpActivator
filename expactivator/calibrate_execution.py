"""Select ExpActivator neighborhood sizes using historical records only.

The memory is the official chronological first-80%-per-user split of
``samples/total.csv``, which provably contains none of the 775 execution-test
episodes.  No AndroidIntent execution-test row, screenshot, or annotation is
opened.  App neighborhood size is evaluated on the chronological final 20% of
each user's history memory.  Trajectory neighborhood size is evaluated by treating
each user's last readable history episode as a held-out execution and using
only earlier episodes as memory.  Every candidate k scores the identical
action set, so their accuracies share one denominator.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.eval_execution.utils import eval_action
from expactivator.execution_core import (
    audit_memory_split,
    execution_memory_frame,
    load_app_labels,
)

COORD_RE = re.compile(r"coordinates=\((\d+),\s*(\d+)\)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=str(REPO))
    parser.add_argument(
        "--cache", default="expactivator/results/execution_history_index_gpu0.pt"
    )
    parser.add_argument(
        "--history-labels", default="expactivator/results/execution_app_labels.json"
    )
    parser.add_argument("--history-root", default="data/Fingertip")
    parser.add_argument("--app-k", default="1,3,5,8,10,16,32")
    parser.add_argument("--trajectory-k", default="1,3,5,8,10,16,20")
    parser.add_argument(
        "--output", default="expactivator/results/execution_history_calibration.json"
    )
    return parser.parse_args()


def normalize_action(action: str) -> str:
    return COORD_RE.sub(
        lambda match: f"coordinates=({match.group(1)},{match.group(2)})", action
    )


def load_trajectories(train: pd.DataFrame, root: Path) -> list[list[tuple[Path, str]]]:
    result = []
    for row in train.itertuples():
        episode = root / str(row.user_id) / str(row.user_id) / str(row.time)
        action_file = episode / "action.jsonl"
        states = []
        if action_file.exists():
            for line in action_file.read_text().splitlines():
                try:
                    entries = json.loads(line)
                except json.JSONDecodeError:
                    continue
                states.extend(
                    (
                        episode / f"Screenshots_{row.time}" / image_name,
                        normalize_action(str(action)),
                    )
                    for image_name, action in entries.items()
                )
        result.append(states)
    return result


@lru_cache(maxsize=50_000)
def thumbnail(path: str) -> tuple[np.ndarray | None, tuple[int, int]]:
    try:
        with Image.open(path) as image:
            size = image.size
            image.draft("RGB", (48, 104))
            value = np.asarray(image.convert("RGB").resize((24, 52)), dtype=np.float32)
            return value / 255.0, size
    except (FileNotFoundError, OSError):
        return None, (1080, 2340)


def vote(
    order: list[int], query: torch.Tensor, keys: torch.Tensor, labels: list[str], k: int
) -> str:
    scores: dict[str, float] = {}
    for rank, index in enumerate(order[:k], start=1):
        label = labels[index]
        similarity = float(query @ keys[index])
        scores[label] = scores.get(label, 0.0) + max(similarity, 0.0) / rank
    return min(scores, key=lambda label: (-scores[label], label))


def main() -> None:
    args = parse_args()
    repo = Path(args.repo).resolve()
    root = Path(args.history_root).resolve()
    app_k = [int(value) for value in args.app_k.split(",")]
    trajectory_k = [int(value) for value in args.trajectory_k.split(",")]
    train = execution_memory_frame(repo)
    history_audit = audit_memory_split(train, repo)
    cache = torch.load(repo / args.cache, map_location="cpu", weights_only=False)
    keys = cache["residual_keys"].float()
    if len(keys) != len(train):
        raise ValueError("Text cache and history memory are misaligned")
    labels = load_app_labels(repo, args.history_labels, train)

    app_hits = {k: 0 for k in app_k}
    app_total = 0
    for _, group in train.groupby("user_id", sort=True):
        indices = sorted(group.index, key=lambda index: str(train.iloc[index].time))
        split = max(1, int(len(indices) * 0.8))
        memory = torch.tensor(indices[:split], dtype=torch.long)
        for query_index in indices[split:]:
            similarities = keys[query_index] @ keys[memory].T
            order = memory[torch.argsort(similarities, descending=True)].tolist()
            app_total += 1
            for k in app_k:
                app_hits[k] += (
                    vote(order, keys[query_index], keys, labels, k)
                    == labels[query_index]
                )

    trajectories = load_trajectories(train, root)
    trajectory_counts = {k: [0, 0] for k in trajectory_k}
    held_out_episodes = 0
    for user, group in train.groupby("user_id", sort=True):
        readable = sorted(
            [int(index) for index in group.index if trajectories[int(index)]],
            key=lambda index: str(train.iloc[index].time),
        )
        if len(readable) < 3:
            continue
        target = readable[-1]
        memory = torch.tensor(readable[:-1], dtype=torch.long)
        similarities = keys[target] @ keys[memory].T
        order = memory[torch.argsort(similarities, descending=True)].tolist()
        order = order[: max(trajectory_k)]
        pools = {}
        for k in trajectory_k:
            states = []
            for index in order[:k]:
                for image_path, action in trajectories[index]:
                    descriptor, _ = thumbnail(str(image_path))
                    if descriptor is not None:
                        states.append((descriptor, action))
            pools[k] = states
        # Every k must score the identical action set, otherwise the selection
        # compares accuracies computed over different denominators.
        if not all(pools[k] for k in trajectory_k):
            continue

        for image_path, target_action in trajectories[target]:
            query_image, (width, height) = thumbnail(str(image_path))
            if query_image is None:
                continue
            for k in trajectory_k:
                _, prediction = min(
                    pools[k],
                    key=lambda state: float(np.mean((query_image - state[0]) ** 2)),
                )
                _, success = eval_action(prediction, target_action, width, height)
                trajectory_counts[k][0] += 1
                trajectory_counts[k][1] += success
        held_out_episodes += 1
        if held_out_episodes % 10 == 0:
            print(
                f"calibrated {held_out_episodes} users; cache={thumbnail.cache_info()}",
                flush=True,
            )

    if not app_total:
        raise ValueError("App calibration produced no validation records")
    totals = {total for total, _ in trajectory_counts.values()}
    if len(totals) != 1 or not totals.pop():
        raise ValueError("Trajectory calibration denominators differ across k")
    app_accuracy = {str(k): app_hits[k] / app_total for k in app_k}
    trajectory_accuracy = {
        str(k): successes / total for k, (total, successes) in trajectory_counts.items()
    }
    result = {
        "protocol": (
            "history memory only; chronological pseudo-held-out app records; "
            "last historical episode per user held out for trajectory calibration"
        ),
        "history_audit": history_audit,
        "test_data_accessed": history_audit["test_data_accessed"],
        "app_validation_records": app_total,
        "app_accuracy": app_accuracy,
        "selected_app_k": min(app_k, key=lambda k: (-app_accuracy[str(k)], k)),
        "trajectory_validation_episodes": held_out_episodes,
        "trajectory_validation_actions": next(iter(trajectory_counts.values()))[0],
        "trajectory_accuracy": trajectory_accuracy,
        "selected_trajectory_k": min(
            trajectory_k, key=lambda k: (-trajectory_accuracy[str(k)], k)
        ),
    }
    destination = repo / args.output
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

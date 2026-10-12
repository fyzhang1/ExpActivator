"""Migrate ExpActivator's sealed task views into one multi-view record memory.

This is an architectural equivalence test, so it deliberately migrates the
already sealed latent/ranking caches byte-for-byte instead of re-encoding them.
The two released history protocols remain separate task masks over shared
``(user_id, time)`` identities; no latent is fused across tasks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from expactivator.build_execution_index import residual_goal
from expactivator.build_proactive_index import chronological_memory_indices
from expactivator.execution_core import (
    audit_memory_split,
    execution_memory_frame,
    load_app_labels,
    load_history_trajectories,
)
from expactivator.shared_memory import SCHEMA, record_id, validate_shared_memory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=str(REPO))
    parser.add_argument("--history-root", default="data/Fingertip")
    parser.add_argument(
        "--history-labels", default="expactivator/results/execution_app_labels.json"
    )
    parser.add_argument(
        "--execution-cache",
        default="expactivator/results/execution_history_index_gpu0.pt",
    )
    parser.add_argument(
        "--proactive-cache",
        default="expactivator/results/proactive_history_index_gpu0.pt",
    )
    parser.add_argument(
        "--execution-index",
        default="expactivator/results/execution_app_predictions_gpu0.json",
    )
    parser.add_argument(
        "--output", default="expactivator/results/shared_multiview_memory_gpu0.pt"
    )
    parser.add_argument(
        "--execution-index-output",
        default="expactivator/results/shared_execution_app_predictions_gpu0.json",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_text(text: str) -> str:
    """Normalize formatting only, for cross-file identity validation."""
    return re.sub(r"\s+", "", str(text))


def main() -> None:
    args = parse_args()
    repo = Path(args.repo).resolve()
    history_root = Path(args.history_root).resolve()
    total_path = repo / "samples/total.csv"
    label_path = repo / args.history_labels
    execution_cache_path = repo / args.execution_cache
    proactive_cache_path = repo / args.proactive_cache
    legacy_index_path = repo / args.execution_index
    train = execution_memory_frame(repo)
    execution_audit = audit_memory_split(train, repo)
    total = pd.read_csv(total_path).reset_index(drop=True)
    if total.duplicated(["user_id", "time"]).any():
        raise ValueError("total.csv does not have unique (user_id, time) records")
    total_lookup = {
        (int(row.user_id), str(row.time)): index
        for index, row in total.iterrows()
    }
    execution_record_indices = [
        total_lookup[(int(row.user_id), str(row.time))]
        for row in train.itertuples()
    ]
    if len(set(execution_record_indices)) != len(train):
        raise ValueError("Execution histories do not map one-to-one into total.csv")

    formatting_overrides = []
    for train_index, record_index in enumerate(execution_record_indices):
        train_text = str(train.iloc[train_index].intentDescription)
        total_text = str(total.iloc[record_index].intentDescription)
        if normalized_text(train_text) != normalized_text(total_text):
            raise ValueError(
                f"History text differs beyond whitespace at train row {train_index}"
            )
        if train_text != total_text:
            formatting_overrides.append(train_index)
        for column in ("scenario", "app", "intentClass"):
            if str(train.iloc[train_index][column]) != str(total.iloc[record_index][column]):
                raise ValueError(f"History {column} mismatch at train row {train_index}")

    # Labels are joined on the stable (user_id, time) identity, so the store
    # stays correct under any history split.
    app_labels = load_app_labels(repo, args.history_labels, train)

    execution_cache = torch.load(
        execution_cache_path, map_location="cpu", weights_only=False
    )
    proactive_cache = torch.load(
        proactive_cache_path, map_location="cpu", weights_only=False
    )
    if execution_cache["embedding_model"] != proactive_cache["embedding_model"]:
        raise ValueError("Task caches do not use the same frozen text encoder")
    if execution_cache["residual_texts"] != [
        residual_goal(str(text)) for text in train.intentDescription
    ]:
        raise ValueError("Sealed residual texts do not match execution histories")
    if len(execution_cache["full_keys"]) != len(train):
        raise ValueError("Execution full-intent cache length changed")
    if len(execution_cache["residual_keys"]) != len(train):
        raise ValueError("Execution residual-goal cache length changed")

    proactive_record_indices = chronological_memory_indices(total)
    if proactive_cache["memory_indices"] != proactive_record_indices:
        raise ValueError("Sealed proactive cache does not match the first-80% mask")
    if len(proactive_cache["keys"]) != len(proactive_record_indices):
        raise ValueError("Proactive full-intent cache length changed")

    trajectories = load_history_trajectories(train, history_root)
    execution_by_record = {
        record_index: train_index
        for train_index, record_index in enumerate(execution_record_indices)
    }
    records = []
    for total_index, row in total.iterrows():
        train_index = execution_by_record.get(total_index)
        trajectory = []
        if train_index is not None:
            trajectory = [
                {
                    "screenshot_path": str(path),
                    "visual_cache_key": str(path),
                    "action": action,
                }
                for path, action in trajectories[train_index]
            ]
        records.append(
            {
                "record_id": record_id(int(row.user_id), str(row.time)),
                "total_index": int(total_index),
                "execution_train_index": train_index,
                "user_id": int(row.user_id),
                "time": str(row.time),
                "scenario": str(row.scenario),
                "app_package": str(row.app),
                "intent_class": str(row.intentClass),
                "full_intent": str(row.intentDescription),
                "execution_full_intent": (
                    str(train.iloc[train_index].intentDescription)
                    if train_index is not None
                    and str(train.iloc[train_index].intentDescription)
                    != str(row.intentDescription)
                    else None
                ),
                "residual_goal": (
                    str(execution_cache["residual_texts"][train_index])
                    if train_index is not None
                    else None
                ),
                "app_preference": (
                    str(app_labels[train_index])
                    if train_index is not None
                    else None
                ),
                "action_length": (
                    int(train.iloc[train_index].action_length)
                    if train_index is not None
                    else None
                ),
                "trajectory": trajectory,
            }
        )

    legacy_index = json.loads(legacy_index_path.read_text())
    query_episode_ids = legacy_index["episode_ids"]
    if len(query_episode_ids) != len(execution_cache["queries"]):
        raise ValueError("Execution query cache and ranked index are misaligned")
    memory = {
        "schema": SCHEMA,
        "protocol": (
            "byte-exact migration into one identity-stable multi-view record "
            "store; task-valid masks retain the released execution and proactive "
            "history splits; no cross-task latent fusion or scoring change"
        ),
        "records": records,
        "tasks": {
            "execution": {
                "record_indices": execution_record_indices,
                "source_indices": list(range(len(train))),
                "history_protocol": (
                    "chronological first 80% per user of total.csv"
                ),
            },
            "proactive": {
                "record_indices": proactive_record_indices,
                "source_indices": proactive_record_indices,
                "history_protocol": "chronological first 80% per user of total.csv",
            },
        },
        "embeddings": {
            "execution_full_intent": {
                "task": "execution",
                "record_indices": execution_record_indices,
                "keys": execution_cache["full_keys"],
                "storage_dtype": str(execution_cache["full_keys"].dtype),
            },
            "residual_goal": {
                "task": "execution",
                "record_indices": execution_record_indices,
                "keys": execution_cache["residual_keys"],
                "storage_dtype": str(execution_cache["residual_keys"].dtype),
            },
            "full_intent": {
                "task": "proactive",
                "record_indices": proactive_record_indices,
                "keys": proactive_cache["keys"],
                "storage_dtype": str(proactive_cache["keys"].dtype),
            },
        },
        "query_views": {
            "execution_ambiguous_goal": {
                "episode_ids": query_episode_ids,
                "keys": execution_cache["queries"],
                "target_answers_accessed": False,
            }
        },
        "embedding_model": execution_cache["embedding_model"],
        "parameter_training": False,
        "audit": {
            "migration_mode": "byte-exact sealed-view migration",
            "record_count": len(records),
            "execution_records": len(execution_record_indices),
            "proactive_records": len(proactive_record_indices),
            "shared_task_records": len(
                set(execution_record_indices) & set(proactive_record_indices)
            ),
            "trajectory_records": sum(bool(row["trajectory"]) for row in records),
            "trajectory_states": sum(len(row["trajectory"]) for row in records),
            "app_preference_records": sum(
                row["app_preference"] is not None for row in records
            ),
            "source_formatting_overrides": formatting_overrides,
            "source_user_time_join_violations": 0,
            "execution_history_audit": execution_audit,
            "total_sha256": sha256(total_path),
            "app_labels_sha256": sha256(label_path),
            "execution_cache_sha256": sha256(execution_cache_path),
            "proactive_cache_sha256": sha256(proactive_cache_path),
            "execution_index_sha256": sha256(legacy_index_path),
            "proactive_evaluation_data_accessed": False,
            "execution_answers_accessed": False,
        },
    }
    validate_shared_memory(memory)
    output = repo / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(memory, output)

    routed_index = dict(legacy_index)
    routed_index["protocol"] = (
        "unchanged sealed residual/full-goal ranking routed through shared "
        "multi-view record identities"
    )
    routed_index["shared_memory_schema"] = SCHEMA
    routed_index["factorized_record_indices"] = [
        [execution_record_indices[index] for index in neighbors]
        for neighbors in legacy_index["factorized_neighbors"]
    ]
    routed_index["full_goal_record_indices"] = [
        [execution_record_indices[index] for index in neighbors]
        for neighbors in legacy_index["full_goal_neighbors"]
    ]
    index_output = repo / args.execution_index_output
    index_output.parent.mkdir(parents=True, exist_ok=True)
    index_output.write_text(json.dumps(routed_index, ensure_ascii=False, indent=2))
    print(
        json.dumps(
            {
                "records": len(records),
                "execution_records": len(execution_record_indices),
                "proactive_records": len(proactive_record_indices),
                "shared_task_records": memory["audit"]["shared_task_records"],
                "formatting_overrides": len(formatting_overrides),
            },
            indent=2,
        )
    )
    print(f"saved {output}")
    print(f"saved {index_output}")


if __name__ == "__main__":
    main()

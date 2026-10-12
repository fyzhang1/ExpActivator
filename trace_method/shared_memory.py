"""Schema and task adapters for TRACE's shared multi-view record memory."""

from __future__ import annotations

from typing import Any

import pandas as pd
import torch

SCHEMA = "trace.shared-multiview-memory.v1"


def record_id(user_id: int, time: str) -> str:
    """Return the dataset-stable identity shared by both task protocols."""
    return f"{int(user_id)}_{time}"


def validate_shared_memory(memory: dict[str, Any]) -> None:
    """Reject artifacts whose task views no longer align with their records."""
    if memory.get("schema") != SCHEMA:
        raise ValueError(f"Unsupported shared-memory schema: {memory.get('schema')!r}")
    records = memory.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("Shared memory has no records")
    ids = [row["record_id"] for row in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Shared record IDs are not unique")

    for task in ("execution", "proactive"):
        task_indices = memory["tasks"][task]["record_indices"]
        if any(index < 0 or index >= len(records) for index in task_indices):
            raise ValueError(f"{task} mask contains an invalid record index")
    for name, view in memory["embeddings"].items():
        task = view["task"]
        task_indices = memory["tasks"][task]["record_indices"]
        if task_indices != view["record_indices"]:
            raise ValueError(f"{task} mask and {name} view are misaligned")
        if len(task_indices) != len(view["keys"]):
            raise ValueError(f"{name} records and keys are misaligned")


def task_frame(memory: dict[str, Any], task: str) -> pd.DataFrame:
    """Materialize a task-valid row view without changing shared identities."""
    validate_shared_memory(memory)
    if task not in ("execution", "proactive"):
        raise ValueError(f"Unknown shared-memory task: {task}")
    indices = memory["tasks"][task]["record_indices"]
    rows = []
    for record_index in indices:
        record = memory["records"][record_index]
        row = {
            "record_index": record_index,
            "record_id": record["record_id"],
            "user_id": record["user_id"],
            "time": record["time"],
            "scenario": record["scenario"],
            "app": record["app_package"],
            "intentDescription": record["full_intent"],
            "intentClass": record["intent_class"],
        }
        if task == "execution":
            row["source_index"] = record["execution_train_index"]
            row["intentDescription"] = record.get("execution_full_intent") or record[
                "full_intent"
            ]
            row["action_length"] = record["action_length"]
            row["app_name"] = record["app_preference"]
            row["residual_goal"] = record["residual_goal"]
            row["trajectory"] = record["trajectory"]
        else:
            row["source_index"] = record["total_index"]
        rows.append(row)
    frame = pd.DataFrame(rows)
    expected = memory["tasks"][task]["source_indices"]
    if frame.source_index.astype(int).tolist() != expected:
        raise ValueError(f"{task} source order changed while routing shared records")
    return frame


def task_keys(memory: dict[str, Any], view: str) -> torch.Tensor:
    """Read one named latent view and validate its task-mask alignment."""
    validate_shared_memory(memory)
    if view not in memory["embeddings"]:
        raise ValueError(f"Unknown shared-memory embedding view: {view}")
    keys = memory["embeddings"][view]["keys"]
    if not isinstance(keys, torch.Tensor):
        raise TypeError(f"Embedding view {view} is not a tensor")
    return keys


def execution_trajectories(
    memory: dict[str, Any],
) -> list[list[tuple[str, str]]]:
    """Return screenshot/action references in the original execution row order."""
    frame = task_frame(memory, "execution")
    return [
        [
            (str(state["visual_cache_key"]), str(state["action"]))
            for state in trajectory
        ]
        for trajectory in frame.trajectory
    ]

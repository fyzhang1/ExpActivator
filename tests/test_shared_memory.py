from __future__ import annotations

import torch

from expactivator.shared_memory import (
    SCHEMA,
    execution_trajectories,
    task_frame,
    task_keys,
    validate_shared_memory,
)


def fixture_memory() -> dict:
    records = [
        {
            "record_id": "1_t0",
            "total_index": 0,
            "execution_train_index": 0,
            "user_id": 1,
            "time": "t0",
            "scenario": "home",
            "app_package": "pkg",
            "intent_class": "class",
            "full_intent": "canonical intent",
            "execution_full_intent": "canonical\nintent",
            "residual_goal": "intent",
            "app_preference": "App",
            "action_length": 1,
            "trajectory": [
                {
                    "screenshot_path": "screen.png",
                    "visual_cache_key": "screen.png",
                    "action": "wait()",
                }
            ],
        },
        {
            "record_id": "1_t1",
            "total_index": 1,
            "execution_train_index": None,
            "user_id": 1,
            "time": "t1",
            "scenario": "work",
            "app_package": "pkg2",
            "intent_class": "class2",
            "full_intent": "proactive only",
            "execution_full_intent": None,
            "residual_goal": None,
            "app_preference": None,
            "action_length": None,
            "trajectory": [],
        },
    ]
    return {
        "schema": SCHEMA,
        "records": records,
        "tasks": {
            "execution": {"record_indices": [0], "source_indices": [0]},
            "proactive": {"record_indices": [0, 1], "source_indices": [0, 1]},
        },
        "embeddings": {
            "execution_full_intent": {
                "task": "execution",
                "record_indices": [0],
                "keys": torch.ones(1, 2),
            },
            "residual_goal": {
                "task": "execution",
                "record_indices": [0],
                "keys": torch.ones(1, 2),
            },
            "full_intent": {
                "task": "proactive",
                "record_indices": [0, 1],
                "keys": torch.ones(2, 2),
            },
        },
    }


def test_task_adapters_read_distinct_views_of_same_record() -> None:
    memory = fixture_memory()
    validate_shared_memory(memory)
    execution = task_frame(memory, "execution")
    proactive = task_frame(memory, "proactive")
    assert execution.loc[0, "record_id"] == proactive.loc[0, "record_id"] == "1_t0"
    assert execution.loc[0, "intentDescription"] == "canonical\nintent"
    assert proactive.loc[0, "intentDescription"] == "canonical intent"
    assert execution.loc[0, "residual_goal"] == "intent"
    assert proactive.scenario.tolist() == ["home", "work"]
    assert task_keys(memory, "residual_goal").shape == (1, 2)
    assert execution_trajectories(memory) == [[("screen.png", "wait()")]]


def test_shared_memory_rejects_task_view_misalignment() -> None:
    memory = fixture_memory()
    memory["embeddings"]["full_intent"]["record_indices"] = [1, 0]
    try:
        validate_shared_memory(memory)
    except ValueError as error:
        assert "misaligned" in str(error)
    else:
        raise AssertionError("misaligned shared view was accepted")

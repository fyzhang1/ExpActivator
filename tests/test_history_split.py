"""Regression guards for the execution history-memory protocol.

The superseded memory (``samples/train_set.csv``) retained 490 of the 775
execution-test episodes together with their explicit original instructions,
which leaked first-action answers and gave TRACE a memory the four paper
baselines never saw.  The previous in-code check could never fire, so these
tests assert both that the split is right and that the audit actually rejects
a wrong one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.eval_execution.build_paper_baseline_contexts import chronological_history
from trace_method.execution_core import (
    audit_memory_split,
    execution_memory_frame,
    held_out_episode_keys,
    history_key,
    load_app_labels,
)

REPO = Path(__file__).resolve().parents[1]


def keys_of(frame: pd.DataFrame) -> set[str]:
    return {history_key(row.user_id, row.time) for row in frame.itertuples()}


def test_memory_is_identical_to_the_paper_baseline_history() -> None:
    """TRACE and the four memory baselines must read the same histories."""
    memory = execution_memory_frame(REPO)
    baseline = chronological_history(pd.read_csv(REPO / "samples/total.csv"))
    assert len(memory) == len(baseline)
    assert keys_of(memory) == keys_of(baseline)


def test_memory_contains_no_execution_test_episode() -> None:
    memory = execution_memory_frame(REPO)
    audit = audit_memory_split(memory, REPO)
    assert audit["test_episodes_in_memory"] == 0
    assert audit["test_data_accessed"] is False
    assert not keys_of(memory) & held_out_episode_keys(REPO)


def test_audit_rejects_the_superseded_leaking_memory() -> None:
    """The audit must fail on train_set.csv, not merely record a zero."""
    leaking = pd.read_csv(REPO / "samples/train_set.csv")
    overlap = keys_of(leaking) & held_out_episode_keys(REPO)
    assert len(overlap) == 490, "fixture drifted; the released overlap is 490 episodes"
    with pytest.raises(ValueError, match="inside the history memory"):
        audit_memory_split(leaking, REPO)


def test_app_labels_are_joined_by_episode_identity(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "user_id": [7, 7, 9],
            "time": ["20250101_000000", "20250102_000000", "20250103_000000"],
        }
    )
    labels = tmp_path / "labels.json"
    # Deliberately stored in a different order from the frame.
    labels.write_text(
        json.dumps(
            {
                "rows": [
                    {"user_id": 9, "time": "20250103_000000", "app_name": "C"},
                    {"user_id": 7, "time": "20250102_000000", "app_name": "B"},
                    {"user_id": 7, "time": "20250101_000000", "app_name": "A"},
                    {"user_id": 1, "time": "20991231_000000", "app_name": "unused"},
                ]
            }
        )
    )
    assert load_app_labels(tmp_path, "labels.json", frame) == ["A", "B", "C"]


def test_missing_app_labels_fail_loudly(tmp_path: Path) -> None:
    frame = pd.DataFrame({"user_id": [7], "time": ["20250101_000000"]})
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({"rows": []}))
    with pytest.raises(ValueError, match="no compiled app label"):
        load_app_labels(tmp_path, "labels.json", frame)

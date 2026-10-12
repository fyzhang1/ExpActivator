from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src/eval_execution"))
sys.path.insert(0, str(REPO))

from evaluation_new_backbones import make_messages, parse_action

from src.eval_execution.build_paper_baseline_contexts import chronological_history
from expactivator.evaluation_stats import paired_binary


def test_parse_nested_mobile_use_json() -> None:
    raw = (
        '<tool_call>{"name":"mobile_use","arguments":'
        '{"action":"click","coordinate":[307,968]}}</tool_call>'
    )
    assert parse_action(raw) == (
        {"action": "click", "coordinate": [307, 968]},
        "json",
    )


def test_parse_direct_action_tool_name() -> None:
    raw = '<tool_call>{"name":"terminate","arguments":{}}</tool_call>'
    assert parse_action(raw) == ({"action": "terminate"}, "json")


def test_recover_malformed_and_truncated_json() -> None:
    malformed_wait = (
        '<tool_call>{"name":"mobile_use","arguments":'
        '{"action":"wait","time:2}}</tool_call>'
    )
    truncated_click = (
        'Action: click\n<tool_call>{"name":"mobile_use","arguments":'
        '{"action":"click","coordinate":[300,971]'
    )
    assert parse_action(malformed_wait) == (
        {"action": "wait"},
        "malformed_json",
    )
    assert parse_action(truncated_click) == (
        {"action": "click", "coordinate": [300, 971]},
        "malformed_json",
    )


def test_paired_binary_accepts_identical_arrays() -> None:
    values = np.asarray([0, 1, 0, 1])
    result = paired_binary(values, values, np.random.default_rng(7), 100)
    assert result["delta"] == 0.0
    assert result["wins"] == 0
    assert result["losses"] == 0
    assert result["exact_mcnemar_p"] == 1.0


def test_chronological_history_takes_first_80_percent_per_user() -> None:
    frame = pd.DataFrame(
        {
            "user_id": [2, 1, 1, 1, 1, 1, 2, 2, 2, 2],
            "time": [5, 5, 1, 4, 2, 3, 1, 4, 2, 3],
        }
    )
    history = chronological_history(frame)
    assert history.groupby("user_id")["time"].apply(list).to_dict() == {
        1: [1, 2, 3, 4],
        2: [1, 2, 3, 4],
    }


def test_memory_context_uses_native_prompt_slot() -> None:
    qwen = make_messages("qwen", "profile", "query", [], 999, 999, ["memory"])
    owl = make_messages("owl", "profile", "query", [], 999, 999, ["memory"])
    assert "User's Preferences" in qwen[1]["content"][1]["text"]
    assert "['memory']" in qwen[1]["content"][1]["text"]
    assert "user history record" in owl[1]["content"][1]["text"]
    assert "['memory']" in owl[1]["content"][1]["text"]

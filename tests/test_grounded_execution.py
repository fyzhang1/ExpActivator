from __future__ import annotations

from src.eval_execution.evaluation_new_backbones import make_messages
from expactivator.analyze_grounded_execution import policy_evidence_interaction
from expactivator.build_execution_grounded_evidence import evidence
from expactivator.execution_core import historical_action_to_qwen


def user_text(messages: list[dict]) -> str:
    return messages[1]["content"][1]["text"]


def test_grounded_action_prompt_keeps_policy_decision() -> None:
    context = evidence(
        {"action": "click", "coordinate": [400, 600]},
        "visual state matching",
    )
    messages = make_messages(
        "qwen",
        "profile",
        "query",
        ["open_app(app_name='A')"],
        999,
        999,
        context,
    )
    text = user_text(messages)
    assert "Grounded Historical Evidence" in text
    assert '"action": "click"' in text
    assert "adopt, adapt, or reject" in text
    assert "current screenshot" in text


def test_grounded_action_prompt_supports_owl_style() -> None:
    context = evidence(
        {"action": "type", "text": "hello"},
        "visual state matching",
    )
    messages = make_messages(
        "owl", "profile", "query", [], 1080, 2340, context
    )
    text = user_text(messages)
    assert "Grounded historical evidence" in text
    assert '"action": "type"' in text
    assert "do not copy it mechanically" in " ".join(text.split())


def test_goal_vote_evidence_is_not_described_as_screen_match() -> None:
    context = evidence(
        {"action": "open_app", "app_name": "地图"},
        "same-user residual-goal retrieval and reciprocal-rank app vote",
    )
    messages = make_messages("qwen", "profile", "query", [], 999, 999, context)
    text = user_text(messages)
    assert "goal-level vote or a goal-and-screen match" in " ".join(text.split())
    assert "retrieved from a historical state" not in text


def test_generic_memory_baseline_keeps_preference_prompt() -> None:
    messages = make_messages(
        "qwen", "profile", "query", [], 999, 999, ["memory record"]
    )
    text = user_text(messages)
    assert "User's Preferences" in text
    assert "Grounded Historical Evidence" not in text


def test_coordinate_evidence_declares_historical_frame() -> None:
    context = evidence(
        {"action": "long_click", "coordinate": [100, 200]},
        "visual state matching",
    )
    assert context["historical_coordinate_frame"] == "0-999 normalized"


def test_policy_evidence_interaction_distinguishes_copy_and_adaptation() -> None:
    evidence_rows = [
        {"output": {"action": "click", "coordinate": [100, 200]}},
        {"output": {"action": "type", "text": "old"}},
    ]
    policy_rows = [
        {"output": {"action": "click", "coordinate": [110, 210]}},
        {"output": {"action": "wait"}},
    ]
    result = policy_evidence_interaction(policy_rows, evidence_rows)
    assert result["exact_candidate_copy_rate"] == 0
    assert result["action_type_agreement_rate"] == 0.5
    assert result["changed_coordinate_rate"] == 1


def test_fenced_json_history_action_is_preserved_as_evidence() -> None:
    typed = historical_action_to_qwen(
        '```json\n{"action": "type", "text": "组会"}\n```', (1080, 2340)
    )
    clicked = historical_action_to_qwen(
        '```json\n{"action": "click", "coordinate": [540, 1170]}\n```',
        (1080, 2340),
    )
    assert typed == {"action": "type", "text": "组会"}
    assert clicked == {"action": "click", "coordinate": [500, 500]}

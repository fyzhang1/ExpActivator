"""Exact prompt-token accounting for every execution memory condition.

Replays the evaluator's own prompt assembly (``make_messages`` plus the chat
template) for all 7,915 test steps under each memory condition and tokenizes
the result with the tokenizer of the checkpoint that produced the run.  No GPU
and no image decoding: the screenshot reaches vLLM as ``multi_modal_data`` and
is identical across conditions, so the text-token difference from Base is
exactly the history-attributable cost of each memory mechanism.

The per-step evidence dictionary is tokenized as it is serialized at inference
time, including its constant ``retrieval_basis`` field, so the reported LEAF
cost is an overestimate rather than a favorable measurement.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

os.environ.setdefault("USE_TF", "0")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src/eval_execution"))

from transformers import AutoTokenizer

from evaluation_new_backbones import load_profiles, make_messages, sha256
from utils import convert_format

RUN = "trace_method/results/grounded/qwen3_vl_8b_grounded_policy_gpu0.json"
TEST = "data/personal_execution/personal_execution.json"
OUT = "trace_method/results/leaf_context_cost/token_cost.json"

CONDITIONS = {
    "Base": None,
    "Recent": "trace_method/results/paper_baselines/recent_contexts.json",
    "Retrieve": "trace_method/results/paper_baselines/retrieve_contexts.json",
    "LLM-UM": "trace_method/results/paper_baselines/llm_um_contexts.json",
    "HIM-Agent": "trace_method/results/paper_baselines/him_agent_contexts.json",
    "LEAF": "trace_method/results/grounded/qwen3_vl_8b_contexts.json",
}


def load_contexts(relative: str) -> tuple[dict, str, str]:
    payload = json.loads((REPO / relative).read_text())
    protocol = payload["protocol"]
    return (
        payload["contexts"],
        protocol.get("context_scope", "episode"),
        protocol.get("method"),
    )


def main() -> None:
    test_path = REPO / TEST
    test = json.loads(test_path.read_text())
    run = json.loads((REPO / RUN).read_text())
    protocol = run["protocol"]
    if protocol["test_sha256"] != sha256(test_path):
        raise ValueError("The grounded run was produced from a different test file")

    profiles = load_profiles(REPO / "samples/user_profile.csv")
    tokenizer = AutoTokenizer.from_pretrained(
        protocol["model_path"], trust_remote_code=True
    )

    prepared = []
    history: list[str] = []
    for step in test:
        if step["id"].endswith("_0"):
            history = []
        prepared.append((step, list(history)))
        history.append(convert_format(step["answer"]))

    report = {
        "protocol": {
            "name": "LEAF execution context-token accounting",
            "backbone": protocol["model_name"],
            "model_revision": protocol["model_revision"],
            "prompt_style": protocol["prompt_style"],
            "test_sha256": protocol["test_sha256"],
            "n_steps": len(test),
            "counts": "text prompt tokens after the chat template; the image is "
                      "passed as multi_modal_data and is identical across conditions",
            "history_tokens": "condition minus Base at the same step",
            "prefix_caching": "not assumed; every step is counted as a full prompt",
        },
        "conditions": {},
    }

    per_step: dict[str, list[int]] = {}
    for name, relative in CONDITIONS.items():
        contexts = scope = method = None
        if relative is not None:
            contexts, scope, method = load_contexts(relative)
        counts = []
        for step, action_history in prepared:
            memory_context = None
            if contexts is not None:
                key = step["id"] if scope == "step" else step["id"].rsplit("_", 1)[0]
                memory_context = contexts[key]
            messages = make_messages(
                protocol["prompt_style"],
                profiles[int(step["user_id"])],
                step["ambiguous_instruction"],
                action_history,
                999,
                999,
                memory_context,
            )
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            counts.append(len(tokenizer(text, add_special_tokens=False)["input_ids"]))
        per_step[name] = counts
        report["conditions"][name] = {
            "context_source": relative,
            "context_source_sha256": sha256(REPO / relative) if relative else None,
            "context_scope": scope,
            "context_method": method,
            "mean_prompt_tokens": sum(counts) / len(counts),
            "total_prompt_tokens": sum(counts),
        }
        print(
            f"{name:12s} mean_prompt={sum(counts) / len(counts):8.1f} "
            f"total={sum(counts):12,d}",
            flush=True,
        )

    base = per_step["Base"]
    for name, counts in per_step.items():
        deltas = [a - b for a, b in zip(counts, base)]
        entry = report["conditions"][name]
        entry["mean_history_tokens"] = sum(deltas) / len(deltas)
        entry["total_history_tokens"] = sum(deltas)
        entry["history_token_ratio_vs_leaf"] = None

    leaf = report["conditions"]["LEAF"]["mean_history_tokens"]
    for name, entry in report["conditions"].items():
        entry["history_token_ratio_vs_leaf"] = (
            entry["mean_history_tokens"] / leaf if leaf else None
        )

    destination = REPO / OUT
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nwrote {destination}")


if __name__ == "__main__":
    main()

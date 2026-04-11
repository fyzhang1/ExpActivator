#!/usr/bin/env python3
"""Minimal demo: from skill root, `pip install -r requirements.txt`, `python scripts/demo.py`."""

import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from him_memory import (
    HIMConfig,
    HIMMemory,
    episodes_from_simple_rows,
    export_openclaw_memory,
)


def main() -> None:
    rows = [
        {
            "episode_id": "1",
            "user_id": "u1",
            "text": "Summarize the weekly report in bullet points",
            "time": "20260110_090000",
            "call_sequence": ["read_file", "summarize", "reply"],
            "evaluation_tools": ["ruff"],
            "context": {"scenario": "office", "project": "alpha"},
        },
        {
            "episode_id": "2",
            "user_id": "u1",
            "text": "Give me bullets for last week's report",
            "time": "20260117_090100",
            "call_sequence": ["read_file", "summarize", "reply"],
            "evaluation_tools": ["ruff"],
            "context": {"scenario": "office", "project": "alpha"},
        },
        {
            "episode_id": "3",
            "user_id": "u1",
            "text": "Refactor this function to use async/await",
            "time": "20260112_140000",
            "call_sequence": ["read_file", "edit", "run_tests"],
            "evaluation_tools": ["pytest", "ruff"],
            "context": {"scenario": "home", "project": "beta"},
        },
        {
            "episode_id": "4",
            "user_id": "u1",
            "text": "Weekly report summary as bullets please",
            "time": "20260124_085500",
            "call_sequence": ["read_file", "summarize", "reply"],
            "evaluation_tools": ["ruff"],
            "context": {"scenario": "office", "project": "alpha"},
        },
    ]
    eps = episodes_from_simple_rows(rows)
    mem = HIMMemory(
        config=HIMConfig(core_cluster_radius=0.45),
    )
    for e in eps:
        mem.observe(e)

    print("=== Preference prototypes (execution-style clusters) ===")
    for p in mem.preference_prototypes():
        print("-", p["center_text"][:80], "| core=", p["is_core"], "| n=", p["cluster_len"])

    print("\n=== Routine candidates (state-style filter, density > threshold) ===")
    for r in mem.build_routine_memory():
        print("cluster", r["cluster_id"], "density=", round(r["total_cluster_density"], 3))
        print("  times:", r["collect_time"], "scenes:", r["collect_scene"])

    print("\n=== Retrieve for a new query ===")
    print(mem.retrieve_snippets("bullet summary of my report"))

    out = str(_SCRIPT_DIR / "him_memory_demo.json")
    mem.save(out)
    mem2 = HIMMemory()
    mem2.load(out)
    print("\n=== Reload OK, clusters:", len(mem2.clusters), "===")

    oc_dir = _SCRIPT_DIR / "openclaw_export"
    paths = export_openclaw_memory(mem, oc_dir)
    print("\n=== OpenClaw export ===")
    for k, v in paths.items():
        print(k, "->", v)


if __name__ == "__main__":
    main()

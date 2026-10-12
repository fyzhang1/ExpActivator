"""Additional fixed-source whole-trajectory prompting comparator.

Uses the native paper memory-baseline prompt, so unlike the ordinal/state pair
this is an ecological comparison of evidence presentation, not a prompt-matched
isolation. Both the source trajectory and the entry-step evidence stay fixed.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from expactivator.execution_core import execution_memory_frame, load_history_trajectories
from src.eval_execution.build_paper_baseline_contexts import record_text


def main():
    root = REPO / "expactivator/results/state_conditioning/qwen3_vl_8b"
    fixed = json.loads((root / "top1_ordinal_contexts.json").read_text())
    history = execution_memory_frame(REPO)
    trajectories = load_history_trajectories(history, Path("/storage/zhangfuyao/datasets/fingertip20k"))
    records = {}
    contexts = {}
    for key, context in fixed["contexts"].items():
        if key.endswith("_0"):
            contexts[key] = context
        else:
            i = fixed["sources"][key]["train_row_index"]
            if i not in records:
                records[i] = record_text(history.iloc[i], trajectories[i])
            contexts[key] = [records[i]]
    output = {"protocol": {**fixed["protocol"], "variant": "top1_trajectory",
        "evidence_form": "complete fixed historical trajectory in native baseline prompt",
        "prompt_matched_to_single_action_controls": False,
        "source_context_sha256": hashlib.sha256((root / "top1_ordinal_contexts.json").read_bytes()).hexdigest()},
        "contexts": contexts, "sources": fixed["sources"]}
    (root / "top1_trajectory_contexts.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(f"sealed {len(contexts)} full-trajectory contexts, {len(records)} unique source records")


if __name__ == "__main__":
    main()

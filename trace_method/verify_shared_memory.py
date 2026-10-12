"""Verify prediction equivalence of legacy and shared-record TRACE routing."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from trace_method.shared_memory import task_frame, validate_shared_memory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=str(REPO))
    parser.add_argument(
        "--shared-memory",
        default="trace_method/results/shared_multiview_memory_gpu0.pt",
    )
    parser.add_argument(
        "--shared-execution-index",
        default="trace_method/results/shared_execution_app_predictions_gpu0.json",
    )
    parser.add_argument(
        "--shared-execution-predictions",
        default="trace_method/results/shared_execution_predictions_gpu0.json",
    )
    parser.add_argument(
        "--shared-proactive-predictions",
        default="trace_method/results/shared_proactive_predictions.json",
    )
    parser.add_argument(
        "--output", default="trace_method/results/shared_memory_equivalence.json"
    )
    return parser.parse_args()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def require_equal(name: str, left: Any, right: Any) -> None:
    if left != right:
        raise ValueError(f"Shared-memory equivalence failed: {name}")


def main() -> None:
    args = parse_args()
    repo = Path(args.repo).resolve()
    shared_path = repo / args.shared_memory
    memory = torch.load(shared_path, map_location="cpu", weights_only=False)
    validate_shared_memory(memory)
    execution_frame = task_frame(memory, "execution")
    proactive_frame = task_frame(memory, "proactive")

    legacy_execution_cache = torch.load(
        repo / "trace_method/results/execution_history_index_gpu0.pt",
        map_location="cpu",
        weights_only=False,
    )
    legacy_proactive_cache = torch.load(
        repo / "trace_method/results/proactive_history_index_gpu0.pt",
        map_location="cpu",
        weights_only=False,
    )
    tensor_checks = {
        "execution_full_intent_fp16": torch.equal(
            memory["embeddings"]["execution_full_intent"]["keys"].half(),
            legacy_execution_cache["full_keys"],
        ),
        "execution_residual_goal_fp16": torch.equal(
            memory["embeddings"]["residual_goal"]["keys"].half(),
            legacy_execution_cache["residual_keys"],
        ),
        "execution_query_fp16": torch.equal(
            memory["query_views"]["execution_ambiguous_goal"]["keys"].half(),
            legacy_execution_cache["queries"],
        ),
        "proactive_full_intent_fp16": torch.equal(
            memory["embeddings"]["full_intent"]["keys"],
            legacy_proactive_cache["keys"],
        ),
    }
    if not all(tensor_checks.values()):
        failed = [name for name, passed in tensor_checks.items() if not passed]
        raise ValueError(f"Shared latent views differ from sealed caches: {failed}")

    legacy_index = json.loads(
        (repo / "trace_method/results/execution_app_predictions_gpu0.json").read_text()
    )
    shared_index = json.loads((repo / args.shared_execution_index).read_text())
    index_fields = [
        "episode_ids",
        "factorized_app_names",
        "full_goal_app_names",
        "factorized_neighbors",
        "full_goal_neighbors",
        "factorized_similarities",
        "full_goal_similarities",
        "factorized_vote_scores",
        "full_goal_vote_scores",
    ]
    index_checks = {}
    for field in index_fields:
        index_checks[field] = shared_index[field] == legacy_index[field]
        require_equal(f"execution index {field}", shared_index[field], legacy_index[field])

    legacy_execution = json.loads(
        (repo / "trace_method/results/execution_predictions_gpu0.json").read_text()
    )
    shared_execution = json.loads(
        (repo / args.shared_execution_predictions).read_text()
    )
    require_equal(
        "all execution outputs", shared_execution["outputs"], legacy_execution["outputs"]
    )
    require_equal(
        "execution metrics", shared_execution["metrics"], legacy_execution["metrics"]
    )

    legacy_proactive = json.loads(
        (repo / "trace_method/results/proactive_predictions.json").read_text()
    )
    legacy_proactive_analysis = json.loads(
        (repo / "trace_method/results/proactive_analysis.json").read_text()
    )
    shared_proactive = json.loads(
        (repo / args.shared_proactive_predictions).read_text()
    )
    require_equal(
        "all proactive predictions",
        shared_proactive["predictions"],
        legacy_proactive["predictions"],
    )
    require_equal(
        "proactive counts", shared_proactive["counts"], legacy_proactive["counts"]
    )
    require_equal(
        "proactive identification",
        shared_proactive["identification"],
        legacy_proactive["identification"],
    )

    result = {
        "claim_tested": (
            "both tasks share identity-stable multi-view records and read only "
            "their task-sufficient views, with no fused latent or scoring change"
        ),
        "passed": True,
        "shared_memory_schema": memory["schema"],
        "record_audit": memory["audit"],
        "task_views": {
            "execution": [
                "residual_goal",
                "app_preference",
                "trajectory",
                "visual_cache_key",
            ],
            "proactive": ["full_intent", "time", "scenario"],
        },
        "task_record_counts": {
            "execution": len(execution_frame),
            "proactive": len(proactive_frame),
            "overlap": memory["audit"]["shared_task_records"],
        },
        "tensor_checks": tensor_checks,
        "execution_index_checks": index_checks,
        "prediction_checks": {
            "execution_steps_exact": len(shared_execution["outputs"]),
            "execution_outputs_sha256": canonical_hash(shared_execution["outputs"]),
            "proactive_states_exact": len(shared_proactive["predictions"]),
            "proactive_predictions_sha256": canonical_hash(
                shared_proactive["predictions"]
            ),
        },
        "metrics": {
            "execution": shared_execution["metrics"]["trace"],
            "proactive": {
                **shared_proactive["identification"],
                "semantic": legacy_proactive_analysis["metrics"]["semantic"],
                "counts": shared_proactive["counts"],
            },
        },
    }
    output = repo / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"saved {output}")


if __name__ == "__main__":
    main()

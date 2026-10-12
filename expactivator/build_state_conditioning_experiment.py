"""Build fixed-trajectory state-alignment controls, then score sealed evidence.

The policy sees only a common evidence schema. Selection never reads target
answers or the original unabridged test instruction. Task-relevance inputs are
sealed before any ablation policy inference and exclude outcome information.
"""
from __future__ import annotations

import argparse
from collections import Counter
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("USE_TF", "0")
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from expactivator.build_execution_grounded_evidence import evidence
from expactivator.build_execution_index import residual_goal
from expactivator.evaluate_execution import image_size, load_selected_features
from expactivator.execution_core import (
    audit_memory_split, execution_memory_frame, historical_action_to_qwen,
    history_key, load_history_trajectories, resolve_test_image,
)
from src.eval_execution.utils import convert_action_qwen

VARIANTS = ("top1_ordinal", "top1_state", "top8_previous")
BASIS = "same-user residual-goal trajectory retrieval and visual state matching"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


@lru_cache(maxsize=None)
def pixels_sha(path: str) -> str:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        digest = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
    return digest


def ordinal_index(position: int, length: int) -> int:
    if position < 1 or length < 1:
        raise ValueError("ordinal lookup requires a non-entry position and nonempty history")
    return min(position - 1, length - 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", default="qwen3_vl_8b")
    parser.add_argument("--history-root", default="/storage/zhangfuyao/datasets/fingertip20k")
    parser.add_argument("--output-root", default="expactivator/results/state_conditioning")
    args = parser.parse_args()
    model = args.model_key
    output = REPO / args.output_root / model
    output.mkdir(parents=True, exist_ok=True)
    grounded = REPO / "expactivator/results/grounded"
    test_path = REPO / "data/personal_execution/personal_execution.json"
    original_context_path = grounded / f"{model}_contexts.json"
    index_path = grounded / "execution_app_predictions_gpu0.json"
    calibration_path = grounded / f"{model}_visual_calibration.json"
    cache_dir = grounded / "cache" / model / "full"
    root = Path(args.history_root)

    history = execution_memory_frame(REPO)
    audit = audit_memory_split(history, REPO)
    trajectories = load_history_trajectories(history, root)
    index = json.loads(index_path.read_text())
    original = json.loads(original_context_path.read_text())
    calibration = json.loads(calibration_path.read_text())
    print(f"{model}: loading fixed visual descriptors", flush=True)
    features, manifest = load_selected_features(
        cache_dir, calibration["selected_mode"], calibration.get("selected_center")
    )
    assert manifest["trajectory_top_k"] == 8
    assert manifest["scope"] == "full" and not manifest["test_answers_accessed"]
    assert manifest["model_revision"] == original["protocol"]["visual_model_revision"]
    # Only these fields are made available to context construction.
    queries = [
        {key: row[key] for key in ("id", "user_id", "ambiguous_instruction", "img_path")}
        for row in json.loads(test_path.read_text())
    ]
    episode_ids = index["episode_ids"]
    assert len(episode_ids) == 775 and len(queries) == 7915
    query_map = {row["id"].rsplit("_", 1)[0]: row for row in queries}
    lengths = Counter(row["id"].rsplit("_", 1)[0] for row in queries)
    top1: dict[str, int] = {}
    relevance_inputs = []
    for k, episode_id in enumerate(episode_ids):
        candidates = [int(i) for i in index["factorized_neighbors"][k] if trajectories[int(i)]]
        if not candidates:
            raise ValueError(f"No historical trajectory: {episode_id}")
        top1[episode_id] = candidates[0]
        source = history.iloc[candidates[0]]
        if history_key(source.user_id, source.time) == episode_id:
            raise ValueError("Self episode in history")
        assert int(source.user_id) == int(query_map[episode_id]["user_id"])
        relevance_inputs.append({
            "episode_id": episode_id,
            "query": query_map[episode_id]["ambiguous_instruction"],
            "historical_goal": str(source.intentDescription),
            "historical_residual_goal": residual_goal(str(source.intentDescription)),
            "source_episode_id": history_key(source.user_id, source.time),
        })
    write_json(output / "task_relevance_inputs.json", {
        "protocol": "Fixed top-one source; query and historical goal only; no target actions, original target instruction, or predictions.",
        "rows": relevance_inputs,
    })

    # Cache matrices by source trajectory; require exact original state order.
    matrices = {}
    for train_i in set(top1.values()):
        paths = [str(path) for path, _ in trajectories[train_i]]
        missing = [path for path in paths if path not in features]
        if missing:
            raise ValueError(f"Incomplete top-one visual cache: {missing[:1]}")
        matrices[train_i] = np.stack([features[path] for path in paths])
    contexts = {variant: {} for variant in VARIANTS}
    sources = {variant: {} for variant in VARIANTS}
    step_metadata = []
    for k, row in enumerate(queries):
        step_id = row["id"]
        episode_id, pos = step_id.rsplit("_", 1)
        position = int(pos)
        if position == 0:
            for variant in VARIANTS:
                contexts[variant][step_id] = original["contexts"][step_id]
                sources[variant][step_id] = original["sources"][step_id]
            continue
        query_path = str(resolve_test_image(row["img_path"], root))
        train_i = top1[episode_id]
        trajectory = trajectories[train_i]
        ordinal = ordinal_index(position, len(trajectory))
        similarities = matrices[train_i] @ features[query_path]
        matched = int(np.argmax(similarities))
        original_source = original["sources"][step_id]
        full_train_i = int(original_source["train_row_index"])
        full_trajectory = trajectories[full_train_i]
        full_j = next(j for j, (path, _) in enumerate(full_trajectory)
                      if str(path) == original_source["image_path"])
        selected = {
            "top1_ordinal": (train_i, ordinal),
            "top1_state": (train_i, matched),
            "top8_previous": (full_train_i, max(0, full_j - 1)),
        }
        for variant, (source_i, source_j) in selected.items():
            path, raw_action = trajectories[source_i][source_j]
            source_row = history.iloc[source_i]
            assert int(source_row.user_id) == int(row["user_id"])
            assert history_key(source_row.user_id, source_row.time) != episode_id
            action = historical_action_to_qwen(raw_action, image_size(str(path)))
            contexts[variant][step_id] = evidence(action, BASIS)
            sources[variant][step_id] = {
                "train_row_index": source_i, "history_state_index": source_j,
                "user_id": int(source_row.user_id), "time": str(source_row.time),
                "image_path": str(path), "raw_historical_action": raw_action,
            }
        same_pixels = pixels_sha(query_path) == pixels_sha(str(trajectory[matched][0]))
        step_metadata.append({
            "step_id": step_id, "episode_id": episode_id, "user_id": int(row["user_id"]),
            "position": position, "episode_steps": lengths[episode_id],
            "source_steps": len(trajectory), "ordinal_history_index": ordinal,
            "matched_history_index": matched,
            "index_displacement": abs(matched - ordinal),
            "ordinal_was_clamped": position - 1 >= len(trajectory),
            "top1_same_pixels": same_pixels,
            "top1_state_similarity": float(similarities[matched]),
            "top1_ordinal_similarity": float(similarities[ordinal]),
            "top8_previous_is_same_state": full_j == 0,
        })
        if (k + 1) % 1000 == 0:
            print(f"{model}: contexts {k + 1}/{len(queries)}", flush=True)

    assert set(original["contexts"]) == {row["id"] for row in queries}
    protocol = {
        "method": "expactivator_grounded", "context_scope": "step", "test_sha256": sha(test_path),
        "experiment": "state_conditioning", "model_key": model,
        "history_audit": audit, "source_contexts_sha256": sha(original_context_path),
        "index_sha256": sha(index_path), "calibration_sha256": sha(calibration_path),
        "visual_manifest_sha256": sha(cache_dir / "manifest.json"),
        "selected_mode": calibration["selected_mode"],
        "prediction_inputs": ["user_id", "ambiguous_instruction", "current_screenshot", "position"],
        "target_answers_used_for_selection": False,
        "target_original_instruction_used_for_selection": False,
        "parameter_training": False, "first_step_identical_to_full_expactivator": True,
        "common_policy_context_basis": BASIS,
    }
    for variant in VARIANTS:
        assert set(contexts[variant]) == set(original["contexts"])
        write_json(output / f"{variant}_contexts.json", {
            "protocol": {**protocol, "variant": variant},
            "contexts": contexts[variant], "sources": sources[variant],
        })
    write_json(output / "step_metadata.json", {"protocol": protocol, "rows": step_metadata})
    print(f"{model}: sealed all contexts; now opening labels for diagnostic scoring", flush=True)

    # No context can change after this point.
    all_contexts = {**contexts, "top8_state": original["contexts"]}
    test = json.loads(test_path.read_text())
    diagnostics = {variant: [] for variant in all_contexts}
    for row in test:
        width, height = image_size(str(resolve_test_image(row["img_path"], root)))
        for variant, ctx in all_contexts.items():
            action = ctx[row["id"]]["retrieved_action"]
            type_acc, sr = convert_action_qwen(action, row["answer"], width, height, coord_map=True)
            diagnostics[variant].append({"episode_id": row["id"], "output": action,
                                         "type_acc": type_acc, "sr": sr})
    write_json(output / "evidence_diagnostics.json", {
        "protocol": {**protocol, "role": "retrieved evidence only, not policy output"},
        "outputs": diagnostics,
    })
    print(json.dumps({v: {"nonentry_success": sum(r["sr"] for r in rows if not r["episode_id"].endswith("_0")) / 7140}
                      for v, rows in diagnostics.items()}), flush=True)


if __name__ == "__main__":
    main()

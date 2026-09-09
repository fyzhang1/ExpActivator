"""Build sealed per-episode contexts for the PersonalAlign paper baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("USE_TF", "0")

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from trace_method.execution_core import load_history_trajectories


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method", choices=("recent", "retrieve", "llm_um", "him_agent"), required=True
    )
    parser.add_argument("--history-csv", default="samples/total.csv")
    parser.add_argument(
        "--test-path", default="data/personal_execution/personal_execution.json"
    )
    parser.add_argument("--history-root", default="data/Fingertip")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--him-dir", default="result/HIM_Agent/preference_memory")
    parser.add_argument(
        "--llm-um-file",
        default="trace_method/results/paper_baselines/llm_um_profiles.json",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_files(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def chronological_history(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the paper's first-80%-per-user history split."""
    groups = []
    for _, group in frame.groupby("user_id", sort=True):
        ordered = group.sort_values("time", kind="stable")
        groups.append(ordered.iloc[: int(len(ordered) * 0.8)])
    return pd.concat(groups).reset_index(drop=True)


def record_text(row: object, actions: list[tuple[Path, str]]) -> str:
    action_list = [action for _, action in actions]
    return (
        f"Intent: {row.intentDescription}; App: {row.app}; "
        f"Actions: {json.dumps(action_list, ensure_ascii=False)}"
    )


def load_episode_queries(test: list[dict]) -> dict[str, dict]:
    result = {}
    for step in test:
        if step["id"].endswith("_0"):
            episode = step["id"].rsplit("_", 1)[0]
            result[episode] = step
    if len(result) != 775:
        raise ValueError(f"Expected 775 episode starts, found {len(result)}")
    return result


def encode(
    texts: list[str], model_path: Path, device: str, batch_size: int
) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(model_path), device=device, local_files_only=True)
    return model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )


def main() -> None:
    args = parse_args()
    history_path = (REPO / args.history_csv).resolve()
    test_path = (REPO / args.test_path).resolve()
    history_root = Path(args.history_root).resolve()
    history = chronological_history(pd.read_csv(history_path))
    trajectories = load_history_trajectories(history, history_root)
    records = [
        record_text(row, trajectories[index])
        for index, row in enumerate(history.itertuples())
    ]
    test = json.loads(test_path.read_text())
    queries = load_episode_queries(test)
    by_user = {
        int(user): group.sort_values("time", kind="stable").index.to_numpy()
        for user, group in history.groupby("user_id", sort=True)
    }

    embedding_model = Path(args.embedding_model).resolve()
    query_embeddings = None
    history_embeddings = None
    if args.method in {"retrieve", "him_agent"}:
        query_embeddings = encode(
            [row["ambiguous_instruction"] for row in queries.values()],
            embedding_model,
            args.device,
            args.batch_size,
        )
    if args.method == "retrieve":
        history_embeddings = encode(
            history["intentDescription"].astype(str).tolist(),
            embedding_model,
            args.device,
            args.batch_size,
        )

    llm_profiles = None
    llm_path = None
    if args.method == "llm_um":
        llm_path = (REPO / args.llm_um_file).resolve()
        llm_value = json.loads(llm_path.read_text())
        llm_profiles = {
            str(row["user_id"]): row["summaries"] for row in llm_value["users"]
        }

    contexts = {}
    sources = {}
    for query_index, (episode, step) in enumerate(queries.items()):
        user = int(step["user_id"])
        candidates = by_user[user]
        if args.method == "recent":
            selected = candidates[-args.top_k :][::-1]
            contexts[episode] = [records[int(index)] for index in selected]
            sources[episode] = [int(index) for index in selected]
        elif args.method == "retrieve":
            similarities = (
                history_embeddings[candidates] @ query_embeddings[query_index]
            )
            order = np.argsort(similarities, kind="stable")[-args.top_k :][::-1]
            selected = candidates[order]
            contexts[episode] = [records[int(index)] for index in selected]
            sources[episode] = [int(index) for index in selected]
        elif args.method == "llm_um":
            contexts[episode] = llm_profiles[str(user)]
            sources[episode] = [user]
        else:
            memory_path = (
                Path(args.him_dir).resolve() / f"user_{user}_preference_memory.json"
            )
            clusters = json.loads(memory_path.read_text())
            eligible = [row for row in clusters if row.get("cluster_len", 0) >= 3]
            if eligible:
                embeddings = np.asarray(
                    [row["center_text_emb"] for row in eligible], dtype=np.float32
                )
                embeddings /= np.maximum(
                    np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12
                )
                similarities = embeddings @ query_embeddings[query_index]
                # Preserve the released HIM execution evaluator's top-three
                # ordering and nested record representation exactly.
                order = np.argsort(similarities, kind="stable")[-args.top_k :]
                chosen = [eligible[int(index)] for index in order]
            else:
                chosen = []
            contexts[episode] = [
                [f"Intent: {row['center_text']}, Actions: {row['center_action']}"]
                for row in chosen
            ]
            sources[episode] = [int(row["cluster_id"]) for row in chosen]

    him_files = (
        list(Path(args.him_dir).resolve().glob("user_*_preference_memory.json"))
        if args.method == "him_agent"
        else []
    )
    result = {
        "protocol": {
            "method": args.method,
            "history_split": "chronological first 80% per user",
            "history_rows": len(history),
            "history_sha256": sha256(history_path),
            "test_sha256": sha256(test_path),
            "n_episodes": len(queries),
            "top_k": args.top_k if args.method != "llm_um" else None,
            "embedding_model": str(embedding_model)
            if args.method in {"retrieve", "him_agent"}
            else None,
            "embedding_model_revision": embedding_model.name
            if args.method in {"retrieve", "him_agent"}
            else None,
            "llm_um_file": str(llm_path) if llm_path else None,
            "llm_um_sha256": sha256(llm_path) if llm_path else None,
            "him_memory_files": len(him_files) if him_files else None,
            "him_memory_sha256": sha256_files(him_files) if him_files else None,
            "test_answers_accessed": False,
        },
        "contexts": contexts,
        "sources": sources,
    }
    output = (REPO / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    lengths = [
        len(json.dumps(value, ensure_ascii=False)) for value in contexts.values()
    ]
    print(
        json.dumps(
            {
                "method": args.method,
                "episodes": len(contexts),
                "history_rows": len(history),
                "mean_context_characters": sum(lengths) / len(lengths),
                "max_context_characters": max(lengths),
                "output": str(output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

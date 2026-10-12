"""Build ExpActivator's frozen, training-free index for proactive suggestion.

The official proactive protocol exposes the chronological first 80% of each
user's records as memory.  This script applies that split before encoding and
never opens either proactive evaluation file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

REPO = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=str(REPO))
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--output", default="expactivator/results/proactive_history_index_gpu0.pt"
    )
    return parser.parse_args()


def chronological_memory_indices(records: pd.DataFrame) -> list[int]:
    indices: list[int] = []
    for _, group in records.groupby("user_id", sort=True):
        ordered = sorted(group.index, key=lambda index: str(records.iloc[index].time))
        indices.extend(ordered[: int(len(ordered) * 0.8)])
    return indices


def mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded = mask.unsqueeze(-1)
    return (hidden * expanded).sum(1) / expanded.sum(1).clamp_min(1)


@torch.inference_mode()
def encode(
    texts: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: str,
    batch_size: int,
) -> torch.Tensor:
    chunks = []
    for start in tqdm(range(0, len(texts), batch_size), desc="proactive memory"):
        batch = tokenizer(
            texts[start : start + batch_size],
            padding=True,
            truncation=True,
            max_length=96,
            return_tensors="pt",
        ).to(device)
        hidden = model(**batch).last_hidden_state
        pooled = mean_pool(hidden, batch["attention_mask"])
        chunks.append(torch.nn.functional.normalize(pooled, dim=-1).cpu().half())
    return torch.cat(chunks)


def main() -> None:
    args = parse_args()
    repo = Path(args.repo).resolve()
    records = pd.read_csv(repo / "samples/total.csv").reset_index(drop=True)
    memory_indices = chronological_memory_indices(records)

    tokenizer = AutoTokenizer.from_pretrained(
        args.embedding_model, local_files_only=True
    )
    model = (
        AutoModel.from_pretrained(args.embedding_model, local_files_only=True)
        .to(args.device)
        .eval()
    )
    keys = encode(
        records.iloc[memory_indices].intentDescription.astype(str).tolist(),
        tokenizer,
        model,
        args.device,
        args.batch_size,
    )

    result = {
        "protocol": (
            "official chronological first-80%-per-user proactive memory; "
            "frozen multilingual encoder; no proactive evaluation file accessed"
        ),
        "source_csv": "samples/total.csv",
        "memory_indices": memory_indices,
        "keys": keys,
        "embedding_model": str(args.embedding_model),
        "device": args.device,
        "parameter_training": False,
        "evaluation_data_accessed": False,
    }
    output = repo / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output)
    print(f"saved {output} ({len(memory_indices)} histories, {tuple(keys.shape)})")


if __name__ == "__main__":
    main()

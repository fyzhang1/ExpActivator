"""Build LLM-UM user preference summaries with a fixed local summarizer."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from vllm.distributed import destroy_distributed_environment, destroy_model_parallel

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.eval_execution.build_paper_baseline_contexts import (
    chronological_history,
    record_text,
    sha256,
)
from src.eval_execution.evaluation_new_backbones import candidate_json_objects
from expactivator.execution_core import load_history_trajectories

SUMMARY_PROMPT = """Act as a user-modeling assistant. Summarize stable app and
content preferences from the user's long-term mobile GUI records.

Return only JSON with this shape:
{{"summaries": [{{"preference": "one concise Chinese app-and-task preference",
"confidence": "High/Medium/Low", "action": ["representative action strings"]}}]}}

Produce multiple entries when supported. Each entry must concern one app.
Prefer repeated evidence, do not invent preferences, and omit unsupported
claims. Keep action strings in their original language. Return at most
{max_summaries} summary entries and at most {max_actions_per_summary}
representative actions per entry so the JSON is complete and concise.

User profile:
{profile}

Chronological history:
{history}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen3.5-9B")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--history-csv", default="samples/total.csv")
    parser.add_argument("--history-root", default="data/Fingertip")
    parser.add_argument("--max-records", type=int, default=150)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-input-tokens", type=int, default=30000)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-summaries", type=int, default=10)
    parser.add_argument("--max-actions-per-summary", type=int, default=3)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help="Regenerate only users whose existing output has no parsed summaries.",
    )
    return parser.parse_args()


def load_profiles(path: Path) -> dict[int, str]:
    frame = pd.read_csv(path)
    result = {}
    for row in frame.itertuples():
        result[int(row.user_id)] = (
            f"{row.sex}，{row.age}岁，职业为{row.occupation}，现居{row.address}，"
            f"{row.family}，使用{row.phone_brand}手机"
        )
    return result


def parse_summary(text: str) -> list[dict]:
    for value in candidate_json_objects(text):
        summaries = value.get("summaries")
        if isinstance(summaries, list):
            return [row for row in summaries if isinstance(row, dict)]
    return []


def main() -> None:
    args = parse_args()
    if args.max_summaries < 1 or args.max_actions_per_summary < 1:
        raise ValueError("Summary and representative-action limits must be positive")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("LLM-UM generation may use only physical GPU 0")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one CUDA device must be visible")

    model_path = Path(args.model_path).resolve()
    history_path = (REPO / args.history_csv).resolve()
    history = chronological_history(pd.read_csv(history_path))
    trajectories = load_history_trajectories(history, Path(args.history_root).resolve())
    record_values = [
        record_text(row, trajectories[index])
        for index, row in enumerate(history.itertuples())
    ]
    profiles = load_profiles(REPO / "samples/user_profile.csv")
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)

    output = (REPO / args.output).resolve()
    existing: dict[int, dict] = {}
    if args.resume_existing and output.exists():
        previous = json.loads(output.read_text())
        existing = {
            int(row["user_id"]): row
            for row in previous["users"]
            if row.get("summaries")
        }

    prompts = []
    metadata = []
    for user, group in history.groupby("user_id", sort=True):
        if int(user) in existing:
            continue
        indices = group.sort_values("time", kind="stable").index.tolist()[
            -args.max_records :
        ]
        records = [record_values[index] for index in indices]
        while True:
            body = SUMMARY_PROMPT.format(
                profile=profiles[int(user)],
                history="\n".join(records),
                max_summaries=args.max_summaries,
                max_actions_per_summary=args.max_actions_per_summary,
            )
            messages = [{"role": "user", "content": body}]
            prompt = processor.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            tokens = len(processor.tokenizer.encode(prompt))
            if tokens <= args.max_input_tokens or len(records) <= 1:
                break
            records.pop(0)
        prompts.append(prompt)
        metadata.append((int(user), len(records), tokens))

    if not prompts:
        raise RuntimeError("No empty user profiles remain to regenerate")

    engine = LLM(
        model=str(model_path),
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        trust_remote_code=True,
        seed=0,
    )
    sampling = SamplingParams(
        temperature=0,
        max_tokens=args.max_new_tokens,
        seed=0,
    )
    generated = engine.generate(prompts, sampling)
    generated_users = {}
    for (user, records_used, input_tokens), result in zip(metadata, generated):
        raw = result.outputs[0].text
        generated_users[user] = {
            "user_id": user,
            "records_used": records_used,
            "input_tokens": input_tokens,
            "output_tokens": len(result.outputs[0].token_ids),
            "max_summaries": args.max_summaries,
            "max_actions_per_summary": args.max_actions_per_summary,
            "summaries": parse_summary(raw),
            "raw_output": raw,
        }

    users = [
        existing[user] if user in existing else generated_users[user]
        for user in sorted(existing.keys() | generated_users.keys())
    ]

    output.parent.mkdir(parents=True, exist_ok=True)
    value = {
        "protocol": {
            "method": "LLM-UM",
            "paper_summarizer": "gpt-5.1-2025-11-13",
            "reproduction_summarizer": args.model_name,
            "reproduction_model_path": str(model_path),
            "reproduction_model_revision": model_path.name,
            "history_split": "chronological first 80% per user",
            "history_sha256": sha256(history_path),
            "max_records": args.max_records,
            "max_input_tokens": args.max_input_tokens,
            "max_new_tokens": args.max_new_tokens,
            "max_summaries": args.max_summaries,
            "max_actions_per_summary": args.max_actions_per_summary,
            "resume_preserved_users": len(existing),
            "deterministic_decoding": True,
            "physical_gpu": 0,
            "visible_cuda_devices": "0",
        },
        "users": users,
    }
    output.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    print(
        json.dumps(
            {
                "users": len(users),
                "nonempty": sum(bool(row["summaries"]) for row in users),
                "mean_input_tokens": sum(row["input_tokens"] for row in users)
                / len(users),
                "mean_output_tokens": sum(row["output_tokens"] for row in users)
                / len(users),
                "output": str(output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    del engine
    destroy_model_parallel()
    destroy_distributed_environment()


if __name__ == "__main__":
    main()

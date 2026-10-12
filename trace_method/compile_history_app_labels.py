"""Compile canonical target-app labels from official factual histories.

The memory is the official chronological first-80%-per-user split of
``samples/total.csv``; no execution-test episode is inside it, and no
execution-test file or label is accessed.  Labels are keyed by the stable
``(user_id, time)`` episode identity rather than by row position, so one
compiled file can serve any history split without silent mis-alignment.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from trace_method.execution_core import (
    audit_memory_split,
    execution_memory_frame,
    history_key,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--repo", default=".")
    parser.add_argument(
        "--output", default="trace_method/results/execution_app_labels.json"
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--checkpoint-every", type=int, default=256)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--reuse",
        default=None,
        help=(
            "Import already-compiled labels from this file by (user_id, time) "
            "and generate only the records it does not cover"
        ),
    )
    return parser.parse_args()


def make_prompt(package: str, instruction: str) -> str:
    return (
        "你在编译手机用户的历史操作记忆。根据 Android 前台包名与显式任务，"
        "给出执行这条任务时应当首先打开或正在使用的目标应用的常用显示名。"
        "若包名是桌面启动器，以任务文字中的目标为准。"
        "只输出一个应用名，不要解释。\n"
        f"包名：{package}\n任务：{instruction}"
    )


def clean_label(text: str) -> str:
    value = text.strip().splitlines()[0] if text.strip() else ""
    value = re.sub(r"^(应用名|目标应用)[：:]\s*", "", value)
    value = value.strip(" `\"'。，、:：")
    return value


def save(path: Path, rows: list[dict], audit: dict) -> None:
    path.write_text(
        json.dumps(
            {
                "protocol": (
                    "frozen Qwen compilation from the official chronological "
                    "first-80%-per-user split of samples/total.csv only"
                ),
                "history_audit": audit,
                "n_rows": len(rows),
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def load_existing(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return {
        history_key(row["user_id"], row["time"]): row
        for row in json.loads(path.read_text())["rows"]
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    repo = Path(args.repo).resolve()
    memory = execution_memory_frame(repo)
    audit = audit_memory_split(memory, repo)
    destination = repo / args.output
    destination.parent.mkdir(parents=True, exist_ok=True)

    # Labels are addressed by episode identity, so an existing file compiled
    # under any earlier split can be reused for every record it already covers.
    existing: dict[str, dict] = {}
    if args.reuse:
        existing.update(load_existing(repo / args.reuse))
    if args.resume:
        existing.update(load_existing(destination))

    keys = [history_key(row.user_id, row.time) for row in memory.itertuples()]
    pending = [index for index, key in enumerate(keys) if key not in existing]
    print(
        f"memory records: {len(memory)}; reused labels: {len(memory) - len(pending)}; "
        f"to compile: {len(pending)}"
    )

    def ordered_rows() -> list[dict]:
        rows = []
        for position, key in enumerate(keys):
            if key not in existing:
                continue
            source = memory.iloc[position]
            rows.append(
                {
                    "record_id": key,
                    "row_index": position,
                    "user_id": int(source.user_id),
                    "time": str(source.time),
                    "package": str(source.app),
                    "app_name": existing[key]["app_name"],
                    "raw_output": existing[key].get("raw_output", ""),
                }
            )
        return rows

    if not pending:
        save(destination, ordered_rows(), audit)
        print("saved", destination, len(keys), "(nothing to compile)")
        return

    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
    processor.tokenizer.padding_side = "left"
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
        local_files_only=True,
    ).eval()

    compiled = 0
    for start in tqdm(
        range(0, len(pending), args.batch_size), desc="compile history app labels"
    ):
        batch = pending[start : start + args.batch_size]
        prompts = []
        for position in batch:
            source = memory.iloc[position]
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": make_prompt(
                                str(source.app), str(source.intentDescription)
                            ),
                        }
                    ],
                }
            ]
            prompts.append(
                processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            )
        tokens = processor.tokenizer(prompts, padding=True, return_tensors="pt").to(
            model.device
        )
        generated = model.generate(
            **tokens,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
        decoded = processor.tokenizer.batch_decode(
            [
                output[len(input_ids) :]
                for output, input_ids in zip(generated, tokens.input_ids)
            ],
            skip_special_tokens=True,
        )
        for position, raw in zip(batch, decoded):
            source = memory.iloc[position]
            existing[keys[position]] = {
                "user_id": int(source.user_id),
                "time": str(source.time),
                "app_name": clean_label(raw),
                "raw_output": raw,
            }
        compiled += len(batch)
        if args.checkpoint_every and compiled % args.checkpoint_every < args.batch_size:
            save(destination, ordered_rows(), audit)

    rows = ordered_rows()
    if len(rows) != len(keys):
        raise ValueError(f"Compiled {len(rows)} labels for {len(keys)} memory records")
    save(destination, rows, audit)
    print("saved", destination, len(rows))


if __name__ == "__main__":
    main()

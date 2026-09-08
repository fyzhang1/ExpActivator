"""Compile canonical target-app labels from official factual train histories.

Only ``samples/train_set.csv`` is read.  The frozen base model converts each
explicit training instruction and its recorded foreground package into the app
display name that an ``open_app`` action should use.  No execution-test file or
label is accessed.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


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


def save(path: Path, rows: list[dict]) -> None:
    path.write_text(
        json.dumps(
            {
                "protocol": "frozen Qwen compilation from official train_set.csv only",
                "n_rows": len(rows),
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    repo = Path(args.repo).resolve()
    train = pd.read_csv(repo / "samples/train_set.csv").reset_index(drop=True)
    destination = repo / args.output
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    if args.resume and destination.exists():
        rows = json.loads(destination.read_text())["rows"]
        for i, row in enumerate(rows):
            if row["row_index"] != i:
                raise ValueError("Existing label file is not an ordered prefix")

    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
    processor.tokenizer.padding_side = "left"
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
        local_files_only=True,
    ).eval()

    for start in tqdm(
        range(len(rows), len(train), args.batch_size), desc="compile train app labels"
    ):
        frame = train.iloc[start : start + args.batch_size]
        prompts = []
        for _, row in frame.iterrows():
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": make_prompt(
                                str(row.app), str(row.intentDescription)
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
        for row_index, ((_, row), raw) in enumerate(
            zip(frame.iterrows(), decoded), start=start
        ):
            rows.append(
                {
                    "row_index": row_index,
                    "user_id": int(row.user_id),
                    "time": str(row.time),
                    "package": str(row.app),
                    "app_name": clean_label(raw),
                    "raw_output": raw,
                }
            )
        if (
            args.checkpoint_every
            and len(rows) % args.checkpoint_every < args.batch_size
        ):
            save(destination, rows)
    save(destination, rows)
    print("saved", destination, len(rows))


if __name__ == "__main__":
    main()

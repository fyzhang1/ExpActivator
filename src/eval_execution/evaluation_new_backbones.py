"""Full, resumable PersonalAlign evaluation for recent local VLM backbones.

The evaluator is intentionally strict about physical GPU selection: it runs
only when ``CUDA_VISIBLE_DEVICES=0`` exposes a single logical CUDA device.  It
supports Qwen3-VL-derived GUI models and Qwen3.5 through Transformers'
``AutoModelForImageTextToText`` interface while preserving the official
teacher-forced PersonalAlign execution protocol.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src/eval_execution"))

from prompt import (
    OWL_PROMPT,
    OWL_USER_QUERY,
    OWL_USER_QUERY_RECORD,
    QWEN3_USER_PROMPT,
    QWEN3_USER_PROMPT_GENERALIZED,
    QWEN3VL_SYSTEM_PROMPT,
)

from utils import convert_action_json, convert_action_qwen, convert_format


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompt-style", choices=("qwen", "owl"), required=True)
    parser.add_argument(
        "--test-path", default="data/personal_execution/personal_execution.json"
    )
    parser.add_argument("--history-root", default="data/Fingertip")
    parser.add_argument("--save-file", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--pixels", type=int, default=200_704)
    parser.add_argument("--checkpoint-every", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--print-first", type=int, default=3)
    parser.add_argument("--attn-implementation", default="sdpa")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_profiles(path: Path) -> dict[int, str]:
    profiles = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            profiles[int(row["user_id"])] = (
                f"{row['sex']}, age: {row['age']}, job: {row['occupation']}, "
                f"living: {row['address']}, {row['family']}, use: {row['phone_brand']}"
            )
    return profiles


def resolve_image(path: str, history_root: Path) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate.resolve()
    normalized = path.replace("\\", "/")
    marker = "data/Fingertip/"
    if marker in normalized:
        candidate = history_root / normalized.split(marker, 1)[1]
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(path)


def parse_scalar(value: str) -> Any:
    value = value.strip()
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value.strip("`\"'")


def candidate_json_objects(text: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    candidates: list[dict[str, Any]] = []
    blocks = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL)
    for block in blocks + [text]:
        for offset, character in enumerate(block):
            if character != "{":
                continue
            try:
                value, _ = decoder.raw_decode(block[offset:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                candidates.append(value)
    return candidates


def expression_action(text: str) -> dict[str, Any] | None:
    match = re.search(
        r"\b(open_app|open|click|long_click|long_press|type|scroll|swipe|"
        r"press_home|press_back|press_recent|wait|finished|terminate)\s*\((.*?)\)",
        text,
        re.DOTALL,
    )
    if not match:
        return None
    action, body = match.groups()
    result: dict[str, Any] = {"action": action}
    app = re.search(r"app_name\s*=\s*['\"](.*?)['\"]", body, re.DOTALL)
    typed = re.search(r"(?:text|content)\s*=\s*['\"](.*?)['\"]", body, re.DOTALL)
    point = re.search(
        r"(?:coordinate|coordinates|point)\s*=\s*(?:['\"]?<point>)?"
        r"[\[(]?\s*(-?\d+(?:\.\d+)?)\s*[, ]\s*(-?\d+(?:\.\d+)?)",
        body,
    )
    direction = re.search(r"direction\s*=\s*['\"]?(up|down|left|right)", body)
    if app:
        result["app_name"] = app.group(1)
    if typed:
        result["text"] = typed.group(1)
    if point:
        result["coordinate"] = [float(point.group(1)), float(point.group(2))]
    if direction:
        result["direction"] = direction.group(1)
    return result


def malformed_json_action(text: str) -> dict[str, Any] | None:
    """Recover an explicit tool action when a model truncates/mangles its JSON."""
    action_match = re.search(
        r"""["']action["']\s*:\s*["']"""
        r"(open_app|open|click|long_click|long_press|type|scroll|swipe|"
        r"""press_home|press_back|press_recent|wait|finished|terminate)["']""",
        text,
    )
    if not action_match:
        return None
    result: dict[str, Any] = {"action": action_match.group(1)}
    for key in ("app_name", "text", "content", "direction"):
        match = re.search(
            rf'"{key}"\s*:\s*"((?:\\.|[^"\\])*)"',
            text,
            re.DOTALL,
        )
        if match:
            try:
                result[key] = json.loads(f'"{match.group(1)}"')
            except json.JSONDecodeError:
                result[key] = match.group(1)
    for key in ("coordinate", "coordinates", "point", "coordinate2", "point2"):
        match = re.search(
            rf"""["']{key}["']\s*:\s*\[\s*"""
            r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)",
            text,
        )
        if match:
            result[key] = [float(match.group(1)), float(match.group(2))]
    return result


def canonical_action(value: dict[str, Any]) -> dict[str, Any]:
    if isinstance(value.get("arguments"), dict):
        outer_name = str(value.get("name", "")).strip().lower()
        value = dict(value["arguments"])
        if "action" not in value and outer_name not in {"", "mobile_use"}:
            value["action"] = outer_name
    elif isinstance(value.get("function"), dict):
        function = value["function"]
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}
        value = dict(arguments) if isinstance(arguments, dict) else {}
        function_name = str(function.get("name", "")).strip().lower()
        if "action" not in value and function_name not in {"", "mobile_use"}:
            value["action"] = function_name
    else:
        value = dict(value)

    action = str(value.get("action", value.get("action_type", ""))).strip().lower()
    aliases = {
        "open": "open_app",
        "long_press": "long_click",
        "finished": "terminate",
        "finish": "terminate",
    }
    action = aliases.get(action, action)
    result: dict[str, Any] = {"action": action or "error"}
    app_name = value.get("app_name", value.get("application", value.get("name")))
    if app_name is not None:
        result["app_name"] = str(app_name)
    typed = value.get("text", value.get("content"))
    if typed is not None:
        result["text"] = str(typed)

    for source, destination in (
        ("coordinate", "coordinate"),
        ("coordinates", "coordinate"),
        ("point", "coordinate"),
        ("coordinate2", "coordinate2"),
        ("point2", "coordinate2"),
    ):
        coordinate = value.get(source)
        if isinstance(coordinate, str):
            numbers = re.findall(r"-?\d+(?:\.\d+)?", coordinate)
            coordinate = [float(item) for item in numbers[:2]]
        if isinstance(coordinate, (list, tuple)) and len(coordinate) >= 2:
            result[destination] = [
                round(float(coordinate[0])),
                round(float(coordinate[1])),
            ]

    direction = str(value.get("direction", "")).lower()
    if action in {"scroll", "swipe"} and "coordinate" not in result:
        endpoints = {
            "up": ([500, 800], [500, 200]),
            "down": ([500, 200], [500, 800]),
            "left": ([800, 500], [200, 500]),
            "right": ([200, 500], [800, 500]),
        }
        if direction in endpoints:
            result["coordinate"], result["coordinate2"] = endpoints[direction]
    return result


def parse_action(text: str) -> tuple[dict[str, Any], str]:
    for value in candidate_json_objects(text):
        action = canonical_action(value)
        if action["action"] != "error":
            return action, "json"

    malformed = malformed_json_action(text)
    if malformed:
        action = canonical_action(malformed)
        if action["action"] != "error":
            return action, "malformed_json"

    function = re.search(r"<function=[^>]+>(.*?)</function>", text, re.DOTALL)
    if function:
        parameters = {
            name.strip(): parse_scalar(value)
            for name, value in re.findall(
                r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>",
                function.group(1),
                re.DOTALL,
            )
        }
        action = canonical_action(parameters)
        if action["action"] != "error":
            return action, "qwen35_tool_xml"

    expression = expression_action(text)
    if expression:
        action = canonical_action(expression)
        if action["action"] != "error":
            return action, "action_expression"
    return {"action": "error"}, "parse_error"


def make_messages(
    prompt_style: str,
    profile: str,
    instruction: str,
    history: list[str],
    resized_width: int,
    resized_height: int,
    memory_context: object | None = None,
) -> list[dict[str, Any]]:
    if prompt_style == "owl":
        system = OWL_PROMPT.replace("{image_width}", str(resized_width)).replace(
            "{image_height}", str(resized_height)
        )
        template = (
            OWL_USER_QUERY_RECORD if memory_context is not None else OWL_USER_QUERY
        )
        user = template.format(
            user_profile=profile,
            instruction=instruction,
            history_actions=history,
            record=memory_context,
        )
    else:
        system = QWEN3VL_SYSTEM_PROMPT
        template = (
            QWEN3_USER_PROMPT_GENERALIZED
            if memory_context is not None
            else QWEN3_USER_PROMPT
        )
        user = template.format(
            user_profile=profile,
            instruction=instruction,
            history_actions=history,
            retrieved_intent=memory_context,
        )
    return [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user},
            ],
        },
    ]


def metrics(outputs: list[dict[str, Any]]) -> dict[str, Any]:
    if not outputs:
        return {"n_steps": 0}
    episodes: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for row in outputs:
        episode, position = row["episode_id"].rsplit("_", 1)
        episodes[episode].append((int(position), int(row["sr"])))
    weighted_success = 0.0
    total_weight = 0.0
    for rows in episodes.values():
        rows.sort()
        raw = [math.exp(-0.6 * index) for index in range(len(rows))]
        scale = len(rows) / sum(raw)
        weights = [value * scale for value in raw]
        weighted_success += sum(weight * row[1] for weight, row in zip(weights, rows))
        total_weight += sum(weights)
    starts = [row for row in outputs if row["episode_id"].endswith("_0")]
    nonfirst = [row for row in outputs if not row["episode_id"].endswith("_0")]
    return {
        "n_steps": len(outputs),
        "n_episodes": len(starts),
        "type_accuracy": sum(row["type_acc"] for row in outputs) / len(outputs),
        "success_rate": sum(row["sr"] for row in outputs) / len(outputs),
        "csr": weighted_success / total_weight,
        "first_step_success_rate": sum(row["sr"] for row in starts) / len(starts),
        "nonfirst_success_rate": (
            sum(row["sr"] for row in nonfirst) / len(nonfirst) if nonfirst else 0.0
        ),
        "parse_error_rate": sum(row["parse_method"] == "parse_error" for row in outputs)
        / len(outputs),
    }


def save_result(
    path: Path, protocol: dict[str, Any], outputs: list[dict[str, Any]]
) -> None:
    result = {"protocol": protocol, "metrics": metrics(outputs), "outputs": outputs}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError(
            "Set CUDA_VISIBLE_DEVICES=0; this evaluator may use only GPU 0"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "Exactly one visible CUDA device (physical GPU 0) is required"
        )
    if args.device != "cuda:0":
        raise ValueError("--device must be cuda:0")

    repo = REPO
    model_path = Path(args.model_path).resolve()
    test_path = (repo / args.test_path).resolve()
    history_root = Path(args.history_root).resolve()
    destination = (repo / args.save_file).resolve()
    test = json.loads(test_path.read_text())
    if args.limit is not None:
        test = test[: args.limit]
    profiles = load_profiles(repo / "samples/user_profile.csv")

    protocol: dict[str, Any] = {
        "model_name": args.model_name,
        "model_path": str(model_path),
        "model_revision": model_path.name,
        "backend": "transformers",
        "prompt_style": args.prompt_style,
        "ambiguous_instruction": True,
        "teacher_forced_history": True,
        "coordinate_protocol": "0-999 normalized coordinates",
        "pixels": args.pixels,
        "max_new_tokens": args.max_new_tokens,
        "requested_batch_size": args.batch_size,
        "deterministic_decoding": True,
        "test_sha256": sha256(test_path),
        "physical_gpu": 0,
        "visible_cuda_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "gpu_name": torch.cuda.get_device_name(0),
        "transformers_version": __import__("transformers").__version__,
        "torch_version": torch.__version__,
    }

    outputs: list[dict[str, Any]] = []
    if args.resume and destination.exists():
        previous = json.loads(destination.read_text())
        outputs = previous["outputs"]
        expected_prefix = [row["id"] for row in test[: len(outputs)]]
        if [row["episode_id"] for row in outputs] != expected_prefix:
            raise ValueError("Resume file is not an ordered prefix of this test run")
        for key in (
            "model_name",
            "model_revision",
            "backend",
            "prompt_style",
            "pixels",
            "max_new_tokens",
            "requested_batch_size",
            "test_sha256",
        ):
            if previous["protocol"].get(key) != protocol[key]:
                raise ValueError(f"Resume protocol mismatch: {key}")
        print(f"resuming from {len(outputs)}/{len(test)} predictions", flush=True)

    print(
        f"loading {args.model_name} on physical GPU 0 with {args.attn_implementation}",
        flush=True,
    )
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    processor.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map={"": args.device},
        attn_implementation=args.attn_implementation,
        local_files_only=True,
    ).eval()
    model.config.use_cache = True

    prepared = []
    history: list[str] = []
    for step in test:
        if step["id"].endswith("_0"):
            history = []
        prepared.append((step, list(history)))
        if args.prompt_style == "owl":
            history.append(convert_action_json(step["answer"]))
        else:
            history.append(convert_format(step["answer"]))

    cursor = len(outputs)
    batch_size = max(1, args.batch_size)
    last_checkpoint = len(outputs) // max(1, args.checkpoint_every)
    progress = tqdm(total=len(test), initial=cursor, desc=args.model_name)
    while cursor < len(test):
        current_size = min(batch_size, len(test) - cursor)
        batch = prepared[cursor : cursor + current_size]
        images: list[Image.Image] = []
        metadata = []
        texts = []
        try:
            for step, action_history in batch:
                path = resolve_image(step["img_path"], history_root)
                with Image.open(path) as source:
                    image = source.convert("RGB")
                width, height = image.size
                resized_width, resized_height = 999, 999
                instruction = step["ambiguous_instruction"]
                messages = make_messages(
                    args.prompt_style,
                    profiles[int(step["user_id"])],
                    instruction,
                    action_history,
                    resized_width,
                    resized_height,
                )
                texts.append(
                    processor.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                )
                images.append(image)
                metadata.append((step, width, height, resized_width, resized_height))

            inputs = processor(
                text=texts,
                images=images,
                padding=True,
                return_tensors="pt",
                min_pixels=args.pixels,
                max_pixels=args.pixels,
            ).to(args.device)
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                )
            trimmed = [
                output[len(input_ids) :]
                for input_ids, output in zip(inputs.input_ids, generated)
            ]
            raw_outputs = processor.batch_decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        except torch.OutOfMemoryError:
            for image in images:
                image.close()
            del images
            torch.cuda.empty_cache()
            if current_size == 1:
                raise
            batch_size = max(1, current_size // 2)
            print(f"CUDA OOM; retrying with batch_size={batch_size}", flush=True)
            continue

        for image in images:
            image.close()
        for raw, (step, width, height, resized_width, resized_height) in zip(
            raw_outputs, metadata
        ):
            action, parse_method = parse_action(raw)
            type_acc, success = convert_action_qwen(
                action, step["answer"], width, height, coord_map=True
            )
            row = {
                "episode_id": step["id"],
                "answer": step["answer"],
                "output": action,
                "raw_output": raw,
                "parse_method": parse_method,
                "type_acc": type_acc,
                "sr": success,
                "result": f"Type: {type_acc}, SR: {success}",
            }
            outputs.append(row)
            if len(outputs) <= args.print_first:
                print(json.dumps(row, ensure_ascii=False, indent=2), flush=True)

        cursor += current_size
        progress.update(current_size)
        checkpoint = len(outputs) // max(1, args.checkpoint_every)
        if checkpoint > last_checkpoint or cursor == len(test):
            protocol["effective_batch_size"] = batch_size
            save_result(destination, protocol, outputs)
            last_checkpoint = checkpoint
            print(
                f"checkpoint {len(outputs)}/{len(test)}: "
                f"{json.dumps(metrics(outputs), ensure_ascii=False)}",
                flush=True,
            )
    progress.close()
    print(json.dumps(metrics(outputs), ensure_ascii=False, indent=2))
    print(f"saved {destination}")


if __name__ == "__main__":
    main()

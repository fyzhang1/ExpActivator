"""Fast, resumable PersonalAlign evaluation for recent local VLM backbones.

This is the continuous-batching counterpart to ``evaluation_new_backbones``.
It keeps the same prompts, parser, coordinate protocol, and teacher-forced
history, while using vLLM to avoid padding every response to the longest
generation in a Transformers batch.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from vllm.distributed import destroy_distributed_environment, destroy_model_parallel

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src/eval_execution"))

from evaluation_new_backbones import (
    load_profiles,
    make_messages,
    metrics,
    parse_action,
    resolve_image,
    save_result,
    sha256,
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
    parser.add_argument("--request-batch-size", type=int, default=512)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--pixels", type=int, default=200_704)
    parser.add_argument("--checkpoint-every", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--print-first", type=int, default=3)
    parser.add_argument("--baseline-name", default="base")
    parser.add_argument("--memory-contexts", default=None)
    return parser.parse_args()


def validate_gpu(args: argparse.Namespace) -> None:
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
    if os.environ.get("VLLM_USE_FLASHINFER_SAMPLER") != "0":
        raise RuntimeError(
            "Set VLLM_USE_FLASHINFER_SAMPLER=0 for a deterministic sampler "
            "that does not require a local CUDA toolkit"
        )


def main() -> None:
    args = parse_args()
    validate_gpu(args)
    model_path = Path(args.model_path).resolve()
    test_path = (REPO / args.test_path).resolve()
    history_root = Path(args.history_root).resolve()
    destination = (REPO / args.save_file).resolve()
    full_test = json.loads(test_path.read_text())
    test = full_test
    if args.limit is not None:
        test = test[: args.limit]
    profiles = load_profiles(REPO / "samples/user_profile.csv")
    contexts = None
    context_path = None
    if args.memory_contexts:
        context_path = (REPO / args.memory_contexts).resolve()
        context_file = json.loads(context_path.read_text())
        contexts = context_file["contexts"]
        episodes = {row["id"].rsplit("_", 1)[0] for row in full_test}
        if set(contexts) != episodes:
            raise ValueError("Memory contexts do not cover test episodes exactly")
        if context_file["protocol"].get("test_sha256") != sha256(test_path):
            raise ValueError("Memory contexts were built for a different test file")
        if context_file["protocol"].get("method") != args.baseline_name:
            raise ValueError("Memory-context method does not match --baseline-name")
    elif args.baseline_name != "base":
        raise ValueError("A non-base baseline requires --memory-contexts")

    import transformers
    import vllm

    protocol: dict[str, Any] = {
        "model_name": args.model_name,
        "model_path": str(model_path),
        "model_revision": model_path.name,
        "backend": "vllm",
        "vllm_version": vllm.__version__,
        "prompt_style": args.prompt_style,
        "ambiguous_instruction": True,
        "teacher_forced_history": True,
        "coordinate_protocol": "0-999 normalized coordinates",
        "pixels": args.pixels,
        "max_new_tokens": args.max_new_tokens,
        "max_model_len": args.max_model_len,
        "request_batch_size": args.request_batch_size,
        "max_num_seqs": args.max_num_seqs,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "deterministic_decoding": True,
        "seed": 0,
        "stop_string": "</tool_call>",
        "test_sha256": sha256(test_path),
        "physical_gpu": 0,
        "visible_cuda_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "gpu_name": torch.cuda.get_device_name(0),
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "baseline_name": args.baseline_name,
        "memory_contexts": str(context_path) if context_path else None,
        "memory_contexts_sha256": sha256(context_path) if context_path else None,
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
            "max_model_len",
            "request_batch_size",
            "max_num_seqs",
            "gpu_memory_utilization",
            "test_sha256",
            "baseline_name",
            "memory_contexts_sha256",
        ):
            if previous["protocol"].get(key) != protocol[key]:
                raise ValueError(f"Resume protocol mismatch: {key}")
        print(f"resuming from {len(outputs)}/{len(test)} predictions", flush=True)

    if len(outputs) == len(test):
        print(json.dumps(metrics(outputs), ensure_ascii=False, indent=2))
        print(f"already complete: {destination}")
        return

    print(f"loading {args.model_name} through vLLM on physical GPU 0", flush=True)
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    processor.tokenizer.padding_side = "left"
    engine = LLM(
        model=str(model_path),
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        limit_mm_per_prompt={"image": 1},
        mm_processor_kwargs={
            "min_pixels": args.pixels,
            "max_pixels": args.pixels,
        },
        trust_remote_code=True,
        seed=0,
    )
    sampling = SamplingParams(
        temperature=0,
        max_tokens=args.max_new_tokens,
        stop=["</tool_call>"],
        include_stop_str_in_output=True,
        seed=0,
    )

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
    request_batch_size = max(1, args.request_batch_size)
    last_checkpoint = len(outputs) // max(1, args.checkpoint_every)
    progress = tqdm(total=len(test), initial=cursor, desc=args.model_name)
    while cursor < len(test):
        batch = prepared[cursor : cursor + request_batch_size]
        prompts = []
        images: list[Image.Image] = []
        metadata = []
        for step, action_history in batch:
            path = resolve_image(step["img_path"], history_root)
            with Image.open(path) as source:
                image = source.convert("RGB")
            width, height = image.size
            messages = make_messages(
                args.prompt_style,
                profiles[int(step["user_id"])],
                step["ambiguous_instruction"],
                action_history,
                999,
                999,
                contexts[step["id"].rsplit("_", 1)[0]] if contexts else None,
            )
            text = processor.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            images.append(image)
            prompts.append({"prompt": text, "multi_modal_data": {"image": image}})
            metadata.append((step, width, height))

        generated = engine.generate(prompts, sampling, use_tqdm=False)
        if len(generated) != len(metadata):
            raise RuntimeError(
                f"vLLM returned {len(generated)} outputs for {len(metadata)} prompts"
            )
        raw_outputs = [item.outputs[0].text for item in generated]
        for image in images:
            image.close()

        for raw, (step, width, height) in zip(raw_outputs, metadata):
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

        cursor += len(batch)
        progress.update(len(batch))
        checkpoint = len(outputs) // max(1, args.checkpoint_every)
        if checkpoint > last_checkpoint or cursor == len(test):
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
    del engine
    destroy_model_parallel()
    destroy_distributed_environment()


if __name__ == "__main__":
    main()

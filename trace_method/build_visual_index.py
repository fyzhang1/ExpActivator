"""Extract frozen Qwen3-VL visual-state features for TRACE.

Two manifests are supported. ``calibration`` contains only historical
pseudo-held-out trajectories and their earlier memories. ``full`` contains
the released execution screenshots and the exact same top-5 historical pools
used by TRACE.  Neither manifest opens target actions while selecting images.

Each image stores normalized global means from three Qwen3-VL deep-stack
layers plus the final visual layer, and four final-layer spatial quadrant
means.  This permits test-free selection of a representation while keeping
the visual model frozen.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("USE_TF", "0")

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from trace_method.execution_core import (
    load_history_trajectories,
    resolve_test_image,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=str(REPO))
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--history-root", default="data/Fingertip")
    parser.add_argument("--scope", choices=("calibration", "full"), required=True)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--pixels", type=int, default=200_704)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--trajectory-top-k", type=int, default=5)
    parser.add_argument(
        "--index", default="trace_method/results/execution_app_predictions_gpu0.json"
    )
    parser.add_argument(
        "--text-cache", default="trace_method/results/execution_history_index_gpu0.pt"
    )
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def selected_trajectories(
    train: pd.DataFrame,
    index: dict,
    trajectories: list[list[tuple[Path, str]]],
    top_k: int,
) -> dict[str, list[int]]:
    result = {}
    for query_index, episode_id in enumerate(index["episode_ids"]):
        user, time = episode_id.split("_", 1)
        selected = []
        for train_index in index["factorized_neighbors"][query_index]:
            row = train.iloc[int(train_index)]
            if int(row.user_id) == int(user) and str(row.time) == time:
                continue
            if not trajectories[int(train_index)]:
                continue
            selected.append(int(train_index))
            if len(selected) == top_k:
                break
        result[episode_id] = selected
    return result


def build_manifest(args: argparse.Namespace) -> list[str]:
    repo = Path(args.repo).resolve()
    history_root = Path(args.history_root).resolve()
    train = pd.read_csv(repo / "samples/train_set.csv").reset_index(drop=True)
    trajectories = load_history_trajectories(train, history_root)
    paths: set[str] = set()

    if args.scope == "full":
        index = json.loads((repo / args.index).read_text())
        selected = selected_trajectories(
            train, index, trajectories, args.trajectory_top_k
        )
        for indices in selected.values():
            paths.update(
                str(image_path)
                for index in indices
                for image_path, _ in trajectories[index]
                if image_path.exists()
            )
        test = json.loads(
            (repo / "data/personal_execution/personal_execution.json").read_text()
        )
        paths.update(
            str(resolve_test_image(step["img_path"], history_root)) for step in test
        )
    else:
        cache = torch.load(
            repo / args.text_cache, map_location="cpu", weights_only=False
        )
        keys = cache["residual_keys"].float()
        for _, group in train.groupby("user_id", sort=True):
            readable = sorted(
                [int(index) for index in group.index if trajectories[int(index)]],
                key=lambda index: str(train.iloc[index].time),
            )
            if len(readable) < 3:
                continue
            target = readable[-1]
            memory = torch.tensor(readable[:-1], dtype=torch.long)
            order = memory[
                torch.argsort(keys[target] @ keys[memory].T, descending=True)
            ].tolist()[: args.trajectory_top_k]
            paths.update(
                str(image_path)
                for index in order + [target]
                for image_path, _ in trajectories[index]
                if image_path.exists()
            )
    return sorted(paths)


def normalized_mean(tokens: torch.Tensor) -> torch.Tensor:
    return F.normalize(tokens.float().mean(0), dim=0)


def extract_batch(
    paths: list[str],
    processor: AutoProcessor,
    vision: torch.nn.Module,
    device: str,
    pixels: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    inputs = processor.image_processor(
        images=images,
        return_tensors="pt",
        min_pixels=pixels,
        max_pixels=pixels,
    )
    grids = inputs.image_grid_thw.to(device)
    with torch.inference_mode():
        final, deep = vision(
            inputs.pixel_values.to(device=device, dtype=vision.dtype),
            grid_thw=grids,
        )
    split_sizes = (grids.prod(-1) // vision.spatial_merge_size**2).tolist()
    final_split = torch.split(final, split_sizes)
    deep_split = [torch.split(layer, split_sizes) for layer in deep]

    global_rows = []
    spatial_rows = []
    for image_index, tokens in enumerate(final_split):
        globals_for_image = [
            normalized_mean(layer[image_index]) for layer in deep_split
        ] + [normalized_mean(tokens)]
        global_rows.append(torch.stack(globals_for_image))

        _, grid_h, grid_w = [int(value) for value in grids[image_index].tolist()]
        height = grid_h // vision.spatial_merge_size
        width = grid_w // vision.spatial_merge_size
        layout = tokens.reshape(height, width, -1)
        row_slices = (slice(0, max(height // 2, 1)), slice(height // 2, height))
        col_slices = (slice(0, max(width // 2, 1)), slice(width // 2, width))
        quadrants = []
        for row_slice in row_slices:
            for col_slice in col_slices:
                block = layout[row_slice, col_slice]
                quadrants.append(normalized_mean(block.reshape(-1, block.shape[-1])))
        spatial_rows.append(torch.stack(quadrants))
    return (
        torch.stack(global_rows).cpu().half(),
        torch.stack(spatial_rows).cpu().half(),
    )


def main() -> None:
    args = parse_args()
    repo = Path(args.repo).resolve()
    output_dir = repo / (
        args.output_dir or f"trace_method/results/cache/visual_{args.scope}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    paths = build_manifest(args)
    manifest = {
        "protocol": (
            "frozen Qwen3-VL visual tower; no target action used to choose images; "
            "normalized deep-stack/final global means and final spatial quadrants"
        ),
        "scope": args.scope,
        "model_path": str(args.model_path),
        "device": args.device,
        "pixels": args.pixels,
        "trajectory_top_k": args.trajectory_top_k,
        "paths": paths,
        "parameter_training": False,
        "test_answers_accessed": False,
    }
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous["paths"] != paths or previous["pixels"] != args.pixels:
            raise ValueError("Existing visual manifest does not match this run")
    else:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    total_chunks = (len(paths) + args.chunk_size - 1) // args.chunk_size
    pending = [
        chunk
        for chunk in range(total_chunks)
        if not (output_dir / f"chunk_{chunk:04d}.pt").exists()
    ]
    if not pending:
        print(f"all {len(paths)} images already encoded in {output_dir}")
        return

    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        device_map={"": args.device},
        attn_implementation="sdpa",
        local_files_only=True,
    ).eval()
    vision = model.model.visual
    del model.model.language_model
    del model.lm_head
    gc.collect()
    torch.cuda.empty_cache()

    for chunk in pending:
        start = chunk * args.chunk_size
        stop = min(start + args.chunk_size, len(paths))
        chunk_paths = paths[start:stop]
        global_parts = []
        spatial_parts = []
        for offset in tqdm(
            range(0, len(chunk_paths), args.batch_size),
            desc=f"visual chunk {chunk + 1}/{total_chunks}",
        ):
            batch_paths = chunk_paths[offset : offset + args.batch_size]
            global_features, spatial_features = extract_batch(
                batch_paths, processor, vision, args.device, args.pixels
            )
            global_parts.append(global_features)
            spatial_parts.append(spatial_features)
        torch.save(
            {
                "start": start,
                "stop": stop,
                "paths": chunk_paths,
                "global_features": torch.cat(global_parts),
                "spatial_features": torch.cat(spatial_parts),
            },
            output_dir / f"chunk_{chunk:04d}.pt",
        )
        print(f"saved visual chunk {chunk + 1}/{total_chunks}", flush=True)


if __name__ == "__main__":
    main()

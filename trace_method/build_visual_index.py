"""Extract frozen backbone-specific visual-state features for TRACE.

Two manifests are supported. ``calibration`` contains only historical
pseudo-held-out trajectories and their earlier memories. ``full`` contains
the released execution screenshots and the same history-selected top-$K$
trajectory pools used by TRACE. Neither manifest opens target actions while
selecting images.

Each image stores normalized global means from visual layers 8, 16, and 24,
plus the final visual layer and four final-layer spatial quadrant means.  For
Qwen3-VL-family checkpoints, the architecture's native deep-stack mergers are
used.  Qwen3.5 has no deep-stack mergers, so its native unmerged visual-token
states are used instead.  This permits history-only representation selection
for every backbone while keeping all model parameters frozen.
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
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from trace_method.execution_core import (
    execution_memory_frame,
    load_history_trajectories,
    resolve_test_image,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=str(REPO))
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-name", default=None)
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
    train = execution_memory_frame(repo)
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


class VisionFeatureAdapter:
    """Expose comparable intermediate visual states across supported models."""

    layer_indices = (8, 16, 24)

    def __init__(self, vision: torch.nn.Module, model_type: str):
        self.vision = vision
        self.model_type = model_type
        self.captures: dict[int, torch.Tensor] = {}
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        if model_type == "qwen3_5":
            for layer_index in self.layer_indices:
                self.handles.append(
                    vision.blocks[layer_index].register_forward_hook(
                        self._capture(layer_index)
                    )
                )

    def _capture(self, layer_index: int):
        def hook(
            _module: torch.nn.Module,
            _inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            self.captures[layer_index] = output

        return hook

    @property
    def readout(self) -> str:
        if self.model_type == "qwen3_vl":
            return "native deep-stack mergers at layers 8/16/24; final merger"
        if self.model_type == "qwen3_5":
            return "native unmerged visual-token states at layers 8/16/24/final"
        raise ValueError(f"Unsupported model type: {self.model_type}")

    def forward(
        self, pixel_values: torch.Tensor, grids: torch.Tensor
    ) -> tuple[list[torch.Tensor], torch.Tensor, list[int], list[tuple[int, int]]]:
        self.captures.clear()
        outputs = self.vision(pixel_values, grid_thw=grids)
        merge = int(self.vision.spatial_merge_size)
        if self.model_type == "qwen3_vl":
            # Transformers releases expose the same values either as a
            # two-tuple ``(final, deepstack)`` or as named output fields.
            if isinstance(outputs, tuple):
                final, deep = outputs
            else:
                deep = outputs.deepstack_features
                final = outputs.pooler_output
            if deep is None or len(deep) != len(self.layer_indices):
                raise ValueError(
                    "Qwen3-VL checkpoint did not expose three deep-stack states"
                )
            split_sizes = (grids.prod(-1) // merge**2).tolist()
            layouts = [(int(grid[1]) // merge, int(grid[2]) // merge) for grid in grids]
            return list(deep), final, split_sizes, layouts
        if self.model_type == "qwen3_5":
            missing = set(self.layer_indices) - set(self.captures)
            if missing:
                raise ValueError(
                    f"Qwen3.5 visual hooks missed layers: {sorted(missing)}"
                )
            final = outputs.last_hidden_state
            split_sizes = grids.prod(-1).tolist()
            layouts = [(int(grid[1]), int(grid[2])) for grid in grids]
            return (
                [self.captures[index] for index in self.layer_indices],
                final,
                split_sizes,
                layouts,
            )
        raise ValueError(f"Unsupported model type: {self.model_type}")

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def extract_batch(
    paths: list[str],
    processor: AutoProcessor,
    adapter: VisionFeatureAdapter,
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
    vision = adapter.vision
    with torch.inference_mode():
        deep, final, split_sizes, layouts = adapter.forward(
            inputs.pixel_values.to(device=device, dtype=vision.dtype),
            grids,
        )
    final_split = torch.split(final, split_sizes)
    deep_split = [torch.split(layer, split_sizes) for layer in deep]

    global_rows = []
    spatial_rows = []
    for image_index, tokens in enumerate(final_split):
        globals_for_image = [
            normalized_mean(layer[image_index]) for layer in deep_split
        ] + [normalized_mean(tokens)]
        global_rows.append(torch.stack(globals_for_image))

        height, width = layouts[image_index]
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


def validate_gpu(device: str) -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError(
            "Set CUDA_VISIBLE_DEVICES=0; visual extraction may use only GPU 0"
        )
    if device != "cuda:0":
        raise ValueError("--device must be cuda:0")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "Exactly one visible CUDA device (physical GPU 0) is required"
        )


def feature_readout(model_type: str) -> str:
    if model_type == "qwen3_vl":
        return "native deep-stack mergers at layers 8/16/24; final merger"
    if model_type == "qwen3_5":
        return "native unmerged visual-token states at layers 8/16/24/final"
    raise ValueError(f"Unsupported model type: {model_type}")


def main() -> None:
    args = parse_args()
    validate_gpu(args.device)
    repo = Path(args.repo).resolve()
    output_dir = repo / (
        args.output_dir or f"trace_method/results/cache/visual_{args.scope}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    paths = build_manifest(args)
    model_path = Path(args.model_path).resolve()
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    model_type = str(config.model_type)
    manifest = {
        "protocol": (
            "frozen model-specific visual tower; no target action used to choose "
            "images; normalized intermediate/final global means and final spatial "
            "quadrants"
        ),
        "scope": args.scope,
        "model_name": args.model_name or model_path.name,
        "model_path": str(model_path),
        "model_revision": model_path.name,
        "model_type": model_type,
        "feature_readout": feature_readout(model_type),
        "feature_modes": [
            "deep8",
            "deep16",
            "deep24",
            "final",
            "deep_mean",
            "spatial",
            "pyramid",
            "deep_pyramid",
        ],
        "device": args.device,
        "physical_gpu": 0,
        "visible_cuda_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "gpu_name": torch.cuda.get_device_name(0),
        "pixels": args.pixels,
        "trajectory_top_k": args.trajectory_top_k,
        "paths": paths,
        "parameter_training": False,
        "test_answers_accessed": False,
    }
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        compared = (
            "paths",
            "scope",
            "model_path",
            "model_type",
            "feature_readout",
            "pixels",
            "trajectory_top_k",
        )
        mismatches = [key for key in compared if previous.get(key) != manifest[key]]
        if mismatches:
            raise ValueError(f"Existing visual manifest mismatch: {mismatches}")
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
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map={"": args.device},
        attn_implementation="sdpa",
        local_files_only=True,
    ).eval()
    vision = model.model.visual
    adapter = VisionFeatureAdapter(vision, model_type)
    if adapter.readout != manifest["feature_readout"]:
        raise ValueError("Loaded visual adapter does not match its manifest")

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
                batch_paths, processor, adapter, args.device, args.pixels
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
    adapter.close()


if __name__ == "__main__":
    main()

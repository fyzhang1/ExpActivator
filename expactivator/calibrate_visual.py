"""Choose a frozen backbone-specific state representation on histories only."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.stats import binomtest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.eval_execution.utils import eval_action
from expactivator.execution_core import (
    audit_memory_split,
    execution_memory_frame,
    load_history_trajectories,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=str(REPO))
    parser.add_argument("--history-root", default="data/Fingertip")
    parser.add_argument("--trajectory-top-k", type=int, default=5)
    parser.add_argument(
        "--text-cache", default="expactivator/results/execution_history_index_gpu0.pt"
    )
    parser.add_argument(
        "--visual-dir", default="expactivator/results/cache/visual_calibration"
    )
    parser.add_argument(
        "--output", default="expactivator/results/execution_visual_calibration.json"
    )
    return parser.parse_args()


def load_features(
    directory: Path,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict]:
    manifest = json.loads((directory / "manifest.json").read_text())
    result = {}
    for chunk_path in sorted(directory.glob("chunk_*.pt")):
        chunk = torch.load(chunk_path, map_location="cpu", weights_only=False)
        global_features = chunk["global_features"].float().numpy()
        spatial_features = chunk["spatial_features"].float().numpy()
        for index, path in enumerate(chunk["paths"]):
            result[path] = (global_features[index], spatial_features[index])
    if len(result) != len(manifest["paths"]):
        raise ValueError(f"Incomplete cache: {len(result)} != {len(manifest['paths'])}")
    return result, manifest


@lru_cache(maxsize=20_000)
def rgb_descriptor(path: str) -> tuple[np.ndarray, tuple[int, int]]:
    with Image.open(path) as image:
        size = image.size
        value = np.asarray(image.convert("RGB").resize((24, 52)), dtype=np.float32)
    return value / 255.0, size


def unit(value: np.ndarray) -> np.ndarray:
    return value / max(float(np.linalg.norm(value)), 1e-12)


def blocks(feature: tuple[np.ndarray, np.ndarray], mode: str) -> np.ndarray:
    global_features, spatial_features = feature
    if mode == "deep8":
        return global_features[0:1]
    if mode == "deep16":
        return global_features[1:2]
    if mode == "deep24":
        return global_features[2:3]
    if mode == "final":
        return global_features[3:4]
    if mode == "deep_mean":
        return global_features
    if mode == "spatial":
        return spatial_features
    if mode == "pyramid":
        return np.concatenate([global_features[3:4], spatial_features], axis=0)
    if mode == "deep_pyramid":
        return np.concatenate([global_features, spatial_features], axis=0)
    raise KeyError(mode)


def visual_similarity(
    query: tuple[np.ndarray, np.ndarray],
    candidate: tuple[np.ndarray, np.ndarray],
    mode: str,
    center: np.ndarray | None,
) -> float:
    left = blocks(query, mode)
    right = blocks(candidate, mode)
    if center is not None:
        left = np.stack([unit(row) for row in left - center])
        right = np.stack([unit(row) for row in right - center])
    return float((left * right).sum(1).mean())


def flattened_descriptor(
    feature: tuple[np.ndarray, np.ndarray],
    mode: str,
    center: np.ndarray | None,
) -> np.ndarray:
    """Return the exact block-cosine descriptor used by visual_similarity.

    Flattening the blocks and dividing by their count turns the per-block mean
    dot product into one vectorized matrix-vector product.
    """
    value = blocks(feature, mode)
    if center is not None:
        value = np.stack([unit(row) for row in value - center])
    return value.reshape(-1) / np.sqrt(value.shape[0])


def main() -> None:
    args = parse_args()
    repo = Path(args.repo).resolve()
    root = Path(args.history_root).resolve()
    train = execution_memory_frame(repo)
    history_audit = audit_memory_split(train, repo)
    trajectories = load_history_trajectories(train, root)
    text_cache = torch.load(
        repo / args.text_cache, map_location="cpu", weights_only=False
    )
    keys = text_cache["residual_keys"].float()
    features, visual_manifest = load_features(repo / args.visual_dir)
    if visual_manifest.get("scope") != "calibration":
        raise ValueError("Visual calibration requires a calibration-scope cache")
    if visual_manifest.get("test_answers_accessed") is not False:
        raise ValueError("Visual feature construction must not access test answers")

    cases = []
    memory_paths: set[str] = set()
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
        pool = [
            (str(path), action)
            for index in order
            for path, action in trajectories[index]
            if str(path) in features
        ]
        memory_paths.update(path for path, _ in pool)
        # The formal state matcher is used only after the open-app action.
        for target_path, target_action in trajectories[target][1:]:
            if str(target_path) in features:
                cases.append((str(target_path), target_action, pool))

    base_modes = (
        "deep8",
        "deep16",
        "deep24",
        "final",
        "deep_mean",
        "spatial",
        "pyramid",
        "deep_pyramid",
    )
    centers = {}
    for mode in base_modes:
        centers[mode] = np.stack(
            [blocks(features[path], mode) for path in sorted(memory_paths)]
        ).mean(0)
    mode_names = [
        f"{mode}_{suffix}" for mode in base_modes for suffix in ("raw", "centered")
    ]
    hits = {"rgb_mse": 0, **{mode: 0 for mode in mode_names}}
    paired = {mode: {"wins": 0, "losses": 0, "ties": 0} for mode in mode_names}

    # Decode each image once.  The original implementation decoded images in
    # the innermost loop; bounded parallel reads preserve values and remove a
    # large shared-filesystem bottleneck.
    all_paths = sorted(
        {query_path for query_path, _, _ in cases}
        | {path for _, _, pool in cases for path, _ in pool}
    )
    with ThreadPoolExecutor(max_workers=16) as executor:
        rgb_values = dict(zip(all_paths, executor.map(rgb_descriptor, all_paths)))
    case_metadata = []
    for query_path, target_action, pool in cases:
        query_rgb, (width, height) = rgb_values[query_path]
        pool_paths = [item[0] for item in pool]
        pool_actions = [item[1] for item in pool]
        pool_rgb = np.stack([rgb_values[path][0] for path in pool_paths])
        rgb_index = int(
            np.argmin(np.mean((pool_rgb - query_rgb[None, ...]) ** 2, axis=(1, 2, 3)))
        )
        rgb_source = (pool_paths[rgb_index], pool_actions[rgb_index])
        _, rgb_success = eval_action(rgb_source[1], target_action, width, height)
        hits["rgb_mse"] += rgb_success
        case_metadata.append(
            (
                query_path,
                target_action,
                pool_paths,
                pool_actions,
                width,
                height,
                bool(rgb_success),
            )
        )

    # Materialize one representation at a time.  The largest variant has eight
    # 4096-D blocks, so this keeps peak host memory bounded.
    for name in mode_names:
        mode, suffix = name.rsplit("_", 1)
        center = centers[mode] if suffix == "centered" else None
        latent_values = {
            path: flattened_descriptor(features[path], mode, center)
            for path in all_paths
        }
        for (
            query_path,
            target_action,
            pool_paths,
            pool_actions,
            width,
            height,
            rgb_success,
        ) in case_metadata:
            query_latent = latent_values[query_path]
            pool_latent = np.stack([latent_values[path] for path in pool_paths])
            source_index = int(np.argmax(pool_latent @ query_latent))
            source = (pool_paths[source_index], pool_actions[source_index])
            _, success = eval_action(source[1], target_action, width, height)
            hits[name] += success
            if success and not rgb_success:
                paired[name]["wins"] += 1
            elif rgb_success and not success:
                paired[name]["losses"] += 1
            else:
                paired[name]["ties"] += 1
        print(f"evaluated {name} on {len(cases)} held-out actions", flush=True)

    accuracy = {name: value / len(cases) for name, value in hits.items()}
    selected = min(mode_names, key=lambda name: (-accuracy[name], name))
    selected_base, selected_suffix = selected.rsplit("_", 1)
    selected_center = (
        centers[selected_base].tolist() if selected_suffix == "centered" else None
    )
    result = {
        "protocol": (
            "history memory only; final readable trajectory per user held out; "
            "first/open-app state excluded; exact same top-"
            f"{args.trajectory_top_k} goal pools for all modes"
        ),
        "history_audit": history_audit,
        "test_data_accessed": history_audit["test_data_accessed"],
        "validation_actions": len(cases),
        "memory_images": len(memory_paths),
        "accuracy": accuracy,
        "paired_vs_rgb": paired,
        "selected_mode": selected,
        "selected_center": selected_center,
        "selected_delta_vs_rgb": accuracy[selected] - accuracy["rgb_mse"],
        "selected_exact_mcnemar_p": float(
            binomtest(
                paired[selected]["wins"],
                paired[selected]["wins"] + paired[selected]["losses"],
                0.5,
            ).pvalue
        ),
        "visual_model": {
            key: visual_manifest.get(key)
            for key in (
                "model_name",
                "model_path",
                "model_revision",
                "model_type",
                "feature_readout",
                "pixels",
            )
        },
    }
    output = repo / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {
                "visual_model": result["visual_model"],
                "validation_actions": result["validation_actions"],
                "selected_mode": selected,
                "selected_accuracy": accuracy[selected],
                "rgb_mse_accuracy": accuracy["rgb_mse"],
                "selected_delta_vs_rgb": result["selected_delta_vs_rgb"],
                "selected_exact_mcnemar_p": result["selected_exact_mcnemar_p"],
            },
            indent=2,
        )
    )
    print(f"saved {output}")


if __name__ == "__main__":
    main()

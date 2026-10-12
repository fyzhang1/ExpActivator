from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from expactivator.evaluate_execution import load_selected_features


def write_cache(directory: Path) -> tuple[torch.Tensor, torch.Tensor]:
    directory.mkdir()
    paths = ["first.png", "second.png"]
    (directory / "manifest.json").write_text(
        json.dumps({"paths": paths, "scope": "full"})
    )
    global_features = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, -1.0]],
            [[0.0, 1.0], [1.0, 0.0], [-1.0, 1.0], [1.0, 1.0]],
        ]
    )
    spatial_features = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],
            [[0.0, 1.0], [1.0, 0.0], [0.0, -1.0], [-1.0, 0.0]],
        ]
    )
    torch.save(
        {
            "paths": paths,
            "global_features": global_features,
            "spatial_features": spatial_features,
        },
        directory / "chunk_0000.pt",
    )
    return global_features, spatial_features


def test_load_raw_multiblock_descriptor(tmp_path: Path) -> None:
    global_features, spatial_features = write_cache(tmp_path / "cache")
    features, _ = load_selected_features(tmp_path / "cache", "deep_pyramid_raw", None)
    expected = torch.cat([global_features[0], spatial_features[0]]).numpy()
    np.testing.assert_allclose(features["first.png"], expected.reshape(-1) / np.sqrt(8))


def test_load_centered_descriptor_normalizes_each_block(tmp_path: Path) -> None:
    global_features, _ = write_cache(tmp_path / "cache")
    center = [[0.5, 0.5]]
    features, _ = load_selected_features(tmp_path / "cache", "deep8_centered", center)
    expected = global_features[0, 0].numpy() - np.asarray(center[0])
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(features["first.png"], expected)


def test_center_presence_must_match_mode(tmp_path: Path) -> None:
    write_cache(tmp_path / "cache")
    try:
        load_selected_features(tmp_path / "cache", "deep8_raw", [[0.0, 0.0]])
    except ValueError as error:
        assert "inconsistent" in str(error)
    else:
        raise AssertionError("raw mode unexpectedly accepted a centering vector")

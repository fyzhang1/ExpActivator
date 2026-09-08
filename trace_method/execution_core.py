"""Shared, prediction-agnostic utilities for TRACE execution evaluation."""

from __future__ import annotations

import json
import re
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.eval_execution.utils import compute_CSR, convert_action_qwen

CLICK_RE = re.compile(r"coordinates=\((\d+),\s*(\d+)\)")
DIRECTION_RE = re.compile(r"direction='([^']+)'", re.DOTALL)
TEXT_RE = re.compile(r"text='(.*?)'", re.DOTALL)


def load_outputs(path: Path) -> list[dict]:
    value = json.loads(path.read_text())
    return value["outputs"] if isinstance(value, dict) else value


def resolve_test_image(path: str, history_root: Path) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate
    normalized = path.replace("\\", "/")
    marker = "data/Fingertip/"
    if marker in normalized:
        candidate = history_root / normalized.split(marker, 1)[1]
        if candidate.exists():
            return candidate
    raise FileNotFoundError(path)


def normalize_action_string(action: str) -> str:
    return CLICK_RE.sub(
        lambda match: f"coordinates=({match.group(1)},{match.group(2)})", action
    )


def load_history_trajectories(
    train: pd.DataFrame, history_root: Path
) -> list[list[tuple[Path, str]]]:
    trajectories = []
    for row in train.itertuples():
        episode = history_root / str(row.user_id) / str(row.user_id) / str(row.time)
        action_file = episode / "action.jsonl"
        states: list[tuple[Path, str]] = []
        if action_file.exists():
            for line in action_file.read_text().splitlines():
                try:
                    entries = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for image_name, action in entries.items():
                    states.append(
                        (
                            episode / f"Screenshots_{row.time}" / image_name,
                            normalize_action_string(str(action)),
                        )
                    )
        trajectories.append(states)
    return trajectories


def make_image_reader(width: int, height: int):
    @lru_cache(maxsize=100_000)
    def read(path: str) -> tuple[np.ndarray | None, tuple[int, int]]:
        try:
            with Image.open(path) as image:
                size = image.size
                image.draft("RGB", (width * 2, height * 2))
                thumbnail = np.asarray(
                    image.convert("RGB").resize((width, height)), dtype=np.float32
                )
                return thumbnail / 255.0, size
        except (FileNotFoundError, OSError):
            return None, (1080, 2340)

    return read


def historical_action_to_qwen(
    action: str, historical_image_size: tuple[int, int]
) -> dict:
    """Translate a factual recorded action without consulting the target."""
    width, height = historical_image_size
    if action.startswith(("click", "long_click")):
        match = CLICK_RE.search(action)
        if not match:
            return {"action": "error"}
        x, y = map(int, match.groups())
        action_type = "long_click" if action.startswith("long_click") else "click"
        return {
            "action": action_type,
            "coordinate": [
                round(x / max(width, 1) * 999),
                round(y / max(height, 1) * 999),
            ],
        }
    if action.startswith("scroll"):
        match = DIRECTION_RE.search(action)
        direction = match.group(1) if match else ""
        endpoints = {
            "up": ([500, 800], [500, 200]),
            "down": ([500, 200], [500, 800]),
            "left": ([800, 500], [200, 500]),
            "right": ([200, 500], [800, 500]),
        }
        if direction not in endpoints:
            return {"action": "error"}
        start, end = endpoints[direction]
        return {"action": "scroll", "coordinate": start, "coordinate2": end}
    if action.startswith("type"):
        match = TEXT_RE.search(action)
        return {"action": "type", "text": match.group(1) if match else ""}
    if action.startswith("finished"):
        return {"action": "terminate"}
    for action_type in ("wait", "press_back", "press_home", "press_recent"):
        if action.startswith(action_type):
            return {"action": action_type}
    if action.startswith("open_app"):
        match = re.search(r"app_name='(.*?)'", action)
        return {"action": "open_app", "app_name": match.group(1) if match else ""}
    return {"action": "error"}


def summarize(outputs: list[dict]) -> dict[str, float | int]:
    starts = [row for row in outputs if row["episode_id"].endswith("_0")]
    nonfirst = [row for row in outputs if not row["episode_id"].endswith("_0")]
    return {
        "n_steps": len(outputs),
        "n_episodes": len(starts),
        "type_accuracy": sum(row["type_acc"] for row in outputs) / len(outputs),
        "success_rate": sum(row["sr"] for row in outputs) / len(outputs),
        "csr": compute_CSR(outputs),
        "first_step_success_rate": sum(row["sr"] for row in starts) / len(starts),
        "nonfirst_success_rate": sum(row["sr"] for row in nonfirst) / len(nonfirst),
    }


def scored_row(step: dict, output: dict, width: int, height: int) -> dict:
    type_acc, success = convert_action_qwen(output, step["answer"], width, height)
    return {
        "episode_id": step["id"],
        "answer": step["answer"],
        "output": output,
        "type_acc": type_acc,
        "sr": success,
        "result": f"Type: {type_acc}, SR: {success}",
    }

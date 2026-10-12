"""Shared, prediction-agnostic utilities for ExpActivator execution evaluation."""

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


def _json_recorded_action(
    action: str, historical_image_size: tuple[int, int]
) -> dict | None:
    """Recover a structured action from fenced JSON history records."""
    if not action.lstrip().startswith("```"):
        return None
    values = []
    for candidate in re.findall(r"\{[^{}]*\}", action, re.DOTALL):
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            values.append(value)
    if not values:
        return None
    value = values[-1]
    action_type = str(value.get("action", value.get("action_type", ""))).lower()
    action_type = {
        "open": "open_app",
        "long_press": "long_click",
        "finished": "terminate",
        "finish": "terminate",
        "swipe": "scroll",
    }.get(action_type, action_type)
    width, height = historical_image_size

    def normalized_coordinate(*names: str) -> list[int] | None:
        coordinate = next((value.get(name) for name in names if name in value), None)
        if not isinstance(coordinate, (list, tuple)) or len(coordinate) < 2:
            return None
        return [
            round(float(coordinate[0]) / max(width, 1) * 999),
            round(float(coordinate[1]) / max(height, 1) * 999),
        ]

    if action_type in {"click", "long_click"}:
        coordinate = normalized_coordinate("coordinate", "coordinates", "point")
        return (
            {"action": action_type, "coordinate": coordinate}
            if coordinate is not None
            else {"action": "error"}
        )
    if action_type == "type":
        return {"action": "type", "text": str(value.get("text", value.get("content", "")))}
    if action_type == "open_app":
        return {"action": "open_app", "app_name": str(value.get("app_name", ""))}
    if action_type == "scroll":
        coordinate = normalized_coordinate("coordinate", "coordinates", "point")
        coordinate2 = normalized_coordinate("coordinate2", "point2")
        if coordinate is not None and coordinate2 is not None:
            return {
                "action": "scroll",
                "coordinate": coordinate,
                "coordinate2": coordinate2,
            }
        endpoints = {
            "up": ([500, 800], [500, 200]),
            "down": ([500, 200], [500, 800]),
            "left": ([800, 500], [200, 500]),
            "right": ([200, 500], [800, 500]),
        }
        direction = str(value.get("direction", "")).lower()
        if direction in endpoints:
            start, end = endpoints[direction]
            return {"action": "scroll", "coordinate": start, "coordinate2": end}
        return {"action": "error"}
    if action_type in {"wait", "press_back", "press_home", "press_recent", "terminate"}:
        return {"action": action_type}
    return {"action": "error"}


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
    json_action = _json_recorded_action(action, historical_image_size)
    if json_action is not None:
        return json_action
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


# --- Official history-memory protocol -------------------------------------
#
# The released benchmark exposes the chronological first 80% of each user's
# records as personal memory; all 775 execution-test episodes are drawn from
# the held-out final 20%.  ``samples/train_set.csv`` is NOT that split: it
# retains 490 of the test episodes together with their explicit original
# instructions, so indexing it leaks first-action answers and gives the method
# a memory the paper baselines never see.  Every execution-branch script must
# therefore construct its memory through ``execution_memory_frame`` so that
# ExpActivator and the four paper baselines read byte-identical histories.

HISTORY_CSV = "samples/total.csv"
TEST_EPISODE_PATH = "data/personal_execution/personal_samples.json"


def history_key(user_id: object, time: object) -> str:
    """Return the dataset-stable episode identity shared by every split."""
    return f"{int(user_id)}_{time}"


def chronological_history(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the official first-80%-per-user chronological history split."""
    groups = []
    for _, group in frame.groupby("user_id", sort=True):
        ordered = group.sort_values("time", kind="stable")
        groups.append(ordered.iloc[: int(len(ordered) * 0.8)])
    return pd.concat(groups).reset_index(drop=True)


def execution_memory_frame(repo: Path) -> pd.DataFrame:
    """Build the protocol memory shared by ExpActivator and the paper baselines."""
    total = pd.read_csv(Path(repo) / HISTORY_CSV)
    if total.duplicated(["user_id", "time"]).any():
        raise ValueError("total.csv does not have unique (user_id, time) records")
    return chronological_history(total)


def held_out_episode_keys(repo: Path, path: str = TEST_EPISODE_PATH) -> set[str]:
    episodes = json.loads((Path(repo) / path).read_text())
    return {history_key(episode["user_id"], episode["time"]) for episode in episodes}


def audit_memory_split(
    frame: pd.DataFrame, repo: Path, path: str = TEST_EPISODE_PATH
) -> dict:
    """Verify by construction that no test episode is inside the memory.

    This replaces the previous self-declared ``test_data_accessed`` constants.
    The returned record is computed from the data, and a non-empty overlap is
    a hard failure rather than a logged counter.
    """
    memory_keys = {
        history_key(row.user_id, row.time) for row in frame.itertuples()
    }
    overlap = sorted(memory_keys & held_out_episode_keys(repo, path))
    if overlap:
        raise ValueError(
            f"{len(overlap)} execution-test episodes are inside the history memory "
            f"(first: {overlap[:3]}); the memory must be the chronological "
            "first-80%-per-user split of samples/total.csv"
        )
    return {
        "history_csv": HISTORY_CSV,
        "history_protocol": "chronological first 80% per user of total.csv",
        "memory_rows": len(frame),
        "test_episodes_in_memory": 0,
        "test_data_accessed": False,
    }


def load_app_labels(repo: Path, label_path: str, frame: pd.DataFrame) -> list[str]:
    """Read compiled app labels by episode identity, not by row position.

    Keying on ``(user_id, time)`` lets one compiled label file serve any
    history split, so changing the split never silently mis-aligns labels.
    """
    rows = json.loads((Path(repo) / label_path).read_text())["rows"]
    lookup: dict[str, str] = {}
    for row in rows:
        key = history_key(row["user_id"], row["time"])
        if key in lookup and lookup[key] != str(row["app_name"]):
            raise ValueError(f"Conflicting compiled app labels for {key}")
        lookup[key] = str(row["app_name"])
    keys = [history_key(row.user_id, row.time) for row in frame.itertuples()]
    missing = [key for key in keys if key not in lookup]
    if missing:
        raise ValueError(
            f"{len(missing)} memory records have no compiled app label "
            f"(first: {missing[:3]}).  Run compile_history_app_labels.py with "
            f"--reuse {label_path} to compile only the missing records."
        )
    return [lookup[key] for key in keys]

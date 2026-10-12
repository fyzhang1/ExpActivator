"""Shared paired statistics for ExpActivator execution experiments."""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.eval_execution.utils import make_weights


def csr_without_print(outputs: list[dict]) -> float:
    episodes: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for row in outputs:
        episode, position = row["episode_id"].rsplit("_", 1)
        episodes[episode].append((int(position), int(row["sr"])))
    numerator = denominator = 0.0
    for rows in episodes.values():
        rows.sort()
        weights = make_weights(len(rows))
        numerator += float(
            np.dot(weights, np.asarray([success for _, success in rows]))
        )
        denominator += float(weights.sum())
    return numerator / denominator


def metrics(outputs: list[dict]) -> dict[str, float | int]:
    starts = [row for row in outputs if row["episode_id"].endswith("_0")]
    nonfirst = [row for row in outputs if not row["episode_id"].endswith("_0")]
    return {
        "n_steps": len(outputs),
        "n_episodes": len(starts),
        "type_accuracy": sum(row["type_acc"] for row in outputs) / len(outputs),
        "success_rate": sum(row["sr"] for row in outputs) / len(outputs),
        "csr": csr_without_print(outputs),
        "first_step_success_rate": sum(row["sr"] for row in starts) / len(starts),
        "nonfirst_success_rate": sum(row["sr"] for row in nonfirst) / len(nonfirst),
    }


def paired_binary(
    reference: np.ndarray,
    method: np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> dict:
    delta = method.astype(float) - reference.astype(float)
    wins = int(((reference == 0) & (method == 1)).sum())
    losses = int(((reference == 1) & (method == 0)).sum())
    samples = []
    remaining = n_bootstrap
    while remaining:
        size = min(remaining, 1_000)
        indices = rng.integers(0, len(delta), size=(size, len(delta)))
        samples.append(delta[indices].mean(1))
        remaining -= size
    bootstrap = np.concatenate(samples)
    discordant = wins + losses
    return {
        "n": len(delta),
        "reference": float(reference.mean()),
        "method": float(method.mean()),
        "delta": float(delta.mean()),
        "delta_bootstrap_95ci": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
        "wins": wins,
        "losses": losses,
        "ties": int(len(delta) - wins - losses),
        "exact_mcnemar_p": (
            float(binomtest(wins, discordant, 0.5).pvalue) if discordant else 1.0
        ),
    }


def full_metric_bootstrap(
    reference: list[dict],
    method: list[dict],
    rng: np.random.Generator,
    n_bootstrap: int,
) -> dict:
    grouped: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    for reference_row, method_row in zip(reference, method):
        if reference_row["episode_id"] != method_row["episode_id"]:
            raise ValueError("Comparison rows are not paired")
        episode = reference_row["episode_id"].rsplit("_", 1)[0]
        grouped[episode].append((reference_row, method_row))
    values = []
    for rows in grouped.values():
        rows.sort(key=lambda pair: int(pair[0]["episode_id"].rsplit("_", 1)[1]))
        weights = make_weights(len(rows))
        values.append(
            (
                len(rows),
                sum(m["sr"] - r["sr"] for r, m in rows),
                sum(m["type_acc"] - r["type_acc"] for r, m in rows),
                sum(
                    weights[index] * (m["sr"] - r["sr"])
                    for index, (r, m) in enumerate(rows)
                ),
            )
        )
    values = np.asarray(values, dtype=float)
    chunks = []
    remaining = n_bootstrap
    while remaining:
        size = min(remaining, 1_000)
        indices = rng.integers(0, len(values), size=(size, len(values)))
        sample = values[indices]
        denominator = sample[:, :, 0].sum(1)
        chunks.append(
            np.stack(
                [
                    sample[:, :, 1].sum(1) / denominator,
                    sample[:, :, 2].sum(1) / denominator,
                    sample[:, :, 3].sum(1) / denominator,
                ],
                axis=1,
            )
        )
        remaining -= size
    bootstrap = np.concatenate(chunks)
    names = ("success_rate", "type_accuracy", "csr")
    return {
        name: {
            "delta": float(values[:, index + 1].sum() / values[:, 0].sum()),
            "delta_bootstrap_95ci": np.quantile(
                bootstrap[:, index], [0.025, 0.975]
            ).tolist(),
        }
        for index, name in enumerate(names)
    }


def action_type(action: str) -> str:
    match = re.match(r"([A-Za-z_]+)", action)
    return match.group(1) if match else "malformed"


def action_breakdown(
    test: list[dict], reference: list[dict], method: list[dict]
) -> dict:
    grouped: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for target, reference_row, method_row in zip(test, reference, method):
        if target["id"].endswith("_0"):
            continue
        values = grouped[action_type(target["answer"])]
        values[0] += 1
        values[1] += reference_row["sr"]
        values[2] += method_row["sr"]
    return {
        name: {
            "n": values[0],
            "reference": values[1] / values[0],
            "method": values[2] / values[0],
            "delta": (values[2] - values[1]) / values[0],
        }
        for name, values in sorted(grouped.items(), key=lambda item: -item[1][0])
    }

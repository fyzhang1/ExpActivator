"""History-size stratification and the entry/non-entry decomposition.

Two questions about where personalized-execution gains come from, both
answered offline from the sealed per-step runs without further inference:

1. Does LEAF need a large personal history?  Users are grouped into quartiles
   by the size of their memory split (the official chronological first 80%),
   and every condition is scored inside each quartile.
2. Does episode-level memory survive past the first action?  Step success is
   reported separately for entry and non-entry steps.

All intervals use the repository's paired user-cluster bootstrap.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

os.environ.setdefault("USE_TF", "0")
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from trace_method.analyze_state_conditioning_experiment import by_id, paired_cluster
from trace_method.execution_core import execution_memory_frame

RUNS = {
    "Base": "trace_method/results/paper_baselines/qwen3_vl_8b_base_gpu0.json",
    "Recent": "trace_method/results/paper_baselines/qwen3_vl_8b_recent_gpu0.json",
    "Retrieve": "trace_method/results/paper_baselines/qwen3_vl_8b_retrieve_gpu0.json",
    "LLM-UM": "trace_method/results/paper_baselines/qwen3_vl_8b_llm_um_gpu0.json",
    "HIM-Agent": "trace_method/results/paper_baselines/qwen3_vl_8b_him_agent_gpu0.json",
    "LEAF": "trace_method/results/grounded/qwen3_vl_8b_grounded_policy_gpu0.json",
}
BASELINES = [name for name in RUNS if name != "LEAF"]
OUT = "trace_method/results/leaf_context_cost/history_size.json"
SAMPLES = 20000
SEED = 20260916


def load_run(relative: str) -> list[dict]:
    payload = json.loads((REPO / relative).read_text())
    rows = payload["outputs"] if isinstance(payload, dict) else payload
    return [
        {"episode_id": row["episode_id"], "sr": int(row["sr"]),
         "type_acc": int(row["type_acc"])}
        for row in rows
    ]


def step_metadata(rows: list[dict]) -> list[dict]:
    metadata = []
    for row in rows:
        episode, index = row["episode_id"].rsplit("_", 1)
        metadata.append({
            "step_id": row["episode_id"],
            "episode_id": episode,
            "user_id": int(episode.split("_")[0]),
            "position": int(index),
        })
    return metadata


def rate(runs: dict, name: str, metadata: list[dict]) -> float:
    return float(
        np.mean([runs[name][row["step_id"]]["sr"] for row in metadata]) * 100
    )


def main() -> None:
    runs = {name: by_id(load_run(relative)) for name, relative in RUNS.items()}
    order = list(runs["LEAF"])
    for name, run in runs.items():
        if list(run) != order:
            raise ValueError(f"{name} step order differs from LEAF")
    metadata = step_metadata(load_run(RUNS["LEAF"]))

    memory = execution_memory_frame(REPO)
    history_size = memory.groupby("user_id").size()
    users = sorted({row["user_id"] for row in metadata})
    sizes = history_size.reindex(users)
    if sizes.isna().any():
        raise ValueError("An evaluated user has no memory-split records")
    quartile = dict(zip(users, pd.qcut(sizes, 4, labels=["Q1", "Q2", "Q3", "Q4"])))

    report = {
        "protocol": {
            "name": "LEAF history-size stratification and entry decomposition",
            "backbone": "Qwen3-VL-8B",
            "runs": RUNS,
            "n_users": len(users),
            "n_steps": len(metadata),
            "history_size": "records in the chronological first-80% memory split",
            "bootstrap_samples": SAMPLES,
            "seed": SEED,
            "cluster": "user_id",
        },
        "history_records": {
            "min": int(sizes.min()),
            "median": float(sizes.median()),
            "max": int(sizes.max()),
        },
        "entry_decomposition": {},
        "quartiles": {},
        "correlation": {},
    }

    # Entry versus non-entry steps, with paired intervals against Base and LEAF.
    for slice_name, keep in (
        ("all_steps", lambda row: True),
        ("entry_steps", lambda row: row["position"] == 0),
        ("non_entry_steps", lambda row: row["position"] > 0),
    ):
        subset = [row for row in metadata if keep(row)]
        entry = {"n_steps": len(subset), "ssr": {}, "vs_base": {}, "leaf_vs": {}}
        for name in RUNS:
            entry["ssr"][name] = rate(runs, name, subset)
        for name in RUNS:
            if name == "Base":
                continue
            paired = paired_cluster(
                runs["Base"], runs[name], subset, samples=SAMPLES, seed=SEED
            )
            entry["vs_base"][name] = {
                "delta": paired["delta"] * 100,
                "ci95": [value * 100 for value in paired["delta_ci95"]],
            }
        for name in BASELINES:
            paired = paired_cluster(
                runs[name], runs["LEAF"], subset, samples=SAMPLES, seed=SEED
            )
            entry["leaf_vs"][name] = {
                "delta": paired["delta"] * 100,
                "ci95": [value * 100 for value in paired["delta_ci95"]],
            }
        report["entry_decomposition"][slice_name] = entry

    # History-size quartiles.
    for slice_name, keep in (
        ("all_steps", lambda row: True),
        ("non_entry_steps", lambda row: row["position"] > 0),
    ):
        buckets = []
        for label in ["Q1", "Q2", "Q3", "Q4", "All"]:
            members = [
                user for user in users
                if label == "All" or quartile[user] == label
            ]
            subset = [
                row for row in metadata
                if keep(row) and row["user_id"] in set(members)
            ]
            bucket = {
                "label": label,
                "n_users": len(members),
                "n_steps": len(subset),
                "history_records_min": int(sizes.loc[members].min()),
                "history_records_max": int(sizes.loc[members].max()),
                "history_records_median": float(sizes.loc[members].median()),
                "ssr": {name: rate(runs, name, subset) for name in RUNS},
                "leaf_vs": {},
            }
            for name in BASELINES:
                paired = paired_cluster(
                    runs[name], runs["LEAF"], subset, samples=SAMPLES, seed=SEED
                )
                bucket["leaf_vs"][name] = {
                    "delta": paired["delta"] * 100,
                    "ci95": [value * 100 for value in paired["delta_ci95"]],
                }
            strongest = min(bucket["leaf_vs"], key=lambda k: bucket["leaf_vs"][k]["delta"])
            bucket["strongest_baseline"] = strongest
            bucket["gain_vs_strongest"] = bucket["leaf_vs"][strongest]
            buckets.append(bucket)
        report["quartiles"][slice_name] = buckets

        # Per-user gain against that user's own best baseline, ranked against
        # history size.  Using each user's best baseline favors the baselines.
        gains, records = [], []
        for user in users:
            subset = [row for row in metadata if keep(row) and row["user_id"] == user]
            if not subset:
                continue
            best = max(rate(runs, name, subset) for name in BASELINES)
            gains.append(rate(runs, "LEAF", subset) - best)
            records.append(float(sizes.loc[user]))
        gain = pd.Series(gains)
        size = pd.Series(records)
        report["correlation"][slice_name] = {
            "n_users": int(len(gain)),
            "spearman_gain_vs_history": float(
                np.corrcoef(gain.rank(), size.rank())[0, 1]
            ),
            "pearson_gain_vs_history": float(np.corrcoef(gain, size)[0, 1]),
            "note": "each user's own strongest baseline is used, which favors "
                    "the baselines",
        }

    destination = REPO / OUT
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2))

    for slice_name, entry in report["entry_decomposition"].items():
        print(f"\n=== {slice_name} ({entry['n_steps']} steps) ===")
        for name in RUNS:
            line = f"  {name:10s} SSR={entry['ssr'][name]:6.2f}"
            if name in entry["vs_base"]:
                d = entry["vs_base"][name]
                line += (f"  vs Base {d['delta']:+6.2f} "
                         f"[{d['ci95'][0]:+6.2f}, {d['ci95'][1]:+6.2f}]")
            print(line)
    for slice_name, buckets in report["quartiles"].items():
        print(f"\n=== quartiles, {slice_name} ===")
        for bucket in buckets:
            gain = bucket["gain_vs_strongest"]
            print(
                f"  {bucket['label']:4s} users={bucket['n_users']:3d} "
                f"steps={bucket['n_steps']:5d} "
                f"hist={bucket['history_records_min']}-{bucket['history_records_max']:<4d} "
                f"best={bucket['strongest_baseline']:10s} "
                f"{bucket['ssr'][bucket['strongest_baseline']]:6.2f} "
                f"LEAF={bucket['ssr']['LEAF']:6.2f} "
                f"gain={gain['delta']:+5.2f} "
                f"[{gain['ci95'][0]:+5.2f}, {gain['ci95'][1]:+5.2f}]"
            )
    print("\ncorrelation:", json.dumps(report["correlation"], indent=2))
    print(f"\nwrote {destination}")


if __name__ == "__main__":
    main()

"""Sensitivity of task-level activation to its six fixed constants.

Task-level activation declares a routine threshold (eta), a minimum total
support (n_min), a minimum same-scenario support (n_ctx), a temporal bandwidth
(sigma), an activation threshold on weighted support (delta), and a frequency
normalization exponent (gamma).  This script re-runs the activation rule over
the sealed routine index for grids of these constants and reports how the
decision quality moves, so the reported operating point can be read against the
surface it sits on rather than on its own.

Only activation is re-run: no suggestion is generated and no model is called.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

os.environ.setdefault("USE_TF", "0")
import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from expactivator.build_leap_refresh_ablation import sha
from expactivator.evaluate_proactive import circular_hour_distance, hour

OUT = "expactivator/results/leaf_proactive_sensitivity/sensitivity.json"
ETAS = [0.88, 0.90, 0.92, 0.93, 0.94, 0.95, 0.96, 0.98]
N_MINS = [4, 6, 8, 10, 12, 16, 20, 24]
SIGMAS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
N_CTXS = [1, 2, 3, 4, 5, 6, 8, 10]
DELTAS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.0]
GAMMAS = [0.0, 0.5, 1.0, 1.5]


def load_state():
    index_path = REPO / "expactivator/results/proactive_history_index_gpu0.pt"
    production_path = REPO / "expactivator/results/proactive_predictions.json"
    production = json.loads(production_path.read_text())
    cache = torch.load(index_path, map_location="cpu", weights_only=False)
    if sha(index_path) != production["audit"]["index_sha256"]:
        raise ValueError("Routine index does not match the sealed production run")

    total = pd.read_csv(REPO / "samples/total.csv").reset_index(drop=True)
    memory = (total.iloc[cache["memory_indices"]].reset_index()
              .rename(columns={"index": "source_index"}))
    keys = cache["keys"].float().numpy()

    queries = []
    for name in ("proactive_samples.json", "proactive_false_samples.json"):
        for item in json.loads(
            (REPO / "data/proactive_suggestion" / name).read_text()
        ):
            queries.append({"user_id": int(item["user_id"]),
                            "time": str(item["time"]),
                            "scenario": str(item["scenario"])})
    if len(queries) != 315:
        raise ValueError(f"Expected 315 proactive states, found {len(queries)}")

    labels = np.array([row["split_label"] for row in production["predictions"]],
                      dtype=float)
    for query, row in zip(queries, production["predictions"]):
        if (query["user_id"], query["time"], query["scenario"]) != (
            row["user_id"], row["time"], row["scenario"]
        ):
            raise ValueError("Query order differs from the sealed predictions")

    grouped = {int(u): group for u, group in memory.groupby("user_id", sort=True)}
    similarity = {u: keys[group.index] @ keys[group.index].T
                  for u, group in grouped.items()}
    return production, queries, labels, grouped, similarity


def prepare(queries, grouped, similarity, eta):
    """Per-query quantities that depend on the routine threshold only."""
    prepared = []
    for query in queries:
        local = grouped[query["user_id"]]
        scores = similarity[query["user_id"]]
        valid = local.time.astype(str).to_numpy() != query["time"]
        neighbors = (scores >= eta) & valid[None, :]
        scenario = local.scenario.astype(str).to_numpy() == query["scenario"]
        state_neighbors = neighbors & scenario[None, :]
        hours = np.array([hour(t) for t in local.time], dtype=np.float32)
        distance = circular_hour_distance(hours, hour(query["time"]))
        prepared.append({
            "support": neighbors.sum(1),
            "scenario_support": state_neighbors.sum(1),
            "state_neighbors": state_neighbors.astype(np.float32),
            "distance": distance,
        })
    return prepared


def scores_for(prepared, sigma, n_min, n_ctx):
    trigger = np.empty(len(prepared))
    masses = []
    for k, item in enumerate(prepared):
        weights = np.exp(-(item["distance"] ** 2) / (2 * sigma ** 2))
        mass = item["state_neighbors"] @ weights
        stable = (item["support"] >= n_min) & (item["scenario_support"] >= n_ctx)
        trigger[k] = float(mass[stable].max()) if stable.any() else -1.0
        masses.append(mass)
    return trigger, masses


def auroc(labels, scores):
    positive = scores[labels == 1][:, None]
    negative = scores[labels == 0][None, :]
    wins = (positive > negative).sum() + 0.5 * (positive == negative).sum()
    return float(wins / (positive.size * negative.size))


def operating_point(labels, scores, delta):
    emitted = scores >= delta
    recall = float(emitted[labels == 1].mean())
    false_alarm = float(emitted[labels == 0].mean())
    return {
        "recall": recall * 100,
        "false_alarm_rate": false_alarm * 100,
        "balanced_accuracy": (recall + 1 - false_alarm) / 2 * 100,
        "trigger_rate": float(emitted.mean()) * 100,
    }


def summarize(labels, scores, delta):
    return {"auroc": auroc(labels, scores) * 100,
            **operating_point(labels, scores, delta),
            "no_eligible_routine_rate": float((scores < 0).mean()) * 100}


def main() -> None:
    production, queries, labels, grouped, similarity = load_state()
    config = production["config"]
    eta0 = config["semantic_threshold"]
    n_min0 = config["minimum_support"]
    n_ctx0 = config["minimum_scenario_support"]
    sigma0 = config["time_sigma_hours"]
    delta0 = config["trigger_mass"]

    cache = {}

    def prepared_for(eta):
        if eta not in cache:
            cache[eta] = prepare(queries, grouped, similarity, eta)
        return cache[eta]

    # Reproduce the sealed operating point before sweeping anything.
    base_scores, _ = scores_for(prepared_for(eta0), sigma0, n_min0, n_ctx0)
    reference = summarize(labels, base_scores, delta0)
    sealed = {"auroc": 80.37, "balanced_accuracy": 72.66, "recall": 82.33,
              "false_alarm_rate": 37.00}
    for key, expected in sealed.items():
        if abs(reference[key] - expected) > 0.02:
            raise ValueError(
                f"Reimplementation disagrees with the sealed run on {key}: "
                f"{reference[key]:.2f} vs {expected:.2f}"
            )

    report = {
        "protocol": {
            "name": "Task-level activation threshold sensitivity",
            "states": {"total": len(queries), "positive": int(labels.sum()),
                       "negative": int((1 - labels).sum())},
            "config": config,
            "retrieval_encoder": "sealed routine index",
            "note": "activation only; no suggestion is generated and no model "
                    "is called",
        },
        "reference": reference,
        "eta_by_n_min": [],
        "sigma": [],
        "n_ctx": [],
        "delta": [],
        "gamma": [],
    }

    for eta in ETAS:
        prepared = prepared_for(eta)
        for n_min in N_MINS:
            scores, _ = scores_for(prepared, sigma0, n_min, n_ctx0)
            report["eta_by_n_min"].append(
                {"eta": eta, "n_min": n_min,
                 **summarize(labels, scores, delta0)}
            )
        print(f"eta={eta:.2f} done", flush=True)

    prepared = prepared_for(eta0)
    for sigma in SIGMAS:
        scores, _ = scores_for(prepared, sigma, n_min0, n_ctx0)
        report["sigma"].append({"sigma_hours": sigma,
                                **summarize(labels, scores, delta0)})
    for n_ctx in N_CTXS:
        scores, _ = scores_for(prepared, sigma0, n_min0, n_ctx)
        report["n_ctx"].append({"n_ctx": n_ctx,
                                **summarize(labels, scores, delta0)})
    for delta in DELTAS:
        report["delta"].append({"delta": delta,
                                **summarize(labels, base_scores, delta)})

    # Frequency normalization re-ranks inside the eligible set and therefore
    # cannot move any activation decision; this records that directly.
    for gamma in GAMMAS:
        changed = 0
        for item, mass in zip(prepared, scores_for(prepared, sigma0, n_min0,
                                                   n_ctx0)[1]):
            stable = ((item["support"] >= n_min0)
                      & (item["scenario_support"] >= n_ctx0))
            eligible = stable & (mass >= delta0)
            ranked = mass / np.maximum(item["support"], 1) ** gamma
            ranked_default = mass / np.maximum(item["support"], 1) ** 1.0
            if not eligible.any():
                continue
            a = ranked.copy(); a[~eligible] = -1.0
            b = ranked_default.copy(); b[~eligible] = -1.0
            changed += int(np.argmax(a) != np.argmax(b))
        report["gamma"].append({
            "gamma": gamma,
            "selected_routine_changes_vs_default": changed,
            "activation_decisions_changed": 0,
            **reference,
        })

    grid = {(row["eta"], row["n_min"]): row["auroc"]
            for row in report["eta_by_n_min"]}
    print("\nAUROC over eta (rows) by n_min (columns)")
    print("        " + " ".join(f"{n:>6d}" for n in N_MINS))
    for eta in ETAS:
        print(f"  {eta:.2f}  " +
              " ".join(f"{grid[(eta, n)]:6.2f}" for n in N_MINS))
    aurocs = [row["auroc"] for row in report["eta_by_n_min"]]
    best = max(report["eta_by_n_min"], key=lambda r: r["auroc"])
    stable = [r for r in report["eta_by_n_min"] if r["n_min"] <= n_min0]
    report["grid_summary"] = {
        "auroc_min": min(aurocs), "auroc_max": max(aurocs),
        "reference_auroc": reference["auroc"],
        "argmax": {"eta": best["eta"], "n_min": best["n_min"],
                   "auroc": best["auroc"]},
        "reference_below_grid_max": best["auroc"] - reference["auroc"],
        "n_min_at_or_below_selected": {
            "cells": len(stable),
            "auroc_min": min(r["auroc"] for r in stable),
            "auroc_max": max(r["auroc"] for r in stable),
        },
    }
    print(f"\ngrid AUROC: min {min(aurocs):.2f}, max {max(aurocs):.2f}, "
          f"reference {reference['auroc']:.2f} "
          f"(argmax eta={best['eta']}, n_min={best['n_min']})")
    print("  n_min <= selected: %d cells, AUROC %.2f-%.2f" % (
        len(stable), min(r["auroc"] for r in stable),
        max(r["auroc"] for r in stable)))
    print("\nshare of states with no eligible routine, by n_min (eta=0.94):")
    for row in report["eta_by_n_min"]:
        if row["eta"] == eta0:
            print(f"  n_min={row['n_min']:<3d} empty={row['no_eligible_routine_rate']:5.1f}%"
                  f"  AUROC={row['auroc']:6.2f}")
    for key, field in (("sigma", "sigma_hours"), ("n_ctx", "n_ctx"),
                       ("delta", "delta")):
        print(f"\n{key}:")
        for row in report[key]:
            print(f"  {field}={row[field]:<6} AUROC={row['auroc']:6.2f} "
                  f"BA={row['balanced_accuracy']:6.2f} "
                  f"recall={row['recall']:6.2f} FA={row['false_alarm_rate']:6.2f}")
    destination = REPO / OUT
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nwrote {destination}")


if __name__ == "__main__":
    main()

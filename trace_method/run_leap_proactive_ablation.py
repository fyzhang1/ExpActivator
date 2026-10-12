"""Sealed context-factor removals using the existing frozen routine index."""
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
from trace_method.evaluate_proactive import hour, circular_hour_distance
from trace_method.build_leap_refresh_ablation import sha, write_once

VARIANTS = ("full", "no_time", "no_scenario", "raw_mass")


def select(local, similarity, query, config, variant):
    valid = local.time.astype(str).to_numpy() != query["time"]
    neighbors = (similarity >= config["semantic_threshold"]) & valid[None, :]
    support = neighbors.sum(1)
    scenario = (local.scenario.astype(str).to_numpy() == query["scenario"])
    if variant == "no_scenario":
        scenario = np.ones(len(local), dtype=bool)
    state_neighbors = neighbors & scenario[None, :]
    scenario_support = state_neighbors.sum(1)
    hours = np.array([hour(t) for t in local.time], dtype=np.float32)
    distance = circular_hour_distance(hours, hour(query["time"]))
    weights = np.exp(-(distance**2) / (2 * config["time_sigma_hours"]**2))
    if variant == "no_time":
        weights = np.ones_like(weights)
    mass = (state_neighbors * weights[None, :]).sum(1)
    stable = support >= config["minimum_support"]
    if variant != "no_scenario":
        stable &= scenario_support >= config["minimum_scenario_support"]
    trigger_score = float(mass[stable].max()) if stable.any() else -1.0
    eligible = stable & (mass >= config["trigger_mass"])
    result = {"trigger_score": trigger_score, "predicted_intent": "False",
              "excluded_exact_states": int((~valid).sum()), "source_index": None}
    if not eligible.any():
        return result
    power = 0.0 if variant == "raw_mass" else 1.0
    scores = mass / np.maximum(support, 1)**power
    scores[~eligible] = -1.0
    center = int(np.argmax(scores))
    members = np.flatnonzero(state_neighbors[center])
    all_members = np.flatnonzero(neighbors[center])
    centrality = similarity[np.ix_(members, all_members)].mean(1)
    tied = members[np.isclose(centrality, float(centrality.max()), rtol=0.0, atol=1e-7)]
    selected = min(tied.tolist(), key=lambda i: int(local.iloc[i].source_index))
    source = local.iloc[selected]
    assert int(source.user_id) == query["user_id"]
    assert str(source.time) != query["time"]
    return {**result, "predicted_intent": str(source.intentDescription),
            "source_index": int(source.source_index), "source_user_id": int(source.user_id),
            "source_time": str(source.time), "source_scenario": str(source.scenario),
            "routine_center_source_index": int(local.iloc[center].source_index),
            "routine_support": int(support[center]), "scenario_support": int(scenario_support[center]),
            "trigger_mass": float(mass[center]), "selection_score": float(scores[center])}


def main():
    out = REPO / "trace_method/results/leap_ablations/proactive"
    out.mkdir(parents=True, exist_ok=True)
    index_path = REPO / "trace_method/results/proactive_history_index_gpu0.pt"
    original_path = REPO / "trace_method/results/proactive_predictions.json"
    production = json.loads(original_path.read_text())
    cache = torch.load(index_path, map_location="cpu", weights_only=False)
    assert sha(index_path) == production["audit"]["index_sha256"]
    total_path = REPO / "samples/total.csv"
    total = pd.read_csv(total_path).reset_index(drop=True)
    memory = total.iloc[cache["memory_indices"]].reset_index().rename(columns={"index": "source_index"})
    keys = cache["keys"].float().numpy()
    queries = []
    query_paths = [REPO / "data/proactive_suggestion" / name for name in
                   ("proactive_samples.json", "proactive_false_samples.json")]
    for path in query_paths:
        for item in json.loads(path.read_text()):
            queries.append({"user_id": int(item["user_id"]), "time": str(item["time"]),
                            "scenario": str(item["scenario"])})
    assert len(queries) == 315
    grouped = {int(u): group for u, group in memory.groupby("user_id", sort=True)}
    similarities = {u: keys[group.index] @ keys[group.index].T for u, group in grouped.items()}
    config = production["config"]
    predictions = {v: [] for v in VARIANTS}
    for k, query in enumerate(queries):
        for variant in VARIANTS:
            predictions[variant].append({"query_index": k, **query,
                 **select(grouped[query["user_id"]], similarities[query["user_id"]], query, config, variant)})
    for old, new in zip(production["predictions"], predictions["full"]):
        assert (old["user_id"], old["time"], old["scenario"]) == (new["user_id"], new["time"], new["scenario"])
        assert old["predicted_intent"] == new["predicted_intent"]
        assert old.get("source_index") == new["source_index"]
    assert [r["predicted_intent"] != "False" for r in predictions["full"]] == [
        r["predicted_intent"] != "False" for r in predictions["raw_mass"]]
    inputs = [index_path, original_path, total_path, *query_paths, Path(__file__),
              REPO / "trace_method/evaluate_proactive.py", REPO / "trace_method/LEAP_PROACTIVE_ABLATION_PROTOCOL.md"]
    protocol = {"experiment": "LEAP proactive context-factor removals", "config": config,
                "retrieval_encoder": cache["embedding_model"], "model_independent": True,
                "target_fields_passed_to_selector": False, "full_matches_production": True,
                "source_hashes": {str(p.relative_to(REPO)): sha(p) for p in inputs}}
    for variant in VARIANTS:
        write_once(out / f"{variant}_predictions.json", {"protocol": {**protocol, "variant": variant},
                                                         "predictions": predictions[variant]})
    print("Sealed four complete conditions. Full matches production; raw mass preserves all trigger decisions.")


if __name__ == "__main__":
    main()

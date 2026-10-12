"""Reproduce official proactive metrics and attach deterministic uncertainty.

Identification follows ``src/eval_proactive/calculate_proactive.py`` exactly.
Semantic similarity is the official mean of edit ratio and embedding cosine,
rounded per example to two decimals before averaging.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("USE_TF", "0")

import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz
from sentence_transformers import SentenceTransformer

REPO = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=str(REPO))
    parser.add_argument(
        "--predictions", default="expactivator/results/proactive_predictions.json"
    )
    parser.add_argument(
        "--positive-csv", default="expactivator/results/proactive_positive.csv"
    )
    parser.add_argument(
        "--negative-csv", default="expactivator/results/proactive_negative.csv"
    )
    parser.add_argument(
        "--rawmass-positive-csv",
        default="expactivator/results/proactive_rawmass_positive.csv",
    )
    parser.add_argument("--embed-model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument(
        "--output", default="expactivator/results/proactive_analysis.json"
    )
    return parser.parse_args()


def interval(values: np.ndarray) -> list[float]:
    low, high = np.quantile(values, [0.025, 0.975])
    return [float(low), float(high)]


def main() -> None:
    args = parse_args()
    repo = Path(args.repo).resolve()
    prediction_record = json.loads((repo / args.predictions).read_text())
    positive = pd.read_csv(repo / args.positive_csv)
    negative = pd.read_csv(repo / args.negative_csv)
    positive["predicted_intent"] = (
        positive.predicted_intent.astype(str)
        .str.replace("**", "", regex=False)
        .str.strip()
    )
    negative["predicted_intent"] = (
        negative.predicted_intent.astype(str)
        .str.replace("**", "", regex=False)
        .str.strip()
    )
    positive_pred = (positive.predicted_intent.str.lower() != "false").to_numpy()
    negative_pred = (negative.predicted_intent.str.lower() != "false").to_numpy()
    tp = int(positive_pred.sum())
    fp = int(negative_pred.sum())
    fn = int(len(positive) - tp)
    tn = int(len(negative) - fp)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    fpr = fp / (fp + tn)
    f1 = 2 * precision * recall / (precision + recall)

    selected = positive.loc[positive_pred]
    model = SentenceTransformer(args.embed_model, device=args.device)
    original = selected.original_intent.astype(str).tolist()
    predicted = selected.predicted_intent.astype(str).tolist()
    embeddings = model.encode(original + predicted)
    count = len(original)
    semantic = []
    for index, (left, right) in enumerate(zip(original, predicted)):
        edit = fuzz.ratio(left, right) / 100.0
        a = embeddings[index]
        b = embeddings[count + index]
        cosine = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        semantic.append(round((edit + cosine) / 2.0, 2))
    semantic_array = np.asarray(semantic, dtype=np.float64)

    rawmass_path = repo / args.rawmass_positive_csv
    rawmass_semantic_array: np.ndarray | None = None
    if rawmass_path.exists():
        rawmass = pd.read_csv(rawmass_path)
        rawmass["predicted_intent"] = (
            rawmass.predicted_intent.astype(str)
            .str.replace("**", "", regex=False)
            .str.strip()
        )
        rawmass_pred = (rawmass.predicted_intent.str.lower() != "false").to_numpy()
        if not np.array_equal(rawmass_pred, positive_pred):
            raise ValueError("Raw-mass ablation changed trigger decisions")
        raw_text = rawmass.loc[rawmass_pred].predicted_intent.astype(str).tolist()
        raw_embeddings = model.encode(raw_text)
        raw_scores = []
        for index, (left, right) in enumerate(zip(original, raw_text)):
            edit = fuzz.ratio(left, right) / 100.0
            a = embeddings[index]
            b = raw_embeddings[index]
            cosine = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
            raw_scores.append(round((edit + cosine) / 2.0, 2))
        rawmass_semantic_array = np.asarray(raw_scores, dtype=np.float64)

    rng = np.random.default_rng(args.seed)
    metric_draws = {name: [] for name in ("precision", "recall", "fpr", "f1")}
    positive_values = positive_pred.astype(np.int64)
    negative_values = negative_pred.astype(np.int64)
    semantic_draws = []
    semantic_delta_draws = []
    for _ in range(args.bootstrap_samples):
        pos = rng.choice(positive_values, size=len(positive_values), replace=True)
        neg = rng.choice(negative_values, size=len(negative_values), replace=True)
        draw_tp = int(pos.sum())
        draw_fp = int(neg.sum())
        draw_precision = draw_tp / (draw_tp + draw_fp) if draw_tp + draw_fp else 0.0
        draw_recall = draw_tp / len(pos)
        draw_fpr = draw_fp / len(neg)
        draw_f1 = (
            2 * draw_precision * draw_recall / (draw_precision + draw_recall)
            if draw_precision + draw_recall
            else 0.0
        )
        metric_draws["precision"].append(draw_precision)
        metric_draws["recall"].append(draw_recall)
        metric_draws["fpr"].append(draw_fpr)
        metric_draws["f1"].append(draw_f1)
        semantic_indices = rng.choice(
            len(semantic_array), size=len(semantic_array), replace=True
        )
        semantic_draws.append(float(semantic_array[semantic_indices].mean()))
        if rawmass_semantic_array is not None:
            semantic_delta_draws.append(
                float(
                    (
                        semantic_array[semantic_indices]
                        - rawmass_semantic_array[semantic_indices]
                    ).mean()
                )
            )

    published_him = {
        "semantic": 0.535,
        "llm_judgement": 0.363,
        "precision": 0.781,
        "recall": 0.814,
        "false_alarm_rate": 0.49,
        "f1": 0.797,
    }
    metrics = {
        "semantic": float(semantic_array.mean()),
        "semantic_ci95": interval(np.asarray(semantic_draws)),
        "semantic_evaluated_predictions": len(semantic_array),
        "precision": precision,
        "precision_ci95": interval(np.asarray(metric_draws["precision"])),
        "recall": recall,
        "recall_ci95": interval(np.asarray(metric_draws["recall"])),
        "false_alarm_rate": fpr,
        "false_alarm_rate_ci95": interval(np.asarray(metric_draws["fpr"])),
        "f1": f1,
        "f1_ci95": interval(np.asarray(metric_draws["f1"])),
        "counts": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }
    result = {
        "protocol": (
            "official proactive identification and Qwen3-Embedding-0.6B semantic "
            "metric; stratified bootstrap uncertainty; GPU0"
        ),
        "metrics": metrics,
        "published_him_reference_not_controlled": published_him,
        "difference_from_published_him": {
            "semantic": metrics["semantic"] - published_him["semantic"],
            "precision": precision - published_him["precision"],
            "recall": recall - published_him["recall"],
            "false_alarm_rate": fpr - published_him["false_alarm_rate"],
            "f1": f1 - published_him["f1"],
        },
        "frequency_normalization_ablation": (
            {
                "raw_occurrence_mass_semantic": float(rawmass_semantic_array.mean()),
                "normalized_state_likelihood_semantic": metrics["semantic"],
                "paired_delta": float((semantic_array - rawmass_semantic_array).mean()),
                "paired_delta_ci95": interval(np.asarray(semantic_delta_draws)),
                "trigger_decisions_identical": True,
            }
            if rawmass_semantic_array is not None
            else {"status": "not_available"}
        ),
        "llm_judgement": {
            "status": "not_run",
            "reason": (
                "the release contains no judgement implementation or configured "
                "judge endpoint; this external metric is not fabricated"
            ),
        },
        "prediction_audit": prediction_record["audit"],
        "config": prediction_record["config"],
        "bootstrap": {"samples": args.bootstrap_samples, "seed": args.seed},
    }
    output = repo / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

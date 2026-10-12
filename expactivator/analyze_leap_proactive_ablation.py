"""Official suggestion scoring and paired user-cluster context diagnostics."""
from __future__ import annotations
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("USE_TF", "0")
import numpy as np
from fuzzywuzzy import fuzz

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from expactivator.build_leap_refresh_ablation import sha

VARIANTS = ("full", "no_time", "no_scenario", "raw_mass")


def roc_curve(labels, scores):
    thresholds = np.r_[np.inf, np.unique(scores)[::-1], -np.inf]
    return [{"threshold": None if not np.isfinite(t) else float(t),
             "threshold_endpoint": "none" if t == np.inf else "all" if t == -np.inf else None,
             "recall": float((scores[labels == 1] >= t).mean()),
             "false_alarm_rate": float((scores[labels == 0] >= t).mean())}
            for t in thresholds]


def bootstrap_metrics(labels, users, scores, emitted, semantic, draws=20000):
    names = np.unique(users)
    group = np.searchsorted(names, users)
    n = len(names)
    pos = np.bincount(group, weights=labels, minlength=n)
    neg = np.bincount(group, weights=1-labels, minlength=n)
    tp = np.bincount(group, weights=labels*emitted, minlength=n)
    fp = np.bincount(group, weights=(1-labels)*emitted, minlength=n)
    semantic_sum = np.bincount(group, weights=labels*semantic, minlength=n)
    positive_indices = np.flatnonzero(labels == 1)
    negative_indices = np.flatnonzero(labels == 0)
    wins = (scores[positive_indices, None] > scores[None, negative_indices]).astype(float)
    wins += 0.5*(scores[positive_indices, None] == scores[None, negative_indices])
    concordance = np.zeros((n, n))
    np.add.at(concordance, (group[positive_indices, None], group[None, negative_indices]), wins)
    rng = np.random.default_rng(20260913)
    w = rng.multinomial(n, np.full(n, 1/n), size=draws)
    sums = w @ np.column_stack([pos, neg, tp, fp, semantic_sum])
    valid = (sums[:, 0] > 0) & (sums[:, 1] > 0) & (sums[:, 2] > 0)
    assert valid.all(), "Unexpected degenerate bootstrap draw; handle consistently before reporting"
    boot = {
        "balanced_accuracy": 0.5*(sums[:, 2]/sums[:, 0] + 1-sums[:, 3]/sums[:, 1]),
        "auroc": np.einsum("bi,ij,bj->b", w, concordance, w, optimize=True)/(sums[:, 0]*sums[:, 1]),
        "semantic_conditional": sums[:, 4]/sums[:, 2],
        "semantic_all_positive": sums[:, 4]/sums[:, 0],
    }
    totals = np.array([pos.sum(), neg.sum(), tp.sum(), fp.sum(), semantic_sum.sum()])
    estimates = {
        "balanced_accuracy": float(0.5*(totals[2]/totals[0] + 1-totals[3]/totals[1])),
        "auroc": float(concordance.sum()/(totals[0]*totals[1])),
        "semantic_conditional": float(totals[4]/totals[2]),
        "semantic_all_positive": float(totals[4]/totals[0]),
        "recall": float(totals[2]/totals[0]), "false_alarm_rate": float(totals[3]/totals[1]),
        "tp": int(totals[2]), "fp": int(totals[3]),
        "n_users": n,
    }
    for name, values in boot.items():
        estimates[name + "_ci95"] = np.quantile(values, [0.025, 0.975]).tolist()
    return estimates, boot


def main():
    out = REPO / "expactivator/results/leap_ablations/proactive"
    files = {v: json.loads((out / f"{v}_predictions.json").read_text()) for v in VARIANTS}
    for record in files.values():
        for name, digest in record["protocol"]["source_hashes"].items():
            assert sha(REPO / name) == digest
    positive = json.loads((REPO / "data/proactive_suggestion/proactive_samples.json").read_text())
    negative = json.loads((REPO / "data/proactive_suggestion/proactive_false_samples.json").read_text())
    labels = np.r_[np.ones(len(positive)), np.zeros(len(negative))].astype(int)
    queries = positive + negative
    predictions = {v: files[v]["predictions"] for v in VARIANTS}
    for rows in predictions.values():
        assert len(rows) == len(queries) == 315
        assert [(r["user_id"], r["time"], r["scenario"]) for r in rows] == [
            (int(q["user_id"]), str(q["time"]), str(q["scenario"])) for q in queries]
    users = np.array([int(q["user_id"]) for q in queries])
    unique_texts = sorted({q["originalInstruction"] for q in positive} | {
        r["predicted_intent"] for rows in predictions.values() for r in rows[:len(positive)]
        if r["predicted_intent"] != "False"})
    encoder_path = "/storage/zhangfuyao/models/Qwen3-Embedding-0.6B"
    cache_path = out / "scoring_embeddings.npz"
    text_path = out / "scoring_texts.json"
    if cache_path.exists():
        assert json.loads(text_path.read_text()) == unique_texts
        vectors = np.load(cache_path)["vectors"]
    else:
        import torch
        from sentence_transformers import SentenceTransformer
        torch.set_num_threads(8)
        print(f"Encoding {len(unique_texts)} unique scoring strings on CPU", flush=True)
        encoder = SentenceTransformer(encoder_path, device="cpu", local_files_only=True)
        vectors = encoder.encode(unique_texts, batch_size=16, show_progress_bar=True)
        np.savez_compressed(cache_path, vectors=vectors)
        text_path.write_text(json.dumps(unique_texts, ensure_ascii=False, indent=2) + "\n")
    lookup = dict(zip(unique_texts, vectors))
    metrics, bootstraps, outputs = {}, {}, {}
    for variant in VARIANTS:
        rows = predictions[variant]
        scores = np.array([r["trigger_score"] for r in rows])
        emitted = np.array([r["predicted_intent"] != "False" for r in rows]).astype(int)
        semantics = np.zeros(len(rows))
        for i, (q, row) in enumerate(zip(positive, rows)):
            if not emitted[i]:
                continue
            left, right = q["originalInstruction"], row["predicted_intent"].replace("**", "").strip()
            a, b = lookup[left], lookup[right]
            cosine = float(np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b)))
            semantics[i] = round((fuzz.ratio(left, right)/100 + cosine)/2, 2)
        metrics[variant], bootstraps[variant] = bootstrap_metrics(labels, users, scores, emitted, semantics)
        outputs[variant] = [{**r, "split_label": int(labels[i]),
                             "semantic_score": float(semantics[i]) if labels[i] else None}
                            for i, r in enumerate(rows)]
        (out / f"{variant}_roc_curve.json").write_text(json.dumps(roc_curve(labels, scores), indent=2) + "\n")
    assert np.array_equal([r["predicted_intent"] != "False" for r in predictions["full"]],
                          [r["predicted_intent"] != "False" for r in predictions["raw_mass"]])
    original_metrics = json.loads((REPO / "expactivator/results/proactive_analysis.json").read_text())["metrics"]
    reproduction = {"existing_semantic": original_metrics["semantic"],
                    "new_semantic": metrics["full"]["semantic_conditional"],
                    "semantic_delta": metrics["full"]["semantic_conditional"]-original_metrics["semantic"]}
    contrasts = {}
    for variant in VARIANTS[1:]:
        contrasts[variant] = {}
        for metric in bootstraps["full"]:
            values = bootstraps["full"][metric] - bootstraps[variant][metric]
            contrasts[variant][metric] = {"delta": metrics["full"][metric]-metrics[variant][metric],
                                         "delta_ci95": np.quantile(values, [0.025, 0.975]).tolist()}
    # Sensitivity to the earlier report's independent-query resampling unit.
    # Added after the primary user-cluster analysis; keep both, not whichever
    # interval is more favorable. Both conditions emit on exactly the same rows.
    normalization_deltas = np.array([
        a["semantic_score"]-b["semantic_score"]
        for a, b in zip(outputs["full"][:len(positive)], outputs["raw_mass"][:len(positive)])
        if a["predicted_intent"] != "False"])
    rng = np.random.default_rng(20260913)
    row_draws = rng.choice(normalization_deltas, size=(20000, len(normalization_deltas)), replace=True).mean(1)
    row_sensitivity = {"cluster": "independent emitted positive query", "n": len(normalization_deltas),
                       "delta": float(normalization_deltas.mean()),
                       "delta_ci95": np.quantile(row_draws, [.025, .975]).tolist(),
                       "status": "post-primary sensitivity matching the earlier report's resampling unit"}
    result = {"status": "complete", "model_independent": True,
              "retrieval_encoder": files["full"]["protocol"]["retrieval_encoder"],
              "semantic_scoring_encoder": encoder_path, "scoring_device": "cpu",
              "bootstrap": {"cluster": "user_id", "draws": 20000, "seed": 20260913},
              "metrics": metrics, "full_minus_ablation": contrasts,
              "production_reproduction": reproduction,
              "normalization_row_bootstrap_sensitivity": row_sensitivity,
              "unretuned_operating_point": True, "threshold_selected_on_test": False,
              "all_declared_conditions_retained": True,
              "output_hashes": {f"{v}_predictions.json": sha(out / f"{v}_predictions.json") for v in VARIANTS}}
    (out / "scored_predictions.json").write_text(json.dumps(outputs, ensure_ascii=False, indent=2) + "\n")
    (out / "analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    report = ["# LEAP 主动建议情境消融", "", "模型无关选择器；四组使用同一历史索引。", "",
              "| 条件 | AUROC (%) | BA (%) | Recall (%) | FA (%) | 条件 Semantic (%) | 全正例 Semantic (%) |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for v in VARIANTS:
        m = metrics[v]
        report.append("| " + v + " | " + " | ".join(f"{100*m[k]:.2f}" for k in
                      ("auroc", "balanced_accuracy", "recall", "false_alarm_rate", "semantic_conditional", "semantic_all_positive")) + " |")
    report += ["", "BA/Recall/FA 使用原固定阈值，移除条件会改变分数尺度；不能单凭该工作点归因。",
               "AUROC 及完整曲线为阈值无关诊断，未从测试曲线选择阈值。全正例 Semantic 将不建议记零，是额外覆盖率诊断。",
               "归一化消融保持触发集合不变；用户聚类配对区间见 analysis.json。所有有效条件均保留。", ""]
    (out / "RESULTS_zh.md").write_text("\n".join(report))
    print("\n".join(report), flush=True)


if __name__ == "__main__":
    main()

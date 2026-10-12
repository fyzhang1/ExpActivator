"""Coverage-preserving donor replacement with the original strict calipers."""
from __future__ import annotations
import copy
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("USE_TF", "0")
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from trace_method.execution_core import execution_memory_frame, audit_memory_split, load_app_labels
from trace_method.build_execution_grounded_evidence import evidence
from trace_method.build_execution_index import ranked_vote
from trace_method.build_leap_user_history_ablation import normalize
from trace_method.build_leap_refresh_ablation import sha, write_once


def main():
    out = REPO / "trace_method/results/leap_ablations/user_history_partial"
    out.mkdir(parents=True, exist_ok=True)
    base = REPO / "trace_method/results/grounded"
    strict = REPO / "trace_method/results/leap_ablations/user_history"
    cache_path = base / "execution_history_index_gpu0.pt"
    index_path = base / "execution_app_predictions_gpu0.json"
    context_path = base / "qwen3_vl_8b_contexts.json"
    label_path = base / "execution_app_labels.json"
    test_path = REPO / "data/personal_execution/personal_execution.json"
    history = execution_memory_frame(REPO)
    history_audit = audit_memory_split(history, REPO)
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    index = json.loads(index_path.read_text())
    original = json.loads(context_path.read_text())
    keys, queries = cache["residual_keys"].float().numpy(), cache["queries"].float().numpy()
    users = history.user_id.to_numpy().astype(int)
    normalized = np.array([normalize(t) for t in cache["residual_texts"]])
    strict_rows = {r["step_id"]: r for r in json.loads((strict / "assignments.json").read_text())["rows"]}
    assignment = []
    for qi, episode in enumerate(index["episode_ids"]):
        sid = episode + "_0"
        prior = strict_rows[sid]
        own = np.array(prior["own_indices"], dtype=int)
        query_cos = keys @ queries[qi]
        donors = np.flatnonzero((users != prior["user_id"]) & np.any(
            np.abs(query_cos[:, None]-query_cos[own][None, :]) <= .02, axis=1))
        pair_cos = keys[own] @ keys[donors].T
        gap = np.abs(query_cos[own, None]-query_cos[donors][None, :])
        eligible = (pair_cos >= .90) & (gap <= .02)
        exact = normalized[own, None] == normalized[donors][None, :]
        cost = (1-exact.astype(float))*10+(1-pair_cos)+gap
        cost[~eligible] = 1e6
        left, right = linear_sum_assignment(cost)
        valid = eligible[left, right]
        l, r = left[valid], right[valid]
        chosen = own.copy()
        chosen[l] = donors[r]
        complete = len(l) == 10
        assert complete == prior["matched"]
        if complete:
            assert chosen.tolist() == prior["donor_indices"]
        assignment.append({"step_id": sid, "episode_id": episode, "user_id": prior["user_id"],
                           "own_indices": own.tolist(), "intervened_indices": chosen.tolist(),
                           "replacement_positions": l.tolist(), "n_replaced": int(len(l)),
                           "pair_goal_cosines": pair_cos[l, r].tolist(),
                           "query_score_gaps": gap[l, r].tolist(),
                           "exact_residual_matches": exact[l, r].tolist(),
                           "complete_match": complete})
    write_once(out / "assignments.json", {"history_audit": history_audit, "rows": assignment})
    labels = load_app_labels(REPO, str(label_path), history)
    tests = [r for r in json.loads(test_path.read_text()) if r["id"].endswith("_0")]
    assert len(tests) == 775
    write_once(out / "entry_test.json", tests)
    contexts = {v: {} for v in ("own", "partial_other")}
    source_records = {v: {} for v in contexts}
    changed_apps = 0
    for qi, item in enumerate(assignment):
        sid = item["step_id"]
        weights = index["factorized_similarities"][qi][:10]
        own_app, own_votes = ranked_vote(item["own_indices"], weights, labels, 10)
        other_app, other_votes = ranked_vote(item["intervened_indices"], weights, labels, 10)
        assert original["contexts"][sid]["retrieved_action"]["app_name"] == own_app
        contexts["own"][sid] = copy.deepcopy(original["contexts"][sid])
        contexts["partial_other"][sid] = evidence({"action": "open_app", "app_name": other_app},
                                                 original["contexts"][sid]["retrieval_basis"])
        source_records["own"][sid] = {"history_indices": item["own_indices"], "vote_weights": weights, "votes": own_votes}
        source_records["partial_other"][sid] = {"history_indices": item["intervened_indices"], "vote_weights": weights, "votes": other_votes}
        changed_apps += own_app != other_app
    strict_contexts = json.loads((strict / "matched_other_contexts.json").read_text())["contexts"]
    for sid, ctx in strict_contexts.items():
        assert ctx == contexts["partial_other"][sid]
    for v in contexts:
        write_once(out / f"{v}_contexts.json", {"protocol": {
            "method": "trace_grounded", "context_scope": "step", "test_sha256": sha(out / "entry_test.json"),
            "experiment": "LEAP partial matched cross-user application evidence", "variant": v,
            "target_fields_used_for_matching": False, "original_rank_weights_preserved": True},
            "contexts": contexts[v], "sources": source_records[v]})
    inputs = [cache_path, index_path, context_path, label_path, test_path, Path(__file__),
              REPO / "trace_method/LEAP_PARTIAL_HISTORY_PROTOCOL.md", REPO / "trace_method/build_leap_user_history_ablation.py",
              strict / "assignments.json", REPO / "trace_method/build_execution_index.py",
              REPO / "src/eval_execution/evaluation_new_backbones_vllm.py",
              REPO / "src/eval_execution/evaluation_new_backbones.py", REPO / "src/eval_execution/prompt.py",
              REPO / "src/eval_execution/utils.py", out / "entry_test.json", out / "assignments.json"]
    inputs += [out / f"{v}_contexts.json" for v in contexts]
    write_once(out / "source_hashes.json", {str(p.relative_to(REPO)): sha(p) for p in inputs})
    summary = {"episodes": len(assignment), "at_least_one_replacement": sum(r['n_replaced']>0 for r in assignment),
               "complete_matches": sum(r['complete_match'] for r in assignment), "changed_reference_apps": changed_apps,
               "replaced_records": sum(r['n_replaced'] for r in assignment), "total_records": 10*len(assignment),
               "mean_goal_pair_cosine": float(np.mean([v for r in assignment for v in r['pair_goal_cosines']])),
               "mean_query_score_gap": float(np.mean([v for r in assignment for v in r['query_score_gaps']]))}
    write_once(out / "matching_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

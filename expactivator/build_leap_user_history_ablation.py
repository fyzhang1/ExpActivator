"""Match application evidence across users without using target preferences."""
from __future__ import annotations
import copy
import json
import os
from pathlib import Path
import re
import sys

os.environ.setdefault("USE_TF", "0")
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from expactivator.execution_core import execution_memory_frame, audit_memory_split, load_app_labels
from expactivator.build_execution_grounded_evidence import evidence
from expactivator.build_execution_index import ranked_vote
from expactivator.build_leap_refresh_ablation import sha, write_once


def normalize(text):
    return "".join(c.lower() for c in text if c.isalnum())


def main():
    out = REPO / "expactivator/results/leap_ablations/user_history"
    out.mkdir(parents=True, exist_ok=True)
    base = REPO / "expactivator/results/grounded"
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
    keys = cache["residual_keys"].float().numpy()
    queries = cache["queries"].float().numpy()
    users = history.user_id.to_numpy().astype(int)
    normalized_goals = np.array([normalize(t) for t in cache["residual_texts"]])
    entry_queries = [{k: row[k] for k in ("id", "user_id", "ambiguous_instruction")}
                     for row in json.loads(test_path.read_text()) if row["id"].endswith("_0")]
    query_by_episode = {r["id"][:-2]: r for r in entry_queries}
    assert set(query_by_episode) == set(index["episode_ids"])
    entry_queries = [query_by_episode[eid] for eid in index["episode_ids"]]
    assignments = []
    for qi, row in enumerate(entry_queries):
        own = np.array(index["factorized_neighbors"][qi][:10], dtype=int)
        assert len(own) == 10 and (users[own] == int(row["user_id"])).all()
        # Eligibility and cost use only the frozen residual-goal representation.
        query_cos = keys @ queries[qi]
        candidate_mask = (users != int(row["user_id"])) & np.any(
            np.abs(query_cos[:, None]-query_cos[own][None, :]) <= 0.02, axis=1)
        donors = np.flatnonzero(candidate_mask)
        pair_cos = keys[own] @ keys[donors].T
        score_gap = np.abs(query_cos[own, None]-query_cos[donors][None, :])
        eligible = (pair_cos >= 0.90) & (score_gap <= 0.02)
        exact = normalized_goals[own, None] == normalized_goals[donors][None, :]
        cost = (1-exact.astype(float))*10 + (1-pair_cos) + score_gap
        cost[~eligible] = 1e6
        if cost.shape[1] >= 10:
            left, right = linear_sum_assignment(cost)
            complete = len(left) == 10 and np.all(eligible[left, right])
        else:
            complete = False
        item = {"step_id": row["id"], "episode_id": row["id"][:-2],
                "user_id": int(row["user_id"]), "matched": bool(complete),
                "own_indices": own.tolist(), "eligible_edges": int(eligible.sum())}
        if complete:
            assert np.array_equal(left, np.arange(10))
            chosen = donors[right]
            assert len(set(chosen.tolist())) == 10 and np.all(users[chosen] != int(row["user_id"]))
            item.update({"donor_indices": chosen.tolist(), "donor_user_ids": users[chosen].tolist(),
                         "pair_goal_cosines": pair_cos[left, right].tolist(),
                         "query_score_gaps": score_gap[left, right].tolist(),
                         "exact_residual_matches": exact[left, right].tolist()})
        assignments.append(item)
    write_once(out / "assignments.json", {"history_audit": history_audit, "rows": assignments})

    # Source app labels are used only after all assignments are fixed.
    labels = load_app_labels(REPO, str(label_path), history)
    selected_ids = {r["step_id"] for r in assignments if r["matched"]}
    # Answers are copied verbatim into the evaluator's test file; they did not
    # participate in matching, filtering, weighting, or reference selection.
    evaluation_rows = [r for r in json.loads(test_path.read_text()) if r["id"] in selected_ids]
    write_once(out / "entry_test.json", evaluation_rows)
    contexts = {v: {} for v in ("own", "matched_other")}
    source_records = {v: {} for v in contexts}
    changed = 0
    for qi, item in enumerate(assignments):
        if not item["matched"]:
            continue
        sid = item["step_id"]
        weights = index["factorized_similarities"][qi][:10]
        own_app, own_votes = ranked_vote(item["own_indices"], weights, labels, 10)
        assert own_app == index["factorized_app_names"][qi]
        assert original["contexts"][sid]["retrieved_action"] == {"action": "open_app", "app_name": own_app}
        other_app, other_votes = ranked_vote(item["donor_indices"], weights, labels, 10)
        contexts["own"][sid] = copy.deepcopy(original["contexts"][sid])
        contexts["matched_other"][sid] = evidence({"action": "open_app", "app_name": other_app},
                                                    original["contexts"][sid]["retrieval_basis"])
        source_records["own"][sid] = {"history_indices": item["own_indices"], "vote_weights": weights, "votes": own_votes}
        source_records["matched_other"][sid] = {"history_indices": item["donor_indices"], "vote_weights": weights, "votes": other_votes}
        changed += own_app != other_app
    inputs = [cache_path, index_path, context_path, label_path, test_path, Path(__file__),
              REPO / "expactivator/LEAP_USER_HISTORY_PROTOCOL.md", REPO / "expactivator/build_execution_index.py",
              REPO / "src/eval_execution/evaluation_new_backbones_vllm.py",
              REPO / "src/eval_execution/evaluation_new_backbones.py", REPO / "src/eval_execution/prompt.py",
              REPO / "src/eval_execution/utils.py"]
    for v in contexts:
        write_once(out / f"{v}_contexts.json", {"protocol": {
            "method": "expactivator_grounded", "context_scope": "step", "test_sha256": sha(out / "entry_test.json"),
            "experiment": "LEAP matched cross-user application evidence", "variant": v,
            "target_fields_used_for_matching": False, "original_rank_weights_preserved": True},
            "contexts": contexts[v], "sources": source_records[v]})
    inputs += [out / "entry_test.json", out / "assignments.json"] + [out / f"{v}_contexts.json" for v in contexts]
    write_once(out / "source_hashes.json", {str(p.relative_to(REPO)): sha(p) for p in inputs})
    matched = [r for r in assignments if r["matched"]]
    summary = {"requested_episodes": len(assignments), "matched_episodes": len(matched),
               "matched_users": len({r['user_id'] for r in matched}), "changed_reference_apps": changed,
               "exact_matched_pairs": sum(sum(r['exact_residual_matches']) for r in matched),
               "total_matched_pairs": 10*len(matched),
               "mean_goal_pair_cosine": float(np.mean([v for r in matched for v in r['pair_goal_cosines']])) if matched else None,
               "mean_query_score_gap": float(np.mean([v for r in matched for v in r['query_score_gaps']])) if matched else None}
    write_once(out / "matching_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

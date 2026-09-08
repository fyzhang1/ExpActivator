"""Build a frozen, non-parametric personal-history index for TRACE.

TRACE never updates an encoder and never fits an auxiliary predictor.  This
compiler performs three deterministic operations:

1. remove the explicit entry/app phrase from each factual history so that it
   can be compared with a query in which that preference was omitted;
2. encode full goals, residual goals, and ambiguous queries with a frozen text
   encoder; and
3. rank same-user records and aggregate their training-side action labels with
   a fixed reciprocal-rank vote.

Target answers, original target instructions, app names, and intent classes
are never read while constructing predictions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

REPO = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=str(REPO))
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-neighbors", type=int, default=32)
    parser.add_argument(
        "--train-labels", default="trace_method/results/execution_app_labels.json"
    )
    parser.add_argument(
        "--cache", default="trace_method/results/execution_history_index_gpu0.pt"
    )
    parser.add_argument(
        "--output", default="trace_method/results/execution_app_predictions_gpu0.json"
    )
    return parser.parse_args()


ENTRY_PATTERN = __import__("re").compile(
    r"^(?:请)?(?:帮我)?(?:打开|启动|进入|使用|用|在)([^，,。；;]{1,24})"
)
ACTION_MARKERS = (
    "查看",
    "搜索",
    "搜",
    "扫描",
    "扫码",
    "导航",
    "播放",
    "购买",
    "设置",
    "发送",
    "进入",
    "打开",
    "点击",
    "选择",
    "领取",
    "修改",
    "发布",
    "添加",
    "开始",
    "阅读",
    "记录",
    "切换",
    "下载",
    "拍摄",
    "删除",
    "出示",
    "开启",
    "关闭",
    "付款",
    "支付",
    "充值",
    "浏览",
    "检索",
    "查询",
    "打卡",
    "签到",
    "预约",
    "创建",
    "新建",
    "关锁",
    "开锁",
    "设定",
    "听",
    "看",
    "买",
    "登录",
    "扫一扫",
    "点赞",
    "评论",
    "分享",
    "更新",
    "升级",
    "卸载",
    "清理",
    "清除",
    "把",
    "将",
)


def residual_goal(instruction: str) -> str:
    """Remove only the leading entry choice; retain the requested GUI goal."""
    instruction = str(instruction).strip()
    match = ENTRY_PATTERN.search(instruction)
    if not match:
        return instruction
    choice = match.group(1)
    cuts = [choice.find(marker) for marker in ACTION_MARKERS]
    cuts = [position for position in cuts if position > 0]
    if cuts:
        choice = choice[: min(cuts)]
    end = match.start(1) + len(choice)
    residual = instruction[end:].lstrip(" ，,：:、")
    return residual or instruction


def mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded = mask.unsqueeze(-1)
    return (hidden * expanded).sum(1) / expanded.sum(1).clamp_min(1)


@torch.inference_mode()
def encode(
    texts: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: str,
    batch_size: int,
) -> torch.Tensor:
    chunks = []
    for start in tqdm(range(0, len(texts), batch_size), desc="frozen text encoding"):
        batch = tokenizer(
            texts[start : start + batch_size],
            padding=True,
            truncation=True,
            max_length=96,
            return_tensors="pt",
        ).to(device)
        hidden = model(**batch).last_hidden_state
        pooled = mean_pool(hidden, batch["attention_mask"])
        chunks.append(torch.nn.functional.normalize(pooled, dim=-1).cpu())
    return torch.cat(chunks)


def ranked_vote(
    indices: list[int], similarities: list[float], labels: list[str], top_k: int
) -> tuple[str, dict[str, float]]:
    scores: dict[str, float] = {}
    for rank, (index, similarity) in enumerate(
        zip(indices[:top_k], similarities[:top_k]), start=1
    ):
        label = labels[index]
        scores[label] = scores.get(label, 0.0) + max(similarity, 0.0) / rank
    if not scores:
        raise ValueError("A user has no historical action labels")
    # Lexical tie-breaking makes the compiler fully deterministic.
    prediction = min(scores, key=lambda label: (-scores[label], label))
    return prediction, scores


def main() -> None:
    args = parse_args()
    repo = Path(args.repo).resolve()
    train = pd.read_csv(repo / "samples/train_set.csv").reset_index(drop=True)
    episodes = json.loads(
        (repo / "data/personal_execution/personal_samples.json").read_text()
    )
    label_file = json.loads((repo / args.train_labels).read_text())
    label_rows = label_file["rows"]
    if len(label_rows) != len(train) or any(
        row["row_index"] != index for index, row in enumerate(label_rows)
    ):
        raise ValueError("Training labels do not match the official training row order")
    labels = [str(row["app_name"]) for row in label_rows]

    tokenizer = AutoTokenizer.from_pretrained(
        args.embedding_model, local_files_only=True
    )
    model = (
        AutoModel.from_pretrained(args.embedding_model, local_files_only=True)
        .to(args.device)
        .eval()
    )

    full_texts = train.intentDescription.astype(str).tolist()
    residual_texts = [residual_goal(text) for text in full_texts]
    query_texts = [str(row["ambiguousInstruction"]) for row in episodes]
    full_keys = encode(full_texts, tokenizer, model, args.device, args.batch_size)
    residual_keys = encode(
        residual_texts, tokenizer, model, args.device, args.batch_size
    )
    queries = encode(query_texts, tokenizer, model, args.device, args.batch_size)

    user_indices = {
        int(user): torch.tensor(group.index.to_numpy(), dtype=torch.long)
        for user, group in train.groupby("user_id", sort=True)
    }
    result: dict[str, object] = {
        "protocol": (
            "frozen multilingual encoder; same-user non-parametric ranking; "
            "reciprocal-rank action vote; no fitted parameters; target inputs "
            "limited to user_id and ambiguousInstruction"
        ),
        "embedding_model": str(args.embedding_model),
        "device": args.device,
        "parameter_training": False,
        "top_k": args.top_k,
        "max_neighbors": args.max_neighbors,
        "episode_ids": [],
        "factorized_app_names": [],
        "full_goal_app_names": [],
        "factorized_neighbors": [],
        "full_goal_neighbors": [],
        "factorized_similarities": [],
        "full_goal_similarities": [],
        "factorized_vote_scores": [],
        "full_goal_vote_scores": [],
    }

    for query_index, episode in enumerate(tqdm(episodes, desc="same-user ranking")):
        user = int(episode["user_id"])
        indices = user_indices[user]
        per_view = []
        for keys in (residual_keys, full_keys):
            similarities = queries[query_index] @ keys[indices].T
            order = torch.argsort(similarities, descending=True)
            keep = order[: min(args.max_neighbors, len(order))]
            neighbors = indices[keep].tolist()
            values = similarities[keep].tolist()
            prediction, scores = ranked_vote(neighbors, values, labels, args.top_k)
            per_view.append((prediction, neighbors, values, scores))

        factorized, full = per_view
        episode_id = f"{episode['user_id']}_{episode['time']}"
        result["episode_ids"].append(episode_id)
        result["factorized_app_names"].append(factorized[0])
        result["full_goal_app_names"].append(full[0])
        result["factorized_neighbors"].append(factorized[1])
        result["full_goal_neighbors"].append(full[1])
        result["factorized_similarities"].append(factorized[2])
        result["full_goal_similarities"].append(full[2])
        result["factorized_vote_scores"].append(factorized[3])
        result["full_goal_vote_scores"].append(full[3])

    destination = repo / args.output
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    cache = repo / args.cache
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "protocol": result["protocol"],
            "full_keys": full_keys.half(),
            "residual_keys": residual_keys.half(),
            "queries": queries.half(),
            "residual_texts": residual_texts,
            "embedding_model": str(args.embedding_model),
        },
        cache,
    )
    print(f"saved {destination}")
    print(f"saved {cache}")


if __name__ == "__main__":
    main()

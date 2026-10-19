from __future__ import annotations

from typing import Callable, Dict, List, Set


def recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hit = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hit / len(relevant_ids)


def evaluate_recall_at_k(
    eval_data: List[dict],
    retrieve_fn: Callable[[str], List[str]],
    k: int,
) -> Dict:
    per_query: List[dict] = []
    total = 0.0
    for item in eval_data:
        query = item["query"]
        relevant = set(item["relevant_ids"])
        retrieved = retrieve_fn(query)
        r = recall_at_k(retrieved, relevant, k)
        per_query.append(
            {
                "query": query,
                "recall": round(r, 4),
                "retrieved": retrieved[:k],
                "relevant": list(relevant),
            }
        )
        total += r
    avg = total / len(eval_data) if eval_data else 0.0
    return {
        "metric": f"recall@{k}",
        "value": round(avg, 4),
        "num_queries": len(eval_data),
        "per_query": per_query,
    }

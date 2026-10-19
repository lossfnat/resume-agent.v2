from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.metrics import evaluate_recall_at_k, recall_at_k


def test_recall_at_k_full_hit():
    assert recall_at_k(["a", "b", "c"], {"a", "b"}, k=5) == 1.0


def test_recall_at_k_partial_hit():
    assert recall_at_k(["a", "x", "y"], {"a", "b"}, k=3) == 0.5


def test_recall_at_k_no_hit():
    assert recall_at_k(["x", "y"], {"a", "b"}, k=3) == 0.0


def test_recall_at_k_topk_truncation():
    assert recall_at_k(["a", "b", "c", "d"], {"a", "b"}, k=1) == 0.5


def test_recall_at_k_empty_relevant():
    assert recall_at_k(["a"], set(), k=5) == 0.0


def test_evaluate_recall_at_k_all_hit():
    eval_data = [
        {"query": "q1", "relevant_ids": ["a"]},
        {"query": "q2", "relevant_ids": ["b", "c"]},
    ]
    result = evaluate_recall_at_k(
        eval_data,
        retrieve_fn=lambda q: {"q1": ["a", "x"], "q2": ["b", "c", "z"]}[q],
        k=5,
    )
    assert result["value"] == 1.0
    assert result["num_queries"] == 2
    assert result["metric"] == "recall@5"


def test_evaluate_recall_at_k_partial():
    eval_data = [
        {"query": "q1", "relevant_ids": ["a"]},
        {"query": "q2", "relevant_ids": ["b", "c"]},
    ]
    result = evaluate_recall_at_k(
        eval_data,
        retrieve_fn=lambda q: {"q1": ["x", "y"], "q2": ["b", "x"]}[q],
        k=5,
    )
    assert abs(result["value"] - 0.25) < 1e-6


def test_evaluate_recall_at_k_empty():
    result = evaluate_recall_at_k([], retrieve_fn=lambda q: [], k=5)
    assert result["value"] == 0.0
    assert result["num_queries"] == 0

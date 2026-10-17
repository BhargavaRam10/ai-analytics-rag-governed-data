"""Offline evaluation: does retrieval return the right governed doc?

Each eval case is (question, expected_doc_id). The eval builds the retriever
with the default embedder, runs every question, and reports top-1 accuracy.
Run: python -m src.eval

This is the honest version of "AI evals" for a portfolio: small, explainable,
and runnable in CI with zero API cost.
"""
from __future__ import annotations

from .answer import answer_question
from .embeddings import default_embedder
from .retrieval import Retriever

EVAL_CASES: list[tuple[str, str]] = [
    ("How is Missed Pickup % calculated?", "mtr_missed_pickup_rate"),
    ("What is the grain of fact_pickups?", "tbl_fact_pickups"),
    ("What does a quarantine table hold?", "gls_quarantine"),
    ("What is the diversion rate formula?", "mtr_diversion_rate"),
    ("Which columns does dim_route have?", "tbl_dim_route"),
]


def main() -> None:
    embedder = default_embedder()
    retriever = Retriever.build(embedder)
    correct = 0
    for question, expected in EVAL_CASES:
        answer = answer_question(question, retriever, embedder, top_k=1)
        top_doc = answer.citations[0] if answer.citations else None
        ok = top_doc == expected
        correct += ok
        print(f"{'PASS' if ok else 'FAIL'}  {question}\n      -> {top_doc} (expected {expected})")
    print(f"\nTop-1 retrieval accuracy: {correct}/{len(EVAL_CASES)}")


if __name__ == "__main__":
    main()

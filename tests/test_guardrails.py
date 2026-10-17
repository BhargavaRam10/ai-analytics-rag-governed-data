"""Guardrail tests: the system must refuse when the corpus has no evidence."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.answer import answer_question
from src.chunking import chunk_corpus
from src.embeddings import MockEmbedder
from src.retrieval import Retriever

DOCS = [
    {"doc_id": "m1", "title": "Missed pickup metric",
     "body": "Missed Pickup % equals missed pickups divided by pickup count."},
]


def _retriever() -> tuple[Retriever, MockEmbedder]:
    embedder = MockEmbedder()
    chunks = chunk_corpus(DOCS)
    vectors = embedder.embed([c.text for c in chunks])
    return Retriever(chunks, vectors), embedder


def test_refuses_question_with_no_evidence():
    retriever, embedder = _retriever()
    answer = answer_question(
        "What is the capital of France?", retriever, embedder, min_score=0.99
    )
    assert answer.refused
    assert answer.citations == []


def test_refusal_explains_itself():
    retriever, embedder = _retriever()
    answer = answer_question(
        "Explain quantum computing", retriever, embedder, min_score=0.99
    )
    assert answer.refused
    assert "governed" in answer.text.lower()


def test_in_scope_question_is_not_refused():
    retriever, embedder = _retriever()
    answer = answer_question(
        "How is Missed Pickup % calculated?", retriever, embedder, min_score=0.0
    )
    assert not answer.refused
    assert "[sources:" in answer.text

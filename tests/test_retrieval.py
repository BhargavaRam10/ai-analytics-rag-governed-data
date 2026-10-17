"""Tests for retrieval: relevant doc ranks first, scores sane, citations present."""
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
    {"doc_id": "t1", "title": "Fact table docs",
     "body": "fact_pickups stores one row per route per day with tonnage measures."},
]


def _retriever() -> tuple[Retriever, MockEmbedder]:
    embedder = MockEmbedder()
    chunks = chunk_corpus(DOCS)
    vectors = embedder.embed([c.text for c in chunks])
    return Retriever(chunks, vectors), embedder


def test_relevant_doc_ranks_first():
    retriever, embedder = _retriever()
    hits = retriever.search("How is Missed Pickup % calculated?", embedder, top_k=2)
    assert hits[0].chunk.doc_id == "m1"


def test_scores_are_bounded():
    retriever, embedder = _retriever()
    hits = retriever.search("tonnage measures", embedder, top_k=2)
    # cosine in [-1, 1] plus a lexical bonus of at most 0.5
    assert all(-1.0 <= h.score <= 1.5 for h in hits)


def test_answer_cites_sources():
    retriever, embedder = _retriever()
    answer = answer_question("How is Missed Pickup % calculated?", retriever, embedder)
    assert not answer.refused
    assert answer.citations, "answer must cite its governed sources"
    assert "m1" in answer.citations

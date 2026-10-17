"""Answer layer with governed-source guardrails.

Rules (see docs/guardrails.md for the rationale):
  1. Answer ONLY from retrieved governed chunks — never from model memory.
  2. Every factual claim cites its source doc_id.
  3. If the top retrieval score is below MIN_SCORE, refuse: the corpus has no
     evidence for this question, and guessing is worse than saying so.

The offline answerer extracts the most relevant sentences from retrieved
chunks (extractive QA). In production, replace _compose with an LLM call that
receives ONLY the retrieved chunks as context — the guardrails don't change.
"""
from __future__ import annotations

from dataclasses import dataclass

from .embeddings import Embedder
from .retrieval import Hit, Retriever

MIN_SCORE = 0.15


@dataclass
class Answer:
    text: str
    citations: list[str]
    refused: bool = False


def _compose(hits: list[Hit]) -> str:
    """Extractive answer: stitch the best sentences from governed chunks."""
    sentences: list[str] = []
    for hit in hits:
        first = hit.chunk.text.split(". ")[0].strip()
        if first and first not in sentences:
            sentences.append(first)
    return ". ".join(sentences) + "."


def answer_question(
    question: str,
    retriever: Retriever,
    embedder: Embedder,
    top_k: int = 3,
    min_score: float = MIN_SCORE,
) -> Answer:
    hits = retriever.search(question, embedder, top_k=top_k)
    if not hits or hits[0].score < min_score:
        return Answer(
            text=(
                "I can't answer that from the governed data catalog — no "
                "document covers this topic. Refusing is by design: this "
                "system never answers from outside its sources."
            ),
            citations=[],
            refused=True,
        )
    citations = sorted({h.chunk.doc_id for h in hits})
    text = _compose(hits)
    cited = f"{text} [sources: {', '.join(citations)}]"
    return Answer(text=cited, citations=citations)

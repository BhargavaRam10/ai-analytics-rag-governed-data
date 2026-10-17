"""Retrieval over the governed corpus with source citations.

Cosine-similarity search over chunk embeddings. Every hit returns the chunk
text PLUS its doc_id/title, so the answer layer can cite exactly which
governed document each claim came from — the core anti-hallucination control.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from .chunking import Chunk, chunk_corpus
from .embeddings import Embedder

INDEX_DIR = Path(__file__).resolve().parent.parent / "data" / "index"
CORPUS_DIR = Path(__file__).resolve().parent.parent / "data" / "corpus"


@dataclass
class Hit:
    chunk: Chunk
    score: float


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def _token_overlap_bonus(query: str, text: str) -> float:
    """Lexical signal: shared content words boost the ranking.

    MockEmbedder is hash-based and semantically weak, so a small term-overlap
    bonus keeps the offline demo meaningful. A real embedding provider makes
    this unnecessary — it is documented, not hidden.
    """
    stopwords = {"the", "a", "an", "is", "of", "what", "how", "which", "does",
                 "do", "in", "to", "for", "it", "and", "or"}
    q = {w.strip("?.,") for w in query.lower().split()} - stopwords
    t = {w.strip("?.,") for w in text.lower().split()} - stopwords
    if not q:
        return 0.0
    return 0.5 * len(q & t) / len(q)


class Retriever:
    def __init__(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        assert len(chunks) == len(vectors)
        self.chunks = chunks
        self.vectors = vectors

    @classmethod
    def build(cls, embedder: Embedder) -> "Retriever":
        docs: list[dict] = []
        for path in sorted(CORPUS_DIR.glob("*.json")):
            docs.extend(json.loads(path.read_text()))
        chunks = chunk_corpus(docs)
        vectors = embedder.embed([c.text for c in chunks])
        return cls(chunks, vectors)

    def search(self, query: str, embedder: Embedder, top_k: int = 3) -> list[Hit]:
        qvec = embedder.embed([query])[0]
        scored = [
            Hit(chunk=c,
                score=cosine(qvec, v) + _token_overlap_bonus(query, c.text))
            for c, v in zip(self.chunks, self.vectors)
        ]
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:top_k]

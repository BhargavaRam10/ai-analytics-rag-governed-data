"""Document chunking with a token budget.

Governance docs vary in length; retrieval works on fixed-size chunks so a
long table doc doesn't drown out short metric definitions. Chunks are cut on
sentence boundaries with a configurable overlap, and each chunk carries its
source doc_id so answers can cite where claims came from.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    doc_id: str
    title: str
    text: str
    chunk_index: int


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token. Good enough for budgeting."""
    return max(1, len(text) // 4)


def chunk_document(
    doc_id: str,
    title: str,
    body: str,
    max_tokens: int = 200,
    overlap_sentences: int = 1,
) -> list[Chunk]:
    """Split a document into sentence-boundary chunks under a token budget."""
    sentences = [s for s in SENTENCE_RE.split(body.strip()) if s]
    chunks: list[Chunk] = []
    current: list[str] = []
    index = 0

    def flush() -> None:
        nonlocal index
        if current:
            chunks.append(Chunk(doc_id, title, " ".join(current), index))
            index += 1

    for sentence in sentences:
        candidate = current + [sentence]
        if estimate_tokens(" ".join(candidate)) > max_tokens and current:
            flush()
            current = current[-overlap_sentences:] if overlap_sentences else []
        current.append(sentence)
    flush()
    return chunks


def chunk_corpus(docs: list[dict], **kwargs) -> list[Chunk]:
    """Chunk every document in a corpus list of {doc_id, title, body}."""
    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(
            chunk_document(doc["doc_id"], doc["title"], doc["body"], **kwargs)
        )
    return chunks

"""Embedding abstraction with a deterministic local fallback.

The system prefers a real embedding API when EMBEDDING_API_KEY is set, but
runs fully offline with MockEmbedding (hash-based, deterministic vectors) so
the project is runnable and testable with zero credentials and zero cost.
Both implement the same interface, so swapping providers changes one line.
"""
from __future__ import annotations

import hashlib
import math
import os
from abc import ABC, abstractmethod

DIM = 64


class Embedder(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text."""


class MockEmbedder(Embedder):
    """Deterministic hash-based embeddings. No network, no API key.

    NOT for production semantic quality — it exists so the full pipeline
    (index -> retrieve -> answer -> evaluate) runs anywhere, including CI.
    """

    def __init__(self, dim: int = DIM) -> None:
        self.dim = dim

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for word in text.lower().split():
            digest = hashlib.md5(word.encode()).digest()
            for i in range(self.dim):
                vec[i] += ((digest[i % len(digest)] / 255.0) - 0.5)
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]


class APIEmbedder(Embedder):
    """Placeholder for a real embedding provider (e.g. OpenAI).

    Wired behind an interface so production can swap it in without touching
    retrieval or answer code. Raises a clear error if the key is missing.
    """

    def __init__(self, api_key: str | None = None, model: str = "text-embedding-3-small") -> None:
        self.api_key = api_key or os.environ.get("EMBEDDING_API_KEY")
        self.model = model
        if not self.api_key:
            raise ValueError(
                "EMBEDDING_API_KEY is not set — use MockEmbedder for offline runs"
            )

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError(
            "Wire your provider's SDK here; interface is stable by design."
        )


def default_embedder() -> Embedder:
    """Mock offline unless a real key is configured."""
    if os.environ.get("EMBEDDING_API_KEY"):
        return APIEmbedder()
    return MockEmbedder()

"""Precompute and persist the retrieval index.

Saves chunks + vectors to data/index/ so the app can load them without
re-embedding the corpus on every start. Run: python -m src.build_index
"""
from __future__ import annotations

import json
from dataclasses import asdict

from .chunking import chunk_corpus
from .embeddings import default_embedder
from .retrieval import CORPUS_DIR, INDEX_DIR


def main() -> None:
    docs: list[dict] = []
    for path in sorted(CORPUS_DIR.glob("*.json")):
        docs.extend(json.loads(path.read_text()))
    embedder = default_embedder()
    chunks = chunk_corpus(docs)
    vectors = embedder.embed([c.text for c in chunks])

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    (INDEX_DIR / "chunks.json").write_text(
        json.dumps([asdict(c) for c in chunks], indent=2)
    )
    (INDEX_DIR / "vectors.json").write_text(json.dumps(vectors))
    print(f"Indexed {len(chunks)} chunks -> {INDEX_DIR}")


if __name__ == "__main__":
    main()

"""Tests for document chunking: budget respected, sentences intact, sources tracked."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunking import chunk_document, estimate_tokens

BODY = (
    "fact_pickups is the operations fact table at one row per route per day. "
    "Grain: (report_date, route_id). Key measures: pickup_count, total_tonnage, "
    "missed_pickups. The missed_pickup_rate column is computed as "
    "div0(missed_pickups, pickup_count). Owner: data engineering."
)


def test_chunks_respect_token_budget():
    chunks = chunk_document("d1", "Title", BODY, max_tokens=30)
    assert len(chunks) > 1
    for chunk in chunks:
        assert estimate_tokens(chunk.text) <= 60  # budget + one sentence overshoot


def test_no_sentence_is_split_midway():
    chunks = chunk_document("d1", "Title", BODY, max_tokens=40)
    joined = " ".join(c.text for c in chunks)
    assert "fact_pickups is the operations fact table" in joined
    assert "Owner: data engineering." in joined


def test_every_chunk_carries_source():
    chunks = chunk_document("tbl_fact_pickups", "Fact docs", BODY, max_tokens=30)
    assert all(c.doc_id == "tbl_fact_pickups" for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

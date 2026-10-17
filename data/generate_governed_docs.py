"""Synthetic governed-data corpus generator.

Creates the "governed warehouse" the RAG system answers from:
  - data/corpus/table_docs.json   — per-table documentation (owner, grain, key columns)
  - data/corpus/metric_defs.json  — metric definitions (formula, owner, refresh cadence)
  - data/corpus/glossary.json     — business-term glossary

These play the role of a real data catalog / dbt docs site. Fixed seed, fully
synthetic — no employer data.
"""
from __future__ import annotations

import json
from pathlib import Path

CORPUS = Path(__file__).resolve().parent / "corpus"

TABLE_DOCS = [
    {
        "doc_id": "tbl_fact_pickups",
        "title": "fact_pickups — fact table documentation",
        "body": (
            "fact_pickups is the operations fact table at one row per route per "
            "day. Grain: (report_date, route_id). Key measures: pickup_count, "
            "total_tonnage, missed_pickups. The missed_pickup_rate column is "
            "computed as div0(missed_pickups, pickup_count). Owner: data "
            "engineering. Refresh: nightly after the quality gate passes."
        ),
    },
    {
        "doc_id": "tbl_dim_route",
        "title": "dim_route — route dimension documentation",
        "body": (
            "dim_route is a slowly changing dimension (type 1) with one row per "
            "route_id. Attributes: zone, truck_id, driver_name, stops_planned. "
            "Truck and driver reassignments overwrite history; zone history is "
            "not tracked. Owner: data engineering. Source: route master file drop."
        ),
    },
    {
        "doc_id": "tbl_zone_kpis",
        "title": "zone_kpis — zone KPI snapshot documentation",
        "body": (
            "zone_kpis aggregates fact_pickups to one row per zone per day. "
            "Columns: active_routes, total_pickups, total_tonnage, "
            "missed_pickups, missed_pickup_rate, avg_tonnage_per_route. "
            "This is the table leadership dashboards read. Owner: analytics."
        ),
    },
]

METRIC_DEFS = [
    {
        "doc_id": "mtr_missed_pickup_rate",
        "title": "Metric: Missed Pickup %",
        "body": (
            "Missed Pickup % = SUM(missed_pickups) / SUM(pickup_count), computed "
            "in the warehouse table fact_pickups as missed_pickup_rate. It is "
            "the headline reliability KPI for operations. Target: below 5%. "
            "Owner: operations analytics. Refresh: nightly."
        ),
    },
    {
        "doc_id": "mtr_diversion_rate",
        "title": "Metric: Diversion Rate %",
        "body": (
            "Diversion Rate % = (SUM(recycle_tonnage) + SUM(organics_tonnage)) / "
            "SUM(total_tonnage). It measures the share of collected tonnage "
            "diverted from landfill. Owner: sustainability reporting. "
            "Refresh: nightly."
        ),
    },
]

GLOSSARY = [
    {
        "doc_id": "gls_route_day",
        "title": "Glossary: route-day grain",
        "body": (
            "Route-day grain means one record per route per calendar day. All "
            "operations KPIs (tonnage, missed pickups, reliability) are reported "
            "at route-day grain; per-pickup detail lives in staging for "
            "drill-through only."
        ),
    },
    {
        "doc_id": "gls_quarantine",
        "title": "Glossary: quarantine table",
        "body": (
            "A quarantine table (silver.<table>_quarantine) holds rows that "
            "failed data-quality rules during the bronze-to-silver transform, "
            "with a quarantined_at timestamp. Quarantined rows are reviewed by "
            "data engineering and never silently dropped."
        ),
    },
]


def main() -> None:
    CORPUS.mkdir(parents=True, exist_ok=True)
    for name, docs in (
        ("table_docs.json", TABLE_DOCS),
        ("metric_defs.json", METRIC_DEFS),
        ("glossary.json", GLOSSARY),
    ):
        (CORPUS / name).write_text(json.dumps(docs, indent=2))
        print(f"Wrote {len(docs)} docs to corpus/{name}")


if __name__ == "__main__":
    main()

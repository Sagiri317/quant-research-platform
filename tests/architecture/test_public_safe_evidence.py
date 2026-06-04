from __future__ import annotations

import json
from pathlib import Path

REQUIRED_AREAS = {
    "processed_panel_parity",
    "factor_value_parity",
    "label_value_parity",
    "masks_universes_parity",
    "research_dataset_parity",
    "evaluation_parity",
    "prediction_oos_smoke_parity",
    "supporting_layers_migration_2014_2025",
    "full_historical_daily_panel_status",
}

REQUIRED_FIELDS = {
    "area",
    "contract_version",
    "privacy_level",
    "period",
    "row_count",
    "key_count",
    "column_count",
    "duplicate_key_count",
    "column_set_hash",
    "dtype_summary_hash",
    "null_summary_hash",
    "aggregate_checksum",
    "max_abs_delta",
    "mean_abs_delta",
    "tolerance",
    "accepted_explained_delta",
    "explanation",
}

FORBIDDEN_PUBLIC_EVIDENCE_PATTERNS = [
    "csmar",
    "CSMAR",
    "C:/",
    "C:\\",
    "/mnt/",
    "/home/",
    "/Users/",
    "data/raw",
    "data/interim",
    "data/processed",
    "reports/private_real",
    "000001",
    "600000",
    "raw row",
    "row sample:",
    "token=",
    "provider_id",
    "source_id",
]


def test_clean_rewrite_public_safe_parity_evidence_json_schema() -> None:
    path = Path("docs/reports/clean-rewrite-0.1-parity-evidence.json")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["contract_version"] == "0.1"
    assert payload["privacy_level"] == "public_safe_aggregate"
    assert {item["area"] for item in payload["areas"]} == REQUIRED_AREAS
    for item in payload["areas"]:
        assert REQUIRED_FIELDS.issubset(item)
        assert item["contract_version"] == "0.1"
        assert item["privacy_level"] == "public_safe_aggregate"


def test_clean_rewrite_public_safe_parity_evidence_has_no_private_samples() -> None:
    paths = [
        Path("docs/reports/clean-rewrite-0.1-parity-evidence.md"),
        Path("docs/reports/clean-rewrite-0.1-parity-evidence.json"),
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in paths)

    for term in FORBIDDEN_PUBLIC_EVIDENCE_PATTERNS:
        assert term not in combined

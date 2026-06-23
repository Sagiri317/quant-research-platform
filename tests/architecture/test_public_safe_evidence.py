from __future__ import annotations

import json
import re
from pathlib import Path

from quant_research_platform.reports.verification_evidence import (
    render_private_verification_summary_markdown,
    render_public_safe_evidence_markdown,
)

DOC_MARKET_PANEL_LOGICAL_ROOT = (
    "processed/a_share/panel/<market_panel_data_id(DataFrequency.DAILY)>"
)
DOC_FULL_HISTORICAL_MARKET_PANEL_STATUS_AREA = (
    "full_historical_market_panel_data_id_daily_status"
)

REQUIRED_AREAS = {
    "processed_panel_parity",
    "factor_value_parity",
    "label_value_parity",
    "masks_universes_parity",
    "research_dataset_parity",
    "evaluation_parity",
    "prediction_oos_smoke_parity",
    "supporting_layers_coverage_2014_2025",
    DOC_FULL_HISTORICAL_MARKET_PANEL_STATUS_AREA,
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
    "verification_kind",
    "verification_scope",
    "evidence_generation_mode",
    "tolerance_owner",
    "downstream_impact",
    "generated_by",
    "source_data_visibility",
    "row_level_output_committed",
    "human_approval_required",
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
    "data/research",
    "reports/private_research",
    "000001",
    "600000",
    "raw row",
    "row sample:",
    "token=",
    "provider_id",
    "source_id",
    "implementation_package",
    "placeholder",
]
HASH_RE = re.compile(r"^[0-9a-f]{64}$")
HASH_FIELDS = {
    "column_set_hash",
    "dtype_summary_hash",
    "null_summary_hash",
    "aggregate_checksum",
}
SUMMARY_TOP_LEVEL_FIELDS = {
    "contract_version",
    "project_version",
    "verification_kind",
    "generated_at",
    "privacy_level",
    "source_data_visibility",
    "row_level_output_committed",
    "dataset_inventory",
    "covered_roots",
    "uncovered_roots",
    "allowed_unverified_roots",
    "merge_ready",
}
INVENTORY_FIELDS = {
    "kind",
    "artifact_id",
    "logical_root",
    "period",
    "row_count",
    "column_count",
    "key_count",
    "duplicate_key_count",
    "column_set_hash",
    "dtype_summary_hash",
    "null_summary_hash",
    "aggregate_checksum",
    "covered_by_evidence_area",
}
PLACEHOLDER_HASH_VALUES = {"passed", "accepted"}
STATUS_TOKENS = {"PASS", "FAIL", "NOT_RUN", "BLOCKED"}


def _hash(seed: str) -> str:
    import hashlib

    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def _synthetic_area(area: str, index: int) -> dict[str, object]:
    period = (
        "2014-01-01 through 2025-12-31"
        if area == DOC_FULL_HISTORICAL_MARKET_PANEL_STATUS_AREA
        else "2024-01-01 through 2024-01-31"
    )
    return {
        "area": area,
        "contract_version": "0.1",
        "privacy_level": "public_safe_aggregate",
        "period": period,
        "row_count": index + 1,
        "key_count": index + 1,
        "column_count": 3,
        "duplicate_key_count": 0,
        "column_set_hash": _hash(f"{area}-columns"),
        "dtype_summary_hash": _hash(f"{area}-dtypes"),
        "null_summary_hash": _hash(f"{area}-nulls"),
        "aggregate_checksum": _hash(f"{area}-aggregate"),
        "key_value_paired_summary_hash": _hash(f"{area}-key-values"),
        "max_abs_delta": 0.0,
        "mean_abs_delta": 0.0,
        "tolerance": 0.0,
        "accepted_explained_delta": False,
        "explanation": "",
        "verification_kind": "private_research_data_aggregate",
        "verification_scope": "aggregate_public_safe_parity",
        "evidence_generation_mode": "local_private_research_public_safe_aggregate",
        "tolerance_owner": "",
        "downstream_impact": "none",
        "generated_by": "synthetic_test_fixture",
        "source_data_visibility": "private_local_only",
        "row_level_output_committed": False,
        "human_approval_required": False,
        "status": "PASS",
    }


def _synthetic_public_safe_evidence_payload() -> dict[str, object]:
    return {
        "contract_version": "0.1",
        "privacy_level": "public_safe_aggregate",
        "artifact_id": "private_research_aggregate_evidence",
        "market_id": "a_share",
        "verification_kind": "private_research_data_aggregate",
        "verification_scope": "aggregate_public_safe_parity",
        "evidence_generation_mode": "local_private_research_public_safe_aggregate",
        "source_data_visibility": "private_local_only",
        "row_level_output_committed": False,
        "merge_ready": False,
        "status": "PASS",
        "areas": [
            _synthetic_area(area, index)
            for index, area in enumerate(sorted(REQUIRED_AREAS))
        ],
    }


def _synthetic_private_verification_summary_payload() -> dict[str, object]:
    inventory = [
        {
            "kind": "dataset",
            "artifact_id": "market_panel_daily",
            "logical_root": DOC_MARKET_PANEL_LOGICAL_ROOT,
            "period": "2014-01-01 through 2025-12-31",
            "row_count": 10,
            "column_count": 3,
            "key_count": 10,
            "duplicate_key_count": 0,
            "column_set_hash": _hash("inventory-columns"),
            "dtype_summary_hash": _hash("inventory-dtypes"),
            "null_summary_hash": _hash("inventory-nulls"),
            "aggregate_checksum": _hash("inventory-aggregate"),
            "key_value_paired_summary_hash": _hash("inventory-key-values"),
            "covered_by_evidence_area": DOC_FULL_HISTORICAL_MARKET_PANEL_STATUS_AREA,
        }
    ]
    return {
        "contract_version": "0.1",
        "project_version": "0.1.0",
        "verification_kind": "private_research_data_aggregate_inventory",
        "generated_at": "2026-01-01T00:00:00+00:00",
        "privacy_level": "public_safe_aggregate",
        "source_data_visibility": "private_local_only",
        "row_level_output_committed": False,
        "dataset_inventory": inventory,
        "covered_roots": [item["logical_root"] for item in inventory],
        "uncovered_roots": [],
        "allowed_unverified_roots": [],
        "merge_ready": False,
        "status": "PASS",
    }


def _write_public_safe_evidence_files(tmp_path: Path) -> tuple[Path, Path]:
    payload = _synthetic_public_safe_evidence_payload()
    json_path = tmp_path / "public-safe-evidence.json"
    md_path = tmp_path / "public-safe-evidence.md"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    md_path.write_text(
        render_public_safe_evidence_markdown(payload),
        encoding="utf-8",
        newline="\n",
    )
    return md_path, json_path


def _write_private_summary_files(tmp_path: Path) -> tuple[Path, Path]:
    payload = _synthetic_private_verification_summary_payload()
    json_path = tmp_path / "private-verification-summary.json"
    md_path = tmp_path / "private-verification-summary.md"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    md_path.write_text(
        render_private_verification_summary_markdown(payload),
        encoding="utf-8",
        newline="\n",
    )
    return md_path, json_path


def test_current_public_safe_parity_evidence_json_schema() -> None:
    payload = _synthetic_public_safe_evidence_payload()

    assert payload["contract_version"] == "0.1"
    assert payload["privacy_level"] == "public_safe_aggregate"
    assert payload["artifact_id"] == "private_research_aggregate_evidence"
    assert payload["market_id"] == "a_share"
    assert payload["verification_kind"] == "private_research_data_aggregate"
    assert payload["verification_scope"] == "aggregate_public_safe_parity"
    assert payload["source_data_visibility"] == "private_local_only"
    assert payload["row_level_output_committed"] is False
    assert payload["status"] in STATUS_TOKENS
    assert len(payload["areas"]) == 9
    assert {item["area"] for item in payload["areas"]} == REQUIRED_AREAS
    for item in payload["areas"]:
        assert REQUIRED_FIELDS.issubset(item)
        assert item["status"] in STATUS_TOKENS
        assert item["contract_version"] == "0.1"
        assert item["privacy_level"] == "public_safe_aggregate"
        for field in HASH_FIELDS:
            value = item[field]
            assert isinstance(value, str)
            assert HASH_RE.fullmatch(value)
            assert not value.startswith("sha256:")
            assert value not in PLACEHOLDER_HASH_VALUES
        assert item["tolerance"] >= 0
        if item["tolerance"] == 0:
            if item["human_approval_required"]:
                assert item["max_abs_delta"] > 0 or item["mean_abs_delta"] > 0
            else:
                assert item["max_abs_delta"] == 0
                assert item["mean_abs_delta"] == 0
        assert item["verification_kind"] == "private_research_data_aggregate"
        assert item["verification_scope"] == "aggregate_public_safe_parity"
        assert (
            item["evidence_generation_mode"]
            == "local_private_research_public_safe_aggregate"
        )
        assert item["source_data_visibility"] == "private_local_only"
        assert item["row_level_output_committed"] is False
        assert isinstance(item["human_approval_required"], bool)
        if item["accepted_explained_delta"] is True:
            assert item["explanation"]
            assert item["tolerance_owner"]
            assert item.get("explanation_details", {}).get("delta_type") != "none"
        if item["max_abs_delta"] > item["tolerance"]:
            assert item["accepted_explained_delta"] or item["human_approval_required"]
            assert item["explanation"]


def test_current_public_safe_parity_evidence_is_aggregate_status_payload() -> None:
    payload = _synthetic_public_safe_evidence_payload()
    text = json.dumps(payload, ensure_ascii=False).lower()

    assert "metadata verification" not in text
    assert payload["verification_kind"] == "private_research_data_aggregate"
    assert payload["evidence_generation_mode"] == (
        "local_private_research_public_safe_aggregate"
    )
    assert payload["row_level_output_committed"] is False
    assert payload["status"] in STATUS_TOKENS
    assert payload["merge_ready"] is False


def test_full_historical_market_panel_daily_covers_full_2014_2025_period() -> None:
    payload = _synthetic_public_safe_evidence_payload()
    area = next(
        item
        for item in payload["areas"]
        if item["area"] == DOC_FULL_HISTORICAL_MARKET_PANEL_STATUS_AREA
    )

    assert area["period"] == "2014-01-01 through 2025-12-31"
    assert area["human_approval_required"] is False
    assert area["accepted_explained_delta"] is False
    assert payload["merge_ready"] is False
    assert payload["status"] in STATUS_TOKENS


def test_public_safe_parity_evidence_has_no_private_samples(tmp_path: Path) -> None:
    paths = _write_public_safe_evidence_files(tmp_path)
    combined = "\n".join(path.read_text(encoding="utf-8") for path in paths)

    for term in FORBIDDEN_PUBLIC_EVIDENCE_PATTERNS:
        assert term not in combined


def test_public_safe_evidence_forbids_physical_research_root_not_logical_root() -> None:
    assert "data/research" in FORBIDDEN_PUBLIC_EVIDENCE_PATTERNS
    assert "research/a_share/demo" not in FORBIDDEN_PUBLIC_EVIDENCE_PATTERNS


def test_public_safe_parity_evidence_uses_lf_newlines(tmp_path: Path) -> None:
    paths = _write_public_safe_evidence_files(tmp_path)

    for path in paths:
        content = path.read_bytes()
        assert b"\r\n" not in content
        assert content.endswith(b"\n")


def test_private_verification_summary_json_schema_and_coverage() -> None:
    payload = _synthetic_private_verification_summary_payload()

    assert SUMMARY_TOP_LEVEL_FIELDS.issubset(payload)
    assert payload["contract_version"] == "0.1"
    assert payload["verification_kind"] == "private_research_data_aggregate_inventory"
    assert payload["privacy_level"] == "public_safe_aggregate"
    assert payload["source_data_visibility"] == "private_local_only"
    assert payload["row_level_output_committed"] is False
    assert payload["status"] in STATUS_TOKENS
    assert payload["merge_ready"] is False
    assert payload["uncovered_roots"] == []
    assert payload["allowed_unverified_roots"] == []
    assert payload["covered_roots"] == [
        item["logical_root"] for item in payload["dataset_inventory"]
    ]

    for item in payload["dataset_inventory"]:
        assert INVENTORY_FIELDS.issubset(item)
        assert item["logical_root"].startswith(
            ("processed/a_share/", "research/a_share/")
        )
        assert "data/processed" not in item["logical_root"]
        assert "data/research" not in item["logical_root"]
        assert item["row_count"] > 0
        assert item["column_count"] > 0
        assert item["key_count"] > 0
        assert item["duplicate_key_count"] == 0
        assert item["covered_by_evidence_area"]
        for field in HASH_FIELDS:
            assert HASH_RE.fullmatch(item[field])


def test_private_verification_summary_covers_full_historical_panel() -> None:
    payload = _synthetic_private_verification_summary_payload()
    market_panel_daily = next(
        item
        for item in payload["dataset_inventory"]
        if item["logical_root"] == DOC_MARKET_PANEL_LOGICAL_ROOT
    )

    assert market_panel_daily["period"] == "2014-01-01 through 2025-12-31"
    assert (
        market_panel_daily["covered_by_evidence_area"]
        == DOC_FULL_HISTORICAL_MARKET_PANEL_STATUS_AREA
    )


def test_private_verification_summary_has_no_private_samples(tmp_path: Path) -> None:
    paths = _write_private_summary_files(tmp_path)
    combined = "\n".join(path.read_text(encoding="utf-8") for path in paths)

    for term in FORBIDDEN_PUBLIC_EVIDENCE_PATTERNS:
        assert term not in combined


def test_private_verification_summary_uses_lf_newlines(tmp_path: Path) -> None:
    paths = _write_private_summary_files(tmp_path)

    for path in paths:
        content = path.read_bytes()
        assert b"\r\n" not in content
        assert content.endswith(b"\n")

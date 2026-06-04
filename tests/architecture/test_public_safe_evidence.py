from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

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
    "reports/private_real",
    "000001",
    "600000",
    "raw row",
    "row sample:",
    "token=",
    "provider_id",
    "source_id",
]
HASH_RE = re.compile(r"^[0-9a-f]{64}$")
HASH_FIELDS = {
    "column_set_hash",
    "dtype_summary_hash",
    "null_summary_hash",
    "aggregate_checksum",
}
PLACEHOLDER_HASH_VALUES = {"passed", "accepted"}


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
        assert item["verification_kind"] == "private_real_data_aggregate"
        assert item["verification_scope"] == "aggregate_public_safe_parity"
        assert (
            item["evidence_generation_mode"]
            == "local_private_data_public_safe_aggregate"
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


def test_clean_rewrite_public_safe_parity_evidence_is_real_aggregate() -> None:
    path = Path("docs/reports/clean-rewrite-0.1-parity-evidence.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    text = json.dumps(payload, ensure_ascii=False).lower()

    assert "metadata verification" not in text
    assert payload["verification_kind"] == "private_real_data_aggregate"
    assert payload["evidence_generation_mode"] == (
        "local_private_data_public_safe_aggregate"
    )
    assert payload["row_level_output_committed"] is False


def test_full_historical_daily_panel_covers_full_period_or_requires_approval() -> None:
    path = Path("docs/reports/clean-rewrite-0.1-parity-evidence.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    area = next(
        item
        for item in payload["areas"]
        if item["area"] == "full_historical_daily_panel_status"
    )

    if area["period"] == "2014-01-01 through 2025-12-31":
        assert area["human_approval_required"] is False
    else:
        assert area["human_approval_required"] is True
        assert area["accepted_explained_delta"] is False
        assert payload["merge_ready"] is False


def test_full_historical_daily_panel_summary_streams_generated_chunks(
    monkeypatch,
) -> None:
    from quant_research_platform.reports import private_aggregate_verification

    frames = (
        pd.DataFrame(
            {
                "market_id": ["a_share"],
                "asset_id": ["asset-a"],
                "timestamp": [pd.Timestamp("2014-01-01")],
            }
        ),
        pd.DataFrame(
            {
                "market_id": ["a_share"],
                "asset_id": ["asset-b"],
                "timestamp": [pd.Timestamp("2025-12-31")],
            }
        ),
    )

    def iter_chunks(project_root: Path):
        assert project_root == Path(".")
        return iter(frames)

    monkeypatch.setattr(
        private_aggregate_verification,
        "_iter_generated_full_daily_panel_chunks",
        iter_chunks,
    )

    summary = private_aggregate_verification._summarize_generated_full_daily_panel(
        Path(".")
    )

    assert summary.row_count == 2
    assert summary.key_count == 2
    assert summary.period == "2014-01-01 through 2025-12-31"


def test_full_historical_daily_panel_generation_prefers_full_frame(
    monkeypatch,
) -> None:
    from quant_research_platform.reports import private_aggregate_verification

    full_frame = pd.DataFrame(
        {
            "market_id": ["a_share"],
            "asset_id": ["asset-a"],
            "timestamp": [pd.Timestamp("2014-01-01")],
        }
    )

    def build_full_frame(project_root: Path):
        assert project_root == Path(".")
        return full_frame

    def iter_stock_chunks(project_root: Path):
        raise AssertionError("stock chunk fallback should not run")
        yield pd.DataFrame()

    monkeypatch.setattr(
        private_aggregate_verification,
        "_build_generated_full_daily_panel_frame",
        build_full_frame,
    )
    monkeypatch.setattr(
        private_aggregate_verification,
        "_iter_generated_full_daily_panel_stock_chunks",
        iter_stock_chunks,
    )

    frames = list(
        private_aggregate_verification._iter_generated_full_daily_panel_chunks(
            Path(".")
        )
    )

    assert frames == [full_frame]


def test_full_historical_daily_panel_generation_falls_back_to_stock_chunks(
    monkeypatch,
) -> None:
    from quant_research_platform.reports import private_aggregate_verification

    chunk = pd.DataFrame(
        {
            "market_id": ["a_share"],
            "asset_id": ["asset-a"],
            "timestamp": [pd.Timestamp("2014-01-01")],
        }
    )

    def build_full_frame(project_root: Path):
        assert project_root == Path(".")
        raise MemoryError

    def iter_stock_chunks(project_root: Path):
        assert project_root == Path(".")
        return iter((chunk,))

    monkeypatch.setattr(
        private_aggregate_verification,
        "_build_generated_full_daily_panel_frame",
        build_full_frame,
    )
    monkeypatch.setattr(
        private_aggregate_verification,
        "_iter_generated_full_daily_panel_stock_chunks",
        iter_stock_chunks,
    )

    frames = list(
        private_aggregate_verification._iter_generated_full_daily_panel_chunks(
            Path(".")
        )
    )

    assert frames == [chunk]


def test_dynamic_stock_chunk_size_uses_one_chunk_when_memory_fits() -> None:
    from quant_research_platform.reports import private_aggregate_verification

    chunk_size = private_aggregate_verification._dynamic_stock_chunk_size(
        total_stock_count=1200,
        estimated_full_frame_bytes=10_000,
        available_memory_bytes=100_000,
        memory_fraction=0.5,
    )

    assert chunk_size == 1200


def test_dynamic_stock_chunk_size_splits_when_memory_is_constrained() -> None:
    from quant_research_platform.reports import private_aggregate_verification

    chunk_size = private_aggregate_verification._dynamic_stock_chunk_size(
        total_stock_count=1200,
        estimated_full_frame_bytes=120_000,
        available_memory_bytes=10_000,
        memory_fraction=0.5,
    )

    assert 1 <= chunk_size < 1200


def test_generated_panel_memory_budget_allows_current_full_frame_scale() -> None:
    from quant_research_platform.reports import private_aggregate_verification

    chunk_size = private_aggregate_verification._dynamic_stock_chunk_size(
        total_stock_count=5420,
        estimated_full_frame_bytes=4_378_832_336,
        available_memory_bytes=8_961_126_400,
        memory_fraction=private_aggregate_verification.GENERATED_PANEL_MEMORY_FRACTION,
    )

    assert chunk_size == 5420


def test_clean_rewrite_public_safe_parity_evidence_has_no_private_samples() -> None:
    paths = [
        Path("docs/reports/clean-rewrite-0.1-parity-evidence.md"),
        Path("docs/reports/clean-rewrite-0.1-parity-evidence.json"),
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in paths)

    for term in FORBIDDEN_PUBLIC_EVIDENCE_PATTERNS:
        assert term not in combined

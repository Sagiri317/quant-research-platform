from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest

from quant_research_platform.storage.datasets import DatasetReadMode

REQUIRED_AREAS = {
    "processed_panel_parity",
    "factor_value_parity",
    "label_value_parity",
    "masks_universes_parity",
    "research_dataset_parity",
    "evaluation_parity",
    "prediction_oos_smoke_parity",
    "supporting_layers_coverage_2014_2025",
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


def test_clean_rewrite_public_safe_parity_evidence_json_schema() -> None:
    path = Path("docs/reports/clean-rewrite-0.1-parity-evidence.json")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["contract_version"] == "0.1"
    assert payload["project_version"] == "0.1.0"
    assert payload["privacy_level"] == "public_safe_aggregate"
    assert payload["artifact_id"] == "a_share_daily_bar"
    assert payload["market_id"] == "a_share"
    assert payload["verification_kind"] == "private_real_data_aggregate"
    assert payload["verification_scope"] == "aggregate_public_safe_parity"
    assert payload["source_data_visibility"] == "private_local_only"
    assert payload["row_level_output_committed"] is False
    assert len(payload["areas"]) == 9
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
    assert payload["status"] == "superseded_by_factor_catalog_refactor"
    assert payload["evidence_status"] == "evidence_pending_regeneration"
    assert payload["merge_ready"] is False


def test_full_historical_daily_panel_covers_full_2014_2025_period() -> None:
    path = Path("docs/reports/clean-rewrite-0.1-parity-evidence.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    area = next(
        item
        for item in payload["areas"]
        if item["area"] == "full_historical_daily_panel_status"
    )

    assert area["period"] == "2014-01-01 through 2025-12-31"
    assert area["human_approval_required"] is False
    assert area["accepted_explained_delta"] is False
    assert payload["merge_ready"] is False
    assert payload["evidence_status"] == "evidence_pending_regeneration"


def test_full_historical_daily_panel_summary_streams_generated_chunks(
    monkeypatch,
) -> None:
    from quant_research_platform.markets.implementations.a_share.verification import (
        private_aggregate,
    )

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
        private_aggregate,
        "_iter_generated_full_daily_panel_chunks",
        iter_chunks,
    )

    summary = private_aggregate._summarize_generated_full_daily_panel(
        Path(".")
    )

    assert summary.row_count == 2
    assert summary.key_count == 2
    assert summary.period == "2014-01-01 through 2025-12-31"


def test_full_historical_daily_panel_generation_always_uses_chunks(
    monkeypatch,
) -> None:
    from quant_research_platform.markets.implementations.a_share.verification import (
        private_aggregate,
    )

    chunk = pd.DataFrame(
        {
            "market_id": ["a_share"],
            "asset_id": ["asset-a"],
            "timestamp": [pd.Timestamp("2014-01-01")],
        }
    )

    def iter_session_chunks(project_root: Path):
        assert project_root == Path(".")
        return iter((chunk,))

    monkeypatch.setattr(
        private_aggregate,
        "_iter_generated_full_daily_panel_session_chunks",
        iter_session_chunks,
    )

    frames = list(
        private_aggregate._iter_generated_full_daily_panel_chunks(
            Path(".")
        )
    )

    assert len(frames) == 1
    assert frames[0].equals(chunk)


def test_generated_panel_generation_uses_runtime_resource_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from quant_research_platform.markets.implementations.a_share.catalog import (
        a_share_processed_catalog,
    )
    from quant_research_platform.markets.implementations.a_share.verification import (
        private_aggregate,
    )

    class Inspection:
        columns = ("asset_id", "timestamp")
        manifest = {"file_size_bytes": 1000}

    sessions = pd.date_range("2014-01-02", periods=6, freq="B")
    policy = private_aggregate.RuntimeResourcePolicy(
        available_memory_bytes=100,
        memory_budget_fraction=1.0,
        chunk_target_bytes=50,
        memory_safety_factor=1.0,
    )

    def fake_from_environment():
        return policy

    def fake_read_dataset(root: Path, **kwargs) -> pd.DataFrame:
        if root.name == "trading_calendar":
            return pd.DataFrame(
                {
                    "timestamp": sessions,
                    "is_trading_day": [True] * len(sessions),
                }
            )
        if root.name in {"stock_master", "market_cap"}:
            return pd.DataFrame({"asset_id": ["asset-a"]})
        assert root.name == "daily_market"
        time_window = kwargs["time_window"]
        timestamps = [
            session
            for session in sessions
            if pd.Timestamp(time_window.start)
            <= session
            <= pd.Timestamp(time_window.end)
        ]
        return pd.DataFrame(
            {
                "asset_id": ["asset-a"] * len(timestamps),
                "timestamp": timestamps,
            }
        )

    required_columns = a_share_processed_catalog().get("daily_panel").required_columns

    def fake_compute_daily_panel(
        daily: pd.DataFrame,
        *,
        stock_master: pd.DataFrame,
        market_cap: pd.DataFrame,
        trading_calendar: pd.DataFrame,
    ) -> pd.DataFrame:
        del stock_master, market_cap, trading_calendar
        return pd.DataFrame(
            {
                column: (
                    list(daily["asset_id"])
                    if column == "asset_id"
                    else list(daily["timestamp"])
                    if column == "timestamp"
                    else ["a_share"] * len(daily)
                    if column == "market_id"
                    else [0] * len(daily)
                )
                for column in required_columns
            }
        )

    from quant_research_platform.markets.implementations.a_share.processed import (
        panels,
    )

    monkeypatch.setattr(
        private_aggregate,
        "inspect_dataset",
        lambda _root: Inspection(),
    )
    monkeypatch.setattr(private_aggregate, "read_dataset", fake_read_dataset)
    monkeypatch.setattr(panels, "compute_daily_panel", fake_compute_daily_panel)
    monkeypatch.setattr(
        private_aggregate.RuntimeResourcePolicy,
        "from_environment",
        staticmethod(fake_from_environment),
    )

    chunks = list(
        private_aggregate._iter_generated_full_daily_panel_session_chunks(Path("."))
    )

    assert len(chunks) > 1
    assert sum(len(chunk) for chunk in chunks) == len(sessions)


def test_private_evidence_does_not_keep_legacy_chunk_policy_helpers() -> None:
    from quant_research_platform.markets.implementations.a_share.verification import (
        private_aggregate,
    )

    for name in (
        "_dynamic_" + "asset_chunk_size",
        "chunked_" + "values",
        "read_dataset_" + "asset_chunk",
        "_available_" + "memory_bytes",
        "_iter_generated_full_daily_panel_" + "asset_chunks",
        "_dataset_" + "asset_ids",
        "_read_dataset_for_" + "assets",
        "_filter_" + "asset_frame",
    ):
        assert not hasattr(private_aggregate, name)


def test_aggregate_summary_checksum_is_invariant_to_chunking() -> None:
    from quant_research_platform.markets.implementations.a_share.verification import (
        private_aggregate,
    )

    combined = pd.DataFrame(
        {
            "market_id": ["a_share"] * 12,
            "asset_id": [f"asset-{index:02d}" for index in range(12)],
            "timestamp": [pd.Timestamp("2025-12-31")] * 12,
            "value": [float(index) + 0.25 for index in range(12)],
        }
    )
    chunks = tuple(
        combined.iloc[start : start + 3].reset_index(drop=True)
        for start in range(0, len(combined), 3)
    )

    full_summary = private_aggregate._summarize_frames((combined,))
    chunk_summary = private_aggregate._summarize_frames(chunks)

    assert "numeric_sum" not in full_summary.content_summary["timestamp"]
    assert "numeric_sum" not in full_summary.content_summary["value"]
    assert chunk_summary.aggregate_checksum == full_summary.aggregate_checksum


def test_aggregate_summary_detects_duplicate_keys_across_chunks() -> None:
    from quant_research_platform.markets.implementations.a_share.verification import (
        private_aggregate,
    )

    first = pd.DataFrame(
        {
            "market_id": ["a_share"],
            "asset_id": ["asset-01"],
            "timestamp": [pd.Timestamp("2025-12-31")],
            "value": [1.0],
        }
    )
    second = pd.DataFrame(
        {
            "market_id": ["a_share"],
            "asset_id": ["asset-01"],
            "timestamp": [pd.Timestamp("2025-12-31")],
            "value": [2.0],
        }
    )

    summary = private_aggregate._summarize_frames((first, second))

    assert summary.row_count == 2
    assert summary.key_count == 1
    assert summary.duplicate_key_count == 1


def test_dataset_root_summary_uses_dataframe_chunks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from quant_research_platform.markets.implementations.a_share.verification import (
        private_aggregate,
    )

    root = tmp_path / "data" / "processed" / "a_share" / "factor" / "fixture"
    calls: list[dict[str, object]] = []

    def fake_read_dataset(dataset_root, *, time_window, mode):
        calls.append(
            {
                "root": dataset_root,
                "time_window": time_window,
                "mode": mode,
            }
        )
        return iter(
            [
                pd.DataFrame(
                    {
                        "market_id": ["a_share"],
                        "asset_id": ["asset-01"],
                        "timestamp": pd.to_datetime(["2025-01-02"]),
                        "value": [1.0],
                    }
                ),
                pd.DataFrame(
                    {
                        "market_id": ["a_share"],
                        "asset_id": ["asset-02"],
                        "timestamp": pd.to_datetime(["2025-01-03"]),
                        "value": [2.0],
                    }
                ),
            ]
        )

    monkeypatch.setattr(private_aggregate, "read_dataset", fake_read_dataset)

    summary = private_aggregate._summarize_dataset_root(root, years=(2025,))

    assert calls == [
        {
            "root": root,
            "time_window": private_aggregate.TimeWindow(
                "2025-01-01",
                "2025-12-31",
            ),
            "mode": DatasetReadMode.CHUNKED,
        }
    ]
    assert summary.row_count == 2
    assert summary.key_count == 2
    assert summary.period == "2025-01-02 through 2025-01-03"


def test_low_level_parquet_fixture_summary_reads_dataset_artifact(
    tmp_path: Path,
) -> None:
    from quant_research_platform.markets.implementations.a_share.verification import (
        private_aggregate,
    )
    from tests.storage.low_level_fixtures import write_low_level_parquet_artifact

    root = tmp_path / "data" / "processed" / "a_share" / "factor" / "fixture"
    write_low_level_parquet_artifact(
        root,
        pd.DataFrame(
            {
                "market_id": ["a_share", "a_share"],
                "asset_id": ["asset-01", "asset-02"],
                "timestamp": pd.to_datetime(["2025-01-02", "2025-01-03"]),
                "value": [1.0, 2.0],
            }
        ),
        primary_key=("market_id", "asset_id", "timestamp"),
        partition_columns=("year",),
    )

    summary = private_aggregate._summarize_dataset_root(root, years=(2025,))

    assert summary.row_count == 2
    assert summary.key_count == 2
    assert summary.period == "2025-01-02 through 2025-01-03"


def test_clean_rewrite_public_safe_parity_evidence_has_no_private_samples() -> None:
    paths = [
        Path("docs/reports/clean-rewrite-0.1-parity-evidence.md"),
        Path("docs/reports/clean-rewrite-0.1-parity-evidence.json"),
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in paths)

    for term in FORBIDDEN_PUBLIC_EVIDENCE_PATTERNS:
        assert term not in combined


def test_clean_rewrite_public_safe_parity_evidence_uses_lf_newlines() -> None:
    paths = [
        Path("docs/reports/clean-rewrite-0.1-parity-evidence.md"),
        Path("docs/reports/clean-rewrite-0.1-parity-evidence.json"),
    ]

    for path in paths:
        content = path.read_bytes()
        assert b"\r\n" not in content
        assert content.endswith(b"\n")


def test_private_verification_summary_json_schema_and_coverage() -> None:
    path = Path("docs/reports/clean-rewrite-0.1-private-verification-summary.json")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert SUMMARY_TOP_LEVEL_FIELDS.issubset(payload)
    assert payload["contract_version"] == "0.1"
    assert payload["project_version"] == "0.1.0"
    assert payload["verification_kind"] == "private_real_data_aggregate_inventory"
    assert payload["privacy_level"] == "public_safe_aggregate"
    assert payload["source_data_visibility"] == "private_local_only"
    assert payload["row_level_output_committed"] is False
    assert payload["status"] == "superseded_by_factor_catalog_refactor"
    assert payload["evidence_status"] == "evidence_pending_regeneration"
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
    path = Path("docs/reports/clean-rewrite-0.1-private-verification-summary.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    daily_panel = next(
        item
        for item in payload["dataset_inventory"]
        if item["logical_root"] == "processed/a_share/panel/daily_panel"
    )

    assert daily_panel["period"] == "2014-01-01 through 2025-12-31"
    assert (
        daily_panel["covered_by_evidence_area"]
        == "full_historical_daily_panel_status"
    )


def test_private_verification_summary_has_no_private_samples() -> None:
    paths = [
        Path("docs/reports/clean-rewrite-0.1-private-verification-summary.md"),
        Path("docs/reports/clean-rewrite-0.1-private-verification-summary.json"),
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in paths)

    for term in FORBIDDEN_PUBLIC_EVIDENCE_PATTERNS:
        assert term not in combined


def test_private_verification_summary_uses_lf_newlines() -> None:
    paths = [
        Path("docs/reports/clean-rewrite-0.1-private-verification-summary.md"),
        Path("docs/reports/clean-rewrite-0.1-private-verification-summary.json"),
    ]

    for path in paths:
        content = path.read_bytes()
        assert b"\r\n" not in content
        assert content.endswith(b"\n")

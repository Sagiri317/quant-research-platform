from __future__ import annotations

import inspect
from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.data_dependency
from tests.architecture.test_share_safe_evidence import (  # noqa: E402
    DOC_FULL_HISTORICAL_MARKET_PANEL_STATUS_AREA,
    HASH_RE,
    _synthetic_local_verification_inventory_payload,
    _synthetic_review_safe_evidence_payload,
)
from tests.helpers import frequency_ids  # noqa: E402


@pytest.fixture
def aggregates():
    pytest.importorskip("pyarrow")
    from quant_research_platform.markets.implementations.a_share.verification import (
        aggregates,
    )

    return aggregates


@pytest.fixture
def dataset_read_mode():
    pytest.importorskip("pyarrow")
    from quant_research_platform.storage.datasets import DatasetReadMode

    return DatasetReadMode


def test_current_review_safe_aggregate_payload_includes_key_value_paired_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    aggregates,
) -> None:
    summary = aggregates.AggregateSummary(
        row_count=1,
        key_count=1,
        column_count=4,
        duplicate_key_count=0,
        period="2024-01-02 through 2024-01-02",
        columns=("market_id", "asset_id", "timestamp", "value"),
        dtype_summary={"value": ("float64",)},
        null_summary={"value": 0},
        content_summary={"value": {"hash_sum": 1, "hash_xor": 1, "non_null_count": 1}},
        key_value_paired_summary={
            "value": {"hash_sum": 2, "hash_xor": 2, "non_null_count": 1}
        },
    )
    monkeypatch.setattr(
        aggregates,
        "_summarize_area_input",
        lambda *args, **kwargs: summary,
    )
    monkeypatch.setattr(
        aggregates,
        "_summarize_dataset_root",
        lambda *args, **kwargs: summary,
    )
    artifact_root = (
        tmp_path / "data" / "processed" / "a_share" / "factor" / "price_volume"
    )
    artifact_root.mkdir(parents=True)
    monkeypatch.setattr(
        aggregates,
        "_official_dataset_artifact_roots",
        lambda _: (artifact_root,),
    )

    area = aggregates._area_record(
        aggregates.AggregateInput(
            area="factor_value_parity",
            roots=(artifact_root,),
            expected_period=summary.period,
            explanation_name="factor values",
        )
    )
    inventory = aggregates._dataset_inventory(tmp_path)

    assert HASH_RE.fullmatch(area["key_value_paired_summary_hash"])
    assert HASH_RE.fullmatch(inventory[0]["key_value_paired_summary_hash"])
    assert "key_value_pairs" in inspect.getsource(
        aggregates.AggregateSummary.aggregate_checksum.fget
    )


def test_full_historical_market_panel_daily_summary_streams_generated_chunks(
    monkeypatch: pytest.MonkeyPatch,
    aggregates,
) -> None:
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

    def iter_chunks(project_root: Path, *, resource_policy=None):
        assert project_root == Path(".")
        return iter(frames)

    monkeypatch.setattr(
        aggregates,
        "_iter_generated_full_market_panel_chunks",
        iter_chunks,
    )

    summary = aggregates._summarize_generated_full_market_panel(Path("."))

    assert summary.row_count == 2
    assert summary.key_count == 2
    assert summary.period == "2014-01-01 through 2025-12-31"


def test_full_historical_market_panel_daily_generation_always_uses_chunks(
    monkeypatch: pytest.MonkeyPatch,
    aggregates,
) -> None:
    chunk = pd.DataFrame(
        {
            "market_id": ["a_share"],
            "asset_id": ["asset-a"],
            "timestamp": [pd.Timestamp("2014-01-01")],
        }
    )

    def iter_session_chunks(project_root: Path, *, resource_policy=None):
        assert project_root == Path(".")
        return iter((chunk,))

    monkeypatch.setattr(
        aggregates,
        "_iter_generated_full_market_panel_session_chunks",
        iter_session_chunks,
    )

    frames = list(aggregates._iter_generated_full_market_panel_chunks(Path(".")))

    assert len(frames) == 1
    assert frames[0].equals(chunk)


def test_generated_panel_generation_uses_runtime_resource_policy(
    monkeypatch: pytest.MonkeyPatch,
    aggregates,
) -> None:
    from quant_research_platform.markets.implementations.a_share.processed import (
        catalog,
        panels,
    )

    class Inspection:
        columns = ("asset_id", "timestamp")
        manifest = {"file_size_bytes": 1000}

    sessions = pd.date_range("2014-01-02", periods=6, freq="B")
    policy = aggregates.RuntimeResourcePolicy(
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
        assert root.name == frequency_ids.MARKET_BARS_DATA_ID
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

    required_columns = (
        catalog.processed_catalog()
        .get(frequency_ids.MARKET_PANEL_DATA_ID)
        .required_columns
    )

    def fake_compute_market_panel(
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

    monkeypatch.setattr(
        aggregates,
        "inspect_dataset",
        lambda _root: Inspection(),
    )
    monkeypatch.setattr(aggregates, "read_dataset", fake_read_dataset)
    monkeypatch.setattr(panels, "compute_market_panel", fake_compute_market_panel)
    monkeypatch.setattr(
        aggregates.RuntimeResourcePolicy,
        "from_environment",
        staticmethod(fake_from_environment),
    )

    chunks = list(
        aggregates._iter_generated_full_market_panel_session_chunks(Path("."))
    )

    assert len(chunks) > 1
    assert sum(len(chunk) for chunk in chunks) == len(sessions)


def test_verification_summary_does_not_keep_legacy_chunk_policy_helpers(
    aggregates,
) -> None:
    for name in (
        "_dynamic_" + "asset_chunk_size",
        "chunked_" + "values",
        "read_dataset_" + "asset_chunk",
        "_available_" + "memory_bytes",
        "_iter_generated_full_market_panel_" + "asset_chunks",
        "_dataset_" + "asset_ids",
        "_read_dataset_for_" + "assets",
        "_filter_" + "asset_frame",
    ):
        assert not hasattr(aggregates, name)


def test_aggregate_summary_checksum_is_invariant_to_chunking(aggregates) -> None:
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

    full_summary = aggregates._summarize_frames((combined,))
    chunk_summary = aggregates._summarize_frames(chunks)

    assert "numeric_sum" not in full_summary.content_summary["timestamp"]
    assert "numeric_sum" not in full_summary.content_summary["value"]
    assert chunk_summary.aggregate_checksum == full_summary.aggregate_checksum


def test_aggregate_summary_detects_duplicate_keys_across_chunks(
    aggregates,
) -> None:
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

    summary = aggregates._summarize_frames((first, second))

    assert summary.row_count == 2
    assert summary.key_count == 1
    assert summary.duplicate_key_count == 1


def test_dataset_root_summary_uses_dataframe_chunks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    aggregates,
    dataset_read_mode,
) -> None:
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

    monkeypatch.setattr(aggregates, "read_dataset", fake_read_dataset)

    summary = aggregates._summarize_dataset_root(root, years=(2025,))

    assert calls == [
        {
            "root": root,
            "time_window": aggregates.TimeWindow(
                "2025-01-01",
                "2025-12-31",
            ),
            "mode": dataset_read_mode.CHUNKED,
        }
    ]
    assert summary.row_count == 2
    assert summary.key_count == 2
    assert summary.period == "2025-01-02 through 2025-01-03"


def test_low_level_parquet_fixture_summary_reads_dataset_artifact(
    tmp_path: Path,
    aggregates,
) -> None:
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

    summary = aggregates._summarize_dataset_root(root, years=(2025,))

    assert summary.row_count == 2
    assert summary.key_count == 2
    assert summary.period == "2025-01-02 through 2025-01-03"


def test_verification_summary_source_fixture_is_not_public_market_contract(
    aggregates,
) -> None:
    source = Path(aggregates.__file__).read_text(encoding="utf-8")
    public_payloads = "\n".join(
        (
            str(_synthetic_review_safe_evidence_payload()),
            str(_synthetic_local_verification_inventory_payload()),
        )
    )

    assert aggregates.LOCAL_VERIFICATION_INTERIM_SOURCE == "csmar"
    assert 'PRIVATE_INTERIM_SOURCE_ID = "csmar"' not in source
    assert 'LOCAL_VERIFICATION_INTERIM_SOURCE = "csmar"' in source
    assert "not an A-share market contract" in source
    assert "csmar" not in public_payloads
    assert "CSMAR" not in public_payloads


def test_private_summary_covers_full_historical_panel_synthetic_contract() -> None:
    payload = _synthetic_local_verification_inventory_payload()
    market_panel_daily = next(
        item
        for item in payload["dataset_inventory"]
        if item["logical_root"].endswith("market_panel_data_id(DataFrequency.DAILY)>")
    )

    assert market_panel_daily["period"] == "2014-01-01 through 2025-12-31"
    assert (
        market_panel_daily["covered_by_evidence_area"]
        == DOC_FULL_HISTORICAL_MARKET_PANEL_STATUS_AREA
    )

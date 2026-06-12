from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from quant_research_platform.catalog.processed import (
    InterimDataCatalog,
    ProcessedDataCatalog,
)
from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.data import (
    DataFrequency,
    DataGrain,
    InterimDataSpec,
    ProcessedDataKind,
)
from quant_research_platform.contracts.ids import SourceId
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.implementations.a_share.processed import (
    materializer,
)
from quant_research_platform.markets.implementations.a_share.processed.catalog import (
    a_share_processed_catalog,
)
from quant_research_platform.runtime.resources import RuntimeResourcePolicy
from quant_research_platform.storage.datasets import (
    inspect_dataset,
    read_dataset,
    write_dataset,
)
from quant_research_platform.storage.layout import StorageLayout

SYNTHETIC_SOURCE_ID = SourceId("synthetic_vendor_a")


def test_materialize_processed_data_writes_requested_panel_and_dependencies(
    tmp_path,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    _write_interim_inputs(layout)

    written = materializer.materialize_processed_data(
        layout=layout,
        source_id=SYNTHETIC_SOURCE_ID,
        time_window=TimeWindow("2014-01-02", "2014-01-03"),
        data_ids=("daily_panel",),
    )

    assert set(written) == {"daily_panel"}
    root = tmp_path / "data" / "processed" / "a_share" / "panel" / "daily_panel"
    panel = read_dataset(root)
    assert len(panel) == 2
    assert panel["asset_id"].unique().tolist() == ["000001"]
    assert "source_asset_id" not in panel.columns


def test_materialize_processed_data_rejects_existing_requested_root(tmp_path) -> None:
    layout = StorageLayout(project_root=tmp_path)
    _write_interim_inputs(layout)
    kwargs = {
        "layout": layout,
        "source_id": SYNTHETIC_SOURCE_ID,
        "time_window": TimeWindow("2014-01-02", "2014-01-03"),
        "data_ids": ("daily_panel",),
    }

    materializer.materialize_processed_data(**kwargs)

    with pytest.raises(ValueError, match="dataset_exists"):
        materializer.materialize_processed_data(**kwargs)


def test_materialize_processed_data_requires_exactly_one_selector(tmp_path) -> None:
    common = {
        "layout": StorageLayout(project_root=tmp_path),
        "source_id": SYNTHETIC_SOURCE_ID,
        "time_window": TimeWindow("2014-01-02", "2014-01-03"),
    }

    with pytest.raises(ValueError, match="data_ids_xor_kinds_required"):
        materializer.materialize_processed_data(**common)
    with pytest.raises(ValueError, match="data_ids_xor_kinds_required"):
        materializer.materialize_processed_data(
            **common,
            data_ids=("daily_panel",),
            kinds=frozenset({ProcessedDataKind.PANEL}),
        )


def test_materialize_processed_data_by_kind_writes_dependency_and_target(
    tmp_path,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    _write_interim_inputs(layout)

    written = materializer.materialize_processed_data(
        layout=layout,
        source_id=SYNTHETIC_SOURCE_ID,
        time_window=TimeWindow("2014-01-02", "2014-01-03"),
        kinds=frozenset({ProcessedDataKind.FACTOR}),
    )

    assert set(written) == {"daily_panel", "price_volume"}
    assert (
        tmp_path / "data" / "processed" / "a_share" / "panel" / "daily_panel"
    ).exists()
    assert (
        tmp_path / "data" / "processed" / "a_share" / "factor" / "price_volume"
    ).exists()


def test_factor_materializer_rejects_unexpected_declared_processed_input(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    _write_interim_inputs(layout)
    catalog = _catalog_with_processed_inputs(
        "price_volume",
        ("daily_panel", "daily_context"),
    )
    monkeypatch.setattr(materializer, "a_share_processed_catalog", lambda: catalog)

    with pytest.raises(
        ValueError,
        match="unexpected_processed_input_for_factor: price_volume: daily_context",
    ):
        materializer.materialize_processed_data(
            layout=layout,
            source_id=SYNTHETIC_SOURCE_ID,
            time_window=TimeWindow("2014-01-02", "2014-01-03"),
            data_ids=("price_volume",),
        )


def test_factor_materializer_requires_family_contract_input_declared_by_spec(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    _write_interim_inputs(layout)
    catalog = _catalog_with_processed_inputs("price_volume", ())
    monkeypatch.setattr(materializer, "a_share_processed_catalog", lambda: catalog)

    with pytest.raises(
        ValueError,
        match="missing_processed_input_for_factor: price_volume: daily_panel",
    ):
        materializer.materialize_processed_data(
            layout=layout,
            source_id=SYNTHETIC_SOURCE_ID,
            time_window=TimeWindow("2014-01-02", "2014-01-03"),
            data_ids=("price_volume",),
        )


def test_materializer_reuses_same_run_dependency_for_empty_window_tail(
    tmp_path,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    _write_interim_inputs(layout)

    written = materializer.materialize_processed_data(
        layout=layout,
        source_id=SYNTHETIC_SOURCE_ID,
        time_window=TimeWindow("2014-01-02", "2014-01-04"),
        kinds=frozenset({ProcessedDataKind.ANNOTATION}),
    )

    assert {
        "daily_panel",
        "daily_context",
        "a_share_default",
        "a_share_as_of",
        "a_share_execution",
        "a_share_evaluation",
    }.issubset(written)


def test_materialize_processed_data_existing_dependency_accepts_non_trading_end(
    tmp_path,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    _write_interim_inputs(layout)
    materializer.materialize_processed_data(
        layout=layout,
        source_id=SYNTHETIC_SOURCE_ID,
        time_window=TimeWindow("2014-01-02", "2014-01-03"),
        data_ids=("daily_panel",),
    )

    written = materializer.materialize_processed_data(
        layout=layout,
        source_id=SYNTHETIC_SOURCE_ID,
        time_window=TimeWindow("2014-01-02", "2014-01-04"),
        data_ids=("a_share_execution",),
    )

    assert "a_share_execution" in written


def test_materialize_processed_data_writes_only_core_trading_sessions(
    tmp_path,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    _write_interim_inputs(
        layout,
        start_date="2017-04-28",
        end_date="2017-05-04",
        closed_dates=("2017-05-01",),
    )

    materializer.materialize_processed_data(
        layout=layout,
        source_id=SYNTHETIC_SOURCE_ID,
        time_window=TimeWindow("2017-04-28", "2017-05-03"),
        data_ids=("price_volume",),
    )

    panel = read_dataset(_processed_root(tmp_path, "panel", "daily_panel"))
    factors = read_dataset(_processed_root(tmp_path, "factor", "price_volume"))

    assert panel["timestamp"].dt.date.astype(str).tolist() == [
        "2017-04-28",
        "2017-05-02",
        "2017-05-03",
    ]
    assert factors["timestamp"].dt.date.astype(str).tolist() == [
        "2017-04-28",
        "2017-05-02",
        "2017-05-03",
    ]


def test_materialize_processed_data_existing_dependency_must_cover_trading_sessions(
    tmp_path,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    _write_interim_inputs(layout, end_date="2014-01-06")
    materializer.materialize_processed_data(
        layout=layout,
        source_id=SYNTHETIC_SOURCE_ID,
        time_window=TimeWindow("2014-01-02", "2014-01-03"),
        data_ids=("daily_panel",),
    )

    with pytest.raises(ValueError, match="insufficient_dataset_coverage"):
        materializer.materialize_processed_data(
            layout=layout,
            source_id=SYNTHETIC_SOURCE_ID,
            time_window=TimeWindow("2014-01-02", "2014-01-06"),
            data_ids=("a_share_execution",),
        )


def test_existing_dependency_coverage_rejects_missing_middle_sessions(
    tmp_path,
) -> None:
    source_layout = StorageLayout(project_root=tmp_path / "source")
    _write_interim_inputs(
        source_layout,
        start_date="2014-01-02",
        end_date="2016-01-04",
    )
    materializer.materialize_processed_data(
        layout=source_layout,
        source_id=SYNTHETIC_SOURCE_ID,
        time_window=TimeWindow("2014-01-02", "2016-01-04"),
        data_ids=("daily_panel",),
    )
    source_panel_root = _processed_root(tmp_path / "source", "panel", "daily_panel")
    panel = read_dataset(source_panel_root)
    panel_without_2015 = panel.loc[
        pd.to_datetime(panel["timestamp"]).dt.year.ne(2015)
    ]

    layout = StorageLayout(project_root=tmp_path / "coverage")
    _write_interim_inputs(
        layout,
        start_date="2014-01-02",
        end_date="2016-01-04",
    )
    panel_root = _processed_root(tmp_path / "coverage", "panel", "daily_panel")
    write_dataset(
        panel_root,
        panel_without_2015,
        spec=a_share_processed_catalog().get("daily_panel"),
    )

    with pytest.raises(
        ValueError,
        match=r"insufficient_dataset_coverage: missing_sessions=",
    ):
        materializer.materialize_processed_data(
            layout=layout,
            source_id=SYNTHETIC_SOURCE_ID,
            time_window=TimeWindow("2014-01-02", "2016-01-04"),
            data_ids=("a_share_execution",),
        )


def test_existing_dependency_coverage_accepts_non_trading_end_date(
    tmp_path,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    _write_interim_inputs(
        layout,
        start_date="2016-12-29",
        end_date="2017-01-03",
    )
    materializer.materialize_processed_data(
        layout=layout,
        source_id=SYNTHETIC_SOURCE_ID,
        time_window=TimeWindow("2016-12-29", "2016-12-30"),
        data_ids=("daily_panel",),
    )

    materializer.materialize_processed_data(
        layout=layout,
        source_id=SYNTHETIC_SOURCE_ID,
        time_window=TimeWindow("2016-12-29", "2016-12-31"),
        data_ids=("a_share_execution",),
    )

    annotation_root = _processed_root(tmp_path, "annotation", "a_share_execution")
    annotation = read_dataset(annotation_root)
    assert annotation["timestamp"].dt.date.astype(str).tolist() == [
        "2016-12-29",
        "2016-12-30",
    ]


def test_annotation_materializer_uses_dependency_lookahead_session(tmp_path) -> None:
    layout = StorageLayout(project_root=tmp_path)
    _write_interim_inputs(
        layout,
        start_date="2014-01-30",
        end_date="2014-02-07",
        closed_dates=(
            "2014-01-31",
            "2014-02-03",
            "2014-02-04",
            "2014-02-05",
            "2014-02-06",
        ),
    )
    materializer.materialize_processed_data(
        layout=layout,
        source_id=SYNTHETIC_SOURCE_ID,
        time_window=TimeWindow("2014-01-30", "2014-02-07"),
        data_ids=("daily_panel",),
    )

    materializer.materialize_processed_data(
        layout=layout,
        source_id=SYNTHETIC_SOURCE_ID,
        time_window=TimeWindow("2014-01-30", "2014-01-30"),
        data_ids=("a_share_execution",),
    )

    execution = read_dataset(
        _processed_root(tmp_path, "annotation", "a_share_execution")
    )
    assert execution["timestamp"].dt.date.astype(str).tolist() == ["2014-01-30"]
    assert execution["next_day_is_trading"].tolist() == [True]
    assert execution["can_enter"].tolist() == [True]
    assert execution["can_exit"].tolist() == [True]


def test_materialize_processed_data_forced_tiny_chunks_equal_default_output(
    tmp_path,
) -> None:
    _assert_forced_tiny_chunks_match_default(
        tmp_path,
        data_id="daily_panel",
        kind="panel",
        end_date="2014-01-08",
        time_window=TimeWindow("2014-01-02", "2014-01-07"),
    )


@pytest.mark.parametrize(
    ("data_id", "kind"),
    [
        ("a_share_execution", "annotation"),
        ("price_volume", "factor"),
        ("forward_returns", "label"),
    ],
)
def test_materialize_processed_data_downstream_forced_tiny_chunks_equal_default_output(
    tmp_path,
    data_id: str,
    kind: str,
) -> None:
    _assert_forced_tiny_chunks_match_default(
        tmp_path,
        data_id=data_id,
        kind=kind,
        end_date="2014-08-29",
        time_window=TimeWindow("2014-01-02", "2014-02-28"),
    )


def _assert_forced_tiny_chunks_match_default(
    tmp_path,
    *,
    data_id: str,
    kind: str,
    end_date: str,
    time_window: TimeWindow,
) -> None:
    default_layout = StorageLayout(project_root=tmp_path / "default")
    tiny_layout = StorageLayout(project_root=tmp_path / "tiny")
    _write_interim_inputs(default_layout, end_date=end_date)
    _write_interim_inputs(tiny_layout, end_date=end_date)

    materializer.materialize_processed_data(
        layout=default_layout,
        source_id=SYNTHETIC_SOURCE_ID,
        time_window=time_window,
        data_ids=(data_id,),
    )
    materializer.materialize_processed_data(
        layout=tiny_layout,
        source_id=SYNTHETIC_SOURCE_ID,
        time_window=time_window,
        data_ids=(data_id,),
        resource_policy=RuntimeResourcePolicy(
            available_memory_bytes=1_000,
            memory_budget_fraction=1.0,
            chunk_target_bytes=250,
            memory_safety_factor=1.0,
        ),
    )

    default_root = _processed_root(tmp_path / "default", kind, data_id)
    tiny_root = _processed_root(tmp_path / "tiny", kind, data_id)
    default = read_dataset(default_root).sort_values(
        ["market_id", "asset_id", "timestamp"]
    )
    tiny = read_dataset(tiny_root).sort_values(["market_id", "asset_id", "timestamp"])

    pd.testing.assert_frame_equal(
        default.reset_index(drop=True),
        tiny.reset_index(drop=True),
    )
    assert (
        inspect_dataset(tiny_root).file_count
        > inspect_dataset(default_root).file_count
    )


def _processed_root(project_root: Path, kind: str, data_id: str) -> Path:
    return project_root / "data" / "processed" / "a_share" / kind / data_id


def _catalog_with_processed_inputs(
    target_data_id: str,
    processed_input_data_ids: tuple[str, ...],
) -> ProcessedDataCatalog:
    catalog = ProcessedDataCatalog()
    for spec in a_share_processed_catalog().list():
        if spec.data_id == target_data_id:
            spec = replace(
                spec,
                processed_input_data_ids=processed_input_data_ids,
            )
        catalog.register(spec)
    return catalog


def _interim_catalog() -> InterimDataCatalog:
    specs = {
        "daily_market": (
            "source_asset_id",
            "timestamp",
            "trade_date",
            "market_type",
            "trading_status_code",
            "filling_flag",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
            "prev_close",
            "limit_status",
            "return_1d",
            "adj_close",
            "adj_open",
            "float_market_cap",
            "total_market_cap",
            "turnover",
        ),
        "stock_master": (
            "source_asset_id",
            "timestamp",
            "stock_name",
            "listing_date",
            "delisting_date",
            "company_status",
            "csrc_2012_industry_code",
            "csrc_2012_industry_name",
            "status_change_date",
        ),
        "market_cap": (
            "source_asset_id",
            "timestamp",
            "share_change_date",
            "total_shares",
            "a_float_shares",
        ),
        "trading_calendar": (
            "timestamp",
            "calendar_date",
            "source_market_type",
            "is_trading_day",
            "source_day_of_week",
            "source_state",
        ),
    }
    catalog = InterimDataCatalog()
    for data_id, columns in specs.items():
        primary_key = (
            ("timestamp",)
            if data_id == "trading_calendar"
            else ("source_asset_id", "timestamp")
        )
        catalog.register(
            InterimDataSpec(
                data_id=data_id,
                source_id=SYNTHETIC_SOURCE_ID,
                frequency=DataFrequency.DAILY,
                grain=DataGrain.ASSET_TIMESTAMP,
                column_roles={
                    column: (
                        ColumnRole.TIMESTAMP
                        if column in {"timestamp", "calendar_date"}
                        else ColumnRole.KEY
                        if column in primary_key
                        else ColumnRole.CONTEXT
                    )
                    for column in columns
                },
                primary_key=primary_key,
                partition_columns=("year",),
            )
        )
    return catalog


def _write_interim_inputs(
    layout: StorageLayout,
    *,
    start_date: str = "2014-01-02",
    end_date: str = "2014-01-03",
    closed_dates: tuple[str, ...] = (),
) -> None:
    catalog = _interim_catalog()
    source_id = SYNTHETIC_SOURCE_ID
    frames = {
        "daily_market": _daily_market_frame(
            start_date=start_date,
            end_date=end_date,
        ),
        "stock_master": _stock_master_frame(),
        "market_cap": _market_cap_frame(),
        "trading_calendar": _trading_calendar_frame(
            end_date=end_date,
            closed_dates=closed_dates,
        ),
    }
    for data_id, frame in frames.items():
        spec = catalog.get(data_id)
        write_dataset(
            layout.interim_data_root(source_id, data_id),
            frame.loc[:, list(spec.required_columns)],
            spec=spec,
        )


def _daily_market_frame(
    *,
    start_date: str = "2014-01-02",
    end_date: str = "2014-01-03",
) -> pd.DataFrame:
    rows = []
    for offset, timestamp in enumerate(
        pd.bdate_range(start_date, end_date),
        start=1,
    ):
        rows.append(
            {
                "source_asset_id": "000001",
                "timestamp": timestamp,
                "trade_date": timestamp,
                "market_type": 4,
                "trading_status_code": 1,
                "filling_flag": "0",
                "open": 10.0 + offset,
                "high": 10.5 + offset,
                "low": 9.8 + offset,
                "close": 10.2 + offset,
                "volume": 1000.0 + offset,
                "amount": 10200.0 + offset,
                "prev_close": 9.9 + offset,
                "limit_status": 0,
                "return_1d": 0.01,
                "adj_close": 10.2 + offset,
                "adj_open": 10.0 + offset,
                "float_market_cap": 100000.0 + offset,
                "total_market_cap": 200000.0 + offset,
                "turnover": 0.01,
            }
        )
    return pd.DataFrame(rows)


def _stock_master_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_asset_id": "000001",
                "timestamp": pd.Timestamp("2010-01-01"),
                "stock_name": "demo",
                "listing_date": pd.Timestamp("2010-01-01"),
                "delisting_date": pd.NaT,
                "company_status": "normal",
                "csrc_2012_industry_code": "I",
                "csrc_2012_industry_name": "industry",
                "status_change_date": pd.Timestamp("2010-01-01"),
            }
        ]
    )


def _market_cap_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_asset_id": "000001",
                "timestamp": pd.Timestamp("2010-01-01"),
                "share_change_date": pd.Timestamp("2010-01-01"),
                "total_shares": 1000.0,
                "a_float_shares": 500.0,
            }
        ]
    )


def _trading_calendar_frame(
    *,
    end_date: str = "2014-01-03",
    closed_dates: tuple[str, ...] = (),
) -> pd.DataFrame:
    closed = {pd.Timestamp(date).date() for date in closed_dates}
    return pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp(date),
                "calendar_date": pd.Timestamp(date),
                "source_market_type": 4,
                "is_trading_day": pd.Timestamp(date).weekday() < 5
                and pd.Timestamp(date).date() not in closed,
                "source_day_of_week": pd.Timestamp(date).dayofweek + 1,
                "source_state": (
                    "O"
                    if pd.Timestamp(date).weekday() < 5
                    and pd.Timestamp(date).date() not in closed
                    else "C"
                ),
            }
            for date in pd.date_range("2010-01-01", end_date, freq="D")
        ]
    )

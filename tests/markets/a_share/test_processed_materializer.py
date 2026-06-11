from __future__ import annotations

import pandas as pd
import pytest

from quant_research_platform.contracts.data import ProcessedDataKind
from quant_research_platform.contracts.ids import SourceId
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.implementations.a_share.processed import (
    materializer,
)
from quant_research_platform.sources.csmar.interim import interim_catalog
from quant_research_platform.storage.datasets import read_dataset, write_dataset
from quant_research_platform.storage.layout import StorageLayout


def test_materialize_processed_data_writes_requested_panel_and_dependencies(
    tmp_path,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    _write_interim_inputs(layout)

    written = materializer.materialize_processed_data(
        layout=layout,
        source_id=SourceId("csmar"),
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
        "source_id": SourceId("csmar"),
        "time_window": TimeWindow("2014-01-02", "2014-01-03"),
        "data_ids": ("daily_panel",),
    }

    materializer.materialize_processed_data(**kwargs)

    with pytest.raises(ValueError, match="dataset_exists"):
        materializer.materialize_processed_data(**kwargs)


def test_materialize_processed_data_requires_exactly_one_selector(tmp_path) -> None:
    common = {
        "layout": StorageLayout(project_root=tmp_path),
        "source_id": SourceId("csmar"),
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
        source_id=SourceId("csmar"),
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


def test_materializer_reuses_same_run_dependency_for_empty_window_tail(
    tmp_path,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    _write_interim_inputs(layout)

    written = materializer.materialize_processed_data(
        layout=layout,
        source_id=SourceId("csmar"),
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


def test_materialize_processed_data_existing_dependency_must_cover_window(
    tmp_path,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    _write_interim_inputs(layout)
    materializer.materialize_processed_data(
        layout=layout,
        source_id=SourceId("csmar"),
        time_window=TimeWindow("2014-01-02", "2014-01-03"),
        data_ids=("daily_panel",),
    )

    with pytest.raises(ValueError, match="insufficient_dataset_coverage"):
        materializer.materialize_processed_data(
            layout=layout,
            source_id=SourceId("csmar"),
            time_window=TimeWindow("2014-01-02", "2014-01-04"),
            data_ids=("a_share_execution",),
        )


def _write_interim_inputs(layout: StorageLayout) -> None:
    catalog = interim_catalog()
    source_id = SourceId("csmar")
    frames = {
        "daily_market": _daily_market_frame(),
        "stock_master": _stock_master_frame(),
        "market_cap": _market_cap_frame(),
        "trading_calendar": _trading_calendar_frame(),
    }
    for data_id, frame in frames.items():
        spec = catalog.get(data_id)
        write_dataset(
            layout.interim_data_root(source_id, data_id),
            frame.loc[:, list(spec.required_columns)],
            spec=spec,
        )


def _daily_market_frame() -> pd.DataFrame:
    rows = []
    for offset, value in enumerate(["2014-01-02", "2014-01-03"], start=1):
        timestamp = pd.Timestamp(value)
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


def _trading_calendar_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp(date),
                "calendar_date": pd.Timestamp(date),
                "source_market_type": 4,
                "is_trading_day": True,
                "source_day_of_week": pd.Timestamp(date).dayofweek + 1,
                "source_state": "O",
            }
            for date in pd.date_range("2010-01-01", "2014-01-03", freq="D")
        ]
    )

from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_research_platform.catalog.defaults import (
    default_processed_catalog,
    default_research_dataset_catalog,
)
from quant_research_platform.contracts.data import ProcessedDataKind
from quant_research_platform.contracts.ids import MarketId, SourceId
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.implementations.a_share.processed import (
    materializer,
)
from quant_research_platform.research.datasets import (
    assemble_research_dataset,
    read_research_dataset,
)
from quant_research_platform.sources.csmar.interim import interim_catalog
from quant_research_platform.storage.datasets import read_dataset, write_dataset
from quant_research_platform.storage.layout import StorageLayout


def test_synthetic_raw_to_research_dataset_correctness_path(tmp_path: Path) -> None:
    layout = StorageLayout(project_root=tmp_path)
    _write_synthetic_interim_world(layout)
    time_window = TimeWindow("2020-01-04", "2020-03-15")

    materializer.materialize_processed_data(
        layout=layout,
        source_id=SourceId("csmar"),
        time_window=time_window,
        kinds=frozenset(
            {
                ProcessedDataKind.ANNOTATION,
                ProcessedDataKind.FACTOR,
                ProcessedDataKind.LABEL,
            }
        ),
    )
    assemble_research_dataset(
        layout=layout,
        dataset_catalog=default_research_dataset_catalog(),
        processed_catalog=default_processed_catalog(),
        dataset_id="a_share_price_volume_forward_returns",
        time_window=time_window,
    )

    panel = read_dataset(
        tmp_path / "data/processed/a_share/panel/daily_panel",
        time_window=time_window,
    )
    execution = read_dataset(
        tmp_path / "data/processed/a_share/annotation/a_share_execution",
        time_window=time_window,
    )
    factors = read_dataset(
        tmp_path / "data/processed/a_share/factor/price_volume",
        time_window=time_window,
    )
    labels = read_dataset(
        tmp_path / "data/processed/a_share/label/forward_returns",
        time_window=time_window,
    )
    research = read_research_dataset(
        layout,
        market_id=MarketId("a_share"),
        dataset_id="a_share_price_volume_forward_returns",
        time_window=time_window,
    )

    assert set(panel["asset_id"].unique()) == {"000001", "000002"}
    assert set(panel["market_type"].unique()) == {4, 32}
    assert "000003" not in set(panel["asset_id"])
    assert panel["timestamp"].min() == pd.Timestamp("2020-01-06")
    assert panel["timestamp"].max() == pd.Timestamp("2020-03-13")
    assert panel.duplicated(["market_id", "asset_id", "timestamp"]).sum() == 0
    assert not (
        panel.select_dtypes("number").isin([float("inf"), float("-inf")]).any().any()
    )

    st_row = panel.loc[
        panel["asset_id"].eq("000002")
        & panel["timestamp"].eq(pd.Timestamp("2020-01-15"))
    ].iloc[0]
    assert bool(st_row["is_st"]) is True
    assert bool(st_row["eligible"]) is False

    next_limit_up = panel.loc[
        panel["asset_id"].eq("000001")
        & panel["timestamp"].eq(pd.Timestamp("2020-01-09"))
    ].iloc[0]
    next_limit_down = panel.loc[
        panel["asset_id"].eq("000001")
        & panel["timestamp"].eq(pd.Timestamp("2020-01-10"))
    ].iloc[0]
    assert bool(next_limit_up["can_enter"]) is False
    assert bool(next_limit_down["can_exit"]) is False

    changed_shares = panel.loc[
        panel["asset_id"].eq("000001")
        & panel["timestamp"].ge(pd.Timestamp("2020-02-03"))
    ]
    assert changed_shares["a_float_shares"].min() == 800.0
    assert execution.duplicated(["market_id", "asset_id", "timestamp"]).sum() == 0
    assert factors.duplicated(["market_id", "asset_id", "timestamp"]).sum() == 0
    assert labels.duplicated(["market_id", "asset_id", "timestamp"]).sum() == 0
    assert research.duplicated(["market_id", "asset_id", "timestamp"]).sum() == 0
    assert len(research) == len(panel)
    assert research["forward_return_1d"].notna().any()
    assert research["vol_20d"].notna().any()


def _write_synthetic_interim_world(layout: StorageLayout) -> None:
    catalog = interim_catalog()
    source_id = SourceId("csmar")
    frames = {
        "daily_market": _daily_market(),
        "stock_master": _stock_master(),
        "market_cap": _market_cap(),
        "trading_calendar": _trading_calendar(),
    }
    for data_id, frame in frames.items():
        spec = catalog.get(data_id)
        write_dataset(
            layout.interim_data_root(source_id, data_id),
            frame.loc[:, list(spec.required_columns)],
            spec=spec,
        )


def _daily_market() -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-02", periods=80)
    rows: list[dict[str, object]] = []
    for asset_id, market_type in (("000001", 4), ("000002", 32), ("000003", 64)):
        for index, timestamp in enumerate(dates, start=1):
            close = 10.0 + index * 0.05 + int(asset_id[-1])
            limit_status = 0
            open_price = close * 0.99
            high = close * 1.01
            low = close * 0.98
            trading_status_code = 1
            if asset_id == "000001" and timestamp == pd.Timestamp("2020-01-10"):
                limit_status = 1
                open_price = high = low = close
            if asset_id == "000001" and timestamp == pd.Timestamp("2020-01-13"):
                limit_status = -1
                open_price = high = low = close
            if asset_id == "000002" and timestamp == pd.Timestamp("2020-01-15"):
                trading_status_code = 2
            rows.append(
                {
                    "source_asset_id": asset_id,
                    "timestamp": timestamp,
                    "trade_date": timestamp,
                    "market_type": market_type,
                    "trading_status_code": trading_status_code,
                    "filling_flag": "0",
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": 10_000.0 + index,
                    "amount": close * (10_000.0 + index),
                    "prev_close": close - 0.05,
                    "limit_status": limit_status,
                    "return_1d": 0.004 + (index % 7) * 0.0003,
                    "adj_close": close,
                    "adj_open": open_price * close / close if close else pd.NA,
                    "float_market_cap": close * 500.0,
                    "total_market_cap": close * 1_000.0,
                    "turnover": 0.02,
                }
            )
    return pd.DataFrame(rows)


def _stock_master() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_asset_id": asset_id,
                "timestamp": pd.Timestamp("2010-01-01"),
                "stock_name": f"demo_{asset_id}",
                "listing_date": pd.Timestamp("2010-01-01"),
                "delisting_date": pd.NaT,
                "company_status": "normal",
                "csrc_2012_industry_code": "I",
                "csrc_2012_industry_name": "industry",
                "status_change_date": pd.Timestamp("2010-01-01"),
            }
            for asset_id in ("000001", "000002", "000003")
        ]
    )


def _market_cap() -> pd.DataFrame:
    rows = []
    for asset_id in ("000001", "000002", "000003"):
        rows.append(
            {
                "source_asset_id": asset_id,
                "timestamp": pd.Timestamp("2010-01-01"),
                "share_change_date": pd.Timestamp("2010-01-01"),
                "total_shares": 1000.0,
                "a_float_shares": 500.0,
            }
        )
    rows.append(
        {
            "source_asset_id": "000001",
            "timestamp": pd.Timestamp("2020-02-03"),
            "share_change_date": pd.Timestamp("2020-02-03"),
            "total_shares": 1200.0,
            "a_float_shares": 800.0,
        }
    )
    return pd.DataFrame(rows)


def _trading_calendar() -> pd.DataFrame:
    rows = []
    for date in pd.date_range("2019-01-01", "2020-05-15", freq="D"):
        open_day = date.weekday() < 5
        for market_type in (4, 32, 64):
            rows.append(
                {
                    "timestamp": date,
                    "calendar_date": date,
                    "source_market_type": market_type,
                    "is_trading_day": open_day,
                    "source_day_of_week": date.dayofweek + 1,
                    "source_state": "O" if open_day else "C",
                }
            )
    return pd.DataFrame(rows)

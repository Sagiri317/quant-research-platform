"""Synthetic source facts shared by native A-share producer tests."""

import numpy as np
import pandas as pd

from quant_research_platform.markets.implementations.a_share.fundamentals import (
    FINANCIAL_VALUE_COLUMNS,
)
from quant_research_platform.markets.implementations.a_share.processed.catalog import (
    processed_catalog,
)
from quant_research_platform.markets.prepare import prepare
from quant_research_platform.sources.choice.interim import (
    interim_catalog as choice_catalog,
)
from quant_research_platform.sources.csmar.interim import (
    interim_catalog as csmar_catalog,
)
from quant_research_platform.sources.csmar.interim import normalize_raw_frame
from quant_research_platform.sources.reference.interim import (
    interim_catalog as reference_catalog,
)
from quant_research_platform.storage.datasets import write_dataset


def _synthetic_financial_frames() -> dict[str, pd.DataFrame]:
    assets = ("000001", "000002", "000003")
    periods = pd.to_datetime(
        [
            "2019-03-31",
            "2019-06-30",
            "2019-09-30",
            "2019-12-31",
            "2020-03-31",
            "2020-06-30",
            "2020-09-30",
            "2020-12-31",
        ]
    )
    frames: dict[str, pd.DataFrame] = {}
    for data_id, columns in FINANCIAL_VALUE_COLUMNS.items():
        rows = [(asset, period) for asset in assets for period in periods]
        frames[data_id] = pd.DataFrame(
            {
                "source_asset_id": [asset for asset, _period in rows],
                "timestamp": [period for _asset, period in rows],
                "report_period": [period for _asset, period in rows],
                "report_type": "A",
                "is_corrected": [False] * len(rows),
                "correction_disclosure_date": pd.Series(
                    pd.NaT, index=range(len(rows)), dtype="datetime64[ns]"
                ),
                "version_identity": "original",
                **{
                    column: [float(index + 1) for index in range(len(rows))]
                    for column in columns
                },
            }
        )
    frames["financial_first_disclosure"] = pd.DataFrame(
        {
            "asset_id": [asset for asset in assets for _period in periods],
            "report_period": [period for _asset in assets for period in periods],
            "first_disclosure_date": [
                period + pd.Timedelta(days=20)
                for _asset in assets
                for period in periods
            ],
        }
    )
    return frames


def _synthetic_interim_frames(sessions: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    assets = ("000001", "000002", "000003")
    rows = [
        (asset, timestamp, asset_index, date_index)
        for date_index, timestamp in enumerate(sessions)
        for asset_index, asset in enumerate(assets, start=1)
    ]
    price = np.array(
        [10.0 + date_index + asset_index for _, _, asset_index, date_index in rows]
    )
    market_bars = pd.DataFrame(
        {
            "source_asset_id": [row[0] for row in rows],
            "timestamp": [row[1] for row in rows],
            "trade_date": [row[1] for row in rows],
            "market_type": 1,
            "trading_status_code": 0,
            "filling_flag": 0,
            "open": price,
            "high": price + 0.5,
            "low": price - 0.5,
            "close": price,
            "volume": 100.0,
            "amount": price * 100.0,
            "after_hours_volume": 0.0,
            "after_hours_amount": 0.0,
            "prev_close": price - 0.1,
            "limit_status": 0.0,
            "period_return": 0.01,
            "adj_close": price,
            "adj_open": price + 0.1,
            "float_market_cap": 1000000.0,
            "total_market_cap": 1200000.0,
            "turnover": np.nan,
        }
    )
    stock_master = pd.DataFrame(
        {
            "source_asset_id": assets,
            "timestamp": sessions[0],
            "stock_name": assets,
            "listing_date": sessions[0] - pd.Timedelta(days=400),
            "delisting_date": pd.NaT,
            "company_status": "normal",
            "industry_level1_code": "A01",
            "industry_level1_name": "industry",
            "status_change_date": sessions[0] - pd.Timedelta(days=400),
        }
    )
    market_cap = pd.DataFrame(
        {
            "source_asset_id": assets,
            "timestamp": sessions[0] - pd.Timedelta(days=400),
            "share_change_date": sessions[0] - pd.Timedelta(days=400),
            "total_shares": 120000.0,
            "a_float_shares": 100000.0,
        }
    )
    calendar_dates = pd.date_range(sessions[0], sessions[-1]).as_unit("ns")
    opened = calendar_dates.isin(sessions)
    calendar = pd.DataFrame(
        {
            "timestamp": calendar_dates,
            "calendar_date": calendar_dates,
            "source_market_type": 1,
            "is_trading_day": opened,
            "source_day_of_week": calendar_dates.dayofweek,
            "source_state": np.where(opened, "O", "C"),
        }
    )
    industry = pd.DataFrame(
        {
            "source_asset_id": assets,
            "timestamp": sessions[0] - pd.Timedelta(days=1),
            "industry_standard_id": "CSRC2012",
            "implement_date": sessions[0] - pd.Timedelta(days=1),
            "industry_level1_code": "A01",
            "industry_level1_name": "industry",
        }
    )
    industry_sets = pd.DataFrame(
        {
            "industry_code": "A01",
            "industry_standard_id": "CSRC2012",
            "industry_name": "industry",
            "parent_industry_code": "A",
            "industry_id": "1",
            "industry_level": "1",
        },
        index=[0],
    )
    industry_versions = pd.DataFrame(
        {
            "industry_standard_id": "CSRC2012",
            "industry_standard_name": "CSRC",
            "timestamp": sessions[0] - pd.Timedelta(days=1),
            "implement_date": sessions[0] - pd.Timedelta(days=1),
            "abandon_date": pd.NaT,
            "institution_name": "CSRC",
        },
        index=[0],
    )
    listing_changes = pd.DataFrame(
        {
            "source_asset_id": assets,
            "timestamp": sessions[0] - pd.Timedelta(days=1),
            "announcement_date": sessions[0] - pd.Timedelta(days=2),
            "effective_date": sessions[0] - pd.Timedelta(days=1),
            "changed_item": "status",
            "value_before": "",
            "value_after": "normal",
        }
    )
    annual = pd.DataFrame(
        {
            "source_asset_id": assets,
            "timestamp": sessions[0] - pd.Timedelta(days=1),
            "effective_date": sessions[0] - pd.Timedelta(days=1),
            "industry_level1_code": "A01",
            "industry_level1_name": "industry",
            "listing_date": sessions[0] - pd.Timedelta(days=400),
            "listing_state": "normal",
        }
    )
    stock_changes = listing_changes.assign(
        changed_item="P0207行业代码", value_after=industry["industry_level1_name"]
    )
    no_limit = pd.DataFrame(
        {
            "source_asset_id": [assets[0]],
            "timestamp": [sessions[0] - pd.Timedelta(days=1)],
            "trade_date": [sessions[0] - pd.Timedelta(days=1)],
            "reason": ["P8606"],
        }
    )
    special = normalize_raw_frame(
        pd.DataFrame(
            {
                "Symbol": [assets[0]],
                "Chgtype": ["BA"],
                "Annoudt": [sessions[0] - pd.Timedelta(days=2)],
                "Execudt": [sessions[0] - pd.Timedelta(days=1)],
            }
        ),
        "special_treatment_changes",
    )
    snapshots = pd.DataFrame(
        {
            "source_asset_id": assets,
            "timestamp": sessions[0],
            "observation_date": sessions[0] - pd.Timedelta(days=2),
            "publication_date": sessions[0] - pd.Timedelta(days=1),
            "industry_standard_id": "CSRC2012",
            "industry_category_code": "A",
            "industry_major_code": "",
            "industry_code": "01",
            "company_name": "synthetic issuer",
            "snapshot_id": "earlier-public-observation",
            "source_locator": "page:1",
        }
    )
    return {
        "a_share_industry_snapshots": snapshots,
        "market_bars_1d": market_bars,
        "stock_master": stock_master,
        "market_cap": market_cap,
        "trading_calendar": calendar,
        "industry_classification": industry,
        "industry_classification_sets": industry_sets,
        "industry_classification_versions": industry_versions,
        "listing_status_changes": listing_changes,
        "stock_info_changes": stock_changes,
        "stock_info_annual": annual,
        "trading_no_limit": no_limit,
        "special_treatment_changes": special,
    }


def _write_inputs(root, frames):
    specs = {
        spec.data_id: spec
        for catalog in (csmar_catalog(), choice_catalog(), reference_catalog())
        for spec in catalog.list()
    }
    frames = dict(frames)
    dates = pd.DatetimeIndex(frames["trading_calendar"].calendar_date)
    history = pd.date_range(
        dates[0] - pd.offsets.BDay(300), dates[0] - pd.Timedelta(days=1)
    ).as_unit("ns")
    frames["trading_calendar"] = pd.concat(
        [
            pd.DataFrame(
                dict(
                    timestamp=history,
                    calendar_date=history,
                    source_market_type=1,
                    is_trading_day=history.dayofweek < 5,
                    source_day_of_week=history.dayofweek,
                    source_state=np.where(history.dayofweek < 5, "O", "C"),
                )
            ),
            frames["trading_calendar"],
        ],
        ignore_index=True,
    )
    paths = {}
    for name, frame in frames.items():
        paths[name] = root / name
        write_dataset(paths[name], frame, spec=specs[name])
    return paths


def _prepare(root, name, inputs, start, end):
    spec = processed_catalog().get(name)
    selected = (*spec.source_input_data_ids, *spec.processed_input_data_ids)
    return prepare(
        {
            "market_id": "a_share",
            "data_id": name,
            "inputs": {key: str(inputs[key]) for key in selected if key in inputs},
            "output_root": str(root / name),
            "time_window": {"start": str(start.date()), "end": str(end.date())},
        }
    )


def _source_catalog():
    from quant_research_platform.catalog.processed import InterimDataCatalog

    specs = {
        spec.data_id: spec
        for catalog in (csmar_catalog(), choice_catalog(), reference_catalog())
        for spec in catalog.list()
    }
    return InterimDataCatalog(tuple(specs.values()))


def _published_inputs(layout, spec):
    """Explicit test fixture destinations; never materialize an upstream dataset."""
    sources = _source_catalog()
    processed = processed_catalog()
    return {
        **{
            name: layout.interim_data_root(sources.get(name).source_id, name)
            for name in spec.source_input_data_ids
        },
        **{
            name: layout.processed_data_root(
                processed.get(name).market_id, processed.get(name).kind, name
            )
            for name in spec.processed_input_data_ids
        },
    }


def _calendar_input(root, sessions_by_market):
    """Write explicit synthetic O/C facts for caller-declared test markets."""
    frames = []
    for market, sessions in sessions_by_market.items():
        sessions = pd.DatetimeIndex(sessions)
        days = pd.date_range(sessions.min(), sessions.max()).as_unit("ns")
        opened = days.isin(sessions)
        frames.append(
            pd.DataFrame(
                {
                    "timestamp": days,
                    "calendar_date": days,
                    "source_market_type": market,
                    "is_trading_day": opened,
                    "source_day_of_week": days.dayofweek,
                    "source_state": np.where(opened, "O", "C"),
                }
            )
        )
    return _write_inputs(
        root, {"trading_calendar": pd.concat(frames, ignore_index=True)}
    )["trading_calendar"]

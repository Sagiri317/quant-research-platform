"""Thin installed-package consumer for the synthetic research chain."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from quant_research_platform.profiles.research import research_profile
from quant_research_platform.runtime.resources import (
    current_runtime_resource_policy,
    runtime_resource_policy_scope,
)
from quant_research_platform.sources.csmar.interim import InterimImportResult
from quant_research_platform.storage.datasets.writing import write_dataset
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.workflows import research


def run_research_synthetic_smoke(project_root: Path) -> dict[str, object]:
    """Exercise public research APIs with an explicit local synthetic fixture."""

    project_root = Path(project_root)
    layout = StorageLayout(project_root=project_root)
    base_profile = research_profile()
    source_frames = _synthetic_source_frames()

    def materialize_fixture(**_kwargs: object) -> InterimImportResult:
        policy = current_runtime_resource_policy()
        if policy is None:
            raise RuntimeError("runtime_resource_policy_not_active")
        roots: dict[str, Path] = {}
        for data_id, frame in source_frames.items():
            spec = base_profile.interim.get(data_id)
            root = layout.interim_data_root(spec.source_id, spec.data_id)
            write_dataset(
                root,
                frame,
                spec=spec,
                resource_policy=policy,
                borrowed_input=True,
            )
            roots[data_id] = root
        return InterimImportResult(
            artifact_roots=roots,
            skipped=(),
            pruning_report_text="synthetic fixture",
        )

    profile = replace(
        base_profile,
        producer_bindings={
            **base_profile.producer_bindings,
            **{data_id: materialize_fixture for data_id in source_frames},
        },
    )
    start = source_frames["trading_calendar"]["timestamp"].min().date()
    end = source_frames["trading_calendar"]["timestamp"].max().date()
    time_window = (start.isoformat(), end.isoformat())
    with runtime_resource_policy_scope(None):
        research.data_materialize(
            profile=profile,
            layout=layout,
            raw_root=project_root / "synthetic-raw",
            start_date=time_window[0],
            end_date=time_window[1],
            data_ids=tuple(source_frames),
        )
        dataset_result = research.dataset_build(
            profile=profile,
            layout=layout,
            dataset_id="pv_all",
            start_date=time_window[0],
            end_date=time_window[1],
        )
        research_result = research.research_run(
            profile=profile,
            layout=layout,
            scenario_id="pv_all",
            start_date=time_window[0],
            end_date=time_window[1],
        )
        report = research.report_build(
            profile=profile,
            layout=layout,
            scenario_id="pv_all",
        )
    return {
        "status": "SYNTHETIC_SMOKE_PASS",
        "stages": [
            "data_materialize",
            "dataset_build",
            "research_run",
            "report_build",
        ],
        "dataset_id": "pv_all",
        "dataset_exists": layout.research_dataset_root("a_share", "pv_all").is_dir(),
        "dataset_status": dataset_result["status"],
        "research_status": research_result["status"],
        "report_status": report["status"],
    }


def _synthetic_source_frames() -> dict[str, pd.DataFrame]:
    sessions = pd.bdate_range("2020-01-01", periods=260)
    prices = pd.Series(np.arange(10.0, 270.0), index=sessions)
    bars = pd.DataFrame(
        {
            "source_asset_id": "000001",
            "timestamp": sessions,
            "trade_date": sessions,
            "market_type": 1,
            "trading_status_code": 0,
            "filling_flag": 0,
            "open": prices,
            "high": prices + 1.0,
            "low": prices - 1.0,
            "close": prices,
            "volume": 100.0,
            "amount": 1_000.0,
            "after_hours_volume": 0.0,
            "after_hours_amount": 0.0,
            "prev_close": prices - 1.0,
            "limit_status": 0,
            "period_return": prices.pct_change().fillna(0.0),
            "adj_close": prices,
            "adj_open": prices,
            "float_market_cap": 1_000_000.0,
            "total_market_cap": 1_200_000.0,
            "turnover": np.nan,
        }
    )
    stock = pd.DataFrame(
        {
            "source_asset_id": ["000001"],
            "timestamp": [sessions[0]],
            "stock_name": ["synthetic"],
            "listing_date": [sessions[0] - pd.Timedelta(days=500)],
            "delisting_date": [pd.NaT],
            "status_change_date": [sessions[0] - pd.Timedelta(days=500)],
            "company_status": ["normal"],
            "industry_level1_code": ["A"],
            "industry_level1_name": ["A"],
        }
    )
    market_cap = pd.DataFrame(
        {
            "source_asset_id": ["000001"],
            "timestamp": [sessions[0] - pd.Timedelta(days=500)],
            "share_change_date": [sessions[0] - pd.Timedelta(days=500)],
            "total_shares": [12_000.0],
            "a_float_shares": [10_000.0],
        }
    )
    calendar = pd.DataFrame(
        {
            "timestamp": sessions,
            "calendar_date": sessions,
            "source_market_type": 1,
            "is_trading_day": True,
            "source_day_of_week": sessions.dayofweek,
            "source_state": "O",
        }
    )
    return {
        "market_bars_1d": bars,
        "stock_master": stock,
        "market_cap": market_cap,
        "trading_calendar": calendar,
    }

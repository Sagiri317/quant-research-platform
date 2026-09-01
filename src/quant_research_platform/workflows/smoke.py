"""Thin installed-package consumer for the synthetic research chain."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.data import InterimDataSpec
from quant_research_platform.contracts.targets import ProducerBinding, ProducerKind
from quant_research_platform.profiles.research import research_profile
from quant_research_platform.runtime.resources import (
    current_runtime_resource_policy,
    runtime_resource_policy_scope,
)
from quant_research_platform.sources.csmar.interim import InterimImportResult
from quant_research_platform.storage.datasets.writing import write_dataset
from quant_research_platform.storage.identity import content_hash_for_files
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.workflows import research


def run_research_synthetic_smoke(project_root: Path) -> dict[str, object]:
    """Exercise public research APIs with an explicit local synthetic fixture."""

    project_root = Path(project_root)
    layout = StorageLayout(project_root=project_root)
    base_profile = research_profile()
    source_frames = _synthetic_source_frames()
    for spec in base_profile.interim.list():
        source_frames.setdefault(spec.data_id, _empty_source_frame(spec))
    raw_root = project_root / "synthetic-raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    for data_id in source_frames:
        (raw_root / f"{data_id}.raw").write_bytes(data_id.encode("utf-8"))
    producer_calls: list[str] = []

    def materialize_fixture(**kwargs: object) -> InterimImportResult:
        policy = current_runtime_resource_policy()
        if policy is None:
            raise RuntimeError("runtime_resource_policy_not_active")
        requested = kwargs.get("data_ids")
        if type(requested) is not tuple or len(requested) != 1:
            raise ValueError("synthetic_data_ids_required")
        producer_layout = kwargs.get("layout")
        instance_metadata = kwargs.get("instance_metadata")
        if not isinstance(producer_layout, StorageLayout) or not isinstance(
            instance_metadata, dict
        ):
            raise ValueError("synthetic_target_output_required")
        roots: dict[str, Path] = {}
        for data_id in requested:
            if type(data_id) is not str:
                raise ValueError("synthetic_data_id_invalid")
            frame = source_frames[data_id]
            spec = base_profile.interim.get(data_id)
            output_root = producer_layout.target_root(
                data_id,
                producer_layout.interim_data_root(spec.source_id, data_id),
            )
            producer_calls.append(data_id)
            write_dataset(
                output_root,
                frame,
                spec=spec,
                instance_metadata=instance_metadata,
                resource_policy=policy,
                borrowed_input=True,
            )
            roots[data_id] = output_root
        return InterimImportResult(
            artifact_roots=roots,
            skipped=(),
            pruning_report_text="synthetic fixture",
        )

    profile = replace(
        base_profile,
        producer_bindings={
            **base_profile.producer_bindings,
            **{
                data_id: ProducerBinding(
                    ProducerKind.INTERIM,
                    materialize_fixture,
                    source_identity=_synthetic_source_identity,
                )
                for data_id in source_frames
            },
        },
    )
    start = source_frames["trading_calendar"]["timestamp"].min().date()
    end = source_frames["trading_calendar"]["timestamp"].max().date()
    time_window = (start.isoformat(), end.isoformat())
    with runtime_resource_policy_scope(None):
        dataset_result = research.dataset_build(
            profile=profile,
            layout=layout,
            dataset_id="pv_all",
            start_date=time_window[0],
            end_date=time_window[1],
            raw_root=raw_root,
        )
        producer_calls_after_first = len(producer_calls)
        repeat_result = research.dataset_build(
            profile=profile,
            layout=layout,
            dataset_id="pv_all",
            start_date=time_window[0],
            end_date=time_window[1],
            raw_root=raw_root,
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
        "status": "BLOCKED",
        "stages": [
            "dataset_build",
            "research_run",
            "report_build",
        ],
        "dataset_id": "pv_all",
        "dataset_exists": layout.research_dataset_root("a_share", "pv_all").is_dir(),
        "dataset_status": dataset_result["status"],
        "dataset_repeat_status": repeat_result["status"],
        "dataset_target_status": next(
            item["status"]
            for item in dataset_result["target_results"]
            if item["target_id"] == "pv_all"
        ),
        "dataset_repeat_target_status": next(
            item["status"]
            for item in repeat_result["target_results"]
            if item["target_id"] == "pv_all"
        ),
        "source_producer_calls": producer_calls_after_first,
        "source_producer_calls_on_repeat": len(producer_calls)
        - producer_calls_after_first,
        "research_status": research_result["status"],
        "report_status": report["status"],
        "blocker": "formal_research_capabilities_not_implemented",
    }


def _empty_source_frame(spec: InterimDataSpec) -> pd.DataFrame:
    return pd.DataFrame(
        {
            column: pd.Series(
                dtype="datetime64[ns]" if role is ColumnRole.TIMESTAMP else "string"
            )
            for column, role in spec.column_roles.items()
        }
    )


def _synthetic_source_identity(raw_root: Path, data_id: str) -> str | None:
    path = Path(raw_root) / f"{data_id}.raw"
    if not path.is_file():
        return None
    return content_hash_for_files(
        ((data_id, path),),
        error="synthetic_source_identity_invalid",
        resource_policy=current_runtime_resource_policy(),
    )


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

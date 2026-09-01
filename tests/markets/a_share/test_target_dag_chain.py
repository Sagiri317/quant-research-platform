from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from quant_research_platform.contracts.targets import ProducerBinding, ProducerKind
from quant_research_platform.markets.implementations.a_share.fundamentals import (
    FINANCIAL_VALUE_COLUMNS,
)
from quant_research_platform.profiles.research import research_profile
from quant_research_platform.sources.csmar.interim import (
    InterimImportResult,
    interim_catalog,
)
from quant_research_platform.storage.datasets import read_dataset
from quant_research_platform.storage.datasets.writing import write_dataset
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.workflows import research


def test_a_share_target_dag_builds_market_labels_and_eligible_dataset(
    tmp_path: Path,
) -> None:
    sessions = pd.bdate_range("2020-01-02", periods=400)
    frames = _synthetic_interim_frames(sessions)
    base = research_profile()

    def source_identity(_raw_root: Path, data_id: str) -> str:
        return (f"{data_id}:{len(sessions)}".encode().hex() + "0" * 64)[:64]

    def producer(**kwargs: object) -> InterimImportResult:
        data_ids = kwargs["data_ids"]
        assert isinstance(data_ids, tuple) and len(data_ids) == 1
        data_id = data_ids[0]
        spec = interim_catalog().get(data_id)
        output_root = kwargs["layout"].target_root(
            data_id,
            kwargs["layout"].interim_data_root(spec.source_id, data_id),
        )
        write_dataset(
            output_root,
            frames[data_id],
            spec=spec,
            instance_metadata=kwargs["instance_metadata"],
            resource_policy=kwargs["resource_policy"],
            borrowed_input=True,
        )
        return InterimImportResult(
            artifact_roots={data_id: output_root},
            skipped=(),
            pruning_report_text="synthetic market source",
        )

    bindings = {
        data_id: ProducerBinding(
            ProducerKind.INTERIM,
            producer,
            source_identity=source_identity,
        )
        for data_id in frames
    }
    profile = replace(
        base,
        producer_bindings={**base.producer_bindings, **bindings},
    )
    layout = StorageLayout(project_root=tmp_path)
    result = research.dataset_build(
        profile=profile,
        layout=layout,
        dataset_id="pv_all",
        start_date=sessions[0].date().isoformat(),
        end_date=sessions[-1].date().isoformat(),
        raw_root=tmp_path / "synthetic-raw",
    )

    assert result["status"] == "COMPLETE"
    target_results = {item["target_id"]: item for item in result["target_results"]}
    assert target_results["market_panel_1d"]["status"] == "BUILT"
    assert target_results["execution_mask"]["status"] == "BUILT"
    assert target_results["forward_returns"]["status"] == "BUILT"
    assert target_results["pv_all"]["status"] == "BUILT"
    logical_root = layout.research_dataset_root("a_share", "pv_all")
    output = read_dataset(
        layout.instance_root(
            logical_root, target_results["pv_all"]["instance_identity"]
        ),
    )
    assert {"eligible", "model_target"}.issubset(output.columns)
    assert {"scenario_universe", "can_enter", "can_exit"}.isdisjoint(output)
    assert output["eligible"].any()
    assert output.loc[output["eligible"], "model_target"].notna().any()


def test_fundamental_pit_and_factors_reach_the_same_target_dag(
    tmp_path: Path,
) -> None:
    sessions = pd.bdate_range("2020-01-02", periods=400)
    frames = _synthetic_interim_frames(sessions)
    frames.update(_synthetic_financial_frames())
    base = research_profile()
    raw_root = tmp_path / "synthetic-raw"
    raw_root.mkdir()
    for data_id in frames:
        (raw_root / f"{data_id}.raw").write_bytes(data_id.encode())

    def source_identity(_raw_root: Path, data_id: str) -> str:
        return hashlib.sha256(data_id.encode()).hexdigest()

    def producer(**kwargs: object) -> InterimImportResult:
        data_id = kwargs["data_ids"][0]
        spec = base.interim.get(data_id)
        output_root = kwargs["layout"].target_root(
            data_id,
            kwargs["layout"].interim_data_root(spec.source_id, data_id),
        )
        write_dataset(
            output_root,
            frames[data_id],
            spec=spec,
            instance_metadata=kwargs["instance_metadata"],
            resource_policy=kwargs["resource_policy"],
            borrowed_input=True,
        )
        return InterimImportResult(
            artifact_roots={data_id: output_root},
            skipped=(),
            pruning_report_text="synthetic financial source",
        )

    bindings = {
        data_id: ProducerBinding(
            ProducerKind.INTERIM,
            producer,
            source_identity=source_identity,
        )
        for data_id in frames
    }
    profile = replace(
        base,
        producer_bindings={**base.producer_bindings, **bindings},
    )
    result = research.dataset_build(
        profile=profile,
        layout=StorageLayout(project_root=tmp_path),
        dataset_id="fusion_nf_core",
        start_date=sessions[0].date().isoformat(),
        end_date=sessions[-1].date().isoformat(),
        raw_root=raw_root,
    )

    assert result["status"] == "COMPLETE"
    target_results = {item["target_id"]: item for item in result["target_results"]}
    assert target_results["fundamental_pit"]["status"] == "BUILT"
    assert target_results["fundamental_factors"]["status"] == "BUILT"
    assert target_results["fusion_nf_core"]["status"] == "BUILT"
    output = read_dataset(
        StorageLayout(project_root=tmp_path).instance_root(
            StorageLayout(project_root=tmp_path).research_dataset_root(
                "a_share", "fusion_nf_core"
            ),
            target_results["fusion_nf_core"]["instance_identity"],
        )
    )
    assert output["eligible"].any()
    assert output["model_target"].notna().any()


def _synthetic_financial_frames() -> dict[str, pd.DataFrame]:
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
        frames[data_id] = pd.DataFrame(
            {
                "source_asset_id": ["000001"] * len(periods),
                "timestamp": periods,
                "stock_name": "synthetic",
                "report_period": periods,
                "report_type": "A",
                "is_corrected": False,
                "correction_disclosure_date": pd.Series(
                    pd.NaT, index=range(len(periods)), dtype="datetime64[ns]"
                ),
                "version_identity": "original",
                **{
                    column: [float(index + 1) for index in range(len(periods))]
                    for column in columns
                },
            }
        )
    frames["financial_first_disclosure"] = pd.DataFrame(
        {
            "asset_id": ["000001"] * len(periods),
            "report_period": periods,
            "first_disclosure_date": periods + pd.Timedelta(days=20),
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
            "float_market_cap": 1_000_000.0,
            "total_market_cap": 1_200_000.0,
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
            "total_shares": 120_000.0,
            "a_float_shares": 100_000.0,
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
    stock_changes = listing_changes.copy()
    no_limit = pd.DataFrame(
        {
            "source_asset_id": [assets[0]],
            "timestamp": [sessions[0] - pd.Timedelta(days=1)],
            "trade_date": [sessions[0] - pd.Timedelta(days=1)],
            "reason": ["P8606"],
        }
    )
    special = pd.DataFrame(
        {
            "source_asset_id": [assets[0]],
            "timestamp": [sessions[0] - pd.Timedelta(days=1)],
            "change_type": ["A"],
            "announcement_date": [sessions[0] - pd.Timedelta(days=2)],
            "effective_date": [sessions[0] - pd.Timedelta(days=1)],
        }
    )
    return {
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

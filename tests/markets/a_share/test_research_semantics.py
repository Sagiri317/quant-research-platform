from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.catalog.datasets import ResearchDatasetCatalog
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.implementations.a_share.processed.catalog import (
    processed_catalog,
)
from quant_research_platform.research.datasets.assembler import (
    _add_a_share_semantic_columns,
    assemble_research_dataset,
)
from quant_research_platform.research.datasets.io import iter_research_dataset_chunks
from quant_research_platform.research.features.transforms import (
    canonical_feature_pipeline,
)
from quant_research_platform.research.protocols.a_share import research_dataset_specs
from quant_research_platform.storage.datasets import write_dataset
from quant_research_platform.storage.layout import StorageLayout


def test_a_share_assembler_owns_eligibility_and_population_model_target() -> None:
    spec = research_dataset_specs()[0]
    assert (
        "model_target"
        not in processed_catalog().get("forward_returns").required_columns
    )
    for dataset_spec in research_dataset_specs():
        assert {
            "scenario_universe",
            "can_enter",
            "can_exit",
        }.isdisjoint(dataset_spec.required_columns)
    frame = pd.DataFrame(
        {
            "market_id": "a_share",
            "asset_id": ["a", "b", "c"],
            "timestamp": pd.Timestamp("2024-01-02"),
            "is_a_share_scope": True,
            "is_trading": True,
            "is_st": [False, False, True],
            "is_pt": False,
            "listing_age_trading_days": 252,
            "close": 10.0,
            "adj_close": 10.0,
            "volume": 100.0,
            "amount": 1_000.0,
            "turnover": 0.1,
            "forward_return_20d": [0.10, 0.20, np.nan],
        }
    )

    result = _add_a_share_semantic_columns(frame, spec)

    assert result["eligible"].tolist() == [True, True, False]
    assert result["model_target"].iloc[:2].tolist() == pytest.approx([-1.0, 1.0])
    assert pd.isna(result.loc[2, "model_target"])
    assert "can_enter" not in result
    assert "can_exit" not in result
    assert "scenario_universe" not in result

    fusion = frame.assign(
        industry_standard_id="CSRC2012",
        industry_level1_code="A01",
        industry_level1_name="industry",
        is_nonfinancial_industry=True,
        has_legal_fundamental_pit=[False, True, True],
    )
    fusion_result = _add_a_share_semantic_columns(fusion, research_dataset_specs()[1])
    assert fusion_result["eligible"].tolist() == [False, True, False]
    assert not _add_a_share_semantic_columns(
        fusion.assign(is_nonfinancial_industry=False), research_dataset_specs()[1]
    )["eligible"].any()


def test_a_share_model_target_zero_variance_does_not_invalidate_other_date() -> None:
    spec = research_dataset_specs()[0]
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"]
            ),
            "is_a_share_scope": True,
            "is_trading": True,
            "is_st": False,
            "is_pt": False,
            "listing_age_trading_days": 252,
            "close": 10.0,
            "adj_close": 10.0,
            "volume": 100.0,
            "amount": 1_000.0,
            "turnover": 0.1,
            "forward_return_20d": [0.10, 0.10, 0.10, 0.20],
        }
    )

    result = _add_a_share_semantic_columns(frame, spec)

    assert result.loc[result["timestamp"].eq("2024-01-02"), "model_target"].isna().all()
    assert result.loc[
        result["timestamp"].eq("2024-01-03"), "model_target"
    ].tolist() == pytest.approx([-1.0, 1.0])


def test_insufficient_neutralization_is_missing_not_observed_zero() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2024-01-02"),
            "eligible": True,
            "industry_level1_code": "A",
            "log_float_market_cap": [1.0, 2.0],
            "factor": [1.0, 2.0],
        }
    )

    result = canonical_feature_pipeline(frame, factor_columns=("factor",))

    assert result["feature_factor"].tolist() == [0.0, 0.0]
    assert result["feature_missing_factor"].tolist() == [1, 1]
    assert result["feature_observed_factor"].tolist() == [0, 0]


def test_a_share_assembler_persists_only_eligible_sample_truth(
    tmp_path,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    processed = processed_catalog()
    dataset_spec = research_dataset_specs()[0]
    timestamps = pd.to_datetime(["2024-01-02"] * 3)
    keys = pd.DataFrame(
        {
            "market_id": "a_share",
            "asset_id": ["a", "b", "c"],
            "timestamp": timestamps,
        }
    )
    panel_spec = processed.get("market_panel_1d")
    panel = keys.copy()
    values = {
        "session_date": timestamps,
        "available_at": timestamps,
        "stock_name": ["a", "b", "c"],
        "market_type": 1,
        "trading_status_code": 0,
        "filling_flag": 0,
        "open": 10.0,
        "high": 10.0,
        "low": 10.0,
        "close": 10.0,
        "adj_open": 10.0,
        "adj_close": 10.0,
        "prev_close": 10.0,
        "volume": 100.0,
        "amount": 1_000.0,
        "after_hours_volume": 0.0,
        "after_hours_amount": 0.0,
        "period_return": 0.0,
        "turnover": 0.1,
        "limit_status": 0.0,
        "is_trading": True,
        "is_a_share_scope": True,
        "is_st": False,
        "is_pt": False,
        "listing_date": pd.Timestamp("2020-01-01"),
        "delisting_date": pd.NaT,
        "listing_age_trading_days": 252,
        "industry_standard_id": "CSRC2012",
        "industry_level1_code": "A",
        "industry_level1_name": "industry",
        "is_nonfinancial_industry": True,
        "share_change_date": pd.Timestamp("2020-01-01"),
        "total_shares": 1_000.0,
        "a_float_shares": 1_000.0,
        "total_market_cap": 10_000.0,
        "float_market_cap": 10_000.0,
    }
    for column, value in values.items():
        panel[column] = value
    write_dataset(
        layout.processed_data_root(
            panel_spec.market_id, panel_spec.kind, panel_spec.data_id
        ),
        panel.loc[:, list(panel_spec.required_columns)],
        spec=panel_spec,
    )

    factor_spec = processed.get("price_volume_facts")
    factors = keys.copy()
    for column in factor_spec.required_columns:
        if column not in factors:
            factors[column] = [1.0, 2.0, 3.0]
    write_dataset(
        layout.processed_data_root(
            factor_spec.market_id, factor_spec.kind, factor_spec.data_id
        ),
        factors.loc[:, list(factor_spec.required_columns)],
        spec=factor_spec,
    )

    label_spec = processed.get("forward_returns")
    labels = keys.copy()
    for column in label_spec.required_columns:
        if column in labels:
            continue
        if column == "forward_return_20d":
            labels[column] = [0.10, 0.20, np.nan]
        elif column.endswith("_available_at"):
            labels[column] = timestamps
        else:
            labels[column] = 0.10
    write_dataset(
        layout.processed_data_root(
            label_spec.market_id, label_spec.kind, label_spec.data_id
        ),
        labels.loc[:, list(label_spec.required_columns)],
        spec=label_spec,
    )

    assemble_research_dataset(
        layout=layout,
        dataset_catalog=ResearchDatasetCatalog((dataset_spec,)),
        processed_catalog=processed,
        dataset_id=dataset_spec.data_id,
        time_window=TimeWindow("2024-01-02", "2024-01-02"),
    )
    result = pd.concat(
        iter_research_dataset_chunks(
            layout,
            market_id=MarketId("a_share"),
            dataset_id=dataset_spec.data_id,
            time_window=TimeWindow("2024-01-02", "2024-01-02"),
        ),
        ignore_index=True,
    )

    assert set(("eligible", "model_target")).issubset(result.columns)
    assert {"scenario_universe", "can_enter", "can_exit"}.isdisjoint(result)
    assert result["eligible"].tolist() == [True, True, True]
    assert result["model_target"].iloc[:2].tolist() == pytest.approx([-1.0, 1.0])
    assert pd.isna(result.loc[2, "model_target"])

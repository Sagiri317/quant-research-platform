from __future__ import annotations

import warnings
from datetime import timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

import quant_research_platform.research.portfolios.runner as portfolio_runner
from quant_research_platform.artifacts._output_bundle_validation import (
    _read_prediction_output_bundle,
)
from quant_research_platform.catalog.portfolios import PortfolioCatalog
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.portfolios import (
    PortfolioDiagnosticsConfig,
    PortfolioSpec,
)
from quant_research_platform.contracts.predictions import PredictionSpec
from quant_research_platform.contracts.signals import SignalSpec
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.research.portfolios.runner import (
    build_portfolio_result_from_research_dataset,
)
from quant_research_platform.storage.datasets import write_dataset
from quant_research_platform.storage.layout import StorageLayout
from tests.helpers import frequency_ids
from tests.helpers.research_specs import (
    make_portfolio_research_dataset_spec,
    make_research_dataset_catalog,
)
from tests.research.prediction_output_helpers import write_prediction_output_and_read


def test_portfolio_read_columns_include_diagnostics_config_columns() -> None:
    spec = PortfolioSpec(
        portfolio_id="portfolio_demo",
        market_id=MarketId("a_share"),
        research_dataset_id="research_demo",
        signals=(
            SignalSpec(
                signal_id="signal",
                source_column="signal",
                source_kind="factor",
            ),
        ),
        return_source_column="forward_return",
        rebalance_frequency=frequency_ids.DAILY_FREQUENCY_SLUG,
        execution_lag_steps=0,
        diagnostics_config=PortfolioDiagnosticsConfig(
            capacity_amount_column="amount",
            capacity_float_market_cap_column="float_market_cap",
            numeric_exposure_columns=("beta",),
            categorical_exposure_columns=("industry",),
        ),
    )

    assert portfolio_runner._portfolio_read_columns(spec) == (
        "market_id",
        "asset_id",
        "timestamp",
        "signal",
        "forward_return",
        "amount",
        "float_market_cap",
        "beta",
        "industry",
    )


def test_portfolio_runner_accepts_equivalent_timezone_representations() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": [
                pd.Timestamp(
                    "2024-01-15T12:00:00",
                    tz=timezone(timedelta(hours=1)),
                ),
                pd.Timestamp(
                    "2024-01-15T12:01:00",
                    tz=ZoneInfo("Europe/Berlin"),
                ),
            ]
        }
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        timestamps = portfolio_runner._timestamp_set(frame)

    assert timestamps == {
        pd.Timestamp("2024-01-15T11:00:00"),
        pd.Timestamp("2024-01-15T11:01:00"),
    }


def test_portfolio_runner_joins_persisted_prediction_primary_key_with_explicit_gaps(
    tmp_path,
) -> None:
    layout, dataset_catalog, portfolio_catalog, prediction, manifest = (
        _prediction_portfolio_inputs(tmp_path)
    )

    result = build_portfolio_result_from_research_dataset(
        layout=layout,
        dataset_catalog=dataset_catalog,
        portfolio_catalog=portfolio_catalog,
        portfolio_id="demo_prediction_portfolio",
        time_window=TimeWindow("2024-01-01", "2024-01-02"),
        prediction_frame=prediction,
        prediction_metadata=manifest["metadata"],
    )
    chunks = list(result.output_chunks)
    returns = pd.concat(
        [chunk["portfolio_returns"] for chunk in chunks],
        ignore_index=True,
    )

    assert returns["timestamp"].tolist() == [pd.Timestamp("2024-01-02")]
    assert returns["gross_portfolio_return"].tolist() == pytest.approx([0.20])


@pytest.mark.parametrize(
    ("corruption", "error"),
    [
        ("duplicate", "duplicate_persisted_prediction_keys"),
        ("missing", "missing_required_prediction_keys"),
        ("misaligned", "misaligned_persisted_prediction_keys"),
    ],
)
def test_portfolio_runner_rejects_invalid_persisted_prediction_coverage(
    tmp_path,
    corruption: str,
    error: str,
) -> None:
    layout, dataset_catalog, portfolio_catalog, prediction, manifest = (
        _prediction_portfolio_inputs(tmp_path)
    )
    if corruption == "duplicate":
        prediction = pd.concat([prediction, prediction.iloc[[0]]], ignore_index=True)
    elif corruption == "missing":
        prediction = prediction.iloc[1:].reset_index(drop=True)
    else:
        prediction = prediction.copy()
        prediction.loc[0, "asset_id"] = "misaligned"

    with pytest.raises(ValueError, match=error):
        result = build_portfolio_result_from_research_dataset(
            layout=layout,
            dataset_catalog=dataset_catalog,
            portfolio_catalog=portfolio_catalog,
            portfolio_id="demo_prediction_portfolio",
            time_window=TimeWindow("2024-01-01", "2024-01-02"),
            prediction_frame=prediction,
            prediction_metadata=manifest["metadata"],
        )
        list(result.output_chunks)


def _prediction_portfolio_inputs(tmp_path):
    layout = StorageLayout(project_root=tmp_path)
    dataset_spec = make_portfolio_research_dataset_spec(
        factor_columns=("model_input",),
        label_columns=("forward_return",),
        filter_columns=("eligible",),
    )
    dataset_catalog = make_research_dataset_catalog(dataset_spec)
    frame = pd.DataFrame(
        {
            "market_id": ["a_share"] * 8,
            "asset_id": ["a", "b", "c", "d"] * 2,
            "timestamp": [pd.Timestamp("2024-01-01")] * 4
            + [pd.Timestamp("2024-01-02")] * 4,
            "model_input": [1.0, 2.0, 3.0, 4.0, 4.0, 1.0, np.nan, 3.0],
            "forward_return": [0.0, 0.0, 0.0, 0.0, 0.1, -0.1, 0.5, 0.2],
            "eligible": [True] * 7 + [False],
        }
    )
    write_dataset(
        layout.research_dataset_root(MarketId("a_share"), dataset_spec.data_id),
        frame,
        spec=dataset_spec,
    )
    prediction_spec = PredictionSpec(
        prediction_id="demo_prediction",
        model_id="linear_regression_demo",
        research_dataset_id=dataset_spec.data_id,
        train_end="2024-01-01",
        test_start="2024-01-02",
        test_end="2024-01-02",
    )
    _prediction, root, _manifest = write_prediction_output_and_read(
        tmp_path,
        (frame,),
        prediction_spec,
        np.array([0.0, 1.0]),
        ("model_input",),
    )
    manifest, prediction = _read_prediction_output_bundle(
        root,
        expected_output_id=prediction_spec.prediction_id,
    )
    portfolio_catalog = PortfolioCatalog()
    portfolio_catalog.register(
        PortfolioSpec(
            portfolio_id="demo_prediction_portfolio",
            market_id=MarketId("a_share"),
            research_dataset_id=dataset_spec.data_id,
            signals=(
                SignalSpec(
                    signal_id=prediction_spec.prediction_id,
                    source_column=prediction_spec.prediction_column,
                    source_kind="prediction",
                ),
            ),
            return_source_column="forward_return",
            rebalance_frequency=frequency_ids.DAILY_FREQUENCY_SLUG,
            execution_lag_steps=0,
        )
    )
    return layout, dataset_catalog, portfolio_catalog, prediction, manifest

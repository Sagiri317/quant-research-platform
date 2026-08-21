from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from quant_research_platform.catalog.datasets import ResearchDatasetCatalog
from quant_research_platform.catalog.evaluations import EvaluationCatalog
from quant_research_platform.catalog.factors import FactorCatalog
from quant_research_platform.catalog.fields import FieldCatalog
from quant_research_platform.catalog.labels import LabelCatalog
from quant_research_platform.catalog.markets import MarketRegistry
from quant_research_platform.catalog.models import ModelCatalog
from quant_research_platform.catalog.outputs import OutputCatalog
from quant_research_platform.catalog.portfolios import PortfolioCatalog
from quant_research_platform.catalog.predictions import PredictionCatalog
from quant_research_platform.catalog.processed import (
    ProcessedDataCatalog,
)
from quant_research_platform.catalog.sources import SourceRegistry
from quant_research_platform.contracts.ids import SourceId
from quant_research_platform.contracts.signals import SignalSpec
from quant_research_platform.markets.implementations.a_share import (
    research_scenarios,
)
from quant_research_platform.research.portfolios.engine import (
    construct_portfolio_positions,
)
from quant_research_platform.runtime.resources import (
    runtime_resource_policy_scope,
)
from quant_research_platform.sources.csmar.interim import InterimImportResult
from quant_research_platform.storage.layout import StorageLayout
from tests.helpers import frequency_ids
from tests.helpers.runtime_resources import capture_test_policy


def test_research_profile_is_only_public_profile_builder() -> None:
    import quant_research_platform.profiles.research as profiles

    assert set(profiles.__all__) == {
        "ResearchProfile",
        "research_profile",
        "research_profile_id",
    }


def test_research_profile_contains_active_catalogs() -> None:
    from quant_research_platform.profiles.research import (
        ResearchProfile,
        research_profile,
        research_profile_id,
    )

    profile = research_profile()

    assert isinstance(profile, ResearchProfile)
    assert profile.profile_id == "a_share_csmar_1d"
    assert (
        research_profile_id("a_share", "csmar", frequency_ids.DAILY_FREQUENCY_SLUG)
        == profile.profile_id
    )
    assert research_profile(profile.profile_id) == profile
    assert isinstance(profile.fields, FieldCatalog)
    assert isinstance(profile.factors, FactorCatalog)
    assert isinstance(profile.labels, LabelCatalog)
    assert isinstance(profile.processed, ProcessedDataCatalog)
    assert isinstance(profile.research_datasets, ResearchDatasetCatalog)
    assert isinstance(profile.outputs, OutputCatalog)
    assert isinstance(profile.evaluations, EvaluationCatalog)
    assert isinstance(profile.models, ModelCatalog)
    assert isinstance(profile.predictions, PredictionCatalog)
    assert isinstance(profile.portfolios, PortfolioCatalog)
    assert isinstance(profile.sources, SourceRegistry)
    assert isinstance(profile.markets, MarketRegistry)
    assert set(profile.source_interim_catalogs) == {"csmar"}
    assert profile.source_id == SourceId("csmar")
    assert callable(profile.materialize_interim_data)
    assert callable(profile.materialize_processed_data)


def test_a_share_profile_materializes_exact_processed_dependency_source_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import quant_research_platform.profiles.research as profiles

    calls: list[dict[str, object]] = []

    def materialize(**kwargs):  # noqa: ANN003
        calls.append(kwargs)
        data_ids = tuple(kwargs.get("data_ids", ("direct",)))
        return InterimImportResult(
            artifact_roots={data_id: tmp_path / data_id for data_id in data_ids},
            skipped=(),
            pruning_report_text="",
        )

    def sessions(*_args, **kwargs):  # noqa: ANN002, ANN003
        assert kwargs["lookback_sessions"] == 252
        assert kwargs["lookahead_sessions"] == 121
        return pd.DatetimeIndex(
            [
                *pd.bdate_range(end="2014-01-01", periods=252),
                pd.Timestamp("2014-01-02"),
                pd.Timestamp("2014-01-03"),
                *pd.bdate_range(start="2014-01-06", periods=121),
            ]
        )

    monkeypatch.setattr(profiles, "_materialize_csmar_interim_data", materialize)
    monkeypatch.setattr(
        profiles.a_share_materialization_planning,
        "available_trading_sessions",
        sessions,
    )
    profile = profiles.research_profile()
    policy = capture_test_policy(available_memory_bytes=64 * 1024**2)

    with runtime_resource_policy_scope(policy):
        result = profile.materialize_interim_data(
            layout=StorageLayout(project_root=tmp_path),
            raw_root=tmp_path / "raw",
            start_date="2014-01-02",
            end_date="2014-01-03",
            resource_policy=policy,
        )

    assert all(call["resource_policy"] is policy for call in calls)
    assert calls[0]["data_ids"] == ("trading_calendar",)
    assert calls[0]["start_date"] is None
    assert calls[0]["end_date"] is None
    assert calls[1]["start_date"] == "2013-01-15"
    assert calls[1]["end_date"] == "2014-06-23"
    assert "trading_calendar" in result.artifact_roots

    calls.clear()
    legacy_result = profile.materialize_interim_data(
        StorageLayout(project_root=tmp_path),
        tmp_path / "raw",
        "2014-01-02",
        "2014-01-03",
    )

    legacy_policy = calls[0]["resource_policy"]
    assert legacy_policy is not None
    assert all(call["resource_policy"] is legacy_policy for call in calls)
    assert legacy_policy is not policy
    assert "trading_calendar" in legacy_result.artifact_roots


@pytest.mark.parametrize("profile_id", (None, "crypto_binance_1m"))
def test_profile_interim_call_reuses_ambient_root(
    tmp_path: Path,
    profile_id: str | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import quant_research_platform.profiles.research as profiles

    calls: list[dict[str, object]] = []

    if profile_id is None:
        def materialize(**kwargs):  # noqa: ANN003
            calls.append(kwargs)
            data_ids = tuple(kwargs.get("data_ids", ("direct",)))
            return InterimImportResult(
                artifact_roots={data_id: tmp_path / data_id for data_id in data_ids},
                skipped=(),
                pruning_report_text="",
            )

        def sessions(*_args, **kwargs):  # noqa: ANN002, ANN003
            assert kwargs["lookback_sessions"] == 252
            assert kwargs["lookahead_sessions"] == 121
            return pd.DatetimeIndex(
                [
                    *pd.bdate_range(end="2014-01-01", periods=252),
                    pd.Timestamp("2014-01-02"),
                    pd.Timestamp("2014-01-03"),
                    *pd.bdate_range(start="2014-01-06", periods=121),
                ]
            )

        monkeypatch.setattr(profiles, "_materialize_csmar_interim_data", materialize)
        monkeypatch.setattr(
            profiles.a_share_materialization_planning,
            "available_trading_sessions",
            sessions,
        )
    else:
        def materialize(**kwargs):  # noqa: ANN003
            calls.append(kwargs)
            return object()

        monkeypatch.setattr(profiles, "_materialize_binance_interim_data", materialize)

    from quant_research_platform.profiles.research import research_profile

    policy = capture_test_policy(available_memory_bytes=64 * 1024**2)
    profile = research_profile(profile_id)

    with runtime_resource_policy_scope(policy):
        profile.materialize_interim_data(
            StorageLayout(project_root=tmp_path),
            tmp_path / "raw",
            "2014-01-02",
            "2014-01-03",
        )

    assert calls
    assert all(call["resource_policy"] is policy for call in calls)


def test_research_profile_covers_required_models_predictions_and_signals() -> None:
    from quant_research_platform.profiles.research import research_profile
    from quant_research_platform.research.models.algorithm_facts import (
        model_algorithm_ids,
    )

    profile = research_profile()
    algorithms = tuple(model_algorithm_ids())
    dataset = profile.research_datasets.get("price_volume_forward_returns")
    evaluation = profile.evaluations.get("signal_forward_return_20d")
    portfolio = research_scenarios.build_portfolio_catalog(
        execution_lag_steps=1,
    ).get("signal_forward_return_20d_portfolio")
    models = tuple(profile.models.list())
    predictions = tuple(
        research_scenarios.build_prediction_catalog(
            train_end="2014-06-30",
            test_start="2014-07-01",
        ).list()
    )
    model_ids = {model.model_id for model in models}

    assert profile.predictions.list() == ()
    assert profile.portfolios.list() == ()
    assert sorted(model.model_algorithm for model in models) == sorted(algorithms)
    assert len(predictions) == len(algorithms)
    assert {prediction.model_id for prediction in predictions} == model_ids
    assert all(
        prediction.research_dataset_id == dataset.data_id for prediction in predictions
    )
    assert dataset.factor_columns
    assert dataset.feature_columns == ("mom_20d_zscore",)
    assert [
        (
            transform.source_columns,
            transform.output_columns,
            transform.transform,
        )
        for transform in dataset.feature_transforms
    ] == [(("mom_20d",), ("mom_20d_zscore",), "cross_sectional_zscore")]
    assert [
        (signal.signal_id, signal.source_column, signal.source_kind)
        for signal in evaluation.signals
    ] == [
        ("mom_20d", "mom_20d", "factor"),
        ("rev_1d", "rev_1d", "factor"),
        ("mom_20d_zscore", "mom_20d_zscore", "feature"),
    ]
    assert [
        (signal.signal_id, signal.source_column, signal.source_kind)
        for signal in portfolio.signals
    ] == [
        ("mom_20d", "mom_20d", "factor"),
        ("rev_1d", "rev_1d", "factor"),
        ("mom_20d_zscore", "mom_20d_zscore", "feature"),
    ]


def test_a_share_scenario_has_distinct_default_and_smoke_model_parameters() -> None:
    default = research_scenarios.build_model_specs(
        model_algorithm_ids=("random_forest_regressor",),
    )[0]
    smoke = research_scenarios.build_model_specs(
        model_algorithm_ids=("random_forest_regressor",),
        hyperparameter_preset="smoke",
    )[0]

    assert dict(default.hyperparameters) == {"n_estimators": 20, "n_jobs": -1}
    assert dict(smoke.hyperparameters) == {
        "n_estimators": 5,
        "max_depth": 3,
        "n_jobs": -1,
    }


def test_a_share_scenario_adds_prediction_portfolio_and_keeps_factor() -> None:
    prediction_id = "linear_regression_mom_20d_prediction_2015"

    portfolios = research_scenarios.build_portfolio_specs(
        prediction_id=prediction_id,
    )

    assert [spec.portfolio_id for spec in portfolios] == [
        "signal_forward_return_20d_portfolio",
        f"{prediction_id}_portfolio",
    ]
    assert portfolios[0].signals == research_scenarios.build_signal_specs()
    assert [
        (signal.signal_id, signal.source_column, signal.source_kind)
        for signal in portfolios[1].signals
    ] == [(prediction_id, "prediction_score", "prediction")]
    for portfolio in portfolios:
        assert portfolio.tradability_config.eligibility_column == "eligible"
        assert portfolio.tradability_config.can_enter_column == "can_enter"
        assert portfolio.tradability_config.can_exit_column == "can_exit"


def test_a_share_factor_and_prediction_portfolios_apply_formal_masks() -> None:
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    frame = pd.DataFrame.from_records(
        [
            {
                "asset_id": asset_id,
                "timestamp": timestamp,
                "factor_score": signal,
                "prediction_score": signal,
                "forward_return_1d": 0.01,
                "eligible": asset_id != "b",
                "can_enter": not (timestamp == dates[1] and asset_id == "a"),
                "can_exit": not (timestamp == dates[1] and asset_id == "d"),
                "amount": 1_000_000.0,
                "float_market_cap": 10_000_000.0,
                "csrc_2012_industry_code": "demo",
            }
            for timestamp, signals in zip(
                dates,
                ((1.0, 2.0, 3.0, 4.0), (4.0, 3.0, 2.0, 1.0)),
                strict=True,
            )
            for asset_id, signal in zip(("a", "b", "c", "d"), signals, strict=True)
        ]
    )
    specs = research_scenarios.build_portfolio_specs(
        signals=(
            SignalSpec(
                signal_id="factor",
                source_kind="factor",
                source_column="factor_score",
            ),
        ),
        prediction_id="prediction",
    )

    for spec in specs:
        result = construct_portfolio_positions(
            frame,
            spec,
            source_columns_by_kind={
                spec.signals[0].source_kind: (spec.signals[0].source_column,)
            },
        )
        final = result.positions.loc[result.positions["timestamp"].eq(dates[1])]
        assert dict(zip(final["asset"], final["position"], strict=True)) == {
            "a": -1.0,
            "d": 1.0,
        }


def test_catalog_profile_source_interim_catalog_requires_single_source(
    monkeypatch,
) -> None:
    import quant_research_platform.profiles.research as profiles

    def fake_catalog() -> object:
        return SimpleNamespace(
            list=lambda: [
                SimpleNamespace(source_id="synthetic_a"),
                SimpleNamespace(source_id="synthetic_b"),
            ]
        )

    monkeypatch.setattr(profiles, "csmar_interim_catalog", fake_catalog)

    with pytest.raises(ValueError, match="unexpected_source_interim_catalog_sources"):
        profiles.research_profile()

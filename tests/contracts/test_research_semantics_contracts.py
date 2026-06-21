from __future__ import annotations

import pytest
from pydantic import ValidationError

from quant_research_platform.compute.panels import ComputePanelSpec
from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.data import DataFrequency, DataGrain
from quant_research_platform.contracts.datasets import ResearchDatasetSpec
from quant_research_platform.contracts.evaluations import EvaluationSpec
from quant_research_platform.contracts.features import FeatureTransformSpec
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.models import ModelSpec
from quant_research_platform.contracts.portfolios import PortfolioSpec, WeightConfig
from quant_research_platform.contracts.signals import SignalSpec
from tests.helpers import frequency_ids


def test_research_dataset_splits_factor_and_feature_columns_without_alias() -> None:
    spec = ResearchDatasetSpec(
        data_id="a_share_research",
        frequency=DataFrequency.DAILY,
        grain=DataGrain.ASSET_TIMESTAMP,
        column_roles={
            "market_id": ColumnRole.KEY,
            "asset_id": ColumnRole.KEY,
            "timestamp": ColumnRole.TIMESTAMP,
            "mom_20d": ColumnRole.FACTOR,
            "mom_20d_zscore": ColumnRole.FEATURE,
            "forward_return_20d": ColumnRole.LABEL,
        },
        primary_key=("market_id", "asset_id", "timestamp"),
        partition_columns=("year",),
        market_id=MarketId("a_share"),
        input_data_ids=("factor_panel", "feature_panel"),
        base_input_data_id="factor_panel",
        factor_columns=("mom_20d",),
        feature_columns=("mom_20d_zscore",),
        feature_transforms=(
            FeatureTransformSpec(
                transform_id="mom_20d_zscore",
                source_columns=("mom_20d",),
                output_columns=("mom_20d_zscore",),
                transform="cross_sectional_zscore",
            ),
        ),
        label_columns=("forward_return_20d",),
    )

    assert spec.factor_columns == ("mom_20d",)
    assert spec.feature_columns == ("mom_20d_zscore",)
    assert "factor_columns" in ResearchDatasetSpec.__dataclass_fields__
    assert "feature_columns" in ResearchDatasetSpec.__dataclass_fields__

    with pytest.raises(ValueError, match="feature_columns_without_transform"):
        ResearchDatasetSpec(
            data_id="bad_missing_transform",
            frequency=DataFrequency.DAILY,
            grain=DataGrain.ASSET_TIMESTAMP,
            column_roles={
                "market_id": ColumnRole.KEY,
                "asset_id": ColumnRole.KEY,
                "timestamp": ColumnRole.TIMESTAMP,
                "mom_20d": ColumnRole.FACTOR,
                "mom_20d_zscore": ColumnRole.FEATURE,
                "forward_return_20d": ColumnRole.LABEL,
            },
            primary_key=("market_id", "asset_id", "timestamp"),
            partition_columns=("year",),
            market_id=MarketId("a_share"),
            input_data_ids=("factor_panel",),
            base_input_data_id="factor_panel",
            factor_columns=("mom_20d",),
            feature_columns=("mom_20d_zscore",),
            label_columns=("forward_return_20d",),
        )

    with pytest.raises(ValueError, match="research_factor_feature_overlap"):
        ResearchDatasetSpec(
            data_id="bad_research",
            frequency=DataFrequency.DAILY,
            grain=DataGrain.ASSET_TIMESTAMP,
            column_roles={
                "market_id": ColumnRole.KEY,
                "asset_id": ColumnRole.KEY,
                "timestamp": ColumnRole.TIMESTAMP,
                "mom_20d": ColumnRole.FACTOR,
                "forward_return_20d": ColumnRole.LABEL,
            },
            primary_key=("market_id", "asset_id", "timestamp"),
            partition_columns=("year",),
            market_id=MarketId("a_share"),
            input_data_ids=("factor_panel",),
            base_input_data_id="factor_panel",
            factor_columns=("mom_20d",),
            feature_columns=("mom_20d",),
            label_columns=("forward_return_20d",),
        )


def test_feature_transform_spec_rejects_raw_factor_pass_through() -> None:
    spec = FeatureTransformSpec(
        transform_id="mom_20d_zscore",
        source_columns=("mom_20d",),
        output_columns=("mom_20d_zscore",),
        transform="cross_sectional_zscore",
    )

    assert spec.source_columns == ("mom_20d",)
    assert spec.output_columns == ("mom_20d_zscore",)

    with pytest.raises(ValueError, match="feature_transform_raw_factor_passthrough"):
        FeatureTransformSpec(
            transform_id="bad_passthrough",
            source_columns=("mom_20d",),
            output_columns=("mom_20d",),
            transform="cross_sectional_zscore",
        )


def test_model_spec_uses_algorithm_and_explicit_factor_feature_inputs() -> None:
    spec = ModelSpec(
        model_id="linear_mom",
        model_algorithm="linear_regression",
        input_factor_columns=("mom_20d",),
        input_feature_columns=("mom_20d_zscore",),
        label_column="forward_return_20d",
    )

    assert spec.model_algorithm == "linear_regression"
    assert spec.input_columns == ("mom_20d", "mom_20d_zscore")
    assert not hasattr(spec, "model_family")
    assert not hasattr(spec, "feature_columns")

    with pytest.raises(ValueError, match="model_input_columns_required"):
        ModelSpec(
            model_id="empty_model",
            model_algorithm="linear_regression",
            input_factor_columns=(),
            input_feature_columns=(),
            label_column="forward_return_20d",
        )


def test_evaluation_and_portfolio_specs_are_signal_owned() -> None:
    signal = SignalSpec(
        signal_id="linear_prediction_score",
        source_column="prediction_score",
        source_kind="prediction",
        direction="higher_is_better",
    )
    evaluation = EvaluationSpec(
        run_id="signal_eval",
        market_id=MarketId("a_share"),
        research_dataset_id="a_share_research",
        signals=(signal,),
        label_columns=("forward_return_20d",),
    )
    portfolio = PortfolioSpec(
        portfolio_id="signal_portfolio",
        market_id=MarketId("a_share"),
        research_dataset_id="a_share_research",
        signals=(signal,),
        return_source_column="forward_return_20d",
        rebalance_frequency=frequency_ids.DAILY_FREQUENCY_SLUG,
        execution_lag_steps=0,
    )

    assert evaluation.signals == (signal,)
    assert portfolio.signals == (signal,)
    assert not hasattr(evaluation, "feature_columns")
    assert not hasattr(portfolio, "prediction_id")

    with pytest.raises(TypeError):
        EvaluationSpec(
            run_id="bad_eval",
            market_id=MarketId("a_share"),
            research_dataset_id="a_share_research",
            feature_columns=("prediction_score",),
            label_columns=("forward_return_20d",),
        )


def test_signal_contracts_reject_filter_and_label_collisions() -> None:
    with pytest.raises(ValueError, match="filter_column_in_input_columns"):
        SignalSpec(
            signal_id="eligible",
            source_column="prediction_score",
            source_kind="prediction",
        )

    with pytest.raises(ValueError, match="evaluation_signal_label_overlap"):
        EvaluationSpec(
            run_id="bad_eval_signal_column",
            market_id=MarketId("a_share"),
            research_dataset_id="a_share_research",
            signals=(
                SignalSpec(
                    signal_id="prediction_score",
                    source_column="forward_return_20d",
                    source_kind="prediction",
                ),
            ),
            label_columns=("forward_return_20d",),
        )

    with pytest.raises(ValueError, match="evaluation_signal_label_overlap"):
        EvaluationSpec(
            run_id="bad_eval_signal_id",
            market_id=MarketId("a_share"),
            research_dataset_id="a_share_research",
            signals=(
                SignalSpec(
                    signal_id="forward_return_20d",
                    source_column="prediction_score",
                    source_kind="prediction",
                ),
            ),
            label_columns=("forward_return_20d",),
        )


def test_signal_contracts_reject_cross_signal_id_column_collisions() -> None:
    own_name_signal = SignalSpec(
        signal_id="prediction_score",
        source_column="prediction_score",
        source_kind="prediction",
    )
    EvaluationSpec(
        run_id="self_named_eval",
        market_id=MarketId("a_share"),
        research_dataset_id="a_share_research",
        signals=(own_name_signal,),
        label_columns=("forward_return_20d",),
    )
    PortfolioSpec(
        portfolio_id="self_named_portfolio",
        market_id=MarketId("a_share"),
        research_dataset_id="a_share_research",
        signals=(own_name_signal,),
        return_source_column="forward_return_20d",
        rebalance_frequency=frequency_ids.DAILY_FREQUENCY_SLUG,
        execution_lag_steps=0,
    )

    cross_collision = (
        SignalSpec(
            signal_id="signal_a",
            source_column="raw_a",
            source_kind="factor",
        ),
        SignalSpec(
            signal_id="raw_a",
            source_column="raw_b",
            source_kind="factor",
        ),
    )
    with pytest.raises(ValueError, match="evaluation_signal_id_column_overlap"):
        EvaluationSpec(
            run_id="bad_eval_cross_signal",
            market_id=MarketId("a_share"),
            research_dataset_id="a_share_research",
            signals=cross_collision,
            label_columns=("forward_return_20d",),
        )
    with pytest.raises(ValueError, match="portfolio_signal_id_column_overlap"):
        PortfolioSpec(
            portfolio_id="bad_portfolio_cross_signal",
            market_id=MarketId("a_share"),
            research_dataset_id="a_share_research",
            signals=cross_collision,
            return_source_column="forward_return_20d",
            rebalance_frequency=frequency_ids.DAILY_FREQUENCY_SLUG,
            execution_lag_steps=0,
        )


def test_portfolio_signal_contract_rejects_duplicate_and_reserved_names() -> None:
    with pytest.raises(
        ValueError,
        match="portfolio_signal_columns_must_not_contain_duplicates",
    ):
        PortfolioSpec(
            portfolio_id="bad_duplicate_signal_columns",
            market_id=MarketId("a_share"),
            research_dataset_id="a_share_research",
            signals=(
                SignalSpec(
                    signal_id="signal_a",
                    source_column="prediction_score",
                    source_kind="prediction",
                ),
                SignalSpec(
                    signal_id="signal_b",
                    source_column="prediction_score",
                    source_kind="prediction",
                ),
            ),
            return_source_column="forward_return_20d",
            rebalance_frequency=frequency_ids.DAILY_FREQUENCY_SLUG,
            execution_lag_steps=0,
        )

    with pytest.raises(ValueError, match="portfolio_signal_return_overlap"):
        PortfolioSpec(
            portfolio_id="bad_label_overlap",
            market_id=MarketId("a_share"),
            research_dataset_id="a_share_research",
            signals=(
                SignalSpec(
                    signal_id="forward_return_20d",
                    source_column="prediction_score",
                    source_kind="prediction",
                ),
            ),
            return_source_column="forward_return_20d",
            rebalance_frequency=frequency_ids.DAILY_FREQUENCY_SLUG,
            execution_lag_steps=0,
        )

    with pytest.raises(TypeError):
        PortfolioSpec(
            portfolio_id="bad_weight_overlap",
            market_id=MarketId("a_share"),
            research_dataset_id="a_share_research",
            signals=(
                SignalSpec(
                    signal_id="prediction_score",
                    source_column="prediction_score",
                    source_kind="prediction",
                ),
            ),
            return_source_column="forward_return_20d",
            rebalance_frequency=frequency_ids.DAILY_FREQUENCY_SLUG,
            execution_lag_steps=0,
            weight_column="prediction_score",
        )


def test_compute_panel_spec_uses_signal_columns_without_feature_alias() -> None:
    spec = ComputePanelSpec(signal_columns=("prediction_score",), label_column="label")

    assert spec.signal_columns == ("prediction_score",)
    assert "feature_columns" not in ComputePanelSpec.model_fields

    with pytest.raises(ValidationError, match="signal_columns"):
        ComputePanelSpec(signal_columns=(), label_column="label")


def test_weight_config_is_portfolio_contract_without_direction() -> None:
    config = WeightConfig(method="rank_weighted_quantile")

    assert config.method == "rank_weighted_quantile"
    assert not hasattr(config, "direction")

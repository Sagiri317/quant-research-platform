from __future__ import annotations

import pytest

from quant_research_platform.contracts.features import FeatureTransformSpec
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.portfolios import (
    CostConfig,
    PortfolioSpec,
    WeightConfig,
)
from quant_research_platform.contracts.signals import SignalSpec
from quant_research_platform.research.models.supervised import model_algorithm_names


def test_signal_spec_uses_source_column_and_closed_direction_names() -> None:
    signal = SignalSpec(
        signal_id="mom_20d_signal",
        source_kind="factor",
        source_column="mom_20d",
        direction="higher_is_better",
    )

    assert signal.source_column == "mom_20d"
    assert not hasattr(signal, "column")

    with pytest.raises(TypeError):
        SignalSpec(  # type: ignore[call-arg]
            signal_id="bad_old_column",
            source_kind="factor",
            column="mom_20d",
        )
    with pytest.raises(ValueError, match="unsupported_signal_direction"):
        SignalSpec(
            signal_id="bad_short_direction",
            source_kind="factor",
            source_column="mom_20d",
            direction="higher",  # type: ignore[arg-type]
        )


def test_feature_transform_spec_uses_source_and_output_columns_only() -> None:
    spec = FeatureTransformSpec(
        transform_id="mom_20d_winsorized_zscore",
        source_columns=("mom_20d",),
        transform="winsorized_zscore",
        output_columns=("mom_20d_winsorized_zscore",),
        winsorize_quantiles=(0.05, 0.95),
    )

    assert spec.source_columns == ("mom_20d",)
    assert spec.output_columns == ("mom_20d_winsorized_zscore",)
    assert not hasattr(spec, "source_factor_columns")
    assert not hasattr(spec, "output_feature_columns")
    assert not hasattr(spec, "market_id")

    with pytest.raises(ValueError, match="unsupported_feature_transform"):
        FeatureTransformSpec(
            transform_id="bad_short_zscore",
            source_columns=("mom_20d",),
            transform="zscore",
            output_columns=("mom_20d_zscore",),
        )
    with pytest.raises(ValueError, match="unsupported_feature_transform"):
        FeatureTransformSpec(
            transform_id="bad_identity_transform",
            source_columns=("mom_20d",),
            transform="identity",
            output_columns=("mom_20d_feature",),
        )
    with pytest.raises(ValueError, match="feature_transform_raw_factor_passthrough"):
        FeatureTransformSpec(
            transform_id="bad_identity_passthrough",
            source_columns=("mom_20d",),
            transform="cross_sectional_rank",
            output_columns=("mom_20d",),
        )


def test_model_algorithm_names_are_final_public_contract() -> None:
    algorithms = set(model_algorithm_names())
    assert algorithms == {
        "linear_regression",
        "ridge_regression",
        "lasso_regression",
        "elastic_net",
        "huber_regression",
        "decision_tree_regressor",
        "random_forest_regressor",
        "extra_trees_regressor",
        "gradient_boosting_regressor",
        "hist_gradient_boosting_regressor",
        "xgboost_regressor",
        "lightgbm_regressor",
        "single_factor_baseline",
        "equal_weight_rank_ensemble",
        "ic_weighted_rank_ensemble",
    }
    assert "linear" not in algorithms
    assert "xgboost" not in algorithms
    assert "lightgbm" not in algorithms


def test_portfolio_contract_uses_return_source_and_cost_config() -> None:
    spec = PortfolioSpec(
        portfolio_id="portfolio_demo",
        market_id=MarketId("a_share"),
        research_dataset_id="research_demo",
        signals=(
            SignalSpec(
                signal_id="mom_signal",
                source_kind="factor",
                source_column="mom_20d",
            ),
        ),
        return_source_column="forward_return_20d",
        rebalance_frequency="1d",
        execution_lag_steps=1,
        cost_config=CostConfig(buy_cost_bps=1.0, sell_cost_bps=1.0),
    )

    assert spec.return_source_column == "forward_return_20d"
    assert spec.rebalance_frequency == "1d"
    assert "frequency" not in PortfolioSpec.__dataclass_fields__
    assert "label_column" not in PortfolioSpec.__dataclass_fields__
    assert "cost_bps" not in WeightConfig.__dataclass_fields__

    with pytest.raises(TypeError):
        PortfolioSpec(  # type: ignore[call-arg]
            portfolio_id="bad_old_label",
            market_id=MarketId("a_share"),
            research_dataset_id="research_demo",
            signals=spec.signals,
            label_column="forward_return_20d",
        )

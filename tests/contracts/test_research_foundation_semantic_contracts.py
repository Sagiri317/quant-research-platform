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
from tests.helpers import frequency_ids


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


def test_contracts_models_do_not_export_algorithm_collection() -> None:
    import quant_research_platform.contracts.models as models

    assert not hasattr(models, "MODEL_ALGORITHMS")


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
        rebalance_frequency=frequency_ids.DAILY_FREQUENCY_SLUG,
        execution_lag_steps=1,
        cost_config=CostConfig(buy_cost_bps=1.0, sell_cost_bps=1.0),
    )

    assert spec.return_source_column == "forward_return_20d"
    assert spec.rebalance_frequency == frequency_ids.DAILY_FREQUENCY_SLUG
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

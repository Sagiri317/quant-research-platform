from __future__ import annotations

from dataclasses import replace

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


def test_feature_transform_quantiles_reject_unrepresentable_numeric_bounds() -> None:
    with pytest.raises(ValueError, match="invalid_winsorize_quantiles"):
        FeatureTransformSpec(
            transform_id="bad_huge_quantile",
            source_columns=("mom_20d",),
            transform="winsorize",
            output_columns=("mom_20d_winsorized",),
            winsorize_quantiles=(0, 10**400),
        )


@pytest.mark.parametrize(
    ("field", "wrong_kind", "error"),
    (
        ("transform", [], "unsupported_feature_transform"),
        ("transform", {}, "unsupported_feature_transform"),
        ("fillna_policy", [], "unsupported_feature_transform_fillna_policy"),
        ("fillna_policy", {}, "unsupported_feature_transform_fillna_policy"),
    ),
    ids=("transform-list", "transform-dict", "fillna-list", "fillna-dict"),
)
def test_feature_transform_rejects_unhashable_enum_values_with_owner_error(
    field: str,
    wrong_kind: object,
    error: str,
) -> None:
    spec = FeatureTransformSpec(
        transform_id="mom_20d_rank",
        source_columns=("mom_20d",),
        transform="cross_sectional_rank",
        output_columns=("mom_20d_rank",),
    )

    with pytest.raises(ValueError, match=error):
        replace(spec, **{field: wrong_kind})


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

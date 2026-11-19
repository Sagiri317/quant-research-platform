"""Declare A-share research catalogs and validate model-family and sample semantics."""

from __future__ import annotations

from dataclasses import replace

from quant_research_platform.catalog.factors import FactorCatalog
from quant_research_platform.catalog.labels import LabelCatalog
from quant_research_platform.contracts.data import (
    DataGrain,
)
from quant_research_platform.contracts.factors import FactorDefinition
from quant_research_platform.contracts.labels import LabelDefinition
from quant_research_platform.contracts.risk import (
    RISK_STYLE_COLUMNS as RISK_STYLE_COLUMNS,
)
from quant_research_platform.markets.implementations.a_share import ids as market_ids
from quant_research_platform.markets.implementations.a_share.stock_pool import (
    A_SHARE_ELIGIBILITY_FACTS,
)
from quant_research_platform.research.factors.expectations.definitions import (
    expectations_factor_definitions,
)
from quant_research_platform.research.factors.fundamental.definitions import (
    fundamental_factor_definitions,
)
from quant_research_platform.research.factors.names import factor_display_name
from quant_research_platform.research.factors.price_volume.definitions import (
    PRICE_VOLUME_LIBRARY,
    PRICE_VOLUME_PATH_FACTORS,
    price_volume_factor_definitions,
)
from quant_research_platform.research.factors.residual import RESIDUAL_FACTOR_WINDOWS
from quant_research_platform.research.labels.forward_returns import (
    forward_return_available_at_column,
    forward_return_label_column,
)

MARKET_MODEL_FACTORS = ("beta_252d", "residual_volatility_60d")
LABEL_HORIZONS = (1, 5, 10, 20, 60, 120)


def _inference_blocker(manifest):
    """Exclude independent robustness runs only from formal joint inference."""
    if manifest.get("selection_is_exploratory", False):
        return "exploratory_selection_excluded"
    if manifest.get("expectations", {}).get("role") == "robustness":
        return "expectations_robustness_excluded"
    if manifest.get("feature_recipe", {}).get("role") == "robustness":
        return "feature_recipe_robustness_excluded"
    if manifest.get("training_weighting", {}).get("role") == "robustness":
        return "training_weighting_robustness_excluded"
    return None


def factor_catalog() -> FactorCatalog:
    from quant_research_platform.research.factors.distributions import (
        definitions as distribution_definitions,
    )
    from quant_research_platform.research.factors.financing import (
        factor_definitions as financing_definitions,
    )
    from quant_research_platform.research.factors.fund_flows import (
        definitions as fund_definitions,
    )
    from quant_research_platform.research.factors.linked_markets import (
        definitions as linked_definitions,
    )
    from quant_research_platform.research.factors.macro_liquidity.definitions import (
        factor_definitions,
    )
    from quant_research_platform.research.factors.market_structure import (
        factor_definitions as market_structure_definitions,
    )
    from quant_research_platform.research.factors.style_structure import (
        factor_definitions as style_definitions,
    )

    return FactorCatalog(
        replace(item, display_name=factor_display_name(item.output_column))
        for item in (
            *financing_definitions(),
            *linked_definitions(),
            *fund_definitions(),
            *distribution_definitions(),
            *factor_definitions(),
            *market_structure_definitions(),
            *style_definitions(),
            *price_volume_factor_definitions(
                market_id=market_ids.MARKET_ID,
                required_input_data_id=market_ids.PRICE_VOLUME_DATA_ID,
                factor_columns=(*PRICE_VOLUME_LIBRARY, *PRICE_VOLUME_PATH_FACTORS),
            ),
            *(
                FactorDefinition(
                    factor_id=column,
                    market_id=market_ids.MARKET_ID,
                    input_columns=tuple(
                        dict.fromkeys(
                            (
                                "period_return",
                                "float_market_cap",
                                "delisting_date",
                                *A_SHARE_ELIGIBILITY_FACTS,
                            )
                        )
                    ),
                    output_column=column,
                    family="market_model",
                    grain=DataGrain.ASSET_TIMESTAMP,
                    required_input_data_id="market_model_factors",
                    lookback_window=312,
                    full_history=True,
                    required_windows=(252, 60),
                    description=(
                        "Prior-cap weighted eligible market; causal 252-session OLS "
                        "and 60-session population residual volatility."
                    ),
                    timing_semantics="same_session_close_available_at_next_session",
                    null_policy="preserve_nulls_after_min_periods",
                )
                for column in MARKET_MODEL_FACTORS
            ),
            *fundamental_factor_definitions(
                market_id=market_ids.MARKET_ID,
                required_input_data_id="fundamental_pit",
            ),
            *(
                FactorDefinition(
                    factor_id=column,
                    market_id=market_ids.MARKET_ID,
                    input_columns=tuple(
                        dict.fromkeys(
                            (
                                "period_return",
                                "listing_date",
                                "is_trading",
                                "industry_standard_id",
                                "industry_level1_code",
                                "total_market_cap",
                                *RISK_STYLE_COLUMNS,
                                *A_SHARE_ELIGIBILITY_FACTS,
                            )
                        )
                    ),
                    output_column=column,
                    family="risk_residual",
                    grain=DataGrain.ASSET_TIMESTAMP,
                    required_input_data_id="risk_residual_factors",
                    lookback_window=window,
                    required_windows=(window,),
                    description=(
                        (
                            "Trailing correlation of own residual and "
                            "contemporaneous-industry leave-one-out peer "
                            "residual; prior industry/style exposures, no future "
                            "residuals."
                        )
                        if "comovement" in column
                        else "Current drawdown from the peak of a complete additive "
                        "daily residual path, including its initial zero; units "
                        "are accumulated decimal residual returns, not compounded "
                        "investable wealth. Prior industry and style exposures."
                        if "drawdown" in column
                        else "Return residuals using prior industry/style exposures."
                    ),
                    timing_semantics="same_session_close_available_at_next_session",
                    null_policy="complete_residual_path"
                    if "drawdown" in column
                    else "full_window_with_minimum_fresh_fraction",
                )
                for column, window in RESIDUAL_FACTOR_WINDOWS.items()
            ),
            *expectations_factor_definitions(
                market_id=market_ids.MARKET_ID,
                required_input_data_id="analyst_forecast",
            ),
        )
    )


def label_definitions() -> tuple[LabelDefinition, ...]:
    return tuple(
        LabelDefinition(
            label_id=f"forward_return_{horizon}d",
            market_id=market_ids.MARKET_ID,
            family="forward_return",
            input_columns=("adj_open", "filling_flag", "is_trading"),
            output_column=forward_return_label_column(horizon),
            label_timing="signal_close_to_next_open_to_horizon_open",
            horizon=horizon,
            lookahead=horizon + 1,
            entry_rule="next_market_session_real_adjusted_open",
            exit_rule="entry_plus_horizon_market_sessions_real_adjusted_open",
        )
        for horizon in LABEL_HORIZONS
    ) + (
        LabelDefinition(
            label_id="next_session_intraday_return",
            market_id=market_ids.MARKET_ID,
            family="next_intraday_return",
            input_columns=("adj_open", "adj_close", "filling_flag", "is_trading"),
            output_column="next_session_intraday_return",
            label_timing="signal_close_to_next_session_open_to_close",
            horizon=1,
            lookahead=1,
            entry_rule="next_market_session_real_adjusted_open",
            exit_rule="same_entry_session_real_adjusted_close",
        ),
    )


def label_catalog() -> LabelCatalog:
    return LabelCatalog(label_definitions())


def forward_label_columns() -> tuple[str, ...]:
    return tuple(forward_return_label_column(horizon) for horizon in LABEL_HORIZONS)


def forward_label_available_columns() -> tuple[str, ...]:
    return tuple(
        forward_return_available_at_column(horizon) for horizon in LABEL_HORIZONS
    )


def validate_model_selection(models):
    """Reject ambiguous A-share champion family names."""
    from quant_research_platform.contracts.models import (
        _validate_a_share_model_family_ids,
    )

    _validate_a_share_model_family_ids(model.model_id for model in models)

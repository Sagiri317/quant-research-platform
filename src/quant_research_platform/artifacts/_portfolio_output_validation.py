"""Portfolio output validation."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from quant_research_platform.artifacts._membership_validation import (
    validate_string_membership_column,
)
from quant_research_platform.artifacts._output_common_validation import (
    require_finite_metadata_number,
    require_metadata_string,
    require_metadata_string_list,
    require_non_negative_metadata_int,
    require_optional_metadata_string,
    require_signal_metadata_list,
)
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.portfolios import (
    PORTFOLIO_CAPACITY_DIAGNOSTIC_COLUMNS,
    PORTFOLIO_CATEGORICAL_EXPOSURE_DIAGNOSTIC_COLUMNS,
    PORTFOLIO_NUMERIC_EXPOSURE_DIAGNOSTIC_COLUMNS,
    CostConfig,
    PortfolioDiagnosticsConfig,
    PortfolioSpec,
    TradabilityConfig,
    WeightConfig,
)
from quant_research_platform.contracts.validation import strict_finite_numeric_series

PORTFOLIO_ARTIFACTS = {
    "positions": "positions.parquet",
    "turnover": "turnover.parquet",
    "cost_breakdown": "cost_breakdown.parquet",
    "portfolio_returns": "portfolio_returns.parquet",
    "additive_cumulative_returns": "additive_cumulative_returns.parquet",
    "performance_summary": "performance_summary.parquet",
    "exposure_summary": "exposure_summary.parquet",
    "concentration": "concentration.parquet",
    "period_breakdown": "period_breakdown.parquet",
    "diagnostics": "diagnostics.parquet",
    "capacity_diagnostics": "capacity_diagnostics.parquet",
    "numeric_exposure_diagnostics": "numeric_exposure_diagnostics.parquet",
    "categorical_exposure_diagnostics": "categorical_exposure_diagnostics.parquet",
}
_PORTFOLIO_OUTPUT_METADATA_KEYS = frozenset(
    {
        "portfolio_id",
        "market_id",
        "research_dataset_id",
        "run_type",
        "signals",
        "return_source_column",
        "rebalance_frequency",
        "execution_lag_steps",
        "weight_config",
        "tradability_config",
        "cost_config",
        "diagnostics_config",
        "row_counts",
    }
)
_WEIGHT_CONFIG_KEYS = frozenset(
    {"method", "portfolio_mode", "top_quantile", "bottom_quantile", "risk_column"}
)
_TRADABILITY_CONFIG_KEYS = frozenset(
    {
        "eligibility_column",
        "can_enter_column",
        "can_exit_column",
        "blocked_entry_policy",
        "blocked_exit_policy",
    }
)
_COST_CONFIG_KEYS = frozenset(
    {"buy_cost_bps", "sell_cost_bps", "slippage_bps", "spread_bps", "impact_model"}
)
_DIAGNOSTICS_CONFIG_KEYS = frozenset(
    {
        "capacity_amount_column",
        "capacity_float_market_cap_column",
        "numeric_exposure_columns",
        "categorical_exposure_columns",
    }
)
_PORTFOLIO_POSITION_COLUMNS = ("timestamp", "asset", "signal", "position")
_PORTFOLIO_TURNOVER_COLUMNS = ("timestamp", "signal", "turnover")
_PORTFOLIO_COST_BREAKDOWN_COLUMNS = (
    "timestamp",
    "signal",
    "buy_turnover",
    "sell_turnover",
    "turnover",
    "buy_cost",
    "sell_cost",
    "slippage_cost",
    "spread_cost",
    "impact_cost",
    "total_cost",
    "cost_model",
)
_PORTFOLIO_RETURN_COLUMNS = (
    "timestamp",
    "signal",
    "return_source_column",
    "portfolio_mode",
    "long_return",
    "short_return",
    "gross_portfolio_return",
    "turnover",
    "cost",
    "net_portfolio_return",
    "long_count",
    "short_count",
)
_PORTFOLIO_ADDITIVE_CUMULATIVE_RETURN_COLUMNS = (
    "timestamp",
    "signal",
    "gross_portfolio_return",
    "net_portfolio_return",
    "cost",
    "additive_cumulative_gross_return",
    "additive_cumulative_net_return",
    "additive_cumulative_cost",
    "high_water_mark",
    "drawdown",
)
_PORTFOLIO_PERFORMANCE_SUMMARY_COLUMNS = (
    "signal",
    "periods",
    "final_additive_cumulative_gross_return",
    "final_additive_cumulative_net_return",
    "mean_net_portfolio_return",
    "volatility_net_portfolio_return",
    "downside_volatility",
    "sharpe_like",
    "sortino_like",
    "calmar_like",
    "best_period_return",
    "worst_period_return",
    "skew",
    "kurtosis",
    "var_95",
    "cvar_95",
    "tail_ratio",
    "positive_return_rate",
    "negative_return_rate",
    "win_rate",
    "loss_rate",
    "profit_factor",
    "payoff_ratio",
    "average_win",
    "average_loss",
    "max_consecutive_wins",
    "max_consecutive_losses",
    "max_drawdown",
    "drawdown_duration",
    "max_drawdown_start",
    "max_drawdown_valley",
    "max_drawdown_recovery",
    "mean_turnover",
    "max_turnover",
    "total_cost",
    "mean_cost",
    "cost_to_gross_return",
    "cost_to_net_return",
    "sharpe_like_reason",
    "sortino_like_reason",
    "calmar_like_reason",
    "skew_reason",
    "kurtosis_reason",
    "tail_ratio_reason",
    "profit_factor_reason",
    "payoff_ratio_reason",
    "average_win_reason",
    "average_loss_reason",
    "cost_to_gross_return_reason",
    "cost_to_net_return_reason",
)
_PORTFOLIO_EXPOSURE_SUMMARY_COLUMNS = (
    "signal",
    "mean_gross_exposure",
    "mean_net_exposure",
    "max_gross_exposure",
    "max_abs_net_exposure",
    "mean_long_count",
    "mean_short_count",
    "mean_position_count",
)
_PORTFOLIO_CONCENTRATION_COLUMNS = (
    "signal",
    "periods",
    "mean_hhi",
    "max_hhi",
    "mean_top_weight",
    "max_top_weight",
    "mean_position_count",
)
_PORTFOLIO_PERIOD_BREAKDOWN_COLUMNS = (
    "signal",
    "year",
    "month",
    "periods",
    "gross_portfolio_return",
    "net_portfolio_return",
    "cost",
    "turnover",
    "hit_rate",
)
_PORTFOLIO_DIAGNOSTIC_COLUMNS = (
    "timestamp",
    "signal",
    "method",
    "universe_count",
    "eligible_count",
    "can_enter_count",
    "can_exit_count",
    "entry_blocked_count",
    "exit_blocked_count",
    "long_count",
    "short_count",
    "fallback_reason",
)
_PORTFOLIO_RATIO_REASON_CODES = frozenset(
    {"zero_denominator", "nonfinite_input", "nonfinite_result"}
)
_PORTFOLIO_MOMENT_REASON_CODES = frozenset(
    {"insufficient_observations", "zero_variance", "nonfinite_result"}
)
_PORTFOLIO_WIN_LOSS_REASON_CODES = frozenset(
    {"no_positive_returns", "no_negative_returns"}
)


def portfolio_required_columns() -> dict[str, tuple[str, ...]]:
    return {
        "positions": _PORTFOLIO_POSITION_COLUMNS,
        "turnover": _PORTFOLIO_TURNOVER_COLUMNS,
        "cost_breakdown": _PORTFOLIO_COST_BREAKDOWN_COLUMNS,
        "portfolio_returns": _PORTFOLIO_RETURN_COLUMNS,
        "additive_cumulative_returns": _PORTFOLIO_ADDITIVE_CUMULATIVE_RETURN_COLUMNS,
        "performance_summary": _PORTFOLIO_PERFORMANCE_SUMMARY_COLUMNS,
        "exposure_summary": _PORTFOLIO_EXPOSURE_SUMMARY_COLUMNS,
        "concentration": _PORTFOLIO_CONCENTRATION_COLUMNS,
        "period_breakdown": _PORTFOLIO_PERIOD_BREAKDOWN_COLUMNS,
        "diagnostics": _PORTFOLIO_DIAGNOSTIC_COLUMNS,
        "capacity_diagnostics": PORTFOLIO_CAPACITY_DIAGNOSTIC_COLUMNS,
        "numeric_exposure_diagnostics": PORTFOLIO_NUMERIC_EXPOSURE_DIAGNOSTIC_COLUMNS,
        "categorical_exposure_diagnostics": (
            PORTFOLIO_CATEGORICAL_EXPOSURE_DIAGNOSTIC_COLUMNS
        ),
    }


def validate_portfolio_artifact_mapping_keys(
    value: object,
    *,
    expected_names: tuple[str, ...],
    error: str,
) -> Mapping[str, pd.DataFrame]:
    if (
        not isinstance(value, Mapping)
        or set(value) != set(expected_names)
        or any(not isinstance(value[name], pd.DataFrame) for name in expected_names)
    ):
        raise ValueError(error)
    return value


def validate_portfolio_row_counts(value: object) -> dict[str, int]:
    if type(value) is not dict or set(value) != set(PORTFOLIO_ARTIFACTS):
        raise ValueError("output_manifest_metadata_invalid")
    for artifact_name in PORTFOLIO_ARTIFACTS:
        require_non_negative_metadata_int(
            value,
            artifact_name,
            "output_manifest_metadata_invalid",
        )
    return value


def validate_portfolio_output_metadata_shape(
    metadata: object,
) -> dict[str, object]:
    if type(metadata) is not dict or set(metadata) != _PORTFOLIO_OUTPUT_METADATA_KEYS:
        raise ValueError("output_manifest_metadata_invalid")
    for key in (
        "portfolio_id",
        "market_id",
        "research_dataset_id",
        "run_type",
        "return_source_column",
        "rebalance_frequency",
    ):
        require_metadata_string(metadata, key, "output_manifest_metadata_invalid")
    signals = require_signal_metadata_list(
        metadata, "signals", "output_manifest_metadata_invalid"
    )
    require_non_negative_metadata_int(
        metadata, "execution_lag_steps", "output_manifest_metadata_invalid"
    )

    weight = metadata["weight_config"]
    if type(weight) is not dict or set(weight) != _WEIGHT_CONFIG_KEYS:
        raise ValueError("output_manifest_metadata_invalid")
    method = require_metadata_string(
        weight, "method", "output_manifest_metadata_invalid"
    )
    portfolio_mode = require_metadata_string(
        weight, "portfolio_mode", "output_manifest_metadata_invalid"
    )
    risk_column = require_optional_metadata_string(
        weight, "risk_column", "output_manifest_metadata_invalid"
    )
    top_quantile = require_finite_metadata_number(
        weight, "top_quantile", "output_manifest_metadata_invalid"
    )
    bottom_quantile = require_finite_metadata_number(
        weight, "bottom_quantile", "output_manifest_metadata_invalid"
    )
    try:
        weight_config = WeightConfig(
            method=method,
            portfolio_mode=portfolio_mode,
            top_quantile=top_quantile,
            bottom_quantile=bottom_quantile,
            risk_column=risk_column,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("output_manifest_metadata_invalid") from exc

    tradability = metadata["tradability_config"]
    if type(tradability) is not dict or set(tradability) != _TRADABILITY_CONFIG_KEYS:
        raise ValueError("output_manifest_metadata_invalid")
    eligibility_column = require_optional_metadata_string(
        tradability, "eligibility_column", "output_manifest_metadata_invalid"
    )
    can_enter_column = require_optional_metadata_string(
        tradability, "can_enter_column", "output_manifest_metadata_invalid"
    )
    can_exit_column = require_optional_metadata_string(
        tradability, "can_exit_column", "output_manifest_metadata_invalid"
    )
    blocked_entry_policy = require_metadata_string(
        tradability, "blocked_entry_policy", "output_manifest_metadata_invalid"
    )
    blocked_exit_policy = require_metadata_string(
        tradability, "blocked_exit_policy", "output_manifest_metadata_invalid"
    )
    try:
        tradability_config = TradabilityConfig(
            eligibility_column=eligibility_column,
            can_enter_column=can_enter_column,
            can_exit_column=can_exit_column,
            blocked_entry_policy=blocked_entry_policy,
            blocked_exit_policy=blocked_exit_policy,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("output_manifest_metadata_invalid") from exc

    cost = metadata["cost_config"]
    if type(cost) is not dict or set(cost) != _COST_CONFIG_KEYS:
        raise ValueError("output_manifest_metadata_invalid")
    buy_cost_bps = require_finite_metadata_number(
        cost, "buy_cost_bps", "output_manifest_metadata_invalid"
    )
    sell_cost_bps = require_finite_metadata_number(
        cost, "sell_cost_bps", "output_manifest_metadata_invalid"
    )
    slippage_bps = require_finite_metadata_number(
        cost, "slippage_bps", "output_manifest_metadata_invalid"
    )
    spread_bps = (
        None
        if cost["spread_bps"] is None
        else require_finite_metadata_number(
            cost, "spread_bps", "output_manifest_metadata_invalid"
        )
    )
    impact_model = require_optional_metadata_string(
        cost, "impact_model", "output_manifest_metadata_invalid"
    )
    try:
        cost_config = CostConfig(
            buy_cost_bps=buy_cost_bps,
            sell_cost_bps=sell_cost_bps,
            slippage_bps=slippage_bps,
            spread_bps=spread_bps,
            impact_model=impact_model,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("output_manifest_metadata_invalid") from exc

    diagnostics = metadata["diagnostics_config"]
    if type(diagnostics) is not dict or set(diagnostics) != _DIAGNOSTICS_CONFIG_KEYS:
        raise ValueError("output_manifest_metadata_invalid")
    capacity_amount_column = require_optional_metadata_string(
        diagnostics, "capacity_amount_column", "output_manifest_metadata_invalid"
    )
    capacity_float_market_cap_column = require_optional_metadata_string(
        diagnostics,
        "capacity_float_market_cap_column",
        "output_manifest_metadata_invalid",
    )
    numeric_columns = require_metadata_string_list(
        diagnostics,
        "numeric_exposure_columns",
        "output_manifest_metadata_invalid",
        allow_empty=True,
    )
    categorical_columns = require_metadata_string_list(
        diagnostics,
        "categorical_exposure_columns",
        "output_manifest_metadata_invalid",
        allow_empty=True,
    )
    try:
        diagnostics_config = PortfolioDiagnosticsConfig(
            capacity_amount_column=capacity_amount_column,
            capacity_float_market_cap_column=capacity_float_market_cap_column,
            numeric_exposure_columns=tuple(numeric_columns),
            categorical_exposure_columns=tuple(categorical_columns),
        )
        portfolio_spec = PortfolioSpec(
            portfolio_id=metadata["portfolio_id"],
            market_id=MarketId(metadata["market_id"]),
            research_dataset_id=metadata["research_dataset_id"],
            signals=signals,
            return_source_column=metadata["return_source_column"],
            rebalance_frequency=metadata["rebalance_frequency"],
            execution_lag_steps=metadata["execution_lag_steps"],
            weight_config=weight_config,
            tradability_config=tradability_config,
            cost_config=cost_config,
            diagnostics_config=diagnostics_config,
            run_type=metadata["run_type"],
        )
        if portfolio_spec.rebalance_frequency != metadata["rebalance_frequency"]:
            raise ValueError("portfolio_rebalance_frequency_not_canonical")
    except (TypeError, ValueError) as exc:
        raise ValueError("output_manifest_metadata_invalid") from exc
    validate_portfolio_row_counts(metadata["row_counts"])
    return metadata


def validate_portfolio_artifact_membership_columns(
    frame: pd.DataFrame,
    *,
    artifact_name: str,
    spec: PortfolioSpec,
) -> None:
    reason_pairs = {
        "sharpe_like": ("sharpe_like_reason", _PORTFOLIO_RATIO_REASON_CODES),
        "sortino_like": ("sortino_like_reason", _PORTFOLIO_RATIO_REASON_CODES),
        "calmar_like": ("calmar_like_reason", _PORTFOLIO_RATIO_REASON_CODES),
        "tail_ratio": ("tail_ratio_reason", _PORTFOLIO_RATIO_REASON_CODES),
        "cost_to_gross_return": (
            "cost_to_gross_return_reason",
            _PORTFOLIO_RATIO_REASON_CODES,
        ),
        "cost_to_net_return": (
            "cost_to_net_return_reason",
            _PORTFOLIO_RATIO_REASON_CODES,
        ),
        "skew": ("skew_reason", _PORTFOLIO_MOMENT_REASON_CODES),
        "kurtosis": ("kurtosis_reason", _PORTFOLIO_MOMENT_REASON_CODES),
        "profit_factor": (
            "profit_factor_reason",
            frozenset({"no_negative_returns"}),
        ),
        "payoff_ratio": (
            "payoff_ratio_reason",
            _PORTFOLIO_WIN_LOSS_REASON_CODES,
        ),
        "average_win": ("average_win_reason", frozenset({"no_positive_returns"})),
        "average_loss": ("average_loss_reason", frozenset({"no_negative_returns"})),
    }
    for value_column, (reason_column, allowed) in reason_pairs.items():
        if value_column in frame and reason_column in frame:
            _validate_statistic_reason_pair(
                frame[value_column],
                frame[reason_column],
                allowed=allowed,
            )
    if "signal" not in frame.columns:
        return
    signal_ids = {signal.signal_id for signal in spec.signals}
    validate_string_membership_column(
        frame,
        "signal",
        signal_ids,
        "portfolio_artifact_signal_mismatch: " + artifact_name,
    )
    if "return_source_column" in frame.columns:
        validate_string_membership_column(
            frame,
            "return_source_column",
            (spec.return_source_column,),
            "portfolio_return_source_column_mismatch",
        )


def _validate_statistic_reason_pair(
    values: pd.Series,
    reasons: pd.Series,
    *,
    allowed: frozenset[str],
) -> None:
    if pd.api.types.is_bool_dtype(values) or (
        not pd.api.types.is_numeric_dtype(values) and not values.isna().all()
    ):
        raise ValueError("invalid_portfolio_statistic_reason")
    try:
        strict_finite_numeric_series(
            values,
            error="invalid_portfolio_statistic_reason",
            allow_null=True,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid_portfolio_statistic_reason") from exc
    for value, reason in zip(values, reasons, strict=True):
        value_is_null = bool(pd.isna(value))
        reason_is_null = bool(pd.isna(reason))
        if value_is_null:
            if reason_is_null or type(reason) is not str or reason not in allowed:
                raise ValueError("invalid_portfolio_statistic_reason")
        elif not reason_is_null:
            raise ValueError("invalid_portfolio_statistic_reason")

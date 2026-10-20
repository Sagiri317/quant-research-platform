"""Multi-table semantic artifact for scientific research statistics."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from fractions import Fraction
from math import ulp
from pathlib import Path

import pandas as pd
import pyarrow as pa

from quant_research_platform.contracts.models import _PREDICTION_LOSS_REASONS
from quant_research_platform.contracts.time import (
    TimeWindow,
    time_window_timestamp_interval,
)
from quant_research_platform.contracts.validation import (
    finite_exact_float,
    strict_finite_numeric_series,
    strict_integer_series,
)
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.storage.atomic import atomic_write_directory
from quant_research_platform.storage.datasets.reading import (
    iter_declared_parquet_member_batches,
)
from quant_research_platform.storage.identity import open_filesystem_file
from quant_research_platform.storage.manifests import read_qrp_json_file, write_manifest
from quant_research_platform.storage.parquet import (
    _opened_parquet,
    parquet_table_writer,
)

_ARROW_SCHEMAS = {
    "factor_stratified_diagnostics.parquet": pa.schema(
        [
            *[
                pa.field(name, pa.string(), nullable=name == "stratum")
                for name in (
                    "signal",
                    "label",
                    "sample_scope",
                    "stratification",
                    "stratum",
                    "statistic",
                )
            ],
            *[
                pa.field(name, pa.int64(), nullable=False)
                for name in (
                    "input_date_count",
                    "observed_date_count",
                    "observation_count",
                    "observation_pairs",
                )
            ],
            *[
                pa.field(name, pa.float64())
                for name in (
                    "mean",
                    "standard_error",
                    "t_statistic",
                    "p_value",
                    "ci_lower",
                    "ci_upper",
                )
            ],
            pa.field("status", pa.string(), nullable=False),
            pa.field("reason", pa.string()),
            pa.field("hac_lag", pa.int64(), nullable=False),
            pa.field("fdr_method", pa.string(), nullable=False),
            pa.field("fdr_alpha", pa.float64(), nullable=False),
            pa.field("fdr_adjusted_p_value", pa.float64()),
            pa.field("defined_hypothesis_count", pa.int64(), nullable=False),
        ]
    ),
    "factor_diagnostics.parquet": pa.schema(
        [
            pa.field("family_id", pa.string()),
            pa.field("factor_id", pa.string()),
            pa.field("observation_count", pa.int64()),
            pa.field("status", pa.string()),
            pa.field("reason", pa.string()),
            pa.field("mean", pa.float64()),
            pa.field("standard_error", pa.float64()),
            pa.field("t_statistic", pa.float64()),
            pa.field("p_value", pa.float64()),
            pa.field("ci_lower", pa.float64()),
            pa.field("ci_upper", pa.float64()),
            pa.field("fdr_method", pa.string()),
            pa.field("fdr_alpha", pa.float64()),
            pa.field("fdr_adjusted_p_value", pa.float64()),
            pa.field("planned_hypothesis_count", pa.int64()),
            pa.field("defined_hypothesis_count", pa.int64()),
            pa.field("missing_hypothesis_count", pa.int64()),
        ]
    ),
    "prediction_diagnostics.parquet": pa.schema(
        [
            pa.field("prediction_id", pa.string()),
            pa.field("observation_count", pa.int64()),
            pa.field("status", pa.string()),
            pa.field("reason", pa.string()),
            pa.field("mean", pa.float64()),
            pa.field("standard_error", pa.float64()),
            pa.field("t_statistic", pa.float64()),
            pa.field("p_value", pa.float64()),
            pa.field("ci_lower", pa.float64()),
            pa.field("ci_upper", pa.float64()),
            pa.field("bootstrap_status", pa.string()),
            pa.field("bootstrap_reason", pa.string()),
            pa.field("bootstrap_lower", pa.float64()),
            pa.field("bootstrap_upper", pa.float64()),
            pa.field("bootstrap_block_size", pa.int64()),
            pa.field("bootstrap_repetitions", pa.int64()),
            pa.field("bootstrap_seed", pa.int64()),
        ]
    ),
    "model_trials.parquet": pa.schema(
        [
            pa.field("selection_group", pa.string()),
            pa.field("decision_id", pa.string()),
            pa.field("family_id", pa.string()),
            pa.field("config_id", pa.string()),
            pa.field("metric_id", pa.string()),
            pa.field("value", pa.float64()),
            pa.field("observation_count", pa.int64()),
            pa.field("status", pa.string()),
            pa.field("reason", pa.string()),
            pa.field("selection_status", pa.string()),
        ]
    ),
    "robustness.parquet": pa.schema(
        [
            pa.field("test_id", pa.string()),
            pa.field("status", pa.string()),
            pa.field("reason", pa.string()),
            pa.field("observation_count", pa.int64()),
        ]
    ),
    "fama_macbeth.parquet": pa.schema(
        [
            pa.field("regression_id", pa.string()),
            pa.field("coefficient_id", pa.string()),
            pa.field("period_count", pa.int64()),
            pa.field("status", pa.string()),
            pa.field("reason", pa.string()),
            pa.field("mean", pa.float64()),
            pa.field("standard_error", pa.float64()),
            pa.field("t_statistic", pa.float64()),
            pa.field("p_value", pa.float64()),
            pa.field("ci_lower", pa.float64()),
            pa.field("ci_upper", pa.float64()),
            pa.field("white_defined_period_count", pa.int64()),
            pa.field("white_undefined_period_count", pa.int64()),
        ]
    ),
    "fama_macbeth_periods.parquet": pa.schema(
        [
            pa.field("regression_id", pa.string()),
            pa.field("coefficient_id", pa.string()),
            pa.field("timestamp", pa.timestamp("ns")),
            pa.field("status", pa.string()),
            pa.field("reason", pa.string()),
            pa.field("observation_count", pa.int64()),
            pa.field("value", pa.float64()),
            pa.field("white_p_value", pa.float64()),
            pa.field("white_reason", pa.string()),
        ]
    ),
    "model_prediction_metrics.parquet": pa.schema(
        [
            pa.field("timestamp", pa.timestamp("ns")),
            pa.field("family_id", pa.string()),
            pa.field("prediction_count", pa.int64()),
            pa.field("target_count", pa.int64()),
            pa.field("mse", pa.float64()),
            pa.field("mse_reason", pa.string()),
            pa.field("baseline_mse", pa.float64()),
            pa.field("oos_r2", pa.float64()),
            pa.field("oos_r2_reason", pa.string()),
            pa.field("head_selected_count", pa.int64()),
            pa.field("head_evaluable_count", pa.int64()),
            pa.field("head_coverage", pa.float64()),
            pa.field("pool_evaluable_count", pa.int64()),
            pa.field("head_mean_raw_return", pa.float64()),
            pa.field("pool_mean_raw_return", pa.float64()),
            pa.field("head_excess_raw_return", pa.float64()),
        ]
    ),
    "model_losses.parquet": pa.schema(
        [
            pa.field("timestamp", pa.timestamp("ns")),
            pa.field("sample_role", pa.string()),
            pa.field("period_id", pa.string()),
            pa.field("family_id", pa.string()),
            pa.field("config_id", pa.string()),
            pa.field("loss", pa.float64()),
            pa.field("observation_count", pa.int64()),
            pa.field("status", pa.string()),
            pa.field("reason", pa.string()),
            pa.field("common_date", pa.int64()),
            pa.field("common_observation_count", pa.int64()),
            pa.field("asset_coverage_id", pa.string()),
        ]
    ),
    "model_comparison.parquet": pa.schema(
        [
            pa.field("sample_role", pa.string()),
            pa.field("benchmark_id", pa.string()),
            pa.field("prediction_id", pa.string()),
            pa.field("observation_count", pa.int64()),
            pa.field("status", pa.string()),
            pa.field("reason", pa.string()),
            pa.field("mean", pa.float64()),
            pa.field("standard_error", pa.float64()),
            pa.field("t_statistic", pa.float64()),
            pa.field("p_value", pa.float64()),
            pa.field("ci_lower", pa.float64()),
            pa.field("ci_upper", pa.float64()),
            pa.field("bootstrap_status", pa.string()),
            pa.field("bootstrap_reason", pa.string()),
            pa.field("bootstrap_lower", pa.float64()),
            pa.field("bootstrap_upper", pa.float64()),
            pa.field("bootstrap_block_size", pa.int64()),
            pa.field("bootstrap_repetitions", pa.int64()),
            pa.field("bootstrap_seed", pa.int64()),
        ]
    ),
    "model_group_comparison.parquet": pa.schema(
        [
            pa.field(name, pa.string())
            for name in (
                "family_id",
                "sample_role",
                "status",
                "reason",
                "fdr_method",
                "bootstrap_status",
                "bootstrap_reason",
            )
        ]
        + [
            pa.field(name, pa.int64())
            for name in (
                "observation_count",
                "calendar_count",
                "fdr_family_size",
                "hac_lag",
                "bootstrap_block_size",
                "bootstrap_repetitions",
                "bootstrap_seed",
            )
        ]
        + [
            pa.field(name, pa.float64())
            for name in (
                "global_mean",
                "quarterly_mean",
                "mean_difference",
                "standard_error",
                "t_statistic",
                "p_value",
                "ci_lower",
                "ci_upper",
                "fdr_adjusted_p_value",
                "fdr_alpha",
                "confidence",
                "bootstrap_lower",
                "bootstrap_upper",
            )
        ]
    ),
    "model_reality_checks.parquet": pa.schema(
        [
            pa.field("test_id", pa.string()),
            pa.field("sample_role", pa.string()),
            pa.field("family_id", pa.string()),
            pa.field("statistic_id", pa.string()),
            pa.field("value", pa.float64()),
            pa.field("status", pa.string()),
            pa.field("reason", pa.string()),
            pa.field("observation_count", pa.int64()),
            pa.field("block_size", pa.int64()),
            pa.field("repetitions", pa.int64()),
            pa.field("alpha", pa.float64()),
            pa.field("seed", pa.int64()),
        ]
    ),
    "portfolio_daily.parquet": pa.schema(
        [
            pa.field("candidate_id", pa.string()),
            pa.field("entry_at", pa.timestamp("ns")),
            pa.field("end_at", pa.timestamp("ns")),
            pa.field("nav_before", pa.float64()),
            pa.field("nav_post", pa.float64()),
            pa.field("cash_post", pa.float64()),
            pa.field("cash_weight_post", pa.float64()),
            pa.field("nav_end", pa.float64()),
            pa.field("gross_return", pa.float64()),
            pa.field("net_return", pa.float64()),
            pa.field("cost_drag", pa.float64()),
            pa.field("target_turnover", pa.float64()),
            pa.field("requested_turnover", pa.float64()),
            pa.field("realized_turnover", pa.float64()),
            pa.field("return_status", pa.string()),
            pa.field("return_reason", pa.string()),
            pa.field("sample_role", pa.string()),
        ]
    ),
    "portfolio_group_comparison.parquet": pa.schema(
        [
            *(
                (name, pa.string())
                for name in (
                    "portfolio_id",
                    "comparison_role",
                    "sample_role",
                    "status",
                    "reason",
                    "bootstrap_status",
                    "bootstrap_reason",
                    "fdr_method",
                    "global_terminal_wealth_reason",
                    "quarterly_terminal_wealth_reason",
                )
            ),
            *(
                (name, pa.int64())
                for name in (
                    "observation_count",
                    "hac_lag",
                    "bootstrap_block_size",
                    "bootstrap_repetitions",
                    "bootstrap_seed",
                    "fdr_family_size",
                )
            ),
            *(
                (name, pa.float64())
                for name in (
                    "global_mean",
                    "quarterly_mean",
                    "mean_difference",
                    "global_terminal_wealth",
                    "quarterly_terminal_wealth",
                    "standard_error",
                    "t_statistic",
                    "p_value",
                    "ci_lower",
                    "ci_upper",
                    "bootstrap_lower",
                    "bootstrap_upper",
                    "confidence",
                    "fdr_alpha",
                    "fdr_adjusted_p_value",
                )
            ),
        ]
    ),
    "portfolio_performance.parquet": pa.schema(
        [
            pa.field("candidate_id", pa.string()),
            pa.field("metric_id", pa.string()),
            pa.field("value", pa.float64()),
            pa.field("status", pa.string()),
            pa.field("reason", pa.string()),
            pa.field("observation_count", pa.int64()),
            pa.field("sample_role", pa.string()),
        ]
    ),
    "portfolio_capacity.parquet": pa.schema(
        [
            pa.field("candidate_id", pa.string()),
            pa.field("decision_timestamp", pa.timestamp("ns")),
            pa.field("entry_at", pa.timestamp("ns")),
            pa.field("participation", pa.float64()),
            pa.field("capacity", pa.float64()),
            pa.field("bottleneck_asset_id", pa.string()),
            pa.field("status", pa.string()),
            pa.field("reason", pa.string()),
        ]
    ),
    "portfolio_capacity_summary.parquet": pa.schema(
        [
            pa.field("candidate_id", pa.string()),
            pa.field("participation", pa.float64()),
            pa.field("period", pa.string()),
            pa.field("metric_id", pa.string()),
            pa.field("value", pa.float64()),
            pa.field("status", pa.string()),
            pa.field("reason", pa.string()),
            pa.field("observation_count", pa.int64()),
        ]
    ),
    "portfolio_attribution.parquet": pa.schema(
        [
            pa.field("candidate_id", pa.string()),
            pa.field("decision_at", pa.timestamp("ns")),
            pa.field("entry_at", pa.timestamp("ns")),
            pa.field("end_at", pa.timestamp("ns")),
            pa.field("component_id", pa.string()),
            pa.field("component_type", pa.string()),
            pa.field("value", pa.float64()),
            pa.field("status", pa.string()),
            pa.field("reason", pa.string()),
        ]
    ),
    "portfolio_robustness.parquet": pa.schema(
        [
            pa.field("candidate_id", pa.string()),
            pa.field("test_id", pa.string()),
            pa.field("value", pa.float64()),
            pa.field("status", pa.string()),
            pa.field("reason", pa.string()),
            pa.field("observation_count", pa.int64()),
            pa.field("candidate_count", pa.int64()),
            pa.field("sample_role", pa.string()),
        ]
    ),
    "portfolio_exposures.parquet": pa.schema(
        [
            pa.field("candidate_id", pa.string()),
            pa.field("decision_at", pa.timestamp("ns")),
            pa.field("entry_at", pa.timestamp("ns")),
            pa.field("factor_id", pa.string()),
            pa.field("value", pa.float64()),
        ]
    ),
    "holding_factor_returns.parquet": pa.schema(
        [
            pa.field("attribution_group", pa.string(), nullable=False),
            pa.field("decision_at", pa.timestamp("ns")),
            pa.field("entry_at", pa.timestamp("ns")),
            pa.field("end_at", pa.timestamp("ns")),
            pa.field("factor_id", pa.string()),
            pa.field("value", pa.float64()),
            pa.field("observation_count", pa.int64()),
        ]
    ),
    "crypto_maker_samples.parquet": pa.schema(
        [
            pa.field("fold_id", pa.string()),
            pa.field("side", pa.string()),
            pa.field("sample_id", pa.int64()),
            pa.field("decision_at", pa.timestamp("ns")),
            pa.field("prediction_end", pa.timestamp("ns")),
            pa.field("label_available_at", pa.timestamp("ns")),
            pa.field("price_prediction", pa.float64()),
            pa.field("observed_return", pa.float64()),
            pa.field("forecast_error", pa.float64()),
            pa.field("filled_quantity", pa.float64()),
            pa.field("quote_quantity", pa.float64()),
            pa.field("terminal_reason", pa.string()),
            pa.field("book_segment_id", pa.string()),
            pa.field("trade_observation_id", pa.string()),
            pa.field("initial_book_event_id", pa.int64()),
            pa.field("initial_bid_index", pa.int64()),
            pa.field("initial_ask_index", pa.int64()),
            pa.field("event_start", pa.int64()),
            pa.field("event_end", pa.int64()),
        ]
    ),
    "crypto_maker_events.parquet": pa.schema(
        [
            pa.field("fold_id", pa.string()),
            pa.field("event_id", pa.int64()),
            pa.field("event_at", pa.timestamp("ns")),
            pa.field("event_kind", pa.string()),
            pa.field("context_id", pa.string()),
            pa.field("side", pa.string()),
            pa.field("level_index", pa.int64()),
            pa.field("price_index", pa.int64()),
            pa.field("quantity", pa.float64()),
            pa.field("known_price_bound", pa.int64(), nullable=False),
        ]
    ),
    "crypto_maker_calibration.parquet": pa.schema(
        [
            pa.field("fold_id", pa.string()),
            pa.field("side", pa.string()),
            pa.field("price_unit", pa.string()),
            pa.field("sample_count", pa.int64()),
            pa.field("zero_fill_count", pa.int64()),
            pa.field("post_only_rejection_count", pa.int64()),
            pa.field("excluded_decisions", pa.int64()),
            pa.field("minimum_samples", pa.int64()),
            pa.field("status", pa.string()),
            pa.field("reason", pa.string()),
        ]
    ),
    "crypto_portfolio_intervals.parquet": pa.schema(
        [
            pa.field("candidate_id", pa.string()),
            pa.field("execution_mode", pa.string()),
            pa.field("episode_id", pa.string()),
            pa.field("kind", pa.string()),
            pa.field("start_at", pa.timestamp("ns")),
            pa.field("end_at", pa.timestamp("ns")),
            pa.field("start_nav", pa.float64()),
            pa.field("end_nav", pa.float64()),
            pa.field("gross_pnl", pa.float64()),
            pa.field("net_pnl", pa.float64()),
            pa.field("fees", pa.float64()),
            pa.field("funding", pa.float64()),
            pa.field("turnover", pa.float64()),
            pa.field("fill_events", pa.int64()),
        ]
    ),
    "crypto_portfolio_daily.parquet": pa.schema(
        [
            pa.field("candidate_id", pa.string()),
            pa.field("execution_mode", pa.string()),
            pa.field("episode_id", pa.string()),
            pa.field("fold_id", pa.string()),
            pa.field("start_at", pa.timestamp("ns")),
            pa.field("end_at", pa.timestamp("ns")),
            pa.field("start_nav", pa.float64()),
            pa.field("end_nav", pa.float64()),
            pa.field("net_return", pa.float64()),
            pa.field("status", pa.string()),
            pa.field("reason", pa.string()),
        ]
    ),
    "crypto_portfolio_statistics.parquet": pa.schema(
        [
            pa.field("candidate_id", pa.string()),
            pa.field("execution_mode", pa.string()),
            pa.field("episode_id", pa.string()),
            pa.field("metric_id", pa.string()),
            pa.field("value", pa.float64()),
            pa.field("observation_count", pa.int64()),
            pa.field("status", pa.string()),
            pa.field("reason", pa.string()),
        ]
    ),
    "crypto_execution_diagnostics.parquet": pa.schema(
        [
            pa.field("candidate_id", pa.string()),
            pa.field("execution_mode", pa.string()),
            pa.field("episode_id", pa.string()),
            pa.field("metric_id", pa.string()),
            pa.field("value", pa.float64()),
            pa.field("text", pa.string()),
            pa.field("status", pa.string()),
            pa.field("reason", pa.string()),
        ]
    ),
    "crypto_cost_sensitivity.parquet": pa.schema(
        [
            pa.field("candidate_id", pa.string()),
            pa.field("execution_mode", pa.string()),
            pa.field("episode_id", pa.string()),
            pa.field("fee_bps", pa.float64()),
            pa.field("additional_spread_bps", pa.float64()),
            pa.field("cost_delta", pa.float64()),
            pa.field("net_pnl", pa.float64()),
            pa.field("method", pa.string()),
        ]
    ),
    "crypto_portfolio_coverage.parquet": pa.schema(
        [
            pa.field("candidate_id", pa.string(), nullable=False),
            pa.field("execution_mode", pa.string(), nullable=False),
            pa.field("start_at", pa.timestamp("ns"), nullable=False),
            pa.field("stop_at", pa.timestamp("ns"), nullable=False),
            pa.field("kind", pa.string(), nullable=False),
            pa.field("reason", pa.string(), nullable=False),
        ]
    ),
}
_COMPARISON_MEMBERS = (
    "portfolio_benchmark_intervals.parquet",
    "portfolio_benchmark_metrics.parquet",
    "portfolio_market_exposure.parquet",
    "portfolio_regime_comparison.parquet",
)
_COMPARISON_IDENTITY = (
    "candidate_id",
    "execution_mode",
    "episode_id",
    "fold_id",
    "reference_id",
    "reference_kind",
    "sample_role",
)
for _member in _COMPARISON_MEMBERS:
    _fields = [
        pa.field(name, pa.string(), nullable=False) for name in _COMPARISON_IDENTITY
    ]
    if _member == "portfolio_benchmark_intervals.parquet":
        _fields += [
            pa.field(name, pa.timestamp("ns"), nullable=False)
            for name in ("start_at", "end_at")
        ]
        _fields += [
            pa.field(name, pa.float64())
            for name in (
                "portfolio_return",
                "benchmark_return",
                "return_difference",
                "portfolio_wealth",
                "benchmark_wealth",
                "relative_wealth",
            )
        ]
        _fields += [
            pa.field(name, pa.string(), nullable=False)
            for name in (
                "portfolio_status",
                "portfolio_reason",
                "benchmark_status",
                "benchmark_reason",
                "status",
                "reason",
            )
        ]
    else:
        if _member == "portfolio_regime_comparison.parquet":
            _fields.append(pa.field("regime", pa.string(), nullable=False))
        _fields.append(
            pa.field(
                "coefficient_id"
                if _member == "portfolio_market_exposure.parquet"
                else "metric_id",
                pa.string(),
                nullable=False,
            )
        )
        _fields.append(pa.field("value", pa.float64()))
        if _member == "portfolio_market_exposure.parquet":
            _fields += [
                pa.field(name, pa.float64())
                for name in (
                    "standard_error",
                    "t_statistic",
                    "p_value",
                    "ci_lower",
                    "ci_upper",
                )
            ]
        _fields += [
            pa.field(name, pa.int64(), nullable=False)
            for name in ("observation_count", "planned_observation_count")
        ]
        _fields += [
            pa.field(name, pa.string(), nullable=False) for name in ("status", "reason")
        ]
        if _member == "portfolio_market_exposure.parquet":
            _fields += [
                pa.field(name, pa.timestamp("ns"), nullable=False)
                for name in ("start_at", "end_at")
            ]
            _fields += [
                pa.field("hac_lag", pa.int64(), nullable=False),
                pa.field("confidence", pa.float64(), nullable=False),
                pa.field("inference_status", pa.string(), nullable=False),
                pa.field("inference_reason", pa.string(), nullable=False),
            ]
    if _member == "portfolio_market_exposure.parquet":
        _fields += [
            pa.field("fdr_method", pa.string(), nullable=False),
            pa.field("planned_hypothesis_count", pa.int64(), nullable=False),
            pa.field("defined_hypothesis_count", pa.int64(), nullable=False),
            pa.field("fdr_adjusted_p_value", pa.float64()),
        ]
    _ARROW_SCHEMAS[_member] = pa.schema(_fields)

STATISTICS_TABLE_GROUPS = {
    "factor": (
        "factor_stratified_diagnostics.parquet",
        "factor_diagnostics.parquet",
        "fama_macbeth.parquet",
        "fama_macbeth_periods.parquet",
    ),
    "research": (
        "prediction_diagnostics.parquet",
        "model_trials.parquet",
        "robustness.parquet",
        "model_losses.parquet",
        "model_prediction_metrics.parquet",
        "model_comparison.parquet",
        "model_group_comparison.parquet",
        "model_reality_checks.parquet",
    ),
    "portfolio": (
        *_COMPARISON_MEMBERS,
        "portfolio_daily.parquet",
        "portfolio_group_comparison.parquet",
        "portfolio_performance.parquet",
        "portfolio_capacity.parquet",
        "portfolio_capacity_summary.parquet",
        "portfolio_attribution.parquet",
        "portfolio_robustness.parquet",
        "portfolio_exposures.parquet",
        "holding_factor_returns.parquet",
    ),
    "crypto_maker_calibration": (
        "crypto_maker_samples.parquet",
        "crypto_maker_events.parquet",
        "crypto_maker_calibration.parquet",
    ),
    "crypto_portfolio": (
        *_COMPARISON_MEMBERS,
        "crypto_portfolio_intervals.parquet",
        "crypto_portfolio_daily.parquet",
        "crypto_portfolio_statistics.parquet",
        "crypto_execution_diagnostics.parquet",
        "crypto_cost_sensitivity.parquet",
        "crypto_portfolio_coverage.parquet",
    ),
}
_EMPTY_TABLE_MEMBERS = frozenset(
    {
        *_COMPARISON_MEMBERS,
        "crypto_maker_events.parquet",
        "crypto_maker_samples.parquet",
        "crypto_portfolio_daily.parquet",
        "crypto_portfolio_intervals.parquet",
        "fama_macbeth_periods.parquet",
        "holding_factor_returns.parquet",
        "model_comparison.parquet",
        "model_losses.parquet",
        "model_group_comparison.parquet",
        "portfolio_capacity.parquet",
        "portfolio_performance.parquet",
        "portfolio_group_comparison.parquet",
        "portfolio_exposures.parquet",
    }
)
for _member in (
    "factor_diagnostics.parquet",
    "prediction_diagnostics.parquet",
    "fama_macbeth.parquet",
    "model_comparison.parquet",
    "model_reality_checks.parquet",
    "robustness.parquet",
):
    _ARROW_SCHEMAS[_member] = _ARROW_SCHEMAS[_member].append(
        pa.field("input_date_count", pa.int64(), nullable=False)
    )

_TABLE_COLUMNS = {
    member: tuple(schema.names) for member, schema in _ARROW_SCHEMAS.items()
}
_STRING_COLUMNS = {
    member: tuple(field.name for field in schema if pa.types.is_string(field.type))
    for member, schema in _ARROW_SCHEMAS.items()
}
_COUNT_COLUMNS = {
    member: tuple(field.name for field in schema if pa.types.is_integer(field.type))
    for member, schema in _ARROW_SCHEMAS.items()
}
_TIMESTAMP_COLUMNS = {
    member: tuple(field.name for field in schema if pa.types.is_timestamp(field.type))
    for member, schema in _ARROW_SCHEMAS.items()
}
STATISTICS_TABLE_COLUMNS = _TABLE_COLUMNS


@dataclass(frozen=True, slots=True)
class StatisticsArtifact:
    metadata: dict
    root: Path
    table_group: str
    _members: tuple[str, ...]

    def _batches(self, member, *, filters=None):
        if member not in self._members:
            raise ValueError("statistics_table_member_invalid")
        with closing_iterator(
            iter_declared_parquet_member_batches(
                self.root / member, parquet_filters=filters
            )
        ) as frames:
            for frame in frames:
                yield _validated_frame(frame, member=member, metadata=self.metadata)

    @contextmanager
    def table(self, member: str):
        if member not in self._members:
            raise ValueError("statistics_table_member_invalid")
        with _opened_parquet(self.root / member) as (parquet, _handle):
            if parquet.schema_arrow != _ARROW_SCHEMAS[member]:
                raise ValueError("statistics_table_schema_invalid")
            frame = parquet.read().to_pandas(integer_object_nulls=True)
        yield _validated_frame(frame, member=member, metadata=self.metadata)


def write_statistics_artifact(
    output_path: Path,
    *,
    metadata: dict,
    tables: Mapping[str, pd.DataFrame] | Callable[..., None],
    table_group: str = "research",
    validate_tables=None,
) -> Path:
    if table_group not in STATISTICS_TABLE_GROUPS:
        raise ValueError("statistics_tables_invalid")
    _validate_metadata(metadata)
    known_members = tuple(STATISTICS_TABLE_GROUPS[table_group])
    expected = {member.removesuffix(".parquet") for member in known_members}
    calibration = table_group == "crypto_maker_calibration"
    if callable(tables) and not calibration:
        raise ValueError("statistics_tables_invalid")
    if calibration and not callable(validate_tables):
        raise ValueError("statistics_business_validator_required")
    if not callable(tables) and (
        not isinstance(tables, Mapping)
        or not tables
        or set(tables) - expected
        or (calibration and set(tables) != expected)
    ):
        raise ValueError("statistics_tables_invalid")
    members = tuple(
        member
        for member in known_members
        if callable(tables) or member.removesuffix(".parquet") in tables
    )
    expected = {member.removesuffix(".parquet") for member in members}

    def write(staging):
        with closing_iterator(ExitStack()) as output:
            writers = {}

            def append(table_name, frame):
                if table_name not in expected:
                    raise ValueError("statistics_tables_invalid")
                member = f"{table_name}.parquet"
                if member not in writers:
                    handle = output.enter_context(
                        closing_iterator(open_filesystem_file(staging / member, "xb"))
                    )
                    writers[member] = output.enter_context(
                        parquet_table_writer(
                            handle,
                            compression="zstd",
                            schema=_ARROW_SCHEMAS[member],
                            statistics=tuple(
                                column
                                for column in (
                                    "fold_id",
                                    "side",
                                    "event_id",
                                    "sample_id",
                                )
                                if column in _TABLE_COLUMNS[member]
                            ),
                        )
                    )
                validated = _validated_frame(frame, member=member, metadata=metadata)
                writers[member](
                    pa.Table.from_pandas(
                        validated, schema=_ARROW_SCHEMAS[member], preserve_index=False
                    )
                )

            if callable(tables):
                tables(on_table=append)
            else:
                for name, frame in tables.items():
                    append(name, frame)
            if set(writers) != set(members):
                raise ValueError("statistics_tables_invalid")
        if callable(validate_tables):
            validate_tables(
                StatisticsArtifact(dict(metadata), staging, table_group, members),
                metadata,
            )
        write_manifest(
            staging / "manifest.json",
            {
                "schema_id": "qrp-statistics",
                "table_group": table_group,
                "metadata": dict(metadata),
                "members": list(members),
            },
        )

    atomic_write_directory(Path(output_path), write)
    return Path(output_path)


def read_statistics_artifact(path: Path, *, validate_tables=None) -> StatisticsArtifact:
    root = Path(path)
    _, manifest = read_qrp_json_file(root / "manifest.json")
    group = manifest.get("table_group")
    if (
        manifest.get("schema_id") != "qrp-statistics"
        or group not in STATISTICS_TABLE_GROUPS
    ):
        raise ValueError("statistics_artifact_contract_invalid")
    known = STATISTICS_TABLE_GROUPS[group]
    declared = manifest.get("members")
    if (
        not isinstance(declared, list)
        or not declared
        or any(
            not isinstance(member, str) or member not in known for member in declared
        )
        or len(set(declared)) != len(declared)
        or declared != [member for member in known if member in declared]
        or (group == "crypto_maker_calibration" and declared != list(known))
    ):
        raise ValueError("statistics_artifact_members_invalid")
    members = tuple(declared)
    metadata = manifest.get("metadata")
    _validate_metadata(metadata)
    artifact = StatisticsArtifact(metadata, root, group, members)
    if group == "crypto_maker_calibration":
        if not callable(validate_tables):
            raise ValueError("statistics_business_validator_required")
        validate_tables(artifact, metadata)
    return artifact


def _validate_metadata(metadata):
    if (
        not isinstance(metadata, dict)
        or type(metadata.get("sample_role")) is not str
        or not metadata["sample_role"].strip()
    ):
        raise ValueError("statistics_sample_role_invalid")
    window = metadata.get("window")
    if not isinstance(window, dict) or set(window) != {"start", "end"}:
        raise ValueError("statistics_window_invalid")
    TimeWindow(window["start"], window["end"])
    candidates = metadata.get("candidate_ids", [])
    if (
        not isinstance(candidates, list)
        or any(not isinstance(x, str) or not x for x in candidates)
        or len(set(candidates)) != len(candidates)
    ):
        raise ValueError("statistics_candidate_ids_invalid")


def _validate_table_binding(frame, *, member, metadata):
    if (
        "sample_role" in frame
        and not frame.sample_role.eq(metadata["sample_role"]).all()
    ):
        raise ValueError("statistics_sample_role_mismatch")
    if member in STATISTICS_TABLE_GROUPS["crypto_maker_calibration"]:
        return  # Fold maturity and event references are checked by its producer.
    if (
        "execution_mode" in frame
        and not frame.execution_mode.eq(metadata.get("execution_mode")).all()
    ):
        raise ValueError("statistics_execution_mode_mismatch")
    if "candidate_id" in frame:
        valid = frame.candidate_id.isin(metadata.get("candidate_ids", []))
        if member == "portfolio_robustness.parquet":
            valid |= frame.candidate_id.eq("joint") & frame.test_id.isin(
                ("cscv_pbo", "validation_only")
            )
        if not valid.all():
            raise ValueError("statistics_candidate_id_mismatch")
    if (
        "portfolio_id" in frame
        and not frame.portfolio_id.isin(
            metadata.get("comparison_portfolio_ids", [])
        ).all()
    ):
        raise ValueError("statistics_candidate_id_mismatch")
    start, stop, inclusive = time_window_timestamp_interval(
        TimeWindow(**metadata["window"])
    )
    for column in _TIMESTAMP_COLUMNS[member]:
        valid = frame[column].ge(start)
        endpoint = member == "crypto_portfolio_coverage.parquet" and column == "stop_at"
        upper = (
            pd.Timestamp(stop) + pd.Timedelta(1, "ns")
            if inclusive and endpoint
            else stop
        )
        valid &= (
            frame[column].le(upper)
            if inclusive or endpoint
            else frame[column].lt(upper)
        )
        if not valid.all():
            raise ValueError("statistics_table_window_mismatch")


def _validated_frame(frame: pd.DataFrame, *, member: str, metadata) -> pd.DataFrame:
    if (
        type(frame) is not pd.DataFrame
        or tuple(frame.columns) != _TABLE_COLUMNS[member]
    ):
        raise ValueError(f"statistics_table_schema_invalid:{member}")
    result = frame.copy()
    for field in _ARROW_SCHEMAS[member]:
        if not field.nullable and result[field.name].isna().any():
            raise ValueError("statistics_table_null_invalid")
    for column in _STRING_COLUMNS[member]:
        if member == "factor_stratified_diagnostics.parquet" and column == "stratum":
            if any(type(value) is not str for value in result[column].dropna()):
                raise ValueError("statistics_stratum_invalid")
            result[column] = result[column].astype("string")
            continue
        values = result[column]
        if values.isna().any():
            result[column] = values.astype("string").fillna("")
        result[column] = result[column].astype(str)
    for column in _COUNT_COLUMNS[member]:
        values = strict_integer_series(
            result[column],
            error="statistics_table_count_invalid",
            allow_null=(
                member in {"prediction_diagnostics.parquet", "model_comparison.parquet"}
                and column
                in {"bootstrap_block_size", "bootstrap_repetitions", "bootstrap_seed"}
                or member == "crypto_maker_events.parquet"
                and column == "price_index"
                and result[column].isna().any()
            ),
        )
        if values.lt(0).any():
            raise ValueError("statistics_table_count_invalid")
        result[column] = values
    if "input_date_count" in result:
        valid_count = result[
            "period_count" if member == "fama_macbeth.parquet" else "observation_count"
        ]
        if (result["input_date_count"] < valid_count).any():
            raise ValueError("statistics_input_date_count_invalid")
    for column in _TIMESTAMP_COLUMNS.get(member, ()):
        values = pd.to_datetime(result[column], errors="raise")
        if values.isna().any():
            raise ValueError("statistics_table_timestamp_invalid")
        result[column] = values.astype("datetime64[ns]")
    numeric_columns = tuple(
        column
        for column in _TABLE_COLUMNS[member]
        if column not in _STRING_COLUMNS[member]
        and column not in _COUNT_COLUMNS[member]
        and (column not in _TIMESTAMP_COLUMNS.get(member, ()))
    )
    for column in numeric_columns:
        result[column] = strict_finite_numeric_series(
            result[column], error="statistics_table_numeric_invalid", allow_null=True
        ).astype("float64")
    if result.empty and member not in _EMPTY_TABLE_MEMBERS:
        raise ValueError("statistics_table_empty")
    if member in (
        *STATISTICS_TABLE_GROUPS["research"],
        *STATISTICS_TABLE_GROUPS["factor"],
    ) and member not in {
        "model_group_comparison.parquet",
        "model_prediction_metrics.parquet",
    }:
        if not result.status.isin(("DEFINED", "UNDEFINED")).all():
            raise ValueError("statistics_table_status_invalid")
        defined = result.status.eq("DEFINED")
        measurements = [
            column
            for column in (
                "value",
                "loss",
                "mean",
                "standard_error",
                "t_statistic",
                "p_value",
                "ci_lower",
                "ci_upper",
                "bootstrap_lower",
                "bootstrap_upper",
            )
            if column in result
            and not (
                member in {"prediction_diagnostics.parquet", "model_comparison.parquet"}
                and column in {"bootstrap_lower", "bootstrap_upper"}
            )
        ]
        if (
            result.loc[defined, measurements].isna().any().any()
            or result.loc[
                ~defined,
                [
                    column
                    for column in measurements
                    if column != "mean"
                    or member
                    not in {
                        "prediction_diagnostics.parquet",
                        "model_comparison.parquet",
                        "factor_stratified_diagnostics.parquet",
                        "factor_diagnostics.parquet",
                        "fama_macbeth.parquet",
                    }
                ],
            ]
            .notna()
            .any()
            .any()
            or result.loc[defined, "reason"].ne("").any()
            or result.loc[~defined, "reason"].str.strip().eq("").any()
        ):
            raise ValueError("statistics_table_status_value_invalid")
        if member in {"prediction_diagnostics.parquet", "model_comparison.parquet"}:
            bootstrap_defined = result.bootstrap_status.eq("DEFINED")
            if (
                not result.bootstrap_status.isin(("DEFINED", "UNDEFINED")).all()
                or result.loc[bootstrap_defined, ["bootstrap_lower", "bootstrap_upper"]]
                .isna()
                .any()
                .any()
                or result.loc[
                    ~bootstrap_defined, ["bootstrap_lower", "bootstrap_upper"]
                ]
                .notna()
                .any()
                .any()
                or result.loc[bootstrap_defined, "bootstrap_reason"].ne("").any()
                or result.loc[~bootstrap_defined, "bootstrap_reason"]
                .str.strip()
                .eq("")
                .any()
                or result.loc[
                    bootstrap_defined,
                    ["bootstrap_block_size", "bootstrap_repetitions", "bootstrap_seed"],
                ]
                .isna()
                .any()
                .any()
            ):
                raise ValueError("statistics_table_bootstrap_status_value_invalid")
        if member in {
            "prediction_diagnostics.parquet",
            "model_comparison.parquet",
            "factor_diagnostics.parquet",
            "fama_macbeth.parquet",
        }:
            count = result[
                "period_count"
                if member == "fama_macbeth.parquet"
                else "observation_count"
            ]
            if result["mean"].notna().ne(count.gt(0)).any():
                raise ValueError("statistics_table_descriptive_mean_count_invalid")
        for column in ("p_value", "fdr_adjusted_p_value", "white_p_value"):
            if column in result and (not result[column].dropna().between(0, 1).all()):
                raise ValueError("statistics_table_probability_invalid")
        if "standard_error" in result and (defined & result.standard_error.le(0)).any():
            raise ValueError("statistics_table_standard_error_invalid")
        for lower, upper in (
            ("ci_lower", "ci_upper"),
            ("bootstrap_lower", "bootstrap_upper"),
        ):
            if lower in result and result[lower].gt(result[upper]).any():
                raise ValueError("statistics_table_interval_invalid")
        if member == "factor_diagnostics.parquet":
            applicable = result.fdr_method.isin(("BH", "BY"))
            if (
                not (applicable | result.fdr_method.eq("NOT_APPLICABLE")).all()
                or result.defined_hypothesis_count.gt(
                    result.planned_hypothesis_count
                ).any()
                or (
                    not result.missing_hypothesis_count.eq(
                        result.planned_hypothesis_count
                        - result.defined_hypothesis_count
                    ).all()
                )
                or result.loc[applicable, "fdr_alpha"].isna().any()
                or (
                    not result.loc[applicable, "fdr_alpha"]
                    .between(0, 1, inclusive="neither")
                    .all()
                )
                or result.loc[applicable & defined, "fdr_adjusted_p_value"].isna().any()
                or result.loc[~applicable | ~defined, "fdr_adjusted_p_value"]
                .notna()
                .any()
                or result.loc[~applicable, "fdr_alpha"].notna().any()
            ):
                raise ValueError("statistics_table_fdr_invalid")
        if member == "fama_macbeth_periods.parquet":
            white_defined = result.white_p_value.notna()
            if (
                (white_defined & ~defined).any()
                or not result.loc[white_defined, "white_p_value"].between(0, 1).all()
                or result.loc[white_defined, "white_reason"].ne("").any()
                or result.loc[~white_defined, "white_reason"].str.strip().eq("").any()
            ):
                raise ValueError("statistics_table_white_invalid")
        if member == "model_reality_checks.parquet":
            indicator = result.statistic_id.isin(("superior", "included"))
            probability = result.statistic_id.isin(
                ("lower_p_value", "consistent_p_value", "upper_p_value", "p_value")
            )
            if (
                not (indicator | probability).all()
                or not result.alpha.between(0, 1, inclusive="neither").all()
                or (not result.loc[defined & indicator, "value"].isin((0, 1)).all())
                or (not result.loc[defined & probability, "value"].between(0, 1).all())
            ):
                raise ValueError("statistics_table_reality_check_invalid")
    if member in {
        "model_losses.parquet",
        "model_comparison.parquet",
        "model_reality_checks.parquet",
    }:
        if (
            result["sample_role"].isna().any()
            or result["sample_role"].str.strip().eq("").any()
            or result["sample_role"].nunique() > 1
        ):
            raise ValueError("statistics_sample_role_invalid")
        keys = {
            "model_losses.parquet": ("timestamp", "family_id"),
            "model_comparison.parquet": ("prediction_id",),
            "model_reality_checks.parquet": ("test_id", "family_id", "statistic_id"),
        }[member]
        if result.duplicated(list(keys)).any():
            raise ValueError("statistics_table_duplicate_keys")
    if member in {
        "model_group_comparison.parquet",
        "portfolio_group_comparison.parquet",
    }:
        key = (
            "family_id"
            if member == "model_group_comparison.parquet"
            else "portfolio_id"
        )
        defined = result.status.eq("DEFINED")
        bootstrap_defined = result.bootstrap_status.eq("DEFINED")
        fdr_defined = defined & result.fdr_method.isin(("BH", "BY"))
        measurements = (
            "standard_error",
            "t_statistic",
            "p_value",
            "ci_lower",
            "ci_upper",
        )
        bootstrap = ("bootstrap_lower", "bootstrap_upper")
        if (
            result.duplicated([key]).any()
            or not result.status.isin(("DEFINED", "UNDEFINED")).all()
            or (not result.bootstrap_status.isin(("DEFINED", "UNDEFINED")).all())
            or result.sample_role.isna().any()
            or result.sample_role.str.strip().eq("").any()
            or (result.sample_role.nunique() > 1)
            or (
                bootstrap_defined
                & result.observation_count.le(result.bootstrap_block_size)
            ).any()
            or result.loc[
                bootstrap_defined, ["global_mean", "quarterly_mean", "mean_difference"]
            ]
            .isna()
            .any()
            .any()
            or result.loc[defined, list(measurements)].isna().any().any()
            or result.loc[defined, ["global_mean", "quarterly_mean", "mean_difference"]]
            .isna()
            .any()
            .any()
            or (defined & result.observation_count.lt(2)).any()
            or result.loc[~defined, list(measurements)].notna().any().any()
            or result.loc[defined, "reason"].ne("").any()
            or result.loc[~defined, "reason"].str.strip().eq("").any()
            or result.loc[bootstrap_defined, list(bootstrap)].isna().any().any()
            or result.loc[~bootstrap_defined, list(bootstrap)].notna().any().any()
            or result.loc[bootstrap_defined, "bootstrap_reason"].ne("").any()
            or result.loc[~bootstrap_defined, "bootstrap_reason"]
            .str.strip()
            .eq("")
            .any()
            or (not result.fdr_method.isin(("BH", "BY", "NOT_APPLICABLE")).all())
            or result.loc[fdr_defined, "fdr_adjusted_p_value"].isna().any()
            or result.loc[~fdr_defined, "fdr_adjusted_p_value"].notna().any()
            or (not result.fdr_alpha.between(0, 1, inclusive="neither").all())
            or (not result.confidence.between(0, 1, inclusive="neither").all())
            or (not result.fdr_family_size.eq(len(result)).all())
            or (defined & result.standard_error.le(0)).any()
            or result.ci_lower.gt(result.ci_upper).any()
            or result.bootstrap_lower.gt(result.bootstrap_upper).any()
            or (not result.p_value.dropna().between(0, 1).all())
            or (not result.fdr_adjusted_p_value.dropna().between(0, 1).all())
        ):
            raise ValueError("statistics_group_comparison_invalid")
        if member == "model_group_comparison.parquet" and (
            result.observation_count.gt(result.calendar_count).any()
            or (
                (defined | bootstrap_defined)
                & result.observation_count.ne(result.calendar_count)
            ).any()
        ):
            raise ValueError("statistics_group_comparison_count_invalid")
        if (
            member == "portfolio_group_comparison.parquet"
            and not result.empty
            and (
                not result.comparison_role.isin(("primary", "secondary")).all()
                or result.comparison_role.eq("primary").sum() != 1
            )
        ):
            raise ValueError("statistics_group_comparison_primary_invalid")
    if member == "portfolio_group_comparison.parquet":
        for group in ("global", "quarterly"):
            column = f"{group}_terminal_wealth"
            missing = result[column].isna()
            if (
                result.loc[~missing, f"{column}_reason"].ne("").any()
                or not result.loc[missing, f"{column}_reason"]
                .isin(("nonfinite_result", "nonpositive_start_nav"))
                .all()
            ):
                raise ValueError("statistics_terminal_wealth_invalid")
    if member == "model_prediction_metrics.parquet":
        incomparable_reasons = _PREDICTION_LOSS_REASONS
        mse_reasons = ("no_valid_targets", "nonfinite_result", *incomparable_reasons)
        no_targets = result.target_count.eq(0)
        incomparable = result.mse_reason.isin(
            incomparable_reasons
        ) | result.oos_r2_reason.isin(incomparable_reasons)
        if (
            result.duplicated(["timestamp", "family_id"]).any()
            or result.target_count.gt(result.prediction_count).any()
            or result.head_selected_count.gt(result.prediction_count).any()
            or result.head_evaluable_count.gt(result.head_selected_count).any()
            or not result.head_coverage.dropna().between(0, 1).all()
            or result.mse.dropna().lt(0).any()
            or result.baseline_mse.dropna().lt(0).any()
            or result.oos_r2.dropna().gt(1).any()
            or result.loc[no_targets, ["mse", "baseline_mse", "oos_r2"]]
            .notna()
            .any()
            .any()
            or result.loc[result.mse.notna(), "mse_reason"].ne("").any()
            or not result.loc[result.mse.isna(), "mse_reason"].isin(mse_reasons).all()
            or (result.mse_reason.eq("no_valid_targets") & ~no_targets).any()
            or result.loc[result.oos_r2.notna(), "oos_r2_reason"].ne("").any()
            or not result.loc[result.oos_r2.isna(), "oos_r2_reason"]
            .isin((*mse_reasons, "zero_baseline_error"))
            .all()
            or (result.oos_r2_reason.eq("no_valid_targets") & ~no_targets).any()
            or (
                result.oos_r2_reason.eq("zero_baseline_error")
                & result.baseline_mse.ne(0)
            ).any()
            or (incomparable & result.oos_r2_reason.ne(result.mse_reason)).any()
        ):
            raise ValueError("statistics_prediction_metrics_invalid")
    if member == "model_losses.parquet":
        if (
            not result.asset_coverage_id.str.fullmatch("[0-9a-f]{64}").all()
            or not result["loss"].dropna().between(-1, 1).all()
            or (not result["common_date"].isin((0, 1)).all())
            or (result["common_observation_count"] > result["observation_count"]).any()
            or result.groupby("timestamp")["common_observation_count"]
            .nunique()
            .gt(1)
            .any()
        ):
            raise ValueError("statistics_model_loss_invalid")
    if member in _COMPARISON_MEMBERS:
        if (
            not result.execution_mode.isin(("continuous", "independent_segments")).all()
            or not result.reference_kind.isin(("benchmark", "control")).all()
            or result[["candidate_id", "episode_id", "reference_id", "sample_role"]]
            .eq("")
            .any()
            .any()
        ):
            raise ValueError("statistics_comparison_identity_invalid")
        continuous = result.execution_mode.eq("continuous")
        if (
            continuous & ~result.episode_id.eq("continuous")
            | ~continuous & ~result.episode_id.str.fullmatch(r"[0-9]+_[0-9]+")
        ).any():
            raise ValueError("statistics_episode_id_invalid")
        suffix = {
            "portfolio_benchmark_intervals.parquet": ("start_at", "end_at"),
            "portfolio_benchmark_metrics.parquet": ("metric_id",),
            "portfolio_market_exposure.parquet": ("coefficient_id",),
            "portfolio_regime_comparison.parquet": ("regime", "metric_id"),
        }[member]
        if result.duplicated([*_COMPARISON_IDENTITY, *suffix]).any():
            raise ValueError("statistics_table_duplicate_keys")
        status_values = [
            (
                "status",
                "reason",
                "return_difference"
                if member == "portfolio_benchmark_intervals.parquet"
                else "value",
            )
        ]
        if member == "portfolio_benchmark_intervals.parquet":
            status_values += [
                ("portfolio_status", "portfolio_reason", "portfolio_return"),
                ("benchmark_status", "benchmark_reason", "benchmark_return"),
            ]
        for status, reason, value in status_values:
            defined = result[status].eq("DEFINED")
            if (
                not result[status]
                .isin(("DEFINED", "UNDEFINED", "NOT_APPLICABLE"))
                .all()
                or not defined.eq(result[value].notna()).all()
                or result.loc[defined, reason].ne("").any()
                or result.loc[~defined, reason].eq("").any()
            ):
                raise ValueError("statistics_table_status_value_invalid")
        if (
            "planned_observation_count" in result
            and result.observation_count.gt(result.planned_observation_count).any()
        ):
            raise ValueError("statistics_comparison_count_invalid")
        if "start_at" in result and not result.start_at.lt(result.end_at).all():
            raise ValueError("statistics_comparison_interval_invalid")
        if (
            member == "portfolio_market_exposure.parquet"
            and not result.confidence.between(0, 1, inclusive="neither").all()
        ):
            raise ValueError("statistics_comparison_confidence_invalid")
    if member == "portfolio_market_exposure.parquet":
        hypothesis = result.coefficient_id.isin(
            ("alpha", "market_return_intercept", "beta")
        )
        inferred = result.inference_status.eq("DEFINED")
        if (
            not result.inference_status.isin(
                ("DEFINED", "UNDEFINED", "NOT_APPLICABLE")
            ).all()
            or not result.inference_status.eq("NOT_APPLICABLE").eq(~hypothesis).all()
            or (inferred & ~result.status.eq("DEFINED")).any()
            or result.loc[inferred, "inference_reason"].ne("").any()
            or result.loc[~inferred, "inference_reason"].eq("").any()
            or result.loc[~inferred, ["t_statistic", "p_value", "fdr_adjusted_p_value"]]
            .notna()
            .any()
            .any()
            or not result.fdr_method.eq("BY").all()
            or result.defined_hypothesis_count.gt(result.planned_hypothesis_count).any()
            or not result.coefficient_id.isin(
                ("alpha", "market_return_intercept", "beta", "correlation", "r_squared")
            ).all()
            or not result.p_value.dropna().between(0, 1).all()
            or not result.fdr_adjusted_p_value.dropna().between(0, 1).all()
            or result.loc[
                inferred,
                [
                    "standard_error",
                    "t_statistic",
                    "p_value",
                    "ci_lower",
                    "ci_upper",
                    "fdr_adjusted_p_value",
                ],
            ]
            .isna()
            .any()
            .any()
            or result.loc[
                ~hypothesis,
                [
                    "standard_error",
                    "t_statistic",
                    "p_value",
                    "ci_lower",
                    "ci_upper",
                    "fdr_adjusted_p_value",
                ],
            ]
            .notna()
            .any()
            .any()
        ):
            raise ValueError("statistics_exposure_inference_invalid")
    if member in {
        "crypto_portfolio_daily.parquet",
        "crypto_portfolio_statistics.parquet",
        "crypto_execution_diagnostics.parquet",
    }:
        if not result.status.isin(("DEFINED", "UNDEFINED")).all():
            raise ValueError("statistics_table_status_invalid")
        defined = result.status.eq("DEFINED")
        value_column = (
            "net_return" if member == "crypto_portfolio_daily.parquet" else "value"
        )
        has_value = result[value_column].notna()
        if member == "crypto_execution_diagnostics.parquet":
            has_text = result.text.str.strip().ne("")
            invalid_value = defined & ~(has_value ^ has_text) | ~defined & (
                has_value | has_text
            )
        else:
            invalid_value = defined.ne(has_value)
        if (
            invalid_value.any()
            or result.loc[defined, "reason"].ne("").any()
            or result.loc[~defined, "reason"].str.strip().eq("").any()
        ):
            raise ValueError("statistics_table_status_value_invalid")
        if member == "crypto_portfolio_daily.parquet" and (
            result.loc[defined, ["start_nav", "end_nav"]].isna().any().any()
            or result.loc[defined, "start_nav"].le(0).any()
        ):
            raise ValueError("statistics_table_status_value_invalid")
    if (
        member in STATISTICS_TABLE_GROUPS["crypto_portfolio"]
        and member not in _COMPARISON_MEMBERS
    ):
        if not result.execution_mode.isin(("continuous", "independent_segments")).all():
            raise ValueError("statistics_execution_mode_invalid")
        if result.candidate_id.str.strip().eq("").any():
            raise ValueError("statistics_candidate_id_invalid")
        if member == "crypto_portfolio_coverage.parquet":
            keys = ("candidate_id", "execution_mode", "start_at", "stop_at", "kind")
            covered = result.kind.eq("covered")
            if (
                not result.kind.isin(("covered", "missing")).all()
                or not result.start_at.lt(result.stop_at).all()
                or result.loc[covered, "reason"].ne("").any()
                or result.loc[~covered, "reason"].str.strip().eq("").any()
            ):
                raise ValueError("statistics_coverage_invalid")
        else:
            aggregate = result.episode_id.eq("all")
            continuous = result.execution_mode.eq("continuous")
            valid_episode = result.episode_id.str.fullmatch(r"[0-9]+_[0-9]+")
            if (
                ~aggregate & continuous & ~result.episode_id.eq("continuous")
                | ~aggregate & ~continuous & ~valid_episode
            ).any():
                raise ValueError("statistics_episode_id_invalid")
            aggregate_metrics = {
                "crypto_portfolio_statistics.parquet": (
                    "covered_seconds",
                    "missing_seconds",
                    "coverage_fraction",
                    "episode_count",
                    "truncated_episode_count",
                    "fees",
                    "funding",
                    "turnover",
                    "fill_events",
                    "fill_quantity",
                    "known_prefix_fees",
                    "known_prefix_turnover",
                    "known_prefix_fill_events",
                    "known_prefix_fill_quantity",
                ),
                "crypto_execution_diagnostics.parquet": (
                    "funding_required_decisions",
                    "funding_covered_decisions",
                    "funding_decision_coverage",
                    "funding_selection_status",
                ),
            }
            if aggregate.any() and (
                member not in aggregate_metrics
                or not result.loc[aggregate, "metric_id"]
                .isin(aggregate_metrics[member])
                .all()
            ):
                raise ValueError("statistics_cross_episode_metric_invalid")
            suffix = {
                "crypto_portfolio_intervals.parquet": ("kind", "start_at", "end_at"),
                "crypto_portfolio_daily.parquet": ("fold_id", "start_at", "end_at"),
                "crypto_portfolio_statistics.parquet": ("metric_id",),
                "crypto_execution_diagnostics.parquet": ("metric_id",),
                "crypto_cost_sensitivity.parquet": ("fee_bps", "additional_spread_bps"),
            }[member]
            keys = ("candidate_id", "execution_mode", "episode_id", *suffix)
        if result.duplicated(list(keys)).any():
            raise ValueError("statistics_table_duplicate_keys")
    if member in STATISTICS_TABLE_GROUPS["portfolio"] and member not in (
        "portfolio_group_comparison.parquet",
        *_COMPARISON_MEMBERS,
    ):
        keys = {
            "portfolio_daily.parquet": ("candidate_id", "entry_at"),
            "portfolio_performance.parquet": ("candidate_id", "metric_id"),
            "portfolio_capacity.parquet": ("candidate_id", "entry_at", "participation"),
            "portfolio_capacity_summary.parquet": (
                "candidate_id",
                "participation",
                "period",
                "metric_id",
            ),
            "portfolio_attribution.parquet": (
                "candidate_id",
                "entry_at",
                "component_id",
            ),
            "portfolio_robustness.parquet": ("candidate_id", "test_id"),
            "portfolio_exposures.parquet": ("candidate_id", "entry_at", "factor_id"),
            "holding_factor_returns.parquet": (
                "attribution_group",
                "entry_at",
                "factor_id",
            ),
        }[member]
        if result.duplicated(list(keys)).any():
            raise ValueError("statistics_table_duplicate_keys")
        if "status" in result:
            if not result.status.isin(("DEFINED", "UNDEFINED", "BLOCKED")).all():
                raise ValueError("statistics_table_status_invalid")
            value_column = (
                "capacity" if member == "portfolio_capacity.parquet" else "value"
            )
            defined = result.status.eq("DEFINED")
            if (
                result.loc[defined, value_column].isna().any()
                or result.loc[~defined, value_column].notna().any()
                or result.loc[~defined, "reason"].eq("").any()
            ):
                raise ValueError("statistics_table_status_value_invalid")
        if member == "portfolio_daily.parquet":
            if (
                result[["cash_post", "nav_before", "nav_post", "nav_end"]]
                .isna()
                .any()
                .any()
                or result[["nav_before", "nav_post", "nav_end"]].lt(0).any().any()
                or not result.cash_post.between(0, result.nav_post).all()
            ):
                raise ValueError("statistics_cash_exposure_invalid")
            expected_cash_weight = pd.Series(
                [
                    finite_exact_float(Fraction(cash) / Fraction(nav))
                    if nav > 0
                    else None
                    for cash, nav in zip(result.cash_post, result.nav_post, strict=True)
                ],
                index=result.index,
                dtype="float64",
            )
            if not (
                result.cash_weight_post.eq(expected_cash_weight)
                | (result.cash_weight_post.isna() & expected_cash_weight.isna())
            ).all():
                raise ValueError("statistics_cash_exposure_invalid")
            positive_start = result.nav_before.gt(0)
            ratios = [
                "net_return",
                "gross_return",
                "cost_drag",
                "target_turnover",
                "requested_turnover",
                "realized_turnover",
            ]
            expected_return = pd.Series(
                [
                    finite_exact_float(
                        (Fraction(end) - Fraction(start)) / Fraction(start)
                    )
                    if start > 0
                    else None
                    for start, end in zip(
                        result.nav_before, result.nav_end, strict=True
                    )
                ],
                index=result.index,
                dtype="float64",
            )
            for actual, expected, start, end in zip(
                result.net_return,
                expected_return,
                result.nav_before,
                result.nav_end,
                strict=True,
            ):
                if pd.isna(actual) or pd.isna(expected):
                    if pd.isna(actual) and pd.isna(expected):
                        continue
                    raise ValueError("statistics_return_status_invalid")
                # Raw-price PnL cancels before rounding; stored money totals round
                # independently. Allow their monetary rounding, not a fixed atol.
                rounding = (
                    2 * (Fraction(ulp(start)) + Fraction(ulp(end))) / Fraction(start)
                    + Fraction(ulp(actual))
                    + Fraction(ulp(expected))
                )
                if abs(Fraction(actual) - Fraction(expected)) > rounding:
                    raise ValueError("statistics_return_status_invalid")
            defined_return = positive_start & result.net_return.notna()
            if (
                not result.return_status.eq(
                    defined_return.map({True: "DEFINED", False: "UNDEFINED"})
                ).all()
                or not result.return_reason.eq(
                    pd.Series(
                        [
                            ""
                            if defined
                            else "nonfinite_result"
                            if positive
                            else "nonpositive_start_nav"
                            for defined, positive in zip(
                                defined_return, positive_start, strict=True
                            )
                        ],
                        index=result.index,
                    )
                ).all()
                or result.loc[~positive_start, ratios].notna().any().any()
            ):
                raise ValueError("statistics_return_status_invalid")
        if "end_at" in result and (not result.entry_at.lt(result.end_at).all()):
            raise ValueError("statistics_holding_interval_invalid")
        if "decision_at" in result and (
            not result.decision_at.lt(result.entry_at).all()
        ):
            raise ValueError("statistics_holding_interval_invalid")
        if member == "portfolio_attribution.parquet" and (
            not result.component_type.isin(("additive", "detail")).all()
        ):
            raise ValueError("statistics_attribution_component_invalid")
    keys = {
        "model_trials.parquet": (
            "selection_group",
            "decision_id",
            "family_id",
            "config_id",
            "metric_id",
        ),
        "prediction_diagnostics.parquet": ("prediction_id",),
        "robustness.parquet": ("test_id",),
        "factor_diagnostics.parquet": ("family_id", "factor_id", "fdr_method"),
        "fama_macbeth.parquet": ("regression_id", "coefficient_id"),
        "fama_macbeth_periods.parquet": (
            "regression_id",
            "coefficient_id",
            "timestamp",
        ),
    }.get(member)
    if keys is not None and result.duplicated(list(keys)).any():
        raise ValueError("statistics_table_duplicate_keys")
    if member == "factor_stratified_diagnostics.parquet":
        _validate_factor_stratified(result, metadata)
    _validate_table_binding(result, member=member, metadata=metadata)
    result.index = pd.RangeIndex(len(result))
    return result


def _validate_factor_stratified(frame, metadata):
    """Bind concrete factor summaries to their persisted EVAL inputs and family."""
    binding = metadata.get("evaluation_binding", {})
    policy = metadata.get("statistics", {})
    if frame.duplicated(
        ["signal", "label", "sample_scope", "stratification", "stratum", "statistic"]
    ).any():
        raise ValueError("statistics_stratified_duplicate_keys")
    signals = {item["signal_id"] for item in binding.get("signals", [])}
    scopes = set(metadata.get("family_scopes", []))
    expected = {
        (signal, label, scope, "all", None, statistic)
        for signal in signals
        for label in binding.get("label_columns", [])
        for scope in scopes
        for statistic in ("ic", "rankic")
    }
    expected.update(
        (signal, binding["strata"]["primary_label"], scope, kind, stratum, statistic)
        for kind, strata in metadata.get("stratum_domains", {}).items()
        for stratum in strata
        for signal in signals
        for scope in scopes
        for statistic in ("ic", "rankic")
    )
    actual = {
        (
            row.signal,
            row.label,
            row.sample_scope,
            row.stratification,
            None if pd.isna(row.stratum) else row.stratum,
            row.statistic,
        )
        for row in frame.itertuples(index=False)
    }
    if actual != expected:
        raise ValueError("statistics_stratified_domain_incomplete")
    if (
        not frame.signal.isin(
            [item["signal_id"] for item in binding.get("signals", [])]
        ).all()
        or not frame.label.isin(binding.get("label_columns", [])).all()
        or not frame.sample_scope.isin(metadata.get("family_scopes", [])).all()
        or not frame.statistic.isin(("ic", "rankic")).all()
    ):
        raise ValueError("statistics_stratified_domain_invalid")
    if (
        not (
            frame.input_date_count.ge(frame.observed_date_count)
            & frame.observed_date_count.ge(frame.observation_count)
        ).all()
        or frame["mean"].notna().ne(frame.observation_count.gt(0)).any()
        or frame.observation_pairs.lt(2 * frame.observation_count).any()
    ):
        raise ValueError("statistics_stratified_count_invalid")
    for scope, block in frame.groupby("sample_scope"):
        if not block.input_date_count.eq(
            metadata.get("input_date_counts", {}).get(scope)
        ).all():
            raise ValueError("statistics_stratified_count_invalid")
    for key in ("hac_lag", "fdr_method", "fdr_alpha"):
        if not frame[key].eq(policy.get(key)).all():
            raise ValueError("statistics_stratified_policy_invalid")
    if (
        not frame.fdr_method.isin(("BH", "BY")).all()
        or not frame.fdr_alpha.between(0, 1, inclusive="neither").all()
    ):
        raise ValueError("statistics_stratified_policy_invalid")
    defined = frame.status.eq("DEFINED")
    if (
        frame.fdr_adjusted_p_value.notna().ne(defined).any()
        or not frame.defined_hypothesis_count.eq(int(defined.sum())).all()
    ):
        raise ValueError("statistics_stratified_fdr_invalid")
    if (
        (defined & frame.observation_count.le(frame.hac_lag)).any()
        or (defined & frame.observation_count.lt(2)).any()
        or not frame["mean"].dropna().between(-1, 1).all()
    ):
        raise ValueError("statistics_stratified_inference_invalid")
    sufficient = frame.observation_count.ge(2) & frame.observation_count.gt(
        frame.hac_lag
    )
    incomplete = frame.observation_count.lt(frame.input_date_count)
    if (
        (defined & incomplete).any()
        or (defined & frame.fdr_adjusted_p_value.lt(frame.p_value)).any()
        or frame.loc[incomplete, "reason"].ne("incomplete_session_clock").any()
        or frame.loc[~defined & ~incomplete & sufficient, "reason"]
        .ne("hac_inference_undefined")
        .any()
        or frame.loc[~incomplete & ~sufficient, "reason"]
        .ne("insufficient_dates_for_hac")
        .any()
    ):
        raise ValueError("statistics_stratified_inference_invalid")
    for kind, block in frame.groupby("stratification", sort=False):
        if kind == "all":
            if (
                block.stratum.notna().any()
                or not block.observed_date_count.eq(block.input_date_count).all()
            ):
                raise ValueError("statistics_stratified_domain_invalid")
        elif (
            kind not in binding.get("strata", {}).get("kinds", [])
            or not block.label.eq(binding["strata"]["primary_label"]).all()
            or not block.stratum.dropna()
            .isin(metadata.get("stratum_domains", {}).get(kind, []))
            .all()
        ):
            raise ValueError("statistics_stratified_domain_invalid")

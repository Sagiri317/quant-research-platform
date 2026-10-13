"""Multi-table semantic artifact for formal research statistics."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import pandas as pd
import pyarrow as pa

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
            pa.field("benchmark_return", pa.float64()),
            pa.field("active_return", pa.float64()),
            pa.field("relative_wealth", pa.float64()),
            pa.field("benchmark_status", pa.string()),
            pa.field("benchmark_reason", pa.string()),
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
            pa.field("benchmark_return", pa.float64()),
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
STATISTICS_TABLE_GROUPS = {
    "factor": (
        "factor_diagnostics.parquet",
        "fama_macbeth.parquet",
        "fama_macbeth_periods.parquet",
    ),
    "research": (
        "prediction_diagnostics.parquet",
        "model_trials.parquet",
        "robustness.parquet",
        "model_losses.parquet",
        "model_comparison.parquet",
        "model_group_comparison.parquet",
        "model_reality_checks.parquet",
    ),
    "portfolio": (
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
        "crypto_maker_events.parquet",
        "crypto_maker_samples.parquet",
        "crypto_portfolio_daily.parquet",
        "fama_macbeth_periods.parquet",
        "holding_factor_returns.parquet",
        "model_comparison.parquet",
        "model_losses.parquet",
        "model_group_comparison.parquet",
        "portfolio_capacity.parquet",
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
            frame = parquet.read().to_pandas()
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
    members = tuple(STATISTICS_TABLE_GROUPS[table_group])
    expected = {member.removesuffix(".parquet") for member in members}
    calibration = table_group == "crypto_maker_calibration"
    if callable(tables) and not calibration:
        raise ValueError("statistics_tables_invalid")
    if calibration and not callable(validate_tables):
        raise ValueError("statistics_business_validator_required")
    if not callable(tables) and (
        not isinstance(tables, Mapping) or set(tables) != expected
    ):
        raise ValueError("statistics_tables_invalid")

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
    members = tuple(STATISTICS_TABLE_GROUPS[group])
    if manifest.get("members") != list(members):
        raise ValueError("statistics_artifact_members_invalid")
    metadata = manifest.get("metadata")
    _validate_metadata(metadata)
    artifact = StatisticsArtifact(metadata, root, group, members)
    if group == "crypto_maker_calibration":
        if not callable(validate_tables):
            raise ValueError("statistics_business_validator_required")
        validate_tables(artifact, metadata)
    return artifact


def _validate_metadata(metadata):
    if not isinstance(metadata, dict) or metadata.get("sample_role") not in {
        "train",
        "validation",
        "test",
    }:
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
        values = result[column]
        if values.isna().any():
            result[column] = values.astype("string").fillna("")
        result[column] = result[column].astype(str)
    for column in _COUNT_COLUMNS[member]:
        values = strict_integer_series(
            result[column], error="statistics_table_count_invalid"
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
    if (
        member
        in (*STATISTICS_TABLE_GROUPS["research"], *STATISTICS_TABLE_GROUPS["factor"])
        and member != "model_group_comparison.parquet"
    ):
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
            if result["mean"].notna().ne(result.observation_count.gt(0)).any():
                raise ValueError("statistics_table_descriptive_mean_count_invalid")
        for column in ("p_value", "fdr_adjusted_p_value", "white_p_value"):
            if column in result and (not result[column].dropna().between(0, 1).all()):
                raise ValueError("statistics_table_probability_invalid")
        if "standard_error" in result and result.standard_error.lt(0).any():
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
            not result["sample_role"].isin(("validation", "test")).all()
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
        fdr_defined = defined & result.fdr_method.eq("BY")
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
            or (not result.sample_role.isin(("validation", "test")).all())
            or (result.sample_role.nunique() > 1)
            or (defined & result.sample_role.ne("test")).any()
            or (bootstrap_defined & ~defined).any()
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
            or (not result.fdr_method.isin(("BY", "NOT_APPLICABLE")).all())
            or result.loc[fdr_defined, "fdr_adjusted_p_value"].isna().any()
            or result.loc[~fdr_defined, "fdr_adjusted_p_value"].notna().any()
            or (not result.fdr_alpha.between(0, 1, inclusive="neither").all())
            or (not result.confidence.between(0, 1, inclusive="neither").all())
            or (not result.fdr_family_size.eq(len(result)).all())
            or result.standard_error.lt(0).any()
            or result.ci_lower.gt(result.ci_upper).any()
            or result.bootstrap_lower.gt(result.bootstrap_upper).any()
            or (not result.p_value.dropna().between(0, 1).all())
            or (not result.fdr_adjusted_p_value.dropna().between(0, 1).all())
        ):
            raise ValueError("statistics_group_comparison_invalid")
        if (
            member == "model_group_comparison.parquet"
            and result.observation_count.gt(result.calendar_count).any()
        ):
            raise ValueError("statistics_group_comparison_count_invalid")
        if (
            member == "portfolio_group_comparison.parquet"
            and not result.empty
            and (
                not result.comparison_role.isin(("primary", "secondary")).all()
                or result.comparison_role.eq("primary").sum() != 1
                or result.loc[
                    ~result.reason.isin(("nonpositive_start_nav", "nonfinite_result")),
                    [
                        "global_mean",
                        "quarterly_mean",
                        "mean_difference",
                    ],
                ]
                .isna()
                .any()
                .any()
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
            result.loc[defined, ["start_nav", "end_nav", "benchmark_return"]]
            .isna()
            .any()
            .any()
            or result.loc[defined, "start_nav"].le(0).any()
        ):
            raise ValueError("statistics_table_status_value_invalid")
    if member in STATISTICS_TABLE_GROUPS["crypto_portfolio"]:
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
    if (
        member in STATISTICS_TABLE_GROUPS["portfolio"]
        and member != "portfolio_group_comparison.parquet"
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
            if not (
                result.net_return.eq(expected_return)
                | (result.net_return.isna() & expected_return.isna())
            ).all():
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
                or result.loc[~defined_return, "active_return"].notna().any()
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
    _validate_table_binding(result, member=member, metadata=metadata)
    result.index = pd.RangeIndex(len(result))
    return result

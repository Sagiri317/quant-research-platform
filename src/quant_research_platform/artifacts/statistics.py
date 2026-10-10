"""Multi-table semantic artifact for formal research statistics."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import pandas as pd
import pyarrow as pa

from quant_research_platform.contracts.business_manifest import BusinessManifestEnvelope
from quant_research_platform.contracts.time import (
    TimeWindow,
    a_share_analysis_windows,
    crypto_execution_windows,
)
from quant_research_platform.contracts.validation import (
    strict_finite_numeric_series,
    strict_integer_series,
)
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.storage.datasets.reading import (
    iter_declared_parquet_member_batches,
)
from quant_research_platform.storage.identity import (
    DatasetFileIdentity,
    dataset_file_identity,
    open_filesystem_file,
)
from quant_research_platform.storage.parquet import (
    _opened_parquet,
    parquet_table_writer,
)
from quant_research_platform.storage.semantic_artifacts import (
    semantic_artifact_read_session,
    verify_semantic_artifact_member,
    verify_semantic_artifact_snapshot,
    write_semantic_artifact,
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
            pa.field("access_window", pa.string()),
            pa.field("quarter_id", pa.string()),
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
            pa.field("access_window", pa.string()),
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
                "access_window",
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
            pa.field("access_window", pa.string()),
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
            pa.field("benchmark_return", pa.float64()),
            pa.field("active_return", pa.float64()),
            pa.field("relative_wealth", pa.float64()),
            pa.field("benchmark_status", pa.string()),
            pa.field("benchmark_reason", pa.string()),
            pa.field("access_window", pa.string()),
        ]
    ),
    "portfolio_group_comparison.parquet": pa.schema(
        [
            *(
                (name, pa.string())
                for name in (
                    "portfolio_id",
                    "comparison_role",
                    "access_window",
                    "status",
                    "reason",
                    "bootstrap_status",
                    "bootstrap_reason",
                    "fdr_method",
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
            pa.field("access_window", pa.string()),
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
            pa.field("access_window", pa.string()),
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
    envelope: dict
    root: Path
    identity: str
    _members: Mapping[str, object]
    _identities: Mapping[str, DatasetFileIdentity]
    _row_counts: Mapping[str, int]
    _snapshot: Mapping[str, DatasetFileIdentity] | None = None
    _directories: Mapping[str, tuple[int, int, int, int]] | None = None

    def _verify_snapshot(self):
        if self._snapshot is not None:
            verify_semantic_artifact_snapshot(
                self.root, self._snapshot, self._directories
            )

    def _verify(self):
        self._verify_snapshot()
        for member, identity in self._identities.items():
            path = self.root / member
            if dataset_file_identity(path) != identity:
                raise ValueError("statistics_table_identity_changed")
            verify_semantic_artifact_member(
                self.root, relative_path=member, expected=dict(self._members[member])
            )
            if dataset_file_identity(path) != identity:
                raise ValueError("statistics_table_identity_changed")

    def _batches(self, member, *, filters=None):
        if member not in self._members:
            raise ValueError("statistics_table_member_invalid")
        with closing_iterator(
            iter_declared_parquet_member_batches(
                self.root / member,
                expected_identity=self._identities[member],
                parquet_filters=filters,
            )
        ) as frames:
            for frame in frames:
                result = _validated_frame(frame, member=member, envelope=self.envelope)
                try:
                    yield result
                except GeneratorExit:
                    return

    @contextmanager
    def table(self, member: str):
        if member not in self._members:
            raise ValueError("statistics_table_member_invalid")
        path = self.root / member
        identity = self._identities[member]
        self._verify_snapshot()
        verify_semantic_artifact_member(
            self.root, relative_path=member, expected=dict(self._members[member])
        )
        with _opened_parquet(path, expected_identity=identity) as (
            parquet,
            _identity,
            _handle,
        ):
            if parquet.schema_arrow != _ARROW_SCHEMAS[member]:
                raise ValueError("statistics_table_schema_invalid")
            table = parquet.read()
            frame = table.to_pandas()
            result = _validated_frame(frame, member=member, envelope=self.envelope)
            primary = None
            try:
                yield result
            except BaseException as error:
                primary = error
                raise
            finally:
                try:
                    self._verify_snapshot()
                except BaseException as error:
                    if primary is None:
                        raise
                    primary.add_note(
                        "statistics snapshot verification also failed: "
                        + type(error).__name__
                    )


def write_statistics_artifact(
    logical_root: Path,
    *,
    envelope: BusinessManifestEnvelope,
    tables: Mapping[str, pd.DataFrame] | Callable[..., None],
    table_group: str = "research",
    validate_tables=None,
) -> Path:
    if table_group not in STATISTICS_TABLE_GROUPS:
        raise ValueError("statistics_tables_invalid")
    envelope_payload = envelope.payload()
    if set(_statistics_members(envelope_payload)) != set(
        STATISTICS_TABLE_GROUPS[table_group]
    ):
        raise ValueError("statistics_artifact_contract_mismatch")
    calibration = table_group == "crypto_maker_calibration"
    if calibration and (not callable(validate_tables)):
        raise ValueError("statistics_business_validator_required")
    expected = {
        member.removesuffix(".parquet")
        for member in STATISTICS_TABLE_GROUPS[table_group]
    }
    if (
        calibration
        and (not callable(tables))
        or (
            not calibration
            and (not isinstance(tables, Mapping) or set(tables) != expected)
        )
    ):
        raise ValueError("statistics_tables_invalid")

    def write_batches(staging, produce):
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

                validated = _validated_frame(
                    frame, member=member, envelope=envelope_payload
                )
                table = pa.Table.from_pandas(
                    validated, schema=_ARROW_SCHEMAS[member], preserve_index=False
                )
                writers[member](table)

            produce(on_table=append)
            return {member.removesuffix(".parquet") for member in writers}

    def write(staging):
        if calibration:
            written = write_batches(staging, tables)
            if written != expected:
                raise ValueError("statistics_tables_invalid")
        else:
            for table_name, frame in tables.items():
                write_batches(
                    staging,
                    lambda on_table, name=table_name, value=frame: on_table(
                        name, value
                    ),
                )

    def validate(staging, members):
        artifact = _statistics_artifact(staging, envelope.payload(), members)
        validate_tables(artifact, envelope.payload())

    return write_semantic_artifact(
        Path(logical_root),
        envelope,
        write,
        staged_validator=validate if calibration else None,
    )


def _statistics_artifact(
    root, envelope, members, member_identities=None, directory_identities=None
):
    by_path = {str(member.get("path")): member for member in members}
    if set(by_path) != set(_statistics_members(envelope)) or len(by_path) != len(
        members
    ):
        raise ValueError("statistics_artifact_members_invalid")
    identities, row_counts = ({}, {})
    for member in by_path:
        path = root / member
        physical = (
            verify_semantic_artifact_member(
                root, relative_path=member, expected=dict(by_path[member])
            )
            if member_identities is None
            else member_identities[member]
        )
        with _opened_parquet(path, expected_identity=physical) as (parquet, _, _handle):
            if parquet.schema_arrow != _ARROW_SCHEMAS[member]:
                raise ValueError("statistics_table_schema_invalid")
            count = parquet.metadata.num_rows
            if count == 0 and member not in _EMPTY_TABLE_MEMBERS:
                raise ValueError("statistics_table_empty")
            identities[member] = physical
            row_counts[member] = count
    return StatisticsArtifact(
        envelope=envelope,
        root=root,
        identity=envelope["instance_identity"],
        _members=MappingProxyType(by_path),
        _identities=MappingProxyType(identities),
        _row_counts=MappingProxyType(row_counts),
        _snapshot=(
            MappingProxyType(dict(member_identities))
            if directory_identities is not None
            else None
        ),
        _directories=(
            MappingProxyType(dict(directory_identities))
            if directory_identities is not None
            else None
        ),
    )


def read_statistics_artifact(
    instance_root: Path,
    *,
    expected_identity: str,
    validate_tables=None,
    expected_member_identities=None,
) -> StatisticsArtifact:
    identities = {}
    directories = {}
    with semantic_artifact_read_session(
        Path(instance_root),
        expected_identity=expected_identity,
        member_identities=identities,
        expected_member_identities=expected_member_identities,
        directory_identities=directories,
    ) as (envelope, members):
        artifact = _statistics_artifact(
            Path(instance_root), envelope, members, identities, directories
        )
        if set(artifact._members) == set(
            STATISTICS_TABLE_GROUPS["crypto_maker_calibration"]
        ):
            if not callable(validate_tables):
                raise ValueError("statistics_business_validator_required")
            validate_tables(artifact, envelope)
        return artifact


def _statistics_members(envelope):
    contract = envelope["semantic_metadata"].get("target_contract")
    if type(contract) is not dict or type(contract.get("protocol")) is not dict:
        raise ValueError("statistics_artifact_contract_invalid")
    declared = contract.get("statistics_tables")
    protocol = contract["protocol"]
    if (
        type(declared) is not dict
        or protocol.get("access_window") != envelope["research_window"]["access"]
    ):
        raise ValueError("statistics_artifact_contract_invalid")
    for members in STATISTICS_TABLE_GROUPS.values():
        if declared == {
            member.removesuffix(".parquet"): list(_TABLE_COLUMNS[member])
            for member in members
        }:
            return members
    raise ValueError("statistics_artifact_contract_mismatch")


def _validate_table_binding(frame, *, member, envelope):
    protocol = envelope["semantic_metadata"]["target_contract"]["protocol"]
    window = envelope["research_window"]
    access = window["access"]
    if "access_window" in frame and not frame.access_window.eq(access).all():
        raise ValueError("statistics_access_window_mismatch")
    if member in STATISTICS_TABLE_GROUPS["crypto_maker_calibration"]:
        # Calibration has its own fold, maturity and event-reference validator.
        return
    start, end = pd.Timestamp(window["start"]), pd.Timestamp(window["end"])
    if member in STATISTICS_TABLE_GROUPS["crypto_portfolio"]:
        crypto_execution_windows(
            protocol["research_plan"], TimeWindow(start, end), access_window=access
        )
        candidates = {item["candidate_id"] for item in protocol["portfolio_candidates"]}
        if not frame.execution_mode.eq(protocol["execution_mode"]).all():
            raise ValueError("statistics_execution_mode_mismatch")
    elif member in STATISTICS_TABLE_GROUPS["portfolio"]:
        candidates = set(protocol["portfolio_candidates"])
        if member == "portfolio_robustness.parquet":
            joint = frame.candidate_id.eq("joint") & frame.test_id.isin(
                ("cscv_pbo", "development_only")
            )
            if not (frame.candidate_id.isin(candidates) | joint).all():
                raise ValueError("statistics_candidate_id_mismatch")
        if (
            "portfolio_id" in frame
            and not frame.portfolio_id.isin(
                protocol["statistics"]["group_comparison"]["portfolio_ids"]
            ).all()
        ):
            raise ValueError("statistics_candidate_id_mismatch")
        if (
            "attribution_group" in frame
            and not frame.attribution_group.isin(protocol["attribution_groups"]).all()
        ):
            raise ValueError("statistics_candidate_id_mismatch")
    else:
        candidates = set()
    if (
        "candidate_id" in frame
        and member != "portfolio_robustness.parquet"
        and not frame.candidate_id.isin(candidates).all()
    ):
        raise ValueError("statistics_candidate_id_mismatch")
    if member in {"model_losses.parquet", "fama_macbeth_periods.parquet"}:
        factor, model = a_share_analysis_windows(protocol, TimeWindow(start, end))
        selected = factor if member == "fama_macbeth_periods.parquet" else model
        start, end = pd.Timestamp(selected.start), pd.Timestamp(selected.end)
        role = protocol[f"{access}_window"]
        role_start = (
            protocol["formal_dataset_window"]["start"]
            if member == "fama_macbeth_periods.parquet" and access == "development"
            else role["start"]
        )
        start, end = (
            max(start, pd.Timestamp(role_start)),
            min(end, pd.Timestamp(role["end"])),
        )
    for column in _TIMESTAMP_COLUMNS[member]:
        upper = (
            end + pd.Timedelta(1, "ns")
            if member == "crypto_portfolio_coverage.parquet" and column == "stop_at"
            else end
        )
        if not frame[column].between(start, upper).all():
            raise ValueError("statistics_table_window_mismatch")


def _validated_frame(frame: pd.DataFrame, *, member: str, envelope) -> pd.DataFrame:
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
            not result["access_window"].isin(("development", "holdout")).all()
            or result["access_window"].nunique() > 1
        ):
            raise ValueError("statistics_access_window_invalid")
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
            or (not result.access_window.isin(("development", "holdout")).all())
            or (result.access_window.nunique() > 1)
            or (defined & result.access_window.ne("holdout")).any()
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
        if member == "portfolio_group_comparison.parquet" and (
            not result.comparison_role.isin(("primary", "secondary")).all()
            or result.comparison_role.eq("primary").sum() != 1
            or result[
                [
                    "global_mean",
                    "quarterly_mean",
                    "mean_difference",
                    "global_terminal_wealth",
                    "quarterly_terminal_wealth",
                ]
            ]
            .isna()
            .any()
            .any()
        ):
            raise ValueError("statistics_group_comparison_primary_invalid")
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
            valid_episode = result.episode_id.str.fullmatch("[0-9a-f]{64}")
            if (
                ~aggregate & continuous & ~result.episode_id.eq("continuous")
                | ~aggregate & ~continuous & ~valid_episode
            ).any():
                raise ValueError("statistics_episode_id_invalid")
            if aggregate.any() and (
                member != "crypto_portfolio_statistics.parquet"
                or not result.loc[aggregate, "metric_id"]
                .isin(
                    (
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
                    )
                )
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
        if member == "portfolio_daily.parquet" and (
            result[["cash_post", "cash_weight_post", "nav_post"]].isna().any().any()
            or result.nav_post.le(0).any()
            or not result.cash_post.between(0, result.nav_post).all()
            or not result.cash_weight_post.eq(result.cash_post / result.nav_post).all()
        ):
            raise ValueError("statistics_cash_exposure_invalid")
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
    _validate_table_binding(result, member=member, envelope=envelope)
    result.index = pd.RangeIndex(len(result))
    return result

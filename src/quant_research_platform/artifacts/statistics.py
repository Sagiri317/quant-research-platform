"""Multi-table semantic artifact for formal research statistics."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import pandas as pd
import pyarrow as pa

from quant_research_platform.contracts.business_manifest import (
    BusinessManifestEnvelope,
)
from quant_research_platform.contracts.validation import (
    strict_finite_numeric_series,
    strict_integer_series,
)
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.runtime.resources import (
    current_runtime_resource_policy,
    pandas_frame_payload_bytes,
    runtime_resource_policy_scope,
)
from quant_research_platform.storage._scratch_ownership import handle_preserving_primary
from quant_research_platform.storage.datasets.reading import (
    iter_declared_parquet_member_batches,
)
from quant_research_platform.storage.identity import (
    DatasetFileIdentity,
    dataset_file_identity,
    native_filesystem_path,
)
from quant_research_platform.storage.parquet import (
    _opened_parquet,
    _parquet_frame_bound,
    _parquet_frame_reservation,
    parquet_decoded_working_set_bounds,
    parquet_table_writer,
)
from quant_research_platform.storage.semantic_artifacts import (
    read_semantic_artifact_record,
    verify_semantic_artifact_member,
    write_semantic_artifact,
)

_TABLE_COLUMNS = {
    "factor_diagnostics.parquet": (
        "family_id",
        "factor_id",
        "observation_count",
        "status",
        "reason",
        "mean",
        "standard_error",
        "t_statistic",
        "p_value",
        "ci_lower",
        "ci_upper",
        "fdr_method",
        "fdr_alpha",
        "fdr_adjusted_p_value",
        "planned_hypothesis_count",
        "defined_hypothesis_count",
        "missing_hypothesis_count",
    ),
    "prediction_diagnostics.parquet": (
        "prediction_id",
        "observation_count",
        "status",
        "reason",
        "mean",
        "standard_error",
        "t_statistic",
        "p_value",
        "ci_lower",
        "ci_upper",
        "bootstrap_lower",
        "bootstrap_upper",
        "bootstrap_block_size",
        "bootstrap_repetitions",
        "bootstrap_seed",
    ),
    "model_trials.parquet": (
        "quarter_id",
        "family_id",
        "config_id",
        "metric_id",
        "value",
        "observation_count",
        "status",
        "reason",
        "selection_status",
    ),
    "robustness.parquet": (
        "test_id",
        "status",
        "reason",
        "observation_count",
    ),
    "fama_macbeth.parquet": (
        "regression_id",
        "coefficient_id",
        "period_count",
        "status",
        "reason",
        "mean",
        "standard_error",
        "t_statistic",
        "p_value",
        "ci_lower",
        "ci_upper",
        "white_defined_period_count",
        "white_undefined_period_count",
    ),
    "fama_macbeth_periods.parquet": (
        "regression_id",
        "coefficient_id",
        "timestamp",
        "status",
        "reason",
        "observation_count",
        "value",
        "white_p_value",
        "white_reason",
    ),
}
_STRING_COLUMNS = {
    "factor_diagnostics.parquet": (
        "family_id",
        "factor_id",
        "status",
        "reason",
        "fdr_method",
    ),
    "prediction_diagnostics.parquet": ("prediction_id", "status", "reason"),
    "model_trials.parquet": (
        "quarter_id",
        "family_id",
        "config_id",
        "metric_id",
        "status",
        "reason",
        "selection_status",
    ),
    "robustness.parquet": ("test_id", "status", "reason"),
    "fama_macbeth.parquet": (
        "regression_id",
        "coefficient_id",
        "status",
        "reason",
    ),
    "fama_macbeth_periods.parquet": (
        "regression_id",
        "coefficient_id",
        "status",
        "reason",
        "white_reason",
    ),
}
_COUNT_COLUMNS = {
    "factor_diagnostics.parquet": (
        "observation_count",
        "planned_hypothesis_count",
        "defined_hypothesis_count",
        "missing_hypothesis_count",
    ),
    "prediction_diagnostics.parquet": (
        "observation_count",
        "bootstrap_block_size",
        "bootstrap_repetitions",
        "bootstrap_seed",
    ),
    "model_trials.parquet": ("observation_count",),
    "robustness.parquet": ("observation_count",),
    "fama_macbeth.parquet": (
        "period_count",
        "white_defined_period_count",
        "white_undefined_period_count",
    ),
    "fama_macbeth_periods.parquet": ("observation_count",),
}
_TABLE_COLUMNS.update(
    {
        "model_losses.parquet": (
            "timestamp",
            "access_window",
            "quarter_id",
            "family_id",
            "config_id",
            "loss",
            "observation_count",
            "status",
            "reason",
            "common_date",
        ),
        "model_comparison.parquet": (
            "access_window",
            "benchmark_id",
            *_TABLE_COLUMNS["prediction_diagnostics.parquet"],
        ),
        "model_reality_checks.parquet": (
            "test_id",
            "access_window",
            "family_id",
            "statistic_id",
            "value",
            "status",
            "reason",
            "observation_count",
            "block_size",
            "repetitions",
            "alpha",
            "seed",
        ),
    }
)
_STRING_COLUMNS.update(
    {
        "model_losses.parquet": (
            "access_window",
            "quarter_id",
            "family_id",
            "config_id",
            "status",
            "reason",
        ),
        "model_comparison.parquet": (
            "access_window",
            "benchmark_id",
            *_STRING_COLUMNS["prediction_diagnostics.parquet"],
        ),
        "model_reality_checks.parquet": (
            "test_id",
            "access_window",
            "family_id",
            "statistic_id",
            "status",
            "reason",
        ),
    }
)
_COUNT_COLUMNS.update(
    {
        "model_losses.parquet": ("observation_count", "common_date"),
        "model_comparison.parquet": _COUNT_COLUMNS["prediction_diagnostics.parquet"],
        "model_reality_checks.parquet": (
            "observation_count",
            "block_size",
            "repetitions",
            "seed",
        ),
    }
)
_TIMESTAMP_COLUMNS = {
    "fama_macbeth_periods.parquet": ("timestamp",),
    "model_losses.parquet": ("timestamp",),
}
STATISTICS_TABLE_GROUPS = {"research": tuple(_TABLE_COLUMNS)}
_PORTFOLIO_COLUMNS = {
    "portfolio_daily.parquet": (
        "candidate_id",
        "entry_at",
        "end_at",
        "nav_before",
        "nav_post",
        "nav_end",
        "gross_return",
        "net_return",
        "cost_drag",
        "target_turnover",
        "requested_turnover",
        "realized_turnover",
        "benchmark_return",
        "active_return",
        "relative_wealth",
        "benchmark_status",
        "benchmark_reason",
        "access_window",
    ),
    "portfolio_performance.parquet": (
        "candidate_id",
        "metric_id",
        "value",
        "status",
        "reason",
        "observation_count",
        "access_window",
    ),
    "portfolio_capacity.parquet": (
        "candidate_id",
        "decision_timestamp",
        "entry_at",
        "participation",
        "capacity",
        "bottleneck_asset_id",
        "status",
        "reason",
    ),
    "portfolio_capacity_summary.parquet": (
        "candidate_id",
        "participation",
        "period",
        "metric_id",
        "value",
        "status",
        "reason",
        "observation_count",
    ),
    "portfolio_attribution.parquet": (
        "candidate_id",
        "decision_at",
        "entry_at",
        "end_at",
        "component_id",
        "component_type",
        "value",
        "status",
        "reason",
    ),
    "portfolio_robustness.parquet": (
        "candidate_id",
        "test_id",
        "value",
        "status",
        "reason",
        "observation_count",
        "candidate_count",
        "access_window",
    ),
    "portfolio_exposures.parquet": (
        "candidate_id",
        "decision_at",
        "entry_at",
        "factor_id",
        "value",
    ),
    "holding_factor_returns.parquet": (
        "decision_at",
        "entry_at",
        "end_at",
        "factor_id",
        "value",
        "observation_count",
    ),
}
STATISTICS_TABLE_GROUPS["portfolio"] = tuple(_PORTFOLIO_COLUMNS)
_TABLE_COLUMNS.update(_PORTFOLIO_COLUMNS)
_STRING_COLUMNS.update(
    {
        "portfolio_daily.parquet": (
            "candidate_id",
            "benchmark_status",
            "benchmark_reason",
            "access_window",
        ),
        "portfolio_performance.parquet": (
            "candidate_id",
            "metric_id",
            "status",
            "reason",
            "access_window",
        ),
        "portfolio_capacity.parquet": (
            "candidate_id",
            "bottleneck_asset_id",
            "status",
            "reason",
        ),
        "portfolio_capacity_summary.parquet": (
            "candidate_id",
            "period",
            "metric_id",
            "status",
            "reason",
        ),
        "portfolio_attribution.parquet": (
            "candidate_id",
            "component_id",
            "component_type",
            "status",
            "reason",
        ),
        "portfolio_robustness.parquet": (
            "candidate_id",
            "test_id",
            "status",
            "reason",
            "access_window",
        ),
        "portfolio_exposures.parquet": ("candidate_id", "factor_id"),
        "holding_factor_returns.parquet": ("factor_id",),
    }
)
_COUNT_COLUMNS.update(
    {
        "portfolio_daily.parquet": (),
        "portfolio_performance.parquet": ("observation_count",),
        "portfolio_capacity.parquet": (),
        "portfolio_capacity_summary.parquet": ("observation_count",),
        "portfolio_attribution.parquet": (),
        "portfolio_exposures.parquet": (),
        "holding_factor_returns.parquet": ("observation_count",),
        "portfolio_robustness.parquet": ("observation_count", "candidate_count"),
    }
)
_TIMESTAMP_COLUMNS.update(
    {
        "portfolio_daily.parquet": ("entry_at", "end_at"),
        "portfolio_capacity.parquet": ("decision_timestamp", "entry_at"),
        "portfolio_attribution.parquet": ("decision_at", "entry_at", "end_at"),
        "portfolio_exposures.parquet": ("decision_at", "entry_at"),
        "holding_factor_returns.parquet": ("decision_at", "entry_at", "end_at"),
    }
)
_CRYPTO_TABLES = {
    "crypto_maker_samples.parquet": (
        "fold_id",
        "side",
        "sample_id",
        "decision_at",
        "prediction_end",
        "label_available_at",
        "price_prediction",
        "observed_return",
        "forecast_error",
        "filled_quantity",
        "quote_quantity",
        "terminal_reason",
        "book_segment_id",
        "trade_connection_id",
        "initial_book_event_id",
        "event_start",
        "event_end",
    ),
    "crypto_maker_events.parquet": (
        "fold_id",
        "event_id",
        "event_at",
        "event_kind",
        "context_id",
        "side",
        "level_index",
        "price_tick",
        "quantity",
    ),
    "crypto_maker_calibration.parquet": (
        "fold_id",
        "side",
        "sample_count",
        "zero_fill_count",
        "post_only_rejection_count",
        "excluded_decisions",
        "minimum_samples",
        "status",
        "reason",
    ),
    "crypto_portfolio_intervals.parquet": (
        "candidate_id",
        "kind",
        "start_at",
        "end_at",
        "start_nav",
        "end_nav",
        "gross_pnl",
        "net_pnl",
        "fees",
        "funding",
        "turnover",
        "fill_events",
    ),
    "crypto_portfolio_daily.parquet": (
        "candidate_id",
        "fold_id",
        "start_at",
        "end_at",
        "start_nav",
        "end_nav",
        "net_return",
        "benchmark_return",
        "status",
        "reason",
    ),
    "crypto_portfolio_statistics.parquet": (
        "candidate_id",
        "metric_id",
        "value",
        "observation_count",
        "status",
        "reason",
    ),
    "crypto_execution_diagnostics.parquet": (
        "candidate_id",
        "metric_id",
        "value",
        "text",
        "status",
        "reason",
    ),
    "crypto_cost_sensitivity.parquet": (
        "candidate_id",
        "fee_bps",
        "additional_spread_bps",
        "cost_delta",
        "net_pnl",
        "method",
    ),
}
STATISTICS_TABLE_GROUPS["crypto_maker_calibration"] = tuple(_CRYPTO_TABLES)[:3]
STATISTICS_TABLE_GROUPS["crypto_portfolio"] = tuple(_CRYPTO_TABLES)[3:]
_TABLE_COLUMNS.update(_CRYPTO_TABLES)
_STRING_COLUMNS.update(
    {
        "crypto_maker_samples.parquet": (
            "fold_id",
            "side",
            "terminal_reason",
            "book_segment_id",
            "trade_connection_id",
        ),
        "crypto_maker_events.parquet": ("fold_id", "event_kind", "context_id", "side"),
        "crypto_maker_calibration.parquet": ("fold_id", "side", "status", "reason"),
        "crypto_portfolio_intervals.parquet": ("candidate_id", "kind"),
        "crypto_portfolio_daily.parquet": (
            "candidate_id",
            "fold_id",
            "status",
            "reason",
        ),
        "crypto_portfolio_statistics.parquet": (
            "candidate_id",
            "metric_id",
            "status",
            "reason",
        ),
        "crypto_execution_diagnostics.parquet": (
            "candidate_id",
            "metric_id",
            "text",
            "status",
            "reason",
        ),
        "crypto_cost_sensitivity.parquet": ("candidate_id", "method"),
    }
)
_COUNT_COLUMNS.update(
    {
        "crypto_maker_samples.parquet": (
            "sample_id",
            "initial_book_event_id",
            "event_start",
            "event_end",
        ),
        "crypto_maker_events.parquet": ("event_id", "level_index", "price_tick"),
        "crypto_maker_calibration.parquet": (
            "sample_count",
            "zero_fill_count",
            "post_only_rejection_count",
            "excluded_decisions",
            "minimum_samples",
        ),
        "crypto_portfolio_intervals.parquet": ("fill_events",),
        "crypto_portfolio_daily.parquet": (),
        "crypto_portfolio_statistics.parquet": ("observation_count",),
        "crypto_execution_diagnostics.parquet": (),
        "crypto_cost_sensitivity.parquet": (),
    }
)
_TIMESTAMP_COLUMNS.update(
    {
        "crypto_maker_samples.parquet": (
            "decision_at",
            "prediction_end",
            "label_available_at",
        ),
        "crypto_maker_events.parquet": ("event_at",),
        "crypto_portfolio_intervals.parquet": ("start_at", "end_at"),
        "crypto_portfolio_daily.parquet": ("start_at", "end_at"),
    }
)
_EMPTY_TABLE_MEMBERS = frozenset(
    {
        "model_losses.parquet",
        "fama_macbeth_periods.parquet",
        "portfolio_capacity.parquet",
        "portfolio_exposures.parquet",
        "holding_factor_returns.parquet",
        "crypto_maker_samples.parquet",
        "crypto_maker_events.parquet",
        "crypto_portfolio_daily.parquet",
    }
)
_ARROW_SCHEMAS = {
    member: pa.schema(
        pa.field(
            column,
            pa.string()
            if column in _STRING_COLUMNS[member]
            else pa.timestamp("ns")
            if column in _TIMESTAMP_COLUMNS.get(member, ())
            else pa.int64()
            if column in _COUNT_COLUMNS[member]
            else pa.float64(),
        )
        for column in columns
    )
    for member, columns in _TABLE_COLUMNS.items()
}
STATISTICS_TABLE_COLUMNS = _TABLE_COLUMNS


@dataclass(frozen=True, slots=True)
class StatisticsArtifact:
    root: Path
    identity: str
    _members: Mapping[str, object]
    _identities: Mapping[str, DatasetFileIdentity]
    _row_counts: Mapping[str, int]

    def _verify(self):
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
        with runtime_resource_policy_scope(current_runtime_resource_policy()) as policy:
            target = max(1, min(1024**2, policy.operation_headroom_bytes // 8))
            with closing_iterator(
                iter_declared_parquet_member_batches(
                    self.root / member,
                    expected_identity=self._identities[member],
                    target_chunk_bytes=target,
                    operation_headroom_bytes=policy.operation_headroom_bytes,
                    parquet_filters=filters,
                )
            ) as frames:
                for frame in frames:
                    with policy.lease(
                        "artifacts.statistics.validation",
                        max(65536, 2 * pandas_frame_payload_bytes(frame)),
                    ):
                        result = None
                        try:
                            result = _validated_frame(frame, member=member)
                            yield result
                        finally:
                            result = None

    @contextmanager
    def table(self, member: str):
        if member not in self._members:
            raise ValueError("statistics_table_member_invalid")
        path = self.root / member
        identity = self._identities[member]
        verify_semantic_artifact_member(
            self.root,
            relative_path=member,
            expected=dict(self._members[member]),
        )
        with runtime_resource_policy_scope(current_runtime_resource_policy()) as policy:
            with _opened_parquet(path, expected_identity=identity) as (
                parquet,
                _identity,
                _handle,
            ):
                if parquet.schema_arrow != _ARROW_SCHEMAS[member]:
                    raise ValueError("statistics_table_schema_invalid")
                row_bytes, fixed_bytes = parquet_decoded_working_set_bounds(
                    parquet, _TABLE_COLUMNS[member]
                )
                working_bytes = max(
                    1, 4 * fixed_bytes + 8 * row_bytes * parquet.metadata.num_rows
                )
                policy.require_working_set(
                    working_bytes, error="statistics_table_exceeds_task_memory_budget"
                )
                with policy.lease(
                    "artifacts.statistics.table",
                    working_bytes,
                ) as working:
                    table = frame = result = None
                    try:
                        table = parquet.read()
                        actual_bound = max(
                            working_bytes, 8 * table.get_total_buffer_size()
                        )
                        additional = actual_bound - working.reserved_bytes
                        policy.require_working_set(
                            additional,
                            error="statistics_table_exceeds_task_memory_budget",
                        )
                        working.grow(additional)
                        frame = table.to_pandas()
                        result = _validated_frame(frame, member=member)
                        yield result
                    finally:
                        table = frame = result = None


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
    calibration = table_group == "crypto_maker_calibration"
    if calibration and not callable(validate_tables):
        raise ValueError("statistics_business_validator_required")
    expected = {
        member.removesuffix(".parquet")
        for member in STATISTICS_TABLE_GROUPS[table_group]
    }
    if (calibration and not callable(tables)) or (
        not calibration and (not isinstance(tables, Mapping) or set(tables) != expected)
    ):
        raise ValueError("statistics_tables_invalid")

    def write_batches(staging, produce):
        policy = current_runtime_resource_policy()
        with handle_preserving_primary(ExitStack()) as output:
            writers = {}

            def append(table_name, frame):
                if table_name not in expected:
                    raise ValueError("statistics_tables_invalid")
                member = f"{table_name}.parquet"
                if member not in writers:
                    handle = output.enter_context(
                        handle_preserving_primary(
                            open(native_filesystem_path(staging / member), "xb")
                        )
                    )
                    writers[member] = output.enter_context(
                        parquet_table_writer(
                            handle,
                            parent=staging,
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
                validated = table = None
                with handle_preserving_primary(ExitStack()) as working:
                    try:
                        if policy is not None:
                            payload_bytes = pandas_frame_payload_bytes(frame)
                            validation_bytes = max(65536, 2 * payload_bytes)
                            policy.require_working_set(
                                validation_bytes
                                + _parquet_frame_bound(frame, key_columns=())
                                - payload_bytes,
                                error="statistics_validation_exceeds_task_memory_budget",
                            )
                            working.enter_context(
                                policy.lease(
                                    "artifacts.statistics.validation", validation_bytes
                                )
                            )
                        working.enter_context(
                            _parquet_frame_reservation(frame, borrowed_input=True)
                        )
                        validated = _validated_frame(frame, member=member)
                        table = pa.Table.from_pandas(
                            validated,
                            schema=_ARROW_SCHEMAS[member],
                            preserve_index=False,
                        )
                        writers[member](table)
                    finally:
                        validated = table = None

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
        artifact = _statistics_artifact(staging, envelope.instance_identity, members)
        validate_tables(artifact, envelope.payload())

    return write_semantic_artifact(
        Path(logical_root),
        envelope,
        write,
        staged_validator=validate if calibration else None,
    )


def _statistics_artifact(root, identity, members):
    by_path = {str(member.get("path")): member for member in members}
    if not any(
        set(by_path) == set(group) for group in STATISTICS_TABLE_GROUPS.values()
    ) or len(by_path) != len(members):
        raise ValueError("statistics_artifact_members_invalid")
    identities, row_counts = {}, {}
    for member in by_path:
        path = root / member
        physical = dataset_file_identity(path)
        verify_semantic_artifact_member(
            root, relative_path=member, expected=dict(by_path[member])
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
        root=root,
        identity=identity,
        _members=MappingProxyType(by_path),
        _identities=MappingProxyType(identities),
        _row_counts=MappingProxyType(row_counts),
    )


def read_statistics_artifact(
    instance_root: Path,
    *,
    expected_identity: str,
    validate_tables=None,
) -> StatisticsArtifact:
    envelope, members = read_semantic_artifact_record(
        Path(instance_root),
        expected_identity=expected_identity,
    )
    with runtime_resource_policy_scope(current_runtime_resource_policy()):
        artifact = _statistics_artifact(Path(instance_root), expected_identity, members)
        if set(artifact._members) == set(
            STATISTICS_TABLE_GROUPS["crypto_maker_calibration"]
        ):
            if not callable(validate_tables):
                raise ValueError("statistics_business_validator_required")
            validate_tables(artifact, envelope)
    return artifact


def _validated_frame(frame: pd.DataFrame, *, member: str) -> pd.DataFrame:
    if (
        type(frame) is not pd.DataFrame
        or tuple(frame.columns) != _TABLE_COLUMNS[member]
    ):
        raise ValueError(f"statistics_table_schema_invalid:{member}")
    result = frame.copy()
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
        and column not in _TIMESTAMP_COLUMNS.get(member, ())
    )
    for column in numeric_columns:
        result[column] = strict_finite_numeric_series(
            result[column], error="statistics_table_numeric_invalid", allow_null=True
        ).astype("float64")
    if result.empty and member not in _EMPTY_TABLE_MEMBERS:
        raise ValueError("statistics_table_empty")
    if member in STATISTICS_TABLE_GROUPS["research"]:
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
            or result.loc[~defined, measurements].notna().any().any()
            or result.loc[defined, "reason"].ne("").any()
            or result.loc[~defined, "reason"].str.strip().eq("").any()
        ):
            raise ValueError("statistics_table_status_value_invalid")
        for column in ("p_value", "fdr_adjusted_p_value", "white_p_value"):
            if column in result and not result[column].dropna().between(0, 1).all():
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
                or not result.missing_hypothesis_count.eq(
                    result.planned_hypothesis_count - result.defined_hypothesis_count
                ).all()
                or result.loc[applicable, "fdr_alpha"].isna().any()
                or not result.loc[applicable, "fdr_alpha"]
                .between(0, 1, inclusive="neither")
                .all()
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
                or not result.loc[defined & indicator, "value"].isin((0, 1)).all()
                or not result.loc[defined & probability, "value"].between(0, 1).all()
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
    if member == "model_losses.parquet":
        if (
            not result["loss"].dropna().between(-1, 1).all()
            or not result["common_date"].isin((0, 1)).all()
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
            invalid_value = (defined & ~(has_value ^ has_text)) | (
                ~defined & (has_value | has_text)
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
    if member in _PORTFOLIO_COLUMNS:
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
            "holding_factor_returns.parquet": ("entry_at", "factor_id"),
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
        if "end_at" in result and not result.entry_at.lt(result.end_at).all():
            raise ValueError("statistics_holding_interval_invalid")
        if "decision_at" in result and not result.decision_at.lt(result.entry_at).all():
            raise ValueError("statistics_holding_interval_invalid")
        if (
            member == "portfolio_attribution.parquet"
            and not result.component_type.isin(("additive", "detail")).all()
        ):
            raise ValueError("statistics_attribution_component_invalid")
    result.index = pd.RangeIndex(len(result))
    return result

"""Multi-table semantic artifact for formal research statistics."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from quant_research_platform.artifacts.outputs.parquet import (
    _artifact_frame_bound,
    _artifact_frame_reservation,
    _artifact_table_write_state,
    _artifact_writer_fixed_reservation,
)
from quant_research_platform.contracts.business_manifest import (
    BusinessManifestEnvelope,
)
from quant_research_platform.contracts.validation import (
    strict_finite_numeric_series,
    strict_integer_series,
)
from quant_research_platform.runtime.resources import (
    current_runtime_resource_policy,
    pandas_frame_payload_bytes,
    runtime_resource_policy_scope,
)
from quant_research_platform.storage.identity import native_filesystem_path
from quant_research_platform.storage.parquet import (
    _opened_parquet,
    _parquet_row_group_size,
    dataset_file_identity,
    parquet_decoded_working_set_bounds,
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

    @contextmanager
    def table(self, member: str):
        if member not in self._members:
            raise ValueError("statistics_table_member_invalid")
        path = self.root / member
        identity = dataset_file_identity(path)
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
    tables: Mapping[str, pd.DataFrame],
    table_group: str = "research",
) -> Path:
    if table_group not in STATISTICS_TABLE_GROUPS or set(tables) != {
        member.removesuffix(".parquet")
        for member in STATISTICS_TABLE_GROUPS[table_group]
    }:
        raise ValueError("statistics_tables_invalid")
    if table_group == "crypto_maker_calibration":
        _validate_crypto_maker_calibration(tables, envelope.payload())

    def write(staging: Path) -> None:
        policy = current_runtime_resource_policy()
        for table_name, frame in tables.items():
            member = f"{table_name}.parquet"
            state = ExitStack()
            validated = table = persisted = None
            primary: BaseException | None = None
            try:
                state.enter_context(
                    _artifact_writer_fixed_reservation(writer_count=1, index_bytes=0)
                )
                if policy is not None:
                    payload_bytes = pandas_frame_payload_bytes(frame)
                    validation_bytes = max(65536, 2 * payload_bytes)
                    policy.require_working_set(
                        validation_bytes
                        + _artifact_frame_bound(frame, key_columns=())
                        - payload_bytes,
                        error="statistics_validation_exceeds_task_memory_budget",
                    )
                    state.enter_context(
                        policy.lease(
                            "artifacts.statistics.validation",
                            validation_bytes,
                        )
                    )
                state.enter_context(
                    _artifact_frame_reservation(frame, borrowed_input=True)
                )
                validated = _validated_frame(frame, member=member)
                table = pa.Table.from_pandas(
                    validated, schema=_ARROW_SCHEMAS[member], preserve_index=False
                )
                persisted, native_state, batch_rows, _retained = (
                    _artifact_table_write_state(table)
                )
                state.enter_context(native_state)
                pq.write_table(
                    persisted,
                    native_filesystem_path(staging / member),
                    row_group_size=_parquet_row_group_size(persisted.num_rows),
                    write_batch_size=batch_rows,
                    max_rows_per_page=batch_rows,
                    compression="zstd",
                    use_dictionary=False,
                    write_statistics=False,
                    write_page_index=False,
                    write_page_checksum=False,
                )
            except BaseException as exc:
                primary = exc
            finally:
                validated = table = persisted = None
                try:
                    state.close()
                except BaseException as close_error:
                    if primary is None:
                        raise
                    primary.add_note(
                        "statistics table state release also failed: "
                        f"{type(close_error).__name__}"
                    )
                if primary is not None:
                    raise primary

    return write_semantic_artifact(Path(logical_root), envelope, write)


def read_statistics_artifact(
    instance_root: Path,
    *,
    expected_identity: str,
) -> StatisticsArtifact:
    envelope, members = read_semantic_artifact_record(
        Path(instance_root),
        expected_identity=expected_identity,
    )
    by_path = {str(member.get("path")): member for member in members}
    if not any(
        set(by_path) == set(group) for group in STATISTICS_TABLE_GROUPS.values()
    ) or len(by_path) != len(members):
        raise ValueError("statistics_artifact_members_invalid")
    artifact = StatisticsArtifact(
        root=Path(instance_root),
        identity=expected_identity,
        _members=MappingProxyType(by_path),
    )
    with runtime_resource_policy_scope(current_runtime_resource_policy()):
        for member in by_path:
            path = Path(instance_root) / member
            with _opened_parquet(
                path, expected_identity=dataset_file_identity(path)
            ) as (parquet, _identity, _handle):
                if parquet.schema_arrow != _ARROW_SCHEMAS[member]:
                    raise ValueError("statistics_table_schema_invalid")
                if (
                    parquet.metadata.num_rows == 0
                    and member not in _EMPTY_TABLE_MEMBERS
                ):
                    raise ValueError("statistics_table_empty")
        if set(by_path) == set(STATISTICS_TABLE_GROUPS["crypto_maker_calibration"]):
            with ExitStack() as reads:
                tables = {
                    member.removesuffix(".parquet"): reads.enter_context(
                        artifact.table(member)
                    )
                    for member in by_path
                }
                _validate_crypto_maker_calibration(tables, envelope)
    return artifact


def _validate_crypto_maker_calibration(tables, envelope):
    protocol = envelope["semantic_metadata"]["target_contract"]["protocol"]
    plan = protocol["research_plan"]
    minimum = plan["execution"]["maker"]["min_calibration_samples"]
    output = envelope["research_window"]
    folds = {
        fold["fold_id"]: fold
        for fold in plan["folds"]
        if pd.Timestamp(fold["development_test"]["start"])
        <= pd.Timestamp(output["end"])
        and pd.Timestamp(fold["development_test"]["end"])
        >= pd.Timestamp(output["start"])
    }
    from quant_research_platform.markets.implementations.crypto.execution import (
        MakerCalibration,
        _MakerOrder,
    )
    from quant_research_platform.markets.implementations.crypto.portfolio import (
        _maker_config,
    )
    from quant_research_platform.policy.lifecycle import closing_iterator

    reference_config = _maker_config(plan, "btc_usdt", calibration=True)

    samples = tables["crypto_maker_samples"]
    events = tables["crypto_maker_events"]
    summary = tables["crypto_maker_calibration"]
    keys = ["fold_id", "side", "sample_id"]
    policy = current_runtime_resource_policy()
    with policy.lease(
        "artifacts.crypto.calibration.validation",
        max(
            65536,
            4 * sum(pandas_frame_payload_bytes(frame) for frame in tables.values()),
        ),
    ):
        if (
            set(zip(summary.fold_id, summary.side, strict=True))
            != {(fold, side) for fold in folds for side in ("buy", "sell")}
            or summary.duplicated(["fold_id", "side"]).any()
            or samples.duplicated(keys).any()
            or events.duplicated(["fold_id", "event_id", "side", "level_index"]).any()
            or not samples.fold_id.isin(folds).all()
            or not samples.side.isin(("buy", "sell")).all()
            or not events.fold_id.isin(folds).all()
        ):
            raise ValueError("crypto_calibration_keys_invalid")
        if not samples.empty and (
            not np.isfinite(
                samples[
                    [
                        "price_prediction",
                        "observed_return",
                        "forecast_error",
                        "filled_quantity",
                        "quote_quantity",
                    ]
                ].to_numpy()
            ).all()
            or not (
                samples.forecast_error.to_numpy()
                == samples.observed_return.to_numpy()
                - samples.price_prediction.to_numpy()
            ).all()
            or not (
                (samples.filled_quantity >= 0)
                & (samples.filled_quantity <= samples.quote_quantity)
                & (samples.quote_quantity > 0)
            ).all()
            or samples.book_segment_id.eq("").any()
            or samples.trade_connection_id.eq("").any()
            or not samples.quote_quantity.eq(reference_config.order_quantity).all()
        ):
            raise ValueError("crypto_calibration_response_invalid")
        if (
            not events.event_kind.isin(("book", "trade", "boundary")).all()
            or not np.isfinite(events.quantity.to_numpy()).all()
            or events.quantity.lt(0).any()
        ):
            raise ValueError("crypto_calibration_event_invalid")
        for fold_id, fold in folds.items():
            sample = samples.loc[samples.fold_id.eq(fold_id)]
            if not sample.empty and not (
                sample.decision_at.between(
                    pd.Timestamp(fold["validation"]["start"]),
                    pd.Timestamp(fold["validation"]["end"]),
                ).all()
                and (
                    sample.prediction_end
                    == sample.decision_at
                    + pd.Timedelta(milliseconds=protocol["label_horizon_ms"])
                ).all()
                and (sample.label_available_at >= sample.prediction_end).all()
                and (
                    sample.label_available_at <= pd.Timestamp(fold["validation"]["end"])
                ).all()
            ):
                raise ValueError("crypto_calibration_outcome_not_mature")
            part = events.loc[events.fold_id.eq(fold_id)]
            headers = part.drop_duplicates("event_id")
            count = len(headers)
            if sample.empty:
                if count:
                    raise ValueError("crypto_calibration_event_coordinates_invalid")
                continue
            if (
                not np.array_equal(headers.event_id.to_numpy(), np.arange(count))
                or (part.event_id.diff().dropna() < 0).any()
                or (headers.event_at.diff().dropna() < pd.Timedelta(0)).any()
                or (headers.event_at > pd.Timestamp(fold["validation"]["end"])).any()
            ):
                raise ValueError("crypto_calibration_event_order_invalid")
            grouped = part.groupby("event_id", sort=False)
            if (
                not grouped[["event_at", "event_kind", "context_id"]]
                .nunique(dropna=False)
                .eq(1)
                .all()
                .all()
            ):
                raise ValueError("crypto_calibration_event_invalid")
            books = part.loc[part.event_kind.eq("book")]
            trades = part.loc[part.event_kind.eq("trade")]
            boundaries = part.loc[part.event_kind.eq("boundary")]
            if (
                not books.side.isin(("buy", "sell")).all()
                or books.context_id.eq("").any()
                or books.price_tick.le(0).any()
                or books.quantity.le(0).any()
                or not trades.side.isin(("buy", "sell")).all()
                or trades.context_id.eq("").any()
                or trades.price_tick.le(0).any()
                or not trades.level_index.eq(0).all()
                or not boundaries.side.eq("").all()
                or not boundaries.price_tick.eq(0).all()
                or not boundaries.quantity.eq(0).all()
                or not boundaries.level_index.eq(0).all()
                or not grouped.size()
                .loc[headers.loc[~headers.event_kind.eq("book"), "event_id"]]
                .eq(1)
                .all()
            ):
                raise ValueError("crypto_calibration_event_invalid")
            book_groups = books.groupby(["event_id", "side"], sort=False)
            if (
                not np.array_equal(
                    books.level_index.to_numpy(), book_groups.cumcount().to_numpy()
                )
                or not books.groupby("event_id").side.nunique().eq(2).all()
            ):
                raise ValueError("crypto_calibration_book_invalid")
            for side in ("buy", "sell"):
                levels = books.loc[books.side.eq(side)]
                identifiers = levels.event_id.to_numpy()
                ticks = levels.price_tick.to_numpy()
                same = identifiers[1:] == identifiers[:-1]
                invalid = (
                    ticks[1:] >= ticks[:-1]
                    if side == "buy"
                    else ticks[1:] <= ticks[:-1]
                )
                if (same & invalid).any():
                    raise ValueError("crypto_calibration_book_invalid")
            best = books.loc[books.level_index.eq(0)].pivot(
                index="event_id", columns="side", values="price_tick"
            )
            if not (best["buy"] < best["sell"]).all():
                raise ValueError("crypto_calibration_book_invalid")
            initial = sample.initial_book_event_id.to_numpy()
            starts, ends = sample.event_start.to_numpy(), sample.event_end.to_numpy()
            if (
                (initial < 0).any()
                or (initial >= count).any()
                or (initial >= starts).any()
                or (starts < 0).any()
                or (starts > ends).any()
                or (ends > count).any()
            ):
                raise ValueError("crypto_calibration_event_coordinates_invalid")
            kinds = headers.event_kind.to_numpy()
            times = headers.event_at.to_numpy().view(np.int64)
            decisions = sample.decision_at.to_numpy().view(np.int64)
            expiries = sample.prediction_end.to_numpy().view(np.int64)
            states = np.flatnonzero(np.isin(kinds, ("book", "boundary")))
            if (
                not (kinds[initial] == "book").all()
                or not np.array_equal(
                    headers.context_id.to_numpy()[initial],
                    sample.book_segment_id.to_numpy(),
                )
                or (times[initial] > decisions).any()
                or not np.array_equal(
                    states[np.searchsorted(states, starts, side="left") - 1], initial
                )
                or (times[starts - 1] > decisions).any()
                or (times[starts[starts < count]] <= decisions[starts < count]).any()
                or (times[ends[ends > starts] - 1] >= expiries[ends > starts]).any()
                or (times[ends[ends < count]] < expiries[ends < count]).any()
            ):
                raise ValueError("crypto_calibration_event_coordinates_invalid")
            covered = np.zeros(count + 1, dtype=np.int64)
            np.add.at(covered, starts, 1)
            np.add.at(covered, ends, -1)
            used = np.cumsum(covered[:-1]) > 0
            used[initial] = True
            if not used.all():
                raise ValueError("crypto_calibration_event_coordinates_invalid")
            paired_columns = [
                column
                for column in sample.columns
                if column not in ("side", "filled_quantity", "terminal_reason")
            ]
            buy = (
                sample.loc[sample.side.eq("buy"), paired_columns]
                .sort_values("sample_id")
                .reset_index(drop=True)
            )
            sell = (
                sample.loc[sample.side.eq("sell"), paired_columns]
                .sort_values("sample_id")
                .reset_index(drop=True)
            )
            if not buy.equals(sell):
                raise ValueError("crypto_calibration_paired_path_invalid")
            event_data = {
                column: part[column].to_numpy()
                for column in part.columns
                if column != "fold_id"
            }
            event_data["event_at"] = event_data["event_at"].view(np.int64)
            offsets = np.r_[
                np.flatnonzero(np.r_[True, np.diff(event_data["event_id"]) != 0]),
                len(part),
            ]
            sample_data = {
                column: sample[column].to_numpy() for column in sample.columns
            }
            sample_data["decision_at"] = sample_data["decision_at"].view(np.int64)
            calibration = MakerCalibration(
                sample_data, event_data, offsets, resource_policy=policy
            )
            try:
                for index, row in enumerate(sample.itertuples(index=False)):
                    state = calibration._book(int(row.initial_book_event_id))
                    order = _MakerOrder(
                        0,
                        row.side,
                        (state.bids if row.side == "buy" else state.asks)[0][0],
                        row.quote_quantity,
                        0.0,
                        0.0,
                        row.book_segment_id,
                        row.trade_connection_id,
                        "1",
                        row.decision_at.value,
                        row.prediction_end.value,
                        row.decision_at.value + reference_config.latency_ms * 1000000,
                    )
                    remaining = row.quote_quantity
                    with closing_iterator(
                        calibration._response(
                            index,
                            state=state,
                            decision_ns=row.decision_at.value,
                            new_order=order,
                            active_order=None,
                            commands=(),
                            lifetime_ns=protocol["label_horizon_ms"] * 1000000,
                            replace_quantity=row.quote_quantity,
                        )
                    ) as response:
                        while True:
                            try:
                                _elapsed, quantity, _price, _new = next(response)
                            except StopIteration as complete:
                                reason = complete.value
                                break
                            remaining -= quantity
                    if (
                        row.filled_quantity != row.quote_quantity - remaining
                        or row.terminal_reason != reason
                    ):
                        raise ValueError(
                            "crypto_calibration_reference_response_invalid"
                        )
            finally:
                calibration.close()
        for row in summary.itertuples(index=False):
            sample = samples.loc[
                samples.fold_id.eq(row.fold_id) & samples.side.eq(row.side)
            ]
            fold = folds[row.fold_id]
            if (
                row.minimum_samples != minimum
                or row.sample_count != len(sample)
                or row.zero_fill_count != int(sample.filled_quantity.eq(0).sum())
                or row.post_only_rejection_count
                != int(sample.terminal_reason.eq("post_only_crossing_rejected").sum())
                or row.status != ("DEFINED" if len(sample) >= minimum else "UNDEFINED")
                or row.reason
                != (
                    "" if len(sample) >= minimum else "insufficient_calibration_samples"
                )
                or not np.array_equal(
                    np.sort(sample.sample_id.to_numpy()), np.arange(len(sample))
                )
            ):
                raise ValueError("crypto_calibration_summary_invalid")


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

"""Spec-driven evaluation orchestration for compute kernels."""

from __future__ import annotations

import math
import sys
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager

import numpy as np
import pandas as pd

from quant_research_platform.compute.dataframe.cross_section import (
    GROUP_SPREAD_SUMMARY_COLUMNS,
    IC_SUMMARY_COLUMNS,
    _summarize_correlation_aggregates,
    compute_ic_and_grouped_label_values,
)
from quant_research_platform.compute.panels import ComputePanelSpec
from quant_research_platform.contracts.evaluations import (
    EvaluationDiagnostics,
    EvaluationSpec,
    strict_non_negative_count_series,
    strict_non_negative_count_value,
)
from quant_research_platform.contracts.filtering import required_filter_columns
from quant_research_platform.contracts.validation import strict_timestamp_series
from quant_research_platform.research._memory import (
    _selected_frame_payload_bytes,
    iter_complete_time_sized_frames,
)
from quant_research_platform.research.signals.materialization import (
    normalize_signal_frame,
)
from quant_research_platform.runtime.resources import (
    RuntimeResourcePolicy,
    pandas_frame_payload_bytes,
)

_EVALUATION_RECORD_WORKING_BYTES = 512
_EVALUATION_TIMESTAMP_FACT_BYTES = 128


class _EvaluationStreamingState:
    def __init__(
        self,
        *,
        spec: EvaluationSpec,
        n_groups: int,
        resource_policy: RuntimeResourcePolicy | None = None,
    ) -> None:
        self._spec = spec
        self._n_groups = n_groups
        self._summary_reservation = ExitStack()
        if resource_policy is not None:
            domain_size = max(1, len(spec.signals)) * max(
                1, len(spec.label_columns)
            )
            self._summary_reservation.enter_context(
                resource_policy.reserve(
                    "research.evaluation.summary.state",
                    64 * 1024 + domain_size * 4096,
                )
            )
        self._closed = False
        self._filter_columns_used: tuple[str, ...] = ()
        self._filter_definition = ""
        self._rows_input = 0
        self._rows_after_filter_mode = 0
        self._rows_after_signal_label_dropna = 0
        self._metric_rows = 0
        self._grouped_rows = 0
        self._timestamp_count = 0
        self._min_timestamp: pd.Timestamp | None = None
        self._max_timestamp: pd.Timestamp | None = None
        self._skipped_timestamp_count = 0
        self._ic: dict[tuple[str, str], dict[str, float | int]] = {}
        self._coverage: dict[tuple[str, str], dict[str, float | int]] = {}
        self._spread: dict[tuple[str, str], dict[str, float | int]] = {}

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._summary_reservation.close()

    def consume(
        self,
        metrics: pd.DataFrame,
        grouped_label_values: pd.DataFrame,
        metadata: Mapping[str, object],
    ) -> None:
        self._rows_input += int(metadata["rows_input"])
        self._rows_after_filter_mode += int(metadata["rows_after_filter_mode"])
        self._filter_columns_used = tuple(
            str(column) for column in metadata["filter_columns_used"]
        )
        self._filter_definition = str(metadata["filter_definition"])
        fact_count = strict_non_negative_count_value(metadata["timestamp_count"])
        if fact_count:
            fact_min = pd.Timestamp(metadata["min_timestamp"])
            fact_max = pd.Timestamp(metadata["max_timestamp"])
            self._timestamp_count += fact_count
            self._min_timestamp = (
                fact_min
                if self._min_timestamp is None
                else min(self._min_timestamp, fact_min)
            )
            self._max_timestamp = (
                fact_max
                if self._max_timestamp is None
                else max(self._max_timestamp, fact_max)
            )
        self._skipped_timestamp_count += strict_non_negative_count_value(
            metadata["skipped_timestamp_count"]
        )
        if not metrics.empty:
            self._metric_rows += len(metrics)
            for row in metrics.itertuples(index=False, name=None):
                (
                    _timestamp,
                    signal,
                    label,
                    ic,
                    rankic,
                    count,
                    _ic_reason,
                    _rankic_reason,
                ) = row
                count = int(count)
                self._rows_after_signal_label_dropna += count
                key = (str(signal), str(label))
                coverage = self._coverage.get(key)
                if coverage is None:
                    coverage = {
                        "rows": 0,
                        "n_obs_sum": 0,
                        "n_obs_min": math.inf,
                        "non_empty": 0,
                    }
                    self._coverage[key] = coverage
                coverage["rows"] += 1
                coverage["n_obs_sum"] += count
                coverage["n_obs_min"] = min(coverage["n_obs_min"], count)
                coverage["non_empty"] += int(count > 0)
                stats = self._ic.get(key)
                if stats is None:
                    stats = {
                        "ic_count": 0,
                        "ic_mean": 0.0,
                        "ic_m2": 0.0,
                        "ic_positive": 0,
                        "rankic_count": 0,
                        "rankic_mean": 0.0,
                        "rankic_m2": 0.0,
                        "rankic_positive": 0,
                    }
                    self._ic[key] = stats
                _add_correlation(stats, "ic", ic)
                _add_correlation(stats, "rankic", rankic)
        if not grouped_label_values.empty:
            self._grouped_rows += len(grouped_label_values)
            current_key: tuple[object, object, object] | None = None
            first_value: float | None = None
            last_value: float | None = None
            previous_value: float | None = None
            monotonicity_sum = 0.0
            monotonicity_count = 0

            def finish_group() -> None:
                if current_key is None:
                    return
                signal, label, _timestamp = current_key
                bottom = math.nan if first_value is None else first_value
                top = math.nan if last_value is None else last_value
                spread_value = top - bottom
                monotonicity = (
                    None
                    if monotonicity_count == 0
                    else monotonicity_sum / monotonicity_count
                )
                _add_group_spread_observation(
                    self._spread,
                    str(signal),
                    str(label),
                    top,
                    bottom,
                    spread_value,
                    float(spread_value > 0),
                    monotonicity,
                )

            for (
                timestamp,
                signal,
                label,
                _group,
                mean_label_value,
                _n_obs,
            ) in grouped_label_values.itertuples(index=False, name=None):
                key = (signal, label, timestamp)
                if key != current_key:
                    finish_group()
                    current_key = key
                    first_value = None
                    last_value = None
                    previous_value = None
                    monotonicity_sum = 0.0
                    monotonicity_count = 0
                if pd.isna(mean_label_value):
                    previous_value = None
                    continue
                value = float(mean_label_value)
                if first_value is None:
                    first_value = value
                if previous_value is not None:
                    monotonicity_sum += float(
                        np.sign(value - previous_value)
                    )
                    monotonicity_count += 1
                previous_value = value
                last_value = value
            finish_group()

    def ic_summary(self) -> pd.DataFrame:
        records: list[dict[str, object]] = []
        for (signal, label), stats in sorted(self._ic.items()):
            ic = _correlation_summary(stats, "ic")
            rankic = _correlation_summary(stats, "rankic")
            records.append(
                {
                    "signal": signal,
                    "label": label,
                    "mean_ic": ic["mean"],
                    "std_ic": ic["std"],
                    "icir": ic["ratio"],
                    "positive_ic_rate": ic["positive_rate"],
                    "valid_ic_count": ic["count"],
                    "mean_rankic": rankic["mean"],
                    "std_rankic": rankic["std"],
                    "rankicir": rankic["ratio"],
                    "positive_rankic_rate": rankic["positive_rate"],
                    "valid_rankic_count": rankic["count"],
                    "ic_reason": ic["reason"],
                    "icir_reason": ic["ratio_reason"],
                    "rankic_reason": rankic["reason"],
                    "rankicir_reason": rankic["ratio_reason"],
                }
            )
        return pd.DataFrame.from_records(records, columns=IC_SUMMARY_COLUMNS)

    def group_spread_summary(self) -> pd.DataFrame:
        records: list[dict[str, object]] = []
        for (signal, label), stats in sorted(self._spread.items()):
            count = int(stats["count"])
            monotonicity_count = int(stats["monotonicity_count"])
            records.append(
                {
                    "signal": signal,
                    "label": label,
                    "top_group_mean_label_value": float(stats["top_sum"]) / count,
                    "bottom_group_mean_label_value": float(
                        stats["bottom_sum"]
                    )
                    / count,
                    "top_bottom_label_value_spread_mean": float(
                        stats["spread_sum"]
                    )
                    / count,
                    "top_bottom_label_value_spread_hit_rate": float(
                        stats["hit_sum"]
                    )
                    / count,
                    "group_monotonicity_score": (
                        None
                        if monotonicity_count == 0
                        else float(stats["monotonicity_sum"])
                        / monotonicity_count
                    ),
                    "group_monotonicity_score_reason": (
                        "no_group_transitions"
                        if monotonicity_count == 0
                        else None
                    ),
                }
            )
        return pd.DataFrame.from_records(
            records,
            columns=GROUP_SPREAD_SUMMARY_COLUMNS,
        )

    def coverage_summary(self) -> pd.DataFrame:
        records = [
            {
                "signal": signal,
                "label": label,
                "mean_n_obs": float(stats["n_obs_sum"]) / int(stats["rows"]),
                "min_n_obs": int(stats["n_obs_min"]),
                "non_empty_period_count": int(stats["non_empty"]),
            }
            for (signal, label), stats in sorted(self._coverage.items())
        ]
        return pd.DataFrame.from_records(
            records,
            columns=[
                "signal",
                "label",
                "mean_n_obs",
                "min_n_obs",
                "non_empty_period_count",
            ],
        )

    def diagnostics(self) -> EvaluationDiagnostics:
        return EvaluationDiagnostics(
            filter_mode=self._spec.filter_mode,
            filter_columns_used=self._filter_columns_used,
            filter_definition=self._filter_definition,
            rows_input=self._rows_input,
            rows_after_filter_mode=self._rows_after_filter_mode,
            rows_after_signal_label_dropna=self._rows_after_signal_label_dropna,
            metric_rows=self._metric_rows,
            grouped_label_value_rows=self._grouped_rows,
            ic_summary_rows=len(self._ic),
            group_spread_summary_rows=len(self._spread),
            timestamp_count=self._timestamp_count,
            min_timestamp=(
                None
                if self._min_timestamp is None
                else self._min_timestamp.date().isoformat()
            ),
            max_timestamp=(
                None
                if self._max_timestamp is None
                else self._max_timestamp.date().isoformat()
            ),
            n_groups=self._n_groups,
            skipped_timestamp_count=self._skipped_timestamp_count,
        )


def _add_correlation(
    stats: dict[str, float | int],
    prefix: str,
    value: object,
) -> None:
    if pd.isna(value):
        return
    numeric = float(value)
    count = int(stats[f"{prefix}_count"]) + 1
    mean_key = f"{prefix}_mean"
    m2_key = f"{prefix}_m2"
    delta = numeric - float(stats[mean_key])
    mean = float(stats[mean_key]) + delta / count
    stats[mean_key] = mean
    stats[m2_key] += delta * (numeric - mean)
    stats[f"{prefix}_count"] = count
    stats[f"{prefix}_positive"] += int(numeric > 0)


def _correlation_summary(
    stats: Mapping[str, float | int],
    prefix: str,
) -> dict[str, float | int | str | None]:
    count = int(stats[f"{prefix}_count"])
    result = _summarize_correlation_aggregates(
        count=count,
        mean=float(stats[f"{prefix}_mean"]),
        m2=float(stats[f"{prefix}_m2"]),
        positive_count=int(stats[f"{prefix}_positive"]),
    )
    result["count"] = result["valid_count"]
    return result


def _add_group_spread_observation(
    spread_state: dict[tuple[str, str], dict[str, float | int]],
    signal: str,
    label: str,
    top: float,
    bottom: float,
    spread_value: float,
    hit_rate: float,
    monotonicity: float | None,
) -> None:
    key = (signal, label)
    spread = spread_state.get(key)
    if spread is None:
        spread = {
            "count": 0,
            "top_sum": 0.0,
            "bottom_sum": 0.0,
            "spread_sum": 0.0,
            "hit_sum": 0.0,
            "monotonicity_sum": 0.0,
            "monotonicity_count": 0,
        }
        spread_state[key] = spread
    spread["count"] += 1
    spread["top_sum"] += top
    spread["bottom_sum"] += bottom
    spread["spread_sum"] += spread_value
    spread["hit_sum"] += hit_rate
    if monotonicity is not None:
        spread["monotonicity_sum"] += monotonicity
        spread["monotonicity_count"] += 1


def _iter_evaluation_output_chunks(
    frames: Iterable[pd.DataFrame],
    *,
    spec: EvaluationSpec,
    signal_columns: tuple[str, ...],
    label_columns: tuple[str, ...],
    n_groups: int,
    asset_column: str,
    resource_policy: RuntimeResourcePolicy,
    source_columns_by_kind: Mapping[str, Sequence[str]] | None = None,
    summary_state: _EvaluationStreamingState | None = None,
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]]:
    """Yield bounded pairs and optionally consume them into research summaries."""

    from quant_research_platform.research.filtering import _filter_mode_scope

    for chunk in _iter_evaluation_work_frames(
        frames,
        spec=spec,
        signal_columns=signal_columns,
        label_columns=label_columns,
        asset_column=asset_column,
        source_columns_by_kind=source_columns_by_kind,
        resource_policy=resource_policy,
    ):
        with _filter_mode_scope(
            chunk,
            mode=spec.filter_mode,
            metadata_key="filter_mode",
        ) as (filtered, filter_metadata):
            try:
                for output in _iter_evaluation_filtered_chunks(
                    filtered,
                    filter_metadata,
                    spec=spec,
                    signal_columns=signal_columns,
                    label_columns=label_columns,
                    n_groups=n_groups,
                    asset_column=asset_column,
                    resource_policy=resource_policy,
                ):
                    if summary_state is not None:
                        summary_state.consume(*output)
                    yield output
            finally:
                filtered = None


def _iter_evaluation_filtered_chunks(
    filtered: pd.DataFrame,
    filter_metadata: Mapping[str, object],
    *,
    spec: EvaluationSpec,
    signal_columns: tuple[str, ...],
    label_columns: tuple[str, ...],
    n_groups: int,
    asset_column: str,
    resource_policy: RuntimeResourcePolicy,
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]]:
    rows_input = strict_non_negative_count_value(
        filter_metadata["rows_input"]
    )
    rows_after_filter = strict_non_negative_count_value(
        filter_metadata["rows_after_filter_mode"]
    )
    compute_specs: list[ComputePanelSpec] = []
    compute_columns_by_label: list[tuple[str, ...]] = []
    for label_column in label_columns:
        compute_spec = ComputePanelSpec(
            signal_columns=signal_columns,
            label_column=label_column,
            timestamp_column="timestamp",
            asset_column=asset_column,
        )
        compute_specs.append(compute_spec)
        compute_columns_by_label.append(
            (
                compute_spec.timestamp_column,
                compute_spec.asset_column,
                *compute_spec.signal_columns,
                compute_spec.label_column,
            )
        )
    metadata_emitted = False
    for compute_frame in _iter_evaluation_compute_batches(
        filtered,
        compute_columns_by_label=tuple(compute_columns_by_label),
        signal_count=len(signal_columns),
        n_groups=n_groups,
        resource_policy=resource_policy,
    ):
        batch_timestamps = strict_timestamp_series(
            compute_frame["timestamp"],
            error="invalid_evaluation_timestamp_values",
        )
        batch_timestamp_count = int(batch_timestamps.nunique())
        batch_min_timestamp = (
            pd.Timestamp(batch_timestamps.min()).date().isoformat()
            if batch_timestamp_count
            else None
        )
        batch_max_timestamp = (
            pd.Timestamp(batch_timestamps.max()).date().isoformat()
            if batch_timestamp_count
            else None
        )
        with _evaluation_batch_diagnostics_scope(
            resource_policy,
            timestamp_count=batch_timestamp_count,
        ) as skipped_timestamps:
            yield from _iter_evaluation_batch_outputs(
                compute_frame,
                compute_specs=compute_specs,
                compute_columns_by_label=tuple(compute_columns_by_label),
                signal_count=len(signal_columns),
                n_groups=n_groups,
                resource_policy=resource_policy,
                filter_metadata=filter_metadata,
                rows_input=rows_input,
                rows_after_filter=rows_after_filter,
                metadata_emitted=metadata_emitted,
                batch_timestamp_count=batch_timestamp_count,
                batch_min_timestamp=batch_min_timestamp,
                batch_max_timestamp=batch_max_timestamp,
                skipped_timestamps=skipped_timestamps,
            )
        metadata_emitted = True
        del compute_frame


@contextmanager
def _evaluation_batch_diagnostics_scope(
    resource_policy: RuntimeResourcePolicy,
    *,
    timestamp_count: int,
) -> Iterator[set[pd.Timestamp]]:
    lease = resource_policy.lease(
        "research.evaluation.diagnostics.batch",
        max(1, timestamp_count) * _EVALUATION_TIMESTAMP_FACT_BYTES,
    )
    skipped_timestamps: set[pd.Timestamp] = set()
    try:
        yield skipped_timestamps
    finally:
        primary = sys.exception()
        try:
            lease.close()
        except BaseException as close_error:
            if primary is None:
                raise
            primary.add_note(
                "evaluation diagnostics lease release also failed: "
                f"{type(close_error).__name__}"
            )


def _iter_evaluation_batch_outputs(
    compute_frame: pd.DataFrame,
    *,
    compute_specs: Sequence[ComputePanelSpec],
    compute_columns_by_label: tuple[tuple[str, ...], ...],
    signal_count: int,
    n_groups: int,
    resource_policy: RuntimeResourcePolicy,
    filter_metadata: Mapping[str, object],
    rows_input: int,
    rows_after_filter: int,
    metadata_emitted: bool,
    batch_timestamp_count: int,
    batch_min_timestamp: str | None,
    batch_max_timestamp: str | None,
    skipped_timestamps: set[pd.Timestamp],
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]]:
    label_metrics: list[pd.DataFrame] = []
    label_grouped: list[pd.DataFrame] = []
    label_leases = []
    concat_lease = None
    try:
        for compute_spec, compute_columns in zip(
            compute_specs, compute_columns_by_label, strict=True
        ):
            kernel_bytes = _evaluation_kernel_bytes(
                compute_frame,
                compute_columns=compute_columns,
                signal_count=signal_count,
                n_groups=n_groups,
            )
            kernel_lease = resource_policy.lease(
                "research.evaluation.kernel",
                kernel_bytes,
            )
            try:
                usable = compute_frame.loc[:, list(compute_columns)].copy()
                metrics, grouped_label_values = (
                    compute_ic_and_grouped_label_values(
                        usable,
                        compute_spec,
                        n_groups=n_groups,
                    )
                )
                del usable
                n_obs = strict_non_negative_count_series(metrics["n_obs"])
                skipped_timestamps.update(
                    pd.Timestamp(timestamp)
                    for timestamp in metrics.loc[
                        n_obs.lt(2), "timestamp"
                    ].drop_duplicates()
                )
                output_bytes = pandas_frame_payload_bytes(
                    metrics
                ) + pandas_frame_payload_bytes(grouped_label_values)
                if output_bytes > kernel_bytes:
                    raise ValueError(
                        "research_evaluation_working_set_exceeds_task_memory_budget"
                    )
                kernel_lease.shrink_to(output_bytes)
                kernel_lease.transfer("research.evaluation.output")
                label_metrics.append(metrics)
                label_grouped.append(grouped_label_values)
                label_leases.append(kernel_lease)
                kernel_lease = None
            finally:
                primary = sys.exception()
                if kernel_lease is not None:
                    try:
                        kernel_lease.close()
                    except BaseException as close_error:
                        if primary is None:
                            raise
                        primary.add_note(
                            "evaluation kernel lease release also failed: "
                            f"{type(close_error).__name__}"
                        )

        if len(label_metrics) == 1:
            metrics = label_metrics[0]
            grouped_label_values = label_grouped[0]
        else:
            retained_output_bytes = sum(
                pandas_frame_payload_bytes(frame)
                for frame in (*label_metrics, *label_grouped)
            )
            concat_sort_bound = max(64 * 1024, retained_output_bytes * 2)
            concat_lease = resource_policy.lease(
                "research.evaluation.concat",
                concat_sort_bound,
            )
            metrics = pd.concat(label_metrics, ignore_index=True).sort_values(
                "timestamp",
                kind="mergesort",
                ignore_index=True,
            )
            grouped_label_values = pd.concat(
                label_grouped,
                ignore_index=True,
            ).sort_values(
                "timestamp",
                kind="mergesort",
                ignore_index=True,
            )
            combined_output_bytes = pandas_frame_payload_bytes(
                metrics
            ) + pandas_frame_payload_bytes(grouped_label_values)
            if combined_output_bytes > concat_sort_bound:
                raise ValueError(
                    "research_evaluation_working_set_exceeds_task_memory_budget"
                )
            concat_lease.shrink_to(combined_output_bytes)
            concat_lease.transfer("research.evaluation.output")
            for lease in label_leases:
                lease.close()
            label_leases.clear()
        label_metrics.clear()
        label_grouped.clear()
        metadata = {
            "rows_input": rows_input if not metadata_emitted else 0,
            "rows_after_filter_mode": rows_after_filter
            if not metadata_emitted
            else 0,
            "filter_columns_used": list(filter_metadata["filter_columns_used"]),
            "filter_definition": str(filter_metadata["filter_definition"]),
            "timestamp_count": batch_timestamp_count,
            "min_timestamp": batch_min_timestamp,
            "max_timestamp": batch_max_timestamp,
            "skipped_timestamp_count": len(skipped_timestamps),
        }
        yield metrics, grouped_label_values, metadata
    finally:
        primary = sys.exception()
        if concat_lease is not None:
            try:
                concat_lease.close()
            except BaseException as close_error:
                if primary is None:
                    raise
                primary.add_note(
                    "evaluation concat lease release also failed: "
                    f"{type(close_error).__name__}"
                )
        for lease in label_leases:
            try:
                lease.close()
            except BaseException as close_error:
                if primary is None:
                    raise
                primary.add_note(
                    "evaluation output lease release also failed: "
                    f"{type(close_error).__name__}"
                )


def _evaluation_kernel_bytes(
    frame: pd.DataFrame,
    *,
    compute_columns: tuple[str, ...],
    signal_count: int,
    n_groups: int,
) -> int:
    row_count = len(frame)
    index_bytes = int(frame.index.memory_usage(deep=True))
    usable_bytes = _selected_frame_payload_bytes(frame, compute_columns)
    timestamp_count = int(frame["timestamp"].nunique())
    output_record_count = timestamp_count * signal_count * (1 + n_groups)
    return int(
        usable_bytes
        + usable_bytes
        + usable_bytes
        + (index_bytes + row_count)
        + (index_bytes + row_count)
        + row_count * (3 * 1 + 10 * 8)
        + output_record_count * _EVALUATION_RECORD_WORKING_BYTES
    )


def _evaluation_output_bound(
    frame: pd.DataFrame,
    *,
    signal_count: int,
    n_groups: int,
) -> int:
    return int(
        frame["timestamp"].nunique()
        * signal_count
        * (1 + n_groups)
        * _EVALUATION_RECORD_WORKING_BYTES
    )


def _evaluation_batch_peak_bytes(
    frame: pd.DataFrame,
    *,
    compute_columns_by_label: tuple[tuple[str, ...], ...],
    signal_count: int,
    n_groups: int,
) -> int:
    kernel_bounds = [
        _evaluation_kernel_bytes(
            frame,
            compute_columns=compute_columns,
            signal_count=signal_count,
            n_groups=n_groups,
        )
        for compute_columns in compute_columns_by_label
    ]
    retained_output_bound = sum(
        _evaluation_output_bound(
            frame,
            signal_count=signal_count,
            n_groups=n_groups,
        )
        for _ in compute_columns_by_label
    )
    concat_sort_bound = max(64 * 1024, retained_output_bound * 2)
    return (
        max(kernel_bounds)
        + retained_output_bound
        + concat_sort_bound
        + max(1, int(frame["timestamp"].nunique()))
        * _EVALUATION_TIMESTAMP_FACT_BYTES
    )


def _iter_evaluation_compute_batches(
    frame: pd.DataFrame,
    *,
    compute_columns_by_label: tuple[tuple[str, ...], ...],
    signal_count: int,
    n_groups: int,
    resource_policy: RuntimeResourcePolicy,
) -> Iterator[pd.DataFrame]:
    """Yield greedy complete-timestamp batches that fit the live kernel budget."""

    if frame.empty:
        yield frame
        return
    timestamps = strict_timestamp_series(
        frame["timestamp"],
        error="invalid_evaluation_timestamp_values",
    )
    values = timestamps.to_numpy(copy=False)
    stops = np.flatnonzero(values[1:] != values[:-1]) + 1
    start = 0
    session_index = 0
    while start < len(frame):
        stop = start
        planned_bytes = 0
        while stop < len(frame):
            session_stop = (
                int(stops[session_index])
                if session_index < len(stops)
                else len(frame)
            )
            session = frame.iloc[stop:session_stop]
            session_bytes = _evaluation_batch_peak_bytes(
                session,
                compute_columns_by_label=compute_columns_by_label,
                signal_count=signal_count,
                n_groups=n_groups,
            )
            candidate_bytes = planned_bytes + session_bytes
            if (
                resource_policy.used_memory_bytes + candidate_bytes
                > resource_policy.memory_budget_bytes
            ):
                if stop == start:
                    with resource_policy.reserve(
                        "research.evaluation.kernel",
                        candidate_bytes,
                    ):
                        pass
                break
            planned_bytes = candidate_bytes
            stop = session_stop
            session_index += 1
        if stop == start:
            raise MemoryError("runtime_memory_budget_exceeded")
        yield frame.iloc[start:stop]
        start = stop


def _iter_evaluation_work_frames(
    frames: Iterable[pd.DataFrame],
    *,
    spec: EvaluationSpec,
    signal_columns: tuple[str, ...],
    label_columns: tuple[str, ...],
    asset_column: str,
    resource_policy: RuntimeResourcePolicy,
    source_columns_by_kind: Mapping[str, Sequence[str]] | None = None,
) -> Iterator[pd.DataFrame]:
    required = {
        "timestamp",
        asset_column,
        *(signal.source_column for signal in spec.signals),
        *label_columns,
    }
    work_columns = tuple(
        dict.fromkeys(
            (
                "timestamp",
                asset_column,
                *(signal.source_column for signal in spec.signals),
                *label_columns,
                *required_filter_columns(spec.filter_mode),
            )
        )
    )
    for frame in iter_complete_time_sized_frames(
        iter(frames),
        timestamp_column="timestamp",
        resource_policy=resource_policy,
    ):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"missing_evaluation_columns: {', '.join(missing)}")
        selected = [column for column in work_columns if column in frame.columns]
        projection_bytes = _selected_frame_payload_bytes(frame, selected)
        selected_set = set(selected)
        added_signal_bytes = 8 * len(frame) * sum(
            signal.signal_id not in selected_set for signal in spec.signals
        )
        normalized_bound = projection_bytes + added_signal_bytes
        normalized_lease = resource_policy.lease(
            "research.evaluation.projection",
            normalized_bound,
        )
        try:
            normalized = normalize_signal_frame(
                frame.loc[:, selected],
                signals=spec.signals,
                timestamp_column="timestamp",
                source_columns_by_kind=source_columns_by_kind,
            )
            normalized_bytes = pandas_frame_payload_bytes(normalized)
            if normalized_bytes > normalized_bound:
                raise ValueError(
                    "research_evaluation_working_set_exceeds_task_memory_budget"
                )
            normalized_lease.shrink_to(normalized_bytes)
            normalized_lease.transfer("research.evaluation.input")
            yield normalized
        finally:
            primary = sys.exception()
            try:
                normalized_lease.close()
            except BaseException as close_error:
                if primary is None:
                    raise
                primary.add_note(
                    "evaluation input lease release also failed: "
                    f"{type(close_error).__name__}"
                )

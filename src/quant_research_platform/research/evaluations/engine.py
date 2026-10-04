"""Spec-driven evaluation orchestration for compute kernels."""

from __future__ import annotations

import math
import sys
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np
import pandas as pd

from quant_research_platform.compute.dataframe.cross_section import (
    _add_correlation_aggregate,
    _CorrelationMoments,
    _summarize_correlation_aggregates,
    compute_ic_and_grouped_label_values,
)
from quant_research_platform.compute.panels import ComputePanelSpec
from quant_research_platform.contracts.evaluations import (
    EVALUATION_GROUP_SPREAD_SUMMARY_COLUMNS,
    EVALUATION_GROUPED_LABEL_VALUE_COLUMNS,
    EVALUATION_IC_SUMMARY_COLUMNS,
    EVALUATION_METRICS_COLUMNS,
    EvaluationDiagnostics,
    EvaluationSpec,
    strict_non_negative_count_series,
    strict_non_negative_count_value,
)
from quant_research_platform.contracts.filtering import (
    filter_definition,
    required_filter_columns,
)
from quant_research_platform.contracts.time import time_value_isoformat
from quant_research_platform.contracts.validation import strict_timestamp_series
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.research.signals.materialization import (
    _signal_normalization_working_bytes,
    normalize_signal_frame,
)
from quant_research_platform.runtime.dataframes import (
    _selected_frame_payload_bytes,
    iter_complete_time_sized_frames,
)
from quant_research_platform.runtime.resources import (
    MemoryLease,
    OwnedValue,
    RuntimeResourcePolicy,
    _close_lease_preserving_primary,
    pandas_frame_payload_bytes,
)

_EVALUATION_RECORD_WORKING_BYTES = 512
_EVALUATION_TIMESTAMP_FACT_BYTES = 128

type _EvaluationOutputChunk = tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, object],
]


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
            domain_size = max(1, len(spec.signals)) * max(1, len(spec.label_columns))
            self._summary_reservation.enter_context(
                resource_policy.lease(
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
        self._ic: dict[tuple[str, str], dict[str, int | Fraction]] = {}
        self._coverage: dict[tuple[str, str], dict[str, float | int]] = {}
        self._spread: dict[tuple[str, str], dict[str, float | int | Fraction]] = {}

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
                        "ic_mean": Fraction(),
                        "ic_m2": Fraction(),
                        "ic_positive": 0,
                        "rankic_count": 0,
                        "rankic_mean": Fraction(),
                        "rankic_m2": Fraction(),
                        "rankic_positive": 0,
                    }
                    self._ic[key] = stats
                _add_correlation_aggregate(stats, "ic", ic)
                _add_correlation_aggregate(stats, "rankic", rankic)
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
                    float(top > bottom),
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
                        (value > previous_value) - (value < previous_value)
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
            if self._spec.evaluation_axis == "single_asset_time_series":
                rankic = {
                    "mean": np.nan,
                    "std": np.nan,
                    "ratio": np.nan,
                    "positive_rate": np.nan,
                    "count": 0,
                    "reason": "not_applicable",
                    "ratio_reason": "not_applicable",
                }
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
        return pd.DataFrame.from_records(records, columns=EVALUATION_IC_SUMMARY_COLUMNS)

    def group_spread_summary(self) -> pd.DataFrame:
        records: list[dict[str, object]] = []
        for (signal, label), stats in sorted(self._spread.items()):
            count = int(stats["count"])
            monotonicity_count = int(stats["monotonicity_count"])
            try:
                spread_mean = float((stats["top_sum"] - stats["bottom_sum"]) / count)
            except OverflowError as error:
                raise ValueError("invalid_evaluation_numeric_values") from error
            records.append(
                {
                    "signal": signal,
                    "label": label,
                    "top_group_mean_label_value": float(stats["top_sum"] / count),
                    "bottom_group_mean_label_value": float(stats["bottom_sum"] / count),
                    "top_bottom_label_value_spread_mean": spread_mean,
                    "top_bottom_label_value_spread_hit_rate": float(stats["hit_sum"])
                    / count,
                    "group_monotonicity_score": (
                        None
                        if monotonicity_count == 0
                        else float(stats["monotonicity_sum"]) / monotonicity_count
                    ),
                    "group_monotonicity_score_reason": (
                        "no_group_transitions" if monotonicity_count == 0 else None
                    ),
                }
            )
        return pd.DataFrame.from_records(
            records,
            columns=EVALUATION_GROUP_SPREAD_SUMMARY_COLUMNS,
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
                else time_value_isoformat(
                    self._min_timestamp,
                    preserve_datetime=True,
                )
            ),
            max_timestamp=(
                None
                if self._max_timestamp is None
                else time_value_isoformat(
                    self._max_timestamp,
                    preserve_datetime=True,
                )
            ),
            n_groups=self._n_groups,
            skipped_timestamp_count=self._skipped_timestamp_count,
        )


@dataclass(frozen=True)
class EvaluationResult:
    output_chunks: Iterable[object]
    _summary_factory: Callable[[], tuple[object, ...]] = field(repr=False)
    _close: Callable[[], None] = field(repr=False, compare=False)

    def summaries(self) -> tuple[object, ...]:
        return self._summary_factory()

    def close(self) -> None:
        self._close()

    def __enter__(self) -> EvaluationResult:
        return self

    def __exit__(
        self,
        _exception_type: object,
        primary: BaseException | None,
        _traceback: object,
    ) -> None:
        try:
            self.close()
        except BaseException as cleanup_error:
            if primary is None:
                raise
            primary.add_note(
                f"evaluation result cleanup also failed: {type(cleanup_error).__name__}"
            )


def build_evaluation_result_from_frames(
    frames: Iterable[pd.DataFrame],
    *,
    spec: EvaluationSpec,
    signal_columns: tuple[str, ...],
    label_columns: tuple[str, ...],
    n_groups: int,
    asset_column: str,
    resource_policy: RuntimeResourcePolicy,
    source_columns_by_kind: Mapping[str, Sequence[str]] | None = None,
) -> EvaluationResult:
    state = _EvaluationStreamingState(
        spec=spec,
        n_groups=n_groups,
        resource_policy=resource_policy,
    )
    output_chunks = (
        _iter_single_asset_time_series_output_chunks(
            frames,
            spec=spec,
            signal_columns=signal_columns,
            label_columns=label_columns,
            asset_column=asset_column,
            source_columns_by_kind=source_columns_by_kind,
            resource_policy=resource_policy,
            summary_state=state,
        )
        if spec.evaluation_axis == "single_asset_time_series"
        else _iter_evaluation_output_chunks(
            frames,
            spec=spec,
            signal_columns=signal_columns,
            label_columns=label_columns,
            n_groups=n_groups,
            asset_column=asset_column,
            source_columns_by_kind=source_columns_by_kind,
            resource_policy=resource_policy,
            summary_state=state,
        )
    )

    def summaries() -> tuple[object, ...]:
        return (
            state.diagnostics(),
            state.ic_summary(),
            state.group_spread_summary(),
            state.coverage_summary(),
        )

    def close() -> None:
        failure: BaseException | None = None
        close_chunks = getattr(output_chunks, "close", None)
        if close_chunks is not None:
            try:
                close_chunks()
            except BaseException as cleanup_error:
                failure = cleanup_error
        try:
            state.close()
        except BaseException as cleanup_error:
            if failure is None:
                failure = cleanup_error
            else:
                failure.add_note(
                    "evaluation summary reservation release also failed: "
                    f"{type(cleanup_error).__name__}"
                )
        if failure is not None:
            raise failure

    return EvaluationResult(output_chunks, summaries, close)


def _correlation_summary(
    stats: Mapping[str, int | Fraction],
    prefix: str,
) -> dict[str, float | int | str | None]:
    count = int(stats[f"{prefix}_count"])
    result = _summarize_correlation_aggregates(
        count=count,
        mean=stats[f"{prefix}_mean"],
        m2=stats[f"{prefix}_m2"],
        positive_count=int(stats[f"{prefix}_positive"]),
    )
    result["count"] = result["valid_count"]
    return result


def _add_group_spread_observation(
    spread_state: dict[tuple[str, str], dict[str, float | int | Fraction]],
    signal: str,
    label: str,
    top: float,
    bottom: float,
    hit_rate: float,
    monotonicity: float | None,
) -> None:
    key = (signal, label)
    spread = spread_state.get(key)
    if spread is None:
        spread = {
            "count": 0,
            "top_sum": Fraction(),
            "bottom_sum": Fraction(),
            "hit_sum": 0.0,
            "monotonicity_sum": 0.0,
            "monotonicity_count": 0,
        }
        spread_state[key] = spread
    spread["count"] += 1
    spread["top_sum"] += Fraction(top)
    spread["bottom_sum"] += Fraction(bottom)
    spread["hit_sum"] += hit_rate
    if monotonicity is not None:
        spread["monotonicity_sum"] += monotonicity
        spread["monotonicity_count"] += 1


def _iter_single_asset_time_series_output_chunks(
    frames: Iterable[pd.DataFrame],
    *,
    spec: EvaluationSpec,
    signal_columns: tuple[str, ...],
    label_columns: tuple[str, ...],
    asset_column: str,
    resource_policy: RuntimeResourcePolicy,
    source_columns_by_kind: Mapping[str, Sequence[str]] | None = None,
    summary_state: _EvaluationStreamingState | None = None,
) -> Iterator[OwnedValue[_EvaluationOutputChunk]]:
    """Yield one bounded across-time correlation row per signal-label pair."""

    from quant_research_platform.research.filtering import _filter_mode_scope

    pair_count = len(signal_columns) * len(label_columns)
    numeric_columns = (*signal_columns, *label_columns)
    numeric_column_count = len(numeric_columns)
    state_lease = resource_policy.lease(
        "research.evaluation.time_series.state",
        max(
            64 * 1024,
            # Five exact sums of binary64 values/products, with int64 row counts,
            # fit in 4 KiB including the largest products and count growth.
            pair_count * 4096,
        ),
    )
    stats = {
        (signal, label): _CorrelationMoments()
        for signal in signal_columns
        for label in label_columns
    }
    rows_input = 0
    rows_after_filter = 0
    filter_columns_used = tuple(required_filter_columns(spec.filter_mode))
    filter_definition_value = filter_definition(spec.filter_mode)
    asset: object | None = None
    timestamp_count = 0
    minimum_timestamp: pd.Timestamp | None = None
    maximum_timestamp: pd.Timestamp | None = None
    output_lease: MemoryLease | None = None
    owned_output: OwnedValue[_EvaluationOutputChunk] | None = None

    def consume_values(values: np.ndarray) -> None:
        label_offset = len(signal_columns)
        for signal_index, signal in enumerate(signal_columns):
            left = values[:, signal_index]
            for label_index, label in enumerate(label_columns):
                right = values[:, label_offset + label_index]
                valid = np.isfinite(left) & np.isfinite(right)
                moments = stats[(signal, label)]
                for x, y in zip(left[valid], right[valid], strict=True):
                    moments.update(float(x), float(y))

    try:
        for chunk in _iter_evaluation_work_frames(
            frames,
            spec=spec,
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
                    rows_input += strict_non_negative_count_value(
                        filter_metadata["rows_input"]
                    )
                    rows_after_filter += strict_non_negative_count_value(
                        filter_metadata["rows_after_filter_mode"]
                    )
                    filter_columns_used = tuple(
                        str(column) for column in filter_metadata["filter_columns_used"]
                    )
                    filter_definition_value = str(filter_metadata["filter_definition"])
                    if filtered.empty:
                        continue
                    kernel_lease = resource_policy.lease(
                        "research.evaluation.time_series.kernel",
                        max(
                            64 * 1024,
                            len(filtered) * (numeric_column_count * 8 + 32),
                        ),
                    )
                    try:
                        assets = filtered[asset_column]
                        unique_assets = assets.drop_duplicates()
                        if len(unique_assets) != 1 or bool(assets.isna().any()):
                            raise ValueError(
                                "single_asset_time_series_evaluation_requires_one_asset"
                            )
                        chunk_asset = unique_assets.iloc[0]
                        if asset is None:
                            asset = chunk_asset
                        elif chunk_asset != asset:
                            raise ValueError(
                                "single_asset_time_series_evaluation_requires_one_asset"
                            )
                        timestamps = strict_timestamp_series(
                            filtered["timestamp"],
                            error="invalid_evaluation_timestamp_values",
                        )
                        if bool(timestamps.duplicated().any()):
                            raise ValueError(
                                "single_asset_time_series_evaluation_duplicate_timestamp"
                            )
                        chunk_minimum = pd.Timestamp(timestamps.min())
                        chunk_maximum = pd.Timestamp(timestamps.max())
                        if (
                            maximum_timestamp is not None
                            and chunk_minimum <= maximum_timestamp
                        ):
                            raise ValueError(
                                "single_asset_time_series_evaluation_time_non_monotonic"
                            )
                        timestamp_count += len(timestamps)
                        minimum_timestamp = (
                            chunk_minimum
                            if minimum_timestamp is None
                            else min(minimum_timestamp, chunk_minimum)
                        )
                        maximum_timestamp = chunk_maximum
                        values = filtered.loc[:, list(numeric_columns)].to_numpy(
                            dtype="float64",
                            copy=True,
                        )
                        consume_values(values)
                    finally:
                        values = timestamps = assets = unique_assets = None
                        _close_lease_preserving_primary(
                            kernel_lease,
                            owner="research.evaluation.time_series.kernel",
                        )
                finally:
                    filtered = chunk = None
        output_bound = max(64 * 1024, pair_count * 2048)
        output_lease = resource_policy.lease(
            "research.evaluation.output",
            output_bound,
        )
        metrics = _time_series_metrics_frame(
            stats if maximum_timestamp is not None else {},
            timestamp=maximum_timestamp,
        )
        grouped_label_values = _empty_grouped_label_values()
        output_bytes = pandas_frame_payload_bytes(metrics) + pandas_frame_payload_bytes(
            grouped_label_values
        )
        if output_bytes > output_bound:
            raise ValueError(
                "research_evaluation_working_set_exceeds_task_memory_budget"
            )
        output_lease.shrink_to(output_bytes)
        metadata = {
            "rows_input": rows_input,
            "rows_after_filter_mode": rows_after_filter,
            "filter_columns_used": list(filter_columns_used),
            "filter_definition": filter_definition_value,
            "timestamp_count": timestamp_count,
            "min_timestamp": time_value_isoformat(
                minimum_timestamp,
                preserve_datetime=True,
            )
            if minimum_timestamp is not None
            else None,
            "max_timestamp": time_value_isoformat(
                maximum_timestamp,
                preserve_datetime=True,
            )
            if maximum_timestamp is not None
            else None,
            "skipped_timestamp_count": 0,
        }
        owned_output = OwnedValue(
            (metrics, grouped_label_values, metadata),
            output_lease,
        )
        output_lease = None
        if summary_state is not None:
            summary_state.consume(*owned_output.value)
        try:
            yield owned_output
        finally:
            if owned_output.lease.active:
                _close_lease_preserving_primary(
                    owned_output.lease,
                    owner="research.evaluation.output",
                )
    finally:
        if output_lease is not None:
            _close_lease_preserving_primary(
                output_lease,
                owner="research.evaluation.output",
            )
        _close_lease_preserving_primary(
            state_lease,
            owner="research.evaluation.time_series.state",
        )


def _time_series_metrics_frame(
    stats: Mapping[tuple[str, str], _CorrelationMoments],
    *,
    timestamp: pd.Timestamp | None,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for (signal, label), values in sorted(stats.items()):
        correlation, reason = values.result()
        records.append(
            {
                "timestamp": timestamp,
                "signal": signal,
                "label": label,
                "ic": correlation,
                "rankic": None,
                "n_obs": values.count,
                "ic_reason": reason,
                "rankic_reason": "not_applicable",
            }
        )
    metrics = pd.DataFrame.from_records(records, columns=EVALUATION_METRICS_COLUMNS)
    metrics["timestamp"] = metrics["timestamp"].astype("datetime64[ns]")
    for column in ("signal", "label", "ic_reason", "rankic_reason"):
        metrics[column] = metrics[column].astype("string")
    for column in ("ic", "rankic"):
        metrics[column] = metrics[column].astype("float64")
    metrics["n_obs"] = metrics["n_obs"].astype("int64")
    return metrics


def _empty_grouped_label_values() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.Series(dtype="datetime64[ns]"),
            "signal": pd.Series(dtype="string"),
            "label": pd.Series(dtype="string"),
            "group": pd.Series(dtype="int64"),
            "mean_label_value": pd.Series(dtype="float64"),
            "n_obs": pd.Series(dtype="int64"),
        },
        columns=EVALUATION_GROUPED_LABEL_VALUE_COLUMNS,
    )


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
) -> Iterator[OwnedValue[_EvaluationOutputChunk]]:
    """Yield bounded pairs and optionally consume them into research summaries."""

    from quant_research_platform.research.filtering import _filter_mode_scope

    work_frames = _iter_evaluation_work_frames(
        frames,
        spec=spec,
        label_columns=label_columns,
        asset_column=asset_column,
        source_columns_by_kind=source_columns_by_kind,
        resource_policy=resource_policy,
    )
    with closing_iterator(work_frames):
        for chunk in work_frames:
            with _filter_mode_scope(
                chunk,
                mode=spec.filter_mode,
                metadata_key="filter_mode",
            ) as (filtered, filter_metadata):
                try:
                    filtered_chunks = _iter_evaluation_filtered_chunks(
                        filtered,
                        filter_metadata,
                        signal_columns=signal_columns,
                        label_columns=label_columns,
                        n_groups=n_groups,
                        asset_column=asset_column,
                        resource_policy=resource_policy,
                    )
                    with closing_iterator(filtered_chunks):
                        for output in filtered_chunks:
                            if summary_state is not None:
                                summary_state.consume(*output.value)
                            try:
                                yield output
                            except GeneratorExit:
                                return
                            finally:
                                output = None
                finally:
                    filtered = chunk = None


def _iter_evaluation_filtered_chunks(
    filtered: pd.DataFrame,
    filter_metadata: Mapping[str, object],
    *,
    signal_columns: tuple[str, ...],
    label_columns: tuple[str, ...],
    n_groups: int,
    asset_column: str,
    resource_policy: RuntimeResourcePolicy,
) -> Iterator[OwnedValue[_EvaluationOutputChunk]]:
    rows_input = strict_non_negative_count_value(filter_metadata["rows_input"])
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
        batch_min = pd.Timestamp(batch_timestamps.min())
        batch_max = pd.Timestamp(batch_timestamps.max())
        batch_min_timestamp = (
            time_value_isoformat(batch_min, preserve_datetime=True)
            if batch_timestamp_count
            else None
        )
        batch_max_timestamp = (
            time_value_isoformat(batch_max, preserve_datetime=True)
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
) -> Iterator[OwnedValue[_EvaluationOutputChunk]]:
    label_metrics: list[pd.DataFrame] = []
    label_grouped: list[pd.DataFrame] = []
    label_leases: list[MemoryLease] = []
    concat_lease: MemoryLease | None = None
    output_lease: MemoryLease | None = None
    owned_output: OwnedValue[_EvaluationOutputChunk] | None = None
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
                metrics, grouped_label_values = compute_ic_and_grouped_label_values(
                    usable,
                    compute_spec,
                    n_groups=n_groups,
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
            output_lease = label_leases.pop()
            label_leases.clear()
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
            output_lease = concat_lease
            concat_lease = None
            for lease in label_leases:
                lease.close()
            label_leases.clear()
        label_metrics.clear()
        label_grouped.clear()
        metadata = {
            "rows_input": rows_input if not metadata_emitted else 0,
            "rows_after_filter_mode": rows_after_filter if not metadata_emitted else 0,
            "filter_columns_used": list(filter_metadata["filter_columns_used"]),
            "filter_definition": str(filter_metadata["filter_definition"]),
            "timestamp_count": batch_timestamp_count,
            "min_timestamp": batch_min_timestamp,
            "max_timestamp": batch_max_timestamp,
            "skipped_timestamp_count": len(skipped_timestamps),
        }
        owned_output = OwnedValue(
            (metrics, grouped_label_values, metadata),
            output_lease,
        )
        output_lease = None
        try:
            yield owned_output
        finally:
            if owned_output.lease.active:
                _close_lease_preserving_primary(
                    owned_output.lease,
                    owner="research.evaluation.output",
                )
    finally:
        primary = sys.exception()
        first_failure: BaseException | None = None
        if output_lease is not None:
            try:
                output_lease.close()
            except BaseException as close_error:
                if primary is not None:
                    primary.add_note(
                        "evaluation output lease release also failed: "
                        f"{type(close_error).__name__}"
                    )
                else:
                    first_failure = close_error
        if concat_lease is not None:
            try:
                concat_lease.close()
            except BaseException as close_error:
                if primary is not None:
                    primary.add_note(
                        "evaluation concat lease release also failed: "
                        f"{type(close_error).__name__}"
                    )
                else:
                    first_failure = close_error
        for lease in label_leases:
            try:
                lease.close()
            except BaseException as close_error:
                if primary is not None:
                    primary.add_note(
                        "evaluation output lease release also failed: "
                        f"{type(close_error).__name__}"
                    )
                elif first_failure is None:
                    first_failure = close_error
                else:
                    first_failure.add_note(
                        "evaluation output lease release also failed: "
                        f"{type(close_error).__name__}"
                    )
        if first_failure is not None:
            raise first_failure


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
        # Pandas objects and one sequential exact-mean workspace.
        # Finite float64 denominators range from 1 through 2**1074.
        + 64 * 1024
        + 256 * min(row_count, 1075)
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
    output_bounds = [
        _evaluation_output_bound(
            frame,
            signal_count=signal_count,
            n_groups=n_groups,
        )
        for _ in compute_columns_by_label
    ]
    retained_before_kernel = 0
    kernel_peak = 0
    for kernel_bound, output_bound in zip(
        kernel_bounds,
        output_bounds,
        strict=True,
    ):
        kernel_peak = max(kernel_peak, retained_before_kernel + kernel_bound)
        retained_before_kernel += output_bound
    concat_peak = (
        retained_before_kernel + max(64 * 1024, retained_before_kernel * 2)
        if len(output_bounds) > 1
        else 0
    )
    diagnostics_bytes = max(1, int(frame["timestamp"].nunique())) * (
        _EVALUATION_TIMESTAMP_FACT_BYTES
    )
    return max(kernel_peak, concat_peak) + diagnostics_bytes


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
                int(stops[session_index]) if session_index < len(stops) else len(frame)
            )
            session = frame.iloc[stop:session_stop]
            session_bytes = _evaluation_batch_peak_bytes(
                session,
                compute_columns_by_label=compute_columns_by_label,
                signal_count=signal_count,
                n_groups=n_groups,
            )
            candidate_bytes = planned_bytes + session_bytes
            if candidate_bytes > resource_policy.planning_headroom_bytes:
                if stop == start:
                    resource_policy.require_working_set(
                        candidate_bytes,
                        error="research_evaluation_working_set_exceeds_task_memory_budget",
                    )
                    planned_bytes = candidate_bytes
                    stop = session_stop
                    session_index += 1
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
    complete_frames = iter_complete_time_sized_frames(
        iter(frames),
        timestamp_column="timestamp",
        resource_policy=resource_policy,
    )
    with closing_iterator(complete_frames) as active_frames:
        for frame in active_frames:
            missing = sorted(required.difference(frame.columns))
            if missing:
                raise ValueError(f"missing_evaluation_columns: {', '.join(missing)}")
            selected = [column for column in work_columns if column in frame.columns]
            projection_bytes = _selected_frame_payload_bytes(frame, selected)
            selected_set = set(selected)
            added_signal_bytes = (
                8
                * len(frame)
                * sum(signal.signal_id not in selected_set for signal in spec.signals)
            )
            normalized_bound = projection_bytes + added_signal_bytes
            resource_policy.require_working_set(
                projection_bytes,
                error="research_evaluation_working_set_exceeds_task_memory_budget",
            )
            normalized_lease = resource_policy.lease(
                "research.evaluation.projection",
                projection_bytes,
            )
            try:
                projected = frame.loc[:, selected]
                normalization_bytes = _signal_normalization_working_bytes(
                    projected, spec.signals
                )
                resource_policy.require_working_set(
                    normalization_bytes,
                    error="research_evaluation_working_set_exceeds_task_memory_budget",
                )
                normalized_lease.grow(normalization_bytes)
                normalized = normalize_signal_frame(
                    projected,
                    signals=spec.signals,
                    timestamp_column="timestamp",
                    source_columns_by_kind=source_columns_by_kind,
                    resource_lease=normalized_lease,
                )
                projected = None
                normalized_bytes = pandas_frame_payload_bytes(normalized)
                if normalized_bytes > normalized_bound:
                    raise ValueError(
                        "research_evaluation_working_set_exceeds_task_memory_budget"
                    )
                normalized_lease.shrink_to(normalized_bytes)
                normalized_lease.transfer("research.evaluation.input")
                yield normalized
            finally:
                projected = normalized = frame = None
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

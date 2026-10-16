"""Spec-driven evaluation orchestration for compute kernels."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
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
from quant_research_platform.research.features.transforms import (
    feature_observation_columns,
    mask_unobserved_features,
)
from quant_research_platform.research.signals.normalization import (
    normalize_signal_frame,
)
from quant_research_platform.runtime.dataframes import iter_complete_time_frames

type _EvaluationOutputChunk = tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]


class _EvaluationStreamingState:
    def __init__(self, *, spec: EvaluationSpec) -> None:
        self._spec = spec
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
                    "std_reason": "not_applicable",
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
                    "std_ic_reason": ic["std_reason"],
                    "icir_reason": ic["ratio_reason"],
                    "rankic_reason": rankic["reason"],
                    "std_rankic_reason": rankic["std_reason"],
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
                    "group_monotonicity_score": None
                    if monotonicity_count == 0
                    else float(stats["monotonicity_sum"]) / monotonicity_count,
                    "group_monotonicity_score_reason": "no_group_transitions"
                    if monotonicity_count == 0
                    else None,
                }
            )
        return pd.DataFrame.from_records(
            records, columns=EVALUATION_GROUP_SPREAD_SUMMARY_COLUMNS
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
            min_timestamp=None
            if self._min_timestamp is None
            else time_value_isoformat(self._min_timestamp, preserve_datetime=True),
            max_timestamp=None
            if self._max_timestamp is None
            else time_value_isoformat(self._max_timestamp, preserve_datetime=True),
            n_groups=self._spec.n_groups,
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
        self, _exception_type: object, primary: BaseException | None, _traceback: object
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
    asset_column: str,
    source_columns_by_kind: Mapping[str, Sequence[str]] | None = None,
) -> EvaluationResult:
    state = _EvaluationStreamingState(
        spec=spec,
    )
    output_chunks = (
        _iter_single_asset_time_series_output_chunks(
            frames,
            spec=spec,
            asset_column=asset_column,
            source_columns_by_kind=source_columns_by_kind,
            summary_state=state,
        )
        if spec.evaluation_axis == "single_asset_time_series"
        else _iter_evaluation_output_chunks(
            frames,
            spec=spec,
            asset_column=asset_column,
            source_columns_by_kind=source_columns_by_kind,
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

    return EvaluationResult(output_chunks, summaries, output_chunks.close)


def _correlation_summary(
    stats: Mapping[str, int | Fraction], prefix: str
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
    asset_column: str,
    source_columns_by_kind: Mapping[str, Sequence[str]] | None = None,
    summary_state: _EvaluationStreamingState | None = None,
) -> Iterator[_EvaluationOutputChunk]:
    """Yield one bounded across-time correlation row per signal-label pair."""
    signal_columns = tuple(signal.signal_id for signal in spec.signals)
    label_columns = spec.label_columns
    numeric_columns = (*signal_columns, *label_columns)
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

    def consume_values(values: np.ndarray) -> None:
        label_offset = len(signal_columns)
        for signal_index, signal in enumerate(signal_columns):
            left = values[:, signal_index]
            for label_index, label in enumerate(label_columns):
                right = values[:, label_offset + label_index]
                valid = np.isfinite(left) & np.isfinite(right)
                moments = stats[signal, label]
                for x, y in zip(left[valid], right[valid], strict=True):
                    moments.update(float(x), float(y))

    work_frames = _iter_evaluation_work_frames(
        frames,
        spec=spec,
        asset_column=asset_column,
        source_columns_by_kind=source_columns_by_kind,
    )
    with closing_iterator(work_frames):
        for filtered, filter_metadata in work_frames:
            rows_input += strict_non_negative_count_value(filter_metadata["rows_input"])
            rows_after_filter += strict_non_negative_count_value(
                filter_metadata["rows_after_filter_mode"]
            )
            filter_columns_used = tuple(
                str(column) for column in filter_metadata["filter_columns_used"]
            )
            filter_definition_value = str(filter_metadata["filter_definition"])
            if filter_metadata["timestamps"].empty:
                continue
            assets = filter_metadata["asset_ids"]
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
                filter_metadata["timestamps"],
                error="invalid_evaluation_timestamp_values",
            )
            if bool(timestamps.duplicated().any()):
                raise ValueError(
                    "single_asset_time_series_evaluation_duplicate_timestamp"
                )
            chunk_minimum = pd.Timestamp(timestamps.min())
            chunk_maximum = pd.Timestamp(timestamps.max())
            if maximum_timestamp is not None and chunk_minimum <= maximum_timestamp:
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
                dtype="float64", copy=True
            )
            consume_values(values)
    metrics = _time_series_metrics_frame(
        stats if maximum_timestamp is not None else {}, timestamp=maximum_timestamp
    )
    grouped_label_values = _empty_grouped_label_values()
    metadata = {
        "rows_input": rows_input,
        "rows_after_filter_mode": rows_after_filter,
        "filter_columns_used": list(filter_columns_used),
        "filter_definition": filter_definition_value,
        "timestamp_count": timestamp_count,
        "min_timestamp": time_value_isoformat(minimum_timestamp, preserve_datetime=True)
        if minimum_timestamp is not None
        else None,
        "max_timestamp": time_value_isoformat(maximum_timestamp, preserve_datetime=True)
        if maximum_timestamp is not None
        else None,
        "skipped_timestamp_count": 0,
    }
    output = (metrics, grouped_label_values, metadata)
    if summary_state is not None:
        summary_state.consume(*output)
    yield output


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
    asset_column: str,
    source_columns_by_kind: Mapping[str, Sequence[str]] | None = None,
    summary_state: _EvaluationStreamingState | None = None,
) -> Iterator[_EvaluationOutputChunk]:
    """Yield bounded pairs and optionally consume them into research summaries."""
    work_frames = _iter_evaluation_work_frames(
        frames,
        spec=spec,
        asset_column=asset_column,
        source_columns_by_kind=source_columns_by_kind,
    )
    with closing_iterator(work_frames):
        for filtered, filter_metadata in work_frames:
            output = _evaluate_filtered_frame(
                filtered,
                filter_metadata,
                spec=spec,
                asset_column=asset_column,
            )
            if summary_state is not None:
                summary_state.consume(*output)
            try:
                yield output
            except GeneratorExit:
                return


def _evaluate_filtered_frame(
    filtered: pd.DataFrame,
    filter_metadata: Mapping[str, object],
    *,
    spec: EvaluationSpec,
    asset_column: str,
) -> _EvaluationOutputChunk:
    signal_columns = tuple(signal.signal_id for signal in spec.signals)
    label_columns = spec.label_columns
    n_groups = spec.n_groups
    timestamps = strict_timestamp_series(
        filter_metadata["timestamps"], error="invalid_evaluation_timestamp_values"
    )
    timestamp_count = int(timestamps.nunique())
    skipped_timestamps: set[pd.Timestamp] = set()
    compute_spec = ComputePanelSpec(
        signal_columns=signal_columns,
        label_columns=label_columns,
        asset_column=asset_column,
    )
    columns = ("timestamp", asset_column, *signal_columns, *label_columns)
    metrics, grouped = compute_ic_and_grouped_label_values(
        filtered.loc[:, list(columns)], compute_spec, n_groups=n_groups
    )
    absent = timestamps.drop_duplicates().loc[
        ~timestamps.drop_duplicates().isin(metrics.timestamp)
    ]
    if not absent.empty:
        missing = pd.DataFrame.from_records(
            [
                {
                    "timestamp": timestamp,
                    "signal": signal,
                    "label": label,
                    "ic": np.nan,
                    "rankic": np.nan,
                    "n_obs": 0,
                    "ic_reason": "insufficient_observations",
                    "rankic_reason": "insufficient_observations",
                }
                for timestamp in absent
                for signal in signal_columns
                for label in label_columns
            ],
            columns=EVALUATION_METRICS_COLUMNS,
        ).astype(metrics.dtypes.to_dict())
        metrics = (
            pd.concat([metrics, missing], ignore_index=True)
            .sort_values(["timestamp", "signal", "label"], kind="stable")
            .reset_index(drop=True)
        )
    n_obs = strict_non_negative_count_series(metrics["n_obs"])
    skipped_timestamps.update(
        pd.Timestamp(value)
        for value in metrics.loc[n_obs.lt(2), "timestamp"].drop_duplicates()
    )
    metadata = {
        "rows_input": strict_non_negative_count_value(filter_metadata["rows_input"]),
        "rows_after_filter_mode": strict_non_negative_count_value(
            filter_metadata["rows_after_filter_mode"]
        ),
        "filter_columns_used": list(filter_metadata["filter_columns_used"]),
        "filter_definition": str(filter_metadata["filter_definition"]),
        "timestamp_count": timestamp_count,
        "min_timestamp": time_value_isoformat(
            pd.Timestamp(timestamps.min()), preserve_datetime=True
        )
        if timestamp_count
        else None,
        "max_timestamp": time_value_isoformat(
            pd.Timestamp(timestamps.max()), preserve_datetime=True
        )
        if timestamp_count
        else None,
        "skipped_timestamp_count": len(skipped_timestamps),
    }
    return (metrics, grouped, metadata)


def _iter_evaluation_work_frames(
    frames: Iterable[pd.DataFrame],
    *,
    spec: EvaluationSpec,
    asset_column: str,
    source_columns_by_kind: Mapping[str, Sequence[str]] | None = None,
) -> Iterator[tuple[pd.DataFrame, dict[str, object]]]:
    from quant_research_platform.research.filtering import apply_filter_mode

    label_columns = spec.label_columns
    feature_columns = tuple(
        signal.source_column
        for signal in spec.signals
        if signal.source_kind == "feature"
    )
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
                *feature_observation_columns(feature_columns),
                *required_filter_columns(spec.filter_mode),
            )
        )
    )
    complete_frames = iter_complete_time_frames(
        iter(frames), timestamp_column="timestamp"
    )
    with closing_iterator(complete_frames) as active_frames:
        for frame in active_frames:
            missing = sorted(required.difference(frame.columns))
            if missing:
                raise ValueError(f"missing_evaluation_columns: {', '.join(missing)}")
            selected = [column for column in work_columns if column in frame.columns]
            projected = frame.loc[:, selected]
            filtered, filter_metadata = apply_filter_mode(
                projected, mode=spec.filter_mode, metadata_key="filter_mode"
            )
            filter_metadata["timestamps"] = projected.timestamp
            filter_metadata["asset_ids"] = projected[asset_column]
            normalized = normalize_signal_frame(
                mask_unobserved_features(filtered, feature_columns),
                signals=spec.signals,
                timestamp_column="timestamp",
                source_columns_by_kind=source_columns_by_kind,
            )

            try:
                yield normalized, filter_metadata
            except GeneratorExit:
                return

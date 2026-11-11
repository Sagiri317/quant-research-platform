"""Compute signal-label metrics, daily strata, and response diagnostics on
aligned samples.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quant_research_platform.compute.dataframe.cross_section import (
    _add_correlation_aggregate,
    _compute_ic_and_grouped_label_values,
    _correlation_metrics_frame,
    _correlation_records,
    _pearson_correlation_result,
    _summarize_correlation_aggregates,
)
from quant_research_platform.compute.panels import ComputePanelSpec
from quant_research_platform.contracts.evaluations import (
    EVALUATION_GROUP_SPREAD_SUMMARY_COLUMNS,
    EVALUATION_GROUPED_LABEL_VALUE_COLUMNS,
    EVALUATION_IC_SUMMARY_COLUMNS,
    EvaluationDiagnostics,
    EvaluationSpec,
)
from quant_research_platform.contracts.filtering import (
    filter_definition,
    required_filter_columns,
)
from quant_research_platform.contracts.signals import DEFAULT_N_GROUPS
from quant_research_platform.contracts.time import time_value_isoformat
from quant_research_platform.contracts.validation import (
    _finite_numeric_series,
    strict_timestamp_series,
)
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.research.features.transforms import _industry_group_codes
from quant_research_platform.research.signals.normalization import (
    normalize_signal_frame,
)
from quant_research_platform.research.statistics import common_correlation_mask
from quant_research_platform.runtime.dataframes import iter_complete_time_frames

type _EvaluationOutputChunk = tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]


class _EvaluationStreamingState:
    def __init__(self, *, spec: EvaluationSpec) -> None:
        self._spec = spec
        self._filter_columns_used = required_filter_columns(spec.filter_mode)
        self._filter_definition = filter_definition(spec.filter_mode)
        self._rows_input = 0
        self._rows_after_filter_mode = 0
        self._rows_after_signal_label_dropna = 0
        self._metric_rows = 0
        self._grouped_rows = 0
        self._timestamp_count = 0
        self._min_timestamp: pd.Timestamp | None = None
        self._max_timestamp: pd.Timestamp | None = None
        self._skipped_timestamp_count = 0
        self._ic: dict[tuple[str, str, str], dict[str, int | float]] = {}
        self._spread: dict[tuple[str, str], dict[str, float | int]] = {}
        self._day = {}
        self._daily_ic = {}

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
        fact_count = metadata["timestamp_count"]
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
        self._skipped_timestamp_count += metadata["skipped_timestamp_count"]
        if not metrics.empty:
            self._metric_rows += len(metrics)
            self._rows_after_signal_label_dropna += sum(
                int(count) for count in metrics.n_obs
            )
            self._consume_correlations(metrics, side="all", metadata=metadata)
        if not metrics.empty and self._spec.top_quantile is not None:
            for side in ("long", "short"):
                columns = {f"{side}_{metric}": metric for metric in ("ic", "rankic")}
                selected = metrics.loc[:, ["timestamp", "signal", "label", *columns]]
                self._consume_correlations(
                    selected.rename(columns=columns), side=side, metadata=metadata
                )
        if not grouped_label_values.empty:
            self._grouped_rows += len(grouped_label_values)
            current_key: tuple[object, object, object] | None = None
            first_value: float | None = None
            last_value: float | None = None
            previous_value: float | None = None
            monotonicity_sum = 0.0
            monotonicity_count = 0
            group_count = 0

            def finish_group() -> None:
                if current_key is None or group_count < 2:
                    return
                signal, label, _timestamp = current_key
                bottom = math.nan if first_value is None else first_value
                top = math.nan if last_value is None else last_value
                monotonicity = (
                    None
                    if monotonicity_count == 0
                    else monotonicity_sum / monotonicity_count
                )
                if not math.isfinite(top) or not math.isfinite(bottom):
                    return
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
                group,
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
                    group_count = 0
                group_count += 1
                if group == 1:
                    first_value = float(mean_label_value)
                if group == self._spec.n_groups:
                    last_value = float(mean_label_value)
                if pd.isna(mean_label_value):
                    previous_value = None
                    continue
                value = float(mean_label_value)
                if previous_value is not None:
                    monotonicity_sum += float(
                        (value > previous_value) - (value < previous_value)
                    )
                    monotonicity_count += 1
                previous_value = value
            finish_group()

    def _consume_correlations(self, metrics, *, side, metadata):
        common = (
            common_correlation_mask(metrics, groups=("timestamp", "label")).to_numpy()
            if metadata.get("common_metrics")
            else None
        )
        for row_number, row in enumerate(metrics.itertuples(index=False)):
            key = (str(row.signal), str(row.label), side)
            stats = self._ic.get(key)
            if stats is None:
                stats = {
                    f"{metric}_{name}": 0
                    for metric in ("ic", "rankic")
                    for name in ("count", "mean", "m2", "positive")
                }
                self._ic[key] = stats
            if self._spec.correlation_clock == "trading_day":
                day = pd.Timestamp(row.timestamp).normalize()
                if side in self._day and day != self._day[side]:
                    self._flush_day(side)
                self._day[side] = day
                daily = self._daily_ic.setdefault(key, [0.0, 0, 0.0, 0])
                for offset, value in ((0, row.ic), (2, row.rankic)):
                    if common is not None and not common[row_number, offset // 2]:
                        continue
                    if pd.notna(value) and math.isfinite(value):
                        daily[offset] += value
                        daily[offset + 1] += 1
            else:
                _add_correlation_aggregate(stats, "ic", row.ic)
                _add_correlation_aggregate(stats, "rankic", row.rankic)

    def _flush_day(self, side=None):
        for key in list(self._daily_ic):
            if side is not None and key[2] != side:
                continue
            daily = self._daily_ic.pop(key)
            for offset, metric in ((0, "ic"), (2, "rankic")):
                if daily[offset + 1]:
                    _add_correlation_aggregate(
                        self._ic[key], metric, daily[offset] / daily[offset + 1]
                    )

    def ic_summary(self) -> pd.DataFrame:
        self._flush_day()
        records = {}
        for (signal, label, side), stats in sorted(self._ic.items()):
            row = records.setdefault(
                (signal, label), {"signal": signal, "label": label}
            )
            prefix = "" if side == "all" else f"{side}_"
            for metric in ("ic", "rankic"):
                result = _correlation_summary(stats, metric)
                name = f"{prefix}{metric}"
                row.update(
                    {
                        f"mean_{name}": result["mean"],
                        f"std_{name}": result["std"],
                        f"{name}ir": result["ratio"],
                        f"positive_{name}_rate": result["positive_rate"],
                        f"valid_{name}_count": result["count"],
                        f"{name}_reason": result["reason"],
                        f"std_{name}_reason": result["std_reason"],
                        f"{name}ir_reason": result["ratio_reason"],
                    }
                )
        frame = pd.DataFrame.from_records(
            list(records.values()), columns=EVALUATION_IC_SUMMARY_COLUMNS
        )
        if self._spec.top_quantile is None:
            for side in ("long", "short"):
                for metric in ("ic", "rankic"):
                    frame[f"valid_{side}_{metric}_count"] = 0
                    for name in (
                        f"{side}_{metric}_reason",
                        f"std_{side}_{metric}_reason",
                        f"{side}_{metric}ir_reason",
                    ):
                        frame[name] = "not_applicable"
        return frame

    def group_spread_summary(self) -> pd.DataFrame:
        records: list[dict[str, object]] = []
        for (signal, label), stats in sorted(self._spread.items()):
            count = int(stats["count"])
            monotonicity_count = int(stats["monotonicity_count"])
            spread_mean = (stats["top_sum"] - stats["bottom_sum"]) / count
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
            ic_summary_rows=sum(key[2] == "all" for key in self._ic),
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
    response_diagnostics=None,
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
            response_diagnostics=response_diagnostics,
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
        )

    return EvaluationResult(output_chunks, summaries, output_chunks.close)


def _correlation_summary(
    stats: Mapping[str, int | float], prefix: str
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
    spread_state: dict[tuple[str, str], dict[str, float | int]],
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
            "top_sum": 0.0,
            "bottom_sum": 0.0,
            "hit_sum": 0.0,
            "monotonicity_sum": 0.0,
            "monotonicity_count": 0,
        }
        spread_state[key] = spread
    spread["count"] += 1
    spread["top_sum"] += top
    spread["bottom_sum"] += bottom
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
    response_diagnostics=None,
) -> Iterator[_EvaluationOutputChunk]:
    """Evaluate each pair over the complete selected narrow time-series input."""
    selected_frames = []
    response_dates = {}
    signal_columns = tuple(signal.signal_id for signal in spec.signals)
    label_columns = spec.label_columns
    numeric_columns = (*signal_columns, *label_columns)
    rows_input = 0
    rows_after_filter = 0
    filter_columns_used = tuple(required_filter_columns(spec.filter_mode))
    filter_definition_value = filter_definition(spec.filter_mode)
    asset: object | None = None
    timestamp_count = 0
    minimum_timestamp: pd.Timestamp | None = None
    maximum_timestamp: pd.Timestamp | None = None

    work_frames = _iter_evaluation_work_frames(
        frames,
        spec=spec,
        asset_column=asset_column,
        source_columns_by_kind=source_columns_by_kind,
    )
    with closing_iterator(work_frames):
        for filtered, filter_metadata in work_frames:
            rows_input += filter_metadata["rows_input"]
            rows_after_filter += filter_metadata["rows_after_filter_mode"]
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
            if (
                response_diagnostics is not None
                or spec.correlation_clock == "trading_day"
            ):
                response_dates.update(
                    timestamps.groupby(timestamps.dt.normalize()).max().to_dict()
                )
            selected_frames.append(filtered.loc[:, ["timestamp", *numeric_columns]])
    frame = (
        pd.concat(selected_frames, ignore_index=True)
        if selected_frames
        else pd.DataFrame(columns=["timestamp", *numeric_columns])
    )
    selected_frames.clear()
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
    if response_diagnostics is not None:
        metadata.update(
            _response_tables(
                frame,
                spec=spec,
                config=response_diagnostics,
                dates=response_dates,
            )
        )
    if spec.correlation_clock == "trading_day":
        by_day = {
            day: block
            for day, block in frame.groupby(frame.timestamp.dt.normalize(), sort=False)
        }
        records = []
        for day in sorted(response_dates):
            block = by_day.get(day, frame.iloc[:0])
            current, _ = _correlation_records(
                block,
                signal_columns=signal_columns,
                label_columns=label_columns,
                timestamp=response_dates[day],
                top_quantile=spec.top_quantile,
                include_groups=False,
            )
            records.extend(current)
        metrics = _time_series_metric_records(records)
    else:
        metrics = _time_series_metrics_frame(
            frame,
            spec=spec,
            timestamp=maximum_timestamp,
        )
    del frame
    output = (metrics, grouped_label_values, metadata)
    if summary_state is not None:
        summary_state.consume(*output)
    yield output


def _time_series_metrics_frame(frame, *, spec, timestamp):
    records = []
    if timestamp is not None:
        records, _ = _correlation_records(
            frame,
            signal_columns=tuple(signal.signal_id for signal in spec.signals),
            label_columns=spec.label_columns,
            timestamp=timestamp,
            top_quantile=spec.top_quantile,
            include_groups=False,
        )
    return _time_series_metric_records(records)


def _time_series_metric_records(records):
    metrics = _correlation_metrics_frame(records)
    return metrics.sort_values(
        ["timestamp", "signal", "label"], kind="stable"
    ).reset_index(drop=True)


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
            yield output


def _evaluate_filtered_frame(
    filtered: pd.DataFrame,
    filter_metadata: Mapping[str, object],
    *,
    spec: EvaluationSpec,
    asset_column: str,
    include_sides=True,
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
    metrics, grouped = _compute_ic_and_grouped_label_values(
        filtered.loc[:, list(columns)],
        compute_spec,
        n_groups=n_groups,
        top_quantile=spec.top_quantile if include_sides else None,
    )
    absent = timestamps.drop_duplicates().loc[
        ~timestamps.drop_duplicates().isin(metrics.timestamp)
    ]
    if not absent.empty:
        missing = _correlation_metrics_frame(
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
            side_reason="insufficient_observations",
        )
        metrics = (
            pd.concat([metrics, missing], ignore_index=True)
            .sort_values(["timestamp", "signal", "label"], kind="stable")
            .reset_index(drop=True)
        )
    n_obs = metrics["n_obs"]
    skipped_timestamps.update(
        pd.Timestamp(value)
        for value in metrics.loc[n_obs.lt(2), "timestamp"].drop_duplicates()
    )
    metadata = {
        "rows_input": filter_metadata["rows_input"],
        "rows_after_filter_mode": filter_metadata["rows_after_filter_mode"],
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
                filtered,
                signals=spec.signals,
                timestamp_column="timestamp",
                source_columns_by_kind=source_columns_by_kind,
            )

            yield normalized, filter_metadata


def _prepare_scoped_frame(raw, source, *, spec, cutoff):
    from quant_research_platform.contracts.experiments import label_maturity_mask
    from quant_research_platform.research.filtering import apply_filter_mode

    selected, metadata = apply_filter_mode(
        raw, mode=spec.filter_mode, metadata_key="filter_mode"
    )
    selected = selected.copy()
    for column in source["signal_columns"]:
        selected[column] = selected[column].where(np.isfinite(selected[column]))
    for label, availability in source["availability"].items():
        if pd.api.types.is_integer_dtype(selected[label].dtype):
            # Keep exact integer targets through maturity masking and
            # peer reindexing instead of filling missing rows with float.
            selected[label] = selected[label].astype(
                "UInt64"
                if pd.api.types.is_unsigned_integer_dtype(selected[label].dtype)
                else "Int64"
            )
        mature = label_maturity_mask(
            selected,
            fitting_cutoff=cutoff,
            availability_column=availability,
            target_column=label,
        )
        selected[label] = selected[label].where(mature & np.isfinite(selected[label]))
    return selected, metadata


def _evaluate_scoped_day(pair, *, spec_values, sources, cutoff, sample_scope, strata):
    from dataclasses import replace

    from quant_research_platform.artifacts.outputs.evaluation_daily import daily_frame

    raw, raw_peers = pair
    spec = EvaluationSpec(**spec_values)
    primary = strata.get("primary_label")
    if raw.empty:
        return None
    focal, metadata = _prepare_scoped_frame(raw, sources[0], spec=spec, cutoff=cutoff)
    metadata["timestamps"] = raw.timestamp
    metadata["asset_ids"] = raw.asset_id
    timestamp = raw.timestamp.iloc[0]
    classification = focal.copy()
    for column in (
        "total_market_cap",
        "amount",
        "median_forecast_age_days_fy1",
    ):
        if column in raw:
            classification[column] = raw.loc[focal.index, column]
    groups, quality = _evaluation_strata(classification, strata)
    mature_counts = {
        label: int(focal[label].notna().sum()) for label in spec.label_columns
    }
    # Normalize on the focal eligible population before intersection.
    normalized = normalize_signal_frame(focal, signals=spec.signals)
    if sample_scope == "common":
        valid = np.isfinite(
            focal[sources[0]["signal_columns"]].to_numpy(dtype=float)
        ).all(axis=1)
        keys = pd.MultiIndex.from_frame(focal[["timestamp", "asset_id"]])
        aligned_peers = []
        for raw_peer, source in zip(raw_peers, sources[1:], strict=True):
            if raw_peer is None:
                valid[:] = False
                continue
            peer, _ = _prepare_scoped_frame(raw_peer, source, spec=spec, cutoff=cutoff)
            aligned = peer.set_index(["timestamp", "asset_id"]).reindex(keys)
            valid &= np.isfinite(
                aligned[source["signal_columns"]].to_numpy(dtype=float)
            ).all(axis=1)
            aligned_peers.append(aligned)
        for label in spec.label_columns:
            domain = valid & focal[label].notna().to_numpy()
            for peer in aligned_peers:
                domain &= peer[label].notna().to_numpy()
            for peer in aligned_peers:
                integral = any(
                    pd.api.types.is_integer_dtype(values.dtype)
                    for values in (focal[label], peer[label])
                )
                left = focal[label].to_numpy(dtype=object if integral else float)[
                    domain
                ]
                right = peer[label].to_numpy(dtype=object if integral else float)[
                    domain
                ]
                if integral:
                    matches = np.equal(left, right)
                else:
                    matches = np.isclose(left, right, rtol=1e-12, atol=0.0)
                if not matches.all():
                    raise ValueError("evaluation_common_label_mismatch")
            normalized[label] = normalized[label].where(domain)
    coverage = daily_frame(
        [
            {
                "timestamp": timestamp,
                "label": label,
                "dataset_rows": len(raw),
                "eligible_rows": len(focal),
                "mature_rows": mature_counts[label],
                "domain_rows": int(normalized[label].notna().sum()),
                **{
                    key: value if label == primary else None
                    for key, value in quality.items()
                },
            }
            for label in spec.label_columns
        ],
        "daily_coverage",
    )
    metrics, grouped, facts = _evaluate_filtered_frame(
        normalized,
        metadata,
        spec=spec,
        asset_column="asset_id",
    )
    facts["classification_samples"] = {
        column: normalized.loc[
            normalized[label].notna() & focal[column].notna(),
            ["timestamp", "asset_id", label],
        ]
        for column, label in sources[0].get("classification_targets", {}).items()
    }
    stratum_records = []
    if primary is not None:
        stratum_spec = replace(
            spec,
            label_columns=(primary,),
            label_availability_columns={
                primary: spec.label_availability_columns[primary]
            },
        )
        for kind, assignments in groups.items():
            labels = (
                [item[0] for item in strata["age_bounds"]]
                if kind == "forecast_age"
                else list(assignments.drop_duplicates())
            )
            if kind == "forecast_age" and assignments.isna().any():
                labels.append(pd.NA)
            for label in labels:
                keep = (
                    assignments.isna()
                    if pd.isna(label)
                    else assignments.eq(label).fillna(False)
                )
                subset = normalized.loc[assignments.index[keep]]
                daily, _, _ = _evaluate_filtered_frame(
                    subset,
                    metadata,
                    spec=stratum_spec,
                    asset_column="asset_id",
                    include_sides=False,
                )
                daily = daily.assign(
                    stratification=kind,
                    stratum=label,
                    stratum_rows=int(keep.sum()),
                )
                stratum_records.extend(daily.to_dict("records"))
    daily = daily_frame(stratum_records, "daily_strata")
    facts.update(
        daily_coverage=coverage,
        daily_strata=daily,
        common_metrics=sample_scope == "common",
    )
    return metrics, grouped, facts


def build_scoped_evaluation_result(
    frames, *, spec, sources, cutoff, sample_scope, strata, workers=1
):
    """Evaluate independent focal dates and consume them in their original order."""
    from concurrent.futures import ProcessPoolExecutor
    from contextlib import nullcontext
    from dataclasses import fields
    from functools import partial
    from itertools import batched
    from multiprocessing import get_context

    if type(workers) is not int or workers < 1:
        raise ValueError("evaluation_workers_invalid")
    state = _EvaluationStreamingState(spec=spec)
    values = {item.name: getattr(spec, item.name) for item in fields(spec)}
    values["label_availability_columns"] = dict(spec.label_availability_columns)
    task = partial(
        _evaluate_scoped_day,
        spec_values=values,
        sources=[
            {
                "signal_columns": source["signal_columns"],
                "availability": source["availability"],
                "classification_targets": source.get("classification_targets", {}),
                "declared": {"column_roles": dict(source["declared"]["column_roles"])},
            }
            for source in sources
        ],
        cutoff=cutoff,
        sample_scope=sample_scope,
        strata=strata,
    )

    def outputs():
        with closing_iterator(iter(frames)) as active:
            with (
                ProcessPoolExecutor(
                    max_workers=workers, mp_context=get_context("spawn")
                )
                if workers > 1
                else nullcontext()
            ) as pool:
                for batch in batched(active, workers, strict=False):
                    results = (
                        pool.map(task, batch) if pool is not None else map(task, batch)
                    )
                    for result in results:
                        if result is None:
                            continue
                        metrics, grouped, facts = result
                        state.consume(metrics, grouped, facts)
                        yield metrics, grouped, facts

    chunks = outputs()
    return EvaluationResult(
        chunks,
        lambda: (
            state.diagnostics(),
            state.ic_summary(),
            state.group_spread_summary(),
        ),
        chunks.close,
    )


def _evaluation_strata(frame, strata):
    """Classify the original mature focal population, retaining local invalidity."""
    quality = dict.fromkeys(
        (
            "age_missing",
            "fractional_boundary_unassigned",
            "age_invalid",
            "size_invalid",
            "liquidity_invalid",
        )
    )
    if not strata:
        return {}, quality
    mature = frame.loc[frame[strata["primary_label"]].notna()]
    groups = {}
    for kind in strata["kinds"]:
        if kind == "industry":
            values = (
                _industry_group_codes(mature)
                if "industry_standard_id" in mature
                else mature.industry_level1_code.astype("string")
            )
        elif kind in {"size", "liquidity"}:
            column = "total_market_cap" if kind == "size" else "amount"
            numeric = mature[column]
            invalid = numeric.notna() & (~np.isfinite(numeric) | numeric.lt(0))
            quality[kind + "_invalid"] = int(invalid.sum())
            values = pd.Series(pd.NA, index=mature.index, dtype="string")
            valid = numeric.notna() & ~invalid
            values.loc[valid] = (
                np.ceil(numeric.loc[valid].rank(method="average", pct=True) * 3)
                .astype("Int64")
                .astype("string")
            )
        else:
            age = mature[strata["age_column"]]
            invalid = age.notna() & (
                ~np.isfinite(age) | ~age.between(0, strata["forecast_freshness_days"])
            )
            values = pd.Series(pd.NA, index=mature.index, dtype="string")
            for label, low, high in strata["age_bounds"]:
                values.loc[age.between(low, high)] = label
            quality["age_missing"] = int(age.isna().sum())
            quality["age_invalid"] = int(invalid.sum())
            fractional = age.notna() & ~invalid & values.isna()
            quality["fractional_boundary_unassigned"] = int(fractional.sum())
            values = values.loc[~fractional]
        if not mature.empty:
            groups[kind] = values
    return groups, quality


def training_quantile_cutpoints(scores, *, n_groups=DEFAULT_N_GROUPS):
    """Freeze empirical training-score bins; ties share their upper-inclusive bin.

    Pass explicit predictions on the legal training reference domain, normally
    in-sample scores from the frozen model. This function neither fits nor proves
    out-of-sample provenance. If OOS reference scores are chosen, declare that
    different reference design. Values must already use the evaluated direction.
    """
    values = np.asarray(scores)
    if (
        type(n_groups) is not int
        or not 2 <= n_groups <= 5
        or values.ndim != 1
        or not len(values)
    ):
        raise ValueError("evaluation_reference_scores_invalid")
    values = _finite_numeric_series(
        pd.Series(scores), error="evaluation_reference_scores_invalid"
    ).to_numpy()
    edges = np.unique(
        np.quantile(values, np.arange(1, n_groups) / n_groups, method="inverted_cdf")
    )
    return edges[edges < values.max()].tolist()


def _response_tables(frame, *, spec, config, dates=None):
    """Calculate both response tables from one selected finite-window cache.

    Exact pooled medians retain O(N * selected columns) values. No sampling,
    synthetic anchors, implicit fitting, inference, or cross-asset pooling occurs.
    """
    from quant_research_platform.artifacts.outputs.evaluation_daily import daily_frame

    def finite(value):
        return float(value) if np.isfinite(value) else np.nan

    def mean_bp(values):
        return finite(np.mean(values) * 10000) if len(values) else np.nan

    def median_bp(values):
        return finite(np.median(values) * 10000) if len(values) else np.nan

    groups_only = config.get("analysis") == "daily_group_returns"
    all_values = frame.loc[:, list(spec.label_columns)].to_numpy(dtype="float64")
    all_mature = np.isfinite(all_values).all(axis=1)
    timestamps = pd.DatetimeIndex(frame.timestamp)
    by_day = frame.groupby(timestamps.normalize(), sort=False).indices
    dates = dates or {
        day: timestamps[positions[-1]] for day, positions in by_day.items()
    }
    metrics, grouped = [], []
    periods = [
        ("daily", stamp, by_day.get(day, np.empty(0, dtype="int64")))
        for day, stamp in sorted(dates.items())
    ]
    if len(dates) and not groups_only:
        periods.append(("pooled", max(dates.values()), slice(None)))
    for signal, edges in config["cutpoints"].items():
        unavailable = edges is None
        n_groups = spec.n_groups if unavailable else len(edges) + 1
        all_scores = frame[signal].to_numpy(dtype="float64")
        all_predicted = np.isfinite(all_scores)
        all_groups = np.zeros(len(frame), dtype="int64")
        if not unavailable:
            all_groups[all_predicted] = (
                np.searchsorted(
                    np.asarray(edges, dtype="float64"),
                    all_scores[all_predicted],
                    side="left",
                )
                + 1
            )
        for period, timestamp, positions in periods:
            scores, labels = all_scores[positions], all_values[positions]
            selected, all_labels, groups = (
                all_predicted[positions],
                all_mature[positions],
                all_groups[positions],
            )
            for scope in ("own",) if groups_only else ("common", "own"):
                denominator = int(selected.sum())
                for index, label in enumerate(spec.label_columns):
                    valid = selected & (
                        all_labels
                        if scope == "common"
                        else np.isfinite(labels[:, index])
                    )
                    count = int(valid.sum())
                    if not groups_only:
                        ic, reason = _pearson_correlation_result(
                            scores[valid], labels[valid, index]
                        )
                    key = dict(
                        timestamp=timestamp,
                        period=period,
                        sample_scope=scope,
                        signal=signal,
                        label=label,
                    )
                    means, counts = [], []
                    for group in range(1, n_groups + 1):
                        values = labels[valid & (groups == group), index]
                        average = mean_bp(values)
                        median = np.nan if groups_only else median_bp(values)
                        means.append(average)
                        counts.append(len(values))
                        grouped.append(
                            dict(
                                **key,
                                group=group,
                                mean_return_bp=average,
                                median_return_bp=median,
                                n_obs=len(values),
                                mean_std_bp=None,
                                valid_day_count=int(
                                    period == "daily" and len(values) > 0
                                ),
                                mean_reason="no_finite_reference"
                                if unavailable
                                else None
                                if math.isfinite(average)
                                else "no_valid_observations"
                                if not len(values)
                                else "nonfinite_result",
                                median_reason="not_requested"
                                if groups_only
                                else "no_finite_reference"
                                if unavailable
                                else None
                                if math.isfinite(median)
                                else "no_valid_observations"
                                if not len(values)
                                else "nonfinite_result",
                            )
                        )
                    if groups_only:
                        continue
                    spread_reason = (
                        "no_finite_reference"
                        if unavailable
                        else "one_signal_group"
                        if n_groups == 1
                        else "empty_extreme_group"
                        if not all(counts[i] for i in (0, -1))
                        else "nonfinite_result"
                        if not all(math.isfinite(means[i]) for i in (0, -1))
                        else None
                    )
                    spread = (
                        finite(means[-1] - means[0])
                        if spread_reason is None
                        else np.nan
                    )
                    if spread_reason is None and not math.isfinite(spread):
                        spread_reason = "nonfinite_result"
                    metrics.append(
                        dict(
                            **key,
                            ic=ic,
                            ic_reason=reason,
                            n_obs=count,
                            signal_count=denominator,
                            coverage=count / denominator if denominator else None,
                            ic_std=None,
                            valid_day_count=int(period == "daily" and ic is not None),
                            top_bottom_mean_bp=spread,
                            spread_reason=spread_reason,
                        )
                    )
    if len(dates) and not groups_only:
        # Summary rows are explicitly means of daily estimates, never pooled IC.
        for records, value, dispersion, keys in (
            (metrics, "ic", "ic_std", ["sample_scope", "signal", "label"]),
            (
                grouped,
                "mean_return_bp",
                "mean_std_bp",
                ["sample_scope", "signal", "label", "group"],
            ),
        ):
            daily = pd.DataFrame(records).query("period == 'daily'")
            for _, block in daily.groupby(keys, sort=False):
                row = block.iloc[0].to_dict()
                row.update(
                    timestamp=max(dates.values()),
                    period="daily_equal_weight",
                    n_obs=int(block.n_obs.sum()),
                )
                values = block[value].dropna().tolist()
                row[value] = finite(np.mean(values)) if values else None
                row[dispersion] = (
                    finite(np.std(values, ddof=1)) if len(values) > 1 else None
                )
                row["valid_day_count"] = len(values)
                if value == "ic":
                    row["ic_reason"] = None if values else "no_valid_observations"
                    row["signal_count"] = int(block.signal_count.sum())
                    row["coverage"] = (
                        row["n_obs"] / row["signal_count"]
                        if row["signal_count"]
                        else None
                    )
                    spreads = block.top_bottom_mean_bp.dropna().tolist()
                    row["top_bottom_mean_bp"] = (
                        finite(np.mean(spreads)) if spreads else None
                    )
                    row["spread_reason"] = (
                        None
                        if spreads
                        else "no_finite_reference"
                        if config["cutpoints"][row["signal"]] is None
                        else "one_signal_group"
                        if not config["cutpoints"][row["signal"]]
                        else "no_valid_observations"
                    )
                else:
                    row["mean_reason"] = (
                        None
                        if values
                        else "no_finite_reference"
                        if config["cutpoints"][row["signal"]] is None
                        else "no_valid_observations"
                    )
                    row["median_return_bp"] = None
                    row["median_reason"] = "not_aggregated"
                records.append(row)
        for records, keys in (
            (metrics, ["sample_scope", "signal", "label"]),
            (grouped, ["sample_scope", "signal", "label", "group"]),
        ):
            counts = {
                tuple(row[key] for key in keys): row["valid_day_count"]
                for row in records
                if row["period"] == "daily_equal_weight"
            }
            for row in records:
                if row["period"] == "pooled":
                    row["valid_day_count"] = counts[tuple(row[key] for key in keys)]
    result = {}
    for name, records in (
        (("response_groups", grouped),)
        if groups_only
        else (("response_metrics", metrics), ("response_groups", grouped))
    ):
        result[name] = (
            daily_frame(records, name)
            .sort_values(
                [
                    "timestamp",
                    "period",
                    "sample_scope",
                    "signal",
                    "label",
                    *(["group"] if name == "response_groups" else []),
                ]
            )
            .reset_index(drop=True)
        )
    return result

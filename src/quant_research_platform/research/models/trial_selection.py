"""Validation-metric-only model trial selection helpers."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Collection, Iterable
from fractions import Fraction

import numpy as np
import pandas as pd

from quant_research_platform.compute.dataframe.cross_section import (
    _add_correlation_aggregate,
    _CorrelationMoments,
    _summarize_correlation_aggregates,
)
from quant_research_platform.contracts.experiments import (
    label_maturity_mask,
    resolve_training_availability_column,
    training_availability_columns,
)
from quant_research_platform.contracts.json_codec import canonical_json_hash
from quant_research_platform.contracts.validation import strict_timestamp_series
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.research.experiments import TrialRecorder
from quant_research_platform.research.features.transforms import (
    model_input_validity_columns,
)
from quant_research_platform.research.filtering import apply_filter_mode
from quant_research_platform.research.models import fitting
from quant_research_platform.research.models.fitting import prediction_callable
from quant_research_platform.research.models.supervised import (
    build_estimator_from_model_spec,
)
from quant_research_platform.runtime.dataframes import iter_complete_time_frames

VALIDATION_METRIC_COLUMNS = ("validation_ic", "validation_rankic", "validation_mse")
VALIDATION_METRIC_REASON_COLUMNS = ("validation_ic_reason", "validation_rankic_reason")
TEST_METRIC_COLUMNS = ("test_ic", "test_rankic", "test_mse")


def validate_trial_metric_columns(
    *,
    selection_metric: str,
    validation_metric_columns: Collection[str],
    test_metric_columns: Collection[str],
) -> None:
    """Ensure model selection uses validation metrics only."""
    metrics = set(validation_metric_columns)
    test_metrics = set(test_metric_columns)
    if selection_metric not in metrics:
        raise ValueError("selection_metric_must_be_validation_metric")
    if selection_metric in test_metrics:
        raise ValueError("selection_metric_must_not_be_test_metric")


def select_best_validation_trial(
    trials: pd.DataFrame,
    *,
    trial_id_column: str,
    selection_metric: str,
    maximize: bool,
    validation_metric_columns: Collection[str],
    test_metric_columns: Collection[str],
    tie_break_order: tuple[str, ...] | None = None,
) -> str:
    """Select the best trial using a validation metric."""
    validate_trial_metric_columns(
        selection_metric=selection_metric,
        validation_metric_columns=validation_metric_columns,
        test_metric_columns=test_metric_columns,
    )
    if trials.empty:
        raise ValueError("empty_trials")
    missing = sorted({trial_id_column, selection_metric}.difference(trials.columns))
    if missing:
        raise ValueError(f"missing_trial_selection_columns: {', '.join(missing)}")
    if bool(trials[trial_id_column].duplicated().any()):
        raise ValueError("duplicate_trial_id")
    metric_series = trials[selection_metric]
    if pd.api.types.is_bool_dtype(metric_series):
        raise ValueError("invalid_trial_selection_metric")
    if not pd.api.types.is_numeric_dtype(metric_series):
        if not bool(metric_series.isna().all()):
            raise ValueError("invalid_trial_selection_metric")
        metric_values = pd.Series(np.nan, index=metric_series.index, dtype="float64")
    else:
        metric_values = metric_series.astype("float64")
    if bool((~metric_values.isna() & ~np.isfinite(metric_values)).any()):
        raise ValueError("invalid_trial_selection_metric")
    defined = metric_values.notna()
    selected_trials = trials.copy()
    ordered_ids = tuple(
        str(value)
        for value in (
            tie_break_order
            if tie_break_order is not None
            else selected_trials[trial_id_column].tolist()
        )
    )
    if set(ordered_ids) != set(
        str(value) for value in selected_trials[trial_id_column]
    ):
        raise ValueError("trial_tie_break_order_mismatch")
    order = {value: index for index, value in enumerate(ordered_ids)}
    ordered = selected_trials.assign(
        _undefined=~defined,
        _selection_metric=metric_values.fillna(-np.inf if maximize else np.inf),
        _tie_break=selected_trials[trial_id_column].map(
            lambda value: order[str(value)]
        ),
    ).sort_values(
        ["_undefined", "_selection_metric", "_tie_break"],
        ascending=[True, not maximize, True],
        kind="mergesort",
    )
    return str(ordered.iloc[0][trial_id_column])


def model_validation_inputs(specs):
    return tuple(
        dict.fromkeys(
            column
            for spec in specs
            for column in model_input_validity_columns(
                spec.input_columns, spec.model_algorithm
            )
        )
    )


def model_validation_coverage_scope(dataset_identity, fold, specs):
    labels = {spec.label_column for spec in specs}
    if len(labels) != 1:
        raise ValueError("model_selection_label_mismatch")
    label = specs[0].label_column
    return {
        "dataset_identity": dataset_identity,
        "fold_id": fold.fold_id,
        "validation_start": str(fold.validation.start),
        "validation_end": str(fold.validation.end),
        "label": label,
        "availability": training_availability_columns(label)[0],
        "inputs": list(model_validation_inputs(specs)),
        "eligible": True,
    }


def crypto_validation_coverage_scope(plan, spec, asset_ids):
    return {
        "dataset_identity": plan.data_snapshot_id,
        "fold": plan.protocol_payload()["folds"][0],
        "label": spec.label_column,
        "availability": training_availability_columns(spec.label_column)[0],
        "inputs": list(spec.input_columns),
        "asset_ids": list(asset_ids),
    }


class ModelSampleCoverage:
    """Identity for per-asset ordered PK streams, independent of interleaving."""

    def __init__(self, scope):
        self.scope = scope
        self.row_count = 0
        self._assets = {}

    def add(self, frame):
        for timestamp, asset in frame.loc[:, ["timestamp", "asset_id"]].itertuples(
            index=False, name=None
        ):
            if pd.isna(timestamp) or type(asset) is not str or (not asset):
                raise ValueError("model_coverage_key_invalid")
            timestamp_ns = pd.Timestamp(timestamp).value
            state = self._assets.get(asset)
            if state is None:
                digest = hashlib.sha256()
            else:
                last_timestamp_ns, digest = state
                if timestamp_ns <= last_timestamp_ns:
                    raise ValueError("model_coverage_key_order_invalid")
            digest.update(timestamp_ns.to_bytes(8, "big", signed=True))
            self._assets[asset] = (timestamp_ns, digest)
            self.row_count += 1

    @property
    def semantic_identity(self):
        digest = hashlib.sha256()
        for asset in sorted(self._assets):
            encoded = asset.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
            digest.update(self._assets[asset][1].digest())
        return canonical_json_hash(
            {
                "scope": self.scope,
                "row_count": self.row_count,
                "keys_sha256": digest.hexdigest(),
            }
        )


def evaluate_model_trial_metrics_from_chunks(
    chunks: Iterable[pd.DataFrame],
    *,
    model_id: str,
    label_column: str,
    prediction_column: str,
    expected_coverage: ModelSampleCoverage | None = None,
) -> dict[str, object]:
    """Evaluate model-selection metrics from bounded prediction chunks."""
    coverage = (
        None
        if expected_coverage is None
        else ModelSampleCoverage(expected_coverage.scope)
    )
    parts = []
    with closing_iterator(iter(chunks)) as source:
        for frame in source:
            if frame.empty:
                continue
            if coverage is not None:
                coverage.add(frame)
            parts.append(
                _metric_chunk(
                    frame,
                    label_column=label_column,
                    prediction_column=prediction_column,
                )
            )
    if coverage is not None and (
        coverage.semantic_identity != expected_coverage.semantic_identity
        or coverage.row_count != expected_coverage.row_count
    ):
        raise ValueError("model_trial_common_coverage_mismatch")
    if not parts:
        raise ValueError("empty_model_selection_validation_window")
    values = pd.concat(parts, ignore_index=True).sort_values(
        "timestamp_ns", kind="stable"
    )
    ranks = values.groupby("timestamp_ns", sort=False)[["label", "prediction"]].rank(
        method="average", pct=True
    )
    values["label_rank"] = ranks["label"]
    values["prediction_rank"] = ranks["prediction"]
    (
        validation_rows,
        validation_mse,
        validation_ic,
        validation_ic_reason,
        validation_rankic,
        validation_rankic_reason,
    ) = _cross_sectional_metric_summary(values.itertuples(index=False, name=None))
    return {
        "model_id": model_id,
        "validation_rows": validation_rows,
        "validation_ic": validation_ic,
        "validation_ic_reason": validation_ic_reason,
        "validation_rankic": validation_rankic,
        "validation_rankic_reason": validation_rankic_reason,
        "validation_mse": validation_mse,
    }


def _cross_sectional_metric_summary(
    rows,
) -> tuple[int, float, float | None, str | None, float | None, str | None]:
    stats: dict[str, int | Fraction] = {
        "ic_count": 0,
        "ic_mean": Fraction(),
        "ic_m2": Fraction(),
        "ic_positive": 0,
        "rankic_count": 0,
        "rankic_mean": Fraction(),
        "rankic_m2": Fraction(),
        "rankic_positive": 0,
    }
    row_count = 0
    squared_errors = _ScaledSquareAccumulator()
    current_timestamp: int | None = None
    timestamp_state: _TimestampMetricState | None = None
    for timestamp_ns, label, prediction, label_rank, prediction_rank in rows:
        timestamp_ns = int(timestamp_ns)
        if current_timestamp != timestamp_ns:
            if timestamp_state is not None:
                row_count += timestamp_state.count
                _add_correlation_aggregate(
                    stats, "ic", timestamp_state.correlation()[0]
                )
                _add_correlation_aggregate(
                    stats, "rankic", timestamp_state.rank.result()[0]
                )
            current_timestamp = timestamp_ns
            timestamp_state = _TimestampMetricState()
        assert timestamp_state is not None
        label = float(label)
        prediction = float(prediction)
        residual = prediction - label
        if not np.isfinite(residual):
            raise ValueError("model_selection_mse_unrepresentable")
        timestamp_state.update(
            label, prediction, float(label_rank), float(prediction_rank)
        )
        squared_errors.add(residual)
    if timestamp_state is not None:
        row_count += timestamp_state.count
        _add_correlation_aggregate(stats, "ic", timestamp_state.correlation()[0])
        _add_correlation_aggregate(stats, "rankic", timestamp_state.rank.result()[0])
    ic = _trial_correlation_summary(stats, "ic")
    rankic = _trial_correlation_summary(stats, "rankic")
    return (
        row_count,
        squared_errors.result(row_count),
        ic[0],
        ic[1],
        rankic[0],
        rankic[1],
    )


class _TimestampMetricState:
    __slots__ = ("count", "ic", "rank")

    def __init__(self) -> None:
        self.count = 0
        self.ic = _CorrelationMoments()
        self.rank = _CorrelationMoments()

    def update(
        self, label: float, prediction: float, label_rank: float, prediction_rank: float
    ) -> None:
        self.ic.update(label, prediction)
        self.rank.update(label_rank, prediction_rank)
        self.count += 1

    def correlation(self) -> tuple[float | None, str | None]:
        return self.ic.result()


class _ScaledSquareAccumulator:
    __slots__ = ("scale", "scaled_sum")

    def __init__(self) -> None:
        self.scale = 0.0
        self.scaled_sum = 0.0

    def add(self, value: float) -> None:
        magnitude = abs(value)
        if magnitude == 0.0:
            return
        if self.scale == 0.0:
            self.scale = magnitude
            self.scaled_sum = 1.0
            return
        if magnitude > self.scale:
            ratio = self.scale / magnitude
            self.scaled_sum = self.scaled_sum * ratio * ratio + 1.0
            self.scale = magnitude
            return
        self.scaled_sum += (magnitude / self.scale) ** 2

    def result(self, count: int) -> float:
        if count <= 0 or self.scale == 0.0:
            return 0.0
        mean_scaled_square = self.scaled_sum / count
        result = self.scale * (self.scale * mean_scaled_square)
        if result == 0.0 or not np.isfinite(result):
            raise ValueError("model_selection_mse_unrepresentable")
        return float(result)


def _trial_correlation_summary(
    stats: dict[str, int | Fraction], prefix: str
) -> tuple[float | None, str | None]:
    result = _summarize_correlation_aggregates(
        count=int(stats[f"{prefix}_count"]),
        mean=stats[f"{prefix}_mean"],
        m2=stats[f"{prefix}_m2"],
        positive_count=int(stats[f"{prefix}_positive"]),
    )
    if int(result["valid_count"]) == 0:
        return (None, str(result["reason"]))
    return (float(result["mean"]), None)


def _metric_chunk(
    frame: pd.DataFrame, *, label_column: str, prediction_column: str
) -> pd.DataFrame:
    missing = sorted(
        {"timestamp", "asset_id", label_column, prediction_column}.difference(
            frame.columns
        )
    )
    if missing:
        raise ValueError(f"missing_trial_metric_columns: {', '.join(missing)}")
    if frame.empty:
        return
    labels = frame[label_column]
    predictions = frame[prediction_column]
    for values in (labels, predictions):
        if not pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(
            values
        ):
            raise ValueError("invalid_trial_metric_values")
    label_values = labels.astype("float64")
    prediction_values = predictions.astype("float64")
    if (
        bool(label_values.isna().any())
        or bool(prediction_values.isna().any())
        or (not bool(np.isfinite(label_values).all()))
        or (not bool(np.isfinite(prediction_values).all()))
    ):
        raise ValueError("invalid_trial_metric_values")
    timestamps = strict_timestamp_series(
        frame["timestamp"], error="invalid_trial_metric_timestamp"
    )
    timestamp_ns = timestamps.astype("int64").to_numpy(copy=False)
    return pd.DataFrame(
        {
            "timestamp_ns": timestamp_ns,
            "label": label_values.to_numpy(dtype="float64", copy=False),
            "prediction": prediction_values.to_numpy(dtype="float64", copy=False),
        }
    )


def fit_validation_candidates(
    *,
    fold,
    candidate_specs,
    dataset_spec,
    chunks,
    validation_input_columns: tuple[str, ...] | None = None,
    trial_recorder: TrialRecorder,
    validation_coverage: ModelSampleCoverage,
    on_validation,
    on_candidate,
):
    """Fit every candidate once and publish its original validation scores."""
    label_columns = {spec.label_column for spec in candidate_specs}
    if len(label_columns) != 1:
        raise ValueError("model_selection_label_column_mismatch")
    label_column = candidate_specs[0].label_column
    common_validation_input_columns = (
        model_validation_inputs(candidate_specs)
        if validation_input_columns is None
        else validation_input_columns
    )
    validation_start = pd.Timestamp(fold.validation.start)
    validation_end = pd.Timestamp(fold.validation.end)
    for candidate_spec in candidate_specs:
        train_read_columns = fitting.model_training_read_columns(
            dataset_spec.primary_key,
            candidate_spec.input_columns,
            label_column=label_column,
            sample_filter_mode=candidate_spec.sample_filter_mode,
        )
        validation_read_columns = tuple(
            dict.fromkeys(
                (
                    *train_read_columns,
                    *common_validation_input_columns,
                    *model_input_validity_columns(
                        candidate_spec.input_columns, candidate_spec.model_algorithm
                    ),
                )
            )
        )

        attempt = (
            trial_recorder.begin(
                candidate_spec, common_coverage_id=validation_coverage.semantic_identity
            )
            if trial_recorder is not None
            else None
        )
        try:
            model_result = fitting._fit_model_result_from_frames(
                fitting.training_scope_chunks(
                    chunks(
                        train_read_columns,
                        input_columns=candidate_spec.input_columns,
                        start=None,
                        end=fold.train.end,
                    ),
                    asset_ids=(),
                ),
                candidate_spec,
                estimator=build_estimator_from_model_spec(candidate_spec),
                fitting_cutoff=pd.Timestamp(fold.train.end),
            )

            def validation_source(
                validation_columns: tuple[str, ...] = validation_read_columns,
                candidate_input_columns: tuple[str, ...] = candidate_spec.input_columns,
            ):
                return chunks(
                    validation_columns,
                    start=fold.validation.start,
                    end=fold.validation.end,
                    input_columns=tuple(
                        dict.fromkeys(
                            (*common_validation_input_columns, *candidate_input_columns)
                        )
                    ),
                )

            validation_chunks = iter_validation_prediction_chunks(
                validation_source,
                candidate_spec=candidate_spec,
                model_result=model_result,
                validation_start=validation_start,
                validation_end=validation_end,
                scoring_input_columns=common_validation_input_columns,
            )

            def observed_chunks(source=validation_chunks, candidate=candidate_spec):
                with closing_iterator(iter(source)) as frames:
                    for item in frames:
                        frame = item
                        on_validation(candidate, frame)
                        mature = frame.loc[
                            label_maturity_mask(
                                frame,
                                fitting_cutoff=validation_end,
                                availability_column="label_available_at",
                                target_column=label_column,
                            )
                            & frame["score_eligible"]
                        ]
                        try:
                            yield mature
                        except GeneratorExit:
                            return

            trial_metrics = evaluate_model_trial_metrics_from_chunks(
                observed_chunks(),
                model_id=candidate_spec.model_id,
                label_column=label_column,
                prediction_column="prediction_score",
                expected_coverage=validation_coverage,
            )
            trial_metrics.update(
                {
                    "resolved_hyperparameters": dict(
                        model_result.metadata["resolved_hyperparameters"]
                    ),
                    "random_seed": model_result.metadata["random_seed"],
                    "n_train": model_result.metadata["rows_after_input_label_dropna"],
                    "dependency_status": dict(
                        model_result.metadata["dependency_status"]
                    ),
                }
            )
            if trial_recorder is not None:
                trial_recorder.finish(
                    attempt, metrics=trial_metrics, model_metadata=model_result.metadata
                )
            on_candidate(candidate_spec, model_result)
        except BaseException as exc:
            if trial_recorder is not None and attempt is not None:
                try:
                    trial_recorder.finish(attempt, error=exc)
                except BaseException as record_error:
                    exc.add_note(
                        "trial failure record also failed: "
                        f"{type(record_error).__name__}"
                    )
            raise


def iter_validation_prediction_chunks(
    chunks: Callable[[], Iterable[pd.DataFrame]],
    *,
    candidate_spec,
    model_result: fitting.ModelFitResult,
    validation_start: pd.Timestamp,
    validation_end: pd.Timestamp,
    scoring_input_columns: tuple[str, ...] | None = None,
    extra_output_columns: tuple[str, ...] = (),
) -> Iterable[object]:
    predictor = prediction_callable(
        candidate_spec.input_columns, model_result.prediction_model
    )
    prediction_inputs = model_input_validity_columns(
        candidate_spec.input_columns, candidate_spec.model_algorithm
    )
    output_columns = (
        "timestamp",
        "asset_id",
        candidate_spec.label_column,
        *extra_output_columns,
    )
    complete_frames = iter_complete_time_frames(
        iter(chunks()), timestamp_column="timestamp"
    )
    with closing_iterator(complete_frames) as source:
        for chunk in source:
            timestamps = strict_timestamp_series(
                chunk["timestamp"], error="invalid_prediction_timestamp_values"
            )
            validation = chunk.loc[
                (timestamps >= validation_start) & (timestamps <= validation_end)
            ]
            if not validation.empty:
                filtered, _filter_metadata = apply_filter_mode(
                    validation,
                    mode=candidate_spec.sample_filter_mode,
                    metadata_key="sample_filter_mode",
                )
                availability_column = resolve_training_availability_column(
                    filtered, label_column=candidate_spec.label_column
                )
                prediction_rows = filtered.loc[
                    filtered.loc[:, list(prediction_inputs)].notna().all(axis=1)
                ]
                usable = prediction_rows.loc[prediction_rows["eligible"]]
                if not usable.empty:
                    predictions = pd.Series(
                        predictor(prediction_rows), index=prediction_rows.index
                    )
                    output = usable.loc[:, list(output_columns)].copy()
                    output["label_available_at"] = usable[availability_column]
                    output["prediction_score"] = predictions.loc[usable.index]
                    if scoring_input_columns is not None:
                        output["score_eligible"] = (
                            usable.loc[:, list(scoring_input_columns)]
                            .notna()
                            .all(axis=1)
                        )
                    try:
                        yield output
                    except GeneratorExit:
                        return

"""Validation-metric-only model trial selection helpers."""

from __future__ import annotations

import sqlite3
import tempfile
from collections.abc import Collection, Iterable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from quant_research_platform.contracts.predictions import (
    MODEL_SELECTION_METRIC_DIRECTIONS,
)
from quant_research_platform.contracts.validation import strict_timestamp_series
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.runtime.resources import (
    _OwnedValue,
    current_runtime_resource_policy,
    require_free_disk_headroom,
)

VALIDATION_METRIC_COLUMNS = ("validation_ic", "validation_rankic", "validation_mse")
VALIDATION_METRIC_REASON_COLUMNS = (
    "validation_ic_reason",
    "validation_rankic_reason",
)
TEST_METRIC_COLUMNS = ("test_ic", "test_rankic", "test_mse")
_MODEL_SELECTION_SQLITE_CACHE_BYTES = 64 * 1024
_MODEL_SELECTION_SQLITE_PAGE_BYTES = 4096
_MODEL_SELECTION_SQLITE_ROW_DISK_BYTES = 512
_MODEL_SELECTION_SQLITE_STARTUP_BYTES = 64 * _MODEL_SELECTION_SQLITE_PAGE_BYTES
_MODEL_SELECTION_METRIC_RESERVE_BYTES = 256 * 1024
_MODEL_SELECTION_METRIC_BATCH_ROWS = 4096


@contextmanager
def _model_selection_scratch() -> Iterator[str]:
    scratch = tempfile.TemporaryDirectory(prefix="qrp-model-selection-")
    scratch_path = scratch.__enter__()
    try:
        yield scratch_path
    except BaseException as primary:
        try:
            scratch.__exit__(type(primary), primary, primary.__traceback__)
        except BaseException as cleanup_error:
            primary.add_note(
                "model selection scratch cleanup also failed: "
                f"{type(cleanup_error).__name__}"
            )
        raise
    else:
        scratch.__exit__(None, None, None)


@contextmanager
def _model_selection_database(
    scratch_path: str,
) -> Iterator[sqlite3.Connection]:
    database = sqlite3.connect(f"{scratch_path}/validation.sqlite")
    try:
        yield database
    except BaseException as primary:
        try:
            database.close()
        except BaseException as cleanup_error:
            primary.add_note(
                "model selection database close also failed: "
                f"{type(cleanup_error).__name__}"
            )
        raise
    else:
        database.close()


@dataclass(slots=True)
class _MetricAccumulator:
    count: int = 0
    label_mean: float = 0.0
    prediction_mean: float = 0.0
    label_m2: float = 0.0
    prediction_m2: float = 0.0
    covariance: float = 0.0
    squared_error: float = 0.0
    label_scale: float = 1.0
    prediction_scale: float = 1.0
    label_min: float | None = None
    label_max: float | None = None
    prediction_min: float | None = None
    prediction_max: float | None = None

    def add(
        self,
        labels: np.ndarray,
        predictions: np.ndarray,
    ) -> None:
        batch_count = len(labels)
        if batch_count == 0:
            return
        batch_label_min = float(labels.min())
        batch_label_max = float(labels.max())
        batch_prediction_min = float(predictions.min())
        batch_prediction_max = float(predictions.max())
        label_scale = max(abs(batch_label_min), abs(batch_label_max))
        prediction_scale = max(abs(batch_prediction_min), abs(batch_prediction_max))
        if label_scale == 0:
            label_scale = 1.0
        if prediction_scale == 0:
            prediction_scale = 1.0
        if self.count == 0:
            self.label_scale = label_scale
            self.prediction_scale = prediction_scale
        else:
            previous_label_scale = self.label_scale
            previous_prediction_scale = self.prediction_scale
            if (
                label_scale < previous_label_scale
                and self.label_min == self.label_max == 0
            ):
                self.label_scale = label_scale
            elif label_scale > previous_label_scale:
                factor = previous_label_scale / label_scale
                if self.label_mean:
                    self.label_mean *= factor
                if self.label_m2:
                    self.label_m2 *= factor * factor
                self.label_scale = label_scale
            if (
                prediction_scale < previous_prediction_scale
                and self.prediction_min == self.prediction_max == 0
            ):
                self.prediction_scale = prediction_scale
            elif prediction_scale > previous_prediction_scale:
                factor = previous_prediction_scale / prediction_scale
                if self.prediction_mean:
                    self.prediction_mean *= factor
                if self.prediction_m2:
                    self.prediction_m2 *= factor * factor
                self.prediction_scale = prediction_scale
            if self.covariance:
                self.covariance *= (
                    previous_label_scale / self.label_scale
                    * previous_prediction_scale
                    / self.prediction_scale
                )
        normalized_labels = labels / self.label_scale
        normalized_predictions = predictions / self.prediction_scale
        label_mean = float(normalized_labels.mean())
        prediction_mean = float(normalized_predictions.mean())
        centered_labels = normalized_labels - label_mean
        centered_predictions = normalized_predictions - prediction_mean
        label_m2 = float(np.dot(centered_labels, centered_labels))
        prediction_m2 = float(np.dot(centered_predictions, centered_predictions))
        covariance = float(np.dot(centered_labels, centered_predictions))
        self.squared_error += float(np.square(predictions - labels).sum())
        if self.count == 0:
            self.count = batch_count
            self.label_mean = label_mean
            self.prediction_mean = prediction_mean
            self.label_m2 = label_m2
            self.prediction_m2 = prediction_m2
            self.covariance = covariance
        else:
            previous_count = self.count
            total_count = previous_count + batch_count
            overlap = previous_count * batch_count / total_count
            label_delta = label_mean - self.label_mean
            prediction_delta = prediction_mean - self.prediction_mean
            self.label_m2 += label_m2 + label_delta * label_delta * overlap
            self.prediction_m2 += (
                prediction_m2 + prediction_delta * prediction_delta * overlap
            )
            self.covariance += covariance + label_delta * prediction_delta * overlap
            self.label_mean += label_delta * batch_count / total_count
            self.prediction_mean += prediction_delta * batch_count / total_count
            self.count = total_count
        self.label_min = (
            batch_label_min
            if self.label_min is None
            else min(self.label_min, batch_label_min)
        )
        self.label_max = (
            batch_label_max
            if self.label_max is None
            else max(self.label_max, batch_label_max)
        )
        self.prediction_min = (
            batch_prediction_min
            if self.prediction_min is None
            else min(self.prediction_min, batch_prediction_min)
        )
        self.prediction_max = (
            batch_prediction_max
            if self.prediction_max is None
            else max(self.prediction_max, batch_prediction_max)
        )

    def correlation(self) -> tuple[float | None, str | None]:
        if self.count < 2:
            return None, "insufficient_observations"
        if (
            self.label_min == self.label_max
            or self.prediction_min == self.prediction_max
        ):
            return None, "zero_variance"
        denominator = np.sqrt(self.label_m2 * self.prediction_m2)
        if denominator == 0:
            return None, "zero_variance"
        if not np.isfinite(denominator):
            return None, "nonfinite_result"
        value = float(self.covariance / denominator)
        if not np.isfinite(value):
            return None, "nonfinite_result"
        return value, None

    @property
    def mean_squared_error(self) -> float:
        return float(self.squared_error / self.count)


def validate_trial_metric_columns(
    *,
    selection_metric: str,
    validation_metric_columns: Collection[str],
    test_metric_columns: Collection[str],
) -> None:
    """Ensure model selection uses validation metrics only."""

    validation_metrics = set(validation_metric_columns)
    test_metrics = set(test_metric_columns)
    if selection_metric not in validation_metrics:
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
        metric_values = pd.Series(
            np.nan,
            index=metric_series.index,
            dtype="float64",
        )
    else:
        metric_values = metric_series.astype("float64")
    if bool((~metric_values.isna() & ~np.isfinite(metric_values)).any()):
        raise ValueError("invalid_trial_selection_metric")
    defined = metric_values.notna()
    if not bool(defined.any()):
        raise ValueError("all_validation_metrics_undefined")

    ordered = (
        trials.loc[defined]
        .assign(_selection_metric=metric_values[defined])
        .sort_values(
            ["_selection_metric", trial_id_column],
            ascending=[not maximize, True],
            kind="mergesort",
        )
    )
    return str(ordered.iloc[0][trial_id_column])


def evaluate_model_trial_metrics_from_chunks(
    chunks: Iterable[pd.DataFrame | _OwnedValue[pd.DataFrame]],
    *,
    model_id: str,
    label_column: str,
    prediction_column: str,
    purge_gap_steps: int,
) -> dict[str, object]:
    """Evaluate model-selection metrics from bounded prediction chunks."""

    policy = current_runtime_resource_policy()
    with ExitStack() as operation:
        if policy is not None:
            operation.enter_context(
                policy.lease(
                    "research.model.selection.metric",
                    _MODEL_SELECTION_METRIC_RESERVE_BYTES,
                )
            )
        with _model_selection_scratch() as scratch:
            with _model_selection_database(scratch) as database:
                database.execute(
                    f"PRAGMA page_size={_MODEL_SELECTION_SQLITE_PAGE_BYTES}"
                )
                database.execute("PRAGMA temp_store=FILE")
                database.execute(
                    f"PRAGMA cache_size=-{_MODEL_SELECTION_SQLITE_CACHE_BYTES // 1024}"
                )
                database.execute("PRAGMA synchronous=NORMAL")
                _admit_model_selection_disk_growth(
                    database,
                    Path(scratch),
                    row_count=0,
                )
                database.execute(
                    "CREATE TABLE validation_rows ("
                    "row_id INTEGER PRIMARY KEY, "
                    "asset_id, "
                    "timestamp_ns INTEGER NOT NULL, "
                    "label REAL NOT NULL, "
                    "prediction REAL NOT NULL"
                    ")"
                )
                source = iter(chunks)
                with closing_iterator(source):
                    for item in source:
                        owned_lease = (
                            item.lease if isinstance(item, _OwnedValue) else None
                        )
                        frame = item.value if isinstance(item, _OwnedValue) else item
                        if frame.empty:
                            if owned_lease is not None:
                                owned_lease.close()
                            continue
                        try:
                            _admit_model_selection_disk_growth(
                                database,
                                Path(scratch),
                                row_count=len(frame),
                            )
                            _insert_metric_chunk(
                                database,
                                frame,
                                label_column=label_column,
                                prediction_column=prediction_column,
                            )
                        except BaseException as primary:
                            if owned_lease is not None:
                                try:
                                    owned_lease.close()
                                except BaseException as cleanup_error:
                                    primary.add_note(
                                        "metric chunk lease release also failed: "
                                        f"{type(cleanup_error).__name__}"
                                    )
                            raise
                        else:
                            if owned_lease is not None:
                                owned_lease.close()
                database.commit()
                validation_row_count = int(
                    database.execute(
                        "SELECT COUNT(*) FROM validation_rows"
                    ).fetchone()[0]
                )
                if validation_row_count == 0:
                    raise ValueError("empty_model_selection_validation_window")
                _admit_model_selection_disk_growth(
                    database,
                    Path(scratch),
                    row_count=validation_row_count,
                )
                if purge_gap_steps > 0:
                    database.execute(
                        "DELETE FROM validation_rows "
                        "WHERE row_id IN ("
                        "SELECT row_id FROM ("
                        "SELECT row_id, DENSE_RANK() OVER ("
                        "PARTITION BY asset_id ORDER BY timestamp_ns DESC"
                        ") AS trailing_rank FROM validation_rows"
                        ") WHERE trailing_rank <= ?"
                        ")",
                        (purge_gap_steps,),
                    )
                if database.execute(
                    "SELECT 1 FROM validation_rows LIMIT 1"
                ).fetchone() is None:
                    raise ValueError("empty_model_selection_validation_window")
                raw = _MetricAccumulator()
                for rows in _metric_batches(
                    database,
                    "SELECT label, prediction FROM validation_rows ORDER BY row_id",
                ):
                    raw.add(
                        np.fromiter(
                            (row[0] for row in rows),
                            dtype="float64",
                            count=len(rows),
                        ),
                        np.fromiter(
                            (row[1] for row in rows),
                            dtype="float64",
                            count=len(rows),
                        ),
                    )
                rank = _MetricAccumulator()
                rank_query = (
                    "SELECT label_rank, prediction_rank FROM ("
                    "SELECT row_id, "
                    "CAST(RANK() OVER (ORDER BY label) AS REAL) + "
                    "(CAST(COUNT(*) OVER (PARTITION BY label) AS REAL) - 1.0) "
                    "/ 2.0 "
                    "AS label_rank, "
                    "CAST(RANK() OVER (ORDER BY prediction) AS REAL) + "
                    "(CAST(COUNT(*) OVER (PARTITION BY prediction) AS REAL) - 1.0) "
                    "/ 2.0 "
                    "AS prediction_rank FROM validation_rows"
                    ") ORDER BY row_id"
                )
                for rows in _metric_batches(database, rank_query):
                    rank.add(
                        np.fromiter(
                            (row[0] for row in rows),
                            dtype="float64",
                            count=len(rows),
                        ),
                        np.fromiter(
                            (row[1] for row in rows),
                            dtype="float64",
                            count=len(rows),
                        ),
                    )
                validation_ic, validation_ic_reason = raw.correlation()
                validation_rankic, validation_rankic_reason = rank.correlation()
                return {
                    "model_id": model_id,
                    "validation_rows": raw.count,
                    "validation_ic": validation_ic,
                    "validation_ic_reason": validation_ic_reason,
                    "validation_rankic": validation_rankic,
                    "validation_rankic_reason": validation_rankic_reason,
                    "validation_mse": raw.mean_squared_error,
                }


def _admit_model_selection_disk_growth(
    database: sqlite3.Connection,
    scratch_path: Path,
    *,
    row_count: int,
) -> None:
    growth_bytes = max(
        _MODEL_SELECTION_SQLITE_STARTUP_BYTES,
        max(1, row_count) * _MODEL_SELECTION_SQLITE_ROW_DISK_BYTES,
    )
    require_free_disk_headroom(
        scratch_path,
        reserve_bytes=2 * growth_bytes,
    )
    growth_pages = (
        growth_bytes + _MODEL_SELECTION_SQLITE_PAGE_BYTES - 1
    ) // _MODEL_SELECTION_SQLITE_PAGE_BYTES
    for schema in ("main", "temp"):
        current_pages = int(
            database.execute(f"PRAGMA {schema}.page_count").fetchone()[0]
        )
        if row_count > 0 and schema == "temp":
            current_pages = int(
                database.execute("PRAGMA temp.max_page_count").fetchone()[0]
            )
        database.execute(
            f"PRAGMA {schema}.max_page_count={current_pages + growth_pages}"
        )


def _insert_metric_chunk(
    database: sqlite3.Connection,
    frame: pd.DataFrame,
    *,
    label_column: str,
    prediction_column: str,
) -> None:
    missing = sorted(
        {
            "timestamp",
            "asset_id",
            label_column,
            prediction_column,
        }.difference(frame.columns)
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
        or not bool(np.isfinite(label_values).all())
        or not bool(np.isfinite(prediction_values).all())
    ):
        raise ValueError("invalid_trial_metric_values")
    timestamps = strict_timestamp_series(
        frame["timestamp"],
        error="invalid_trial_metric_timestamp",
    )
    timestamp_ns = timestamps.astype("int64").to_numpy(copy=False)
    assets = frame["asset_id"].to_numpy(copy=False)
    database.executemany(
        "INSERT INTO validation_rows(asset_id, timestamp_ns, label, prediction) "
        "VALUES (?, ?, ?, ?)",
        (
            (
                asset,
                int(timestamp),
                float(label),
                float(prediction),
            )
            for asset, timestamp, label, prediction in zip(
                assets,
                timestamp_ns,
                label_values.to_numpy(dtype="float64", copy=False),
                prediction_values.to_numpy(dtype="float64", copy=False),
                strict=True,
            )
        ),
    )


def _metric_batches(
    database: sqlite3.Connection,
    query: str,
) -> Iterable[list[tuple[float, float]]]:
    cursor = database.execute(query)
    try:
        while rows := cursor.fetchmany(_MODEL_SELECTION_METRIC_BATCH_ROWS):
            yield rows
    finally:
        cursor.close()


def build_model_selection_metadata(
    *,
    configured_model_id: str,
    selected_model_id: str,
    selected_model_input_columns: tuple[str, ...],
    candidate_model_ids: tuple[str, ...],
    selection_metric: str,
    validation_start: str,
    validation_end: str,
    trial_metrics: pd.DataFrame,
) -> dict[str, object]:
    """Build prediction model selection metadata from validation metrics."""

    if selection_metric not in MODEL_SELECTION_METRIC_DIRECTIONS:
        raise ValueError("invalid_model_selection_metric")
    if selected_model_id not in candidate_model_ids:
        raise ValueError("selected_model_not_in_candidates")
    records = trial_metrics.loc[
        :, ("model_id", *VALIDATION_METRIC_COLUMNS, *VALIDATION_METRIC_REASON_COLUMNS)
    ].to_dict(
        "records"
    )
    for record in records:
        for key in (
            "validation_ic",
            "validation_rankic",
            "validation_ic_reason",
            "validation_rankic_reason",
        ):
            if pd.isna(record[key]):
                record[key] = None
        record["validation_rows"] = int(
            trial_metrics.loc[
                trial_metrics["model_id"] == record["model_id"],
                "validation_rows",
            ].iloc[0]
        )
    return {
        "status": "selected",
        "configured_model_id": configured_model_id,
        "selected_model_id": selected_model_id,
        "selected_model_input_columns": list(selected_model_input_columns),
        "candidate_model_ids": list(candidate_model_ids),
        "selection_metric": selection_metric,
        "selection_direction": MODEL_SELECTION_METRIC_DIRECTIONS[selection_metric],
        "validation_start": validation_start,
        "validation_end": validation_end,
        "candidate_count": len(candidate_model_ids),
        "trial_metrics": records,
    }

"""Validation-metric-only model trial selection helpers."""

from __future__ import annotations

import sqlite3
import tempfile
from collections.abc import Collection, Iterable, Iterator
from contextlib import ExitStack, closing, contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

from quant_research_platform.compute.dataframe.cross_section import (
    _add_correlation_aggregate,
    _pearson_correlation_from_moments,
    _summarize_correlation_aggregates,
)
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
_CROSS_SECTIONAL_METRIC_QUERY = """
WITH ranked AS (
    SELECT
        row_id,
        timestamp_ns,
        label,
        prediction,
        FIRST_VALUE(label) OVER (
            PARTITION BY timestamp_ns ORDER BY row_id
        ) AS label_reference,
        FIRST_VALUE(prediction) OVER (
            PARTITION BY timestamp_ns ORDER BY row_id
        ) AS prediction_reference,
        RANK() OVER (
            PARTITION BY timestamp_ns ORDER BY label
        ) AS label_rank,
        RANK() OVER (
            PARTITION BY timestamp_ns ORDER BY prediction
        ) AS prediction_rank,
        COUNT(*) OVER (
            PARTITION BY timestamp_ns, label
        ) AS label_tie_count,
        COUNT(*) OVER (
            PARTITION BY timestamp_ns, prediction
        ) AS prediction_tie_count,
        COUNT(*) OVER (
            PARTITION BY timestamp_ns
        ) AS timestamp_count
    FROM validation_rows
), differences AS (
    SELECT
        *,
        label - label_reference AS label_difference,
        prediction - prediction_reference AS prediction_difference
    FROM ranked
), scaled AS (
    SELECT
        *,
        MAX(ABS(label_difference)) OVER (
            PARTITION BY timestamp_ns
        ) AS label_difference_scale,
        MAX(ABS(prediction_difference)) OVER (
            PARTITION BY timestamp_ns
        ) AS prediction_difference_scale,
        MAX(ABS(label)) OVER (
            PARTITION BY timestamp_ns
        ) AS label_raw_scale,
        MAX(ABS(prediction)) OVER (
            PARTITION BY timestamp_ns
        ) AS prediction_raw_scale
    FROM differences
), normalized AS (
    SELECT
        timestamp_ns,
        label,
        prediction,
        CASE
            WHEN label_difference_scale = 0 THEN 0.0
            WHEN label_difference_scale <= 1.7976931348623157e308
                THEN label_difference / label_difference_scale
            ELSE label / label_raw_scale
        END AS label_value,
        CASE
            WHEN prediction_difference_scale = 0 THEN 0.0
            WHEN prediction_difference_scale <= 1.7976931348623157e308
                THEN prediction_difference / prediction_difference_scale
            ELSE prediction / prediction_raw_scale
        END AS prediction_value,
        (
            CAST(label_rank AS REAL)
            + (CAST(label_tie_count AS REAL) - 1.0) / 2.0
        ) / CAST(timestamp_count AS REAL) AS label_rank_value,
        (
            CAST(prediction_rank AS REAL)
            + (CAST(prediction_tie_count AS REAL) - 1.0) / 2.0
        ) / CAST(timestamp_count AS REAL) AS prediction_rank_value
    FROM scaled
)
SELECT
    timestamp_ns,
    COUNT(*) AS observation_count,
    SUM((prediction - label) * (prediction - label)) AS squared_error,
    SUM(label_value) AS label_sum,
    SUM(prediction_value) AS prediction_sum,
    SUM(label_value * label_value) AS label_square_sum,
    SUM(prediction_value * prediction_value) AS prediction_square_sum,
    SUM(label_value * prediction_value) AS label_prediction_sum,
    SUM(label_rank_value) AS label_rank_sum,
    SUM(prediction_rank_value) AS prediction_rank_sum,
    SUM(label_rank_value * label_rank_value) AS label_rank_square_sum,
    SUM(prediction_rank_value * prediction_rank_value) AS prediction_rank_square_sum,
    SUM(label_rank_value * prediction_rank_value) AS label_prediction_rank_sum
FROM normalized
GROUP BY timestamp_ns
ORDER BY timestamp_ns
"""


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
                (
                    validation_rows,
                    validation_mse,
                    validation_ic,
                    validation_ic_reason,
                    validation_rankic,
                    validation_rankic_reason,
                ) = _cross_sectional_metric_summary(database)
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
    database: sqlite3.Connection,
) -> tuple[int, float, float | None, str | None, float | None, str | None]:
    stats: dict[str, float | int] = {
        "ic_count": 0,
        "ic_mean": 0.0,
        "ic_m2": 0.0,
        "ic_positive": 0,
        "ic_min": None,
        "ic_max": None,
        "rankic_count": 0,
        "rankic_mean": 0.0,
        "rankic_m2": 0.0,
        "rankic_positive": 0,
        "rankic_min": None,
        "rankic_max": None,
    }
    row_count = 0
    squared_error = 0.0
    with closing(database.execute(_CROSS_SECTIONAL_METRIC_QUERY)) as rows:
        for row in rows:
            (
                _timestamp,
                timestamp_rows,
                timestamp_squared_error,
                label_sum,
                prediction_sum,
                label_square_sum,
                prediction_square_sum,
                label_prediction_sum,
                label_rank_sum,
                prediction_rank_sum,
                label_rank_square_sum,
                prediction_rank_square_sum,
                label_prediction_rank_sum,
            ) = row
            count = int(timestamp_rows)
            row_count += count
            squared_error += float(timestamp_squared_error)
            ic = _pearson_correlation_from_sums(
                count=count,
                left_sum=float(label_sum),
                right_sum=float(prediction_sum),
                left_square_sum=float(label_square_sum),
                right_square_sum=float(prediction_square_sum),
                cross_sum=float(label_prediction_sum),
            )[0]
            rankic = _pearson_correlation_from_sums(
                count=count,
                left_sum=float(label_rank_sum),
                right_sum=float(prediction_rank_sum),
                left_square_sum=float(label_rank_square_sum),
                right_square_sum=float(prediction_rank_square_sum),
                cross_sum=float(label_prediction_rank_sum),
            )[0]
            _add_correlation_aggregate(stats, "ic", ic)
            _add_correlation_aggregate(stats, "rankic", rankic)
    ic = _trial_correlation_summary(stats, "ic")
    rankic = _trial_correlation_summary(stats, "rankic")
    return (
        row_count,
        squared_error / row_count,
        ic[0],
        ic[1],
        rankic[0],
        rankic[1],
    )


def _pearson_correlation_from_sums(
    *,
    count: int,
    left_sum: float,
    right_sum: float,
    left_square_sum: float,
    right_square_sum: float,
    cross_sum: float,
) -> tuple[float | None, str | None]:
    return _pearson_correlation_from_moments(
        count=count,
        left_m2=left_square_sum - left_sum * left_sum / count,
        right_m2=right_square_sum - right_sum * right_sum / count,
        covariance=cross_sum - left_sum * right_sum / count,
    )


def _trial_correlation_summary(
    stats: dict[str, float | int],
    prefix: str,
) -> tuple[float | None, str | None]:
    result = _summarize_correlation_aggregates(
        count=int(stats[f"{prefix}_count"]),
        mean=float(stats[f"{prefix}_mean"]),
        m2=float(stats[f"{prefix}_m2"]),
        positive_count=int(stats[f"{prefix}_positive"]),
        minimum=stats[f"{prefix}_min"],
        maximum=stats[f"{prefix}_max"],
    )
    if int(result["valid_count"]) == 0:
        return None, str(result["reason"])
    return float(result["mean"]), None


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

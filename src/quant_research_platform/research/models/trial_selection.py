"""Validation-metric-only model trial selection helpers."""

from __future__ import annotations

import hashlib
import sqlite3
import tempfile
from collections.abc import Callable, Collection, Iterable, Iterator
from contextlib import ExitStack, closing, contextmanager
from fractions import Fraction
from math import log
from pathlib import Path

import numpy as np
import pandas as pd

from quant_research_platform.compute.dataframe.cross_section import (
    _add_correlation_aggregate,
    _CorrelationMoments,
    _summarize_correlation_aggregates,
)
from quant_research_platform.contracts.experiments import (
    legal_training_rows,
    resolve_training_availability_column,
    training_availability_columns,
)
from quant_research_platform.contracts.filtering import required_filter_columns
from quant_research_platform.contracts.json_codec import canonical_json_hash
from quant_research_platform.contracts.models import (
    MODEL_SELECTION_METRIC_DIRECTIONS,
    _model_algorithm_facts,
)
from quant_research_platform.contracts.validation import strict_timestamp_series
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.research.experiments import TrialRecorder
from quant_research_platform.research.features.transforms import (
    canonical_feature_validity_columns,
)
from quant_research_platform.research.filtering import (
    _filter_working_set_bound,
    apply_filter_mode,
)
from quant_research_platform.research.models import fitting
from quant_research_platform.research.models.fitting import prediction_callable
from quant_research_platform.research.models.supervised import (
    build_estimator_from_model_spec,
)
from quant_research_platform.runtime.dataframes import iter_complete_time_sized_frames
from quant_research_platform.runtime.resources import (
    OwnedValue,
    current_runtime_resource_policy,
    pandas_frame_payload_bytes,
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
)
SELECT
    timestamp_ns,
    label,
    prediction,
    (
        CAST(label_rank AS REAL)
        + (CAST(label_tie_count AS REAL) - 1.0) / 2.0
    ) / CAST(timestamp_count AS REAL) AS label_rank_value,
    (
        CAST(prediction_rank AS REAL)
        + (CAST(prediction_tie_count AS REAL) - 1.0) / 2.0
    ) / CAST(timestamp_count AS REAL) AS prediction_rank_value
FROM ranked
ORDER BY timestamp_ns, row_id
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
            for column in (
                *canonical_feature_validity_columns(spec.input_columns),
                *(
                    spec.input_columns
                    if _model_algorithm_facts(spec.model_algorithm).input_kind
                    != "rank_ensemble_inputs"
                    else ()
                ),
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
            if pd.isna(timestamp) or type(asset) is not str or not asset:
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
    chunks: Iterable[pd.DataFrame | OwnedValue[pd.DataFrame]],
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
                            item.lease if isinstance(item, OwnedValue) else None
                        )
                        frame = item.value if isinstance(item, OwnedValue) else item
                        if frame.empty:
                            if owned_lease is not None:
                                owned_lease.close()
                            continue
                        try:
                            if coverage is not None:
                                coverage.add(frame)
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
                if coverage is not None and (
                    coverage.semantic_identity != expected_coverage.semantic_identity
                    or coverage.row_count != expected_coverage.row_count
                ):
                    raise ValueError("model_trial_common_coverage_mismatch")
                database.commit()
                validation_row_count = int(
                    database.execute("SELECT COUNT(*) FROM validation_rows").fetchone()[
                        0
                    ]
                )
                if validation_row_count == 0:
                    raise ValueError("empty_model_selection_validation_window")
                _admit_model_selection_disk_growth(
                    database,
                    Path(scratch),
                    row_count=validation_row_count,
                )
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
    with closing(database.execute(_CROSS_SECTIONAL_METRIC_QUERY)) as rows:
        for (
            timestamp_ns,
            label,
            prediction,
            label_rank,
            prediction_rank,
        ) in rows:
            timestamp_ns = int(timestamp_ns)
            if current_timestamp != timestamp_ns:
                if timestamp_state is not None:
                    row_count += timestamp_state.count
                    _add_correlation_aggregate(
                        stats,
                        "ic",
                        timestamp_state.correlation()[0],
                    )
                    _add_correlation_aggregate(
                        stats,
                        "rankic",
                        timestamp_state.rank.result()[0],
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
                label,
                prediction,
                float(label_rank),
                float(prediction_rank),
            )
            squared_errors.add(residual)
    if timestamp_state is not None:
        row_count += timestamp_state.count
        _add_correlation_aggregate(stats, "ic", timestamp_state.correlation()[0])
        _add_correlation_aggregate(
            stats,
            "rankic",
            timestamp_state.rank.result()[0],
        )
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
    __slots__ = (
        "count",
        "ic",
        "rank",
    )

    def __init__(self) -> None:
        self.count = 0
        self.ic = _CorrelationMoments()
        self.rank = _CorrelationMoments()

    def update(
        self,
        label: float,
        prediction: float,
        label_rank: float,
        prediction_rank: float,
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
        log_result = 2.0 * log(self.scale) + log(mean_scaled_square)
        if log_result > log(float(np.finfo(np.float64).max)) or log_result < log(
            float(np.nextafter(0.0, 1.0))
        ):
            raise ValueError("model_selection_mse_unrepresentable")
        result = self.scale * (self.scale * mean_scaled_square)
        if result == 0.0 or not np.isfinite(result):
            raise ValueError("model_selection_mse_unrepresentable")
        return float(result)


def _trial_correlation_summary(
    stats: dict[str, int | Fraction],
    prefix: str,
) -> tuple[float | None, str | None]:
    result = _summarize_correlation_aggregates(
        count=int(stats[f"{prefix}_count"]),
        mean=stats[f"{prefix}_mean"],
        m2=stats[f"{prefix}_m2"],
        positive_count=int(stats[f"{prefix}_positive"]),
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
    tie_break_order: tuple[str, ...] = (),
) -> dict[str, object]:
    """Build prediction model selection metadata from validation metrics."""

    if selection_metric not in MODEL_SELECTION_METRIC_DIRECTIONS:
        raise ValueError("invalid_model_selection_metric")
    if selected_model_id not in candidate_model_ids:
        raise ValueError("selected_model_not_in_candidates")
    tie_break_order = tie_break_order or candidate_model_ids
    if set(tie_break_order) != set(candidate_model_ids):
        raise ValueError("trial_tie_break_order_mismatch")
    missing_fact_columns = sorted(
        {
            "resolved_hyperparameters",
            "random_seed",
            "n_train",
            "dependency_status",
        }.difference(trial_metrics.columns)
    )
    if missing_fact_columns:
        raise ValueError(
            "model_selection_trial_facts_missing: " + ", ".join(missing_fact_columns)
        )
    records = trial_metrics.loc[
        :, ("model_id", *VALIDATION_METRIC_COLUMNS, *VALIDATION_METRIC_REASON_COLUMNS)
    ].to_dict("records")
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
        source = trial_metrics.loc[
            trial_metrics["model_id"] == record["model_id"]
        ].iloc[0]
        for key in (
            "resolved_hyperparameters",
            "random_seed",
            "n_train",
            "dependency_status",
        ):
            value = source[key]
            if hasattr(value, "item"):
                value = value.item()
            if key == "random_seed" and pd.isna(value):
                value = None
            if key == "dependency_status" and type(value) is dict:
                value = dict(value)
            record[key] = value
    return {
        "status": "selected",
        "configured_model_id": configured_model_id,
        "selected_model_id": selected_model_id,
        "selected_model_input_columns": list(selected_model_input_columns),
        "candidate_model_ids": list(candidate_model_ids),
        "tie_break_order": list(tie_break_order),
        "selection_metric": selection_metric,
        "selection_direction": MODEL_SELECTION_METRIC_DIRECTIONS[selection_metric],
        "validation_start": validation_start,
        "validation_end": validation_end,
        "candidate_count": len(candidate_model_ids),
        "trial_metrics": records,
    }


def select_prediction_model_spec(
    *,
    prediction_spec,
    dataset_spec,
    model_catalog,
    chunks,
    validation_input_columns: tuple[str, ...] | None = None,
    trial_recorder: TrialRecorder | None = None,
    validation_coverage: ModelSampleCoverage | None = None,
):
    """Select one model spec from validation trials for model-owned callers."""

    candidate_specs = tuple(
        model_catalog.get(model_id)
        for model_id in (prediction_spec.model_id, *prediction_spec.candidate_model_ids)
    )
    label_columns = {spec.label_column for spec in candidate_specs}
    if len(label_columns) != 1:
        raise ValueError("model_selection_label_column_mismatch")
    label_column = candidate_specs[0].label_column
    common_validation_input_columns = (
        model_validation_inputs(candidate_specs)
        if validation_input_columns is None
        else validation_input_columns
    )
    validation_start = pd.Timestamp(prediction_spec.model_selection_validation_start)
    validation_end = pd.Timestamp(prediction_spec.model_selection_validation_end)
    trial_metric_records: list[dict[str, object]] = []
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
                    *dataset_spec.primary_key,
                    *common_validation_input_columns,
                    *candidate_spec.input_columns,
                    *canonical_feature_validity_columns(candidate_spec.input_columns),
                    label_column,
                    *training_availability_columns(label_column),
                    *required_filter_columns(prediction_spec.sample_filter_mode),
                )
            )
        )
        validation_chunks = None
        cleanup_owner = None
        cleanup_lease = None
        primary_error: BaseException | None = None
        owned_model = model_result = model_lease = None
        attempt = (
            trial_recorder.begin(
                candidate_spec,
                common_coverage_id=validation_coverage.semantic_identity,
            )
            if trial_recorder is not None
            else None
        )
        try:
            owned_model = fitting._fit_model_result_owned_from_frames(  # noqa: SLF001
                fitting.training_scope_chunks(
                    chunks(
                        train_read_columns,
                        input_columns=candidate_spec.input_columns,
                        start=None,
                        end=prediction_spec.train_end,
                    ),
                    asset_ids=prediction_spec.training_asset_ids,
                ),
                candidate_spec,
                estimator=build_estimator_from_model_spec(candidate_spec),
                fitting_cutoff=pd.Timestamp(prediction_spec.train_end),
            )
            model_result = (
                owned_model.value
                if isinstance(owned_model, OwnedValue)
                else owned_model
            )
            model_lease = (
                owned_model.lease if isinstance(owned_model, OwnedValue) else None
            )

            def validation_source(
                validation_columns: tuple[str, ...] = validation_read_columns,
                candidate_input_columns: tuple[str, ...] = candidate_spec.input_columns,
            ):
                return chunks(
                    validation_columns,
                    start=prediction_spec.model_selection_validation_start,
                    end=prediction_spec.model_selection_validation_end,
                    input_columns=tuple(
                        dict.fromkeys(
                            (*common_validation_input_columns, *candidate_input_columns)
                        )
                    ),
                )

            validation_chunks = iter_validation_prediction_chunks(
                validation_source,
                prediction_spec=prediction_spec,
                candidate_spec=candidate_spec,
                model_result=model_result,
                validation_start=validation_start,
                validation_end=validation_end,
                usable_input_columns=common_validation_input_columns,
            )
            trial_metrics = evaluate_model_trial_metrics_from_chunks(
                validation_chunks,
                model_id=candidate_spec.model_id,
                label_column=label_column,
                prediction_column=prediction_spec.prediction_column,
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
            trial_metric_records.append(trial_metrics)
        except BaseException as exc:
            primary_error = exc
            if trial_recorder is not None and attempt is not None:
                try:
                    trial_recorder.finish(attempt, error=exc)
                except BaseException as record_error:
                    exc.add_note(
                        "trial failure record also failed: "
                        f"{type(record_error).__name__}"
                    )
            raise
        finally:
            cleanup_owner = owned_model if isinstance(owned_model, OwnedValue) else None
            cleanup_lease = model_lease
            validation_chunks = None
            model_result = None
            owned_model = None
            model_lease = None
            try:
                if cleanup_owner is not None and cleanup_owner.lease.active:
                    cleanup_owner.release(owner="research.model.selection")
                elif cleanup_lease is not None and cleanup_lease.active:
                    cleanup_lease.close()
            except BaseException as cleanup_error:
                if primary_error is not None:
                    primary_error.add_note(
                        "candidate model lease release also failed: "
                        f"{type(cleanup_error).__name__}"
                    )
                else:
                    raise
            finally:
                cleanup_owner = None
                cleanup_lease = None
    trial_metrics = pd.DataFrame.from_records(trial_metric_records)
    direction = (
        MODEL_SELECTION_METRIC_DIRECTIONS[prediction_spec.model_selection_metric]
        == "maximize"
    )
    selected_model_id = select_best_validation_trial(
        trial_metrics,
        trial_id_column="model_id",
        selection_metric=prediction_spec.model_selection_metric,
        maximize=direction,
        validation_metric_columns=VALIDATION_METRIC_COLUMNS,
        test_metric_columns=TEST_METRIC_COLUMNS,
        tie_break_order=(prediction_spec.selection_tie_break_order or None),
    )
    metadata = build_model_selection_metadata(
        configured_model_id=prediction_spec.model_id,
        selected_model_id=selected_model_id,
        candidate_model_ids=tuple(spec.model_id for spec in candidate_specs),
        selection_metric=prediction_spec.model_selection_metric,
        validation_start=prediction_spec.model_selection_validation_start,
        validation_end=prediction_spec.model_selection_validation_end,
        tie_break_order=(
            prediction_spec.selection_tie_break_order
            or tuple(spec.model_id for spec in candidate_specs)
        ),
        selected_model_input_columns=model_catalog.get(selected_model_id).input_columns,
        trial_metrics=trial_metrics,
    )
    return model_catalog.get(selected_model_id), metadata


def iter_validation_prediction_chunks(
    chunks: Callable[[], Iterable[pd.DataFrame]],
    *,
    prediction_spec,
    candidate_spec,
    model_result: fitting.ModelFitResult,
    validation_start: pd.Timestamp,
    validation_end: pd.Timestamp,
    usable_input_columns: tuple[str, ...] | None = None,
    include_availability: bool = False,
    extra_output_columns: tuple[str, ...] = (),
) -> Iterable[object]:
    predictor = prediction_callable(
        candidate_spec.input_columns,
        model_result.prediction_model,
    )
    rank_inputs_allow_missing = (
        _model_algorithm_facts(candidate_spec.model_algorithm).input_kind
        == "rank_ensemble_inputs"
    )
    output_columns = (
        "timestamp",
        "asset_id",
        candidate_spec.label_column,
        *extra_output_columns,
    )
    policy = current_runtime_resource_policy()
    complete_frames = iter_complete_time_sized_frames(
        iter(chunks()), timestamp_column="timestamp", resource_policy=policy
    )
    with closing_iterator(complete_frames) as source:
        for chunk in source:
            full_chunk_payload = pandas_frame_payload_bytes(chunk)
            index_bytes = int(chunk.index.memory_usage(deep=True))
            chunk_bound = max(
                3 * full_chunk_payload
                + index_bytes
                + len(chunk)
                * 8
                * (len(usable_input_columns or candidate_spec.input_columns) + 6),
                _filter_working_set_bound(
                    chunk,
                    mode=prediction_spec.sample_filter_mode,
                ),
            )
            chunk_lease = (
                policy.lease(
                    "research.model.selection.validation.chunk",
                    max(1, chunk_bound),
                )
                if policy is not None
                else None
            )
            try:
                timestamps = strict_timestamp_series(
                    chunk["timestamp"],
                    error="invalid_prediction_timestamp_values",
                )
                validation = chunk.loc[
                    (timestamps >= validation_start) & (timestamps <= validation_end)
                ]
                if not validation.empty:
                    filtered, _filter_metadata = apply_filter_mode(
                        validation,
                        mode=prediction_spec.sample_filter_mode,
                        metadata_key="sample_filter_mode",
                    )
                    availability_column = resolve_training_availability_column(
                        filtered,
                        label_column=candidate_spec.label_column,
                    )
                    prediction_inputs = tuple(
                        dict.fromkeys(
                            (
                                *canonical_feature_validity_columns(
                                    candidate_spec.input_columns
                                ),
                                *(
                                    candidate_spec.input_columns
                                    if not rank_inputs_allow_missing
                                    else ()
                                ),
                            )
                        )
                    )
                    prediction_rows = filtered.loc[
                        filtered.loc[:, list(prediction_inputs)].notna().all(axis=1)
                    ]
                    usable = legal_training_rows(
                        prediction_rows,
                        fitting_cutoff=validation_end,
                        availability_column=availability_column,
                        feature_columns=(
                            usable_input_columns
                            if usable_input_columns is not None
                            else canonical_feature_validity_columns(
                                candidate_spec.input_columns
                            )
                            if rank_inputs_allow_missing
                            else candidate_spec.input_columns
                        ),
                        target_column=candidate_spec.label_column,
                    )
                    if not usable.empty:
                        predictions = pd.Series(
                            predictor(prediction_rows), index=prediction_rows.index
                        )
                        output = usable.loc[:, list(output_columns)].copy()
                        if include_availability:
                            output["label_available_at"] = usable[availability_column]
                        output[prediction_spec.prediction_column] = predictions.loc[
                            usable.index
                        ]
                        if chunk_lease is not None:
                            output_bytes = pandas_frame_payload_bytes(output)
                            if output_bytes > chunk_lease.reserved_bytes:
                                raise ValueError(
                                    "prediction_validation_working_set_exceeds_task_memory_budget"
                                )
                            chunk_lease.shrink_to(output_bytes)
                            chunk_lease.transfer(
                                "research.model.selection.metric.chunk"
                            )
                            item = OwnedValue(output, chunk_lease)
                        else:
                            item = output
                        del (
                            chunk,
                            timestamps,
                            validation,
                            filtered,
                            prediction_rows,
                            predictions,
                            usable,
                            output,
                        )
                        try:
                            yield item
                        finally:
                            item = None
            except BaseException as primary:
                if chunk_lease is not None and chunk_lease.active:
                    try:
                        chunk_lease.close()
                    except BaseException as cleanup_error:
                        primary.add_note(
                            "validation chunk lease release also failed: "
                            f"{type(cleanup_error).__name__}"
                        )
                raise
            else:
                if chunk_lease is not None and chunk_lease.active:
                    chunk_lease.close()
            finally:
                chunk = timestamps = validation = filtered = None
                prediction_rows = predictions = usable = output = None

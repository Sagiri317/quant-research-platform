"""Validation-metric-only model trial selection helpers."""

from __future__ import annotations

import hashlib
import sqlite3
import sys
import tempfile
from collections.abc import Callable, Collection, Iterable
from contextlib import closing, contextmanager
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
    label_maturity_mask,
    resolve_training_availability_column,
    training_availability_columns,
)
from quant_research_platform.contracts.filtering import required_filter_columns
from quant_research_platform.contracts.json_codec import canonical_json_hash
from quant_research_platform.contracts.validation import strict_timestamp_series
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.research.experiments import TrialRecorder
from quant_research_platform.research.features.transforms import (
    model_input_validity_columns,
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
    runtime_resource_policy_scope,
)
from quant_research_platform.storage._scratch_ownership import ScratchReservation

VALIDATION_METRIC_COLUMNS = ("validation_ic", "validation_rankic", "validation_mse")
VALIDATION_METRIC_REASON_COLUMNS = (
    "validation_ic_reason",
    "validation_rankic_reason",
)
TEST_METRIC_COLUMNS = ("test_ic", "test_rankic", "test_mse")
_MODEL_SELECTION_SQLITE_CACHE_BYTES = 64 * 1024
_MODEL_SELECTION_SQLITE_PAGE_BYTES = 4096
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
def _model_selection_database(policy):
    """Retain scratch admission until both the connection and directory close."""

    with ScratchReservation(
        Path(tempfile.gettempdir()), resource_policy=policy
    ) as reservation:
        directory = tempfile.TemporaryDirectory(prefix="qrp-model-selection-")
        database = None
        metric_lease = None
        try:
            policy.require_working_set(
                _MODEL_SELECTION_METRIC_RESERVE_BYTES,
                error="model_selection_working_set_exceeds_memory_capacity",
            )
            metric_lease = policy.lease(
                "research.model.selection.metric", _MODEL_SELECTION_METRIC_RESERVE_BYTES
            )
            database = sqlite3.connect(f"{directory.name}/validation.sqlite")
            yield database, reservation, metric_lease
        finally:
            primary = sys.exception()
            failure = None
            closed = False
            try:
                if database is not None:
                    database.close()
                closed = True
                if metric_lease is not None:
                    metric_lease.close()
            except BaseException as exc:
                failure = exc
            try:
                directory.cleanup()
            except BaseException as exc:
                if failure is None:
                    failure = exc
                else:
                    failure.add_note(
                        "model selection directory cleanup also failed: "
                        f"{type(exc).__name__}"
                    )
            else:
                if closed:
                    reservation.resize(0)
            if failure is not None:
                if primary is None:
                    raise failure
                primary.add_note(
                    "model selection scratch cleanup also failed: "
                    f"{type(failure).__name__}"
                )


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

    with (
        runtime_resource_policy_scope() as policy,
        _model_selection_database(policy) as (database, scratch, metric_lease),
    ):
        database.execute(f"PRAGMA page_size={_MODEL_SELECTION_SQLITE_PAGE_BYTES}")
        database.execute("PRAGMA temp_store=FILE")
        database.execute(
            f"PRAGMA cache_size=-{_MODEL_SELECTION_SQLITE_CACHE_BYTES // 1024}"
        )
        database.execute("PRAGMA synchronous=NORMAL")
        database.execute("PRAGMA threads=0")
        database.execute("PRAGMA mmap_size=0")
        row_count = 0
        _admit_model_selection_storage(
            database, scratch, policy, metric_lease, row_count=0
        )
        database.execute(
            "CREATE TABLE validation_rows ("
            "row_id INTEGER PRIMARY KEY, "
            "timestamp_ns INTEGER NOT NULL, "
            "label REAL NOT NULL, "
            "prediction REAL NOT NULL"
            ")"
        )
        source = iter(chunks)
        with closing_iterator(source):
            for item in source:
                owned_lease = item.lease if isinstance(item, OwnedValue) else None
                frame = item.value if isinstance(item, OwnedValue) else item
                if frame.empty:
                    if owned_lease is not None:
                        owned_lease.close()
                    continue
                try:
                    if coverage is not None:
                        coverage.add(frame)
                    _admit_model_selection_storage(
                        database,
                        scratch,
                        policy,
                        metric_lease,
                        row_count=row_count + len(frame),
                    )
                    _insert_metric_chunk(
                        database,
                        frame,
                        label_column=label_column,
                        prediction_column=prediction_column,
                    )
                    row_count += len(frame)
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
            database.execute("SELECT COUNT(*) FROM validation_rows").fetchone()[0]
        )
        if validation_row_count == 0:
            raise ValueError("empty_model_selection_validation_window")
        _admit_model_selection_storage(
            database,
            scratch,
            policy,
            metric_lease,
            row_count=validation_row_count,
            query=True,
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


def _admit_model_selection_storage(
    database,
    scratch,
    policy,
    metric_lease,
    *,
    row_count: int,
    query: bool = False,
) -> None:
    native_bytes = _MODEL_SELECTION_METRIC_RESERVE_BYTES
    scratch_bytes = _MODEL_SELECTION_SQLITE_STARTUP_BYTES + 128 * row_count
    if query:
        options = dict(
            value.split("=", 1)
            for (value,) in database.execute("PRAGMA compile_options")
            if "=" in value
        )
        page = max(
            _MODEL_SELECTION_SQLITE_PAGE_BYTES,
            int(options.get("DEFAULT_PAGE_SIZE", "4096")),
        )
        default_cache = int(options.get("DEFAULT_CACHE_SIZE", "-2000"))
        cache_bytes = (
            -default_cache * 1024 if default_cache < 0 else default_cache * page
        )
        # This SQL has six sorters and five independent numeric window B-trees.
        # A ten-field record is <=91 bytes (<=128 with SorterRecord/alignment).
        # B-tree cells are <=103 bytes; 384/row covers one-third-full pages.
        if options.get("TEMP_STORE") == "3":
            # This compile option forces memory even after PRAGMA temp_store=FILE.
            # SQLite uses one allocation per record, no PMAs and no bulk realloc;
            # every temporary B-tree page must also remain resident.
            native_bytes += 6 * 128 * row_count
            native_bytes += 10 * (4 * page + 384 * row_count)
        else:
            # 4096/row covers main, all five trees, PMAs and their merge files.
            scratch_bytes = 64 * page + 4096 * row_count
            # QRP does not change SQLITE_CONFIG_PMASZ or install a custom cache.
            # SQLite's built-in defaults can be overridden by compile options.
            # See src/vdbesort.c, src/btree.c and cache_size/cache_spill pragmas.
            pma_bytes = max(
                int(options.get("SORTER_PMASZ", "250"))
                * _MODEL_SELECTION_SQLITE_PAGE_BYTES,
                _MODEL_SELECTION_SQLITE_CACHE_BYTES,
            )
            # threads=0 permits one realloc: six buffers plus its old one.
            native_bytes += 7 * min(
                2 * max(_MODEL_SELECTION_SQLITE_PAGE_BYTES, 128 * row_count),
                pma_bytes,
            )
            # Each tree has four cursors. 128 extra pages cover pinned paths
            # (SQLite's max depth is 20), balancing and spill beyond cache_size.
            # Doubling storage covers native page headers and allocator space.
            native_bytes += 10 * min(
                4 * page + 384 * row_count,
                cache_bytes + 128 * page,
            )
            # A full PMA holds at least threshold-128 bytes before the next row.
            # All leaf readers and each level of the 16-way merge tree coexist;
            # each owns a page, a <=128-byte record and bounded metadata.
            if 128 * row_count > pma_bytes:
                readers = level = (128 * row_count + pma_bytes - 129) // (
                    pma_bytes - 128
                )
                while level > 1:
                    level = (level + 15) // 16
                    readers += level
                native_bytes += 6 * readers * (_MODEL_SELECTION_SQLITE_PAGE_BYTES + 512)
    growth = native_bytes - metric_lease.reserved_bytes
    if growth > 0:
        policy.require_working_set(
            growth, error="model_selection_working_set_exceeds_memory_capacity"
        )
        metric_lease.grow(growth)
    scratch.resize(scratch_bytes)
    # main.max_page_count constrains this database, not anonymous sorters/trees;
    # their complete lifetimes are covered by the reservation above.
    main_bytes = _MODEL_SELECTION_SQLITE_STARTUP_BYTES + 128 * row_count
    database.execute(
        f"PRAGMA main.max_page_count={main_bytes // _MODEL_SELECTION_SQLITE_PAGE_BYTES}"
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
    database.executemany(
        "INSERT INTO validation_rows(timestamp_ns, label, prediction) VALUES (?, ?, ?)",
        (
            (
                int(timestamp),
                float(label),
                float(prediction),
            )
            for timestamp, label, prediction in zip(
                timestamp_ns,
                label_values.to_numpy(dtype="float64", copy=False),
                prediction_values.to_numpy(dtype="float64", copy=False),
                strict=True,
            )
        ),
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
                    *dataset_spec.primary_key,
                    *common_validation_input_columns,
                    *candidate_spec.input_columns,
                    *model_input_validity_columns(
                        candidate_spec.input_columns, candidate_spec.model_algorithm
                    ),
                    label_column,
                    *training_availability_columns(label_column),
                    *required_filter_columns(candidate_spec.sample_filter_mode),
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
                        end=fold.train.end,
                    ),
                    asset_ids=(),
                ),
                candidate_spec,
                estimator=build_estimator_from_model_spec(candidate_spec),
                fitting_cutoff=pd.Timestamp(fold.train.end),
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
                        frame = item.value if isinstance(item, OwnedValue) else item
                        on_validation(candidate, frame)
                        with current_runtime_resource_policy().lease(
                            "research.model.validation.maturity",
                            max(1, 3 * pandas_frame_payload_bytes(frame)),
                        ):
                            mature = frame.loc[
                                label_maturity_mask(
                                    frame,
                                    fitting_cutoff=validation_end,
                                    availability_column="label_available_at",
                                    target_column=label_column,
                                )
                                & frame["score_eligible"]
                            ]
                            if isinstance(item, OwnedValue):
                                yield OwnedValue(mature, item.lease)
                            else:
                                yield mature

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
        candidate_spec.input_columns,
        model_result.prediction_model,
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
                * (len(scoring_input_columns or candidate_spec.input_columns) + 6),
                _filter_working_set_bound(
                    chunk,
                    mode=candidate_spec.sample_filter_mode,
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
                        mode=candidate_spec.sample_filter_mode,
                        metadata_key="sample_filter_mode",
                    )
                    availability_column = resolve_training_availability_column(
                        filtered,
                        label_column=candidate_spec.label_column,
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

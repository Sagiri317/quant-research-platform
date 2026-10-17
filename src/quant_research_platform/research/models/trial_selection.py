"""Validation-metric-only model trial selection helpers."""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping
from fractions import Fraction

import numpy as np
import pandas as pd

from quant_research_platform.compute.dataframe.cross_section import (
    _add_correlation_aggregate,
    _CorrelationMoments,
    _summarize_correlation_aggregates,
)
from quant_research_platform.contracts.columns import _validate_asset_time_key
from quant_research_platform.contracts.evaluations import (
    EVALUATION_AXES,
    EvaluationAxis,
)
from quant_research_platform.contracts.experiments import (
    legal_training_mask,
    training_availability_columns,
)
from quant_research_platform.contracts.ids import validate_input_market
from quant_research_platform.contracts.json_codec import (
    _strict_config,
)
from quant_research_platform.contracts.models import (
    _PREDICTION_LOSS_REASONS,
    _prediction_loss_reason,
)
from quant_research_platform.contracts.time import _window_payload
from quant_research_platform.contracts.validation import strict_timestamp_series
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.research.features.transforms import (
    model_input_validity_columns,
)
from quant_research_platform.research.models.rank_ensemble import _label_predictions
from quant_research_platform.runtime.dataframes import iter_complete_time_frames
from quant_research_platform.storage.datasets.reading import (
    _dataset_input_options,
    iter_dataset_chunks,
    open_dataset_session,
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
) -> str | None:
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
    if not defined.any():
        return None
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


def evaluate_model_trial_metrics_from_chunks(
    chunks: Iterable[pd.DataFrame],
    *,
    model_id: str,
    label_column: str,
    prediction_column: str,
    validation_axis: EvaluationAxis = "cross_sectional",
    label_calibration: tuple[float, float] | None = None,
    label_prediction_column: str | None = None,
) -> dict[str, object]:
    """Evaluate time-ordered chunks, including timestamps split across chunks."""
    if type(validation_axis) is not str or validation_axis not in EVALUATION_AXES:
        raise ValueError("model_validation_axis_invalid")

    def metric_chunks():
        asset = None
        with closing_iterator(
            iter_complete_time_frames(iter(chunks), timestamp_column="timestamp")
        ) as source:
            for frame in source:
                if frame.empty:
                    continue
                values = _metric_chunk(
                    frame,
                    label_column=label_column,
                    prediction_column=prediction_column,
                )
                if validation_axis == "single_asset_time_series":
                    assets = frame["asset_id"]
                    unique_assets = assets.drop_duplicates()
                    if (
                        len(unique_assets) != 1
                        or assets.isna().any()
                        or (asset is not None and unique_assets.iloc[0] != asset)
                    ):
                        raise ValueError(
                            "single_asset_time_series_evaluation_requires_one_asset"
                        )
                    asset = unique_assets.iloc[0]
                if validation_axis == "cross_sectional":
                    ranks = values.groupby("timestamp_ns", sort=False)[
                        ["label", "prediction"]
                    ].rank(method="average")
                    values["label_rank"] = ranks["label"]
                    values["prediction_rank"] = ranks["prediction"]
                else:
                    values["label_rank"] = values["prediction_rank"] = 0.0
                if label_prediction_column is not None:
                    values["label_prediction"] = frame[
                        label_prediction_column
                    ].to_numpy(dtype="float64")
                yield values

    with closing_iterator(metric_chunks()) as source:
        summary = _cross_sectional_metric_summary(
            (
                row
                for values in source
                for row in values.itertuples(index=False, name=None)
            ),
            validation_axis=validation_axis,
            label_calibration=label_calibration,
        )
    if summary["validation_rows"] == 0:
        raise ValueError("empty_model_selection_validation_window")
    return {"model_id": model_id, **summary}


def _cross_sectional_metric_summary(
    rows,
    *,
    validation_axis: EvaluationAxis,
    label_calibration: tuple[float, float] | None = None,
) -> dict[str, object]:
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
    squared_errors = _MeanSquaredError()
    date_errors = _MeanSquaredError()
    current_timestamp: int | None = None
    timestamp_state: _TimestampMetricState | None = None
    for row in rows:
        timestamp_ns, label, prediction, label_rank, prediction_rank, *economic = row
        timestamp_ns = int(timestamp_ns) if validation_axis == "cross_sectional" else 0
        if current_timestamp != timestamp_ns:
            if timestamp_state is not None:
                squared_errors.add_period(date_errors)
                date_errors = _MeanSquaredError()
                row_count += timestamp_state.ic.count
                _add_correlation_aggregate(stats, "ic", timestamp_state.ic.result()[0])
                _add_correlation_aggregate(
                    stats, "rankic", timestamp_state.rank.result()[0]
                )
            current_timestamp = timestamp_ns
            timestamp_state = _TimestampMetricState()
        assert timestamp_state is not None
        label = float(label)
        prediction = float(prediction)
        label_prediction = (
            economic[0]
            if economic
            else _label_predictions(prediction, label_calibration)
        )
        timestamp_state.ic.update(label, prediction)
        if validation_axis == "cross_sectional":
            timestamp_state.rank.update(float(label_rank), float(prediction_rank))
        date_errors.add(label_prediction, label)
    if timestamp_state is not None:
        squared_errors.add_period(date_errors)
        row_count += timestamp_state.ic.count
        _add_correlation_aggregate(stats, "ic", timestamp_state.ic.result()[0])
        _add_correlation_aggregate(stats, "rankic", timestamp_state.rank.result()[0])
    mse = squared_errors.result()
    result = {"validation_rows": row_count, "validation_mse": mse}
    for prefix in ("ic", "rankic"):
        summary = _summarize_correlation_aggregates(
            count=int(stats[f"{prefix}_count"]),
            mean=stats[f"{prefix}_mean"],
            m2=stats[f"{prefix}_m2"],
            positive_count=int(stats[f"{prefix}_positive"]),
        )
        if validation_axis == "single_asset_time_series":
            if prefix == "ic":
                summary["mean"], summary["reason"] = (
                    timestamp_state.ic.result()
                    if row_count >= 2
                    else (None, "no_valid_observations")
                )
            else:
                summary["mean"], summary["reason"] = None, "not_applicable"
                summary["ratio_reason"] = "not_applicable"
                summary["std_reason"] = "not_applicable"
        for suffix, key in (
            (prefix, "mean"),
            (f"{prefix}_reason", "reason"),
            (f"std_{prefix}", "std"),
            (f"std_{prefix}_reason", "std_reason"),
            (f"{prefix}ir", "ratio"),
            (f"positive_{prefix}_rate", "positive_rate"),
            (f"valid_{prefix}_count", "valid_count"),
            (f"{prefix}ir_reason", "ratio_reason"),
        ):
            value = summary[key]
            result[f"validation_{suffix}"] = (
                None if isinstance(value, float) and not np.isfinite(value) else value
            )
    return result


class _TimestampMetricState:
    __slots__ = ("ic", "rank")

    def __init__(self) -> None:
        self.ic = _CorrelationMoments()
        self.rank = _CorrelationMoments()


class _MeanSquaredError:
    """Accumulate exact daily losses and their equally weighted period means."""

    __slots__ = ("moments", "total", "count")

    def __init__(self) -> None:
        self.moments = _CorrelationMoments()
        self.total = Fraction()
        self.count = 0

    def add(self, prediction: float, label: float) -> None:
        self.count += 1
        if self.total is not None:
            if np.isfinite(prediction) and np.isfinite(label):
                self.moments.update(float(prediction), float(label))
            else:
                self.total = None

    def add_period(self, errors: _MeanSquaredError) -> None:
        self.count += 1
        value = errors.mean()
        self.total = (
            self.total + value if self.total is not None and value is not None else None
        )

    def mean(self) -> Fraction | None:
        if not self.count or self.total is None:
            return None
        moments = self.moments
        squared_error = Fraction(
            moments.left_square + moments.right_square - 2 * moments.product,
            1 << (2 * moments.exponent),
        )
        return (self.total + squared_error) / self.count

    def result(self) -> float | None:
        value = self.mean()
        if value is None:
            return None
        try:
            result = float(value)
        except OverflowError:
            return None
        return result if np.isfinite(result) and (result != 0 or value == 0) else None


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


def _json_metrics(metrics):
    return {
        key: None
        if isinstance(value, (float, np.floating)) and np.isnan(value)
        else value.item()
        if isinstance(value, np.generic)
        else value
        for key, value in metrics.items()
    }


def score(config):
    """Score an explicit complete candidate/fold matrix on a freshly aligned domain."""
    from pathlib import Path

    from quant_research_platform.artifacts.outputs.prediction import (
        prediction_model_metadata,
        read_predictions,
    )
    from quant_research_platform.contracts.filtering import required_filter_columns
    from quant_research_platform.contracts.models import (
        _fit_windows,
        _model_spec,
        validate_model_input_roles,
    )
    from quant_research_platform.contracts.time import TimeWindow
    from quant_research_platform.research.predictions.outputs import prediction_rows
    from quant_research_platform.storage.atomic import (
        atomic_write_json,
        reject_if_exists,
    )

    config = _strict_config(
        config,
        {
            "predictions",
            "labels",
            "candidate_ids",
            "validation_windows",
            "scoring_window",
            "cutoff",
            "selection_group",
            "validation_axis",
            "output",
        },
        {"label_spec", "test_window", "label_unit"},
    )
    reject_if_exists(Path(config["output"]))
    ids, paths = config["candidate_ids"], config["predictions"]
    if (
        type(ids) is not list
        or not ids
        or len(set(ids)) != len(ids)
        or any(type(item) is not str or not item for item in ids)
        or type(paths) is not list
        or len(paths) != len(ids)
    ):
        raise ValueError("complete_ordered_candidates_required")
    if config["selection_group"] not in {"global", "quarterly"}:
        raise ValueError("selection_group_invalid")
    cutoff = strict_timestamp_series(
        pd.Series([config["cutoff"]]), error="invalid_selection_cutoff"
    ).iloc[0]
    window = TimeWindow(**config["scoring_window"])
    test = TimeWindow(**config["test_window"]) if "test_window" in config else None
    validation_windows = _fit_windows(config["validation_windows"], cutoff)
    if (
        window.end is None
        or pd.Timestamp(window.end) > cutoff
        or (test is not None and cutoff >= pd.Timestamp(test.start))
    ):
        raise ValueError("selection_test_isolation_invalid")
    if config["selection_group"] == "global" and (
        pd.Timestamp(window.start) != pd.Timestamp(validation_windows[0].start)
        or pd.Timestamp(window.end) != pd.Timestamp(validation_windows[-1].end)
    ):
        raise ValueError("global_selection_requires_complete_validation")
    source = config["labels"]
    if config.get("label_spec") is not None:
        if isinstance(source, Mapping):
            raise ValueError("duplicate_input_spec")
        source = {"path": source, "spec": config["label_spec"]}
    options = _dataset_input_options(source)
    label_projection = options.get("columns")
    options.setdefault("columns", ("timestamp", "asset_id"))
    label_session = open_dataset_session(**options)
    _validate_asset_time_key(
        label_session.spec["primary_key"], label_session.spec["column_roles"]
    )
    declared = label_session.spec
    labels = None
    candidates, frames, common, target, market = [], [], None, None, None
    expected_domains = {}
    for candidate_id, candidate_paths in zip(ids, paths, strict=True):
        if type(candidate_paths) is dict:
            if set(candidate_paths) != {"trial"}:
                raise ValueError("failed_candidate_trial_reference_invalid")
            from quant_research_platform.storage.experiments import read_trial_ledger

            records = read_trial_ledger(Path(candidate_paths["trial"]))
            record = records[-1]
            if (
                record["status"] not in {"failed", "abandoned"}
                or record["model"]["model_id"] != candidate_id
            ):
                raise ValueError("failed_candidate_trial_required")
            if pd.Timestamp(record["fit_cutoff"]) >= pd.Timestamp(
                validation_windows[0].start
            ):
                raise ValueError("validation_fit_overlap")
            config.setdefault(
                "label_unit",
                declared.get("column_units", {}).get(record["model"]["target_column"]),
            )
            if type(config["label_unit"]) is not str or not config["label_unit"]:
                raise ValueError("label_unit_required")
            if market is not None and record["market_id"] != market:
                raise ValueError("selection_label_units_or_market_mismatch")
            if target is not None and record["model"]["target_column"] != target:
                raise ValueError("model_selection_label_mismatch")
            market, target = record["market_id"], record["model"]["target_column"]
            candidates.append(
                {
                    "model_id": candidate_id,
                    "family_id": record["family_id"],
                    "asset_ids": record.get("asset_ids"),
                    "model": record["model"],
                    "candidate_parameters": record["candidate_parameters"],
                    "predictions": [],
                    "fit_windows": record["fit_windows"],
                    "trial": candidate_paths["trial"],
                    "status": record["status"],
                    "failure_reason": record.get("error"),
                }
            )
            frames.append(None)
            continue
        candidate_paths = (
            [candidate_paths] if type(candidate_paths) is str else candidate_paths
        )
        if (
            type(candidate_paths) is not list
            or len(candidate_paths) != len(validation_windows)
            or len(set(candidate_paths)) != len(candidate_paths)
        ):
            raise ValueError("candidate_validation_windows_incomplete")
        pieces, training_windows, first_facts = [], [], None
        mse_reason = None
        for path, declared_window in zip(
            candidate_paths, validation_windows, strict=True
        ):
            predictions, prediction = read_predictions(
                Path(path), columns=("timestamp", "asset_id", "prediction_score")
            )
            if prediction["model_id"] != candidate_id or prediction[
                "prediction_window"
            ] != _window_payload(declared_window):
                raise ValueError("selection_requires_declared_validation_candidate")
            facts = prediction["model_facts"]
            model_metadata = prediction_model_metadata(prediction)
            if pd.Timestamp(facts["fit_cutoff"]) >= pd.Timestamp(declared_window.start):
                raise ValueError("validation_fit_overlap")
            config.setdefault(
                "label_unit",
                declared.get("column_units", {}).get(model_metadata["target_column"]),
            )
            if type(config["label_unit"]) is not str or not config["label_unit"]:
                raise ValueError("label_unit_required")
            if market is not None and facts["market_id"] != market:
                raise ValueError("selection_label_units_or_market_mismatch")
            market = facts["market_id"]
            if market == "crypto":
                training_assets = facts.get("asset_ids")
                if training_assets is None:
                    training_assets = model_metadata["training_asset_ids"]
                expected_axis = (
                    "single_asset_time_series"
                    if len(training_assets) == 1
                    else "cross_sectional"
                )
                if not training_assets or config["validation_axis"] != expected_axis:
                    raise ValueError("crypto_training_scope_validation_axis_mismatch")
            label = model_metadata["target_column"]
            if target is not None and target != label:
                raise ValueError("model_selection_label_mismatch")
            target = label
            if labels is None:
                if label_projection is not None and target not in label_projection:
                    raise ValueError("scoring_label_projection_missing")
                if declared["column_roles"].get(target) != "label":
                    raise ValueError("scoring_target_role_mismatch")
                unit = declared.get("column_units", {}).get(target)
                if unit is not None and unit != config["label_unit"]:
                    raise ValueError("scoring_label_unit_mismatch")
                availability = declared.get("label_availability_columns", {}).get(
                    target, training_availability_columns(target)[0]
                )
                with closing_iterator(
                    iter_dataset_chunks(
                        label_session.filtered_view(
                            TimeWindow(
                                validation_windows[0].start, validation_windows[-1].end
                            )
                        ).projected_view(
                            ("timestamp", "asset_id", "eligible", target, availability)
                        )
                    )
                ) as label_chunks:
                    labels = pd.concat(list(label_chunks), ignore_index=True)
                if labels.duplicated(["timestamp", "asset_id"]).any() or labels.loc[
                    :, ["timestamp", "asset_id"]
                ].isna().any(axis=None):
                    raise ValueError("invalid_scoring_label_keys")
                labels["timestamp"] = strict_timestamp_series(
                    labels.timestamp, error="invalid_scoring_timestamp"
                )
            if first_facts is not None:
                if facts.get("asset_ids") != first_facts.get("asset_ids"):
                    raise ValueError(
                        "candidate_training_asset_scope_changed_across_folds"
                    )
                for key in (
                    "model_algorithm",
                    "input_factor_columns",
                    "input_feature_columns",
                    "target_column",
                    "sample_filter_mode",
                    "training_weighting",
                    "max_train_rows",
                ):
                    if facts["model"][key] != first_facts["model"][key]:
                        raise ValueError("candidate_configuration_changed_across_folds")
                if facts["candidate_parameters"] != first_facts["candidate_parameters"]:
                    raise ValueError("candidate_parameters_changed_across_folds")
            else:
                first_facts = facts
            columns = tuple(
                dict.fromkeys(
                    (
                        "timestamp",
                        "asset_id",
                        *model_metadata["input_columns"],
                        *required_filter_columns(model_metadata["sample_filter_mode"]),
                    )
                )
            )
            from quant_research_platform.contracts.json_codec import (
                canonical_identity_bytes,
            )

            domain = canonical_identity_bytes(
                {
                    "inputs": prediction["inputs"],
                    "input_spec": prediction["input_spec"],
                    "window": prediction["prediction_window"],
                    "columns": list(columns),
                    "assets": prediction.get("asset_ids"),
                    "algorithm": model_metadata["model_algorithm"],
                    "filter": model_metadata["sample_filter_mode"],
                }
            )
            if domain not in expected_domains:
                source = prediction["inputs"]
                if prediction["input_spec"] is not None:
                    if isinstance(source, Mapping):
                        raise ValueError("duplicate_input_spec")
                    source = {"path": source, "spec": prediction["input_spec"]}
                options = _dataset_input_options(source)
                selected_columns = options.get("columns")
                options.setdefault("columns", ("timestamp", "asset_id"))
                session = open_dataset_session(**options)
                validate_model_input_roles(
                    session.spec,
                    _model_spec(facts["model"]),
                    market_id=facts["market_id"],
                    selected_columns=selected_columns,
                )
                read_columns = tuple(
                    dict.fromkeys(
                        (
                            *columns,
                            *model_input_validity_columns(
                                tuple(model_metadata["input_columns"]),
                                model_metadata["model_algorithm"],
                                session.spec["column_roles"],
                            ),
                        )
                    )
                )
                keys = []
                with closing_iterator(
                    iter_dataset_chunks(
                        session.filtered_view(declared_window).projected_view(
                            read_columns
                        )
                    )
                ) as input_chunks:
                    for frame in input_chunks:
                        expected = prediction_rows(
                            frame,
                            model_metadata,
                            declared_window,
                            declared_columns=session.spec["column_roles"],
                        )
                        if prediction.get("asset_ids") is not None:
                            expected = expected.loc[
                                expected.asset_id.isin(prediction["asset_ids"])
                            ]
                        keys.append(expected.loc[:, ["timestamp", "asset_id"]])
                expected_keys = pd.MultiIndex.from_frame(
                    pd.concat(keys, ignore_index=True)
                )
                if not expected_keys.is_unique:
                    raise ValueError("candidate_input_duplicate_keys")
                expected_domains[domain] = expected_keys
            expected_keys = expected_domains[domain]
            actual_keys = pd.MultiIndex.from_frame(
                predictions.loc[:, ["timestamp", "asset_id"]]
            )
            if len(expected_keys) != len(actual_keys) or len(
                expected_keys.difference(actual_keys)
            ):
                raise ValueError("candidate_prediction_coverage_missing")
            joined = predictions.merge(
                labels,
                on=["timestamp", "asset_id"],
                how="left",
                validate="one_to_one",
                indicator=True,
            )
            if not joined["_merge"].eq("both").all():
                raise ValueError("candidate_label_coverage_missing")
            mature = legal_training_mask(
                joined,
                fitting_cutoff=cutoff,
                availability_column=availability,
                target_column=label,
            )
            from quant_research_platform.contracts.time import (
                time_window_timestamp_interval,
            )

            lower, upper, closed = time_window_timestamp_interval(window)
            mature &= joined.timestamp.ge(pd.Timestamp(lower)) & (
                joined.timestamp.le(pd.Timestamp(upper))
                if closed
                else joined.timestamp.lt(pd.Timestamp(upper))
            )
            joined = joined.loc[mature].drop(columns="_merge")
            calibration = model_metadata.get("rank_parameters", {}).get(
                "label_calibration"
            )
            reason = _prediction_loss_reason(
                facts,
                target,
                declared,
                label_session.manifest,
                target_unit=config["label_unit"],
            )
            mse_reason = mse_reason or reason
            joined["prediction_label_value"] = (
                _label_predictions(joined.prediction_score, calibration)
                if reason is None
                else np.nan
            )
            pieces.append(joined)
            training_windows.extend(facts["fit_windows"])
        candidate_frame = pd.concat(pieces, ignore_index=True).set_index(
            ["timestamp", "asset_id"]
        )
        if not candidate_frame.index.is_unique:
            raise ValueError("candidate_validation_keys_duplicate")
        common = (
            candidate_frame.index
            if common is None
            else common.intersection(candidate_frame.index, sort=False)
        )
        frames.append(candidate_frame)
        candidates.append(
            {
                "model_id": candidate_id,
                "family_id": first_facts["family_id"],
                "asset_ids": first_facts.get("asset_ids"),
                "model": first_facts["model"],
                "candidate_parameters": first_facts["candidate_parameters"],
                "predictions": candidate_paths,
                "fit_windows": training_windows,
                "mse_reason": mse_reason,
            }
        )
    validate_input_market(declared, market)
    if market == "a_share":
        from quant_research_platform.research.protocols.a_share import (
            validate_model_selection,
        )

        validate_model_selection([_model_spec(item["model"]) for item in candidates])
    if common is None or not len(common):
        raise ValueError("empty_model_selection_validation_window")
    rows = []
    for candidate, frame in zip(candidates, frames, strict=True):
        if frame is None:
            from quant_research_platform.contracts.models import (
                _MODEL_SELECTION_TRIAL_METRIC_KEYS,
            )

            metrics = {key: None for key in _MODEL_SELECTION_TRIAL_METRIC_KEYS}
            metrics.update(model_id=candidate["model_id"], validation_rows=0)
            for prefix in ("ic", "rankic"):
                metrics[f"validation_valid_{prefix}_count"] = 0
                for suffix in (
                    f"{prefix}_reason",
                    f"std_{prefix}_reason",
                    f"{prefix}ir_reason",
                ):
                    metrics[f"validation_{suffix}"] = "no_valid_observations"
            rows.append(metrics)
            continue
        comparable = (
            frame.loc[common]
            .reset_index()
            .sort_values(["timestamp", "asset_id"], kind="stable")
        )
        metrics = evaluate_model_trial_metrics_from_chunks(
            [comparable],
            model_id=candidate["model_id"],
            label_column=target,
            prediction_column="prediction_score",
            validation_axis=config["validation_axis"],
            label_prediction_column="prediction_label_value",
        )
        rows.append(_json_metrics(metrics))
    payload = {
        "candidate_ids": ids,
        "candidates": candidates,
        "metrics": rows,
        "validation_windows": [_window_payload(item) for item in validation_windows],
        "scoring_window": _window_payload(window),
        "cutoff": cutoff.isoformat(),
        "test_window": None if test is None else _window_payload(test),
        "selection_group": config["selection_group"],
        "validation_axis": config["validation_axis"],
        "common_rows": len(common),
        "label_unit": config["label_unit"],
        "target_column": target,
        "market_id": market,
        "inference_blocker": "global_validation_selection_reuses_validation"
        if config["selection_group"] == "global"
        else None,
    }
    return atomic_write_json(Path(config["output"]), payload)


def _selection_choice(value):
    from quant_research_platform.contracts.models import (
        MODEL_SELECTION_METRIC_DIRECTIONS,
    )

    ids, order = value["candidate_ids"], value["tie_break_order"]
    if (
        not ids
        or len(ids) != len(set(ids))
        or len(order) != len(set(order))
        or set(order) != set(ids)
    ):
        raise ValueError("selection_candidate_coverage_missing")
    if [item["model_id"] for item in value["candidates"]] != ids or [
        item["model_id"] for item in value["metrics"]
    ] != ids:
        raise ValueError("selection_candidate_coverage_missing")
    from pathlib import Path

    from quant_research_platform.contracts.models import (
        _validate_model_selection_trial_record,
    )

    for candidate, metrics in zip(value["candidates"], value["metrics"], strict=True):
        if candidate.get("mse_reason") is not None and (
            candidate["mse_reason"] not in _PREDICTION_LOSS_REASONS
            or metrics["validation_mse"] is not None
        ):
            raise ValueError("selection_mse_reason_invalid")
        if candidate.get("status") in {"failed", "abandoned"}:
            from quant_research_platform.storage.experiments import read_trial_ledger

            terminal = read_trial_ledger(Path(candidate["trial"]))[-1]
            if (
                terminal["status"] != candidate["status"]
                or terminal["model"] != candidate["model"]
                or terminal["family_id"] != candidate["family_id"]
                or terminal.get("error") != candidate.get("failure_reason")
                or terminal["candidate_parameters"] != candidate["candidate_parameters"]
                or terminal["fit_windows"] != candidate["fit_windows"]
                or terminal.get("asset_ids") != candidate.get("asset_ids")
            ):
                raise ValueError("selection_failed_trial_reference_mismatch")
            if metrics["validation_rows"] != 0 or any(
                metrics[key] is not None for key in MODEL_SELECTION_METRIC_DIRECTIONS
            ):
                raise ValueError("failed_trial_cannot_have_validation_metrics")
        else:
            _validate_model_selection_trial_record(metrics)
    if (
        MODEL_SELECTION_METRIC_DIRECTIONS.get(value["selection_metric"])
        != value["selection_direction"]
    ):
        raise ValueError("selection_metric_direction_invalid")
    eligible = {
        item["model_id"]
        for item in value["candidates"]
        if item.get("status") not in {"failed", "abandoned"}
    }
    if not eligible:
        return None
    return select_best_validation_trial(
        pd.DataFrame([row for row in value["metrics"] if row["model_id"] in eligible]),
        trial_id_column="model_id",
        selection_metric=value["selection_metric"],
        maximize=value["selection_direction"] == "maximize",
        validation_metric_columns=tuple(MODEL_SELECTION_METRIC_DIRECTIONS),
        test_metric_columns=(),
        tie_break_order=tuple(item for item in order if item in eligible),
    )


def select(config):
    """Freeze a complete validation comparison, without automatically refitting."""
    from pathlib import Path

    from quant_research_platform.contracts.json_codec import (
        _strict_config,
        parse_qrp_json_object,
    )
    from quant_research_platform.storage.atomic import atomic_write_json

    config = _strict_config(
        config,
        {
            "scores",
            "output",
            "selection_metric",
            "selection_direction",
            "tie_break_order",
            "refit_windows",
            "refit_cutoff",
        },
    )
    scores = parse_qrp_json_object(Path(config["scores"]).read_bytes())
    result = {
        **scores,
        **{
            key: config[key]
            for key in (
                "scores",
                "selection_metric",
                "selection_direction",
                "tie_break_order",
                "refit_windows",
                "refit_cutoff",
            )
        },
    }
    result.update(_selection_decisions(result))
    result["refit_windows"], result["refit_cutoff"] = _validate_refit_scope(result)
    return atomic_write_json(Path(config["output"]), result)


def _selection_decisions(value):
    _selection_choice(value)
    family_order = list(
        dict.fromkeys(item["family_id"] for item in value["candidates"])
    )
    families, winners = [], []
    for family in family_order:
        members = [
            item["model_id"]
            for item in value["candidates"]
            if item["family_id"] == family
        ]
        subset = {
            **value,
            "candidate_ids": members,
            "candidates": [
                item for item in value["candidates"] if item["model_id"] in members
            ],
            "metrics": [
                item for item in value["metrics"] if item["model_id"] in members
            ],
            "tie_break_order": [
                item for item in value["tie_break_order"] if item in members
            ],
        }
        winner = _selection_choice(subset)
        families.append(
            {
                "family_id": family,
                "selected_config_id": winner,
                "candidate_ids": members,
                "trial_metrics": subset["metrics"],
                "failure_reasons": {
                    item["model_id"]: item.get("failure_reason")
                    for item in subset["candidates"]
                    if item.get("status") in {"failed", "abandoned"}
                },
            }
        )
        if winner is None:
            continue
        winners.append(
            {
                **next(row for row in subset["metrics"] if row["model_id"] == winner),
                "model_id": family,
            }
        )
    if not winners:
        return {
            "family_tie_break_order": family_order,
            "families": families,
            "selected_family_id": None,
            "selected_model_id": None,
        }
    selected_family = select_best_validation_trial(
        pd.DataFrame(winners),
        trial_id_column="model_id",
        selection_metric=value["selection_metric"],
        maximize=value["selection_direction"] == "maximize",
        validation_metric_columns=(value["selection_metric"],),
        test_metric_columns=(),
        tie_break_order=tuple(
            family
            for family in family_order
            if any(row["model_id"] == family for row in winners)
        ),
    )
    return {
        "family_tie_break_order": family_order,
        "families": families,
        "selected_family_id": selected_family,
        "selected_model_id": next(
            item["selected_config_id"]
            for item in families
            if item["family_id"] == selected_family
        ),
    }


def _validate_refit_scope(value):
    from quant_research_platform.contracts.models import _fit_windows
    from quant_research_platform.contracts.time import (
        _window_payload,
    )

    cutoff = strict_timestamp_series(
        pd.Series([value["refit_cutoff"]]), error="invalid_refit_cutoff"
    ).iloc[0]
    if cutoff < pd.Timestamp(value["cutoff"]) or (
        value.get("test_window") is not None
        and cutoff >= pd.Timestamp(value["test_window"]["start"])
    ):
        raise ValueError("refit_cutoff_outside_frozen_domain")
    eligible_families = [
        item for item in value["families"] if item["selected_config_id"] is not None
    ]
    if not eligible_families:
        return {}, cutoff.isoformat()
    declared = value["refit_windows"]
    if type(declared) is list:
        declared = {item["family_id"]: declared for item in eligible_families}
    if type(declared) is not dict or set(declared) != {
        item["family_id"] for item in eligible_families
    }:
        raise ValueError("refit_family_windows_incomplete")

    normalized = {
        family["family_id"]: [
            _window_payload(window)
            for window in _fit_windows(declared[family["family_id"]], cutoff)
        ]
        for family in eligible_families
    }
    return normalized, cutoff.isoformat()


def read_selection(path):
    from pathlib import Path

    from quant_research_platform.contracts.json_codec import parse_qrp_json_object

    value = parse_qrp_json_object(Path(path).read_bytes())
    _validate_refit_scope(value)
    decisions = _selection_decisions(value)
    if any(value.get(key) != item for key, item in decisions.items()):
        raise ValueError("selection_choice_mismatch")
    original = parse_qrp_json_object(Path(value["scores"]).read_bytes())
    if any(value.get(key) != item for key, item in original.items()):
        raise ValueError("selection_score_reference_mismatch")
    cutoff = strict_timestamp_series(
        pd.Series([value["cutoff"]]), error="invalid_selection_cutoff"
    ).iloc[0]
    if value.get("test_window") is not None and cutoff >= pd.Timestamp(
        value["test_window"]["start"]
    ):
        raise ValueError("selection_test_isolation_invalid")
    return value

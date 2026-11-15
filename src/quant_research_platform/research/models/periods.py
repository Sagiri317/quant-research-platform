"""Summarize frozen fits, saved historical CV, and future daily evaluation."""

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from quant_research_platform.artifacts.outputs.manifest import read_output_manifest
from quant_research_platform.artifacts.statistics import (
    STATISTICS_TABLE_COLUMNS,
    write_statistics_artifact,
)
from quant_research_platform.compute.dataframe.streaming import (
    iter_complete_time_frames,
)
from quant_research_platform.contracts.experiments import label_maturity_mask
from quant_research_platform.contracts.filtering import required_filter_columns
from quant_research_platform.contracts.json_codec import (
    _strict_config,
    render_qrp_json_object,
)
from quant_research_platform.contracts.models import (
    SEQUENCE_ALGORITHMS,
    _prediction_target_reason,
    _validate_model_prediction_window,
    compare_model_facts,
    sequence_context_columns,
    validate_model_input_roles,
)
from quant_research_platform.contracts.time import (
    PredictionWindow,
    TimeWindow,
    prediction_window_mask,
    time_window_timestamp_interval,
)
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.research.evaluations.time_series import (
    _DAILY_DTYPES,
    _evaluation_axes,
    daily_metric_summary,
    daily_signal_metrics,
)
from quant_research_platform.research.models.persisted import load_model
from quant_research_platform.research.models.production import _fit_read_windows
from quant_research_platform.research.predictions.runner import iter_frozen_predictions
from quant_research_platform.storage.atomic import reject_if_exists
from quant_research_platform.storage.datasets.reading import (
    _dataset_input_options,
    iter_dataset_time_chunks,
    materialize_dataset_session,
    open_dataset_session,
)

_METRICS = ("mse", "rmse", "mae", "ic", "rank_ic", "r2_vs_zero")


def _bounds(window):
    lower, upper, closed = time_window_timestamp_interval(window)
    return pd.Timestamp(lower), pd.Timestamp(upper) - pd.Timedelta(
        0 if closed else 1, "ns"
    )


def _window_days(sessions, windows):
    selected = np.zeros(len(sessions), dtype=bool)
    for window in windows:
        start, end = _bounds(window)
        selected |= (sessions >= start.normalize()) & (sessions <= end.normalize())
    return sessions[selected]


def _training_daily(inputs, result, facts, *, sessions, evaluation_axis):
    """Replay calendar-month feature blocks with one prediction closure."""
    metadata = result.metadata
    windows = tuple(TimeWindow(**item) for item in facts["fit_windows"])
    span = TimeWindow(windows[0].start, windows[-1].end)
    sequence = metadata["model_algorithm"] in SEQUENCE_ALGORITHMS
    read_windows = tuple(_fit_read_windows(windows, sequence=sequence))
    assets = tuple(metadata["training_asset_ids"])
    options = _dataset_input_options(
        inputs,
        discovery_windows=tuple(window for window, _ in read_windows),
        row_filters=(("asset_id", "in", assets),),
    )
    selected_columns = options.get("columns")
    options.setdefault("columns", ("timestamp", "asset_id"))
    session = open_dataset_session(**options)
    validate_model_input_roles(
        session.spec,
        facts["model"],
        market_id=facts["market_id"],
        include_target=True,
        selected_columns=selected_columns,
        input_units=facts["input_units"],
        hyperparameters=metadata["hyperparameters"],
    )
    target = metadata["target_column"]
    reason = _prediction_target_reason(facts, target, session.spec, session.manifest)
    if reason is not None:
        raise ValueError(f"model_period_target_mismatch: {reason}")
    availability = metadata["label_availability_column"]
    columns = tuple(
        dict.fromkeys(
            (
                "timestamp",
                "asset_id",
                *metadata["input_columns"],
                *sequence_context_columns(
                    metadata["model_algorithm"], metadata["hyperparameters"]
                ),
                *required_filter_columns(metadata["sample_filter_mode"]),
                target,
                availability,
            )
        )
    )
    context_columns = tuple(c for c in columns if c not in {target, availability})

    def sample_mask(frame):
        selected = pd.Series(False, index=frame.index)
        for window in windows:
            selected |= prediction_window_mask(frame.timestamp, window)
        return selected

    def frames():
        for read_window, context_only in read_windows:
            first, last = _bounds(read_window)
            for month in pd.period_range(first, last, freq="M"):
                # Bound even the chronological reader's overlapping-member sort.
                window = TimeWindow(
                    max(first, month.start_time), min(last, month.end_time)
                )
                with closing_iterator(
                    iter_dataset_time_chunks(
                        session.filtered_view(window).projected_view(
                            context_columns if context_only else columns
                        )
                    )
                ) as chunks:
                    for frame in chunks:
                        if not frame.timestamp.dt.normalize().isin(sessions).all():
                            raise ValueError("evaluation_input_session_not_declared")
                        yield frame

    def values():
        with closing_iterator(
            iter_frozen_predictions(
                frames(), result, read_window=span, output_window=span
            )
        ) as predictions:
            for frame in predictions:
                frame = frame.loc[sample_mask(frame)]
                if frame.empty:
                    continue
                mature = label_maturity_mask(
                    frame,
                    fitting_cutoff=facts["fit_cutoff"],
                    availability_column=availability,
                    target_column=target,
                )
                yield (
                    frame.loc[:, ["timestamp", "asset_id"]]
                    .assign(
                        signal=frame.prediction_score.where(mature),
                        label=frame[target].where(mature),
                        session=frame.timestamp.dt.normalize(),
                    )
                    .copy(deep=True)
                )

    parts = []
    with closing_iterator(
        iter_complete_time_frames(
            values(), timestamp_column="session", one_timestamp=True
        )
    ) as days:
        for frame in days:
            parts.append(
                daily_signal_metrics(
                    frame,
                    evaluation_axis=evaluation_axis,
                    sessions=[frame.session.iloc[0]],
                    asset_ids=assets,
                    regression=True,
                    baseline_columns=("zero",),
                )
            )
    dates = _window_days(sessions, windows)
    observed = pd.DatetimeIndex([part.timestamp.iloc[0] for part in parts])
    missing = dates.difference(observed)
    if len(missing):
        empty = pd.DataFrame(
            {
                "timestamp": pd.Series(dtype="datetime64[ns]"),
                "asset_id": pd.Series(dtype="string"),
                "signal": pd.Series(dtype="float64"),
                "label": pd.Series(dtype="float64"),
            }
        )
        parts.append(
            daily_signal_metrics(
                empty,
                evaluation_axis=evaluation_axis,
                sessions=missing,
                asset_ids=assets,
                regression=True,
                baseline_columns=("zero",),
            )
        )
    daily = pd.concat(parts, ignore_index=True)
    if (
        int(daily.loc[daily.metric.eq("mse"), "observation_count"].sum())
        != metadata["rows_after_input_label_dropna"]
    ):
        raise ValueError("model_period_training_sample_mismatch")
    return daily


def _prediction_source(path, model_id, config):
    single = config["evaluation_axis"] == "single_asset_time_series"
    session = open_dataset_session(
        Path(path),
        columns=(*_DAILY_DTYPES, "candidate_id", "sample_scope"),
        row_filters=(
            ("candidate_id", "==", model_id),
            ("sample_scope", "==", "own"),
            *((("asset_id", "in", tuple(config["asset_ids"])),) if single else ()),
        ),
    )
    parameters = session.manifest["parameters"]
    if parameters["evaluation_axis"] != config["evaluation_axis"]:
        raise ValueError("model_period_prediction_axis_mismatch")
    actual, requested = set(parameters["asset_ids"]), set(config["asset_ids"])
    if not requested <= actual or (not single and actual != requested):
        raise ValueError("model_period_prediction_assets_mismatch")
    return (
        session,
        parameters,
        [
            read_output_manifest(Path(path))["metadata"]
            for path in parameters["predictions"][model_id]
        ],
    )


def _prediction_daily(source, model_id, facts, window, dates):
    session, parameters, predictions = source
    if parameters["label"] != facts["model"]["target_column"]:
        raise ValueError("model_period_prediction_target_mismatch")
    clock = pd.DatetimeIndex(pd.to_datetime(parameters["sessions"]))
    if not _window_days(clock, (window,)).equals(dates):
        raise ValueError("model_period_prediction_sessions_mismatch")
    start, end = _bounds(window)
    for metadata in predictions:
        if metadata["prediction_rows"] == 0:
            continue
        extent = PredictionWindow(**metadata["prediction_window"])
        lower = pd.Timestamp(extent.start)
        upper = (
            pd.Timestamp(metadata["effective_prediction_end"])
            if extent.end is None
            else _bounds(extent)[1]
        )
        if lower > end or upper < start:
            continue
        compare_model_facts(metadata["model_facts"], facts)
        if any(
            metadata["model_facts"][key] != facts[key]
            for key in ("fit_windows", "fit_cutoff")
        ):
            raise ValueError("model_period_prediction_fit_mismatch")
    daily = materialize_dataset_session(session.filtered_view(window))
    return daily, parameters["cutoff"]


def _summary_rows(daily, common):
    rows = []
    summary = daily_metric_summary(daily)
    missing_zero = summary.loc[
        summary.metric.eq("mse")
        & ~summary.asset_id.isin(
            summary.loc[summary.metric.eq("r2_vs_zero"), "asset_id"]
        )
    ].assign(
        metric="r2_vs_zero",
        value=np.nan,
        reason="not_saved",
        aggregation="not_saved",
        valid_session_count=None,
        observation_count=None,
    )
    for row in pd.concat([summary, missing_zero], ignore_index=True).to_dict("records"):
        if row["metric"] not in _METRICS:
            continue
        observed = daily.loc[
            daily.asset_id.eq(row["asset_id"])
            & daily.metric.eq(row["metric"])
            & daily.value.notna(),
            "timestamp",
        ]
        rows.append(
            {
                **common,
                "cutoff": pd.Timestamp(
                    TimeWindow(common["cutoff"], common["cutoff"]).start
                ),
                **{
                    key: row[key]
                    for key in (
                        "asset_id",
                        "metric",
                        "value",
                        "reason",
                        "aggregation",
                        "session_count",
                        "valid_session_count",
                        "observation_count",
                    )
                },
                "status": "DEFINED" if pd.notna(row["value"]) else "UNDEFINED",
                "first_valid_session": observed.min(),
                "last_valid_session": observed.max(),
            }
        )
    return rows


def _validation_rows(selection, common):
    winner = (
        None
        if selection is None
        else next(
            item
            for item in selection["candidates"]
            if item["hyperparameters"] == selection["selected_hyperparameters"]
        )
    )
    window = None if selection is None else TimeWindow(**selection["validation_window"])
    stopping = None if winner is None else winner.get("early_stopping")
    mse = np.nan if winner is None else winner["validation_daily_mse"]
    return [
        {
            **common,
            "stage": "validation",
            "asset_id": "__common_validation__",
            "metric": metric,
            "value": mse
            if metric == "mse"
            else np.sqrt(mse)
            if metric == "rmse"
            else np.nan,
            "status": "DEFINED"
            if winner is not None and metric in ("mse", "rmse")
            else "UNDEFINED",
            "reason": None
            if winner is not None and metric in ("mse", "rmse")
            else "not_saved",
            "source": "saved_common_validation",
            "evaluation_axis": "pooled_daily_rows",
            "aggregation": "sqrt_daily_equal_weight_mse"
            if metric == "rmse"
            else "daily_equal_weight_mean",
            "window_start": None if window is None else _bounds(window)[0],
            "window_end": None if window is None else _bounds(window)[1],
            "cutoff": None
            if selection is None
            else pd.Timestamp(
                TimeWindow(
                    selection["validation_cutoff"], selection["validation_cutoff"]
                ).start
            ),
            "session_count": None
            if selection is None
            else selection["common_sessions"],
            "valid_session_count": selection["common_sessions"]
            if winner is not None and metric in ("mse", "rmse")
            else None,
            "observation_count": None
            if selection is None
            else selection["common_rows"],
            "first_valid_session": None,
            "last_valid_session": None,
            "parameters_json": None
            if selection is None
            else render_qrp_json_object(selection["selected_hyperparameters"]),
            "iterations": None if stopping is None else stopping["best_iteration"] + 1,
        }
        for metric in _METRICS
    ]


def _fit_iterations(result):
    algorithm = result.metadata["model_algorithm"]
    if algorithm == "lasso_regression":
        return int(result.prediction_model.estimator.n_iter_)
    if algorithm == "xgboost_regressor":
        return result.prediction_model.estimator.get_booster().num_boosted_rounds()
    if algorithm in SEQUENCE_ALGORITHMS:
        stopping = result.metadata.get("early_stopping")
        return (
            stopping["best_iteration"] + 1
            if stopping is not None
            else len(result.metadata["learning_curve"]["epochs"])
        )
    return None


def evaluate_model_periods(config: Mapping) -> Path:
    """Write phase metrics and saved final-fit curves without fitting any model."""
    config = _strict_config(
        config,
        {
            "inputs",
            "periods",
            "prediction_daily",
            "sessions",
            "asset_ids",
            "evaluation_axis",
            "output",
        },
        set(),
    )
    reject_if_exists(Path(config["output"]))
    dates, _ = _evaluation_axes(
        config["evaluation_axis"], config["sessions"], config["asset_ids"]
    )
    periods = config["periods"]
    if not isinstance(periods, list) or not periods:
        raise ValueError("model_periods_required")
    ids = [item["period_id"] for item in periods]
    if any(type(item) is not str or not item for item in ids) or len(ids) != len(
        set(ids)
    ):
        raise ValueError("model_period_ids_invalid")
    sources = {}
    records, epochs, lineage = [], [], []
    for period in periods:
        period = _strict_config(
            period, {"period_id", "prediction_window", "models"}, set()
        )
        window = TimeWindow(**period["prediction_window"])
        start, end = _bounds(window)
        if start != start.normalize() or end != end.normalize() + pd.Timedelta(
            days=1
        ) - pd.Timedelta(1, "ns"):
            raise ValueError("model_period_prediction_window_must_cover_whole_days")
        if not isinstance(period["models"], Mapping) or not period["models"]:
            raise ValueError("model_period_models_required")
        for model_id, path in period["models"].items():
            if type(model_id) is not str or not model_id:
                raise ValueError("model_period_model_id_invalid")
            result, facts = load_model(Path(path))
            metadata = result.metadata
            if metadata["model_algorithm"] == "deeplob_classifier":
                raise ValueError("model_periods_require_regression")
            _validate_model_prediction_window(
                facts, PredictionWindow(window.start, window.end)
            )
            if model_id not in sources:
                sources[model_id] = _prediction_source(
                    config["prediction_daily"][model_id], model_id, config
                )
            prediction_daily, cutoff = _prediction_daily(
                sources[model_id],
                model_id,
                facts,
                window,
                _window_days(dates, (window,)),
            )
            selection = facts.get("training_selection")
            common = {
                "period_id": period["period_id"],
                "model_id": model_id,
                "label": metadata["target_column"],
                "label_unit": facts["label_unit"],
                "evaluation_axis": config["evaluation_axis"],
                "parameters_json": render_qrp_json_object(
                    facts["candidate_parameters"]
                ),
                "iterations": _fit_iterations(result),
            }
            daily = _training_daily(
                config["inputs"],
                result,
                facts,
                sessions=dates,
                evaluation_axis=config["evaluation_axis"],
            )
            daily = daily.assign(candidate_id=model_id, sample_scope="fit_sample")
            records.extend(
                _summary_rows(
                    daily,
                    {
                        **common,
                        "stage": "training",
                        "source": "frozen_model_forward",
                        "window_start": _bounds(TimeWindow(**facts["fit_windows"][0]))[
                            0
                        ],
                        "window_end": _bounds(TimeWindow(**facts["fit_windows"][-1]))[
                            1
                        ],
                        "cutoff": facts["fit_cutoff"],
                    },
                )
            )
            records.extend(_validation_rows(selection, common))
            records.extend(
                _summary_rows(
                    prediction_daily,
                    {
                        **common,
                        "stage": "prediction",
                        "source": "saved_prediction_daily",
                        "window_start": _bounds(window)[0],
                        "window_end": _bounds(window)[1],
                        "cutoff": cutoff,
                    },
                )
            )
            curve = metadata.get("learning_curve")
            if curve is not None:
                epochs.extend(
                    {
                        "period_id": period["period_id"],
                        "model_id": model_id,
                        "label_unit": facts["label_unit"],
                        "metric": curve["metric"],
                        "training_loss_scope": curve["training_loss_scope"],
                        "validation_loss_scope": curve["validation_loss_scope"],
                        **row,
                    }
                    for row in curve["epochs"]
                )
            lineage.append(
                {
                    "period_id": period["period_id"],
                    "model_id": model_id,
                    "model": str(path),
                    "model_algorithm": metadata["model_algorithm"],
                    "fit_windows": facts["fit_windows"],
                    "fit_cutoff": facts["fit_cutoff"],
                    "train_start": metadata["train_start"],
                    "train_end": metadata["train_end"],
                    "training_weighting": metadata["training_weighting"],
                    "sample_filter_mode": metadata["sample_filter_mode"],
                    "label_availability_column": metadata["label_availability_column"],
                    "label_available_max": metadata["label_available_max"],
                    "target_definition": facts["target_definition"],
                    "training_selection": selection,
                    "final_fit_curve": "saved" if curve is not None else "not_saved",
                    "historical_cv_curves": "not_saved",
                    "final_fit_early_stopping": metadata.get("early_stopping"),
                }
            )
            del result
    return write_statistics_artifact(
        Path(config["output"]),
        table_group="model_periods",
        tables={
            name: pd.DataFrame(
                rows, columns=STATISTICS_TABLE_COLUMNS[name + ".parquet"]
            )
            for name, rows in (
                ("model_period_metrics", records),
                ("model_epoch_curves", epochs),
            )
        },
        metadata={
            "parameters": config,
            "models": lineage,
            "training_metric_scope": (
                "frozen final model; unweighted daily metrics "
                "on the original fit sample"
            ),
            "epoch_loss_scope": (
                "saved optimizer-time losses; not frozen final-model training errors"
            ),
        },
    )

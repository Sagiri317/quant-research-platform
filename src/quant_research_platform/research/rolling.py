"""Select on historical sessions, refit expanding history and predict later folds."""

from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from quant_research_platform.contracts.evaluations import evaluation_sessions
from quant_research_platform.contracts.experiments import (
    label_maturity_mask,
    training_availability_columns,
)
from quant_research_platform.contracts.filtering import required_filter_columns
from quant_research_platform.contracts.json_codec import _strict_config
from quant_research_platform.contracts.models import (
    SEQUENCE_ALGORITHMS,
    _model_spec,
    model_spec_payload,
    sequence_context_columns,
    validate_model_input_roles,
)
from quant_research_platform.contracts.time import (
    TimeWindow,
    _window_payload,
)
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.research.filtering import apply_filter_mode
from quant_research_platform.research.models.fitting import (
    model_training_read_columns,
    prediction_callable,
)
from quant_research_platform.research.predictions.outputs import prediction_input_mask
from quant_research_platform.storage.datasets.reading import (
    _dataset_input_options,
    iter_dataset_chunks,
    materialize_dataset_session,
    open_dataset_session,
)


@dataclass(frozen=True)
class RollingFold:
    fold_id: str
    train: TimeWindow
    validation: TimeWindow
    prediction: TimeWindow


def _fold(fold_id, past, prediction, validation_sessions):
    return RollingFold(
        fold_id,
        TimeWindow(past[0].date(), past[-validation_sessions - 1].date()),
        TimeWindow(past[-validation_sessions].date(), past[-1].date()),
        prediction,
    )


def quarterly_folds(
    training_start, rolling_window, *, sessions, validation_sessions=20
):
    """Reserve the last historical sessions; the initial quarter is warmup."""
    start = date.fromisoformat(training_start)
    window = TimeWindow(**rolling_window)
    if type(window.start) is not date or start > window.start:
        raise ValueError("quarterly_training_start_invalid")
    if type(validation_sessions) is not int or validation_sessions < 1:
        raise ValueError("rolling_validation_sessions_invalid")
    days = evaluation_sessions(sessions)
    days = days[days >= pd.Timestamp(start)]
    folds = []
    for period in pd.period_range(window.start, window.end, freq="Q"):
        if period.start_time.date() <= start:
            continue
        prediction_start = max(window.start, period.start_time.date())
        end = min(window.end, period.end_time.date())
        past = days[days < pd.Timestamp(prediction_start)]
        if len(past) <= validation_sessions:
            continue
        folds.append(
            _fold(
                str(period),
                past,
                TimeWindow(prediction_start, end),
                validation_sessions,
            )
        )
    if not folds:
        raise ValueError("quarterly_validation_requires_prior_training")
    return tuple(folds)


def session_folds(
    sessions, *, min_train_sessions=60, validation_sessions=20, prediction_sessions=20
):
    """Expand history every prediction block on the caller's complete session axis."""
    if (
        type(min_train_sessions) is not int
        or min_train_sessions < 60
        or type(validation_sessions) is not int
        or validation_sessions < 20
        or type(prediction_sessions) is not int
        or prediction_sessions < 1
    ):
        raise ValueError("rolling_session_counts_invalid")
    days = evaluation_sessions(sessions)
    return tuple(
        _fold(
            days[offset].date().isoformat(),
            days[:offset],
            TimeWindow(
                days[offset].date(),
                days[min(offset + prediction_sessions, len(days)) - 1].date(),
            ),
            validation_sessions,
        )
        for offset in range(
            min_train_sessions + validation_sessions, len(days), prediction_sessions
        )
    )


def _config(config, *, session_mode):
    return _strict_config(
        config,
        {
            "inputs",
            "output",
            "market_id",
            "models",
            "training_start",
            "rolling_window",
            "evaluation",
            "evaluation_cutoff",
        },
        {
            "label_unit",
            "asset_ids",
            "trading_calendar",
            "sessions",
            "validation_sessions",
            "parameter_candidates",
            *({"min_train_sessions", "prediction_sessions"} if session_mode else ()),
        },
    )


def _sessions(config):
    if "sessions" in config:
        return evaluation_sessions(config["sessions"])
    from quant_research_platform.markets.implementations.a_share.calendar import (
        unified_trading_sessions,
    )

    calendar = materialize_dataset_session(
        open_dataset_session(**_dataset_input_options(config["trading_calendar"]))
    )
    return unified_trading_sessions(calendar)


def run_quarterly(config):
    config = _config(config, session_mode=False)
    days = _sessions(config)
    folds = quarterly_folds(
        config["training_start"],
        config["rolling_window"],
        sessions=days,
        validation_sessions=config.get("validation_sessions", 20),
    )
    return _run(config, folds, days)


def run_sessions(config):
    config = _config(config, session_mode=True)
    window = TimeWindow(**config["rolling_window"])
    days = _sessions(config)
    days = days[
        (days >= pd.Timestamp(config["training_start"]))
        & (days <= pd.Timestamp(window.end))
    ]
    folds = tuple(
        replace(
            fold,
            prediction=TimeWindow(
                max(fold.prediction.start, window.start), fold.prediction.end
            ),
        )
        for fold in session_folds(
            days,
            min_train_sessions=config.get("min_train_sessions", 60),
            validation_sessions=config.get("validation_sessions", 20),
            prediction_sessions=config.get("prediction_sessions", 20),
        )
        if fold.prediction.end >= window.start
    )
    if not folds:
        raise ValueError("rolling_prediction_requires_prior_sessions")
    return _run(config, folds, days)


def _columns(spec, availability):
    return tuple(
        dict.fromkeys(
            (
                *model_training_read_columns(
                    ("timestamp", "asset_id"),
                    spec.input_columns,
                    label_column=spec.target_column,
                    sample_filter_mode=spec.sample_filter_mode,
                    availability_column=availability,
                ),
                *sequence_context_columns(spec.model_algorithm, spec.hyperparameters),
                *(
                    ("market_type",)
                    if spec.training_weighting.half_life_periods is not None
                    else ()
                ),
            )
        )
    )


def _historical_validation(fold, session, candidates, days, count):
    target = candidates[0][1].target_column
    availability = session.spec.get("label_availability_columns", {}).get(
        target, training_availability_columns(target)[0]
    )
    history = TimeWindow(fold.train.start, fold.validation.end)
    cutoff = pd.Timestamp(fold.prediction.start) - pd.Timedelta(1, "ns")
    modes = tuple(dict.fromkeys(spec.sample_filter_mode for _, spec in candidates))
    columns = tuple(
        dict.fromkeys(
            (
                "timestamp",
                target,
                availability,
                *(column for mode in modes for column in required_filter_columns(mode)),
            )
        )
    )
    observed = set()
    first_ready = None
    with closing_iterator(
        iter_dataset_chunks(session.filtered_view(history).projected_view(columns))
    ) as frames:
        for frame in frames:
            for mode in modes:
                frame, _ = apply_filter_mode(
                    frame, mode=mode, metadata_key="sample_filter_mode"
                )
            mature = label_maturity_mask(
                frame,
                fitting_cutoff=cutoff,
                availability_column=availability,
                target_column=target,
            )
            observed.update(frame.loc[mature, "timestamp"].dt.normalize())
            ready = frame.loc[mature, ["timestamp", availability]].max(axis=1).min()
            if pd.notna(ready) and (first_ready is None or ready < first_ready):
                first_ready = ready
    mature_days = days[days.isin(observed)]
    if len(mature_days) < count:
        return None
    validation = mature_days[-count:]
    before = days[(days >= pd.Timestamp(history.start)) & (days < validation[0])]
    if before.empty or first_ready >= validation[0]:
        return None
    return replace(
        fold,
        train=TimeWindow(before[0].date(), before[-1].date()),
        validation=TimeWindow(validation[0].date(), validation[-1].date()),
    )


def _select_inputs(fold, session, candidates):
    groups = {}
    for _, spec in candidates:
        key = (
            spec.input_factor_columns,
            spec.input_feature_columns,
            spec.sample_filter_mode,
        )
        if key in groups:
            continue
        availability = session.spec.get("label_availability_columns", {}).get(
            spec.target_column, training_availability_columns(spec.target_column)[0]
        )
        pending, first, varying = set(spec.input_columns), {}, set()
        cutoff = pd.Timestamp(fold.validation.start) - pd.Timedelta(1, "ns")
        with closing_iterator(
            iter_dataset_chunks(
                session.filtered_view(fold.train).projected_view(
                    _columns(spec, availability)
                )
            )
        ) as frames:
            for frame in frames:
                mature = label_maturity_mask(
                    frame,
                    fitting_cutoff=cutoff,
                    availability_column=availability,
                    target_column=spec.target_column,
                )
                legal, _ = apply_filter_mode(
                    frame.loc[mature],
                    mode=spec.sample_filter_mode,
                    metadata_key="sample_filter_mode",
                )
                for column in tuple(pending):
                    values = legal[column].dropna()
                    if values.empty:
                        continue
                    first.setdefault(column, values.iloc[0])
                    if values.ne(first[column]).any():
                        varying.add(column)
                        pending.remove(column)
                if not pending:
                    break
        if not varying:
            raise ValueError("rolling_no_varying_inputs")
        groups[key] = {
            "retained": [column for column in spec.input_columns if column in varying],
            "all_missing": [
                column for column in spec.input_columns if column not in first
            ],
            "constant": [
                column
                for column in spec.input_columns
                if column in first and column not in varying
            ],
        }
    selected, facts = [], {}
    for model_id, spec in candidates:
        key = (
            spec.input_factor_columns,
            spec.input_feature_columns,
            spec.sample_filter_mode,
        )
        kept = groups[key]["retained"]
        facts[model_id] = groups[key]
        selected.append(
            (
                model_id,
                replace(
                    spec,
                    input_factor_columns=tuple(
                        column for column in spec.input_factor_columns if column in kept
                    ),
                    input_feature_columns=tuple(
                        column
                        for column in spec.input_feature_columns
                        if column in kept
                    ),
                ),
            )
        )
    return selected, facts


def _select(config, fold, session, candidates, calendar):
    from quant_research_platform.research.models.production import (
        _fit_read_windows,
        fit_model,
    )

    candidates, input_groups = _select_inputs(fold, session, candidates)
    target = candidates[0][1].target_column
    availability = session.spec.get("label_availability_columns", {}).get(
        target, training_availability_columns(target)[0]
    )
    columns = tuple(
        dict.fromkeys(
            column for _, spec in candidates for column in _columns(spec, availability)
        )
    )
    sequence = any(
        spec.model_algorithm in SEQUENCE_ALGORITHMS for _, spec in candidates
    )
    context_columns = tuple(
        dict.fromkeys(
            (
                "timestamp",
                "asset_id",
                *(
                    column
                    for _, spec in candidates
                    for column in (
                        *spec.input_columns,
                        *sequence_context_columns(
                            spec.model_algorithm, spec.hyperparameters
                        ),
                    )
                ),
            )
        )
    )
    validation_frames = [
        materialize_dataset_session(
            session.filtered_view(window).projected_view(
                context_columns if context_only else columns
            )
        ).reindex(columns=columns)
        for window, context_only in _fit_read_windows(
            (fold.validation,),
            sequence=sequence,
            context_start=fold.train.end if sequence else None,
        )
    ]
    validation = validation_frames[-1]
    scoring_context = pd.concat(validation_frames, ignore_index=True)
    cutoff = pd.Timestamp(fold.prediction.start) - pd.Timedelta(1, "ns")
    common = label_maturity_mask(
        validation,
        fitting_cutoff=cutoff,
        availability_column=availability,
        target_column=target,
    ).to_numpy()
    for mode in dict.fromkeys(spec.sample_filter_mode for _, spec in candidates):
        filtered, _ = apply_filter_mode(
            validation, mode=mode, metadata_key="sample_filter_mode"
        )
        common = common & validation.index.isin(filtered.index)
    predictions, rows = [], []
    for model_id, spec in candidates:
        validate_model_input_roles(
            session.spec,
            model_spec_payload(spec),
            market_id=config["market_id"],
            include_target=True,
            hyperparameters=spec.hyperparameters,
        )
        patience = spec.hyperparameters.get("early_stopping_rounds")
        result = fit_model(
            iter_dataset_chunks(
                session.filtered_view(fold.train).projected_view(
                    _columns(spec, availability)
                )
            ),
            spec,
            fit_windows=[_window_payload(fold.train)],
            fit_cutoff=pd.Timestamp(fold.validation.start) - pd.Timedelta(1, "ns"),
            label_availability_column=availability,
            trading_calendar=calendar,
            **(
                {
                    "validation_frames": iter(validation_frames),
                    "validation_cutoff": cutoff,
                    "validation_windows": [_window_payload(fold.validation)],
                }
                if patience is not None
                else {}
            ),
        )
        values = prediction_callable(spec.input_columns, result.prediction_model)(
            scoring_context
        )[-len(validation) :]
        if values.shape != (len(validation),):
            raise ValueError("invalid_prediction_values")
        common = common & prediction_input_mask(validation, result.metadata).to_numpy()
        predictions.append(values)
        rows.append(
            {
                "model_id": model_id,
                "hyperparameters": dict(spec.hyperparameters),
                **(
                    {"early_stopping": result.metadata["early_stopping"]}
                    if patience is not None
                    else {}
                ),
            }
        )
        del result
    if not common.any():
        raise ValueError("rolling_validation_sample_empty")
    actual = validation[target].to_numpy(dtype=float)[common]
    dates = validation.timestamp.loc[common].dt.normalize().reset_index(drop=True)
    for values, row in zip(predictions, rows, strict=True):
        if not np.isfinite(values[common]).all():
            raise ValueError("invalid_prediction_values")
        losses = pd.Series((values[common] - actual) ** 2).groupby(dates).mean()
        row["validation_daily_mse"] = float(losses.mean())
    selection = {
        "fit_window": _window_payload(fold.train),
        "validation_window": _window_payload(fold.validation),
        "validation_cutoff": cutoff.isoformat(),
        "validation_rule": "latest_mature_signal_sessions",
        "metric": "daily_mse",
        "common_rows": int(common.sum()),
        "common_sessions": int(dates.nunique()),
        "label_available_max": pd.Timestamp(
            validation.loc[common, availability].max()
        ).isoformat(),
    }
    selected = []
    for model_id in dict.fromkeys(model_id for model_id, _ in candidates):
        spec = next(spec for name, spec in candidates if name == model_id)
        scores = [row for row in rows if row["model_id"] == model_id]
        winner = min(scores, key=lambda row: row["validation_daily_mse"])
        parameters = dict(winner["hyperparameters"])
        if "early_stopping" in winner:
            parameters[
                "n_estimators"
                if spec.model_algorithm == "xgboost_regressor"
                else "epochs"
            ] = winner["early_stopping"]["best_iteration"] + 1
            parameters.pop("early_stopping_rounds")
        selected.append(
            (
                model_spec_payload(spec),
                parameters,
                {
                    **selection,
                    "candidates": scores,
                    "selected_hyperparameters": winner["hyperparameters"],
                    "inputs": input_groups[model_id],
                },
            )
        )
    return selected


def _run(config, folds, days):
    from quant_research_platform.research.evaluations.runner import evaluate
    from quant_research_platform.research.models.production import fit
    from quant_research_platform.research.predictions.runner import predict
    from quant_research_platform.storage.atomic import reject_if_exists

    models = config["models"]
    if (
        type(models) is not list
        or not models
        or len({model["model_id"] for model in models}) != len(models)
    ):
        raise ValueError("quarterly_models_invalid")
    root = Path(config["output"])
    reject_if_exists(root)
    grids = config.get("parameter_candidates", {})
    candidates = [
        (
            model["model_id"],
            _model_spec(
                {
                    **model,
                    "hyperparameters": {
                        **model.get("hyperparameters", {}),
                        **parameters,
                    },
                }
            ),
        )
        for model in models
        for parameters in grids.get(model["model_id"], [{}])
    ]
    if (
        not candidates
        or any(not grids.get(model["model_id"], [{}]) for model in models)
        or set(grids) - {model["model_id"] for model in models}
        or len({spec.target_column for _, spec in candidates}) != 1
        or any(spec.model_algorithm == "deeplob_classifier" for _, spec in candidates)
    ):
        raise ValueError("rolling_regression_candidates_invalid")
    options = _dataset_input_options(
        config["inputs"],
        row_filters=(("asset_id", "in", tuple(config["asset_ids"])),)
        if config.get("asset_ids") is not None
        else (),
    )
    session = open_dataset_session(**options)
    calendar = None
    if any(
        spec.training_weighting.half_life_periods is not None for _, spec in candidates
    ):
        calendar = materialize_dataset_session(
            open_dataset_session(**_dataset_input_options(config["trading_calendar"]))
        )
    prediction_columns = [
        signal["source_column"]
        for signal in config["evaluation"]["signals"]
        if signal["source_kind"] == "prediction"
    ]
    if len(prediction_columns) != 1:
        raise ValueError("quarterly_one_prediction_signal_required")
    results = []
    for fold in folds:
        history = TimeWindow(fold.train.start, fold.validation.end)
        selected_fold = _historical_validation(
            fold, session, candidates, days, config.get("validation_sessions", 20)
        )
        if selected_fold is None:
            results.append(
                {"fold_id": fold.fold_id, "reason": "insufficient_mature_history"}
            )
            continue
        fold = selected_fold
        window = _window_payload(fold.prediction)
        cutoff = pd.Timestamp(fold.prediction.start) - pd.Timedelta(nanoseconds=1)
        for model, parameters, selection in _select(
            config, fold, session, candidates, calendar
        ):
            destination = root / fold.fold_id / model["model_id"]
            trained = fit(
                {
                    "inputs": config["inputs"],
                    "output": str(destination / "model"),
                    "market_id": config["market_id"],
                    "model": {**model, "hyperparameters": parameters},
                    "fit_windows": [_window_payload(history)],
                    "fit_cutoff": cutoff.isoformat(),
                    "training_selection": selection,
                    **{
                        key: config[key]
                        for key in (
                            "label_unit",
                            "asset_ids",
                            "trading_calendar",
                        )
                        if key in config
                    },
                }
            )
            predicted = predict(
                {
                    "inputs": config["inputs"],
                    "model": str(trained),
                    "output": str(destination / "predictions"),
                    "prediction_window": window,
                    "sample_role": "rolling_prediction",
                    **{key: config[key] for key in ("asset_ids",) if key in config},
                }
            )
            evaluated = evaluate(
                {
                    "evaluation": config["evaluation"],
                    "inputs": config["inputs"],
                    "predictions": {prediction_columns[0]: str(predicted)},
                    "output": str(destination / "evaluation"),
                    "window": window,
                    "cutoff": config["evaluation_cutoff"],
                    "sample_role": "rolling_prediction",
                }
            )
            results.append(
                {
                    "fold_id": fold.fold_id,
                    "model": str(trained),
                    "predictions": str(predicted),
                    "evaluation": str(evaluated),
                }
            )
    return results

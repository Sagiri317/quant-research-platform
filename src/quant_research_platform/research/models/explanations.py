"""Explain one frozen model on explicit data without fitting or changing it."""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
from importlib.metadata import version
from math import fsum
from pathlib import Path
from statistics import mean as _mean

import numpy as np
import pandas as pd

from quant_research_platform.artifacts.explanations import (
    ATTRIBUTION_METHODS,
    LINEAR_EXPLANATION_ALGORITHMS,
    attribution_additivity,
    attribution_relative_tolerance,
    validate_attribution_method,
)
from quant_research_platform.compute.dataframe.cross_section import (
    _average_ranks,
    _pearson_correlation_result,
)
from quant_research_platform.contracts.experiments import (
    DEFAULT_ROOT_SEED,
    label_maturity_mask,
)
from quant_research_platform.contracts.filtering import required_filter_columns
from quant_research_platform.contracts.json_codec import _strict_config
from quant_research_platform.contracts.models import (
    _model_spec,
    _prediction_loss_reason,
    _validate_model_prediction_window,
    validate_model_input_roles,
)
from quant_research_platform.contracts.time import (
    PredictionWindow,
    TimeWindow,
    _window_payload,
    prediction_window_mask,
)
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.research.features.transforms import (
    model_input_validity_columns,
)
from quant_research_platform.research.models.fitting import prediction_callable
from quant_research_platform.research.models.persisted import load_model
from quant_research_platform.research.models.rank_ensemble import (
    ICWeightedRankEnsemble,
    RankEnsemblePredictor,
    _label_predictions,
    _rank_frame_columns,
)
from quant_research_platform.research.models.scaler import LinearScaledPredictor
from quant_research_platform.research.models.trial_selection import (
    _MeanSquaredError,
)
from quant_research_platform.research.predictions.outputs import prediction_rows
from quant_research_platform.research.randomness import derive_seed
from quant_research_platform.runtime.dataframes import iter_complete_time_frames
from quant_research_platform.storage.atomic import reject_if_exists
from quant_research_platform.storage.datasets.reading import (
    _dataset_input_options,
    iter_dataset_chunks,
    open_dataset_session,
)


def _positive(value, name, *, allow_none=False):
    if allow_none and value is None:
        return
    if type(value) is not int or value < 1:
        raise ValueError(f"explanation_{name}_invalid")


def _seed(value):
    if type(value) is not int or not 0 <= value < 2**32:
        raise ValueError("explanation_seed_invalid")
    return value


def _selection(value):
    selected = _strict_config(
        value, {"max_rows", "max_dates"}, {"asset_ids", "root_seed"}
    )
    for name in ("max_rows", "max_dates"):
        _positive(selected[name], name, allow_none=name == "max_dates")
    _seed(selected.setdefault("root_seed", DEFAULT_ROOT_SEED))
    assets = selected.get("asset_ids")
    if assets is not None and (
        type(assets) is not list
        or not assets
        or any(type(a) is not str or not a for a in assets)
        or len(set(assets)) != len(assets)
    ):
        raise ValueError("explanation_asset_ids_invalid")
    return selected


def _session(source, input_spec, result, facts, *, labels=False):
    if input_spec is not None:
        if isinstance(source, Mapping):
            raise ValueError("duplicate_input_spec")
        source = {"path": source, "spec": input_spec}
    options = _dataset_input_options(source)
    # Input row selection defines a different rank intervention domain. Keep this
    # operation's domain complete; select display/scoring assets only afterwards.
    if options.get("row_filters"):
        raise ValueError("explanation_input_row_filters_not_supported")
    selected_columns = options.get("columns")
    options.setdefault("columns", ("timestamp", "asset_id"))
    session = open_dataset_session(**options)
    validate_model_input_roles(
        session.spec,
        _model_spec(facts["model"]),
        market_id=facts["market_id"],
        include_target=labels,
        selected_columns=selected_columns,
        input_units=facts["input_units"],
    )
    return session


def _prediction_frames(session, result, window, *, extra=()):
    metadata = result.metadata
    columns = tuple(
        dict.fromkeys(
            (
                "timestamp",
                "asset_id",
                *metadata["input_columns"],
                *model_input_validity_columns(
                    tuple(metadata["input_columns"]),
                    metadata["model_algorithm"],
                    session.spec["column_roles"],
                ),
                *required_filter_columns(metadata["sample_filter_mode"]),
                *extra,
            )
        )
    )
    frames = iter_dataset_chunks(
        session.filtered_view(TimeWindow(window.start, window.end)).projected_view(
            columns
        )
    )
    with closing_iterator(
        iter_complete_time_frames(iter(frames), timestamp_column="timestamp")
    ) as source:
        for frame in source:
            usable = prediction_rows(
                frame, metadata, window, declared_columns=session.spec["column_roles"]
            )
            for _, block in usable.groupby("timestamp", sort=True):
                yield block


def _context(session, result, window, selection, *, extra=()):
    retained = []
    rng = np.random.default_rng(
        derive_seed(selection["root_seed"], "explanation_dates")
    )
    seen = 0
    limit = selection["max_dates"]
    with closing_iterator(
        _prediction_frames(session, result, window, extra=extra)
    ) as source:
        for block in source:
            seen += 1
            if limit is None or len(retained) < limit:
                retained.append(block.copy())
            else:
                index = int(rng.integers(seen))
                if index < limit:
                    retained[index] = block.copy()
    if not retained:
        raise ValueError("explanation_sample_empty")
    context = (
        pd.concat(retained, ignore_index=True)
        .sort_values(["timestamp", "asset_id"])
        .reset_index(drop=True)
    )
    return context, seen


def _selected_indices(context, selection):
    indices = np.arange(len(context))
    if selection.get("asset_ids") is not None:
        indices = indices[context.asset_id.isin(selection["asset_ids"])]
    if len(indices) > selection["max_rows"]:
        indices = np.sort(
            np.random.default_rng(
                derive_seed(selection["root_seed"], "explanation_rows")
            ).choice(indices, selection["max_rows"], replace=False)
        )
    if not len(indices):
        raise ValueError("explanation_sample_empty")
    return indices


def _background(config, result, facts):
    config = _strict_config(
        config, {"inputs", "window", "max_rows"}, {"input_spec", "root_seed"}
    )
    _positive(config["max_rows"], "background_rows")
    _seed(config.setdefault("root_seed", DEFAULT_ROOT_SEED))
    window = PredictionWindow(**config["window"])
    if window.end is None or pd.Timestamp(window.end) > pd.Timestamp(
        facts["fit_cutoff"]
    ):
        raise ValueError("explanation_background_after_fit_cutoff")
    session = _session(config["inputs"], config.get("input_spec"), result, facts)
    if not hasattr(result.prediction_model, "scaler"):
        raise ValueError("explanation_background_method_not_supported")
    rng = np.random.default_rng(
        derive_seed(config["root_seed"], "explanation_background")
    )
    selected = pd.DataFrame()
    priorities = np.empty(0)
    with closing_iterator(_prediction_frames(session, result, window)) as source:
        for block in source:
            legal = pd.Series(False, index=block.index)
            for item in facts["fit_windows"]:
                legal |= prediction_window_mask(
                    block.timestamp, PredictionWindow(**item)
                )
            if (
                not legal.all()
                or not block.timestamp.le(pd.Timestamp(facts["fit_cutoff"])).all()
                or not block.asset_id.isin(result.metadata["training_asset_ids"]).all()
            ):
                raise ValueError("explanation_background_outside_fit_domain")
            selected = pd.concat([selected, block], ignore_index=True)
            priorities = np.concatenate([priorities, rng.random(len(block))])
            if len(selected) > config["max_rows"]:
                keep = np.argsort(priorities, kind="stable")[: config["max_rows"]]
                selected = selected.iloc[keep].reset_index(drop=True)
                priorities = priorities[keep]
    if selected.empty:
        raise ValueError("explanation_background_empty")
    selected = selected.sort_values(["timestamp", "asset_id"])
    matrix = result.prediction_model.scaler.transform(
        selected.loc[:, result.metadata["input_columns"]]
    )
    return matrix, {
        "kind": "historical_reference",
        "distribution": "uniform_selected_rows",
        "inputs": config["inputs"],
        "window": _window_payload(window),
        "root_seed": config["root_seed"],
        "row_count": len(selected),
        "sample_keys": _sample_keys(selected),
        "training_sample_reconstructed": False,
    }


def _sample_keys(frame):
    return [
        [timestamp.isoformat(), asset]
        for timestamp, asset in frame[["timestamp", "asset_id"]].itertuples(
            index=False, name=None
        )
    ]


def _linear_parameters(predictor):
    if isinstance(predictor, LinearScaledPredictor):
        return np.asarray(predictor.coefficients[1:]), predictor.coefficients[0]
    return np.asarray(predictor.estimator.coef_, dtype="float64"), float(
        predictor.estimator.intercept_
    )


def _identity_output(result):
    algorithm = result.metadata["model_algorithm"]
    estimator = getattr(result.prediction_model, "estimator", None)
    if algorithm == "xgboost_regressor":
        return estimator.get_xgb_params().get("objective") in {
            "reg:squarederror",
            "reg:squaredlogerror",
            "reg:absoluteerror",
            "reg:pseudohubererror",
            "reg:quantileerror",
        }
    if algorithm == "lightgbm_regressor":
        objective = str(
            estimator.booster_.params.get("objective", "regression")
        ).split()[0]
        return objective in {
            "regression",
            "regression_l1",
            "huber",
            "fair",
            "quantile",
            "mape",
        }
    return True


def _output(result, frame, output_space):
    predictor = result.prediction_model
    raw = prediction_callable(tuple(result.metadata["input_columns"]), predictor)(frame)
    if output_space == "calibrated_label":
        calibration = getattr(predictor, "label_calibration", None)
        if calibration is None:
            raise ValueError("explanation_frozen_calibration_required")
        explained = _label_predictions(raw, calibration)
    elif output_space == "raw_margin":
        matrix = predictor.scaler.transform(
            frame.loc[:, result.metadata["input_columns"]]
        )
        algorithm = result.metadata["model_algorithm"]
        if algorithm == "xgboost_regressor":
            explained = predictor.estimator.predict(matrix, output_margin=True)
        elif algorithm == "lightgbm_regressor":
            explained = predictor.estimator.predict(matrix, raw_score=True)
        else:
            raise ValueError("explanation_raw_margin_not_supported")
    elif output_space == "model_output":
        explained = raw
    else:
        raise ValueError("explanation_output_space_invalid")
    if (
        np.shape(raw) != (len(frame),)
        or np.shape(explained) != (len(frame),)
        or not np.isfinite(raw).all()
        or not np.isfinite(explained).all()
    ):
        raise ValueError("explanation_prediction_nonfinite")
    return raw, np.asarray(explained, dtype="float64")


def _attributions(result, context, indices, method, background, output_space):
    predictor = result.prediction_model
    algorithm = result.metadata["model_algorithm"]
    validate_attribution_method(method, algorithm)
    columns = result.metadata["input_columns"]
    selected = context.iloc[indices]
    raw, explained = _output(result, context, output_space)
    method_facts = {
        "method": method,
        "implementation": "qrp",
        "background": {"kind": "none"},
    }
    if method == "rank":
        if not isinstance(predictor, (RankEnsemblePredictor, ICWeightedRankEnsemble)):
            raise ValueError("explanation_rank_model_required")
        weights = (
            np.asarray(predictor.directions) / len(predictor.columns)
            if isinstance(predictor, RankEnsemblePredictor)
            else np.asarray(predictor.weights)
        )
        contributions = np.zeros((len(selected), len(columns)))
        ranks = _rank_frame_columns(context, predictor.columns)[indices]
        contributions[:, [columns.index(c) for c in predictor.columns]] = (
            ranks * weights
        )
        base = np.zeros(len(selected))
        method_facts["baseline"] = "zero_centered_cross_section_rank"
    else:
        if not hasattr(predictor, "scaler"):
            raise ValueError("explanation_method_not_supported_for_model")
        matrix = predictor.scaler.transform(selected.loc[:, columns])
        if method in {"affine", "linear_shap"}:
            coefficients, intercept = _linear_parameters(predictor)
            if method == "affine":
                with np.errstate(over="ignore", invalid="ignore"):
                    contributions = matrix.to_numpy() * coefficients
                # Match the existing linear predictor's exact underflow recovery.
                if isinstance(predictor, LinearScaledPredictor):
                    for row, column in zip(
                        *np.nonzero(
                            (np.abs(matrix.to_numpy()) < np.finfo(float).tiny)
                            | ~np.isfinite(matrix.to_numpy())
                        ),
                        strict=True,
                    ):
                        scale = predictor.scaler
                        try:
                            contributions[row, column] = float(
                                Fraction(float(coefficients[column]))
                                * (
                                    Fraction(
                                        float(
                                            scale.input_matrix(
                                                selected.loc[:, columns]
                                            ).iloc[row, column]
                                        )
                                    )
                                    - Fraction(scale.origin_[column])
                                    - scale.offset_[column]
                                )
                                / Fraction(scale.scale_[column])
                            )
                        except OverflowError:
                            contributions[row, column] = np.nan
                base = np.full(len(matrix), intercept)
                method_facts["baseline"] = "training_scaled_zero"
            else:
                import shap

                masker = shap.maskers.Independent(
                    background[0], max_samples=len(background[0])
                )
                values = shap.LinearExplainer((coefficients, intercept), masker)(matrix)
                contributions, base = values.values, values.base_values
                method_facts.update(
                    implementation=f"shap:{version('shap')}",
                    baseline="historical_reference_expectation",
                    background=background[1],
                )
        elif method in {"native_tree", "tree_shap_path", "tree_shap_interventional"}:
            if output_space == "model_output" and not _identity_output(result):
                raise ValueError("explanation_nonidentity_link_requires_raw_margin")
            estimator = predictor.estimator
            if method == "native_tree":
                if algorithm == "xgboost_regressor":
                    import xgboost

                    booster = estimator.get_booster()
                    if estimator.get_xgb_params().get("booster", "gbtree") not in (
                        None,
                        "gbtree",
                    ):
                        raise ValueError("explanation_xgboost_booster_not_supported")
                    end = getattr(estimator, "best_iteration", None)
                    values = booster.predict(
                        xgboost.DMatrix(matrix),
                        pred_contribs=True,
                        iteration_range=(0, end + 1 if end is not None else 0),
                    )
                    implementation = f"xgboost:{version('xgboost')}:pred_contribs"
                elif algorithm == "lightgbm_regressor":
                    if estimator.booster_.params.get("linear_tree", False):
                        raise ValueError(
                            "explanation_lightgbm_linear_tree_not_supported"
                        )
                    values = estimator.predict(matrix, pred_contrib=True)
                    implementation = f"lightgbm:{version('lightgbm')}:pred_contrib"
                else:
                    raise ValueError("explanation_native_tree_not_supported")
                contributions, base = (
                    np.asarray(values)[:, :-1],
                    np.asarray(values)[:, -1],
                )
                method_facts.update(
                    implementation=implementation,
                    baseline="model_internal_tree_expectation",
                    background={"kind": "model_internal_tree_statistics"},
                )
            else:
                import shap

                interventional = method == "tree_shap_interventional"
                data = (
                    shap.maskers.Independent(
                        background[0], max_samples=len(background[0])
                    )
                    if interventional
                    else None
                )
                explainer = shap.TreeExplainer(
                    estimator,
                    data=data,
                    feature_perturbation="interventional"
                    if interventional
                    else "tree_path_dependent",
                    model_output="raw",
                )
                values = explainer(matrix, check_additivity=False)
                contributions, base = values.values, values.base_values
                method_facts.update(
                    implementation=f"shap:{version('shap')}",
                    baseline="historical_reference_expectation"
                    if interventional
                    else "model_internal_tree_expectation",
                    background=background[1]
                    if interventional
                    else {"kind": "model_internal_tree_statistics"},
                )
        else:
            raise ValueError("explanation_method_invalid")
    contributions = np.asarray(contributions, dtype="float64")
    if getattr(getattr(predictor, "scaler", None), "medians", ()):
        contributions = (
            contributions[:, : len(columns)] + contributions[:, len(columns) :]
        )
        method_facts["missing_indicators"] = "aggregated_with_original_feature"
    base = np.broadcast_to(np.asarray(base, dtype="float64"), (len(selected),)).copy()
    if output_space == "calibrated_label":
        intercept, slope = predictor.label_calibration
        contributions *= slope
        base = intercept + slope * base
        method_facts["calibration"] = {
            "intercept": intercept,
            "slope": slope,
            "source": "saved_training_calibration",
        }
    if contributions.shape != (len(selected), len(columns)):
        raise ValueError("explanation_contributions_invalid")
    relative = attribution_relative_tolerance(method, len(columns))
    residual = np.full(len(selected), np.nan)
    tolerance = np.full(len(selected), np.nan)
    reasons = []
    for row in range(len(selected)):
        residual[row], tolerance[row], reason = attribution_additivity(
            base[row], contributions[row], explained[indices[row]], relative
        )
        if reason:
            contributions[row, :] = np.nan
            base[row] = residual[row] = tolerance[row] = np.nan
        reasons.append(reason)
    method_facts["additivity_relative_tolerance"] = relative
    predictions = selected[["timestamp", "asset_id"]].copy()
    predictions["base_value"] = base
    predictions["explained_output"] = explained[indices]
    predictions["prediction_score"] = raw[indices]
    predictions["additivity_residual"] = residual
    predictions["additivity_tolerance"] = tolerance
    predictions["status"] = ["UNDEFINED" if reason else "DEFINED" for reason in reasons]
    predictions["reason"] = reasons
    rows = selected[["timestamp", "asset_id", *columns]].melt(
        id_vars=["timestamp", "asset_id"], var_name="feature", value_name="input_value"
    )
    rows["contribution"] = contributions.T.reshape(-1)
    return predictions, rows, method_facts


def _native_importance(result, facts):
    algorithm = result.metadata["model_algorithm"]
    predictor = result.prediction_model
    values = result.metadata.get("feature_importance")
    if values is None:
        raise ValueError("explanation_native_importance_unavailable")
    signed = {}
    unit = "dimensionless"
    if algorithm in LINEAR_EXPLANATION_ALGORITHMS:
        coefficients, _ = _linear_parameters(predictor)
        if predictor.scaler.medians:
            count = len(result.metadata["input_columns"])
            coefficients = coefficients[:count] + coefficients[count:]
        signed = dict(zip(result.metadata["input_columns"], coefficients, strict=True))
        kind, unit = "absolute_standardized_coefficient", facts["label_unit"]
    elif isinstance(predictor, (RankEnsemblePredictor, ICWeightedRankEnsemble)):
        weights = (
            np.asarray(predictor.directions) / len(predictor.columns)
            if isinstance(predictor, RankEnsemblePredictor)
            else predictor.weights
        )
        signed = dict(zip(predictor.columns, weights, strict=True))
        kind = "absolute_rank_weight"
    elif algorithm == "xgboost_regressor":
        kind = "xgboost_" + (predictor.estimator.importance_type or "gain")
        unit = "normalized_native_importance"
    elif algorithm == "lightgbm_regressor":
        kind = "lightgbm_" + predictor.estimator.importance_type
        unit = (
            "split_count"
            if predictor.estimator.importance_type == "split"
            else "native_gain"
        )
    else:
        kind, unit = "normalized_impurity_decrease", "dimensionless"
    return pd.DataFrame(
        [
            {
                "feature": column,
                "value": values.get(column),
                "signed_value": signed.get(column),
                "kind": kind,
                "unit": unit,
                "status": "DEFINED" if column in values else "NOT_APPLICABLE",
                "reason": "" if column in values else "excluded_at_fit",
            }
            for column in result.metadata["input_columns"]
        ]
    )


def _summaries(rows, period):
    if period not in {"month", "quarter"}:
        raise ValueError("explanation_period_invalid")
    work = rows.copy()
    work["absolute"] = work.contribution.abs()
    work["period"] = work.timestamp.dt.to_period(
        "M" if period == "month" else "Q"
    ).astype(str)
    results = []

    def stable_mean(values):
        values = values.dropna()
        return float(_mean(map(float, values))) if len(values) else np.nan

    for name, block in [("all", work), *work.groupby("period", sort=True)]:
        daily = block.groupby(["timestamp", "feature"])[
            ["contribution", "absolute"]
        ].agg(stable_mean)
        summary = (
            daily.groupby("feature")
            .agg(stable_mean)
            .rename(
                columns={
                    "contribution": "mean_signed_contribution",
                    "absolute": "mean_abs_contribution",
                }
            )
        )
        summary["observation_count"] = block.groupby("feature").contribution.count()
        summary["date_count"] = (
            block.loc[block.contribution.notna()]
            .groupby("feature")
            .timestamp.nunique()
            .reindex(summary.index, fill_value=0)
        )
        summary["sample_count"] = block.groupby("feature").size()
        summary["sample_date_count"] = block.groupby("feature").timestamp.nunique()
        # Normalize at the largest feature magnitude so sum(abs) cannot overflow.
        largest = summary.mean_abs_contribution.max()
        scaled = (
            summary.mean_abs_contribution / largest
            if largest > 0
            else summary.mean_abs_contribution * np.nan
        )
        summary["absolute_share"] = scaled / scaled.sum()
        summary["rank"] = summary.mean_abs_contribution.rank(
            method="min", ascending=False
        ).astype("Int64")
        summary["period"] = name
        summary["status"] = np.where(
            summary.observation_count.gt(0), "DEFINED", "UNDEFINED"
        )
        summary["reason"] = np.where(
            summary.observation_count.gt(0), "", "no_defined_contributions"
        )
        results.append(summary.reset_index())
    return pd.concat(results, ignore_index=True)


def _temporal_step_nanoseconds(value):
    if (
        type(value) not in {int, float}
        or (type(value) is float and not np.isfinite(value))
        or value <= 0
    ):
        raise ValueError("explanation_step_seconds_invalid")
    # Interpret declared seconds exactly at the timestamp's nanosecond resolution.
    nanoseconds = Fraction(str(value)) * 1_000_000_000
    if nanoseconds.denominator != 1 or nanoseconds > pd.Timedelta.max.value:
        raise ValueError("explanation_step_seconds_invalid")
    return int(nanoseconds)


def _permutation_config(value, columns):
    config = _strict_config(
        value,
        {"metrics", "cutoff", "axis", "groups", "repeats"},
        {"episode_column", "block_size", "step_seconds", "root_seed"},
    )
    if (
        type(config["metrics"]) is not list
        or not config["metrics"]
        or len(set(config["metrics"])) != len(config["metrics"])
        or set(config["metrics"]) - {"ic", "rankic", "mse"}
    ):
        raise ValueError("explanation_permutation_metrics_invalid")
    _positive(config["repeats"], "repeats")
    _seed(config.setdefault("root_seed", DEFAULT_ROOT_SEED))
    groups = config["groups"]
    if not isinstance(groups, dict) or not groups:
        raise ValueError("explanation_permutation_groups_required")
    for name, members in groups.items():
        if (
            not name
            or type(members) is not list
            or not members
            or len(set(members)) != len(members)
            or set(members) - set(columns)
        ):
            raise ValueError("explanation_permutation_group_invalid")
    temporal = {"episode_column", "block_size", "step_seconds"}
    if config["axis"] == "within_timestamp":
        if temporal & set(config):
            raise ValueError("explanation_unused_temporal_parameters")
    elif config["axis"] == "temporal_blocks":
        if (
            temporal - set(config)
            or not isinstance(config["episode_column"], str)
            or not config["episode_column"]
        ):
            raise ValueError("explanation_temporal_parameters_required")
        _positive(config["block_size"], "block_size")
        _temporal_step_nanoseconds(config["step_seconds"])
        if "rankic" in config["metrics"]:
            raise ValueError("explanation_temporal_rankic_not_supported")
    else:
        raise ValueError("explanation_permutation_axis_invalid")
    return config


def _permutation_segments(frame, config):
    if config["axis"] == "within_timestamp":
        segments = [
            np.asarray(index)
            for index in frame.groupby("timestamp", sort=True).indices.values()
        ]
        if any(len(index) < 2 for index in segments):
            raise ValueError("explanation_permutation_cross_section_too_small")
        return segments
    step_ns = _temporal_step_nanoseconds(config["step_seconds"])
    episode = config["episode_column"]
    if frame[episode].isna().any() or frame.asset_id.nunique() != 1:
        raise ValueError("explanation_temporal_requires_one_asset_and_episodes")
    segments = []
    previous = None
    seen = set()
    current = []
    for index, row in frame.iterrows():
        key = row[episode]
        if (
            previous is None
            or key != previous[0]
            or row.timestamp.value - previous[1].value != step_ns
        ):
            if current:
                segments.append(np.asarray(current))
            if previous is not None and key != previous[0] and key in seen:
                raise ValueError("explanation_episode_not_contiguous")
            current = []
            seen.add(key)
        current.append(index)
        previous = key, row.timestamp
    if current:
        segments.append(np.asarray(current))
    if any(len(index) <= config["block_size"] for index in segments):
        raise ValueError("explanation_temporal_episode_too_short")
    return segments


def _permutation(
    result,
    facts,
    frame,
    indices,
    availability,
    config,
    output_space,
    *,
    loss_reason=None,
):
    target = result.metadata["target_column"]
    rank = isinstance(
        result.prediction_model, (RankEnsemblePredictor, ICWeightedRankEnsemble)
    )
    if "mse" in config["metrics"] and (
        output_space == "raw_margin" or rank and output_space != "calibrated_label"
    ):
        raise ValueError("explanation_mse_requires_label_output")
    selected = pd.Series(False, index=frame.index)
    selected.iloc[indices] = True
    selected &= label_maturity_mask(
        frame,
        fitting_cutoff=config["cutoff"],
        availability_column=availability,
        target_column=target,
    )
    if not selected.any():
        raise ValueError("explanation_mature_labels_empty")
    segments = _permutation_segments(frame, config)
    baseline = _output(result, frame, output_space)[1]
    predictions = [("baseline", 0, baseline)]
    for name, columns in config["groups"].items():
        for repeat in range(config["repeats"]):
            shuffled = frame.copy()
            for segment in segments:
                # Shared sample identities keep paired model interventions aligned.
                rng = np.random.default_rng(
                    derive_seed(
                        config["root_seed"],
                        "explanation_permutation",
                        config["axis"],
                        name,
                        repeat,
                        _sample_keys(frame.loc[segment]),
                    )
                )
                if config["axis"] == "within_timestamp":
                    order = rng.permutation(segment)
                else:
                    blocks = [
                        segment[i : i + config["block_size"]]
                        for i in range(0, len(segment), config["block_size"])
                    ]
                    order = np.concatenate(
                        [blocks[i] for i in rng.permutation(len(blocks))]
                    )
                shuffled.loc[segment, columns] = frame.loc[order, columns].to_numpy()
            predictions.append(
                (name, repeat, _output(result, shuffled, output_space)[1])
            )
    temporal = config["axis"] == "temporal_blocks"
    eligible = frame.loc[selected].copy()
    periods = (
        [("all", eligible.index.to_numpy())]
        if temporal
        else list(eligible.groupby("timestamp", sort=True).indices.items())
    )
    if not temporal:
        periods = [
            (timestamp, eligible.index.to_numpy()[positions])
            for timestamp, positions in periods
        ]
    measured = []
    for name, repeat, values in predictions:
        per_period = {}
        for timestamp, positions in periods:
            labels = frame.loc[positions, target].to_numpy(dtype="float64")
            scores = values[positions]
            metrics = {}
            for metric in config["metrics"]:
                if metric == "mse" and loss_reason is not None:
                    metrics[metric] = None
                elif metric == "mse":
                    accumulator = _MeanSquaredError()
                    for score, label in zip(scores, labels, strict=True):
                        accumulator.add(score, label)
                    metrics[metric] = accumulator
                else:
                    left, right = (
                        (_average_ranks(scores), _average_ranks(labels))
                        if metric == "rankic"
                        else (scores, labels)
                    )
                    metrics[metric] = _pearson_correlation_result(left, right)[0]
            per_period[str(timestamp)] = (metrics, len(positions))
        measured.append((name, repeat, per_period))
    rows = []
    common_periods = {}
    for metric in config["metrics"]:
        key = metric
        common = [
            str(timestamp)
            for timestamp, _ in periods
            if all(item[2][str(timestamp)][0].get(key) is not None for item in measured)
        ]
        common_periods[metric] = common
        count = sum(measured[0][2][period][1] for period in common)

        def aggregate(values, common=common, metric=metric, key=key):
            if not common:
                return None
            if metric != "mse":
                return float(_mean(float(values[period][0][key]) for period in common))
            accumulator = _MeanSquaredError()
            for period in common:
                accumulator.add_period(values[period][0][key])
            result = accumulator.result()
            if result is None:
                raise ValueError("explanation_mse_unrepresentable")
            return result

        baseline_value = aggregate(measured[0][2])
        for name, repeat, values in measured[1:]:
            perturbed = aggregate(values)
            delta = (
                None
                if perturbed is None
                else perturbed - baseline_value
                if metric == "mse"
                else baseline_value - perturbed
            )
            rows.append(
                {
                    "group": name,
                    "repeat": repeat,
                    "metric": metric,
                    "baseline": baseline_value,
                    "perturbed": perturbed,
                    "delta": delta,
                    "observation_count": count,
                    "period_count": len(common),
                    "unit": facts["label_unit"] + "^2"
                    if metric == "mse"
                    else "dimensionless",
                    "status": "DEFINED" if delta is not None else "UNDEFINED",
                    "reason": ""
                    if delta is not None
                    else (
                        loss_reason
                        if metric == "mse" and loss_reason is not None
                        else "no_common_defined_periods"
                    ),
                }
            )
    repeats = pd.DataFrame(rows)
    summaries = []
    for _, block in repeats.groupby(["group", "metric"], sort=False):
        row = block.iloc[0].drop(labels=["repeat", "perturbed", "delta"]).to_dict()
        mean = deviation = None
        if block.delta.notna().all():
            values = block.delta.to_numpy(dtype="float64")
            scale = float(np.max(np.abs(values)))
            normalized = values / scale if scale else values
            center = fsum(map(float, normalized)) / len(values)
            mean = float(_mean(map(float, values)))
            # Population SD cannot exceed max(abs(values)); normalize before
            # squaring to retain finite large-MSE permutation dispersion.
            variance = fsum((float(value) - center) ** 2 for value in normalized) / len(
                values
            )
            deviation = scale * min(1.0, variance**0.5)
        row.update(delta_mean=mean, delta_std=deviation, repeats=len(block))
        summaries.append(row)
    return (
        repeats,
        pd.DataFrame(summaries),
        {
            "config": config,
            "mature_scoring_rows": int(selected.sum()),
            "perturbation_coordinate": "assembled_model_inputs",
            "std_definition": "permutation_population_sd_not_confidence_interval",
            "scoring_keys": _sample_keys(frame.loc[selected]),
            "common_metric_periods": common_periods,
        },
    )


def explain(config: Mapping) -> Path:
    """Publish frozen-model importance, contributions and OOS perturbations."""
    config = _strict_config(
        config,
        {"model", "output", "methods"},
        {
            "inputs",
            "input_spec",
            "window",
            "sample_role",
            "selection",
            "output_space",
            "background",
            "period",
            "permutation",
        },
    )
    output = Path(config["output"])
    reject_if_exists(output)
    methods = config["methods"]
    if (
        type(methods) is not list
        or not methods
        or any(type(value) is not str for value in methods)
        or len(set(methods)) != len(methods)
        or set(methods) - ATTRIBUTION_METHODS - {"native_importance", "permutation"}
    ):
        raise ValueError("explanation_methods_invalid")
    attribution = set(methods) & ATTRIBUTION_METHODS
    if len(attribution) > 1:
        raise ValueError("explanation_single_attribution_method_required")
    needs_data = bool(attribution or "permutation" in methods)
    data_fields = {
        "inputs",
        "input_spec",
        "window",
        "sample_role",
        "selection",
        "output_space",
        "background",
        "period",
        "permutation",
    }
    if not needs_data and data_fields & set(config):
        raise ValueError("explanation_unused_data_parameters")
    if ("permutation" in methods) != ("permutation" in config):
        raise ValueError("explanation_permutation_config_required")
    needs_background = bool(attribution & {"linear_shap", "tree_shap_interventional"})
    if needs_background != ("background" in config):
        raise ValueError("explanation_background_config_mismatch")
    if not attribution and "period" in config:
        raise ValueError("explanation_unused_period")
    result, facts = load_model(Path(config["model"]))
    metadata = {
        "kind": "MODEL_EXPLANATION",
        "model": config["model"],
        "model_id": facts["model"]["model_id"],
        "model_algorithm": result.metadata["model_algorithm"],
        "input_columns": result.metadata["input_columns"],
        "input_units": facts["input_units"],
        "target_column": result.metadata["target_column"],
        "label_unit": facts["label_unit"],
        "methods": methods,
    }
    tables = {}
    if "native_importance" in methods:
        tables["native_importance"] = _native_importance(result, facts)
    if needs_data:
        if {"inputs", "window", "sample_role", "selection", "output_space"} - set(
            config
        ):
            raise ValueError("explanation_data_parameters_required")
        if type(config["sample_role"]) is not str or not config["sample_role"].strip():
            raise ValueError("explanation_sample_role_invalid")
        window = PredictionWindow(**config["window"])
        if window.end is None:
            raise ValueError("explanation_finite_window_required")
        _validate_model_prediction_window(facts, window)
        selection = _selection(config["selection"])
        permutation = (
            _permutation_config(config["permutation"], result.metadata["input_columns"])
            if "permutation" in config
            else None
        )
        if (
            permutation
            and permutation["axis"] == "temporal_blocks"
            and selection["max_dates"] is not None
        ):
            raise ValueError("explanation_temporal_requires_contiguous_window")
        session = _session(
            config["inputs"],
            config.get("input_spec"),
            result,
            facts,
            labels=permutation is not None,
        )
        availability = None
        extra = []
        if permutation:
            availability = session.spec.get("label_availability_columns", {}).get(
                result.metadata["target_column"],
                result.metadata["label_availability_column"],
            )
            extra.extend((result.metadata["target_column"], availability))
            if permutation["axis"] == "temporal_blocks":
                extra.append(permutation["episode_column"])
        context, dates = _context(session, result, window, selection, extra=extra)
        indices = _selected_indices(context, selection)
        rank = isinstance(
            result.prediction_model, (RankEnsemblePredictor, ICWeightedRankEnsemble)
        )
        unit = (
            "raw_margin"
            if config["output_space"] == "raw_margin"
            else "dimensionless_rank_score"
            if rank and config["output_space"] == "model_output"
            else facts["label_unit"]
        )
        metadata.update(
            inputs=config["inputs"],
            window=_window_payload(window),
            sample_role=config["sample_role"],
            selection=selection,
            available_date_count=dates,
            context_row_count=len(context),
            selected_row_count=len(indices),
            context_timestamps=[
                item.isoformat() for item in context.timestamp.drop_duplicates()
            ],
            selected_timestamps=[
                item.isoformat()
                for item in context.iloc[indices].timestamp.drop_duplicates()
            ],
            output_space=config["output_space"],
            unit=unit,
            input_coordinate="assembled_model_inputs",
            aggregation="equal_timestamp_means",
            period=config.get("period", "month"),
        )
        if attribution:
            method = next(iter(attribution))
            background = (
                _background(config["background"], result, facts)
                if needs_background
                else None
            )
            predictions, contributions, method_facts = _attributions(
                result, context, indices, method, background, config["output_space"]
            )
            tables.update(
                predictions=predictions,
                contributions=contributions,
                summary=_summaries(contributions, config.get("period", "month")),
            )
            metadata["attribution"] = method_facts
        if permutation:
            repeats, summary, details = _permutation(
                result,
                facts,
                context,
                indices,
                availability,
                permutation,
                config["output_space"],
                loss_reason=_prediction_loss_reason(
                    facts,
                    result.metadata["target_column"],
                    session.spec,
                    session.manifest,
                ),
            )
            tables.update(permutation_repeats=repeats, permutation_summary=summary)
            metadata["permutation"] = details
    from quant_research_platform.artifacts.explanations import write_explanation

    return write_explanation(output, metadata=metadata, tables=tables)

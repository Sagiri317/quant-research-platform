"""Daily equal-weight prediction comparisons on declared session and sample axes."""

from pathlib import Path

import numpy as np
import pandas as pd

from quant_research_platform.artifacts.outputs.prediction import read_predictions
from quant_research_platform.compute.classification import classification_metrics
from quant_research_platform.compute.dataframe.cross_section import (
    _pearson_correlation_result,
)
from quant_research_platform.contracts.data import DataSpec
from quant_research_platform.contracts.evaluations import (
    EVALUATION_AXES,
    evaluation_sessions,
)
from quant_research_platform.contracts.experiments import label_maturity_mask
from quant_research_platform.contracts.json_codec import _strict_config
from quant_research_platform.contracts.models import (
    PROBABILITY_COLUMNS,
    _prediction_target_reason,
)
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.research.statistics import (
    _validate_bootstrap_configuration,
    stationary_bootstrap_mean_ci,
    time_series_inference_summary,
)
from quant_research_platform.storage.atomic import atomic_write_directory
from quant_research_platform.storage.datasets import write_dataset
from quant_research_platform.storage.datasets.reading import (
    _dataset_input_options,
    materialize_dataset_session,
    open_dataset_session,
)

_CROSS_SECTION_ASSET = "__cross_section__"
_DAILY_DTYPES = {
    "timestamp": "datetime64[ns]",
    "asset_id": "string",
    "metric": "string",
    "value": "float64",
    "reason": "string",
    "observation_count": "int64",
    "input_count": "int64",
    "block_count": "int64",
    "valid_block_count": "int64",
}


def _evaluation_axes(evaluation_axis, sessions, asset_ids):
    dates = evaluation_sessions(sessions)
    if evaluation_axis not in EVALUATION_AXES:
        raise ValueError("evaluation_axis_invalid")
    assets = tuple(asset_ids)
    if not assets or len(set(assets)) != len(assets):
        raise ValueError("evaluation_asset_ids_invalid")
    if _CROSS_SECTION_ASSET in assets:
        raise ValueError("evaluation_asset_id_reserved")
    return dates, assets


def _daily_blocks(frame, *, evaluation_axis, sessions, asset_ids):
    selected = frame.loc[frame.asset_id.isin(asset_ids)]
    if selected.duplicated(["timestamp", "asset_id"]).any():
        raise ValueError("evaluation_input_keys_duplicate")
    dates = selected.timestamp.dt.normalize()
    if not dates.isin(sessions).all():
        raise ValueError("evaluation_input_session_not_declared")
    cross_sectional = evaluation_axis == "cross_sectional"
    grouped = dict(
        tuple(
            selected.groupby(
                dates if cross_sectional else [selected.asset_id, dates], sort=False
            )
        )
    )
    for asset in (_CROSS_SECTION_ASSET,) if cross_sectional else asset_ids:
        for day in sessions:
            block = grouped.get(
                day if cross_sectional else (asset, day), selected.iloc[:0]
            )
            parts = (
                [part for _, part in block.groupby("timestamp", sort=False)]
                if cross_sectional
                else [block]
                if len(block)
                else []
            )
            yield asset, day, block, parts


def _finite_mean(values):
    if not len(values):
        return np.nan, "no_observations"
    value = float(np.mean(values))
    return (value, None) if np.isfinite(value) else (np.nan, "nonfinite_result")


def _signal_metric_blocks(
    frame,
    *,
    evaluation_axis,
    sessions,
    asset_ids,
    regression=False,
    baseline_columns=(),
    prediction_column="signal",
):
    """Compute each timestamp cross section or asset-day exactly once."""
    sessions, asset_ids = _evaluation_axes(evaluation_axis, sessions, asset_ids)
    baselines = tuple(baseline_columns)
    if len(set(baselines)) != len(baselines):
        raise ValueError("evaluation_baselines_invalid")
    rows = []
    for asset, day, block, parts in _daily_blocks(
        frame, evaluation_axis=evaluation_axis, sessions=sessions, asset_ids=asset_ids
    ):
        for part in parts or [block]:
            usable = part.loc[np.isfinite(part.signal) & np.isfinite(part.label)]
            x = usable.signal.to_numpy(dtype=float)
            y = usable.label.to_numpy(dtype=float)
            metrics = {}
            for metric, left, right in (
                ("ic", x, y),
                (
                    "rank_ic",
                    usable.signal.rank().to_numpy(),
                    usable.label.rank().to_numpy(),
                ),
            ):
                value, reason = _pearson_correlation_result(left, right)
                metrics[metric] = (value, reason, len(y))
            if regression:
                economic = usable.loc[np.isfinite(usable[prediction_column])]
                x = economic[prediction_column].to_numpy(dtype=float)
                y = economic.label.to_numpy(dtype=float)
                with np.errstate(over="ignore", invalid="ignore"):
                    squared, absolute = (x - y) ** 2, np.abs(x - y)
                for metric, values in (("mse", squared), ("mae", absolute)):
                    metrics[metric] = (*_finite_mean(values), len(y))
                for baseline in baselines:
                    prediction = (
                        np.zeros(len(economic))
                        if baseline == "zero"
                        else economic[baseline].to_numpy(dtype=float)
                    )
                    paired = np.isfinite(prediction)
                    with np.errstate(over="ignore", invalid="ignore"):
                        numerator = (x[paired] - y[paired]) ** 2
                        denominator = (prediction[paired] - y[paired]) ** 2
                    model_value, model_reason = _finite_mean(numerator)
                    baseline_value, baseline_reason = _finite_mean(denominator)
                    reason = model_reason or baseline_reason
                    for metric, value in (
                        (f"matched_model_mse_vs_{baseline}", model_value),
                        (f"matched_baseline_mse_vs_{baseline}", baseline_value),
                    ):
                        metrics[metric] = (
                            value if reason is None else np.nan,
                            reason,
                            int(paired.sum()),
                        )
            timestamp = (
                part.timestamp.iloc[0]
                if len(part) and evaluation_axis == "cross_sectional"
                else day
            )
            for metric, (value, reason, count) in metrics.items():
                rows.append(
                    dict(
                        timestamp=timestamp,
                        asset_id=asset,
                        metric=metric,
                        value=value,
                        reason=reason,
                        observation_count=count,
                        input_count=len(part),
                        block_count=int(bool(len(part))),
                        valid_block_count=int(reason is None),
                    )
                )
    return pd.DataFrame(rows, columns=_DAILY_DTYPES).astype(_DAILY_DTYPES)


def _common_metric_blocks(blocks, *, metric_candidates):
    """Match defined blocks independently within each applicable metric family."""
    result = blocks.reset_index(drop=True).copy()
    for metric, candidates in metric_candidates.items():
        selected = result.loc[
            result.metric.eq(metric) & result.candidate_id.isin(candidates)
        ]
        if selected.empty:
            continue
        defined = (
            np.isfinite(selected.value)
            .groupby([selected.timestamp, selected.asset_id], sort=False)
            .transform("all")
        )
        unmatched = selected.index[~defined]
        result.loc[unmatched, "value"] = np.nan
        result.loc[unmatched, "reason"] = "no_common_defined_blocks"
        result.loc[unmatched, ["observation_count", "valid_block_count"]] = 0
    return result


def _daily_signal_metrics_from_blocks(blocks):
    """Reduce one candidate's existing block estimates, preserving empty days."""
    rows = []
    days = blocks.timestamp.dt.normalize()
    for (day, asset, metric), part in blocks.groupby(
        [days, "asset_id", "metric"], sort=False
    ):
        value, reason = _finite_mean(part.value.dropna())
        reasons = part.reason.dropna()
        if reason is not None and len(reasons):
            reason = (
                reasons.iloc[0] if reasons.nunique() == 1 else "no_valid_observations"
            )
        if metric not in {"ic", "rank_ic"} and reasons.eq("nonfinite_result").any():
            value, reason = np.nan, "nonfinite_result"
        rows.append(
            dict(
                timestamp=day,
                asset_id=asset,
                metric=metric,
                value=value,
                reason=reason,
                **{
                    column: int(part[column].sum())
                    for column in (
                        "observation_count",
                        "input_count",
                        "block_count",
                        "valid_block_count",
                    )
                },
            )
        )
    daily = pd.DataFrame(rows, columns=_DAILY_DTYPES).astype(_DAILY_DTYPES)
    mse = daily.loc[daily.metric.eq("mse")].copy()
    if not mse.empty:
        rows.extend(
            mse.assign(metric="rmse", value=np.sqrt(mse.value)).to_dict("records")
        )
    indexed = daily.set_index(["timestamp", "asset_id", "metric"])
    matched = daily.loc[daily.metric.str.startswith("matched_model_mse_vs_")]
    for row in matched.to_dict("records"):
        baseline = row["metric"].removeprefix("matched_model_mse_vs_")
        denominator = indexed.loc[
            (row["timestamp"], row["asset_id"], f"matched_baseline_mse_vs_{baseline}")
        ]
        value = np.nan
        reason = row["reason"] if pd.notna(row["reason"]) else denominator.reason
        if pd.isna(reason):
            reason = None
        if reason is None:
            if denominator.value == 0:
                reason = "zero_baseline_error"
            else:
                value = 1 - row["value"] / denominator.value
                if not np.isfinite(value):
                    value, reason = np.nan, "nonfinite_result"
        rows.append(
            {
                **row,
                "metric": f"r2_vs_{baseline}",
                "value": value,
                "reason": reason,
                "valid_block_count": row["valid_block_count"] if reason is None else 0,
            }
        )
    return pd.DataFrame(rows, columns=_DAILY_DTYPES).astype(_DAILY_DTYPES)


def daily_signal_metrics(
    frame,
    *,
    evaluation_axis,
    sessions,
    asset_ids,
    regression=False,
    baseline_columns=(),
    prediction_column="signal",
):
    """Reduce finite signal/label pairs on the declared daily axis.

    Keep excluded rows and mask signal or label with NaN to retain input
    coverage. Ranks are recomputed on the retained block. Regression alone
    intersects the label-scale prediction_column domain; baseline missingness
    affects only matched errors. Cross-sectional blocks are timestamps and
    time-series blocks are asset-days. RMSE and R-squared derive from mean
    squared errors, never from means of block ratios or square roots.
    """
    return _daily_signal_metrics_from_blocks(
        _signal_metric_blocks(
            frame,
            evaluation_axis=evaluation_axis,
            sessions=sessions,
            asset_ids=asset_ids,
            regression=regression,
            baseline_columns=baseline_columns,
            prediction_column=prediction_column,
        )
    )


def daily_metric_summary(daily):
    """Summarize day-weighted errors; RMSE and R-squared are not daily means."""
    keys = ["asset_id", "candidate_id", "sample_scope"]
    rows = []
    for identity, group in daily.groupby(keys, sort=False):
        series = {
            metric: part.set_index("timestamp")
            for metric, part in group.groupby("metric", sort=False)
        }
        for metric, part in series.items():
            values = part.value
            coverage = part
            aggregation = "daily_equal_weight_mean"
            if metric == "rmse":
                values = series["mse"].value
                coverage = series["mse"]
                aggregation = "sqrt_daily_equal_weight_mse"
            elif metric.startswith("r2_vs_"):
                baseline = metric.removeprefix("r2_vs_")
                numerator = series[f"matched_model_mse_vs_{baseline}"].value
                denominator = series[f"matched_baseline_mse_vs_{baseline}"].value
                keep = numerator.notna() & denominator.notna()
                values = numerator.loc[keep]
                coverage = series[f"matched_model_mse_vs_{baseline}"].loc[keep]
                aggregation = "ratio_daily_equal_weight_mse"
            valid = values.dropna()
            value, reason = _finite_mean(valid)
            if reason is not None and not len(valid):
                declared_reasons = part.reason.dropna().unique()
                if len(declared_reasons) == 1:
                    reason = declared_reasons[0]
            if metric == "rmse" and reason is None:
                value = np.sqrt(value)
            elif metric.startswith("r2_vs_") and reason is None:
                baseline_mean = float(denominator.loc[keep].mean())
                if baseline_mean == 0:
                    value, reason = np.nan, "zero_baseline_error"
                else:
                    value = 1 - value / baseline_mean
                    if not np.isfinite(value):
                        value, reason = np.nan, "nonfinite_result"
            rows.append(
                dict(
                    zip(keys, identity, strict=True),
                    metric=metric,
                    value=value,
                    reason=reason,
                    aggregation=aggregation,
                    session_count=len(part),
                    valid_session_count=len(valid),
                    observation_count=int(coverage.observation_count.sum()),
                    input_count=int(part.input_count.sum()),
                    block_count=int(part.block_count.sum()),
                    valid_block_count=int(coverage.valid_block_count.sum()),
                )
            )
    dtypes = {
        **dict.fromkeys(keys, "string"),
        "metric": "string",
        "value": "float64",
        "reason": "string",
        "aggregation": "string",
        **dict.fromkeys(
            (
                "session_count",
                "valid_session_count",
                "observation_count",
                "input_count",
                "block_count",
                "valid_block_count",
            ),
            "int64",
        ),
    }
    return pd.DataFrame(rows, columns=dtypes).astype(dtypes)


def _daily_classification(frame, *, evaluation_axis, sessions, asset_ids, threshold):
    rows = []
    for asset, day, block, parts in _daily_blocks(
        frame, evaluation_axis=evaluation_axis, sessions=sessions, asset_ids=asset_ids
    ):
        daily = {}
        count = 0
        for part in parts:
            usable = part.loc[np.isfinite(part.signal) & np.isfinite(part.label)]
            count += len(usable)
            for metric, value in classification_metrics(
                usable.label.to_numpy(),
                usable[list(PROBABILITY_COLUMNS)].to_numpy(),
                threshold,
            ).items():
                daily.setdefault(metric, []).append(value)
        if not parts:
            daily = {
                metric: []
                for metric in classification_metrics(
                    np.array([]), np.empty((0, len(PROBABILITY_COLUMNS))), threshold
                )
            }
        for metric, values in daily.items():
            values = [
                value for value in values if value is not None and np.isfinite(value)
            ]
            value, reason = _finite_mean(values)
            rows.append(
                dict(
                    timestamp=day,
                    asset_id=asset,
                    metric=metric,
                    value=value,
                    reason=reason,
                    observation_count=count,
                    input_count=len(block),
                    block_count=len(parts),
                    valid_block_count=len(values),
                )
            )
    return pd.DataFrame(rows, columns=_DAILY_DTYPES).astype(_DAILY_DTYPES)


def daily_inference(daily, *, groups, statistics, sessions):
    """Infer each declared family without compressing missing sessions."""
    policy = _strict_config(
        statistics,
        {
            "hac_lag",
            "block_size",
            "repetitions",
            "seed",
            "confidence",
            "fdr_method",
            "alpha",
        },
        set(),
    )
    _validate_bootstrap_configuration(
        policy["block_size"], policy["repetitions"], policy["seed"]
    )
    results = []
    for (asset, scope, metric), rows in daily.groupby(
        ["asset_id", "sample_scope", "metric"], sort=False
    ):
        matrix = rows.pivot(
            index="timestamp", columns="candidate_id", values="value"
        ).reindex(sessions)
        for family, candidates in groups.items():
            selected = [name for name in candidates if name in matrix]
            if not selected:
                continue
            table = (
                matrix.reindex(columns=candidates)
                .rename_axis("timestamp")
                .reset_index()
            )
            inferred = time_series_inference_summary(
                table,
                identity_column="timestamp",
                max_lags=policy["hac_lag"],
                fdr_method=policy["fdr_method"],
                fdr_alpha=policy["alpha"],
                confidence=policy["confidence"],
            ).rename(columns={"timestamp": "candidate_id"})
            inferred["bootstrap_lower"] = np.nan
            inferred["bootstrap_upper"] = np.nan
            inferred["bootstrap_reason"] = "incomplete_session_clock"
            for index, row in inferred.iterrows():
                values = table[row.candidate_id]
                if values.isna().any() or len(values) < 2:
                    continue
                interval = stationary_bootstrap_mean_ci(
                    values,
                    block_size=policy["block_size"],
                    repetitions=policy["repetitions"],
                    seed=policy["seed"],
                    confidence=policy["confidence"],
                )
                inferred.loc[index, ["bootstrap_lower", "bootstrap_upper"]] = (
                    interval.lower,
                    interval.upper,
                )
                inferred.loc[index, "bootstrap_reason"] = ""
            results.append(
                inferred.assign(
                    asset_id=asset,
                    sample_scope=scope,
                    metric=metric,
                    family=family,
                    aggregation="mean_daily_value",
                )
            )
    dtypes = {
        "candidate_id": "string",
        "observation_count": "int64",
        "status": "string",
        "reason": "string",
        **dict.fromkeys(
            (
                "mean",
                "standard_error",
                "t_statistic",
                "p_value",
                "ci_lower",
                "ci_upper",
            ),
            "float64",
        ),
        "fdr_method": "string",
        "fdr_alpha": "float64",
        "fdr_adjusted_p_value": "float64",
        "bootstrap_lower": "float64",
        "bootstrap_upper": "float64",
        **dict.fromkeys(
            (
                "bootstrap_reason",
                "asset_id",
                "sample_scope",
                "metric",
                "family",
                "aggregation",
            ),
            "string",
        ),
    }
    result = (
        pd.concat(results, ignore_index=True)
        if results
        else pd.DataFrame(columns=dtypes)
    )
    return result.astype(dtypes)


def _write_table(root, name, frame, key, config):
    roles = dict.fromkeys(frame.columns, "context")
    roles.update(
        {column: "timestamp" if column == "timestamp" else "key" for column in key}
    )
    write_dataset(
        root / name,
        frame,
        spec=DataSpec(
            data_id=name,
            frequency="1d",
            grain="research_row",
            column_roles=roles,
            primary_key=key,
            partition_columns=(),
        ),
        metadata={"parameters": config},
    )


def evaluate_time_series(config):
    """Evaluate saved forecasts or factors, preserving own and paired coverage."""
    config = _strict_config(
        config,
        {
            "inputs",
            "predictions",
            "factor_columns",
            "label",
            "window",
            "cutoff",
            "sessions",
            "asset_ids",
            "comparison_groups",
            "inference_groups",
            "statistics",
            "output",
        },
        {"comparisons", "evaluation_axis", "r2_baselines"},
    )
    config.setdefault("evaluation_axis", "single_asset_time_series")
    config.setdefault("r2_baselines", ["zero"])
    window = TimeWindow(**config["window"])
    dates, asset_ids = _evaluation_axes(
        config["evaluation_axis"], config["sessions"], config["asset_ids"]
    )
    if len(dates) and (
        dates[0] < pd.Timestamp(window.start).normalize()
        or (window.end is not None and dates[-1] > pd.Timestamp(window.end).normalize())
    ):
        raise ValueError("evaluation_sessions_outside_window")
    factors = config["factor_columns"]
    candidates = config["predictions"]
    names = [*candidates, *factors]
    if not names or len(set(names)) != len(names):
        raise ValueError("evaluation_candidate_ids_invalid")
    baselines = config["r2_baselines"]
    if (
        type(baselines) is not list
        or len(set(baselines)) != len(baselines)
        or any(name != "zero" and name not in candidates for name in baselines)
    ):
        raise ValueError("evaluation_baselines_invalid")
    for members in config["comparison_groups"].values():
        if (
            not members
            or len(set(members)) != len(members)
            or set(members) - set(names)
        ):
            raise ValueError("evaluation_comparison_group_invalid")
    session = open_dataset_session(
        **_dataset_input_options(
            config["inputs"],
            time_window=window,
            row_filters=(("asset_id", "in", asset_ids),),
        )
    )
    label = config["label"]
    availability = session.spec["label_availability_columns"][label]
    columns = ["timestamp", "asset_id", "eligible", label, availability, *factors]
    frame = materialize_dataset_session(session.projected_view(columns))
    if frame.duplicated(["timestamp", "asset_id"]).any():
        raise ValueError("evaluation_input_keys_duplicate")
    mature = label_maturity_mask(
        frame,
        fitting_cutoff=config["cutoff"],
        availability_column=availability,
        target_column=label,
    )
    frame.loc[~(mature & frame.eligible.fillna(False)), label] = np.nan
    base = frame.set_index(["timestamp", "asset_id"])
    scores, probability, thresholds, loss_reasons = {}, {}, {}, {}
    for candidate, paths in candidates.items():
        if type(paths) is not list or not paths:
            raise ValueError("evaluation_predictions_required")
        parts = []
        identity = None
        for path in paths:
            part, metadata = read_predictions(
                Path(path), time_window=window, asset_ids=asset_ids
            )
            model = metadata["model_facts"]["model"]
            if model["target_column"] != label:
                raise ValueError("evaluation_prediction_target_mismatch")
            current_identity = (
                model["model_algorithm"],
                metadata["model_facts"]["label_unit"],
            )
            if identity is not None and current_identity != identity:
                raise ValueError("evaluation_candidate_signal_mismatch")
            identity = current_identity
            reason = _prediction_target_reason(
                metadata["model_facts"], label, session.spec, session.manifest
            )
            if reason is not None:
                loss_reasons[candidate] = reason
            if (
                session.spec.get("market_id") is not None
                and metadata["model_facts"]["market_id"] != session.spec["market_id"]
            ):
                raise ValueError("evaluation_market_mismatch")
            if model["model_algorithm"] == "deeplob_classifier":
                threshold = model["hyperparameters"]["class_threshold"]
                if candidate in thresholds and thresholds[candidate] != threshold:
                    raise ValueError("evaluation_class_threshold_mismatch")
                thresholds[candidate] = threshold
            parts.append(part)
        predicted = pd.concat(parts, ignore_index=True).set_index(
            ["timestamp", "asset_id"]
        )
        if not predicted.index.is_unique:
            raise ValueError("evaluation_prediction_keys_duplicate")
        predicted = predicted.reindex(base.index)
        scores[candidate] = predicted.prediction_score
        if candidate in thresholds:
            probability[candidate] = predicted[list(PROBABILITY_COLUMNS)]
    scores.update({name: base[name] for name in factors})
    score_frame = pd.DataFrame(scores, index=base.index)
    if np.isinf(score_frame.to_numpy(dtype=float)).any():
        raise ValueError("evaluation_signal_nonfinite")
    valid = np.isfinite(score_frame).mul(np.isfinite(base[label]), axis=0)
    scopes = {"own": names, **config["comparison_groups"]}
    if "own" in config["comparison_groups"]:
        raise ValueError("evaluation_scope_reserved")
    rows = []
    baseline_columns = {
        name: "zero" if name == "zero" else f"baseline_{name}" for name in baselines
    }
    baseline_metrics = {
        prefix + column: prefix + name
        for name, column in baseline_columns.items()
        for prefix in (
            "matched_model_mse_vs_",
            "matched_baseline_mse_vs_",
            "r2_vs_",
        )
    }
    for scope, members in scopes.items():
        paired = valid[members].all(axis=1)
        scope_blocks, classifications = [], []
        for candidate in members:
            mask = valid[candidate] if scope == "own" else paired
            selected = pd.DataFrame(
                {"signal": score_frame[candidate].where(mask), "label": base[label]}
            )
            for name, column in baseline_columns.items():
                if name != "zero":
                    selected[column] = score_frame[name]
            if candidate in thresholds:
                selected = selected.join(probability[candidate])
            selected = selected.reset_index()
            kwargs = dict(
                evaluation_axis=config["evaluation_axis"],
                sessions=dates,
                asset_ids=asset_ids,
            )
            metrics = _signal_metric_blocks(
                selected,
                **kwargs,
                regression=candidate in candidates and candidate not in thresholds,
                baseline_columns=tuple(baseline_columns.values()),
            )
            metrics["metric"] = metrics.metric.map(
                lambda name: baseline_metrics.get(name, name)
            )
            if candidate in thresholds:
                classified = _daily_classification(
                    selected, **kwargs, threshold=thresholds[candidate]
                )
                if candidate in loss_reasons:
                    classified["value"] = np.nan
                    classified["reason"] = loss_reasons[candidate]
                    classified["valid_block_count"] = 0
                classifications.append(
                    classified.assign(candidate_id=candidate, sample_scope=scope)
                )
            if candidate in loss_reasons:
                affected = ~metrics.metric.isin(["ic", "rank_ic"])
                metrics.loc[affected, "value"] = np.nan
                metrics.loc[affected, "reason"] = loss_reasons[candidate]
                metrics.loc[affected, "valid_block_count"] = 0
            for baseline in baselines:
                baseline_reason = (
                    "prediction_direction_score_not_return"
                    if baseline in thresholds
                    else loss_reasons.get(baseline)
                )
                if baseline_reason:
                    affected = metrics.metric.isin(
                        [
                            prefix + baseline
                            for prefix in (
                                "matched_model_mse_vs_",
                                "matched_baseline_mse_vs_",
                                "r2_vs_",
                            )
                        ]
                    )
                    metrics.loc[affected, "value"] = np.nan
                    metrics.loc[affected, "reason"] = baseline_reason
                    metrics.loc[affected, "valid_block_count"] = 0
            scope_blocks.append(metrics.assign(candidate_id=candidate))
        blocks = pd.concat(scope_blocks, ignore_index=True)
        if scope != "own":
            regression_members = [
                name
                for name in members
                if name in candidates
                and name not in thresholds
                and name not in loss_reasons
            ]
            metric_candidates = {
                "ic": members,
                "rank_ic": members,
                "mse": regression_members,
                "mae": regression_members,
                **{
                    prefix + baseline: regression_members
                    for baseline in baselines
                    if baseline not in thresholds and baseline not in loss_reasons
                    for prefix in ("matched_model_mse_vs_", "matched_baseline_mse_vs_")
                },
            }
            blocks = _common_metric_blocks(blocks, metric_candidates=metric_candidates)
        for candidate, candidate_blocks in blocks.groupby("candidate_id", sort=False):
            rows.append(
                _daily_signal_metrics_from_blocks(candidate_blocks).assign(
                    candidate_id=candidate, sample_scope=scope
                )
            )
        rows.extend(classifications)
    daily = (
        pd.concat(rows, ignore_index=True)
        if rows
        else pd.DataFrame(
            columns=[*_DAILY_DTYPES, "candidate_id", "sample_scope"]
        ).astype({**_DAILY_DTYPES, "candidate_id": "string", "sample_scope": "string"})
    )
    summary = daily_metric_summary(daily)
    comparisons = []
    for comparison in config.get("comparisons", []):
        comparison = _strict_config(
            comparison, {"comparison_id", "left", "right", "sample_scope"}, set()
        )
        scope = comparison["sample_scope"]
        if (
            scope == "own"
            or scope not in scopes
            or not {comparison["left"], comparison["right"]} <= set(scopes[scope])
            or comparison["comparison_id"] in names
        ):
            raise ValueError("evaluation_paired_comparison_invalid")
        selected = daily.loc[daily.sample_scope.eq(scope)]
        keys = ["timestamp", "asset_id", "metric"]
        left = selected.loc[selected.candidate_id.eq(comparison["left"])].set_index(
            keys
        )
        right = selected.loc[selected.candidate_id.eq(comparison["right"])].set_index(
            keys
        )
        paired = left.join(
            right[["value"]].rename(columns={"value": "value_right"}), how="inner"
        )
        paired["value"] -= paired.pop("value_right")
        paired["candidate_id"] = comparison["comparison_id"]
        paired["reason"] = np.where(paired.value.notna(), "", "paired_metric_undefined")
        comparisons.append(paired.reset_index())
    if comparisons:
        daily = pd.concat([daily, *comparisons], ignore_index=True)
    inference = daily_inference(
        daily,
        groups=config["inference_groups"],
        statistics=config["statistics"],
        sessions=dates,
    )
    output = Path(config["output"])

    def publish(root):
        _write_table(
            root,
            "daily",
            daily,
            ("timestamp", "asset_id", "candidate_id", "sample_scope", "metric"),
            config,
        )
        _write_table(
            root,
            "inference",
            inference,
            ("asset_id", "candidate_id", "sample_scope", "metric", "family"),
            config,
        )

        _write_table(
            root,
            "summary",
            summary,
            ("asset_id", "candidate_id", "sample_scope", "metric"),
            config,
        )

    atomic_write_directory(output, publish)
    return output

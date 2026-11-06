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
                    asset_id=asset, sample_scope=scope, metric=metric, family=family
                )
            )
    return pd.concat(results, ignore_index=True)


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
        {"comparisons"},
    )
    window = TimeWindow(**config["window"])
    dates = pd.DatetimeIndex(config["sessions"])
    if (
        not len(dates)
        or dates.hasnans
        or not dates.is_unique
        or not dates.is_monotonic_increasing
        or not dates.equals(dates.normalize())
    ):
        raise ValueError("evaluation_sessions_invalid")
    if dates[0] < pd.Timestamp(window.start) or dates[-1] > pd.Timestamp(window.end):
        raise ValueError("evaluation_sessions_outside_window")
    factors = config["factor_columns"]
    candidates = config["predictions"]
    names = [*candidates, *factors]
    if not names or len(set(names)) != len(names):
        raise ValueError("evaluation_candidate_ids_invalid")
    for members in config["comparison_groups"].values():
        if (
            not members
            or len(set(members)) != len(members)
            or set(members) - set(names)
        ):
            raise ValueError("evaluation_comparison_group_invalid")
    session = open_dataset_session(
        **_dataset_input_options(config["inputs"], time_window=window)
    )
    label = config["label"]
    availability = session.spec["label_availability_columns"][label]
    columns = ["timestamp", "asset_id", "eligible", label, availability, *factors]
    frame = materialize_dataset_session(session.projected_view(columns))
    frame = frame.loc[frame.asset_id.isin(config["asset_ids"])].copy()
    if frame.duplicated(["timestamp", "asset_id"]).any():
        raise ValueError("evaluation_input_keys_duplicate")
    mature = label_maturity_mask(
        frame,
        fitting_cutoff=config["cutoff"],
        availability_column=availability,
        target_column=label,
    )
    frame.loc[~(mature & frame.eligible), label] = np.nan
    base = frame.set_index(["timestamp", "asset_id"])
    scores, probability, thresholds, loss_reasons = {}, {}, {}, {}
    for candidate, paths in candidates.items():
        if type(paths) is not list or not paths:
            raise ValueError("evaluation_predictions_required")
        parts = []
        identity = None
        for path in paths:
            part, metadata = read_predictions(Path(path), time_window=window)
            model = metadata["model_facts"]["model"]
            if model["target_column"] != label:
                raise ValueError("evaluation_prediction_target_mismatch")
            current_identity = (model["model_algorithm"], metadata["label_unit"])
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
    valid = score_frame.notna().mul(base[label].notna(), axis=0)
    scopes = {"own": names, **config["comparison_groups"]}
    if "own" in config["comparison_groups"]:
        raise ValueError("evaluation_scope_reserved")
    rows = []
    for scope, members in scopes.items():
        paired = valid[members].all(axis=1)
        for candidate in members:
            mask = valid[candidate] if scope == "own" else paired
            selected = pd.DataFrame(
                {"signal": score_frame[candidate], "label": base[label], "keep": mask}
            ).reset_index()
            selected["date"] = selected.timestamp.dt.normalize()
            blocks = dict(tuple(selected.groupby(["asset_id", "date"], sort=False)))
            for asset in config["asset_ids"]:
                for day in dates:
                    block = blocks.get((asset, day), selected.iloc[:0])
                    usable = block.loc[block.keep]
                    x, y = usable.signal.to_numpy(), usable.label.to_numpy()
                    ic, reason = _pearson_correlation_result(x, y)
                    rank, rank_reason = _pearson_correlation_result(
                        pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy()
                    )
                    metrics = {"ic": (ic, reason), "rank_ic": (rank, rank_reason)}
                    if candidate in thresholds:
                        keys = pd.MultiIndex.from_frame(
                            usable[["timestamp", "asset_id"]]
                        )
                        metrics.update(
                            {
                                name: (value, None if len(y) else "no_observations")
                                for name, value in classification_metrics(
                                    y,
                                    probability[candidate].reindex(keys).to_numpy(),
                                    thresholds[candidate],
                                ).items()
                            }
                        )
                    elif candidate in candidates:
                        metrics["mse"] = (
                            float(np.mean((x - y) ** 2)) if len(y) else None,
                            None if len(y) else "no_observations",
                        )
                    if candidate in loss_reasons:
                        for metric in set(metrics) - {"ic", "rank_ic"}:
                            metrics[metric] = (None, loss_reasons[candidate])
                    for metric, (value, reason) in metrics.items():
                        rows.append(
                            dict(
                                timestamp=day,
                                asset_id=asset,
                                candidate_id=candidate,
                                sample_scope=scope,
                                metric=metric,
                                value=value,
                                reason=reason,
                                observation_count=len(y),
                                input_count=len(block),
                            )
                        )
    daily = pd.DataFrame(rows)
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
        paired = left.join(right[["value"]], rsuffix="_right", how="inner")
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

    atomic_write_directory(output, publish)
    return output

"""Production A-share statistics artifact producer."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from quant_research_platform.contracts.experiments import (
    _model_inference_blocker,
    legal_training_rows,
    training_availability_columns,
)
from quant_research_platform.contracts.json_codec import canonical_json_hash
from quant_research_platform.contracts.time import (
    TimeWindow,
    time_window_timestamp_interval,
)
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.research.features.transforms import (
    model_input_validity_columns,
)
from quant_research_platform.research.protocols.a_share import (
    _inference_blocker,
)
from quant_research_platform.research.randomness import derive_seed
from quant_research_platform.research.statistics import (
    _apply_fdr,
    cross_sectional_rank_ic,
    fama_macbeth_period,
    hac_mean,
    model_confidence_set,
    spa_reality_check,
    stationary_bootstrap_mean_ci,
    stepm_superior_models,
    time_series_inference_summary,
)
from quant_research_platform.runtime.dataframes import iter_complete_session_frames
from quant_research_platform.storage.datasets.reading import read_dataset
from quant_research_platform.storage.datasets.types import DatasetReadMode

_PREDICTION_COLUMN = "__prediction_alpha__"
_MODEL_ROBUSTNESS_TESTS = (
    "spa_reality_check",
    "stepm_superior_models",
    "model_confidence_set",
)
_FAMA_MACBETH_PERIOD_COLUMNS = (
    "regression_id",
    "coefficient_id",
    "timestamp",
    "status",
    "reason",
    "observation_count",
    "value",
    "white_p_value",
    "white_reason",
)


def _statistics_window_bounds(window):
    start, stop, inclusive = time_window_timestamp_interval(window)
    return pd.Timestamp(start), pd.Timestamp(stop) if inclusive else pd.Timestamp(
        stop
    ) - pd.Timedelta(1, "ns")


def build_a_share_factor_statistics_tables(
    *,
    dataset_path,
    dataset_spec,
    config: Mapping[str, object],
    time_window: TimeWindow,
) -> dict[str, pd.DataFrame]:
    """Publish factor inference once, independently of fitted model artifacts."""
    spec = dataset_spec
    evaluation = config["evaluation"]
    primary_label = evaluation["primary_label"]
    availability = _statistics_label_availability(config["evaluation"], dataset_spec)
    factors = tuple(evaluation["factors"])
    statistics = config["statistics"]
    families = _factor_families(statistics["factor_fdr_families"], factors)
    core = tuple(statistics["fama_macbeth_core_factors"])
    if any(factor not in factors for factor in core):
        raise ValueError("statistics_core_factors_missing")
    max_lags = statistics["hac_lag"]
    methods = (
        _fdr_method(statistics["fdr_primary"], expected="BH"),
        _fdr_method(statistics["fdr_robustness"], expected="BY"),
    )
    window = time_window
    columns = tuple(
        dict.fromkeys(
            (
                *spec.primary_key,
                "eligible",
                primary_label,
                availability,
                *factors,
                "total_market_cap",
            )
        )
    )
    ranks, periods, input_dates = [], [], []
    chunks = iter_complete_session_frames(
        read_dataset(
            dataset_path,
            spec=dataset_spec if Path(dataset_path).is_file() else None,
            time_window=window,
            columns=columns,
            mode=DatasetReadMode.CHUNKED,
        ),
        timestamp_column="timestamp",
    )
    with closing_iterator(chunks):
        for session in chunks:
            input_dates.append(pd.Timestamp(session["timestamp"].iloc[0]))
            mature = legal_training_rows(
                session,
                fitting_cutoff=_statistics_window_bounds(window)[1],
                availability_column=availability,
                target_column=primary_label,
            )
            if mature.empty:
                continue
            ranks.extend(
                cross_sectional_rank_ic(
                    mature,
                    value_columns=factors,
                    label_column=primary_label,
                ).to_dict("records")
            )
            periods.extend(
                _fama_macbeth_period_records(
                    mature,
                    return_column=primary_label,
                    factors=factors,
                    core_factors=core,
                )
            )
    period_frame = pd.DataFrame.from_records(
        periods, columns=_FAMA_MACBETH_PERIOD_COLUMNS
    )
    return {
        "factor_diagnostics": _factor_summary(
            pd.DataFrame.from_records(ranks, columns=["timestamp", *factors])
            .set_index("timestamp")
            .reindex(input_dates)
            .rename_axis("timestamp")
            .reset_index(),
            factors=factors,
            families=families,
            methods=methods
            if config["sample_role"] == "validation"
            and _inference_blocker(config) is None
            else (None,),
            max_lags=max_lags,
        ),
        "fama_macbeth": _fama_macbeth_summary(
            period_frame,
            factors=factors,
            core_factors=core,
            max_lags=max_lags,
            input_date_count=len(input_dates),
        ),
        "fama_macbeth_periods": period_frame,
    }


def build_a_share_statistics_tables(
    *,
    dataset_path,
    dataset_spec,
    prediction_id: str,
    model_entries: Mapping,
    selections: Mapping,
    period_windows: Mapping,
    trial_rows: pd.DataFrame,
    config: Mapping[str, object],
    time_window: TimeWindow,
    prediction_lookup: object,
    global_statistics: object | None,
) -> dict[str, pd.DataFrame]:
    dataset_id = str(dataset_spec.data_id)
    selection_group = config["selection_group"]
    inference_blocker = _model_inference_blocker(
        sample_role=config["sample_role"],
        selection_groups=(selection_group,),
    )
    joint_blocker = _inference_blocker(config)
    primary_label = config["evaluation"]["primary_label"]
    availability = _statistics_label_availability(config["evaluation"], dataset_spec)
    statistics = config.get("statistics")
    root_seed = config.get("root_seed")
    if (
        type(primary_label) is not str
        or type(availability) is not str
        or type(statistics) is not dict
        or (type(root_seed) is not int)
    ):
        raise ValueError("statistics_protocol_invalid")
    max_lags = statistics.get("hac_lag")
    block_size = statistics.get("bootstrap_block")
    repetitions = statistics.get("bootstrap_repetitions")
    if (
        type(max_lags) is not int
        or type(block_size) is not int
        or type(repetitions) is not int
    ):
        raise ValueError("statistics_protocol_invalid")
    model_window = time_window
    period_bounds = {
        key: _statistics_window_bounds(window) for key, window in period_windows.items()
    }
    model_start, model_end = _statistics_window_bounds(model_window)
    family_ids = tuple(
        family["family_id"] for family in next(iter(selections.values()))["families"]
    )
    family_columns = {family: index for index, family in enumerate(family_ids)}
    prediction_columns = tuple(family_columns.values())
    benchmark_id = config["comparison_model_id"]
    if benchmark_id not in family_ids:
        raise ValueError("statistics_model_benchmark_missing")
    read_columns = tuple(
        dict.fromkeys(
            (
                *dataset_spec.primary_key,
                "eligible",
                primary_label,
                availability,
                *(
                    column
                    for item in model_entries.values()
                    if item["model_metadata"] is not None
                    for column in model_input_validity_columns(
                        item["model_metadata"]["input_columns"],
                        item["model_metadata"]["model_algorithm"],
                    )
                ),
            )
        )
    )
    rank_records: list[dict[str, object]] = []
    comparison_records: list[dict[str, object]] = []
    loss_records: list[dict[str, object]] = []
    chunks = iter_complete_session_frames(
        read_dataset(
            dataset_path,
            spec=dataset_spec if Path(dataset_path).is_file() else None,
            time_window=model_window,
            columns=read_columns,
            mode=DatasetReadMode.CHUNKED,
        ),
        timestamp_column="timestamp",
    )
    with closing_iterator(chunks):
        for session in chunks:
            mature = legal_training_rows(
                session,
                fitting_cutoff=_statistics_window_bounds(model_window)[1],
                availability_column=availability,
                target_column=primary_label,
            )
            timestamp = pd.Timestamp(session["timestamp"].iloc[0])
            if model_start <= timestamp <= model_end:
                matches = [
                    key
                    for key, (start, end) in period_bounds.items()
                    if start <= timestamp <= end
                ]
                if len(matches) != 1:
                    raise ValueError("statistics_prediction_period_coverage_invalid")
                period_id = matches[0]
                selection = selections[period_id]
                for family_id in family_ids:
                    entry = model_entries[period_id, family_id]
                    metadata = entry["model_metadata"]
                    if metadata is None:
                        mature[family_columns[family_id]] = np.nan
                        continue
                    inputs = model_input_validity_columns(
                        metadata["input_columns"], metadata["model_algorithm"]
                    )
                    eligible = session["eligible"].eq(True) & session.loc[
                        :, list(inputs)
                    ].notna().all(axis=1)
                    keys = session.loc[eligible, ["timestamp", "asset_id"]].copy()
                    keys["family_id"] = family_id
                    values, found = prediction_lookup.values_for_frame(
                        keys, mark_matched=True
                    )
                    if not all(found):
                        raise ValueError("missing_required_prediction_keys")
                    scores = pd.Series(
                        np.asarray(values, dtype="float64"), index=keys.index
                    )
                    mature[family_columns[family_id]] = scores.reindex(mature.index)
                mature[_PREDICTION_COLUMN] = mature[
                    family_columns[selection["selected_family_id"]]
                ]
                rank = cross_sectional_rank_ic(
                    mature,
                    value_columns=(
                        _PREDICTION_COLUMN,
                        *prediction_columns,
                    ),
                    label_column=primary_label,
                )
                if mature.empty:
                    rank = pd.DataFrame(
                        [
                            {
                                "timestamp": timestamp,
                                **dict.fromkeys(
                                    (_PREDICTION_COLUMN, *prediction_columns), np.nan
                                ),
                            }
                        ]
                    )
                rank_records.extend(rank.to_dict("records"))
                daily = rank.iloc[0]
                paired_rank, paired_count = (
                    (rank.loc[:, ["timestamp", *prediction_columns]], 0)
                    if mature.empty
                    else _common_model_rank_ic(
                        mature,
                        family_ids=prediction_columns,
                        label_column=primary_label,
                    )
                )
                comparison_records.extend(paired_rank.to_dict("records"))
                common = bool(
                    np.isfinite(
                        paired_rank.loc[:, list(prediction_columns)].to_numpy(
                            dtype="float64"
                        )
                    ).all()
                )
                for family_id in family_ids:
                    column = family_columns[family_id]
                    value = daily[column]
                    defined = bool(np.isfinite(value))
                    paired_assets = (
                        mature.loc[
                            mature[[column, primary_label]].notna().all(axis=1),
                            "asset_id",
                        ]
                        .astype(str)
                        .sort_values()
                    )
                    loss_records.append(
                        {
                            "timestamp": timestamp,
                            "sample_role": config["sample_role"],
                            "period_id": period_id,
                            "family_id": family_id,
                            "config_id": model_entries[period_id, family_id][
                                "model_id"
                            ],
                            "loss": -float(value) if defined else np.nan,
                            "observation_count": len(paired_assets),
                            "asset_coverage_id": canonical_json_hash(
                                paired_assets.tolist()
                            ),
                            "status": "DEFINED" if defined else "UNDEFINED",
                            "reason": None
                            if defined
                            else "family_training_failed"
                            if model_entries[period_id, family_id]["model_metadata"]
                            is None
                            else "cross_sectional_rankic_undefined",
                            "common_date": int(common),
                            "common_observation_count": paired_count,
                        }
                    )
    if prediction_lookup.coverage_counts() != (0, 0):
        raise ValueError("prediction_dataset_coverage_mismatch")
    rank_by_date = pd.DataFrame.from_records(
        rank_records, columns=["timestamp", _PREDICTION_COLUMN]
    )
    prediction_summary = _prediction_summary(
        rank_by_date,
        prediction_id=prediction_id,
        inference_blocker=inference_blocker,
        max_lags=max_lags,
        block_size=block_size,
        repetitions=repetitions,
        seed=derive_seed(
            root_seed,
            "a_share_statistics_bootstrap",
            dataset_id,
            prediction_id,
        ),
    )
    model_trials = trial_rows.copy()
    model_losses = pd.DataFrame.from_records(
        loss_records,
        columns=(
            "timestamp",
            "sample_role",
            "period_id",
            "family_id",
            "config_id",
            "loss",
            "observation_count",
            "status",
            "reason",
            "common_date",
            "common_observation_count",
            "asset_coverage_id",
        ),
    )
    comparison_by_date = (
        pd.DataFrame.from_records(
            comparison_records, columns=["timestamp", *prediction_columns]
        )
        .set_index("timestamp")
        .rename(columns={column: family for family, column in family_columns.items()})
    )
    comparison, tests, robustness = _model_comparison(
        comparison_by_date,
        family_ids=family_ids,
        benchmark_id=benchmark_id,
        sample_role=config["sample_role"],
        inference_blocker=inference_blocker,
        joint_blocker=joint_blocker,
        max_lags=max_lags,
        block_size=block_size,
        repetitions=repetitions,
        root_seed=root_seed,
        dataset_id=dataset_id,
    )
    group_comparison = _paired_group_rankic(
        model_losses,
        global_statistics=global_statistics,
        selection_group=selection_group,
        family_ids=family_ids,
        config=config,
    )
    tables = {
        "prediction_diagnostics": prediction_summary,
        "model_trials": model_trials,
        "robustness": robustness,
        "model_losses": model_losses,
        "model_comparison": comparison,
        "model_group_comparison": group_comparison,
        "model_reality_checks": tests,
    }
    return tables


def _paired_group_rankic(
    quarterly, *, global_statistics, selection_group, family_ids, config
):
    columns = (
        "family_id",
        "sample_role",
        "status",
        "reason",
        "fdr_method",
        "bootstrap_status",
        "bootstrap_reason",
        "observation_count",
        "calendar_count",
        "fdr_family_size",
        "hac_lag",
        "bootstrap_block_size",
        "bootstrap_repetitions",
        "bootstrap_seed",
        "global_mean",
        "quarterly_mean",
        "mean_difference",
        "standard_error",
        "t_statistic",
        "p_value",
        "ci_lower",
        "ci_upper",
        "fdr_adjusted_p_value",
        "fdr_alpha",
        "confidence",
        "bootstrap_lower",
        "bootstrap_upper",
    )
    if (
        selection_group == "global"
        or config["statistics"].get("group_comparison") is None
    ):
        return pd.DataFrame(columns=columns)
    if global_statistics is None:
        raise ValueError("global_comparison_statistics_required")
    statistics = config["statistics"]
    declaration = statistics["group_comparison"]
    keys = ["timestamp", "family_id"]
    with global_statistics.table("model_losses.parquet") as global_losses:
        left = global_losses.set_index(keys).sort_index()
        right = quarterly.set_index(keys).sort_index()
        if (
            left.index.has_duplicates
            or right.index.has_duplicates
            or left.empty != right.empty
            or (
                not left.empty
                and (
                    not left.index.equals(right.index)
                    or not left.loc[
                        left.status.eq("DEFINED") & right.status.eq("DEFINED"),
                        "asset_coverage_id",
                    ].equals(
                        right.loc[
                            left.status.eq("DEFINED") & right.status.eq("DEFINED"),
                            "asset_coverage_id",
                        ]
                    )
                    or not left.loc[
                        left.status.eq("DEFINED") & right.status.eq("DEFINED"),
                        "observation_count",
                    ].equals(
                        right.loc[
                            left.status.eq("DEFINED") & right.status.eq("DEFINED"),
                            "observation_count",
                        ]
                    )
                    or (not left["sample_role"].equals(right["sample_role"]))
                )
            )
        ):
            raise ValueError("model_group_comparison_coverage_invalid")
        if left.empty:
            differences = pd.DataFrame(
                index=pd.DatetimeIndex([], name="timestamp"),
                columns=family_ids,
                dtype="float64",
            )
            global_values = differences.copy()
            quarterly_values = differences.copy()
        else:
            differences = (left["loss"] - right["loss"]).unstack("family_id")
            global_values = -left["loss"].unstack("family_id")
            quarterly_values = -right["loss"].unstack("family_id")
    if set(differences.columns) != set(family_ids):
        raise ValueError("model_group_comparison_family_incomplete")
    access = config["sample_role"]
    inference_blocker = _model_inference_blocker(
        sample_role=access, selection_groups=("global", "quarterly")
    )
    fdr_method = (
        None if _inference_blocker(config) is not None else declaration["fdr_method"]
    )
    inference = (
        time_series_inference_summary(
            differences.set_axis(range(len(family_ids)), axis=1).reset_index(),
            identity_column="timestamp",
            max_lags=statistics["hac_lag"],
            fdr_method=fdr_method,
            fdr_alpha=declaration["alpha"],
            fdr_family_size=len(family_ids),
        )
        .set_index("timestamp")
        .set_axis(differences.columns)
        if inference_blocker is None and (not differences.empty)
        else None
    )
    rows = []
    for family in family_ids:
        paired = np.isfinite(global_values[family]) & np.isfinite(
            quarterly_values[family]
        )
        values = differences.loc[paired, family].to_numpy(dtype="float64")
        row = dict.fromkeys(columns, np.nan)
        row.update(
            family_id=family,
            sample_role=access,
            observation_count=len(values),
            calendar_count=len(differences),
            global_mean=float(global_values.loc[paired, family].mean()),
            quarterly_mean=float(quarterly_values.loc[paired, family].mean()),
            mean_difference=float(values.mean()) if len(values) else np.nan,
            status="UNDEFINED",
            reason=inference_blocker if len(values) else "no_common_observations",
            fdr_family_size=len(family_ids),
            fdr_method=fdr_method or "NOT_APPLICABLE",
            fdr_alpha=declaration["alpha"],
            hac_lag=statistics["hac_lag"],
            bootstrap_block_size=statistics["bootstrap_block"],
            bootstrap_repetitions=statistics["bootstrap_repetitions"],
            bootstrap_seed=declaration["seed"],
            confidence=declaration["confidence"],
            bootstrap_status="UNDEFINED",
            bootstrap_reason=inference_blocker
            if len(values)
            else "no_common_observations",
        )
        if inference is not None:
            estimate = inference.loc[family]
            for key in (
                "status",
                "reason",
                "standard_error",
                "t_statistic",
                "p_value",
                "ci_lower",
                "ci_upper",
                "fdr_adjusted_p_value",
            ):
                row[key] = estimate[key]
            if (
                estimate["status"] == "DEFINED"
                and len(values) > statistics["bootstrap_block"]
            ):
                interval = stationary_bootstrap_mean_ci(
                    values,
                    block_size=statistics["bootstrap_block"],
                    repetitions=statistics["bootstrap_repetitions"],
                    seed=declaration["seed"],
                    confidence=declaration["confidence"],
                )
                row.update(
                    bootstrap_lower=interval.lower,
                    bootstrap_upper=interval.upper,
                    bootstrap_status="DEFINED",
                    bootstrap_reason=None,
                )
            else:
                row["bootstrap_reason"] = (
                    estimate["reason"]
                    if estimate["status"] != "DEFINED"
                    else "bootstrap_block_exceeds_observations"
                )
        rows.append(row)
    return pd.DataFrame.from_records(rows, columns=columns)


def _fdr_method(value: object, *, expected: str) -> str:
    if type(value) is not str or value != f"{expected}_5pct":
        raise ValueError("statistics_protocol_invalid")
    return expected


def _factor_families(
    value: dict[str, object], factors: tuple[str, ...]
) -> dict[str, tuple[str, ...]]:
    result: dict[str, tuple[str, ...]] = {}
    for family_id, raw_factors in value.items():
        if (
            type(family_id) is not str
            or type(raw_factors) is not list
            or (not raw_factors)
            or any(type(factor) is not str for factor in raw_factors)
            or (len(set(raw_factors)) != len(raw_factors))
        ):
            raise ValueError("statistics_protocol_invalid")
        if any(factor not in factors for factor in raw_factors):
            raise ValueError("statistics_fdr_family_factors_missing")
        result[family_id] = tuple(raw_factors)
    return result


def _factor_summary(
    rank_by_date: pd.DataFrame,
    *,
    factors: tuple[str, ...],
    families: Mapping[str, tuple[str, ...]],
    methods: tuple[str | None, ...],
    max_lags: int,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    assigned: set[str] = set()
    for family_id, planned_factors in families.items():
        present = tuple(factor for factor in planned_factors if factor in factors)
        assigned.update(present)
        family_frame = rank_by_date.reindex(columns=["timestamp", *planned_factors])
        inference = time_series_inference_summary(
            family_frame,
            identity_column="timestamp",
            max_lags=max_lags,
            fdr_method=None,
            fdr_alpha=0.05,
            fdr_family_size=len(planned_factors),
        ).rename(columns={"timestamp": "factor_id"})
        for method in methods:
            part = _apply_fdr(inference.copy(), method, 0.05, len(planned_factors))
            if method is None:
                part["fdr_method"] = "NOT_APPLICABLE"
            part.insert(0, "family_id", family_id)
            defined_count = int(part["status"].eq("DEFINED").sum())
            part["planned_hypothesis_count"] = len(planned_factors)
            part["defined_hypothesis_count"] = defined_count
            part["missing_hypothesis_count"] = len(planned_factors) - defined_count
            parts.append(part)
    diagnostics = tuple(factor for factor in factors if factor not in assigned)
    if diagnostics:
        part = time_series_inference_summary(
            rank_by_date.loc[:, ["timestamp", *diagnostics]],
            identity_column="timestamp",
            max_lags=max_lags,
            fdr_method=None,
            fdr_alpha=0.05,
        ).rename(columns={"timestamp": "factor_id"})
        part.insert(0, "family_id", "diagnostic_only")
        part["fdr_method"] = "NOT_APPLICABLE"
        defined_count = int(part["status"].eq("DEFINED").sum())
        part["planned_hypothesis_count"] = len(diagnostics)
        part["defined_hypothesis_count"] = defined_count
        part["missing_hypothesis_count"] = len(diagnostics) - defined_count
        parts.append(part)
    result = pd.concat(parts, ignore_index=True)
    result["input_date_count"] = len(rank_by_date)
    return result


def _prediction_summary(
    rank_by_date: pd.DataFrame,
    *,
    prediction_id: str,
    inference_blocker: str | None,
    max_lags: int,
    block_size: int,
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    values = (
        rank_by_date[_PREDICTION_COLUMN].dropna()
        if _PREDICTION_COLUMN in rank_by_date
        else pd.Series(dtype="float64")
    )
    count = len(values)
    row = {
        "prediction_id": prediction_id,
        "observation_count": count,
        "status": "UNDEFINED",
        "reason": inference_blocker or "hac_lag_exceeds_observations",
        "mean": float(values.mean()) if count else np.nan,
        "standard_error": np.nan,
        "t_statistic": np.nan,
        "p_value": np.nan,
        "ci_lower": np.nan,
        "ci_upper": np.nan,
        "bootstrap_lower": np.nan,
        "bootstrap_upper": np.nan,
        "bootstrap_block_size": block_size,
        "bootstrap_repetitions": repetitions,
        "bootstrap_seed": seed,
        "input_date_count": len(rank_by_date),
    }
    if inference_blocker is not None or count < max(2, max_lags + 1):
        return pd.DataFrame([row])
    try:
        inference = hac_mean(values, max_lags=max_lags)
        bootstrap = stationary_bootstrap_mean_ci(
            values, block_size=block_size, repetitions=repetitions, seed=seed
        )
    except ValueError as exc:
        if str(exc) not in {"hac_inference_undefined", "bootstrap_inference_undefined"}:
            raise
        row["reason"] = str(exc)
        return pd.DataFrame([row])
    row.update(
        status="DEFINED",
        reason=None,
        mean=inference.mean,
        standard_error=inference.standard_error,
        t_statistic=inference.t_statistic,
        p_value=inference.p_value,
        ci_lower=inference.confidence_interval[0],
        ci_upper=inference.confidence_interval[1],
        bootstrap_lower=bootstrap.lower,
        bootstrap_upper=bootstrap.upper,
    )
    return pd.DataFrame([row])


def _common_model_rank_ic(frame, *, family_ids, label_column):
    """Compare predictions only on the exact intersection of valid OOS rows."""
    columns = [*family_ids, label_column]
    valid = np.isfinite(frame.loc[:, columns].to_numpy(dtype="float64")).all(axis=1)
    shared = frame.loc[valid]
    if shared.empty:
        return (
            pd.DataFrame(
                [
                    {
                        "timestamp": frame["timestamp"].iloc[0],
                        **dict.fromkeys(family_ids, np.nan),
                    }
                ]
            ),
            0,
        )
    return (
        cross_sectional_rank_ic(
            shared, value_columns=family_ids, label_column=label_column
        ),
        len(shared),
    )


def _model_comparison(
    rank_by_date,
    *,
    family_ids,
    benchmark_id,
    sample_role,
    inference_blocker,
    max_lags,
    block_size,
    repetitions,
    root_seed,
    dataset_id,
    joint_blocker=None,
):
    common = rank_by_date.loc[:, list(family_ids)].dropna()
    losses = -common
    challengers = tuple(family for family in family_ids if family != benchmark_id)
    comparisons = []
    for family in challengers:
        difference = common[family] - common[benchmark_id]
        summary = _prediction_summary(
            pd.DataFrame({_PREDICTION_COLUMN: difference}),
            prediction_id=family,
            inference_blocker=inference_blocker,
            max_lags=max_lags,
            block_size=block_size,
            repetitions=repetitions,
            seed=derive_seed(
                root_seed,
                "a_share_family_comparison",
                dataset_id,
                sample_role,
                family,
            ),
        )
        summary.insert(0, "sample_role", sample_role)
        summary.insert(1, "benchmark_id", benchmark_id)
        comparisons.append(summary)
    test_rows = []
    robustness_rows = []
    joint_blocker = joint_blocker or inference_blocker
    for test_id in _MODEL_ROBUSTNESS_TESTS:
        seed = derive_seed(
            root_seed, "a_share_model_reality_check", dataset_id, sample_role, test_id
        )
        result = None
        if (
            joint_blocker is None
            and sample_role == "validation"
            and challengers
            and (len(common) > block_size)
        ):
            options = dict(block_size=block_size, repetitions=repetitions, seed=seed)
            if test_id == "spa_reality_check":
                result = spa_reality_check(
                    losses[benchmark_id], losses.loc[:, list(challengers)], **options
                )
            elif test_id == "stepm_superior_models":
                result = stepm_superior_models(
                    losses[benchmark_id],
                    losses.loc[:, list(challengers)],
                    alpha=0.05,
                    **options,
                )
            else:
                result = model_confidence_set(losses, alpha=0.05, **options)
        status = result.status if result is not None else "UNDEFINED"
        reason = (
            result.reason
            if result is not None
            else joint_blocker
            if joint_blocker is not None
            else "test_excluded"
            if sample_role != "validation"
            else "no_comparison_models"
            if not challengers
            else "bootstrap_block_exceeds_common_observations"
        )
        robustness_rows.append(
            dict(
                test_id=test_id,
                status=status,
                reason=reason,
                observation_count=len(common),
            )
        )
        if test_id == "spa_reality_check":
            values = [
                (
                    "joint",
                    name,
                    getattr(result, name) if status == "DEFINED" else np.nan,
                )
                for name in ("lower_p_value", "consistent_p_value", "upper_p_value")
            ]
        elif test_id == "stepm_superior_models":
            values = [
                (
                    family,
                    "superior",
                    float(family in result.superior_models)
                    if status == "DEFINED"
                    else np.nan,
                )
                for family in challengers
            ]
        else:
            values = [
                (
                    family,
                    name,
                    (
                        float(result.p_values.loc[family].iloc[0])
                        if name == "p_value"
                        else float(family in result.included)
                    )
                    if status == "DEFINED"
                    else np.nan,
                )
                for family in family_ids
                for name in ("p_value", "included")
            ]
        test_rows.extend(
            (
                dict(
                    test_id=test_id,
                    sample_role=sample_role,
                    family_id=family,
                    statistic_id=name,
                    value=value,
                    status=status,
                    reason=reason,
                    observation_count=len(common),
                    block_size=block_size,
                    repetitions=repetitions,
                    alpha=0.05,
                    seed=seed,
                )
                for family, name, value in values
            )
        )
    if comparisons:
        comparison_table = pd.concat(comparisons, ignore_index=True)
    else:
        comparison_table = _prediction_summary(
            pd.DataFrame(),
            prediction_id=benchmark_id,
            inference_blocker=inference_blocker,
            max_lags=max_lags,
            block_size=block_size,
            repetitions=repetitions,
            seed=root_seed,
        ).iloc[:0]
        comparison_table.insert(0, "sample_role", pd.Series(dtype="str"))
        comparison_table.insert(1, "benchmark_id", pd.Series(dtype="str"))
    tables = (comparison_table, pd.DataFrame(test_rows), pd.DataFrame(robustness_rows))
    for table in tables:
        table["input_date_count"] = len(rank_by_date)
    return tables


def _fama_macbeth_definitions(factors, core_factors):
    definitions = tuple((f"single:{factor}", (factor,)) for factor in factors)
    if core_factors:
        definitions += (("core_multifactor", core_factors),)
    return tuple(
        (f"{weighting}:{regression}", columns, weight_column)
        for weighting, weight_column in (
            ("equal", None),
            ("total_market_cap", "total_market_cap"),
        )
        for regression, columns in definitions
    )


def _fama_macbeth_period_records(
    frame: pd.DataFrame,
    *,
    return_column: str,
    factors: tuple[str, ...],
    core_factors: tuple[str, ...],
) -> list[dict[str, object]]:
    timestamp = pd.Timestamp(frame["timestamp"].iloc[0])
    rows: list[dict[str, object]] = []
    for regression_id, regression_factors, weight_column in _fama_macbeth_definitions(
        factors, core_factors
    ):
        coefficient_ids = ("intercept", *regression_factors)
        complete_count = len(
            frame.loc[
                :,
                [
                    return_column,
                    *regression_factors,
                    *(() if weight_column is None else (weight_column,)),
                ],
            ].dropna()
        )
        try:
            result = fama_macbeth_period(
                frame,
                return_column=return_column,
                factor_columns=regression_factors,
                weight_column=weight_column,
            )
        except ValueError as exc:
            if str(exc) != "fama_macbeth_period_insufficient":
                raise
            rows.extend(
                {
                    "regression_id": regression_id,
                    "coefficient_id": coefficient_id,
                    "timestamp": timestamp,
                    "status": "UNDEFINED",
                    "reason": "insufficient_complete_cross_section",
                    "observation_count": complete_count,
                    "value": np.nan,
                    "white_p_value": np.nan,
                    "white_reason": "white_test_undefined",
                }
                for coefficient_id in coefficient_ids
            )
            continue
        coefficients = result.coefficients.rename(index={"const": "intercept"})
        rows.extend(
            {
                "regression_id": regression_id,
                "coefficient_id": coefficient_id,
                "timestamp": timestamp,
                "status": "DEFINED",
                "reason": None,
                "observation_count": result.observation_count,
                "value": float(coefficients[coefficient_id]),
                "white_p_value": result.white_p_value
                if result.white_p_value is not None
                else np.nan,
                "white_reason": result.white_reason,
            }
            for coefficient_id in coefficient_ids
        )
    return rows


def _fama_macbeth_summary(
    periods: pd.DataFrame,
    *,
    factors: tuple[str, ...],
    core_factors: tuple[str, ...],
    max_lags: int,
    input_date_count: int | None = None,
) -> pd.DataFrame:
    if input_date_count is None:
        input_date_count = periods["timestamp"].nunique()
    rows: list[dict[str, object]] = []
    definitions = [
        (regression, coefficient)
        for regression, columns, _weight in _fama_macbeth_definitions(
            factors, core_factors
        )
        for coefficient in ("intercept", *columns)
    ]
    groups = dict(
        tuple(periods.groupby(["regression_id", "coefficient_id"], sort=False))
    )
    empty = periods.iloc[:0]
    for regression_id, coefficient_id in definitions:
        group = groups.get((regression_id, coefficient_id), empty)
        defined = group.loc[group["status"].eq("DEFINED")]
        values = defined["value"].dropna()
        count = len(values)
        white_defined = int(defined["white_p_value"].notna().sum())
        white_undefined = int(group["white_p_value"].isna().sum())
        row = {
            "regression_id": regression_id,
            "coefficient_id": coefficient_id,
            "period_count": count,
            "status": "UNDEFINED",
            "reason": "hac_lag_exceeds_observations",
            "mean": np.nan,
            "standard_error": np.nan,
            "t_statistic": np.nan,
            "p_value": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "white_defined_period_count": white_defined,
            "white_undefined_period_count": white_undefined,
            "input_date_count": input_date_count,
        }
        if count >= max(2, max_lags + 1):
            try:
                inference = hac_mean(values, max_lags=max_lags)
            except ValueError as exc:
                if str(exc) != "hac_inference_undefined":
                    raise
                row["reason"] = "hac_inference_undefined"
                rows.append(row)
                continue
            row.update(
                status="DEFINED",
                reason=None,
                mean=inference.mean,
                standard_error=inference.standard_error,
                t_statistic=inference.t_statistic,
                p_value=inference.p_value,
                ci_lower=inference.confidence_interval[0],
                ci_upper=inference.confidence_interval[1],
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _statistics_input_spec(config):
    from quant_research_platform.contracts.data import DataSpec
    from quant_research_platform.storage.manifests import read_qrp_json_file

    path = Path(config["dataset"])
    if path.is_file():
        payload = config.get("input_spec")
        if not isinstance(payload, dict):
            raise ValueError("parquet_semantic_spec_required")
        values = dict(payload)
    else:
        if "input_spec" in config:
            raise ValueError("dataset_spec_already_declared")
        _, payload = read_qrp_json_file(path / "spec.json")
        values = {
            key: payload[key]
            for key in (
                "data_id",
                "frequency",
                "grain",
                "column_roles",
                "primary_key",
                "partition_columns",
            )
        }
    for key in ("primary_key", "partition_columns"):
        values[key] = tuple(values[key])
    return DataSpec(**values)


def _statistics_label_availability(evaluation, spec):
    label = evaluation["primary_label"]
    availability = evaluation.get("label_availability_columns", {})
    if not isinstance(availability, dict) or set(availability) - {label}:
        raise ValueError("statistics_label_availability_invalid")
    column = availability.get(label, training_availability_columns(label)[0])
    if not isinstance(column, str) or not column or column not in spec.column_roles:
        raise ValueError("statistics_label_availability_invalid")
    if str(spec.column_roles.get(label)) != "label":
        raise ValueError("statistics_label_role_invalid")
    if any(
        str(spec.column_roles.get(factor)) != "factor"
        for factor in evaluation.get("factors", ())
    ):
        raise ValueError("statistics_factor_role_invalid")
    return column


def factor_statistics(config):
    """Calculate and publish factor inference from one explicitly selected dataset."""
    from pathlib import Path

    from quant_research_platform.artifacts.statistics import write_statistics_artifact
    from quant_research_platform.contracts.json_codec import qrp_json_mapping_snapshot
    from quant_research_platform.storage.atomic import reject_if_exists

    config = qrp_json_mapping_snapshot(config)
    required = {
        "dataset",
        "output",
        "window",
        "sample_role",
        "evaluation",
        "statistics",
    }
    optional = {
        "input_spec",
        "selection_is_exploratory",
        "expectations",
        "feature_recipe",
        "training_weighting",
    }
    if set(config) - required - optional or required - set(config):
        raise ValueError("factor_statistics_config_invalid")
    if config["sample_role"] not in {"validation", "test"}:
        raise ValueError("statistics_sample_role_invalid")
    evaluation, policy = config["evaluation"], config["statistics"]
    if (
        not isinstance(evaluation, dict)
        or set(evaluation) - {"factors", "primary_label", "label_availability_columns"}
        or {"factors", "primary_label"} - set(evaluation)
    ):
        raise ValueError("factor_statistics_evaluation_invalid")
    if not isinstance(policy, dict) or set(policy) != {
        "factor_fdr_families",
        "fama_macbeth_core_factors",
        "hac_lag",
        "fdr_primary",
        "fdr_robustness",
    }:
        raise ValueError("factor_statistics_policy_invalid")
    output = Path(config["output"])
    reject_if_exists(output)
    window = TimeWindow(**config["window"])
    spec = _statistics_input_spec(config)
    tables = build_a_share_factor_statistics_tables(
        dataset_path=Path(config["dataset"]),
        dataset_spec=spec,
        config=config,
        time_window=window,
    )
    return write_statistics_artifact(
        output,
        metadata={key: value for key, value in config.items() if key != "output"},
        tables=tables,
        table_group="factor",
    )


def _selection_trial_rows(selections, selection_group):
    rows = []
    for decision_id, selection in selections.items():
        for family in selection["families"]:
            for metrics in family["trial_metrics"]:
                for metric_id, value in metrics.items():
                    if (
                        not metric_id.startswith("validation_")
                        or metric_id.endswith("_reason")
                        or metric_id == "validation_rows"
                    ):
                        continue
                    if value is not None and (
                        type(value) not in {int, float} or not np.isfinite(value)
                    ):
                        raise ValueError("statistics_trial_metric_invalid")
                    if "rankic" in metric_id:
                        count = metrics["validation_valid_rankic_count"]
                    elif metric_id == "validation_mse":
                        count = metrics["validation_rows"]
                    else:
                        count = metrics["validation_valid_ic_count"]
                    reason = metrics.get(metric_id + "_reason")
                    if metric_id == "validation_icir":
                        reason = metrics.get("validation_icir_reason")
                    if metric_id == "validation_rankicir":
                        reason = metrics.get("validation_rankicir_reason")
                    rows.append(
                        {
                            "selection_group": selection_group,
                            "decision_id": decision_id,
                            "family_id": family["family_id"],
                            "config_id": metrics["model_id"],
                            "metric_id": metric_id,
                            "value": value,
                            "observation_count": count,
                            "status": "DEFINED" if value is not None else "UNDEFINED",
                            "reason": ""
                            if value is not None
                            else reason or "metric_undefined",
                            "selection_status": "CHAMPION"
                            if metrics["model_id"] == selection["selected_model_id"]
                            else "FAMILY_WINNER"
                            if metrics["model_id"] == family["selected_config_id"]
                            else "TRIAL",
                        }
                    )
    if not rows:
        raise ValueError("statistics_model_trials_empty")
    return pd.DataFrame(rows)


def model_statistics(config):
    """Compare explicit saved predictions over caller-selected disjoint periods."""
    from pathlib import Path

    import pyarrow as pa

    from quant_research_platform.artifacts.outputs.bundle_validation import (
        validate_output_bundle,
    )
    from quant_research_platform.artifacts.outputs.prediction import (
        _PredictionLookup,
        iter_predictions,
    )
    from quant_research_platform.artifacts.statistics import (
        read_statistics_artifact,
        write_statistics_artifact,
    )
    from quant_research_platform.contracts.json_codec import qrp_json_mapping_snapshot
    from quant_research_platform.research.models.trial_selection import read_selection
    from quant_research_platform.storage.atomic import reject_if_exists
    from quant_research_platform.storage.exact_keys import ExactPrimaryKeyIndex

    config = qrp_json_mapping_snapshot(config)
    required = {
        "dataset",
        "output",
        "window",
        "sample_role",
        "evaluation",
        "statistics",
        "root_seed",
        "selection_group",
        "comparison_model_id",
        "periods",
        "prediction_id",
    }
    optional = {
        "input_spec",
        "global_statistics",
        "selection_is_exploratory",
        "expectations",
        "feature_recipe",
        "training_weighting",
    }
    if required - set(config) or set(config) - required - optional:
        raise ValueError("model_statistics_config_invalid")
    if config["sample_role"] not in {"validation", "test"}:
        raise ValueError("statistics_sample_role_invalid")
    evaluation, policy = config["evaluation"], config["statistics"]
    if (
        not isinstance(evaluation, dict)
        or set(evaluation) - {"primary_label", "label_availability_columns"}
        or "primary_label" not in evaluation
    ):
        raise ValueError("model_statistics_evaluation_invalid")
    if (
        not isinstance(policy, dict)
        or set(policy)
        - {"hac_lag", "bootstrap_block", "bootstrap_repetitions", "group_comparison"}
        or {"hac_lag", "bootstrap_block", "bootstrap_repetitions"} - set(policy)
    ):
        raise ValueError("model_statistics_policy_invalid")
    paired = "group_comparison" in policy
    if (
        paired != ("global_statistics" in config)
        or paired
        and config["selection_group"] != "quarterly"
    ):
        raise ValueError("model_statistics_comparison_inputs_invalid")
    if paired and (
        not isinstance(policy["group_comparison"], dict)
        or set(policy["group_comparison"])
        != {"fdr_method", "alpha", "seed", "confidence"}
    ):
        raise ValueError("model_statistics_comparison_policy_invalid")
    output = Path(config["output"])
    reject_if_exists(output)
    window = TimeWindow(**config["window"])
    if not isinstance(config["periods"], list) or not config["periods"]:
        raise ValueError("statistics_prediction_periods_required")
    periods, entries, selections = {}, {}, {}
    family_ids = None
    previous_end = None
    index = ExactPrimaryKeyIndex(primary_key=("timestamp", "asset_id", "family_id"))
    lookup = _PredictionLookup(
        index=index,
        primary_key=("timestamp", "asset_id", "family_id"),
        prediction_column="prediction_score",
    )
    try:
        for item in config["periods"]:
            if not isinstance(item, dict) or set(item) != {
                "period_id",
                "window",
                "selection",
                "predictions",
            }:
                raise ValueError("statistics_prediction_period_invalid")
            period_id = item["period_id"]
            if not isinstance(period_id, str) or not period_id or period_id in periods:
                raise ValueError("statistics_prediction_period_id_invalid")
            period = TimeWindow(**item["window"])
            start, end = _statistics_window_bounds(period)
            requested_start, requested_end = _statistics_window_bounds(window)
            if (
                start < requested_start
                or end > requested_end
                or previous_end is not None
                and start <= previous_end
            ):
                raise ValueError("statistics_prediction_period_window_invalid")
            previous_end = end
            periods[period_id] = period
            selection = read_selection(Path(item["selection"]))
            if selection["selection_group"] != config["selection_group"]:
                raise ValueError("statistics_selection_group_mismatch")
            if config["selection_group"] == "quarterly" and pd.Timestamp(
                selection["cutoff"]
            ) >= pd.Timestamp(period.start):
                raise ValueError("statistics_selection_after_prediction_start")
            if config["sample_role"] == "test" and (
                start
                < _statistics_window_bounds(TimeWindow(**selection["test_window"]))[0]
                or end
                > _statistics_window_bounds(TimeWindow(**selection["test_window"]))[1]
            ):
                raise ValueError("statistics_test_window_mismatch")
            selections[period_id] = selection
            wanted = {
                family["family_id"]: family["selected_config_id"]
                for family in selection["families"]
            }
            if family_ids is None:
                family_ids = tuple(wanted)
            elif tuple(wanted) != family_ids:
                raise ValueError("statistics_candidate_family_coverage_mismatch")
            for family, winner in wanted.items():
                if winner is None:
                    entries[period_id, family] = {
                        "model_metadata": None,
                        "model_id": "",
                    }
            supplied = set()
            for path in item["predictions"]:
                metadata = validate_output_bundle(
                    Path(path), expected_definition_id="prediction_outputs"
                )[0]["metadata"]
                family = metadata["family_id"]
                if (
                    family in supplied
                    or wanted.get(family) != metadata["model_id"]
                    or metadata["sample_role"] != config["sample_role"]
                ):
                    raise ValueError("statistics_prediction_family_mismatch")
                candidate = next(
                    entry
                    for entry in selection["candidates"]
                    if entry["model_id"] == metadata["model_id"]
                )
                if config["sample_role"] == "test":
                    if (
                        not metadata.get("selection")
                        or read_selection(Path(metadata["selection"])) != selection
                    ):
                        raise ValueError("statistics_test_selection_mismatch")
                elif Path(metadata["model"]) not in {
                    Path(path) for path in candidate["model_paths"]
                }:
                    raise ValueError("statistics_selected_training_model_mismatch")
                supplied.add(family)
                model_metadata = metadata["model_metadata"]
                if model_metadata["target_column"] != selection["target_column"]:
                    raise ValueError("statistics_prediction_target_mismatch")
                entries[period_id, family] = {
                    "model_metadata": model_metadata,
                    "model_id": metadata["model_id"],
                }
                with closing_iterator(
                    iter_predictions(
                        Path(path),
                        columns=("timestamp", "asset_id", "prediction_score"),
                        time_window=period,
                    )
                ) as frames:
                    for frame in frames:
                        frame = frame.copy()
                        frame["family_id"] = family
                        lookup.row_count += index.add_float_value_table(
                            pa.Table.from_pandas(frame, preserve_index=False),
                            namespace="prediction",
                            value_column="prediction_score",
                            validation_column="prediction_score",
                            retain_values=True,
                        )
            if supplied != {
                family for family, winner in wanted.items() if winner is not None
            }:
                raise ValueError("statistics_prediction_family_coverage_missing")
        spec = _statistics_input_spec(config)
        tables = build_a_share_statistics_tables(
            dataset_path=Path(config["dataset"]),
            dataset_spec=spec,
            prediction_id=config["prediction_id"],
            model_entries=entries,
            selections=selections,
            period_windows=periods,
            trial_rows=_selection_trial_rows(selections, config["selection_group"]),
            config=config,
            time_window=window,
            prediction_lookup=lookup,
            global_statistics=read_statistics_artifact(
                Path(config["global_statistics"])
            )
            if "global_statistics" in config
            else None,
        )
    finally:
        lookup.close()
    return write_statistics_artifact(
        output,
        metadata={key: value for key, value in config.items() if key != "output"},
        tables=tables,
    )

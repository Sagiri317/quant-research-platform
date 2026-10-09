"""Production A-share statistics artifact producer."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from quant_research_platform.catalog.datasets import ResearchDatasetCatalog
from quant_research_platform.contracts.experiments import (
    _model_inference_blocker,
    legal_training_rows,
)
from quant_research_platform.contracts.json_codec import canonical_json_hash
from quant_research_platform.contracts.time import TimeWindow, a_share_analysis_windows
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.research.datasets.io import iter_research_dataset_chunks
from quant_research_platform.research.features.transforms import (
    model_input_validity_columns,
)
from quant_research_platform.research.protocols.a_share import (
    _inference_blocker,
    label_catalog,
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
from quant_research_platform.storage.layout import StorageLayout

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


def build_a_share_factor_statistics_tables(
    *,
    layout: StorageLayout,
    dataset_catalog: ResearchDatasetCatalog,
    dataset_id: str,
    protocol_manifest: Mapping[str, object],
    time_window: TimeWindow,
) -> dict[str, pd.DataFrame]:
    """Publish factor inference once, independently of fitted model artifacts."""
    spec = dataset_catalog.get(dataset_id)
    evaluation = protocol_manifest["research_configuration"]["evaluation"]
    primary_label = evaluation["primary_label"]
    availability = label_catalog().get(primary_label).available_at_column
    factors = tuple(evaluation["factors"])
    statistics = protocol_manifest["statistics"]
    families = _factor_families(statistics["factor_fdr_families"], factors)
    core = tuple(statistics["fama_macbeth_core_factors"])
    if any(factor not in factors for factor in core):
        raise ValueError("statistics_core_factors_missing")
    max_lags = statistics["hac_lag"]
    methods = (
        _fdr_method(statistics["fdr_primary"], expected="BH"),
        _fdr_method(statistics["fdr_robustness"], expected="BY"),
    )
    window, _ = a_share_analysis_windows(protocol_manifest, time_window)
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
    ranks, periods = [], []
    chunks = iter_complete_session_frames(
        iter_research_dataset_chunks(
            layout,
            market_id=spec.market_id,
            dataset_id=dataset_id,
            time_window=window,
            columns=columns,
            access_window=protocol_manifest["access_window"],
        ),
        timestamp_column="timestamp",
    )
    with closing_iterator(chunks):
        for session in chunks:
            mature = legal_training_rows(
                session,
                fitting_cutoff=window.end,
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
            pd.DataFrame.from_records(ranks, columns=["timestamp", *factors]),
            factors=factors,
            families=families,
            methods=methods
            if protocol_manifest["access_window"] == "development"
            and _inference_blocker(protocol_manifest) is None
            else (None,),
            max_lags=max_lags,
        ),
        "fama_macbeth": _fama_macbeth_summary(
            period_frame,
            factors=factors,
            core_factors=core,
            max_lags=max_lags,
        ),
        "fama_macbeth_periods": period_frame,
    }


def build_a_share_statistics_tables(
    *,
    layout: StorageLayout,
    dataset_catalog: ResearchDatasetCatalog,
    dataset_id: str,
    prediction_id: str,
    persisted_model: object,
    protocol_manifest: Mapping[str, object],
    time_window: TimeWindow,
    prediction_lookup: object,
    global_statistics: object | None,
) -> dict[str, pd.DataFrame]:
    dataset_spec = dataset_catalog.get(dataset_id)
    selection_group = protocol_manifest["selection_group"]
    inference_blocker = _model_inference_blocker(
        access_window=protocol_manifest["access_window"],
        selection_groups=(selection_group,),
    )
    joint_blocker = _inference_blocker(protocol_manifest)
    primary_label = protocol_manifest["research_configuration"]["evaluation"][
        "primary_label"
    ]
    availability = label_catalog().get(primary_label).available_at_column
    statistics = protocol_manifest.get("statistics")
    root_seed = protocol_manifest.get("root_seed")
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
    _factor_window, model_window = a_share_analysis_windows(
        protocol_manifest, time_window
    )
    model_entries = {
        (item["quarter_id"], item["family_id"]): item
        for item in persisted_model.prediction_models(
            model_window,
            selection_group=selection_group,
            access_window=protocol_manifest["access_window"],
        )
    }
    family_ids = tuple(
        dict.fromkeys(item["family_id"] for item in model_entries.values())
    )
    family_columns = {family: index for index, family in enumerate(family_ids)}
    prediction_columns = tuple(family_columns.values())
    benchmark_id = protocol_manifest["research_configuration"]["comparison_model_id"]
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
        iter_research_dataset_chunks(
            layout,
            market_id=dataset_spec.market_id,
            dataset_id=dataset_id,
            time_window=model_window,
            columns=read_columns,
            access_window=protocol_manifest["access_window"],
        ),
        timestamp_column="timestamp",
    )
    with closing_iterator(chunks):
        for session in chunks:
            mature = legal_training_rows(
                session,
                fitting_cutoff=model_window.end,
                availability_column=availability,
                target_column=primary_label,
            )
            timestamp = pd.Timestamp(session["timestamp"].iloc[0])
            if (
                pd.Timestamp(model_window.start)
                <= timestamp
                <= pd.Timestamp(model_window.end)
            ):
                quarter_id = str(timestamp.to_period("Q"))
                selection = persisted_model.selection(
                    selection_group,
                    quarter_id,
                    access_window=protocol_manifest["access_window"],
                )
                for family_id in family_ids:
                    entry = model_entries[quarter_id, family_id]
                    metadata = entry["model_metadata"]
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
                if mature.empty:
                    continue
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
                rank_records.extend(rank.to_dict("records"))
                daily = rank.iloc[0]
                paired_rank, paired_count = _common_model_rank_ic(
                    mature,
                    family_ids=prediction_columns,
                    label_column=primary_label,
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
                            "access_window": protocol_manifest["access_window"],
                            "quarter_id": quarter_id,
                            "family_id": family_id,
                            "config_id": model_entries[quarter_id, family_id][
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
            protocol_manifest.get("scenario_id"),
            prediction_id,
        ),
    )
    model_trials = _model_trials(persisted_model, selection_group)
    model_losses = pd.DataFrame.from_records(
        loss_records,
        columns=(
            "timestamp",
            "access_window",
            "quarter_id",
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
        access_window=protocol_manifest["access_window"],
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
        protocol_manifest=protocol_manifest,
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
    quarterly, *, global_statistics, selection_group, family_ids, protocol_manifest
):
    columns = (
        "family_id",
        "access_window",
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
    if selection_group == "global":
        return pd.DataFrame(columns=columns)
    statistics = protocol_manifest["statistics"]
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
                    or not left["asset_coverage_id"].equals(right["asset_coverage_id"])
                    or (
                        not left["observation_count"].equals(right["observation_count"])
                    )
                    or (not left["access_window"].equals(right["access_window"]))
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
    access = protocol_manifest["access_window"]
    inference_blocker = _model_inference_blocker(
        access_window=access, selection_groups=("global", "quarterly")
    )
    fdr_method = (
        None
        if _inference_blocker(protocol_manifest) is not None
        else declaration["fdr_method"]
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
        values = differences[family].dropna().to_numpy(dtype="float64")
        row = dict.fromkeys(columns, np.nan)
        row.update(
            family_id=family,
            access_window=access,
            observation_count=len(values),
            calendar_count=len(differences),
            global_mean=float(global_values[family].mean()),
            quarterly_mean=float(quarterly_values[family].mean()),
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
    return pd.concat(parts, ignore_index=True)


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


def _model_trials(persisted_model, selection_group) -> pd.DataFrame:
    rows = []
    for decision_id, selection in persisted_model.selections[selection_group].items():
        for family in selection["families"]:
            family_id = family["family_id"]
            for metrics in family["trial_metrics"]:
                config_id = metrics["model_id"]
                for metric_id, reason_key in (
                    ("validation_ic", "validation_ic_reason"),
                    ("validation_rankic", "validation_rankic_reason"),
                    ("validation_mse", None),
                    *(
                        (f"validation_{suffix}", f"validation_{prefix}ir_reason")
                        for prefix in ("ic", "rankic")
                        for suffix in (
                            f"std_{prefix}",
                            f"{prefix}ir",
                            f"positive_{prefix}_rate",
                            f"valid_{prefix}_count",
                        )
                    ),
                ):
                    value = metrics[metric_id]
                    defined = type(value) in {int, float} and np.isfinite(value)
                    rows.append(
                        {
                            "selection_group": selection_group,
                            "decision_id": decision_id,
                            "family_id": family_id,
                            "config_id": config_id,
                            "metric_id": metric_id,
                            "value": float(value) if defined else np.nan,
                            "observation_count": (
                                metrics["validation_rows"]
                                if metric_id == "validation_mse"
                                else metrics["validation_valid_rankic_count"]
                                if "rankic" in metric_id
                                else metrics["validation_valid_ic_count"]
                            ),
                            "status": "DEFINED" if defined else "UNDEFINED",
                            "reason": None if defined else metrics[reason_key],
                            "selection_status": "CHAMPION"
                            if family_id == selection["selected_family_id"]
                            and config_id == family["selected_config_id"]
                            else "FAMILY_WINNER"
                            if config_id == family["selected_config_id"]
                            else "TRIAL",
                        }
                    )
    if not rows:
        raise ValueError("statistics_model_trials_empty")
    return pd.DataFrame.from_records(rows)


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
    access_window,
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
                access_window,
                family,
            ),
        )
        summary.insert(0, "access_window", access_window)
        summary.insert(1, "benchmark_id", benchmark_id)
        comparisons.append(summary)
    test_rows = []
    robustness_rows = []
    joint_blocker = joint_blocker or inference_blocker
    for test_id in _MODEL_ROBUSTNESS_TESTS:
        seed = derive_seed(
            root_seed, "a_share_model_reality_check", dataset_id, access_window, test_id
        )
        result = None
        if (
            joint_blocker is None
            and access_window == "development"
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
            else "holdout_excluded"
            if access_window != "development"
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
                    access_window=access_window,
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
        comparison_table.insert(0, "access_window", pd.Series(dtype="str"))
        comparison_table.insert(1, "benchmark_id", pd.Series(dtype="str"))
    return (comparison_table, pd.DataFrame(test_rows), pd.DataFrame(robustness_rows))


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
) -> pd.DataFrame:
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

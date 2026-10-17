"""Production A-share statistics artifact producer."""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path
from statistics import mean

import numpy as np
import pandas as pd

from quant_research_platform.contracts.experiments import (
    DEFAULT_ROOT_SEED,
    legal_training_rows,
    training_availability_columns,
)
from quant_research_platform.contracts.ids import validate_input_market
from quant_research_platform.contracts.json_codec import canonical_json_hash
from quant_research_platform.contracts.models import _prediction_loss_reason
from quant_research_platform.contracts.time import (
    TimeWindow,
    time_window_timestamp_interval,
)
from quant_research_platform.contracts.validation import strict_real_numeric_series
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.research.features.transforms import (
    feature_observation_columns,
    mask_unobserved_features,
)
from quant_research_platform.research.models.rank_ensemble import _label_predictions
from quant_research_platform.research.models.trial_selection import _MeanSquaredError
from quant_research_platform.research.protocols.a_share import (
    _inference_blocker,
)
from quant_research_platform.research.randomness import derive_seed
from quant_research_platform.research.statistics import (
    _apply_fdr,
    _selected_analyses,
    _validate_bootstrap_configuration,
    cross_sectional_rank_ic,
    fama_macbeth_period,
    hac_mean,
    model_confidence_set,
    spa_reality_check,
    stationary_bootstrap_mean_ci,
    stepm_superior_models,
    time_series_inference_summary,
)
from quant_research_platform.runtime.dataframes import (
    iter_complete_session_frames,
    iter_complete_time_frames,
)
from quant_research_platform.storage.datasets.reading import (
    _dataset_input_options,
    iter_dataset_chunks,
    open_dataset_session,
    read_dataset,
)
from quant_research_platform.storage.datasets.schema import data_spec_payload
from quant_research_platform.storage.datasets.types import DatasetReadMode

_PREDICTION_COLUMN = "__prediction_alpha__"
_MODEL_ROBUSTNESS_TESTS = (
    "spa_reality_check",
    "stepm_superior_models",
    "model_confidence_set",
)
_MODEL_ANALYSES = (
    "model_losses",
    "prediction_hac",
    "prediction_bootstrap",
    "model_comparison",
    *_MODEL_ROBUSTNESS_TESTS,
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
    dataset_session=None,
) -> dict[str, pd.DataFrame]:
    """Publish factor inference once, independently of fitted model artifacts."""
    spec = dataset_spec
    evaluation = config["evaluation"]
    primary_label = evaluation["primary_label"]
    availability = _statistics_label_availability(config["evaluation"], dataset_spec)
    factors = tuple(evaluation["factors"])
    analyses = _selected_analyses(config, ("rank_ic", "fama_macbeth"))
    statistics = config["statistics"]
    families = (
        _factor_families(statistics["factor_fdr_families"], factors)
        if "rank_ic" in analyses
        else {}
    )
    core = (
        tuple(statistics["fama_macbeth_core_factors"])
        if "fama_macbeth" in analyses
        else ()
    )
    if any(factor not in factors for factor in core):
        raise ValueError("statistics_core_factors_missing")
    max_lags = statistics["hac_lag"]
    methods = (
        (
            _fdr_method(statistics["fdr_primary"], expected="BY"),
            _fdr_method(statistics["fdr_robustness"], expected="BH"),
        )
        if "rank_ic" in analyses
        else ()
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
                *feature_observation_columns(factors, spec.column_roles),
                *(("total_market_cap",) if "fama_macbeth" in analyses else ()),
            )
        )
    )
    cutoff = (
        _statistics_window_bounds(
            TimeWindow(config["label_cutoff"], config["label_cutoff"])
        )[1]
        if "label_cutoff" in config
        else _statistics_window_bounds(window)[1]
    )
    business_columns = {
        primary_label,
        *factors,
        *(("total_market_cap",) if "fama_macbeth" in analyses else ()),
    }
    if dataset_session is not None and business_columns - set(
        dataset_session.logical_columns
    ):
        raise ValueError("statistics_required_columns_excluded")
    ranks, periods, input_dates = [], [], []
    chunks = iter_complete_time_frames(
        read_dataset(
            dataset_path,
            spec=dataset_spec if Path(dataset_path).is_file() else None,
            time_window=window,
            columns=columns,
            mode=DatasetReadMode.CHUNKED,
        )
        if dataset_session is None
        else iter_dataset_chunks(dataset_session.projected_view(columns)),
        timestamp_column="timestamp",
    )
    with closing_iterator(chunks):
        for batch in chunks:
            input_dates.extend(map(pd.Timestamp, batch["timestamp"].drop_duplicates()))
            mature = legal_training_rows(
                batch,
                fitting_cutoff=cutoff,
                availability_column=availability,
                target_column=primary_label,
            )
            if mature.empty:
                continue
            mature = mask_unobserved_features(mature, factors, spec.column_roles)
            if "rank_ic" in analyses:
                ranks.extend(
                    cross_sectional_rank_ic(
                        mature,
                        value_columns=factors,
                        label_column=primary_label,
                    ).to_dict("records")
                )
            if "fama_macbeth" in analyses:
                for _, session in mature.groupby("timestamp", sort=False):
                    periods.extend(
                        _fama_macbeth_period_records(
                            session,
                            return_column=primary_label,
                            factors=factors,
                            core_factors=core,
                        )
                    )
    tables = {}
    if "rank_ic" in analyses:
        tables["factor_diagnostics"] = _factor_summary(
            pd.DataFrame.from_records(ranks, columns=["timestamp", *factors])
            .set_index("timestamp")
            .reindex(input_dates)
            .rename_axis("timestamp")
            .reset_index(),
            factors=factors,
            families=families,
            methods=methods if _inference_blocker(config) is None else (None,),
            max_lags=max_lags,
        )
    if "fama_macbeth" in analyses:
        period_frame = pd.DataFrame.from_records(
            periods, columns=_FAMA_MACBETH_PERIOD_COLUMNS
        )
        tables["fama_macbeth"] = _fama_macbeth_summary(
            period_frame,
            factors=factors,
            core_factors=core,
            max_lags=max_lags,
            input_date_count=len(input_dates),
        )
        tables["fama_macbeth_periods"] = period_frame
    return tables


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
    global_prediction_lookup: object | None = None,
    global_inference_blocker: Mapping[str, str] | str | None = None,
    dataset_session=None,
) -> dict[str, pd.DataFrame]:
    declared = (
        dataset_session.spec
        if dataset_session is not None
        else data_spec_payload(dataset_spec)
    )
    manifest = dataset_session.manifest if dataset_session is not None else None
    dataset_id = str(dataset_spec.data_id)
    selection_group = config.get("selection_group", "explicit")
    inference_blocker = (
        "selection_reuses_evaluation_window"
        if any(
            (cutoff := (entry.get("model_facts") or {}).get("selection_cutoff"))
            is not None
            and pd.Timestamp(cutoff)
            >= _statistics_window_bounds(period_windows[period])[0]
            for (period, _family), entry in model_entries.items()
        )
        or any(
            selection.get("cutoff") is not None
            and pd.Timestamp(selection["cutoff"])
            >= _statistics_window_bounds(period_windows[period])[0]
            for period, selection in selections.items()
        )
        else None
    )
    joint_blocker = _inference_blocker(config)
    primary_label = config["evaluation"]["primary_label"]
    availability = _statistics_label_availability(config["evaluation"], dataset_spec)
    analyses = _selected_analyses(config, _MODEL_ANALYSES)
    statistics = config.get("statistics")
    root_seed = config.get("root_seed", DEFAULT_ROOT_SEED)
    if (
        type(primary_label) is not str
        or type(availability) is not str
        or type(statistics) is not dict
    ):
        raise ValueError("statistics_protocol_invalid")
    head_fraction = config["evaluation"].get("head_fraction", 0.2)
    if type(head_fraction) not in (int, float) or not 0 < head_fraction <= 1:
        raise ValueError("model_statistics_head_fraction_invalid")
    raw_label = config["evaluation"].get(
        "raw_return_label",
        "forward_return_20d"
        if "forward_return_20d" in dataset_spec.column_roles
        else None,
    )
    if raw_label is not None and (type(raw_label) is not str or not raw_label):
        raise ValueError("model_statistics_raw_return_label_invalid")
    if "model_losses" not in analyses:
        raw_label = None
    raw_availability = (
        _statistics_label_availability({"primary_label": raw_label}, dataset_spec)
        if raw_label is not None
        else None
    )
    max_lags = statistics.get("hac_lag")
    block_size = statistics.get("bootstrap_block")
    repetitions = statistics.get("bootstrap_repetitions", 2000)
    model_window = time_window
    period_bounds = {
        key: _statistics_window_bounds(window) for key, window in period_windows.items()
    }
    model_start, model_end = _statistics_window_bounds(model_window)
    family_ids = (
        tuple(
            family["family_id"]
            for family in next(iter(selections.values()))["families"]
        )
        if selections
        else tuple(config["family_ids"])
    )
    family_columns = {family: index for index, family in enumerate(family_ids)}
    prediction_columns = tuple(family_columns.values())
    benchmark_id = config.get("comparison_model_id")
    needs_benchmark = bool(
        set(analyses)
        & {"model_comparison", "spa_reality_check", "stepm_superior_models"}
    )
    if needs_benchmark and benchmark_id not in family_ids:
        raise ValueError("statistics_model_benchmark_missing")
    read_columns = tuple(
        dict.fromkeys(
            (
                *dataset_spec.primary_key,
                "eligible",
                primary_label,
                availability,
                *((raw_label, raw_availability) if raw_label is not None else ()),
            )
        )
    )
    business_columns = {
        primary_label,
        *((raw_label,) if raw_label is not None else ()),
    }
    if dataset_session is not None and business_columns - set(
        dataset_session.logical_columns
    ):
        raise ValueError("statistics_required_columns_excluded")
    needs_losses = "model_losses" in analyses
    needs_comparison = bool(
        set(analyses) & {"model_comparison", *_MODEL_ROBUSTNESS_TESTS}
    )
    rank_records: list[dict[str, object]] = []
    comparison_records: list[dict[str, object]] = []
    loss_records: list[dict[str, object]] = []
    predictive_records: list[dict[str, object]] = []
    group_records: list[dict[str, object]] = []
    group_blockers = (
        dict(global_inference_blocker)
        if isinstance(global_inference_blocker, Mapping)
        else dict.fromkeys(family_ids, global_inference_blocker)
    )
    chunks = iter_complete_session_frames(
        read_dataset(
            dataset_path,
            spec=dataset_spec if Path(dataset_path).is_file() else None,
            time_window=model_window,
            columns=read_columns,
            mode=DatasetReadMode.CHUNKED,
        )
        if dataset_session is None
        else iter_dataset_chunks(dataset_session.projected_view(read_columns)),
        timestamp_column="timestamp",
    )
    with closing_iterator(chunks):
        for session in chunks:
            if session.empty:
                continue
            mature = legal_training_rows(
                session,
                fitting_cutoff=(
                    _statistics_window_bounds(
                        TimeWindow(config["label_cutoff"], config["label_cutoff"])
                    )[1]
                    if "label_cutoff" in config
                    else _statistics_window_bounds(model_window)[1]
                ),
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
                selection = selections.get(period_id)
                raw_mature = (
                    legal_training_rows(
                        session,
                        fitting_cutoff=(
                            _statistics_window_bounds(
                                TimeWindow(
                                    config["label_cutoff"], config["label_cutoff"]
                                )
                            )[1]
                            if "label_cutoff" in config
                            else _statistics_window_bounds(model_window)[1]
                        ),
                        availability_column=raw_availability,
                        target_column=raw_label,
                    )[raw_label]
                    if raw_label is not None
                    else pd.Series(dtype="float64")
                )
                for family_id in family_ids:
                    entry = model_entries[period_id, family_id]
                    metadata = entry["model_metadata"]
                    if metadata is None:
                        mature[family_columns[family_id]] = np.nan
                        if "model_losses" in analyses:
                            predictive_records.append(
                                dict(
                                    timestamp=timestamp,
                                    family_id=family_id,
                                    **_prediction_quality(
                                        pd.Series(dtype="float64"),
                                        mature[primary_label],
                                        raw_mature,
                                        head_fraction=head_fraction,
                                    ),
                                )
                            )
                        continue
                    expected = session
                    if entry.get("asset_ids") is not None:
                        expected = expected.loc[
                            expected.asset_id.isin(entry["asset_ids"])
                        ]
                    keys = expected.loc[:, ["timestamp", "asset_id"]].copy()
                    keys["family_id"] = family_id
                    values, _found = prediction_lookup.values_for_frame(
                        keys, mark_matched=True
                    )
                    scores = pd.Series(
                        np.asarray(values, dtype="float64"), index=keys.index
                    )
                    mature[family_columns[family_id]] = scores.reindex(mature.index)
                    if "model_losses" in analyses:
                        eligible_scores = scores.reindex(
                            session.index[session.eligible.fillna(False)]
                        )
                        facts = entry.get("model_facts", {})
                        loss_reason = _prediction_loss_reason(
                            facts,
                            primary_label,
                            declared,
                            manifest,
                            model_target=metadata["target_column"],
                        )
                        predictive_records.append(
                            dict(
                                timestamp=timestamp,
                                family_id=family_id,
                                **_prediction_quality(
                                    eligible_scores,
                                    mature[primary_label],
                                    raw_mature,
                                    head_fraction=head_fraction,
                                    label_predictions=_label_predictions(
                                        eligible_scores,
                                        metadata.get("rank_parameters", {}).get(
                                            "label_calibration"
                                        ),
                                    ),
                                    loss_reason=loss_reason,
                                ),
                            )
                        )
                if global_prediction_lookup is not None:
                    for family_id in family_ids:
                        keys = session.loc[:, ["timestamp", "asset_id"]].copy()
                        keys["family_id"] = family_id
                        values, _found = global_prediction_lookup.values_for_frame(
                            keys, mark_matched=True
                        )
                        shared = mature.loc[
                            :, ["timestamp", "asset_id", primary_label]
                        ].copy()
                        shared[0] = pd.Series(values, index=keys.index).reindex(
                            mature.index
                        )
                        shared[1] = mature[family_columns[family_id]]
                        paired_rank, count = (
                            _common_model_rank_ic(
                                shared, family_ids=(0, 1), label_column=primary_label
                            )
                            if not shared.empty
                            else (pd.DataFrame([{0: np.nan, 1: np.nan}]), 0)
                        )
                        if count:
                            entry = model_entries[period_id, family_id]
                            cutoffs = tuple(
                                value
                                for value in (
                                    selection.get("cutoff")
                                    if selection is not None
                                    else None,
                                    (entry.get("model_facts") or {}).get(
                                        "selection_cutoff"
                                    ),
                                )
                                if value is not None
                            )
                            if not cutoffs:
                                if group_blockers.get(family_id) is None:
                                    group_blockers[family_id] = (
                                        "selection_chronology_not_established"
                                    )
                            elif any(
                                pd.Timestamp(value) >= timestamp for value in cutoffs
                            ):
                                group_blockers[family_id] = (
                                    "selection_reuses_evaluation_window"
                                )
                        group_records.append(
                            {
                                "timestamp": timestamp,
                                "family_id": family_id,
                                "global": paired_rank.iloc[0][0],
                                "quarterly": paired_rank.iloc[0][1],
                                "observation_count": count,
                            }
                        )
                if selection is not None:
                    winner = selection["selected_family_id"]
                    mature[_PREDICTION_COLUMN] = (
                        mature[family_columns[winner]] if winner is not None else np.nan
                    )
                rank = cross_sectional_rank_ic(
                    mature,
                    value_columns=(
                        *((_PREDICTION_COLUMN,) if selection is not None else ()),
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
                if needs_losses or needs_comparison:
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
                for family_id in family_ids if needs_losses else ():
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
    if not rank_records:
        raise ValueError("statistics_model_observations_empty")
    rank_by_date = pd.DataFrame.from_records(rank_records)
    prediction_summaries = []
    if set(analyses) & {"prediction_hac", "prediction_bootstrap"}:
        summaries = (
            [(prediction_id, _PREDICTION_COLUMN)]
            if selections
            else list(family_columns.items())
            if "analyses" in config
            else []
        )
        for identity, column in summaries:
            prediction_summaries.append(
                _prediction_summary(
                    rank_by_date.loc[:, [column]].rename(
                        columns={column: _PREDICTION_COLUMN}
                    ),
                    prediction_id=identity,
                    inference_blocker=inference_blocker,
                    max_lags=max_lags if "prediction_hac" in analyses else None,
                    block_size=block_size
                    if "prediction_bootstrap" in analyses
                    else None,
                    repetitions=repetitions
                    if "prediction_bootstrap" in analyses
                    else None,
                    seed=derive_seed(
                        root_seed, "a_share_statistics_bootstrap", dataset_id, identity
                    )
                    if "prediction_bootstrap" in analyses
                    else None,
                )
            )
    if needs_losses:
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
    if needs_comparison:
        comparison_by_date = (
            pd.DataFrame.from_records(
                comparison_records, columns=["timestamp", *prediction_columns]
            )
            .set_index("timestamp")
            .rename(
                columns={column: family for family, column in family_columns.items()}
            )
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
            analyses=analyses,
        )
    if "group_comparison" in statistics:
        group_comparison = _paired_group_rankic(
            pd.DataFrame.from_records(
                group_records,
                columns=[
                    "timestamp",
                    "family_id",
                    "global",
                    "quarterly",
                    "observation_count",
                ],
            ),
            selection_group=selection_group,
            family_ids=family_ids,
            config=config,
            inference_blocker=group_blockers,
        )

    tables = {}
    if "model_losses" in analyses:
        tables["model_losses"] = model_losses
        tables["model_prediction_metrics"] = pd.DataFrame.from_records(
            predictive_records
        )
    if "model_comparison" in analyses:
        tables["model_comparison"] = comparison
    if set(analyses) & set(_MODEL_ROBUSTNESS_TESTS):
        tables.update(robustness=robustness, model_reality_checks=tests)
    if prediction_summaries:
        tables["prediction_diagnostics"] = pd.concat(
            prediction_summaries, ignore_index=True
        )
    if selections and "analyses" not in config:
        tables["model_trials"] = trial_rows.copy()
    if "group_comparison" in statistics:
        tables["model_group_comparison"] = group_comparison
    return tables


def _prediction_quality(
    scores,
    target,
    raw_returns,
    *,
    head_fraction,
    label_predictions=None,
    loss_reason=None,
):
    """Keep raw-score cohorts fixed and compute losses only in the target scale."""
    scores = scores.loc[np.isfinite(scores)]
    paired = target.reindex(scores.index)
    valid = np.isfinite(paired)
    predictions = (
        scores if label_predictions is None else label_predictions.reindex(scores.index)
    )
    errors, baseline_errors = _MeanSquaredError(), _MeanSquaredError()
    for label, prediction in zip(
        paired.loc[valid], predictions.loc[valid], strict=True
    ):
        if loss_reason is None:
            errors.add(prediction, label)
        baseline_errors.add(0.0, label)
    count = int(valid.sum())

    def represent(value):
        try:
            result = float(value)
        except OverflowError:
            return np.nan
        return result if np.isfinite(result) else np.nan

    error, baseline = errors.mean(), baseline_errors.mean()
    mse = errors.result()
    baseline_mse = baseline_errors.result()
    r2 = (
        represent(1 - error / baseline)
        if loss_reason is None and error is not None and baseline
        else np.nan
    )
    head = scores.iloc[:0]
    if len(scores):
        cutoff = scores.nlargest(int(np.ceil(len(scores) * head_fraction))).iloc[-1]
        head = scores.loc[scores.ge(cutoff)]
    returns = raw_returns.reindex(head.index)
    observed = returns.loc[np.isfinite(returns)]
    pool = raw_returns.loc[np.isfinite(raw_returns)]
    head_mean = float(mean(observed)) if len(observed) else np.nan
    pool_mean = float(mean(pool)) if len(pool) else np.nan
    excess = (
        represent(Fraction(head_mean) - Fraction(pool_mean))
        if len(observed) and len(pool)
        else np.nan
    )
    return {
        "prediction_count": len(scores),
        "target_count": count,
        "mse": np.nan if mse is None else mse,
        "mse_reason": None
        if mse is not None
        else loss_reason or ("no_valid_targets" if not count else "nonfinite_result"),
        "baseline_mse": np.nan if baseline_mse is None else baseline_mse,
        "oos_r2": r2,
        "oos_r2_reason": None
        if np.isfinite(r2)
        else loss_reason
        if loss_reason is not None
        else "no_valid_targets"
        if not count
        else "zero_baseline_error"
        if not baseline
        else "nonfinite_result",
        "head_selected_count": len(head),
        "head_evaluable_count": len(observed),
        "head_coverage": len(observed) / len(head) if len(head) else np.nan,
        "pool_evaluable_count": len(pool),
        "head_mean_raw_return": head_mean,
        "pool_mean_raw_return": pool_mean,
        "head_excess_raw_return": excess,
    }


def _model_target_metadata(config, spec, window):
    """Describe the actual evaluation target without binding unrelated inputs."""
    label = config["evaluation"]["primary_label"]
    availability = _statistics_label_availability(config["evaluation"], spec)
    cutoff = (
        _statistics_window_bounds(
            TimeWindow(config["label_cutoff"], config["label_cutoff"])
        )[1]
        if "label_cutoff" in config
        else _statistics_window_bounds(window)[1]
    )
    return {
        "selection_group": config.get("selection_group", "explicit"),
        "coverage_basis": "persisted_prediction_keys",
        "sample_role": config["sample_role"],
        "window": {"start": str(window.start), "end": str(window.end)},
        "evaluation": {
            **config["evaluation"],
            "label_availability_columns": {label: availability},
        },
        "prediction_metrics": {
            "head_fraction": config["evaluation"].get("head_fraction", 0.2),
            "raw_return_label": config["evaluation"].get(
                "raw_return_label",
                "forward_return_20d"
                if "forward_return_20d" in spec.column_roles
                else None,
            ),
            "baseline": "zero_prediction",
            "aggregation": "date_equal_mse; oos_r2=1-mean(mse)/mean(baseline_mse)",
            "head_selection": (
                "finite_predictions_before_labels; ceil_fraction_cutoff_including_ties"
            ),
            "return_basis": (
                "gross_raw_horizon; "
                "same_date_eligible_pool_equal_weight; no_compounding"
            ),
        },
        "evaluation_axis": "cross_sectional",
        **(
            {"group_comparison_target": "explicit_current_evaluation"}
            if "group_comparison" in config["statistics"]
            and config.get("selection_group") == "quarterly"
            else {}
        ),
        "column_roles": {label: str(spec.column_roles[label])},
        "column_units": {
            column: unit
            for column, unit in spec.column_units.items()
            if column == label
        },
        "label_cutoff": cutoff.isoformat(),
    }


def _paired_group_rankic(
    paired_rankic,
    *,
    selection_group,
    family_ids,
    config,
    inference_blocker=None,
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
    statistics = config["statistics"]
    declaration = statistics["group_comparison"]
    keys = ["timestamp", "family_id"]
    if paired_rankic.duplicated(keys).any():
        raise ValueError("model_group_comparison_coverage_invalid")
    if paired_rankic.empty:
        global_values = pd.DataFrame(
            index=pd.DatetimeIndex([], name="timestamp"),
            columns=family_ids,
            dtype="float64",
        )
        quarterly_values = global_values.copy()
    else:
        paired = paired_rankic.set_index(keys).sort_index()
        global_values = paired["global"].unstack("family_id")
        quarterly_values = paired["quarterly"].unstack("family_id")
    if set(global_values.columns) != set(family_ids):
        raise ValueError("model_group_comparison_family_incomplete")
    differences = quarterly_values - global_values
    access = config["sample_role"]
    fdr_method = (
        None if _inference_blocker(config) is not None else declaration["fdr_method"]
    )
    blockers = (
        dict(inference_blocker)
        if isinstance(inference_blocker, Mapping)
        else dict.fromkeys(family_ids, inference_blocker)
    )
    inferential = differences.copy()
    for family in family_ids:
        if blockers.get(family) is not None:
            inferential[family] = np.nan
    inference = (
        time_series_inference_summary(
            inferential.set_axis(range(len(family_ids)), axis=1).reset_index(),
            identity_column="timestamp",
            max_lags=statistics["hac_lag"],
            fdr_method=fdr_method,
            fdr_alpha=declaration["alpha"],
            fdr_family_size=len(family_ids),
            confidence=declaration["confidence"],
        )
        .set_index("timestamp")
        .set_axis(differences.columns)
        if not differences.empty
        and any(blockers.get(family) is None for family in family_ids)
        else None
    )
    rows = []
    for family in family_ids:
        blocker = blockers.get(family)
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
            reason=blocker if len(values) else "no_common_observations",
            fdr_family_size=len(family_ids),
            fdr_method=fdr_method or "NOT_APPLICABLE",
            fdr_alpha=declaration["alpha"],
            hac_lag=statistics["hac_lag"],
            bootstrap_block_size=statistics["bootstrap_block"],
            bootstrap_repetitions=statistics["bootstrap_repetitions"],
            bootstrap_seed=declaration["seed"],
            confidence=declaration["confidence"],
            bootstrap_status="UNDEFINED",
            bootstrap_reason=blocker if len(values) else "no_common_observations",
        )
        if inference is not None and blocker is None and len(values):
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
            if len(values) != len(differences) or not np.isfinite(values).all():
                row["bootstrap_reason"] = "incomplete_session_clock"
            elif len(values) > statistics["bootstrap_block"]:
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
                row["bootstrap_reason"] = "bootstrap_block_exceeds_observations"
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
    max_lags: int | None,
    block_size: int | None,
    repetitions: int | None,
    seed: int | None,
) -> pd.DataFrame:
    values = (
        rank_by_date[_PREDICTION_COLUMN].dropna()
        if _PREDICTION_COLUMN in rank_by_date
        else pd.Series(dtype="float64")
    )
    count = len(values)
    if count != len(rank_by_date):
        inference_blocker = inference_blocker or "incomplete_session_clock"
    row = {
        "prediction_id": prediction_id,
        "observation_count": count,
        "status": "UNDEFINED",
        "reason": inference_blocker
        or ("not_requested" if max_lags is None else "hac_lag_exceeds_observations"),
        "mean": float(mean(values)) if count else np.nan,
        "standard_error": np.nan,
        "t_statistic": np.nan,
        "p_value": np.nan,
        "ci_lower": np.nan,
        "ci_upper": np.nan,
        "bootstrap_status": "UNDEFINED",
        "bootstrap_reason": inference_blocker
        or (
            "not_requested"
            if block_size is None
            else "bootstrap_insufficient_observations"
        ),
        "bootstrap_lower": np.nan,
        "bootstrap_upper": np.nan,
        "bootstrap_block_size": block_size,
        "bootstrap_repetitions": repetitions,
        "bootstrap_seed": seed,
        "input_date_count": len(rank_by_date),
    }
    if inference_blocker is not None:
        return pd.DataFrame([row])
    if max_lags is not None and count >= max(2, max_lags + 1):
        try:
            inference = hac_mean(values, max_lags=max_lags)
        except ValueError as exc:
            if str(exc) != "hac_inference_undefined":
                raise
            row["reason"] = str(exc)
        else:
            row.update(
                status="DEFINED",
                reason=None,
                standard_error=inference.standard_error,
                t_statistic=inference.t_statistic,
                p_value=inference.p_value,
                ci_lower=inference.confidence_interval[0],
                ci_upper=inference.confidence_interval[1],
            )
    if block_size is not None and count >= 2:
        try:
            bootstrap = stationary_bootstrap_mean_ci(
                values, block_size=block_size, repetitions=repetitions, seed=seed
            )
        except ValueError as exc:
            if str(exc) != "bootstrap_inference_undefined":
                raise
            row["bootstrap_reason"] = str(exc)
        else:
            row.update(
                bootstrap_status="DEFINED",
                bootstrap_reason=None,
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
    analyses=_MODEL_ANALYSES,
):
    common = rank_by_date.loc[:, list(family_ids)].dropna()
    if len(common) != len(rank_by_date):
        inference_blocker = inference_blocker or "incomplete_session_clock"
    losses = -common
    challengers = tuple(family for family in family_ids if family != benchmark_id)
    comparisons = []
    for family in challengers if "model_comparison" in analyses else ():
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
    for test_id in (test for test in _MODEL_ROBUSTNESS_TESTS if test in analyses):
        seed = derive_seed(
            root_seed, "a_share_model_reality_check", dataset_id, sample_role, test_id
        )
        result = None
        has_candidates = (
            len(family_ids) > 1
            if test_id == "model_confidence_set"
            else bool(challengers)
        )
        if joint_blocker is None and has_candidates and (len(common) > block_size):
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
            else "no_comparison_models"
            if not has_candidates
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
        complete = (
            frame.loc[:, [return_column, *regression_factors]].notna().all(axis=1)
        )
        if weight_column is not None:
            weights = strict_real_numeric_series(
                frame[weight_column],
                error="fama_macbeth_weights_invalid",
                errors="coerce",
            )
            complete &= np.isfinite(weights) & weights.gt(0)
        complete_count = int(complete.sum())
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
            "reason": "incomplete_session_clock"
            if count != input_date_count
            else "hac_lag_exceeds_observations",
            "mean": float(mean(values)) if count else np.nan,
            "standard_error": np.nan,
            "t_statistic": np.nan,
            "p_value": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "white_defined_period_count": white_defined,
            "white_undefined_period_count": white_undefined,
            "input_date_count": input_date_count,
        }
        if count == input_date_count and count >= max(2, max_lags + 1):
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


def _statistics_input_session(config, window):
    from dataclasses import fields

    from quant_research_platform.contracts.data import DataSpec

    value = config["dataset"]
    if "input_spec" in config:
        if isinstance(value, dict):
            if "spec" in value:
                raise ValueError("dataset_spec_already_declared")
            value = {**value, "spec": config["input_spec"]}
        else:
            value = {"path": value, "spec": config["input_spec"]}
    options = _dataset_input_options(value)
    session = open_dataset_session(**options).filtered_view(window)
    payload = session.spec
    values = {
        field.name: payload[field.name]
        for field in fields(DataSpec)
        if field.init and field.name in payload
    }
    for key in ("primary_key", "partition_columns"):
        values[key] = tuple(values[key])
    from quant_research_platform.contracts.columns import _validate_asset_time_key

    spec = DataSpec(**values)
    _validate_asset_time_key(spec.primary_key, spec.column_roles)
    return session, spec


def _statistics_label_availability(evaluation, spec):
    label = evaluation["primary_label"]
    availability = evaluation.get("label_availability_columns", {})
    if not isinstance(availability, dict) or set(availability) - {label}:
        raise ValueError("statistics_label_availability_invalid")
    declared = getattr(spec, "label_availability_columns", {}).get(label)
    if (
        label in availability
        and declared is not None
        and availability[label] != declared
    ):
        raise ValueError("statistics_label_availability_conflict")
    column = availability.get(label, declared)
    if column is None:
        column = training_availability_columns(label)[0]
    if not isinstance(column, str) or not column or column not in spec.column_roles:
        raise ValueError("statistics_label_availability_invalid")
    if str(spec.column_roles.get(label)) != "label":
        raise ValueError("statistics_label_role_invalid")
    if any(
        str(spec.column_roles.get(factor)) not in {"factor", "feature"}
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
    if "evaluation_inputs" in config:
        return _factor_statistics_from_evaluations(config)
    required = {
        "dataset",
        "output",
        "window",
        "evaluation",
        "statistics",
    }
    optional = {
        "sample_role",
        "input_spec",
        "analyses",
        "label_cutoff",
        "selection_is_exploratory",
        "expectations",
        "feature_recipe",
        "training_weighting",
    }
    if set(config) - required - optional or required - set(config):
        raise ValueError("factor_statistics_config_invalid")
    config.setdefault("sample_role", "exploratory")
    if type(config["sample_role"]) is not str or not config["sample_role"].strip():
        raise ValueError("statistics_sample_role_invalid")
    evaluation, policy = config["evaluation"], config["statistics"]
    if (
        not isinstance(evaluation, dict)
        or set(evaluation) - {"factors", "primary_label", "label_availability_columns"}
        or {"factors", "primary_label"} - set(evaluation)
    ):
        raise ValueError("factor_statistics_evaluation_invalid")
    analyses = _selected_analyses(config, ("rank_ic", "fama_macbeth"))
    expected = {"hac_lag"}
    if "rank_ic" in analyses:
        if isinstance(policy, dict):
            policy = {"fdr_primary": "BY_5pct", "fdr_robustness": "BH_5pct", **policy}
            config["statistics"] = policy
        expected.update(("factor_fdr_families", "fdr_primary", "fdr_robustness"))
    if "fama_macbeth" in analyses:
        expected.add("fama_macbeth_core_factors")
    known = {
        "hac_lag",
        "factor_fdr_families",
        "fdr_primary",
        "fdr_robustness",
        "fama_macbeth_core_factors",
    }
    if (
        not isinstance(policy, dict)
        or expected - set(policy)
        or set(policy) - known
        or type(policy["hac_lag"]) is not int
        or policy["hac_lag"] < 0
    ):
        raise ValueError("factor_statistics_policy_invalid")
    output = Path(config["output"])
    reject_if_exists(output)
    window = TimeWindow(**config["window"])
    session, spec = _statistics_input_session(config, window)
    tables = build_a_share_factor_statistics_tables(
        dataset_path=session.root,
        dataset_session=session,
        dataset_spec=spec,
        config=config,
        time_window=window,
    )
    return write_statistics_artifact(
        output,
        metadata={
            **{key: value for key, value in config.items() if key != "output"},
            "inference_scope": "exploratory",
            "formal_selection_qualification": "not_established",
            "column_units": {
                column: unit
                for column, unit in spec.column_units.items()
                if column in {evaluation["primary_label"], *evaluation["factors"]}
            },
        },
        tables=tables,
        table_group="factor",
    )


def _factor_statistics_from_evaluations(config):
    """Reduce published daily facts without reopening their dataset producers."""
    from contextlib import ExitStack

    from quant_research_platform.artifacts.outputs.evaluation import (
        _evaluation_spec,
        _iter_evaluation_member_batches,
    )
    from quant_research_platform.artifacts.outputs.evaluation_daily import (
        daily_frame,
        validate_daily_counts,
    )
    from quant_research_platform.artifacts.outputs.manifest import (
        validate_output_bundle,
    )
    from quant_research_platform.artifacts.statistics import (
        STATISTICS_TABLE_COLUMNS,
        write_statistics_artifact,
    )
    from quant_research_platform.research.statistics import fdr_adjust
    from quant_research_platform.storage.atomic import reject_if_exists

    required = {"evaluation_inputs", "output", "statistics"}
    optional = {"window", "analyses", "sample_role", "evaluation"}
    if required - set(config) or set(config) - required - optional:
        raise ValueError("factor_statistics_evaluation_config_invalid")
    paths = config["evaluation_inputs"]
    if (
        type(paths) is not list
        or not paths
        or any(type(path) is not str or not path for path in paths)
    ):
        raise ValueError("factor_statistics_evaluation_inputs_invalid")
    analyses = _selected_analyses(config, ("rank_ic", "stratified_ic"))
    if "rank_ic" in analyses and len(paths) != 1:
        raise ValueError("factor_statistics_rank_ic_multiple_evaluations")
    policy = config["statistics"]
    needed = {"hac_lag"}
    if isinstance(policy, dict):
        defaults = {}
        if "rank_ic" in analyses:
            defaults.update(fdr_primary="BY_5pct", fdr_robustness="BH_5pct")
        if "stratified_ic" in analyses:
            defaults.update(fdr_method="BY", fdr_alpha=0.05)
        policy = defaults | policy
        config["statistics"] = policy
    if "stratified_ic" in analyses:
        needed |= {"fdr_method", "fdr_alpha"}
    if "rank_ic" in analyses:
        needed |= {"factor_fdr_families", "fdr_primary", "fdr_robustness"}
    if (
        type(policy) is not dict
        or set(policy) != needed
        or type(policy.get("hac_lag")) is not int
        or policy["hac_lag"] < 0
    ):
        raise ValueError("factor_statistics_policy_invalid")
    lag = policy["hac_lag"]
    if "stratified_ic" in analyses:
        fdr_adjust([0.5], method=policy["fdr_method"], alpha=policy["fdr_alpha"])
    reject_if_exists(Path(config["output"]))
    entries = []
    scopes = set()
    binding = None
    for path in paths:
        manifest, _ = validate_output_bundle(
            Path(path),
            expected_definition_id="evaluation_outputs",
            selected_files=("metrics.parquet",),
        )
        metadata = manifest["metadata"]
        if (
            "input_facts" not in metadata
            or metadata["evaluation_axis"] != "cross_sectional"
        ):
            raise ValueError("factor_statistics_evaluation_science_missing")
        scope = metadata["sample_scope"]
        if scope in scopes:
            raise ValueError("factor_statistics_evaluation_duplicate_scope")
        scopes.add(scope)
        config.setdefault("window", metadata["window"])
        current = {
            key: metadata[key]
            for key in (
                "market_id",
                "evaluation_axis",
                "signals",
                "label_columns",
                "label_availability_columns",
                "filter_mode",
                "window",
                "cutoff",
                "input_facts",
                "strata",
                "common_rule",
            )
        }
        current["window"] = config["window"]
        current["input_facts"] = {
            key: value
            for key, value in current["input_facts"].items()
            if key != "selector"
        }
        if binding is not None and current != binding:
            raise ValueError("factor_statistics_evaluation_inputs_incompatible")
        binding = current
        requested = TimeWindow(**config["window"])
        low, high = _statistics_window_bounds(requested)
        original_low, original_high = _statistics_window_bounds(
            TimeWindow(**metadata["window"])
        )
        if low < original_low or high > original_high:
            raise ValueError("factor_statistics_evaluation_window_outside")
        entries.append(
            (
                Path(path),
                metadata,
                None if low == original_low and high == original_high else requested,
            )
        )
    config.setdefault("sample_role", entries[0][1]["sample_role"])
    if type(config["sample_role"]) is not str or not config["sample_role"].strip():
        raise ValueError("statistics_sample_role_invalid")
    series = {}
    input_counts = {}
    input_dates = {}
    for path, metadata, window in entries:
        scope = metadata["sample_scope"]
        input_dates[scope] = []
        for signal in metadata["signals"]:
            for label in metadata["label_columns"]:
                series[(signal["signal_id"], label, scope, "all", None)] = {
                    "ic": [],
                    "rankic": [],
                    "pairs": 0,
                    "dates": 0,
                    "rankic_daily": [],
                }
        members = ["metrics"]
        if metadata["optional_member_rows"]:
            members += ["daily_coverage", "daily_strata"]
        validate_output_bundle(
            path,
            expected_definition_id="evaluation_outputs",
            selected_files=tuple(name + ".parquet" for name in members),
        )
        date_count = 0
        with closing_iterator(ExitStack()) as stack:
            streams = {
                name: stack.enter_context(
                    closing_iterator(
                        iter_complete_session_frames(
                            _iter_evaluation_member_batches(
                                path, name=name, metadata=metadata, time_window=window
                            ),
                            timestamp_column="timestamp",
                        )
                    )
                )
                for name in members
            }

            def next_date(stream):
                return next((frame for frame in stream if not frame.empty), None)

            heads = {
                name: next_date(stream)
                for name, stream in streams.items()
                if name != "metrics"
            }
            for metrics in streams["metrics"]:
                if metrics.empty:
                    continue
                timestamp = metrics.timestamp.iloc[0]
                date_count += 1
                input_dates[scope].append(timestamp)
                daily = {}
                for name in heads:
                    head = heads[name]
                    if (
                        head is not None
                        and not head.empty
                        and head.timestamp.iloc[0] < timestamp
                    ):
                        raise ValueError("evaluation_daily_dates_mismatch")
                    daily[name] = (
                        head
                        if head is not None
                        and not head.empty
                        and head.timestamp.iloc[0] == timestamp
                        else None
                    )
                    if daily[name] is not None:
                        heads[name] = next_date(streams[name])
                coverage = daily.get("daily_coverage")
                if "daily_coverage" in members and coverage is None:
                    raise ValueError("evaluation_daily_dates_mismatch")
                validate_daily_counts(
                    metrics,
                    coverage,
                    daily.get("daily_strata")
                    if daily.get("daily_strata") is not None
                    else daily_frame([], "daily_strata")
                    if "daily_strata" in members
                    else None,
                    spec=_evaluation_spec(metadata),
                    science=metadata,
                )
                for name, frame in (
                    ("metrics", metrics),
                    ("daily_strata", daily.get("daily_strata")),
                ):
                    if frame is None:
                        continue
                    for row in frame.itertuples(index=False):
                        kind = "all" if name == "metrics" else row.stratification
                        stratum = (
                            None
                            if name == "metrics" or pd.isna(row.stratum)
                            else row.stratum
                        )
                        values = series.setdefault(
                            (row.signal, row.label, scope, kind, stratum),
                            {"ic": [], "rankic": [], "pairs": 0, "dates": 0},
                        )
                        if "rank_ic" in analyses and name == "metrics":
                            values["rankic_daily"].append(float(row.rankic))
                        values["dates"] += 1
                        values["pairs"] += int(row.n_obs)
                        for statistic in ("ic", "rankic"):
                            value = getattr(row, statistic)
                            if pd.notna(value):
                                values[statistic].append(float(value))
            if any(head is not None and not head.empty for head in heads.values()):
                raise ValueError("evaluation_daily_dates_mismatch")
        input_counts[scope] = date_count
    tables = {}
    if "stratified_ic" in analyses:
        rows = []
        for key, values in series.items():
            for statistic in ("ic", "rankic"):
                observations = values[statistic]
                row = dict(
                    zip(
                        (
                            "signal",
                            "label",
                            "sample_scope",
                            "stratification",
                            "stratum",
                        ),
                        key,
                        strict=True,
                    )
                )
                row.update(
                    statistic=statistic,
                    input_date_count=input_counts[key[2]],
                    observed_date_count=values["dates"],
                    observation_count=len(observations),
                    observation_pairs=values["pairs"],
                    mean=float(np.mean(observations)) if observations else np.nan,
                    standard_error=np.nan,
                    t_statistic=np.nan,
                    p_value=np.nan,
                    ci_lower=np.nan,
                    ci_upper=np.nan,
                    status="UNDEFINED",
                    reason="incomplete_session_clock"
                    if len(observations) != input_counts[key[2]]
                    else "insufficient_dates_for_hac",
                    hac_lag=lag,
                    fdr_method=policy["fdr_method"],
                    fdr_alpha=policy["fdr_alpha"],
                    fdr_adjusted_p_value=np.nan,
                    defined_hypothesis_count=0,
                )
                if len(observations) == input_counts[key[2]] and len(
                    observations
                ) >= max(2, lag + 1):
                    try:
                        inference = hac_mean(observations, max_lags=lag)
                    except ValueError as error:
                        if str(error) != "hac_inference_undefined":
                            raise
                        row["reason"] = str(error)
                    else:
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
        result = pd.DataFrame.from_records(
            rows,
            columns=STATISTICS_TABLE_COLUMNS["factor_stratified_diagnostics.parquet"],
        )
        result["defined_hypothesis_count"] = int(result.status.eq("DEFINED").sum())
        tables["factor_stratified_diagnostics"] = _apply_fdr(
            result, policy["fdr_method"], policy["fdr_alpha"], len(result)
        )
    if "rank_ic" in analyses:
        evaluation = config.get("evaluation")
        if not isinstance(evaluation, dict) or set(evaluation) != {
            "primary_label",
            "factors",
        }:
            raise ValueError("factor_statistics_evaluation_invalid")
        factors = tuple(evaluation["factors"])
        if (
            set(factors) - {signal["signal_id"] for signal in entries[0][1]["signals"]}
            or evaluation["primary_label"] not in entries[0][1]["label_columns"]
        ):
            raise ValueError("factor_statistics_evaluation_invalid")
        scope = entries[0][1]["sample_scope"]
        ranks = pd.DataFrame(
            {
                "timestamp": input_dates[scope],
                **{
                    factor: series[
                        (factor, evaluation["primary_label"], scope, "all", None)
                    ]["rankic_daily"]
                    for factor in factors
                },
            }
        )
        tables["factor_diagnostics"] = _factor_summary(
            ranks,
            factors=factors,
            families=_factor_families(policy["factor_fdr_families"], factors),
            methods=(
                _fdr_method(policy["fdr_primary"], expected="BY"),
                _fdr_method(policy["fdr_robustness"], expected="BH"),
            ),
            max_lags=lag,
        )
    return write_statistics_artifact(
        Path(config["output"]),
        table_group="factor",
        tables=tables,
        metadata={
            **{key: value for key, value in config.items() if key != "output"},
            "evaluation_binding": binding,
            "family_scopes": sorted(scopes),
            "scope_facts": {
                metadata["sample_scope"]: {
                    "comparison_inputs": metadata["comparison_inputs"],
                    "sample_role": metadata["sample_role"],
                }
                for _, metadata, _ in entries
            },
            "stratum_domains": {
                kind: sorted(
                    {key[4] for key in series if key[3] == kind},
                    key=lambda value: (value is not None, value or ""),
                )
                for kind in binding["strata"].get("kinds", [])
            },
            "input_date_counts": input_counts,
            "inference_scope": "exploratory",
            "formal_selection_qualification": "not_established",
            "hac_lag_unit": "observed_dates_with_defined_statistic",
            "multiple_testing_scope": (
                "all_defined_signals_labels_strata_scopes_ic_rankic_in_explicit_inputs"
            ),
        },
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

    from quant_research_platform.artifacts.outputs.manifest import (
        validate_output_bundle,
    )
    from quant_research_platform.artifacts.outputs.prediction import (
        _iter_prediction_member,
        _PredictionLookup,
        prediction_model_metadata,
    )
    from quant_research_platform.artifacts.statistics import write_statistics_artifact
    from quant_research_platform.contracts.json_codec import qrp_json_mapping_snapshot
    from quant_research_platform.research.models.trial_selection import read_selection
    from quant_research_platform.storage.atomic import reject_if_exists
    from quant_research_platform.storage.exact_keys import ExactPrimaryKeyIndex

    config = qrp_json_mapping_snapshot(config)
    required = {
        "dataset",
        "output",
        "window",
        "evaluation",
        "statistics",
        "periods",
    }
    optional = {
        "analyses",
        "root_seed",
        "comparison_model_id",
        "sample_role",
        "family_ids",
        "selection_group",
        "prediction_id",
        "label_cutoff",
        "input_spec",
        "global_predictions",
        "selection_is_exploratory",
        "expectations",
        "feature_recipe",
        "training_weighting",
    }
    if required - set(config) or set(config) - required - optional:
        raise ValueError("model_statistics_config_invalid")
    analyses = _selected_analyses(config, _MODEL_ANALYSES)
    config.setdefault("sample_role", "exploratory")
    if type(config["sample_role"]) is not str or not config["sample_role"].strip():
        raise ValueError("statistics_sample_role_invalid")
    evaluation, policy = config["evaluation"], config["statistics"]
    if (
        not isinstance(evaluation, dict)
        or set(evaluation)
        - {
            "primary_label",
            "label_availability_columns",
            "head_fraction",
            "raw_return_label",
        }
        or "primary_label" not in evaluation
    ):
        raise ValueError("model_statistics_evaluation_invalid")
    if not isinstance(policy, dict):
        raise ValueError("model_statistics_policy_invalid")
    needed = set()
    if (
        set(analyses) & {"prediction_hac", "model_comparison"}
        or "group_comparison" in policy
    ):
        needed.add("hac_lag")
    if "group_comparison" in policy:
        needed |= {"bootstrap_block", "bootstrap_repetitions"}
    if set(analyses) & {
        "prediction_bootstrap",
        "model_comparison",
        *_MODEL_ROBUSTNESS_TESTS,
    }:
        needed |= {"bootstrap_block", "bootstrap_repetitions"}
        seed = config.setdefault("root_seed", DEFAULT_ROOT_SEED)
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("model_statistics_config_invalid")
    if set(analyses) & {
        "model_comparison",
        "spa_reality_check",
        "stepm_superior_models",
    }:
        if not isinstance(config.get("comparison_model_id"), str):
            raise ValueError("model_statistics_config_invalid")
    if "bootstrap_repetitions" in needed:
        policy.setdefault("bootstrap_repetitions", 2000)
    if (
        not isinstance(policy, dict)
        or set(policy)
        - {"hac_lag", "bootstrap_block", "bootstrap_repetitions", "group_comparison"}
        or needed - set(policy)
        or any(
            type(policy[key]) is not int
            or policy[key]
            < {"hac_lag": 0, "bootstrap_block": 1, "bootstrap_repetitions": 2}[key]
            for key in needed
        )
    ):
        raise ValueError("model_statistics_policy_invalid")
    paired = "group_comparison" in policy
    if (
        paired != ("global_predictions" in config)
        or paired
        and config.get("selection_group") != "quarterly"
    ):
        raise ValueError("model_statistics_comparison_inputs_invalid")
    if paired and (
        not isinstance(policy["group_comparison"], dict)
        or set(policy["group_comparison"])
        != {"fdr_method", "alpha", "seed", "confidence"}
    ):
        raise ValueError("model_statistics_comparison_policy_invalid")
    if paired:
        declaration = policy["group_comparison"]
        if (
            type(declaration["confidence"]) is not float
            or not 0 < declaration["confidence"] < 1
            or type(declaration["alpha"]) is not float
            or not 0 < declaration["alpha"] < 1
            or declaration["fdr_method"] not in ("BH", "BY")
        ):
            raise ValueError("model_statistics_comparison_policy_invalid")
        _validate_bootstrap_configuration(
            policy["bootstrap_block"],
            policy["bootstrap_repetitions"],
            declaration["seed"],
        )
    if paired and (
        not isinstance(config["global_predictions"], list)
        or not config["global_predictions"]
        or any(
            not isinstance(path, str) or not path
            for path in config["global_predictions"]
        )
    ):
        raise ValueError("model_statistics_comparison_inputs_invalid")
    output = Path(config["output"])
    reject_if_exists(output)
    window = TimeWindow(**config["window"])
    if not isinstance(config["periods"], list) or not config["periods"]:
        raise ValueError("statistics_prediction_periods_required")
    direct = all(
        isinstance(item, dict) and "selection" not in item for item in config["periods"]
    )
    if not direct and any("selection" not in item for item in config["periods"]):
        raise ValueError("statistics_prediction_selection_mixed")
    if direct:
        family_ids = config.get("family_ids")
        if (
            not isinstance(family_ids, list)
            or not family_ids
            or any(not isinstance(value, str) or not value for value in family_ids)
            or len(set(family_ids)) != len(family_ids)
        ):
            raise ValueError("statistics_candidate_family_declaration_required")
        family_ids = tuple(family_ids)
    else:
        if not {"selection_group", "prediction_id"} <= set(config):
            raise ValueError("model_statistics_selection_fields_required")
        family_ids = None
    periods, entries, selections = {}, {}, {}
    market = None
    previous_end = None
    index = ExactPrimaryKeyIndex(primary_key=("timestamp", "asset_id", "family_id"))
    lookup = _PredictionLookup(
        index=index,
        primary_key=("timestamp", "asset_id", "family_id"),
        prediction_column="prediction_score",
    )
    reference_lookup = None
    reference_blocker = {}
    try:
        for item in config["periods"]:
            if not isinstance(item, dict) or set(item) != (
                {"period_id", "window", "predictions"}
                | (set() if direct else {"selection"})
            ):
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
            selection = None
            if direct:
                wanted = dict.fromkeys(family_ids)
            else:
                selection = read_selection(Path(item["selection"]))
                market = selection["market_id"] if market is None else market
                validate_input_market(selection, market)
                if selection["selection_group"] != config["selection_group"]:
                    raise ValueError("statistics_selection_group_mismatch")
                if config["selection_group"] == "quarterly" and pd.Timestamp(
                    selection["cutoff"]
                ) >= pd.Timestamp(period.start):
                    raise ValueError("statistics_selection_after_prediction_start")
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
                    or family not in wanted
                    or (not direct and wanted[family] != metadata["model_id"])
                ):
                    raise ValueError("statistics_prediction_family_mismatch")
                facts = metadata["model_facts"]
                market = facts["market_id"] if market is None else market
                validate_input_market(facts, market)
                if facts["family_id"] != family:
                    raise ValueError("statistics_prediction_family_mismatch")
                if selection is not None:
                    from quant_research_platform.contracts.models import (
                        validate_model_facts,
                    )

                    selected = next(
                        candidate
                        for candidate in selection["candidates"]
                        if candidate["model_id"] == wanted[family]
                    )
                    validate_model_facts(facts, expected=selected)
                supplied.add(family)
                model_metadata = prediction_model_metadata(metadata)
                if (
                    selection is not None
                    and model_metadata["target_column"] != selection["target_column"]
                ):
                    raise ValueError("statistics_prediction_target_mismatch")
                entries[period_id, family] = {
                    "model_metadata": model_metadata,
                    "model_id": metadata["model_id"],
                    "model_facts": facts,
                    "asset_ids": metadata.get("asset_ids"),
                }
                with closing_iterator(
                    _iter_prediction_member(
                        Path(path),
                        metadata,
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
                family
                for family, winner in wanted.items()
                if direct or winner is not None
            }:
                raise ValueError("statistics_prediction_family_coverage_missing")
        if paired:
            reference_index = ExactPrimaryKeyIndex(
                primary_key=("timestamp", "asset_id", "family_id")
            )
            reference_lookup = _PredictionLookup(
                index=reference_index,
                primary_key=("timestamp", "asset_id", "family_id"),
                prediction_column="prediction_score",
            )
            for path in config["global_predictions"]:
                metadata = validate_output_bundle(
                    Path(path), expected_definition_id="prediction_outputs"
                )[0]["metadata"]
                family = metadata["family_id"]
                if family not in family_ids:
                    raise ValueError("statistics_prediction_family_mismatch")
                facts = metadata["model_facts"]
                validate_input_market(facts, market)
                if facts["family_id"] != family:
                    raise ValueError("statistics_prediction_family_mismatch")
                cutoff = facts.get("selection_cutoff")
                with closing_iterator(
                    _iter_prediction_member(
                        Path(path),
                        metadata,
                        columns=("timestamp", "asset_id", "prediction_score"),
                        time_window=window,
                    )
                ) as frames:
                    for frame in frames:
                        if not frame.empty:
                            if cutoff is None:
                                reference_blocker.setdefault(
                                    family, "selection_chronology_not_established"
                                )
                            elif pd.Timestamp(cutoff) >= pd.Timestamp(
                                frame.timestamp.min()
                            ):
                                reference_blocker[family] = (
                                    "selection_reuses_evaluation_window"
                                )
                        frame = frame.copy()
                        frame["family_id"] = family
                        duplicates = reference_index.add_float_value_table(
                            pa.Table.from_pandas(frame, preserve_index=False),
                            namespace="prediction",
                            value_column="prediction_score",
                            validation_column="prediction_score",
                            retain_values=True,
                        )
                        if duplicates:
                            raise ValueError("statistics_prediction_keys_duplicate")
                        reference_lookup.row_count += len(frame)
        session, spec = _statistics_input_session(config, window)
        validate_input_market(session.spec, market)
        tables = build_a_share_statistics_tables(
            dataset_path=session.root,
            dataset_session=session,
            dataset_spec=spec,
            prediction_id=config.get("prediction_id"),
            model_entries=entries,
            selections=selections,
            period_windows=periods,
            trial_rows=(
                _selection_trial_rows(selections, config["selection_group"])
                if selections
                else pd.DataFrame()
            ),
            config=config,
            time_window=window,
            prediction_lookup=lookup,
            global_prediction_lookup=reference_lookup,
            global_inference_blocker=reference_blocker,
        )
    finally:
        lookup.close()
        if reference_lookup is not None:
            reference_lookup.close()
    return write_statistics_artifact(
        output,
        metadata={
            **{key: value for key, value in config.items() if key != "output"},
            **_model_target_metadata(config, spec, window),
            "inference_scope": "exploratory",
            "formal_selection_qualification": "not_established",
            "selection_cutoffs": sorted(
                {
                    str(cutoff)
                    for cutoff in (
                        *(selection.get("cutoff") for selection in selections.values()),
                        *(
                            (entry.get("model_facts") or {}).get("selection_cutoff")
                            for entry in entries.values()
                        ),
                    )
                    if cutoff is not None
                }
            ),
        },
        tables=tables,
    )

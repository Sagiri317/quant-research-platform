"""Production A-share statistics artifact producer."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import ExitStack

import numpy as np
import pandas as pd

from quant_research_platform.catalog.datasets import ResearchDatasetCatalog
from quant_research_platform.contracts.experiments import legal_training_rows
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.research.datasets.io import iter_research_dataset_chunks
from quant_research_platform.research.features.transforms import (
    model_input_validity_columns,
)
from quant_research_platform.research.protocols.a_share import (
    analysis_windows,
    label_catalog,
)
from quant_research_platform.research.randomness import derive_seed
from quant_research_platform.research.statistics import (
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
from quant_research_platform.runtime.resources import (
    OwnedValue,
    RuntimeResourcePolicy,
    current_runtime_resource_policy,
    pandas_frame_payload_bytes,
)
from quant_research_platform.storage.experiments import (
    read_model_trial_ledger_reference,
)
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
    resource_policy: RuntimeResourcePolicy,
) -> OwnedValue[dict[str, pd.DataFrame]]:
    dataset_spec = dataset_catalog.get(dataset_id)
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
        or type(root_seed) is not int
    ):
        raise ValueError("statistics_protocol_invalid")
    max_lags = statistics.get("hac_lag")
    block_size = statistics.get("bootstrap_block")
    repetitions = statistics.get("bootstrap_repetitions")
    family_specs = statistics.get("factor_fdr_families")
    core_factors = statistics.get("fama_macbeth_core_factors")
    if (
        type(max_lags) is not int
        or max_lags < 0
        or type(block_size) is not int
        or block_size <= 0
        or type(repetitions) is not int
        or repetitions <= 0
        or type(family_specs) is not dict
        or type(core_factors) is not list
        or any(type(value) is not str for value in core_factors)
    ):
        raise ValueError("statistics_protocol_invalid")
    fdr_methods = (
        _fdr_method(statistics.get("fdr_primary"), expected="BH"),
        _fdr_method(statistics.get("fdr_robustness"), expected="BY"),
    )
    factors = tuple(
        protocol_manifest["research_configuration"]["evaluation"]["factors"]
    )
    families = _factor_families(family_specs, factors)
    core = tuple(core_factors)
    if any(factor not in factors for factor in core):
        raise ValueError("statistics_core_factors_missing")
    factor_window, model_window = analysis_windows(protocol_manifest, time_window)
    model_entries = {
        (item["quarter_id"], item["family_id"]): item
        for item in persisted_model.prediction_models(
            model_window, access_window=protocol_manifest["access_window"]
        )
    }
    family_ids = tuple(
        persisted_model.quarterly_selections[
            next(iter(persisted_model.quarterly_selections))
        ]["tie_break_order"]
    )
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
                *factors,
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
    calendar_days = (
        pd.Timestamp(factor_window.end) - pd.Timestamp(factor_window.start)
    ).days + 1
    accumulator_bound = max(
        1,
        calendar_days
        * max(1, 3 * len(factors) + len(core) + 8 * len(family_ids) + 8)
        * 1024,
    )
    rank_records: list[dict[str, object]] = []
    comparison_records: list[dict[str, object]] = []
    loss_records: list[dict[str, object]] = []
    fmb_period_records: list[dict[str, object]] = []
    with ExitStack() as ownership:
        accumulator_lease = ownership.enter_context(
            resource_policy.lease("research.statistics.accumulators", accumulator_bound)
        )
        chunks = iter_complete_session_frames(
            iter_research_dataset_chunks(
                layout,
                market_id=dataset_spec.market_id,
                dataset_id=dataset_id,
                time_window=factor_window,
                columns=read_columns,
                access_window=protocol_manifest["access_window"],
            ),
            timestamp_column="timestamp",
            resource_policy=resource_policy,
        )
        with closing_iterator(chunks):
            for session in chunks:
                with resource_policy.lease(
                    "research.statistics.session",
                    max(
                        1,
                        5 * pandas_frame_payload_bytes(session)
                        + len(session) * (256 + 48 * len(family_ids)),
                    ),
                ):
                    mature = keys = scores = values = found = rank = daily = (
                        eligible
                    ) = None
                    try:
                        mature = legal_training_rows(
                            session,
                            fitting_cutoff=factor_window.end,
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
                            source_quarter = model_entries[(quarter_id, family_ids[0])][
                                "source_quarter"
                            ]
                            selection = persisted_model.quarterly_selections[
                                source_quarter
                            ]
                            for family_id in family_ids:
                                entry = model_entries[(quarter_id, family_id)]
                                metadata = entry["model_metadata"]
                                inputs = model_input_validity_columns(
                                    metadata["input_columns"],
                                    metadata["model_algorithm"],
                                )
                                eligible = session["eligible"].eq(True) & session.loc[
                                    :, list(inputs)
                                ].notna().all(axis=1)
                                keys = session.loc[
                                    eligible, ["timestamp", "asset_id"]
                                ].copy()
                                keys["family_id"] = family_id
                                with prediction_lookup.values_for_frame(
                                    keys, mark_matched=True
                                ) as (values, found):
                                    if not all(found):
                                        raise ValueError(
                                            "missing_required_prediction_keys"
                                        )
                                    scores = pd.Series(
                                        np.asarray(values, dtype="float64"),
                                        index=keys.index,
                                    )
                                mature[family_id] = scores.reindex(mature.index)
                            if mature.empty:
                                continue
                            mature[_PREDICTION_COLUMN] = mature[
                                selection["selected_family_id"]
                            ]
                            rank = cross_sectional_rank_ic(
                                mature,
                                value_columns=(
                                    *factors,
                                    _PREDICTION_COLUMN,
                                    *family_ids,
                                ),
                                label_column=primary_label,
                            )
                            rank_records.extend(rank.to_dict("records"))
                            daily = rank.iloc[0]
                            paired_rank, paired_count = _common_model_rank_ic(
                                mature,
                                family_ids=family_ids,
                                label_column=primary_label,
                            )
                            comparison_records.extend(paired_rank.to_dict("records"))
                            common = bool(
                                np.isfinite(
                                    paired_rank.loc[:, list(family_ids)].to_numpy(
                                        dtype="float64"
                                    )
                                ).all()
                            )
                            for family_id in family_ids:
                                value = daily[family_id]
                                defined = bool(np.isfinite(value))
                                loss_records.append(
                                    {
                                        "timestamp": timestamp,
                                        "access_window": protocol_manifest[
                                            "access_window"
                                        ],
                                        "quarter_id": quarter_id,
                                        "family_id": family_id,
                                        "config_id": model_entries[
                                            (quarter_id, family_id)
                                        ]["model_id"],
                                        "loss": -float(value) if defined else np.nan,
                                        "observation_count": int(
                                            mature[[family_id, primary_label]]
                                            .dropna()
                                            .shape[0]
                                        ),
                                        "status": "DEFINED" if defined else "UNDEFINED",
                                        "reason": None
                                        if defined
                                        else "cross_sectional_rankic_undefined",
                                        "common_date": int(common),
                                        "common_observation_count": paired_count,
                                    }
                                )
                        else:
                            if mature.empty:
                                continue
                            rank = cross_sectional_rank_ic(
                                mature,
                                value_columns=factors,
                                label_column=primary_label,
                            )
                            rank_records.extend(rank.to_dict("records"))
                        fmb_period_records.extend(
                            _fama_macbeth_period_records(
                                mature,
                                return_column=primary_label,
                                factors=factors,
                                core_factors=core,
                            )
                        )
                    finally:
                        mature = keys = scores = values = found = rank = daily = (
                            eligible
                        ) = session = None

        if prediction_lookup.coverage_counts() != (0, 0):
            raise ValueError("prediction_dataset_coverage_mismatch")

        rank_by_date = pd.DataFrame.from_records(
            rank_records,
            columns=["timestamp", *factors, _PREDICTION_COLUMN, *family_ids],
        )
        factor_summary = _factor_summary(
            rank_by_date,
            factors=factors,
            families=families,
            methods=fdr_methods
            if protocol_manifest["access_window"] == "development"
            else (None,),
            max_lags=max_lags,
        )
        prediction_summary = _prediction_summary(
            rank_by_date,
            prediction_id=prediction_id,
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
        model_trials = _model_trials(
            persisted_model, layout, output_lease=accumulator_lease
        )
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
            ),
        )
        comparison_by_date = pd.DataFrame.from_records(
            comparison_records, columns=["timestamp", *family_ids]
        )
        common_count = int(
            comparison_by_date.loc[:, list(family_ids)].notna().all(axis=1).sum()
        )
        bootstrap_workspace = (
            64 * repetitions * len(family_ids) ** 2
            + 256 * len(rank_by_date) * len(family_ids)
            if common_count > block_size
            else 65536
        )
        with resource_policy.lease(
            "research.statistics.bootstrap_workspace", bootstrap_workspace
        ):
            comparison, tests, robustness = _model_comparison(
                comparison_by_date,
                family_ids=family_ids,
                benchmark_id=benchmark_id,
                access_window=protocol_manifest["access_window"],
                max_lags=max_lags,
                block_size=block_size,
                repetitions=repetitions,
                root_seed=root_seed,
                dataset_id=dataset_id,
            )
        fmb_periods = pd.DataFrame.from_records(
            fmb_period_records, columns=_FAMA_MACBETH_PERIOD_COLUMNS
        )
        del rank_records, comparison_records, fmb_period_records, loss_records
        fmb_summary = _fama_macbeth_summary(
            fmb_periods,
            factors=factors,
            core_factors=core,
            max_lags=max_lags,
        )
        tables = {
            "factor_diagnostics": factor_summary,
            "prediction_diagnostics": prediction_summary,
            "model_trials": model_trials,
            "robustness": robustness,
            "fama_macbeth": fmb_summary,
            "fama_macbeth_periods": fmb_periods,
            "model_losses": model_losses,
            "model_comparison": comparison,
            "model_reality_checks": tests,
        }
        output_bytes = sum(
            pandas_frame_payload_bytes(table) for table in tables.values()
        )
        if output_bytes > accumulator_lease.reserved_bytes:
            raise ValueError("statistics_accumulator_bound_invalid")
        accumulator_lease.shrink_to(max(1, output_bytes))
        accumulator_lease.transfer("research.statistics.output")
        ownership.pop_all()
        return OwnedValue(tables, accumulator_lease)


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
            or not raw_factors
            or any(type(factor) is not str for factor in raw_factors)
            or len(set(raw_factors)) != len(raw_factors)
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
        for method in methods:
            part = time_series_inference_summary(
                family_frame,
                identity_column="timestamp",
                max_lags=max_lags,
                fdr_method=method,
                fdr_alpha=0.05,
                fdr_family_size=len(planned_factors),
            ).rename(columns={"timestamp": "factor_id"})
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
    if not parts:
        return _undefined_factor_table()
    return pd.concat(parts, ignore_index=True)


def _prediction_summary(
    rank_by_date: pd.DataFrame,
    *,
    prediction_id: str,
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
        "reason": "hac_lag_exceeds_observations",
        "mean": np.nan,
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
    if count < max(2, max_lags + 1):
        return pd.DataFrame([row])
    try:
        inference = hac_mean(values, max_lags=max_lags)
        bootstrap = stationary_bootstrap_mean_ci(
            values,
            block_size=block_size,
            repetitions=repetitions,
            seed=seed,
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


def _model_trials(
    persisted_model: object, layout: StorageLayout, *, output_lease
) -> pd.DataFrame:
    groups = []
    pending = {}
    for quarter_id, selection in persisted_model.quarterly_selections.items():
        if type(quarter_id) is not str or type(selection) is not dict:
            raise ValueError("statistics_model_selection_metadata_missing")
        for family in selection.get("families", ()):
            if type(family) is not dict or type(family.get("family_id")) is not str:
                raise ValueError("statistics_model_family_metadata_invalid")
            identity = family["trial_ledger_semantic_identity"]
            if identity in pending:
                raise ValueError("statistics_trial_reference_mismatch")
            rows = []
            groups.append(rows)
            pending[identity] = (
                quarter_id,
                selection.get("selected_family_id"),
                family,
                rows,
            )
    for reference in persisted_model.trial_ledgers:
        with read_model_trial_ledger_reference(layout, reference) as (plan, records):
            if plan.stage != "selection":
                continue
            identity = reference["semantic_identity"]
            if identity not in pending:
                raise ValueError("statistics_trial_reference_mismatch")
            quarter_id, selected_family, family, rows = pending.pop(identity)
            family_id = family["family_id"]
            selected_config = family.get("selected_config_id")
            if plan.candidate_family_id != family_id:
                raise ValueError("statistics_trial_reference_mismatch")
            for record in records:
                config_id = record.candidate_id
                value = record.metrics["validation_rankic"]
                defined = type(value) in {int, float} and np.isfinite(value)
                # Nine scalar columns, row dictionaries, and DataFrame construction.
                row_bytes = 4096 + 4 * sum(
                    len(value) for value in (quarter_id, family_id, config_id)
                )
                current_runtime_resource_policy().require_working_set(
                    row_bytes, error="statistics_accumulator_bound_invalid"
                )
                output_lease.grow(row_bytes)
                rows.append(
                    {
                        "quarter_id": quarter_id,
                        "family_id": family_id,
                        "config_id": config_id,
                        "metric_id": "validation_rankic",
                        "value": float(value) if defined else np.nan,
                        "observation_count": record.metrics["validation_rows"],
                        "status": "DEFINED" if defined else "UNDEFINED",
                        "reason": None
                        if defined
                        else record.metrics["validation_rankic_reason"],
                        "selection_status": (
                            "CHAMPION"
                            if family_id == selected_family
                            and config_id == selected_config
                            else "FAMILY_WINNER"
                            if config_id == selected_config
                            else "TRIAL"
                        ),
                    }
                )
    if pending:
        raise ValueError("statistics_trial_reference_mismatch")
    if not any(groups):
        raise ValueError("statistics_model_trials_empty")
    return pd.DataFrame(row for group in groups for row in group)


def _common_model_rank_ic(frame, *, family_ids, label_column):
    """Compare predictions only on the exact intersection of valid OOS rows."""
    columns = [*family_ids, label_column]
    valid = np.isfinite(frame.loc[:, columns].to_numpy(dtype="float64")).all(axis=1)
    shared = frame.loc[valid]
    if shared.empty:
        return pd.DataFrame(
            [
                {
                    "timestamp": frame["timestamp"].iloc[0],
                    **dict.fromkeys(family_ids, np.nan),
                }
            ]
        ), 0
    return cross_sectional_rank_ic(
        shared, value_columns=family_ids, label_column=label_column
    ), len(shared)


def _model_comparison(
    rank_by_date,
    *,
    family_ids,
    benchmark_id,
    access_window,
    max_lags,
    block_size,
    repetitions,
    root_seed,
    dataset_id,
):
    common = rank_by_date.set_index("timestamp").loc[:, list(family_ids)].dropna()
    losses = -common
    challengers = tuple(family for family in family_ids if family != benchmark_id)
    comparisons = []
    for family in challengers:
        difference = common[family] - common[benchmark_id]
        summary = _prediction_summary(
            pd.DataFrame({_PREDICTION_COLUMN: difference}),
            prediction_id=family,
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
    for test_id in _MODEL_ROBUSTNESS_TESTS:
        seed = derive_seed(
            root_seed, "a_share_model_reality_check", dataset_id, access_window, test_id
        )
        result = None
        if access_window == "development" and challengers and len(common) > block_size:
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
    if comparisons:
        comparison_table = pd.concat(comparisons, ignore_index=True)
    else:
        comparison_table = _prediction_summary(
            pd.DataFrame(),
            prediction_id=benchmark_id,
            max_lags=max_lags,
            block_size=block_size,
            repetitions=repetitions,
            seed=root_seed,
        ).iloc[:0]
        comparison_table.insert(0, "access_window", pd.Series(dtype="str"))
        comparison_table.insert(1, "benchmark_id", pd.Series(dtype="str"))
    return (
        comparison_table,
        pd.DataFrame(test_rows),
        pd.DataFrame(robustness_rows),
    )


def _fama_macbeth_period_records(
    frame: pd.DataFrame,
    *,
    return_column: str,
    factors: tuple[str, ...],
    core_factors: tuple[str, ...],
) -> list[dict[str, object]]:
    timestamp = pd.Timestamp(frame["timestamp"].iloc[0])
    definitions = tuple((f"single:{factor}", (factor,)) for factor in factors)
    if core_factors:
        definitions += (("core_multifactor", core_factors),)
    rows: list[dict[str, object]] = []
    for regression_id, regression_factors in definitions:
        coefficient_ids = ("intercept", *regression_factors)
        complete_count = len(
            frame.loc[:, [return_column, *regression_factors]].dropna()
        )
        try:
            result = fama_macbeth_period(
                frame,
                return_column=return_column,
                factor_columns=regression_factors,
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
                "white_p_value": (
                    result.white_p_value if result.white_p_value is not None else np.nan
                ),
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
        (f"single:{factor}", coefficient)
        for factor in factors
        for coefficient in ("intercept", factor)
    ]
    definitions.extend(
        ("core_multifactor", coefficient)
        for coefficient in ("intercept", *core_factors)
    )
    for regression_id, coefficient_id in definitions:
        group = periods.loc[
            periods["regression_id"].eq(regression_id)
            & periods["coefficient_id"].eq(coefficient_id)
        ]
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


def _undefined_factor_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "family_id": "none",
                "factor_id": "none",
                "observation_count": 0,
                "status": "UNDEFINED",
                "reason": "no_factor_columns",
                "mean": np.nan,
                "standard_error": np.nan,
                "t_statistic": np.nan,
                "p_value": np.nan,
                "ci_lower": np.nan,
                "ci_upper": np.nan,
                "fdr_method": "NOT_APPLICABLE",
                "fdr_alpha": np.nan,
                "fdr_adjusted_p_value": np.nan,
                "planned_hypothesis_count": 0,
                "defined_hypothesis_count": 0,
                "missing_hypothesis_count": 0,
            }
        ]
    )

"""Compute factor inference and diagnostics on declared mature samples.

Read explicit datasets or saved evaluations and publish native statistics tables,
including missing-sample and undefined-inference outcomes."""

from __future__ import annotations

from collections.abc import Mapping
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from quant_research_platform.compute.dataframe.streaming import (
    iter_complete_session_frames,
)
from quant_research_platform.contracts.evaluations import CORRELATION_METRICS
from quant_research_platform.contracts.experiments import (
    legal_training_rows,
)
from quant_research_platform.contracts.signals import resolve_top_quantile
from quant_research_platform.contracts.time import (
    TimeWindow,
)
from quant_research_platform.contracts.validation import strict_real_numeric_series
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.research.datasets.statistics import (
    _statistics_input_session,
    _statistics_label_availability,
    _statistics_window_bounds,
)
from quant_research_platform.research.protocols.a_share import (
    _inference_blocker,
)
from quant_research_platform.research.statistics import (
    _apply_fdr,
    _selected_analyses,
    common_correlation_mask,
    cross_sectional_correlations,
    daily_mean_frame,
    fama_macbeth_period,
    hac_mean,
    time_series_inference_summary,
)
from quant_research_platform.storage.datasets.reading import (
    DatasetSession,
    iter_dataset_time_chunks,
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


def _factor_analyses(config, *, saved_evaluation=False):
    defaults = ("rank_ic", "stratified_ic") if saved_evaluation else ("rank_ic",)
    return _selected_analyses(
        {"analyses": config.get("analyses", list(defaults))},
        (
            *defaults,
            "annual_stability",
            *(("fama_macbeth", "factor_correlation") if not saved_evaluation else ()),
        ),
    )


def build_a_share_factor_statistics_tables(
    *,
    dataset_session: DatasetSession,
    config: Mapping[str, object],
    time_window: TimeWindow,
) -> dict[str, pd.DataFrame]:
    """Publish factor inference once, independently of fitted model artifacts."""
    spec = dataset_session.spec
    evaluation = config["evaluation"]
    factors = tuple(evaluation["factors"])
    analyses = _factor_analyses(config)
    top_quantile = resolve_top_quantile(evaluation.get("top_quantile"))
    metrics = CORRELATION_METRICS if "analyses" not in config else ("rankic",)
    uses_label = bool(set(analyses) - {"factor_correlation"})
    primary_label = evaluation["primary_label"] if uses_label else None
    availability = (
        _statistics_label_availability(evaluation, spec) if uses_label else None
    )
    if not uses_label and any(
        str(spec["column_roles"].get(factor)) not in {"factor", "feature"}
        for factor in factors
    ):
        raise ValueError("statistics_factor_role_invalid")
    statistics = config["statistics"]
    families = (
        _factor_families(statistics["factor_fdr_families"], factors)
        if {"rank_ic", "annual_stability"}.intersection(analyses)
        else {}
    )
    core = (
        tuple(statistics["fama_macbeth_core_factors"])
        if "fama_macbeth" in analyses
        else ()
    )
    if any(factor not in factors for factor in core):
        raise ValueError("statistics_core_factors_missing")
    max_lags = statistics.get("hac_lag", 0)
    methods = (
        (
            _fdr_method(statistics["fdr_primary"], expected="BY"),
            _fdr_method(statistics["fdr_robustness"], expected="BH"),
        )
        if {"rank_ic", "annual_stability"}.intersection(analyses)
        else ()
    )
    window = time_window
    columns = tuple(
        dict.fromkeys(
            (
                *spec["primary_key"],
                "eligible",
                *((primary_label, availability) if uses_label else ()),
                *factors,
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
        *((primary_label,) if uses_label else ()),
        *factors,
        *(("total_market_cap",) if "fama_macbeth" in analyses else ()),
    }
    if business_columns - set(dataset_session.logical_columns):
        raise ValueError("statistics_required_columns_excluded")
    ranks, periods, input_dates = [], [], []
    correlations = {
        method: np.zeros((4, len(factors), len(factors)), dtype="float64")
        for method in ("pearson", "spearman")
        if "factor_correlation" in analyses
    }
    chunks = iter_dataset_time_chunks(dataset_session.projected_view(columns))
    with closing_iterator(chunks):
        for batch in chunks:
            input_dates.extend(map(pd.Timestamp, batch["timestamp"].drop_duplicates()))
            if correlations:
                _update_factor_correlations(
                    batch.loc[batch.eligible.eq(True)], factors, correlations
                )
            if not uses_label:
                continue
            mature = legal_training_rows(
                batch,
                fitting_cutoff=cutoff,
                availability_column=availability,
                target_column=primary_label,
            )
            if {"rank_ic", "annual_stability"}.intersection(analyses):
                candidates = batch.loc[batch.eligible.eq(True)].copy()
                labels = candidates[primary_label]
                if pd.api.types.is_integer_dtype(labels.dtype):
                    labels = labels.astype(
                        "UInt64"
                        if pd.api.types.is_unsigned_integer_dtype(labels.dtype)
                        else "Int64"
                    )
                candidates[primary_label] = labels.where(
                    candidates.index.isin(mature.index)
                )
                ranks.extend(
                    cross_sectional_correlations(
                        candidates,
                        value_columns=factors,
                        label_column=primary_label,
                        top_quantile=top_quantile if len(metrics) > 1 else None,
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
                            sample_scope=statistics.get(
                                "fama_macbeth_sample_scope", "own"
                            ),
                        )
                    )
    tables = {}
    if {"rank_ic", "annual_stability"}.intersection(analyses):
        rank_by_date = pd.DataFrame.from_records(
            ranks, columns=["timestamp", "signal", *metrics]
        ).pivot(index="timestamp", columns="signal", values=list(metrics))
        rank_by_date = rank_by_date.reindex(
            index=input_dates,
            columns=pd.MultiIndex.from_product([metrics, factors]),
        )
        rank_by_date.columns = pd.Index(
            rank_by_date.columns.to_list(), tupleize_cols=False
        )
        rank_by_date = (
            daily_mean_frame(rank_by_date, sessions=config.get("sessions"))
            .rename_axis("timestamp")
            .reset_index()
        )
        summary_options = dict(
            factors=factors,
            families=families,
            metrics=metrics,
            methods=methods if _inference_blocker(config) is None else (None,),
            max_lags=max_lags,
        )
        if "rank_ic" in analyses:
            tables["factor_diagnostics"] = _factor_correlations_summary(
                rank_by_date, **summary_options
            )
        if "annual_stability" in analyses:
            tables["factor_annual_diagnostics"] = _annual_factor_summary(
                rank_by_date, summary=_factor_correlations_summary, **summary_options
            )
    if correlations:
        tables["factor_correlation"] = _factor_correlation_summary(
            correlations, factors, input_date_count=len(input_dates)
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


def _factor_correlations_summary(
    rank_by_date, *, factors, families, metrics=CORRELATION_METRICS, **options
):
    """Infer all requested correlations in one factor family, retaining old keys."""
    from quant_research_platform.artifacts.statistics import (
        FACTOR_CORRELATION_FIELDS,
        STATISTICS_TABLE_COLUMNS,
    )

    expanded = _factor_summary(
        rank_by_date,
        factors=tuple((metric, factor) for metric in metrics for factor in factors),
        families={
            family: tuple((metric, factor) for metric in metrics for factor in members)
            for family, members in families.items()
        },
        **options,
    )
    expanded["statistic"] = expanded.factor_id.map(lambda value: value[0])
    expanded["factor_id"] = expanded.factor_id.map(lambda value: value[1])
    keys = ["family_id", "factor_id", "fdr_method"]
    result = (
        expanded.loc[expanded.statistic.eq("rankic")]
        .drop(columns="statistic")
        .set_index(keys)
    )
    for metric in metrics[1:]:
        selected = expanded.loc[expanded.statistic.eq(metric)].set_index(keys)
        result = result.join(
            selected.loc[:, list(FACTOR_CORRELATION_FIELDS)].add_prefix(f"{metric}_")
        )
    result = result.reset_index()
    return result.loc[
        :,
        [
            *STATISTICS_TABLE_COLUMNS["factor_diagnostics.parquet"],
            *(
                f"{metric}_{name}"
                for metric in metrics[1:]
                for name in FACTOR_CORRELATION_FIELDS
            ),
        ],
    ]


def _annual_factor_summary(rank_by_date, *, summary=_factor_summary, **summary_options):
    from quant_research_platform.artifacts.statistics import STATISTICS_TABLE_COLUMNS

    parts = [
        summary(group, **summary_options).assign(year=int(year))
        for year, group in rank_by_date.groupby(
            pd.to_datetime(rank_by_date["timestamp"]).dt.year, sort=True
        )
    ]
    return (
        pd.concat(parts, ignore_index=True)
        if parts
        else pd.DataFrame(
            columns=STATISTICS_TABLE_COLUMNS["factor_annual_diagnostics.parquet"]
        )
    )


def _update_factor_correlations(frame, factors, accumulators):
    """Accumulate each pair on its own finite domain, independent of labels."""
    for _, session in frame.groupby("timestamp", sort=False):
        values = session.loc[:, factors]
        present = np.isfinite(values.to_numpy(dtype="float64"))
        values = values.where(present)
        if len(values) < 2:
            continue
        pairs = present.astype("int64").T @ present.astype("int64")
        for method, totals in accumulators.items():
            correlation = values.corr(method=method, min_periods=2).to_numpy()
            valid = np.isfinite(correlation)
            totals[0] += valid
            totals[1] += np.where(valid, correlation, 0.0)
            totals[2] += np.where(valid, np.abs(correlation), 0.0)
            totals[3] += np.where(valid, pairs, 0)


def _factor_correlation_summary(accumulators, factors, *, input_date_count):
    from quant_research_platform.artifacts.statistics import STATISTICS_TABLE_COLUMNS

    rows = []
    for left, right in combinations(range(len(factors)), 2):
        for method, totals in accumulators.items():
            count, total, absolute_total, pairs = totals[:, left, right]
            rows.append(
                {
                    "factor_id": factors[left],
                    "comparison_factor_id": factors[right],
                    "statistic": method,
                    "input_date_count": input_date_count,
                    "observation_count": int(count),
                    "observation_pairs": int(pairs),
                    "status": "DEFINED" if count else "UNDEFINED",
                    "reason": None if count else "no_defined_cross_sections",
                    "mean": float(total / count) if count else np.nan,
                    "mean_absolute": float(absolute_total / count) if count else np.nan,
                }
            )
    return pd.DataFrame.from_records(
        rows, columns=STATISTICS_TABLE_COLUMNS["factor_correlation.parquet"]
    )


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
    sample_scope: str = "own",
) -> list[dict[str, object]]:
    timestamp = pd.Timestamp(frame["timestamp"].iloc[0])
    if sample_scope not in ("own", "common"):
        raise ValueError("fama_macbeth_sample_scope_invalid")
    if sample_scope == "common":
        # The family includes every declared regression and both weightings.
        values = frame.loc[:, [return_column, *factors]].to_numpy(dtype="float64")
        weights = strict_real_numeric_series(
            frame["total_market_cap"],
            error="fama_macbeth_weights_invalid",
            errors="coerce",
        )
        common = np.isfinite(values).all(axis=1) & np.isfinite(weights) & weights.gt(0)
        frame = frame.loc[common]
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
            if frame.empty:
                raise ValueError("fama_macbeth_period_insufficient")
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
        rows.extend(
            {
                "regression_id": regression_id,
                "coefficient_id": coefficient_id,
                "timestamp": timestamp,
                "status": "DEFINED",
                "reason": None,
                "observation_count": result.observation_count,
                "value": float(result.coefficients[coefficient_id]),
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
            "mean": float(np.mean(values)) if count else np.nan,
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
        "sessions",
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
    analyses = _factor_analyses(config)
    if (
        not isinstance(evaluation, dict)
        or set(evaluation)
        - {"factors", "primary_label", "label_availability_columns", "top_quantile"}
        or (
            {"factors", "primary_label"}
            if set(analyses) - {"factor_correlation"}
            else {"factors"}
        )
        - set(evaluation)
    ):
        raise ValueError("factor_statistics_evaluation_invalid")
    evaluation["top_quantile"] = resolve_top_quantile(evaluation.get("top_quantile"))
    expected = {"hac_lag"} if set(analyses) != {"factor_correlation"} else set()
    if {"rank_ic", "annual_stability"}.intersection(analyses):
        if isinstance(policy, dict):
            policy = {"fdr_primary": "BY_5pct", "fdr_robustness": "BH_5pct", **policy}
            config["statistics"] = policy
        expected.update(("factor_fdr_families", "fdr_primary", "fdr_robustness"))
    if "fama_macbeth" in analyses:
        expected.add("fama_macbeth_core_factors")
        if isinstance(policy, dict):
            policy.setdefault("fama_macbeth_sample_scope", "own")
    known = {
        "hac_lag",
        "factor_fdr_families",
        "fdr_primary",
        "fdr_robustness",
        "fama_macbeth_core_factors",
        "fama_macbeth_sample_scope",
    }
    if (
        not isinstance(policy, dict)
        or expected - set(policy)
        or set(policy) - known
        or type(policy.get("hac_lag", 0)) is not int
        or policy.get("hac_lag", 0) < 0
        or policy.get("fama_macbeth_sample_scope", "own") not in ("own", "common")
    ):
        raise ValueError("factor_statistics_policy_invalid")
    output = Path(config["output"])
    reject_if_exists(output)
    window = TimeWindow(**config["window"])
    session = _statistics_input_session(config, window)
    tables = build_a_share_factor_statistics_tables(
        dataset_session=session,
        config=config,
        time_window=window,
    )
    return write_statistics_artifact(
        output,
        metadata={
            **{key: value for key, value in config.items() if key != "output"},
            "inference_scope": "exploratory",
            "formal_selection_qualification": "not_established",
            "correlation_clock": "trading_day",
            "hac_lag_unit": "trading_session",
            "column_units": {
                column: unit
                for column, unit in session.spec["column_units"].items()
                if column in {evaluation.get("primary_label"), *evaluation["factors"]}
            },
        },
        tables=tables,
        table_group="factor",
    )


def _factor_statistics_from_evaluations(config):
    """Reduce published daily facts without reopening their dataset producers."""
    from contextlib import ExitStack

    from quant_research_platform.artifacts.outputs.evaluation import (
        _iter_evaluation_member_batches,
    )
    from quant_research_platform.artifacts.outputs.manifest import (
        read_output_manifest,
    )
    from quant_research_platform.artifacts.statistics import (
        STATISTICS_TABLE_COLUMNS,
        write_statistics_artifact,
    )
    from quant_research_platform.research.statistics import fdr_adjust
    from quant_research_platform.storage.atomic import reject_if_exists

    required = {"evaluation_inputs", "output", "statistics"}
    optional = {"window", "analyses", "sample_role", "evaluation", "sessions"}
    if required - set(config) or set(config) - required - optional:
        raise ValueError("factor_statistics_evaluation_config_invalid")
    paths = config["evaluation_inputs"]
    if (
        type(paths) is not list
        or not paths
        or any(type(path) is not str or not path for path in paths)
    ):
        raise ValueError("factor_statistics_evaluation_inputs_invalid")
    analyses = _factor_analyses(config, saved_evaluation=True)
    if {"rank_ic", "annual_stability"}.intersection(analyses) and len(paths) != 1:
        raise ValueError("factor_statistics_rank_ic_multiple_evaluations")
    policy = config["statistics"]
    needed = {"hac_lag"}
    if isinstance(policy, dict):
        defaults = {}
        if {"rank_ic", "annual_stability"}.intersection(analyses):
            defaults.update(fdr_primary="BY_5pct", fdr_robustness="BH_5pct")
        if "stratified_ic" in analyses:
            defaults.update(fdr_method="BY", fdr_alpha=0.05)
        policy = defaults | policy
        config["statistics"] = policy
    if "stratified_ic" in analyses:
        needed |= {"fdr_method", "fdr_alpha"}
    if {"rank_ic", "annual_stability"}.intersection(analyses):
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
        manifest = read_output_manifest(Path(path))
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
        if "stratified_ic" in analyses:
            current["top_quantile"] = metadata.get("top_quantile")
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
    session_axes = {}
    for path, metadata, window in entries:
        scope = metadata["sample_scope"]
        input_dates[scope] = []
        for signal in metadata["signals"]:
            for label in metadata["label_columns"]:
                series[(signal["signal_id"], label, scope, "all", None)] = {
                    "ic": [],
                    "rankic": [],
                    "pairs": 0,
                    "timestamps": [],
                }
        members = ["metrics"]
        if "stratified_ic" in analyses:
            for signal in metadata["signals"]:
                for label in metadata["label_columns"]:
                    for side in ("long", "short"):
                        series[(signal["signal_id"], label, scope, "side", side)] = {
                            "ic": [],
                            "rankic": [],
                            "pairs": 0,
                            "timestamps": [],
                        }
            if metadata["strata"] or "daily_strata" in metadata["optional_member_rows"]:
                members.append("daily_strata")

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
                input_dates[scope].append(timestamp)
                daily = {}
                for name in heads:
                    head = heads[name]
                    daily[name] = (
                        head
                        if head is not None and head.timestamp.iloc[0] == timestamp
                        else None
                    )
                    if daily[name] is not None:
                        heads[name] = next_date(streams[name])
                domains = [
                    ("metrics", metrics),
                    ("daily_strata", daily.get("daily_strata")),
                ]
                if "stratified_ic" in analyses:
                    for side in ("long", "short"):
                        columns = {
                            f"{side}_{name}": name for name in ("ic", "rankic", "n_obs")
                        }
                        domains.append(
                            (
                                side,
                                metrics.loc[
                                    :, ["timestamp", "signal", "label", *columns]
                                ].rename(columns=columns),
                            )
                        )
                for name, frame in domains:
                    if frame is None:
                        continue
                    if (
                        scope == "common"
                        and metadata.get("correlation_clock") == "trading_day"
                    ):
                        frame = frame.copy()
                        frame[["ic", "rankic"]] = frame[["ic", "rankic"]].where(
                            common_correlation_mask(
                                frame,
                                groups=("timestamp", "label")
                                + (
                                    ("stratification", "stratum")
                                    if name == "daily_strata"
                                    else ()
                                ),
                            )
                        )
                    for row in frame.itertuples(index=False):
                        kind = (
                            "all"
                            if name == "metrics"
                            else "side"
                            if name in {"long", "short"}
                            else row.stratification
                        )
                        stratum = (
                            name
                            if name in {"long", "short"}
                            else None
                            if name == "metrics" or pd.isna(row.stratum)
                            else row.stratum
                        )
                        values = series.setdefault(
                            (row.signal, row.label, scope, kind, stratum),
                            {"ic": [], "rankic": [], "pairs": 0, "timestamps": []},
                        )
                        values["timestamps"].append(timestamp)
                        values["pairs"] += int(row.n_obs)
                        for statistic in ("ic", "rankic"):
                            value = getattr(row, statistic)
                            values[statistic].append(
                                float(value) if pd.notna(value) else np.nan
                            )
        declared_sessions = config.get("sessions", metadata.get("sessions"))
        if declared_sessions is not None:
            from quant_research_platform.contracts.evaluations import (
                evaluation_sessions,
            )

            declared_sessions = evaluation_sessions(declared_sessions)
            low, high = _statistics_window_bounds(TimeWindow(**config["window"]))
            declared_sessions = declared_sessions[
                (declared_sessions >= low.normalize())
                & (declared_sessions <= high.normalize())
            ]
        session_axes[scope] = daily_mean_frame(
            pd.DataFrame(index=pd.DatetimeIndex(input_dates[scope])),
            sessions=declared_sessions,
        ).index
        input_counts[scope] = len(session_axes[scope])
    stratum_domains = {
        kind: sorted(
            {key[4] for key in series if key[3] == kind},
            key=lambda value: (value is not None, value or ""),
        )
        for kind in binding["strata"].get("kinds", [])
        if "stratified_ic" in analyses
    }
    tables = {}
    if "stratified_ic" in analyses:
        # The joint family includes strata absent from an individual scope.
        # Keep them empty: neither observations nor dates may be manufactured.
        for kind, strata in stratum_domains.items():
            for stratum in strata:
                for signal in binding["signals"]:
                    for scope in sorted(scopes):
                        series.setdefault(
                            (
                                signal["signal_id"],
                                binding["strata"]["primary_label"],
                                scope,
                                kind,
                                stratum,
                            ),
                            {"ic": [], "rankic": [], "pairs": 0, "timestamps": []},
                        )
        stratum_domains["side"] = ["long", "short"]
        rows = []
        for key, values in series.items():
            daily = daily_mean_frame(
                pd.DataFrame(
                    {
                        "timestamp": values["timestamps"],
                        "ic": values["ic"],
                        "rankic": values["rankic"],
                    }
                ).set_index("timestamp"),
                sessions=session_axes[key[2]],
            )
            for statistic in ("ic", "rankic"):
                observations = daily[statistic].dropna().tolist()
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
                    observed_date_count=pd.DatetimeIndex(values["timestamps"])
                    .normalize()
                    .nunique(),
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
    if {"rank_ic", "annual_stability"}.intersection(analyses):
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
        metrics = CORRELATION_METRICS if "analyses" not in config else ("rankic",)
        ranks = {}
        for metric in metrics:
            side, _, name = metric.rpartition("_")
            kind, stratum = ("side", side) if side else ("all", None)
            for factor in factors:
                ranks[(metric, factor)] = series[
                    (factor, evaluation["primary_label"], scope, kind, stratum)
                ][name or metric]
        ranks = pd.DataFrame(ranks, index=input_dates[scope])
        ranks.columns = pd.Index(ranks.columns.to_list(), tupleize_cols=False)
        ranks = ranks.rename_axis("timestamp").reset_index()
        ranks = daily_mean_frame(
            ranks.set_index("timestamp"),
            sessions=session_axes[scope],
        ).reset_index()
        summary_options = dict(
            factors=factors,
            metrics=metrics,
            families=_factor_families(policy["factor_fdr_families"], factors),
            methods=(
                _fdr_method(policy["fdr_primary"], expected="BY"),
                _fdr_method(policy["fdr_robustness"], expected="BH"),
            ),
            max_lags=lag,
        )
        if "rank_ic" in analyses:
            tables["factor_diagnostics"] = _factor_correlations_summary(
                ranks, **summary_options
            )
        if "annual_stability" in analyses:
            tables["factor_annual_diagnostics"] = _annual_factor_summary(
                ranks, summary=_factor_correlations_summary, **summary_options
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
                    "sessions": [
                        date.date().isoformat()
                        for date in session_axes[metadata["sample_scope"]]
                    ],
                }
                for _, metadata, _ in entries
            },
            "stratum_domains": stratum_domains,
            "input_date_counts": input_counts,
            "inference_scope": "exploratory",
            "formal_selection_qualification": "not_established",
            "correlation_clock": "trading_day",
            "hac_lag_unit": "trading_session",
            "multiple_testing_scope": (
                "all_defined_signals_labels_strata_scopes_ic_rankic_in_explicit_inputs"
            ),
        },
    )

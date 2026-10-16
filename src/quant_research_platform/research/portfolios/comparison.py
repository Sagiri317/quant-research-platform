"""Explicit same-interval reference comparisons and descriptive market research."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from statistics import mean

import numpy as np
import pandas as pd

from quant_research_platform.contracts.validation import finite_exact_float
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.research.statistics import (
    _return_sharpes,
    fdr_adjust,
    return_standard_deviation,
)
from quant_research_platform.storage.datasets.reading import (
    _dataset_input_options,
    iter_dataset_chunks,
    open_dataset_session,
)

COMPARISON_ANALYSES = (
    "benchmark_comparison",
    "market_exposure",
    "regime_comparison",
    "control_comparison",
)
COMPARISON_TABLES = (
    "portfolio_benchmark_intervals",
    "portfolio_benchmark_metrics",
    "portfolio_market_exposure",
    "portfolio_regime_comparison",
)
IDENTITY_COLUMNS = (
    "candidate_id",
    "execution_mode",
    "episode_id",
    "fold_id",
    "reference_id",
    "reference_kind",
    "sample_role",
)
_FACTS = ("valuation_currency", "timestamp_basis", "return_basis", "cost_basis")
_BASES = {
    "price",
    "gross_total_return",
    "net_total_return",
    "strategy_net",
    "constant_cash",
}
_COSTS = {"frictionless", "execution_costs", "zero_cash"}
_CLOCKS = {"trading_day_open", "trading_day_close", "utc"}


def _nav_return_roundoff(start: object, end: object, returns: object) -> np.ndarray:
    """Enclose NAV-ratio rounding using adjacent float64 operand values.

    One outward ULP encloses each rounded NAV. Outward division/subtraction
    bounds cover both (end / start - 1) and ((end - start) / start).
    """
    before = np.asarray(start, dtype="float64")
    after = np.asarray(end, dtype="float64")
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        lower = np.nextafter(
            np.nextafter(after, -np.inf)
            / np.nextafter(before, np.where(after >= 0, np.inf, -np.inf)),
            -np.inf,
        )
        upper = np.nextafter(
            np.nextafter(after, np.inf)
            / np.nextafter(before, np.where(after >= 0, -np.inf, np.inf)),
            np.inf,
        )
        lower = np.nextafter(lower - 1.0, -np.inf)
        upper = np.nextafter(upper - 1.0, np.inf)
        data = np.asarray(returns, dtype="float64")
        return np.where(
            (before > 0) & np.isfinite(data),
            np.maximum(np.abs(data - lower), np.abs(upper - data)),
            np.inf,
        )


def selected_portfolio_analyses(config, allowed):
    from quant_research_platform.research.statistics import _selected_analyses

    return _selected_analyses({"analyses": ["performance"], **config}, allowed)


def endpoint_return(start, end):
    """A bad endpoint invalidates only intervals that actually depend on it."""
    if pd.isna(start) or pd.isna(end):
        return np.nan, "reference_endpoint_missing"
    if not np.isfinite([start, end]).all() or min(start, end) <= 0:
        return np.nan, "reference_endpoint_invalid"
    value = finite_exact_float(Fraction(float(end)) / Fraction(float(start)) - 1)
    return (np.nan, "nonfinite_result") if value is None else (value, "")


def _descriptor(item):
    if not isinstance(item, dict):
        raise ValueError("comparison_reference_invalid")
    representation = item.get("representation")
    clocks = (
        {"timestamp_column"}
        if representation == "endpoint_level"
        else {"start_column", "end_column"}
    )
    required = {"reference_id", "input", "representation", "value_column", *clocks}
    if (
        representation not in {"endpoint_level", "interval_simple_return"}
        or required - set(item)
        or set(item) - required - set(_FACTS) - {"return_unit"}
        or any(
            type(item[name]) is not str or not item[name].strip()
            for name in required - {"input"}
        )
    ):
        raise ValueError("comparison_reference_invalid")
    _dataset_input_options(item["input"])
    for key, allowed in (
        ("timestamp_basis", _CLOCKS),
        ("return_basis", _BASES),
        ("cost_basis", _COSTS),
    ):
        if key in item and item[key] not in allowed:
            raise ValueError(f"comparison_{key}_invalid")
    if "valuation_currency" in item and (
        type(item["valuation_currency"]) is not str
        or not item["valuation_currency"].isascii()
        or not item["valuation_currency"].isalpha()
        or not item["valuation_currency"].isupper()
    ):
        raise ValueError("comparison_currency_invalid")
    if "return_unit" in item and item["return_unit"] not in {"decimal", "percent"}:
        raise ValueError("comparison_return_unit_invalid")
    if representation == "endpoint_level" and "return_unit" in item:
        raise ValueError("comparison_level_return_unit_invalid")
    return item


def validate_comparison_config(config, analyses):
    """Validate only selected research; unused reference declarations are not read."""
    if not set(analyses).intersection(COMPARISON_ANALYSES):
        return
    for name in ("comparison_benchmarks", "comparison_controls"):
        needed = (
            name == "comparison_controls" and "control_comparison" in analyses
        ) or (
            name == "comparison_benchmarks"
            and bool(set(analyses) & set(COMPARISON_ANALYSES[:3]))
        )
        if not needed:
            continue
        items = config.get(name)
        if not isinstance(items, list) or not items:
            raise ValueError(f"{name}_required")
        ids = [_descriptor(item)["reference_id"] for item in items]
        if len(ids) != len(set(ids)):
            raise ValueError("comparison_reference_ids_duplicate")
    if "market_exposure" in analyses:
        policy = config.get("market_exposure")
        if (
            not isinstance(policy, dict)
            or {"hac_lag", "confidence", "minimum_observations"} - set(policy)
            or set(policy)
            - {"hac_lag", "confidence", "minimum_observations", "risk_free"}
            or type(policy["hac_lag"]) is not int
            or policy["hac_lag"] < 0
            or type(policy["minimum_observations"]) is not int
            or policy["minimum_observations"] < 3
            or type(policy["confidence"]) not in {float, int}
            or not 0 < policy["confidence"] < 1
        ):
            raise ValueError("market_exposure_policy_invalid")
        if "risk_free" in policy:
            _descriptor(policy["risk_free"])
    if "regime_comparison" in analyses:
        policy = config.get("regime_comparison")
        if (
            not isinstance(policy, dict)
            or set(policy) != {"threshold"}
            or type(policy["threshold"]) not in {int, float}
            or not np.isfinite(policy["threshold"])
        ):
            raise ValueError("regime_comparison_policy_invalid")


def _read_reference(item):
    options = _dataset_input_options(item["input"])
    requested = options.pop("columns", None)
    value = item["value_column"]
    time_columns = (
        [item["timestamp_column"]]
        if item["representation"] == "endpoint_level"
        else [item["start_column"], item["end_column"]]
    )
    if requested is not None and not {value, *time_columns} <= set(requested):
        raise ValueError("comparison_input_columns_excluded")
    declared = options.get("spec")
    if declared is not None and not {value, *time_columns} <= set(
        declared.required_columns
    ):
        raise ValueError("comparison_input_columns_missing")
    # Parse the selector before deciding that a missing path is a local data gap.
    try:
        session = open_dataset_session(**options)
    except FileNotFoundError:
        session = None
    spec = (
        session.spec
        if session is not None
        else {
            "column_units": getattr(declared, "column_units", {}),
            "parameters": getattr(declared, "parameters", {}),
        }
    )
    manifest = session.manifest if session is not None else {}
    inherited = {}
    for source in (spec.get("parameters", {}), manifest.get("parameters", {})):
        for key in (*_FACTS, "return_unit"):
            if key in source:
                if key in inherited and inherited[key] != source[key]:
                    raise ValueError("comparison_source_facts_conflict")
                inherited[key] = source[key]
    if item["representation"] == "interval_simple_return":
        unit = spec.get("column_units", {}).get(value)
        if unit is not None:
            if unit not in {"decimal", "percent"}:
                raise ValueError("comparison_return_unit_invalid")
            if "return_unit" in inherited and inherited["return_unit"] != unit:
                raise ValueError("comparison_source_facts_conflict")
            inherited["return_unit"] = unit
    facts = {}
    for key in (*_FACTS, "return_unit"):
        if key in item and key in inherited and item[key] != inherited[key]:
            raise ValueError("comparison_source_facts_conflict")
        facts[key] = inherited.get(key, item.get(key))
    if any(facts[key] is None for key in _FACTS):
        raise ValueError("comparison_source_facts_required")
    if (
        facts["timestamp_basis"] not in _CLOCKS
        or facts["return_basis"] not in _BASES
        or facts["cost_basis"] not in _COSTS
    ):
        raise ValueError("comparison_source_facts_invalid")
    if item["representation"] == "interval_simple_return" and facts[
        "return_unit"
    ] not in {"decimal", "percent"}:
        raise ValueError("comparison_return_unit_required")
    currency = facts["valuation_currency"]
    if (
        type(currency) is not str
        or not currency.isascii()
        or not currency.isalpha()
        or not currency.isupper()
    ):
        raise ValueError("comparison_currency_invalid")
    if facts["return_basis"] == "constant_cash" and facts["cost_basis"] != "zero_cash":
        raise ValueError("comparison_constant_cash_invalid")
    if session is None:
        return pd.DataFrame(), facts, "reference_input_missing"
    columns = tuple(dict.fromkeys((*time_columns, value, *spec["primary_key"])))
    session = replace(session.projected_view(columns), _allow_infinite_columns=(value,))
    with closing_iterator(iter_dataset_chunks(session)) as chunks:
        parts = list(chunks)
    frame = (
        pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)
    )
    if frame.empty:
        return frame, facts, "reference_selection_empty"
    identity = [column for column in spec["primary_key"] if column not in time_columns]
    if identity and len(frame[identity].drop_duplicates()) != 1:
        raise ValueError("comparison_ambiguous_series")
    if frame.duplicated(time_columns).any():
        raise ValueError("comparison_duplicate_intervals")
    for column in time_columns:
        if not pd.api.types.is_datetime64_any_dtype(frame[column]):
            raise ValueError("comparison_timestamp_dtype_invalid")
        frame[column] = pd.to_datetime(frame[column], utc=True).dt.tz_localize(None)
        if frame[column].isna().any():
            raise ValueError("comparison_timestamp_invalid")
        if (
            facts["timestamp_basis"] != "utc"
            and not frame[column].eq(frame[column].dt.normalize()).all()
        ):
            raise ValueError("comparison_trading_day_label_invalid")
    if not pd.api.types.is_numeric_dtype(frame[value]) or pd.api.types.is_bool_dtype(
        frame[value]
    ):
        raise ValueError("comparison_value_dtype_invalid")
    frame = frame.sort_values(time_columns).reset_index(drop=True)
    if len(time_columns) == 2:
        start, end = time_columns
        if (
            not frame[start].lt(frame[end]).all()
            or (
                frame[start].iloc[1:].to_numpy() < frame[end].iloc[:-1].to_numpy()
            ).any()
        ):
            raise ValueError("comparison_intervals_overlap")
        if facts["return_unit"] == "percent":
            frame[value] = frame[value] / 100.0
    if facts["return_basis"] == "constant_cash":
        observed = frame[value].to_numpy(dtype=float)
        valid = np.isfinite(observed)
        if item["representation"] == "endpoint_level":
            valid &= observed > 0
        if facts["cost_basis"] != "zero_cash" or (
            valid.any()
            and (
                np.ptp(observed[valid]) != 0
                if item["representation"] == "endpoint_level"
                else np.any(observed[valid] != 0)
            )
        ):
            raise ValueError("comparison_constant_cash_invalid")
    return frame, facts, ""


def _reference_returns(axis, item, loaded, *, currency, clock):
    frame, facts, missing = loaded
    incompatible = (
        "valuation_currency_mismatch"
        if facts["valuation_currency"] != currency
        else "timestamp_basis_mismatch"
        if facts["timestamp_basis"] != clock
        else ""
    )
    if missing or incompatible:
        return [(np.nan, incompatible or missing)] * len(axis)
    value = item["value_column"]
    if item["representation"] == "endpoint_level":
        levels = frame.set_index(item["timestamp_column"])[value]
        return [
            endpoint_return(levels.get(start, np.nan), levels.get(end, np.nan))
            for start, end in zip(axis.start_at, axis.end_at, strict=True)
        ]
    begin, stop = item["start_column"], item["end_column"]
    by_start = frame.set_index(begin)
    returns = []
    for start, end in zip(axis.start_at, axis.end_at, strict=True):
        cursor, wealth, reason = start, Fraction(1), ""
        while cursor < end:
            if cursor not in by_start.index:
                reason = "reference_interval_missing"
                break
            row = by_start.loc[cursor]
            result, following = row[value], row[stop]
            if following > end:
                reason = "reference_interval_misaligned"
                break
            if (
                pd.isna(result)
                or not np.isfinite(result)
                or (result < -1 and facts["return_basis"] != "strategy_net")
            ):
                reason = "reference_return_invalid"
                break
            wealth *= 1 + Fraction(float(result))
            cursor = following
        result = None if reason else finite_exact_float(wealth - 1)
        returns.append(
            (
                np.nan if result is None else result,
                reason or ("nonfinite_result" if result is None else ""),
            )
        )
    return returns


def _complete_daily(axis, clock):
    if axis.empty:
        return False
    adjacent = np.array_equal(
        axis.start_at.iloc[1:].to_numpy(), axis.end_at.iloc[:-1].to_numpy()
    )
    if clock != "utc":
        return adjacent
    day = pd.Timedelta(days=1)
    return (
        adjacent
        and axis.start_at.eq(axis.start_at.dt.normalize()).all()
        and axis.end_at.sub(axis.start_at).eq(day).all()
    )


def _wealth(returns):
    wealth = Fraction(1)
    result = []
    for value in returns:
        if pd.isna(value) or not np.isfinite(value):
            wealth = None
        if wealth is not None:
            wealth *= 1 + Fraction(float(value))
        converted = None if wealth is None else finite_exact_float(wealth)
        result.append(np.nan if converted is None else converted)
    return np.asarray(result)


def comparison_tables(axis, *, config, analyses, currencies, clock, metadata):
    """Consume ledger-owned net interval returns without redoing its accounting."""
    from quant_research_platform.artifacts.statistics import STATISTICS_TABLE_COLUMNS

    chosen = set(analyses) & set(COMPARISON_ANALYSES)
    if not chosen:
        return {}
    validate_comparison_config(config, analyses)
    references = (
        [("benchmark", item) for item in config.get("comparison_benchmarks", [])]
        if chosen & set(COMPARISON_ANALYSES[:3])
        else []
    )
    if "control_comparison" in chosen:
        references += [("control", item) for item in config["comparison_controls"]]
    ids = [item["reference_id"] for _, item in references]
    if len(ids) != len(set(ids)):
        raise ValueError("comparison_reference_ids_duplicate")
    loaded = {item["reference_id"]: _read_reference(item) for _, item in references}
    rf_item = (
        config.get("market_exposure", {}).get("risk_free")
        if "market_exposure" in chosen
        else None
    )
    rf = _read_reference(rf_item) if rf_item is not None else None
    if rf is not None and rf[1]["return_basis"] == "constant_cash":
        raise ValueError("market_exposure_cash_is_not_risk_free")
    metadata["comparison_method"] = {
        "analysis_scope": "exploratory_descriptive",
        "causal_identification": "not_established",
        "portfolio_return_basis": "strategy_net",
        "portfolio_cost_basis": "execution_costs",
        "portfolio_timestamp_basis": clock,
        "control_design_matching": "not_established",
        "reference_facts": {
            key: facts for key, (_data, facts, _reason) in loaded.items()
        },
        "risk_free_facts": None if rf is None else rf[1],
        "exposure_hypothesis_family": (
            "all_declared_candidate_reference_episode_fold_intercepts_and_betas"
        ),
        "exposure_fdr_method": "BY",
        "annualization": "complete_daily_axis_only",
        "annualization_periods": config.get("statistics", {}).get(
            "annualization_periods", 365.0
        )
        if clock == "utc"
        else 252.0,
        "annualization_minimum_observations": config.get("statistics", {}).get(
            "annualization_minimum_observations", 2
        ),
        "missing_policy": "no_fill_complete_path_required",
        "regime_inference": "descriptive_only_no_compressed_hac",
        "regime_label_timing": "ex_post_descriptive",
    }
    records = {name: [] for name in COMPARISON_TABLES}
    group_columns = [
        "candidate_id",
        "execution_mode",
        "episode_id",
        "fold_id",
        "sample_role",
    ]
    for keys, selected in axis.groupby(group_columns, sort=False, dropna=False):
        selected = selected.sort_values(["start_at", "end_at"]).reset_index(drop=True)
        if (
            selected.duplicated(["start_at", "end_at"]).any()
            or (
                selected.start_at.iloc[1:].to_numpy()
                < selected.end_at.iloc[:-1].to_numpy()
            ).any()
        ):
            raise ValueError("comparison_portfolio_intervals_invalid")
        identity = dict(zip(group_columns, keys, strict=True))
        currency = currencies[identity["candidate_id"]]
        daily = _complete_daily(selected, clock)
        for kind, item in references:
            identity = {
                **identity,
                "reference_id": item["reference_id"],
                "reference_kind": kind,
            }
            facts = loaded[item["reference_id"]][1]
            if kind == "control" and (
                facts["return_basis"] != "strategy_net"
                or facts["cost_basis"] != "execution_costs"
            ):
                raise ValueError("comparison_control_basis_invalid")
            aligned = _reference_returns(
                selected,
                item,
                loaded[item["reference_id"]],
                currency=currency,
                clock=clock,
            )
            rp = selected.portfolio_return.to_numpy(dtype=float)
            rb = np.array([value for value, _reason in aligned])
            wp, wb = _wealth(rp), _wealth(rb)
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                differences = rp - rb
                relative = np.where(wb > 0, wp / wb, np.nan)
            differences[~np.isfinite(differences)] = np.nan
            relative[~np.isfinite(relative)] = np.nan
            for index, row in selected.iterrows():
                reference_reason = aligned[index][1]
                reason = (
                    row.portfolio_reason
                    or reference_reason
                    or (
                        "nonfinite_result"
                        if not np.isfinite(differences[index])
                        else ""
                    )
                )
                records["portfolio_benchmark_intervals"].append(
                    {
                        **identity,
                        "start_at": row.start_at,
                        "end_at": row.end_at,
                        "portfolio_return": rp[index],
                        "benchmark_return": rb[index],
                        "return_difference": differences[index],
                        "portfolio_wealth": wp[index],
                        "benchmark_wealth": wb[index],
                        "relative_wealth": relative[index],
                        "portfolio_status": "DEFINED"
                        if np.isfinite(rp[index])
                        else "UNDEFINED",
                        "portfolio_reason": row.portfolio_reason,
                        "benchmark_status": "DEFINED"
                        if not reference_reason
                        else "NOT_APPLICABLE"
                        if reference_reason.endswith("mismatch")
                        else "UNDEFINED",
                        "benchmark_reason": reference_reason,
                        "status": "DEFINED"
                        if not reason
                        else "NOT_APPLICABLE"
                        if reason.endswith("mismatch")
                        else "UNDEFINED",
                        "reason": reason,
                    }
                )
            paired = np.isfinite(differences)
            n, planned = int(paired.sum()), len(selected)
            complete = n == planned and planned > 0
            common_reason = next(
                (
                    row.portfolio_reason or aligned[i][1]
                    for i, row in selected.iterrows()
                    if row.portfolio_reason or aligned[i][1]
                ),
                "incomplete_paired_path",
            )
            values = {
                "portfolio_total_return": wp[-1] - 1,
                "benchmark_total_return": wb[-1] - 1,
                "mean_return_difference": float(mean(differences))
                if complete
                else np.nan,
                "relative_wealth": relative[-1],
                "relative_wealth_return": relative[-1] - 1,
                "maximum_relative_drawdown": float(
                    np.min(
                        relative / np.maximum.accumulate(np.r_[1.0, relative])[1:] - 1
                    )
                )
                if np.isfinite(relative).all()
                else np.nan,
            }
            periods = (
                config.get("statistics", {}).get("annualization_periods", 365.0)
                if clock == "utc"
                else 252.0
            )
            minimum = config.get("statistics", {}).get(
                "annualization_minimum_observations", 2
            )
            annual_reason = (
                "incomplete_daily_axis"
                if not daily
                else "insufficient_complete_daily_observations"
                if planned < minimum
                else ""
            )
            roundoff = np.zeros(planned)
            if {"start_nav", "end_nav"} <= set(selected):
                roundoff += _nav_return_roundoff(
                    selected.start_nav, selected.end_nav, rp
                )
            if item["representation"] == "endpoint_level" and complete:
                levels = loaded[item["reference_id"]][0].set_index(
                    item["timestamp_column"]
                )[item["value_column"]]
                roundoff += _nav_return_roundoff(
                    levels.reindex(selected.start_at).to_numpy(),
                    levels.reindex(selected.end_at).to_numpy(),
                    rb,
                )
            roundoff += np.spacing(np.abs(differences))
            sd = (
                float(return_standard_deviation(differences, roundoff=roundoff))
                if complete and planned >= 2
                else np.nan
            )
            values.update(
                annualized_return_difference=float(mean(differences) * periods)
                if complete and not annual_reason
                else np.nan,
                tracking_error=sd * np.sqrt(periods)
                if complete and not annual_reason
                else np.nan,
                information_ratio=float(
                    _return_sharpes(differences, roundoff=roundoff) * np.sqrt(periods)
                )
                if complete and not annual_reason and sd > 0
                else np.nan,
            )
            for metric_id, value in values.items():
                if not complete:
                    value = np.nan
                reason = (
                    ""
                    if np.isfinite(value)
                    else common_reason
                    if not complete
                    else annual_reason
                    if metric_id
                    in {
                        "annualized_return_difference",
                        "tracking_error",
                        "information_ratio",
                    }
                    and annual_reason
                    else "statistic_not_identified"
                )
                records["portfolio_benchmark_metrics"].append(
                    {
                        **identity,
                        "metric_id": metric_id,
                        "value": value if np.isfinite(value) else np.nan,
                        "observation_count": n,
                        "planned_observation_count": planned,
                        "status": "DEFINED"
                        if not reason
                        else "NOT_APPLICABLE"
                        if reason.endswith("mismatch")
                        else "UNDEFINED",
                        "reason": reason,
                    }
                )
            if kind == "benchmark" and "market_exposure" in chosen:
                rf_values = (
                    None
                    if rf is None
                    else np.array(
                        [
                            value
                            for value, _ in _reference_returns(
                                selected, rf_item, rf, currency=currency, clock=clock
                            )
                        ]
                    )
                )
                records["portfolio_market_exposure"].extend(
                    _exposure(
                        identity,
                        selected,
                        rp,
                        rb,
                        rf_values,
                        daily,
                        config["market_exposure"],
                    )
                )
            if kind == "benchmark" and "regime_comparison" in chosen:
                threshold = config["regime_comparison"]["threshold"]
                for label, mask in (
                    ("above_threshold", rb > threshold),
                    ("at_or_below_threshold", rb <= threshold),
                    ("unknown", ~np.isfinite(rb)),
                ):
                    count = int((mask & paired).sum())
                    for metric, values_array in (
                        ("mean_portfolio_return", rp),
                        ("mean_benchmark_return", rb),
                        ("mean_return_difference", differences),
                    ):
                        value = (
                            float(mean(values_array[mask & paired]))
                            if count
                            else np.nan
                        )
                        records["portfolio_regime_comparison"].append(
                            {
                                **identity,
                                "regime": label,
                                "metric_id": metric,
                                "value": value,
                                "observation_count": count,
                                "planned_observation_count": int(mask.sum()),
                                "status": "DEFINED" if count else "UNDEFINED",
                                "reason": ""
                                if count
                                else "no_paired_regime_observations",
                            }
                        )
    requested = set()
    if chosen & {"benchmark_comparison", "control_comparison"}:
        requested.update(COMPARISON_TABLES[:2])
    if "market_exposure" in chosen:
        requested.add(COMPARISON_TABLES[2])
    if "regime_comparison" in chosen:
        requested.add(COMPARISON_TABLES[3])
    exposure = records["portfolio_market_exposure"]
    hypotheses = [
        row
        for row in exposure
        if row["coefficient_id"] in {"alpha", "market_return_intercept", "beta"}
    ]
    defined = sum(np.isfinite(row["p_value"]) for row in hypotheses)
    adjusted = (
        fdr_adjust(
            [
                row["p_value"] if np.isfinite(row["p_value"]) else 1.0
                for row in hypotheses
            ],
            method="BY",
        ).adjusted_p_values
        if hypotheses
        else []
    )
    for row in exposure:
        row.update(
            fdr_method="BY",
            planned_hypothesis_count=len(hypotheses),
            defined_hypothesis_count=defined,
            fdr_adjusted_p_value=np.nan,
        )
    for row, value in zip(hypotheses, adjusted, strict=True):
        if np.isfinite(row["p_value"]):
            row["fdr_adjusted_p_value"] = value
    result = {}
    for name in requested:
        frame = pd.DataFrame.from_records(
            records[name], columns=STATISTICS_TABLE_COLUMNS[f"{name}.parquet"]
        )
        for column in ("start_at", "end_at"):
            if column in frame:
                frame[column] = frame[column].astype("datetime64[ns]")
        result[name] = frame
    return result


def _exposure(identity, axis, rp, rb, rf, daily, policy):
    """HAC is fitted on the intact paired axis, never on NaN-compressed dates."""
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools.tools import add_constant

    x, y = (rb, rp) if rf is None else (rb - rf, rp - rf)
    finite = np.isfinite(x) & np.isfinite(y)
    planned, count = len(x), int(finite.sum())
    reason = (
        "incomplete_paired_path"
        if count != planned
        else "incomplete_daily_axis"
        if not daily
        else "insufficient_observations"
        if count < max(policy["minimum_observations"], policy["hac_lag"] + 2)
        else "market_variance_zero"
        if np.ptp(x) == 0
        else ""
    )
    values = {}
    names = (
        "market_return_intercept" if rf is None else "alpha",
        "beta",
        "correlation",
        "r_squared",
    )
    if not reason:
        x_scale = np.max(np.abs(x))
        y_scale = np.max(np.abs(y)) or 1.0
        scaled_x, scaled_y = x / x_scale, y / y_scale
        design = add_constant(scaled_x, has_constant="add")
        try:
            if np.linalg.matrix_rank(design) < 2:
                reason = "regression_not_identified"
            else:
                fit = OLS(scaled_y, design).fit(
                    cov_type="HAC",
                    cov_kwds={"maxlags": policy["hac_lag"], "use_correction": True},
                )
                intervals = fit.conf_int(alpha=1 - policy["confidence"])
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    for index, name in enumerate(names[:2]):
                        scale = y_scale if index == 0 else y_scale / x_scale
                        values[name] = (
                            fit.params[index] * scale,
                            fit.bse[index] * scale,
                            fit.tvalues[index],
                            fit.pvalues[index],
                            *(intervals[index] * scale),
                        )
                    values["correlation"] = (
                        np.corrcoef(scaled_x, scaled_y)[0, 1]
                        if np.ptp(scaled_y)
                        else np.nan,
                        *([np.nan] * 5),
                    )
                    values["r_squared"] = (
                        fit.rsquared if np.ptp(scaled_y) else np.nan,
                        *([np.nan] * 5),
                    )
        except np.linalg.LinAlgError:
            reason = "regression_not_identified"
    result = []
    for name in names:
        measurements = values.get(name, (np.nan,) * 6)
        valid = np.isfinite(measurements[0]) and (
            name in names[2:] or np.isfinite(measurements).all()
        )
        result.append(
            {
                **identity,
                "coefficient_id": name,
                **dict(
                    zip(
                        (
                            "value",
                            "standard_error",
                            "t_statistic",
                            "p_value",
                            "ci_lower",
                            "ci_upper",
                        ),
                        measurements if valid else (np.nan,) * 6,
                        strict=True,
                    )
                ),
                "observation_count": count,
                "planned_observation_count": planned,
                "status": "DEFINED" if valid else "UNDEFINED",
                "reason": "" if valid else reason or "regression_not_identified",
                "start_at": axis.start_at.iloc[0],
                "end_at": axis.end_at.iloc[-1],
                "hac_lag": policy["hac_lag"],
                "confidence": policy["confidence"],
            }
        )
    return result

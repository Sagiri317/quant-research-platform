"""Concrete daily coverage and stratification tables for factor evaluation."""

import pandas as pd
import pyarrow as pa

from quant_research_platform.artifacts.outputs.evaluation_validation import (
    _validate_correlation_metrics,
)
from quant_research_platform.contracts.time import (
    TimeWindow,
    time_window_timestamp_interval,
)
from quant_research_platform.contracts.validation import (
    strict_integer_series,
    strict_timestamp_series,
)

METRICS_SCHEMA = pa.schema(
    [
        pa.field("timestamp", pa.timestamp("ns")),
        pa.field("signal", pa.large_string()),
        pa.field("label", pa.large_string()),
        pa.field("ic", pa.float64()),
        pa.field("rankic", pa.float64()),
        pa.field("n_obs", pa.int64()),
        pa.field("ic_reason", pa.large_string()),
        pa.field("rankic_reason", pa.large_string()),
    ]
)

DAILY_SCHEMAS = {
    "daily_coverage": pa.schema(
        [
            pa.field("timestamp", pa.timestamp("ns"), nullable=False),
            pa.field("label", pa.string(), nullable=False),
            *[
                pa.field(name, pa.int64(), nullable=False)
                for name in (
                    "dataset_rows",
                    "eligible_rows",
                    "mature_rows",
                    "domain_rows",
                )
            ],
            *[
                pa.field(name, pa.int64())
                for name in (
                    "age_missing",
                    "fractional_boundary_unassigned",
                    "age_invalid",
                    "size_invalid",
                    "liquidity_invalid",
                )
            ],
        ]
    ),
    "daily_strata": pa.schema(
        [
            pa.field("timestamp", pa.timestamp("ns"), nullable=False),
            pa.field("signal", pa.string(), nullable=False),
            pa.field("label", pa.string(), nullable=False),
            pa.field("ic", pa.float64()),
            pa.field("rankic", pa.float64()),
            pa.field("n_obs", pa.int64(), nullable=False),
            pa.field("ic_reason", pa.string()),
            pa.field("rankic_reason", pa.string()),
            pa.field("stratification", pa.string(), nullable=False),
            pa.field("stratum", pa.string()),
            pa.field("stratum_rows", pa.int64(), nullable=False),
        ]
    ),
}
DAILY_KEYS = {
    "daily_coverage": ("timestamp", "label"),
    "daily_strata": ("timestamp", "signal", "label", "stratification", "stratum"),
}


def daily_frame(records, name):
    """Materialize the concrete schema, including empty and nullable count columns."""
    schema = DAILY_SCHEMAS[name]
    frame = pd.DataFrame.from_records(records, columns=schema.names)
    for field in schema:
        if pa.types.is_timestamp(field.type):
            frame[field.name] = frame[field.name].astype("datetime64[ns]")
        elif pa.types.is_integer(field.type):
            frame[field.name] = frame[field.name].astype(
                "Int64" if field.nullable else "int64"
            )
        elif pa.types.is_floating(field.type):
            frame[field.name] = frame[field.name].astype("float64")
        else:
            frame[field.name] = frame[field.name].astype("string")
    return frame


def daily_key_table(frame, name):
    """Encode null strata separately from literal industry strings in exact keys."""
    keys = frame.loc[:, list(DAILY_KEYS[name])].copy()
    if name == "daily_strata":
        keys["stratum"] = keys.stratum.map(
            lambda value: "0" if pd.isna(value) else "1" + value
        )
    return pa.Table.from_pandas(keys, preserve_index=False)


def validate_daily_frame(frame, *, name, spec, science, check_domains=True):
    schema = DAILY_SCHEMAS[name]
    if tuple(frame.columns) != tuple(schema.names):
        raise ValueError("evaluation_daily_schema_invalid")
    for field in schema:
        values = frame[field.name]
        if not field.nullable and values.isna().any():
            raise ValueError("evaluation_daily_null_invalid")
        if pa.types.is_integer(field.type):
            selected = values.dropna() if field.nullable else values
            if (
                strict_integer_series(selected, error="evaluation_daily_count_invalid")
                .lt(0)
                .any()
            ):
                raise ValueError("evaluation_daily_count_invalid")
        elif pa.types.is_string(field.type):
            if any(type(value) is not str for value in values.dropna()):
                raise ValueError("evaluation_daily_string_invalid")
    timestamps = strict_timestamp_series(
        frame.timestamp, error="evaluation_daily_time_invalid"
    )
    start, stop, closed = time_window_timestamp_interval(
        TimeWindow(**science["window"])
    )
    if not (
        timestamps.ge(start) & (timestamps.le(stop) if closed else timestamps.lt(stop))
    ).all():
        raise ValueError("evaluation_daily_window_mismatch")
    if (
        not frame.label.isin(spec.label_columns).all()
        or frame.duplicated(list(DAILY_KEYS[name])).any()
    ):
        raise ValueError("evaluation_daily_domain_or_keys_invalid")
    if name.startswith("response_"):
        _validate_response_frame(frame, name=name, science=science)
        return
    strata = science["strata"]
    if name == "daily_coverage":
        if not (
            frame.dataset_rows.ge(frame.eligible_rows)
            & frame.eligible_rows.ge(frame.mature_rows)
            & frame.mature_rows.ge(frame.domain_rows)
        ).all():
            raise ValueError("evaluation_coverage_counts_invalid")
        if (
            science["sample_scope"] == "own"
            and not frame.domain_rows.eq(frame.mature_rows).all()
        ):
            raise ValueError("evaluation_coverage_domain_invalid")
        for column, kind in (
            ("age_missing", "forecast_age"),
            ("fractional_boundary_unassigned", "forecast_age"),
            ("age_invalid", "forecast_age"),
            ("size_invalid", "size"),
            ("liquidity_invalid", "liquidity"),
        ):
            active = frame.label.eq(strata.get("primary_label")) & (
                kind in strata.get("kinds", ())
            )
            if (
                frame[column].notna().ne(active).any()
                or frame.loc[active, column].gt(frame.loc[active, "mature_rows"]).any()
            ):
                raise ValueError("evaluation_coverage_quality_invalid")
        age = frame[
            ["age_missing", "fractional_boundary_unassigned", "age_invalid"]
        ].sum(axis=1, min_count=1)
        if age.gt(frame.mature_rows).any():
            raise ValueError("evaluation_coverage_quality_invalid")
        return
    if (
        not frame.label.eq(strata.get("primary_label")).all()
        or not frame.stratification.isin(strata.get("kinds", ())).all()
    ):
        raise ValueError("evaluation_stratum_domain_invalid")
    for kind, block in frame.groupby("stratification", sort=False):
        actual = {None if pd.isna(value) else value for value in block.stratum}
        if check_domains and not actual.issubset(science["stratum_domains"][kind]):
            raise ValueError("evaluation_stratum_domain_invalid")
        allowed = (
            ["1", "2", "3"]
            if kind in {"size", "liquidity"}
            else [bound[0] for bound in strata["age_bounds"]]
            if kind == "forecast_age"
            else None
        )
        if allowed is not None and not block.stratum.dropna().isin(allowed).all():
            raise ValueError("evaluation_stratum_domain_invalid")
    if frame.n_obs.gt(frame.stratum_rows).any():
        raise ValueError("evaluation_stratum_counts_invalid")
    _validate_correlation_metrics(frame, spec=spec)


def validate_daily_counts(metrics, coverage, strata, *, spec, science):
    """Cross-member checks apply only when those members are actually consumed."""
    if coverage is None:
        return
    denominators = coverage.set_index(["timestamp", "label"])
    for name, frame in (("metrics", metrics), ("daily_strata", strata)):
        if frame is None or frame.empty:
            continue
        aligned = denominators.reindex(
            pd.MultiIndex.from_frame(frame[["timestamp", "label"]])
        )
        if (
            aligned.domain_rows.isna().any()
            or (frame.n_obs.to_numpy() > aligned.domain_rows.to_numpy()).any()
        ):
            raise ValueError("evaluation_daily_coverage_mismatch")
        if (
            name == "metrics"
            and science.get("sample_scope", "own") == "common"
            and not (frame.n_obs.to_numpy() == aligned.domain_rows.to_numpy()).all()
        ):
            raise ValueError("evaluation_common_metric_count_mismatch")
        if (
            name == "daily_strata"
            and (frame.stratum_rows.to_numpy() > aligned.mature_rows.to_numpy()).any()
        ):
            raise ValueError("evaluation_daily_coverage_mismatch")

    signals = {signal.signal_id for signal in spec.signals}
    if metrics is not None:
        expected = {
            (row.timestamp, signal, row.label)
            for row in coverage.itertuples()
            for signal in signals
        }
        actual = set(
            metrics[["timestamp", "signal", "label"]].itertuples(index=False, name=None)
        )
        if expected != actual:
            raise ValueError("evaluation_daily_metric_domain_mismatch")
    for _, block in coverage.groupby("timestamp", sort=False):
        if (
            set(block.label) != set(spec.label_columns)
            or block.dataset_rows.nunique() != 1
            or block.eligible_rows.nunique() != 1
        ):
            raise ValueError("evaluation_daily_coverage_mismatch")
    definition = science.get("strata", {})
    if strata is None or not definition:
        return
    primary = coverage.loc[coverage.label.eq(definition["primary_label"])].set_index(
        "timestamp"
    )
    actual_groups = set()
    for (timestamp, kind, signal), block in strata.groupby(
        ["timestamp", "stratification", "signal"], sort=False
    ):
        actual_groups.add((timestamp, kind, signal))
        denominator = int(primary.loc[timestamp, "mature_rows"])
        if kind == "forecast_age":
            denominator -= int(primary.loc[timestamp, "fractional_boundary_unassigned"])
            if set(block.stratum.dropna()) != {
                bound[0] for bound in definition["age_bounds"]
            }:
                raise ValueError("evaluation_age_bins_incomplete")
        elif block.stratum_rows.eq(0).any():
            raise ValueError("evaluation_empty_stratum_invalid")
        if metrics is not None:
            total = metrics.loc[
                metrics.timestamp.eq(timestamp)
                & metrics.signal.eq(signal)
                & metrics.label.eq(definition["primary_label"]),
                "n_obs",
            ]
            if len(total) != 1:
                raise ValueError("evaluation_stratum_observations_mismatch")
            unassigned = int(total.iloc[0]) - int(block.n_obs.sum())
            maximum_unassigned = (
                int(primary.loc[timestamp, "fractional_boundary_unassigned"])
                if kind == "forecast_age"
                else 0
            )
            if not 0 <= unassigned <= maximum_unassigned:
                raise ValueError("evaluation_stratum_observations_mismatch")
        if int(block.stratum_rows.sum()) != denominator:
            raise ValueError("evaluation_stratum_population_mismatch")
    expected_groups = {
        (timestamp, kind, signal)
        for timestamp, row in primary.iterrows()
        if row.mature_rows > 0
        for kind in definition["kinds"]
        for signal in signals
    }
    if actual_groups != expected_groups:
        raise ValueError("evaluation_stratum_population_mismatch")
    for _, block in strata.groupby(
        ["timestamp", "stratification", "stratum"], dropna=False, sort=False
    ):
        if block.stratum_rows.nunique() != 1 or set(block.signal) != signals:
            raise ValueError("evaluation_stratum_population_mismatch")


_RESPONSE_KEYS = (
    "timestamp",
    "period",
    "sample_scope",
    "signal",
    "label",
)
_RESPONSE_FIELDS = [
    pa.field("timestamp", pa.timestamp("ns"), nullable=False),
    *[pa.field(name, pa.string(), nullable=False) for name in _RESPONSE_KEYS[1:]],
]
DAILY_SCHEMAS.update(
    {
        "response_metrics": pa.schema(
            [
                *_RESPONSE_FIELDS,
                pa.field("ic", pa.float64()),
                pa.field("ic_reason", pa.string()),
                pa.field("n_obs", pa.int64(), nullable=False),
                pa.field("signal_count", pa.int64(), nullable=False),
                pa.field("coverage", pa.float64()),
                pa.field("ic_std", pa.float64()),
                pa.field("valid_day_count", pa.int64(), nullable=False),
                pa.field("top_bottom_mean_bp", pa.float64()),
                pa.field("spread_reason", pa.string()),
            ]
        ),
        "response_groups": pa.schema(
            [
                *_RESPONSE_FIELDS,
                pa.field("group", pa.int64(), nullable=False),
                pa.field("mean_return_bp", pa.float64()),
                pa.field("median_return_bp", pa.float64()),
                pa.field("n_obs", pa.int64(), nullable=False),
                pa.field("mean_reason", pa.string()),
                pa.field("median_reason", pa.string()),
                pa.field("mean_std_bp", pa.float64()),
                pa.field("valid_day_count", pa.int64(), nullable=False),
            ]
        ),
    }
)
DAILY_KEYS.update(
    {"response_metrics": _RESPONSE_KEYS, "response_groups": (*_RESPONSE_KEYS, "group")}
)


def _validate_response_frame(frame, *, name, science):
    import numpy as np

    signal, cutpoints = next(iter(science["response_diagnostics"]["cutpoints"].items()))
    if (
        not frame.signal.eq(signal).all()
        or not frame.period.isin(["daily", "pooled", "daily_equal_weight"]).all()
        or not frame.sample_scope.isin(["own", "common"]).all()
    ):
        raise ValueError("evaluation_response_domain_invalid")
    for column in frame.select_dtypes(include="floating"):
        if np.isinf(frame[column].to_numpy(dtype=float, na_value=np.nan)).any():
            raise ValueError("evaluation_response_numeric_invalid")
    summary = frame.period.eq("daily_equal_weight")
    if name == "response_metrics":
        if (
            frame.n_obs.gt(frame.signal_count).any()
            or frame.ic.dropna().abs().gt(1).any()
            or frame.ic.isna().ne(frame.ic_reason.notna()).any()
            or frame.top_bottom_mean_bp.isna().ne(frame.spread_reason.notna()).any()
            or frame.coverage.isna().ne(frame.signal_count.eq(0)).any()
            or not np.allclose(
                frame.coverage,
                frame.n_obs / frame.signal_count,
                equal_nan=True,
                rtol=1e-12,
                atol=0,
            )
        ):
            raise ValueError("evaluation_response_metric_invalid")
    else:
        if (
            not frame.group.between(1, len(cutpoints) + 1).all()
            or (frame.n_obs.eq(0) & frame.mean_return_bp.notna()).any()
            or (frame.n_obs.eq(0) & frame.median_return_bp.notna()).any()
            or frame.mean_return_bp.isna().ne(frame.mean_reason.notna()).any()
            or frame.median_return_bp.isna().ne(frame.median_reason.notna()).any()
            or frame.loc[summary, "median_return_bp"].notna().any()
        ):
            raise ValueError("evaluation_response_groups_invalid")


def validate_response_counts(metrics, groups, *, science):
    """Check concrete paired response rows when both members are consumed."""
    if metrics is None or groups is None or metrics.empty:
        return
    key = list(_RESPONSE_KEYS)
    sums = groups.groupby(key, dropna=False, sort=False).n_obs.sum()
    counts = metrics.set_index(key).n_obs
    if not sums.sort_index().equals(counts.sort_index()):
        raise ValueError("evaluation_response_group_count_mismatch")
    expected = set(
        range(
            1,
            len(next(iter(science["response_diagnostics"]["cutpoints"].values()))) + 2,
        )
    )
    if any(
        set(block.group) != expected for _, block in groups.groupby(key, sort=False)
    ):
        raise ValueError("evaluation_response_group_domain_incomplete")
    for _, block in metrics.groupby(["timestamp", "period", "signal"], sort=False):
        if (
            block.signal_count.nunique() != 1
            or block.loc[block.sample_scope.eq("common"), "n_obs"].nunique() != 1
        ):
            raise ValueError("evaluation_response_common_count_mismatch")
        paired = block.pivot(index="label", columns="sample_scope", values="n_obs")
        if (
            set(paired.columns) != {"own", "common"}
            or paired.common.gt(paired.own).any()
        ):
            raise ValueError("evaluation_response_common_count_mismatch")

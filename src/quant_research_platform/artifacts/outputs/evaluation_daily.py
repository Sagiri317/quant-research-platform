"""Concrete daily coverage and stratification tables for factor evaluation."""

import pandas as pd
import pyarrow as pa

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

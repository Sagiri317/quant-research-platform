"""A-share benchmark-weight facts and independent industry history."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_research_platform.contracts.validation import (
    strict_finite_numeric_series,
    strict_string_series,
    strict_timestamp_series,
)
from quant_research_platform.markets.implementations.a_share.ids import (
    FORMAL_BENCHMARKS,
)

_WEIGHT_COLUMNS = (
    "market_id",
    "benchmark_id",
    "asset_id",
    "timestamp",
    "weight",
)


def iter_benchmark_weight_snapshots(index_info_interim, weight_frames):
    """Validate every available benchmark without requiring unrelated benchmarks."""
    observed = set()
    for frame in weight_frames:
        normalized = _normalize_benchmark_weight_snapshots(index_info_interim, frame)
        observed.update(normalized["benchmark_id"])
        yield normalized
        del normalized, frame
    if not observed:
        raise ValueError("a_share_benchmark_coverage_incomplete")


def _normalize_benchmark_weight_snapshots(
    index_info_interim: pd.DataFrame,
    weight_interim: pd.DataFrame,
) -> pd.DataFrame:

    if (
        type(index_info_interim) is not pd.DataFrame
        or type(weight_interim) is not pd.DataFrame
    ):
        raise ValueError("a_share_benchmark_interim_schema_invalid")
    if not {"source_asset_id", "index_name"}.issubset(index_info_interim.columns):
        raise ValueError("a_share_benchmark_index_info_schema_invalid")
    required_weights = {
        "source_asset_id",
        "constituent_asset_id",
        "effective_date",
        "weight",
    }
    if not required_weights.issubset(weight_interim.columns):
        raise ValueError("a_share_benchmark_weight_schema_invalid")
    weights = weight_interim.loc[
        weight_interim["source_asset_id"].isin(FORMAL_BENCHMARKS)
    ].copy()
    weights["source_asset_id"] = strict_string_series(
        weights["source_asset_id"],
        error="a_share_benchmark_source_index_invalid",
    )

    index_info = index_info_interim.loc[
        index_info_interim["source_asset_id"].isin(weights["source_asset_id"].unique())
    ].copy()
    index_info["source_asset_id"] = strict_string_series(
        index_info["source_asset_id"],
        error="a_share_benchmark_source_index_invalid",
    )
    index_info["index_name"] = strict_string_series(
        index_info["index_name"],
        error="a_share_benchmark_index_name_invalid",
    )

    for source_id in weights["source_asset_id"].unique():
        _benchmark_id, expected_name = FORMAL_BENCHMARKS[source_id]
        names = set(
            index_info.loc[index_info["source_asset_id"].eq(source_id), "index_name"]
        )
        if names != {expected_name}:
            raise ValueError("a_share_benchmark_index_identity_invalid")

    weights["benchmark_id"] = weights["source_asset_id"].map(
        {source_id: values[0] for source_id, values in FORMAL_BENCHMARKS.items()}
    )
    weights["asset_id"] = strict_string_series(
        weights["constituent_asset_id"],
        error="a_share_benchmark_asset_id_invalid",
    )
    weights["timestamp"] = strict_timestamp_series(
        weights["effective_date"],
        error="a_share_benchmark_weight_date_invalid",
    ).dt.normalize()
    weights["weight"] = strict_finite_numeric_series(
        weights["weight"],
        error="a_share_benchmark_weight_invalid",
    ).astype("float64")
    if bool(weights["weight"].le(0.0).any()):
        raise ValueError("a_share_benchmark_weight_invalid")
    key = ["benchmark_id", "asset_id", "timestamp"]
    if bool(weights.duplicated(key, keep=False).any()):
        raise ValueError("a_share_benchmark_weight_key_duplicate")
    grouped = weights.groupby(["benchmark_id", "timestamp"])["weight"]
    totals = grouped.transform("sum")
    counts = grouped.transform("size")
    # CSMAR N10,4 percent weights have a 1e-6 fractional quantum per observation.
    tolerance = counts * 1e-6 + (counts + 2) * np.finfo(float).eps * np.maximum(
        1.0, totals.abs()
    )
    if not bool(np.isfinite(totals).all()) or not bool(
        np.isclose(totals.to_numpy(), 1.0, atol=tolerance, rtol=0.0).all()
    ):
        raise ValueError("a_share_benchmark_weight_sum_invalid")
    weights["weight"] = weights["weight"].div(totals)
    return (
        weights.assign(market_id="a_share")
        .loc[:, list(_WEIGHT_COLUMNS)]
        .sort_values(["timestamp", "benchmark_id", "asset_id"], kind="mergesort")
        .reset_index(drop=True)
    )

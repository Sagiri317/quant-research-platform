"""A-share benchmark-weight facts and independent industry history."""

from __future__ import annotations

from types import MappingProxyType

import numpy as np
import pandas as pd

from quant_research_platform.contracts.validation import (
    strict_finite_numeric_series,
    strict_string_series,
    strict_timestamp_series,
)

FORMAL_BENCHMARKS = MappingProxyType(
    {
        "000905": ("csi500", "中证小盘500指数"),
        "000852": ("csi1000", "中证1000指数"),
    }
)
APPROVED_BENCHMARK_IDS = frozenset(
    benchmark_id for benchmark_id, _name in FORMAL_BENCHMARKS.values()
)
_WEIGHT_COLUMNS = (
    "market_id",
    "benchmark_id",
    "asset_id",
    "timestamp",
    "weight",
)


def iter_benchmark_weight_snapshots(index_info_interim, weight_frames):
    """Validate complete dates and require both benchmarks across the input window."""
    observed = set()
    for frame in weight_frames:
        normalized = _normalize_benchmark_weight_snapshots(index_info_interim, frame)
        observed.update(normalized["benchmark_id"])
        yield normalized
        del normalized, frame
    if observed != APPROVED_BENCHMARK_IDS:
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
    index_info = index_info_interim.loc[
        index_info_interim["source_asset_id"].isin(FORMAL_BENCHMARKS)
    ].copy()
    index_info["source_asset_id"] = strict_string_series(
        index_info["source_asset_id"],
        error="a_share_benchmark_source_index_invalid",
    )
    index_info["index_name"] = strict_string_series(
        index_info["index_name"],
        error="a_share_benchmark_index_name_invalid",
    )
    for source_id, (_benchmark_id, expected_name) in FORMAL_BENCHMARKS.items():
        names = set(
            index_info.loc[index_info["source_asset_id"].eq(source_id), "index_name"]
        )
        if names != {expected_name}:
            raise ValueError("a_share_benchmark_index_identity_invalid")

    weights = weight_interim.loc[
        weight_interim["source_asset_id"].isin(FORMAL_BENCHMARKS)
    ].copy()
    weights["source_asset_id"] = strict_string_series(
        weights["source_asset_id"],
        error="a_share_benchmark_source_index_invalid",
    )
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
    totals = weights.groupby(["benchmark_id", "timestamp"])["weight"].sum()
    if not bool(np.isclose(totals.to_numpy(), 1.0, atol=1e-6, rtol=0.0).all()):
        raise ValueError("a_share_benchmark_weight_sum_invalid")
    return (
        weights.assign(market_id="a_share")
        .loc[:, list(_WEIGHT_COLUMNS)]
        .sort_values(["timestamp", "benchmark_id", "asset_id"], kind="mergesort")
        .reset_index(drop=True)
    )

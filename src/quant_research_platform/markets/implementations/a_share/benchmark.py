"""CSI500/CSI1000 effective-date benchmark and independent industry history."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

APPROVED_BENCHMARK_IDS = frozenset({"csi500", "csi1000"})


def normalize_benchmark_history(
    *,
    membership_interim: pd.DataFrame,
    weight_interim: pd.DataFrame,
    return_interim: pd.DataFrame,
    index_id_mapping: Mapping[str, str],
) -> dict[str, pd.DataFrame]:
    """Validate exact interim tables and bind provider index codes once."""

    mapping = dict(index_id_mapping)
    if set(mapping.values()) != APPROVED_BENCHMARK_IDS:
        raise ValueError("a_share_benchmark_mapping_invalid")
    membership = membership_interim.copy()
    weights = weight_interim.copy()
    returns = return_interim.copy()
    for frame, required in (
        (
            membership,
            {
                "source_asset_id",
                "constituent_asset_id",
                "effective_date",
                "announcement_date",
                "change_type",
            },
        ),
        (
            weights,
            {
                "source_asset_id",
                "constituent_asset_id",
                "effective_date",
                "weight",
            },
        ),
        (returns, {"source_asset_id", "timestamp", "period_return"}),
    ):
        if not required.issubset(frame.columns):
            raise ValueError("a_share_benchmark_interim_schema_invalid")
        frame["benchmark_id"] = frame["source_asset_id"].map(mapping)
        if bool(frame["benchmark_id"].isna().any()):
            raise ValueError("a_share_benchmark_unmapped_index")
        if set(frame["benchmark_id"]) != APPROVED_BENCHMARK_IDS:
            raise ValueError("a_share_benchmark_coverage_incomplete")
    membership["is_member"] = (
        membership["change_type"].astype("string").map({"1": True, "2": False})
    )
    if bool(membership["is_member"].isna().any()):
        raise ValueError("a_share_benchmark_change_type_invalid")
    membership_key = ["benchmark_id", "constituent_asset_id", "effective_date"]
    weight_key = ["benchmark_id", "constituent_asset_id", "effective_date"]
    return_key = ["benchmark_id", "timestamp"]
    if bool(membership.duplicated(membership_key, keep=False).any()):
        raise ValueError("a_share_benchmark_membership_key_duplicate")
    if bool(weights.duplicated(weight_key, keep=False).any()):
        raise ValueError("a_share_benchmark_weight_key_duplicate")
    if bool(returns.duplicated(return_key, keep=False).any()):
        raise ValueError("a_share_benchmark_return_key_duplicate")
    if bool((weights["weight"] < 0.0).any()):
        raise ValueError("a_share_benchmark_weight_invalid")
    totals = weights.groupby(["benchmark_id", "effective_date"])["weight"].sum()
    if not bool(np.isclose(totals.to_numpy(), 1.0, atol=1e-6).all()):
        raise ValueError("a_share_benchmark_weight_sum_invalid")
    membership_state = pd.merge(
        weights,
        membership.loc[
            :,
            (
                "benchmark_id",
                "constituent_asset_id",
                "effective_date",
                "is_member",
            ),
        ],
        on=["benchmark_id", "constituent_asset_id", "effective_date"],
        how="left",
        validate="many_to_one",
    )
    if bool(membership_state["is_member"].ne(True).any()):
        raise ValueError("a_share_benchmark_weight_membership_invalid")
    return {
        "membership": membership.sort_values(membership_key).reset_index(drop=True),
        "weights": weights.sort_values(weight_key).reset_index(drop=True),
        "returns": returns.sort_values(return_key).reset_index(drop=True),
    }


def materialize_benchmark_weights(
    observations: pd.DataFrame,
    weights: pd.DataFrame,
    membership: pd.DataFrame,
) -> pd.DataFrame:
    """As-of materialize historical constituent weights by effective date."""

    required = {"benchmark_id", "constituent_asset_id", "timestamp"}
    if not required.issubset(observations.columns):
        raise ValueError("a_share_benchmark_observation_schema_invalid")
    left = observations.loc[
        :, ("benchmark_id", "constituent_asset_id", "timestamp")
    ].copy()
    right = weights.loc[
        :,
        ("benchmark_id", "constituent_asset_id", "effective_date", "weight"),
    ].copy()
    for column in ("benchmark_id", "constituent_asset_id"):
        left[column] = left[column].astype("string")
        right[column] = right[column].astype("string")
    left["timestamp"] = pd.Series(
        pd.to_datetime(left["timestamp"]), index=left.index, dtype="datetime64[ns]"
    )
    right["effective_date"] = pd.Series(
        pd.to_datetime(right["effective_date"]),
        index=right.index,
        dtype="datetime64[ns]",
    )
    membership_right = membership.loc[
        :, ("benchmark_id", "constituent_asset_id", "effective_date", "is_member")
    ].copy()
    for column in ("benchmark_id", "constituent_asset_id"):
        membership_right[column] = membership_right[column].astype("string")
    membership_right["effective_date"] = pd.to_datetime(
        membership_right["effective_date"]
    ).astype("datetime64[ns]")
    state = pd.merge_asof(
        left.sort_values("timestamp"),
        membership_right.sort_values("effective_date"),
        left_on="timestamp",
        right_on="effective_date",
        by=["benchmark_id", "constituent_asset_id"],
        direction="backward",
        allow_exact_matches=True,
    )
    exact_weights = right.rename(columns={"effective_date": "timestamp"})
    result = state.merge(
        exact_weights,
        on=["benchmark_id", "constituent_asset_id", "timestamp"],
        how="left",
        validate="one_to_one",
    )
    result["weight"] = result["weight"].where(result["is_member"].eq(True))
    return result.sort_values(
        ["benchmark_id", "constituent_asset_id", "timestamp"], kind="mergesort"
    ).reset_index(drop=True)


def materialize_industry_history(
    observations: pd.DataFrame,
    stock_master_interim: pd.DataFrame,
) -> pd.DataFrame:
    """As-of materialize industry independently of any benchmark identity."""

    if not {"asset_id", "timestamp"}.issubset(observations.columns) or not {
        "source_asset_id",
        "timestamp",
        "industry_standard_id",
        "industry_level1_code",
        "industry_level1_name",
    }.issubset(stock_master_interim.columns):
        raise ValueError("a_share_industry_schema_invalid")
    return (
        pd.merge_asof(
            observations.sort_values("timestamp"),
            stock_master_interim.loc[
                :,
                (
                    "source_asset_id",
                    "timestamp",
                    "industry_standard_id",
                    "industry_level1_code",
                    "industry_level1_name",
                ),
            ]
            .rename(columns={"source_asset_id": "asset_id"})
            .sort_values("timestamp"),
            on="timestamp",
            by="asset_id",
            direction="backward",
            allow_exact_matches=True,
        )
        .sort_values(["asset_id", "timestamp"], kind="mergesort")
        .reset_index(drop=True)
    )

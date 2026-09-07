from __future__ import annotations

import pandas as pd
import pytest

from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.implementations.a_share.benchmark import (
    iter_benchmark_weight_snapshots,
)
from quant_research_platform.markets.implementations.a_share.processed import (
    producer as processed_producer,
)
from quant_research_platform.markets.implementations.a_share.processed.catalog import (
    processed_catalog,
)
from quant_research_platform.storage.layout import StorageLayout


def _benchmark_facts() -> tuple[pd.DataFrame, pd.DataFrame]:
    index_info = pd.DataFrame(
        {
            "source_asset_id": ["000905", "000852", "399905", "399852", "000300"],
            "index_name": [
                "中证小盘500指数",
                "中证1000指数",
                "中证小盘500指数",
                "中证1000指数",
                "沪深300指数",
            ],
        }
    )
    rows = []
    for source_id, date, first_weight in (
        ("000905", "2020-01-02", 0.4),
        ("000852", "2020-01-02", 0.3),
        ("000905", "2020-01-03", 0.6),
        ("000852", "2020-01-03", 0.7),
    ):
        rows.extend(
            [
                {
                    "source_asset_id": source_id,
                    "constituent_asset_id": "000001",
                    "effective_date": date,
                    "weight": first_weight,
                },
                {
                    "source_asset_id": source_id,
                    "constituent_asset_id": "000002",
                    "effective_date": date,
                    "weight": 1.0 - first_weight,
                },
            ]
        )
    rows.append(
        {
            "source_asset_id": "399905",
            "constituent_asset_id": "000001",
            "effective_date": "2020-01-02",
            "weight": 1.0,
        }
    )
    return index_info, pd.DataFrame(rows)


def test_benchmark_filters_repeated_codes_and_exact_snapshots():
    index_info, weights = _benchmark_facts()
    normalized = pd.concat(
        iter_benchmark_weight_snapshots(index_info, (weights,)), ignore_index=True
    )

    assert tuple(normalized.columns) == (
        "market_id",
        "benchmark_id",
        "asset_id",
        "timestamp",
        "weight",
    )
    assert set(normalized["benchmark_id"]) == {"csi500", "csi1000"}
    assert set(normalized["asset_id"]) == {"000001", "000002"}
    assert len(normalized) == 8


def test_processed_benchmark_target_streams_exact_dates_and_requires_global_coverage(
    tmp_path,
):
    from quant_research_platform.profiles.research import research_profile
    from quant_research_platform.runtime.dataframes import iter_complete_session_frames
    from quant_research_platform.runtime.resources import runtime_resource_policy_scope
    from quant_research_platform.storage.datasets.reading import (
        DatasetReadMode,
        read_dataset,
    )
    from quant_research_platform.storage.datasets.writing import write_dataset

    profile = research_profile()
    layout = StorageLayout(tmp_path)
    index_info, _ = _benchmark_facts()
    dates = pd.bdate_range("2020-01-02", periods=130)
    index_info["timestamp"] = dates[0]
    weights = pd.DataFrame(
        [
            {
                "source_asset_id": "000905" if offset < 64 else "000852",
                "constituent_asset_id": asset,
                "timestamp": date,
                "effective_date": date,
                "constituent_name": asset,
                "weight": weight,
            }
            for offset, date in enumerate(dates)
            for asset, weight in (("000001", 0.4), ("000002", 0.6))
        ]
    )
    with runtime_resource_policy_scope(explicit_cap_bytes=128 * 1024**2) as policy:
        for data_id, frame in (("index_info", index_info), ("index_weights", weights)):
            source_spec = profile.interim.get(data_id)
            frame = frame.reindex(columns=source_spec.required_columns)
            write_dataset(
                layout.interim_data_root(source_spec.source_id, data_id),
                frame,
                spec=source_spec,
                resource_policy=policy,
                borrowed_input=True,
            )
        spec = processed_catalog().get("benchmark_weights")
        root = layout.processed_data_root(spec.market_id, spec.kind, spec.data_id)
        with processed_producer.produce_a_share_processed_data(
            spec,
            layout=layout,
            time_window=TimeWindow(dates[0], dates[-1]),
            processed_catalog=profile.processed,
            source_catalog=profile.interim,
            resource_policy=policy,
        ) as (frames, metadata, verify):
            write_dataset(
                root,
                frames,
                spec=spec,
                resource_policy=policy,
                instance_metadata=metadata,
                staged_validator=verify,
            )
        actual = read_dataset(root, resource_policy=policy)
        expected = pd.concat(
            iter_benchmark_weight_snapshots(index_info, (weights,)), ignore_index=True
        )
        pd.testing.assert_frame_equal(actual, expected)
        assert len(tuple((root / "data").rglob("*.parquet"))) > 1
        streamed = list(
            iter_complete_session_frames(
                read_dataset(
                    root, mode=DatasetReadMode.CHUNKED, resource_policy=policy
                ),
                timestamp_column="timestamp",
                resource_policy=policy,
            )
        )
        pd.testing.assert_frame_equal(pd.concat(streamed, ignore_index=True), expected)
        assert policy.used_memory_bytes == 0

    with pytest.raises(ValueError, match="coverage_incomplete"):
        tuple(iter_benchmark_weight_snapshots(index_info, (weights.iloc[:128],)))


@pytest.mark.parametrize(
    ("mutation", "error"),
    (
        ("duplicate", "key_duplicate"),
        ("negative", "weight_invalid"),
        ("sum", "weight_sum_invalid"),
        ("identity", "index_identity_invalid"),
    ),
)
def test_benchmark_normalization_rejects_fact_mutations(
    mutation: str,
    error: str,
) -> None:
    index_info, weights = _benchmark_facts()
    if mutation == "duplicate":
        weights = pd.concat([weights, weights.iloc[[0]]], ignore_index=True)
    elif mutation == "negative":
        weights.loc[0, "weight"] = -0.1
    elif mutation == "sum":
        weights.loc[1, "weight"] = 0.7
    else:
        index_info.loc[0, "index_name"] = "wrong"
    with pytest.raises(ValueError, match=error):
        pd.concat(
            iter_benchmark_weight_snapshots(index_info, (weights,)), ignore_index=True
        )

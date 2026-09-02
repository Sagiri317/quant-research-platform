from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.implementations.a_share.benchmark import (
    materialize_benchmark_weights,
    normalize_benchmark_weight_snapshots,
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
    normalized = normalize_benchmark_weight_snapshots(index_info, weights)

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

    observations = pd.DataFrame(
        {
            "benchmark_id": ["csi500", "csi500"],
            "asset_id": ["000001", "000001"],
            "timestamp": pd.to_datetime(["2020-01-02", "2020-01-04"]),
        }
    )
    looked_up = materialize_benchmark_weights(observations, normalized)
    assert looked_up["weight"].iloc[0] == pytest.approx(0.4)
    assert pd.isna(looked_up["weight"].iloc[1])
    with pytest.raises(ValueError, match="market_id_invalid"):
        materialize_benchmark_weights(
            observations,
            normalized.assign(market_id="crypto"),
        )


def test_processed_benchmark_target_reads_required_index_facts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    index_info, weights = _benchmark_facts()
    frames = {"index_info": index_info, "index_weights": weights}
    observed: list[tuple[str, bool, bool]] = []

    monkeypatch.setattr(processed_producer, "_source_available", lambda *_: True)

    def read_source(
        _layout,
        spec,
        *,
        time_window,
        policy,
        optional,
        full_history,
    ):
        del time_window, policy
        observed.append((spec.data_id, optional, full_history))
        return frames[spec.data_id]

    monkeypatch.setattr(processed_producer, "_read_source", read_source)
    spec = processed_catalog().get("benchmark_weights")
    result = processed_producer.produce_a_share_processed_frame(
        spec,
        layout=StorageLayout(project_root=tmp_path),
        time_window=TimeWindow("2020-01-02", "2020-01-03"),
        processed_catalog=processed_catalog(),
        source_catalog=SimpleNamespace(
            get=lambda data_id: SimpleNamespace(data_id=data_id)
        ),
        resource_policy=object(),
    )

    assert tuple(result.columns) == (
        "market_id",
        "benchmark_id",
        "asset_id",
        "timestamp",
        "weight",
    )
    assert observed == [
        ("index_info", False, True),
        ("index_weights", False, False),
    ]


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
        normalize_benchmark_weight_snapshots(index_info, weights)

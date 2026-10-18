from __future__ import annotations

from decimal import ROUND_HALF_EVEN, Decimal

import numpy as np
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
from quant_research_platform.sources.csmar.interim import normalize_raw_frame
from tests.helpers.a_share_sources import _published_inputs, _source_catalog
from tests.helpers.paths import StorageLayout


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
    return (index_info, pd.DataFrame(rows))


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
    tmp_path, monkeypatch
):
    from quant_research_platform.runtime.dataframes import iter_complete_session_frames
    from quant_research_platform.storage.datasets.reading import (
        DatasetReadMode,
        read_dataset,
    )
    from quant_research_platform.storage.datasets.writing import write_dataset

    slabs = []
    read_frame = processed_producer.materialize_dataset_session

    def observed_read(*args):
        frame = read_frame(*args)
        if "constituent_asset_id" in frame:
            slabs.append(frame["timestamp"].nunique())
        return frame

    monkeypatch.setattr(
        processed_producer, "materialize_dataset_session", observed_read
    )
    monkeypatch.setattr(processed_producer, "DEFAULT_BATCH_ROWS", 256)
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
    for data_id, frame in (("index_info", index_info), ("index_weights", weights)):
        source_spec = _source_catalog().get(data_id)
        frame = frame.reindex(columns=source_spec.required_columns)
        write_dataset(
            layout.interim_data_root(source_spec.source_id, data_id),
            frame,
            spec=source_spec,
        )
    spec = processed_catalog().get("benchmark_weights")
    root = layout.processed_data_root(spec.market_id, spec.kind, spec.data_id)
    with processed_producer.produce_a_share_processed_data(
        spec,
        inputs=_published_inputs(layout, spec),
        time_window=TimeWindow(dates[0], dates[-1]),
        processed_catalog=processed_catalog(),
        source_catalog=_source_catalog(),
    ) as (frames, metadata):
        write_dataset(root, frames, spec=spec, metadata=metadata)
    actual = read_dataset(root)
    expected = pd.DataFrame(
        [
            ("a_share", benchmark, asset, date, weight)
            for benchmark, window in (("csi500", dates[:64]), ("csi1000", dates[64:]))
            for date in window
            for asset, weight in (("000001", 0.4), ("000002", 0.6))
        ],
        columns=("market_id", "benchmark_id", "asset_id", "timestamp", "weight"),
    )
    expected["timestamp"] = expected["timestamp"].astype("datetime64[ns]")
    expected["asset_id"] = expected["asset_id"].astype("object")
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    assert len(tuple((root / "data").rglob("*.parquet"))) > 1
    assert len(slabs) > 1 and sum(slabs) == len(dates)
    streamed = list(
        iter_complete_session_frames(
            read_dataset(root, mode=DatasetReadMode.CHUNKED),
            timestamp_column="timestamp",
        )
    )
    pd.testing.assert_frame_equal(pd.concat(streamed, ignore_index=True), expected)
    single = pd.concat(
        iter_benchmark_weight_snapshots(index_info, (weights.iloc[:128],)),
        ignore_index=True,
    )
    assert set(single.benchmark_id) == {"csi500"}


@pytest.mark.parametrize(
    ("mutation", "error"),
    (
        ("duplicate", "key_duplicate"),
        ("negative", "weight_invalid"),
        ("zero", "weight_invalid"),
        ("null", "weight_invalid"),
        ("missing", "weight_sum_invalid"),
        ("overflow", "weight_sum_invalid"),
        ("sum", "weight_sum_invalid"),
        ("identity", "index_identity_invalid"),
    ),
)
def test_benchmark_normalization_rejects_fact_mutations(
    mutation: str, error: str
) -> None:
    index_info, weights = _benchmark_facts()
    if mutation == "duplicate":
        weights = pd.concat([weights, weights.iloc[[0]]], ignore_index=True)
    elif mutation == "negative":
        weights.loc[0, "weight"] = -0.1
    elif mutation == "zero":
        weights.loc[0, "weight"] = 0.0
    elif mutation == "null":
        weights.loc[0, "weight"] = np.nan
    elif mutation == "missing":
        weights = weights.drop(index=1)
    elif mutation == "overflow":
        weights.loc[[0, 1], "weight"] = 1e308
        assert np.isfinite(weights["weight"]).all()
    elif mutation == "sum":
        weights.loc[1, "weight"] = 0.7
    else:
        index_info.loc[0, "index_name"] = "wrong"
    with pytest.raises(ValueError, match=error):
        pd.concat(
            iter_benchmark_weight_snapshots(index_info, (weights,)), ignore_index=True
        )


@pytest.mark.parametrize("raw_weight", (0.3333333, 0.3333334))
def test_rounded_benchmark_weights_publish_as_unit_portfolio(tmp_path, raw_weight):
    from quant_research_platform.research.portfolios.production import (
        _iter_benchmark_weights,
    )
    from quant_research_platform.storage.datasets.writing import write_dataset

    info, _ = _benchmark_facts()
    timestamp = pd.Timestamp("2020-01-02")
    raw = pd.DataFrame(
        [
            {
                "source_asset_id": code,
                "constituent_asset_id": asset,
                "effective_date": timestamp,
                "weight": raw_weight,
            }
            for code in ("000905", "000852")
            for asset in ("000001", "000002", "000003")
        ]
    )
    original = raw.copy(deep=True)
    normalized = pd.concat(
        iter_benchmark_weight_snapshots(info, (raw,)), ignore_index=True
    )
    spec = processed_catalog().get("benchmark_weights")
    root = StorageLayout(tmp_path).processed_data_root(
        spec.market_id, spec.kind, spec.data_id
    )
    write_dataset(root, normalized, spec=spec)
    for benchmark in ("csi500", "csi1000"):
        (weights,) = _iter_benchmark_weights(
            root, benchmark, pd.DatetimeIndex([timestamp])
        )
        np.testing.assert_array_max_ulp(
            weights.to_numpy(), np.full(3, 1.0 / 3.0), maxulp=1
        )
        np.testing.assert_array_max_ulp(
            np.array([(weights * 100.0).sum()]), np.array([100.0]), maxulp=2
        )
    pd.testing.assert_frame_equal(raw, original, check_exact=True)


@pytest.mark.parametrize("raw_weight", (0.333332, 0.333335))
def test_benchmark_normalization_does_not_hide_invalid_total(raw_weight):
    info, _ = _benchmark_facts()
    weights = pd.DataFrame(
        [
            {
                "source_asset_id": code,
                "constituent_asset_id": asset,
                "effective_date": "2020-01-02",
                "weight": raw_weight,
            }
            for code in ("000905", "000852")
            for asset in ("000001", "000002", "000003")
        ]
    )
    with pytest.raises(ValueError, match="^a_share_benchmark_weight_sum_invalid$"):
        tuple(iter_benchmark_weight_snapshots(info, (weights,)))


def test_benchmark_normalization_preserves_unequal_relative_weights():
    info, _ = _benchmark_facts()
    frame = pd.DataFrame(
        [
            {
                "source_asset_id": code,
                "constituent_asset_id": asset,
                "effective_date": "2020-01-02",
                "weight": weight,
            }
            for code in ("000905", "000852")
            for asset, weight in (
                ("000001", 0.25),
                ("000002", 0.5),
                ("000003", 0.2500001),
            )
        ]
    )
    normalized = pd.concat(
        iter_benchmark_weight_snapshots(info, (frame,)), ignore_index=True
    )
    for _, group in normalized.groupby("benchmark_id"):
        weights = group.set_index("asset_id")["weight"]
        assert weights["000002"] == 2.0 * weights["000001"]
        assert weights["000003"] > weights["000001"]
        np.testing.assert_array_max_ulp(
            np.array([weights.sum()]), np.array([1.0]), maxulp=1
        )


@pytest.mark.parametrize(
    ("percentages", "accepted"),
    (
        (("10.0000", "90.0002"), True),
        (("10.0000", "90.0003"), False),
        (("10.0000", "20.0000", "30.0000", "39.9996"), True),
        (("10.0000", "20.0000", "30.0000", "39.9995"), False),
    ),
)
def test_source_percentage_quantization_boundary_uses_actual_group_size(
    percentages, accepted
):
    info, _ = _benchmark_facts()
    raw = pd.DataFrame(
        [
            {
                "Indexcd": code,
                "Enddt": "2020-01-02",
                "Stkcd": f"{position + 1:06d}",
                "Constdnme": "synthetic",
                "Weight": weight,
            }
            for code in ("000905", "000852")
            for position, weight in enumerate(percentages)
        ]
    )
    original = raw.copy(deep=True)
    interim = normalize_raw_frame(raw, "index_weights")
    if not accepted:
        with pytest.raises(ValueError, match="^a_share_benchmark_weight_sum_invalid$"):
            tuple(iter_benchmark_weight_snapshots(info, (interim,)))
    else:
        normalized = pd.concat(
            iter_benchmark_weight_snapshots(info, (interim,)), ignore_index=True
        )
        total_percent = sum(map(Decimal, percentages))
        expected = np.array(
            [float(Decimal(weight) / total_percent) for weight in percentages]
        )
        for _, group in normalized.groupby("benchmark_id"):
            np.testing.assert_array_max_ulp(
                group["weight"].to_numpy(), expected, maxulp=2
            )
            np.testing.assert_array_max_ulp(
                np.array([group["weight"].sum()]), np.array([1.0]), maxulp=2
            )
    pd.testing.assert_frame_equal(raw, original, check_exact=True)


@pytest.mark.parametrize("direction", (-1, 1))
def test_complete_500_and_1000_source_rounding_preserves_relative_weights(direction):
    info, _ = _benchmark_facts()
    rows = []
    expected = {}
    for code, benchmark, count in (
        ("000905", "csi500", 500),
        ("000852", "csi1000", 1000),
    ):
        base = Decimal(1) / count
        offset = direction * Decimal("0.0000004")
        exact = [base + offset] * (count - 1) + [base - (count - 1) * offset]
        assert sum(exact) == Decimal(1)
        percentages = [
            (weight * 100).quantize(Decimal("0.0001"), rounding=ROUND_HALF_EVEN)
            for weight in exact
        ]
        total_percent = sum(percentages)
        assert abs(total_percent / 100 - 1) > Decimal("0.000001")
        expected[benchmark] = np.array(
            [float(weight / total_percent) for weight in percentages]
        )
        rows.extend(
            (
                {
                    "Indexcd": code,
                    "Enddt": "2020-01-02",
                    "Stkcd": f"{position + 1:06d}",
                    "Constdnme": "synthetic",
                    "Weight": str(weight),
                }
                for position, weight in enumerate(percentages)
            )
        )
    raw = pd.DataFrame(rows)
    original = raw.copy(deep=True)
    interim = normalize_raw_frame(raw, "index_weights")
    normalized = pd.concat(
        iter_benchmark_weight_snapshots(info, (interim,)), ignore_index=True
    )
    assert normalized.groupby("benchmark_id").size().to_dict() == {
        "csi500": 500,
        "csi1000": 1000,
    }
    for benchmark, group in normalized.groupby("benchmark_id"):
        np.testing.assert_array_max_ulp(
            group["weight"].to_numpy(), expected[benchmark], maxulp=2
        )
        np.testing.assert_array_max_ulp(
            np.array([group["weight"].sum()]), np.array([1.0]), maxulp=2
        )
    pd.testing.assert_frame_equal(raw, original, check_exact=True)


@pytest.mark.parametrize("benchmark_id", ["csi500", "csi1000"])
def test_single_benchmark_requires_its_own_identity(benchmark_id, subtests):
    from quant_research_platform.markets.implementations.a_share.ids import (
        FORMAL_BENCHMARKS,
    )

    # The production mapping owns vendor identifiers and names.
    source_id = next(
        key for key, value in FORMAL_BENCHMARKS.items() if value[0] == benchmark_id
    )
    info = pd.DataFrame(
        {
            "source_asset_id": [source_id],
            "index_name": [FORMAL_BENCHMARKS[source_id][1]],
        }
    )
    weights = pd.DataFrame(
        {
            "source_asset_id": [source_id, source_id],
            "constituent_asset_id": ["000001", "000002"],
            "effective_date": pd.Timestamp("2024-01-02"),
            "weight": [0.4, 0.6],
        }
    )
    other_id = next(key for key in FORMAL_BENCHMARKS if key != source_id)
    for case, selected, name, error in (
        ("unselected absent", False, "absent", None),
        ("unselected unknown", False, None, None),
        ("unselected wrong", False, "wrong", None),
        ("selected absent", True, "absent", "index_identity_invalid"),
        ("selected unknown", True, None, "index_name_invalid"),
        ("selected wrong", True, "wrong", "index_identity_invalid"),
    ):
        with subtests.test(case=case):
            reference = info.iloc[:0] if selected else info.copy()
            if name != "absent":
                reference = pd.concat(
                    [
                        reference,
                        pd.DataFrame(
                            {
                                "source_asset_id": [
                                    source_id if selected else other_id
                                ],
                                "index_name": [name],
                            }
                        ),
                    ],
                    ignore_index=True,
                )
            if error is not None:
                with pytest.raises(ValueError, match=error):
                    list(iter_benchmark_weight_snapshots(reference, [weights]))
            else:
                result = pd.concat(
                    iter_benchmark_weight_snapshots(reference, [weights])
                )
                assert set(result.benchmark_id) == {benchmark_id}
                assert result.weight.tolist() == [0.4, 0.6]
    # A reference unused by an earlier block becomes required when consumed.
    reference = pd.concat(
        [info, pd.DataFrame({"source_asset_id": [other_id], "index_name": [None]})]
    )
    frames = iter_benchmark_weight_snapshots(
        reference, [weights, weights.assign(source_asset_id=other_id)]
    )
    assert set(next(frames).benchmark_id) == {benchmark_id}
    with pytest.raises(ValueError, match="index_name_invalid"):
        next(frames)
    with pytest.raises(ValueError, match="coverage_incomplete"):
        list(iter_benchmark_weight_snapshots(info, [weights.iloc[:0]]))

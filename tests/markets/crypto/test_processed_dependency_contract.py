from __future__ import annotations

import inspect
from pathlib import Path

import pandas as pd
import pytest

from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.implementations.crypto.processed import (
    materialization_validation,
    materialize_factors,
    materialize_labels,
    materializer,
)
from quant_research_platform.markets.implementations.crypto.processed.catalog import (
    processed_catalog,
)
from quant_research_platform.storage._dataset_types import DatasetInspection


def test_input_dataset_identities_distinguish_source_and_manifest() -> None:
    first = _inspection(
        "market_bars_1m",
        spec_extra={"source_id": "source_a"},
        manifest_extra={"row_count": 3},
    )
    other_source = _inspection(
        "market_bars_1m",
        spec_extra={"source_id": "source_b"},
        manifest_extra={"row_count": 3},
    )
    changed_manifest = _inspection(
        "market_bars_1m",
        spec_extra={"source_id": "source_a"},
        manifest_extra={"row_count": 4},
    )

    first_identity = materialization_validation.input_dataset_identities(
        {"market_bars_1m": first}
    )
    other_source_identity = materialization_validation.input_dataset_identities(
        {"market_bars_1m": other_source}
    )
    changed_manifest_identity = materialization_validation.input_dataset_identities(
        {"market_bars_1m": changed_manifest}
    )

    assert first_identity != other_source_identity
    assert first_identity != changed_manifest_identity
    assert set(first_identity["market_bars_1m"]) == {
        "data_id",
        "spec_hash",
        "manifest_hash",
    }


def test_materialization_input_identity_validation_is_exact() -> None:
    source = _inspection("market_bars_1m")
    expected = materialization_validation.input_dataset_identities(
        {"market_bars_1m": source}
    )
    target = _inspection(
        "market_panel_1m",
        manifest_extra={"input_dataset_identities": expected},
    )

    materialization_validation.validate_materialization_input_identities(
        target,
        expected=expected,
        error="crypto_panel_input_identity_mismatch",
    )

    with pytest.raises(ValueError, match="crypto_panel_input_identity_mismatch"):
        materialization_validation.validate_materialization_input_identities(
            _inspection("market_panel_1m"),
            expected=expected,
            error="crypto_panel_input_identity_mismatch",
        )


@pytest.mark.parametrize(
    "mutation",
    ["missing_asset", "missing_middle", "extra", "market_mismatch", "duplicate"],
)
def test_exact_dependency_key_coverage_rejects_any_full_key_delta(
    mutation: str,
) -> None:
    expected = _key_frame()
    actual = expected.copy()
    if mutation == "missing_asset":
        actual = actual.loc[actual["asset_id"].ne("asset_b")].copy()
    elif mutation == "missing_middle":
        actual = actual.loc[
            ~(
                actual["asset_id"].eq("asset_a")
                & actual["timestamp"].eq(pd.Timestamp("2026-07-06 00:01"))
            )
        ].copy()
    elif mutation == "extra":
        actual = pd.concat(
            [
                actual,
                actual.iloc[[0]].assign(timestamp=pd.Timestamp("2026-07-06 00:03")),
            ],
            ignore_index=True,
        )
    elif mutation == "market_mismatch":
        actual.loc[0, "market_id"] = "other_market"
    else:
        actual = pd.concat([actual, actual.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="invalid_crypto_dependency_key_coverage"):
        materialization_validation.validate_exact_dependency_key_coverage(
            expected,
            actual,
            primary_key=("market_id", "asset_id", "timestamp"),
            error="invalid_crypto_dependency_key_coverage",
        )


def test_dependency_chunk_order_sorts_within_chunk_and_rejects_overlap() -> None:
    watermarks: dict[tuple[str, str], pd.Timestamp] = {}
    first = _key_frame().iloc[[2, 0, 1, 5, 3, 4]].copy()

    ordered = materialization_validation.validated_ordered_dependency_chunk(
        first,
        watermarks=watermarks,
        error="out_of_order_crypto_factor_dependency",
    )

    pd.testing.assert_frame_equal(
        ordered,
        first.sort_values(
            ["market_id", "asset_id", "timestamp"], kind="mergesort"
        ).reset_index(drop=True),
        check_dtype=False,
    )
    assert watermarks == {
        ("crypto", "asset_a"): pd.Timestamp("2026-07-06 00:02"),
        ("crypto", "asset_b"): pd.Timestamp("2026-07-06 00:02"),
    }

    with pytest.raises(ValueError, match="out_of_order_crypto_factor_dependency"):
        materialization_validation.validated_ordered_dependency_chunk(
            _key_frame().iloc[[2]].copy(),
            watermarks=watermarks,
            error="out_of_order_crypto_factor_dependency",
        )


@pytest.mark.parametrize(
    ("module_name", "error"),
    [
        ("factor", "out_of_order_crypto_factor_dependency"),
        ("label", "out_of_order_crypto_label_dependency"),
    ],
)
@pytest.mark.parametrize("second_start", [0, 4], ids=["reversed", "overlap"])
def test_factor_and_label_materializers_reject_non_monotonic_chunks(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    error: str,
    second_start: int,
) -> None:
    if second_start == 0:
        first = _panel_frame(start=5, minutes=5)
    else:
        first = _panel_frame(start=0, minutes=5)
    second = _panel_frame(start=second_start, minutes=5)
    spec = processed_catalog().get(
        "microstructure" if module_name == "factor" else "forward_returns"
    )
    module = materialize_factors if module_name == "factor" else materialize_labels
    monkeypatch.setattr(
        module,
        "read_dataset",
        lambda *_args, **_kwargs: iter((first, second)),
    )
    iterator = (
        module._iter_factor_chunks(
            spec,
            panel_root=Path("unused"),
            time_window=TimeWindow("2026-07-06 00:00", "2026-07-06 00:10"),
        )
        if module_name == "factor"
        else module._iter_label_chunks(
            spec,
            panel_root=Path("unused"),
            time_window=TimeWindow("2026-07-06 00:00", "2026-07-06 00:10"),
        )
    )

    with pytest.raises(ValueError, match=error):
        list(iterator)


@pytest.mark.parametrize(
    "module",
    [
        materialization_validation,
        materializer,
        materialize_factors,
        materialize_labels,
    ],
)
def test_crypto_materialization_hot_path_does_not_fingerprint_content(module) -> None:
    assert "fingerprint_dataset" not in inspect.getsource(module)


def _inspection(
    data_id: str,
    *,
    spec_extra: dict[str, object] | None = None,
    manifest_extra: dict[str, object] | None = None,
) -> DatasetInspection:
    spec = {"schema_id": "qrp-dataset-spec", "data_id": data_id}
    manifest = {"schema_id": "qrp-dataset-manifest", "data_id": data_id}
    spec.update(spec_extra or {})
    manifest.update(manifest_extra or {})
    return DatasetInspection(
        root=Path(data_id),
        data_root=Path(data_id) / "data",
        spec=spec,
        columns=("market_id", "asset_id", "timestamp"),
        schema_hash="schema",
        file_count=1,
        row_count=int(manifest.get("row_count", 0)),
        partitions=("year=2026",),
        partition_row_counts={"year=2026": int(manifest.get("row_count", 0))},
        date_min="2026-07-06",
        date_max="2026-07-06",
        manifest=manifest,
    )


def _key_frame() -> pd.DataFrame:
    timestamps = pd.date_range("2026-07-06", periods=3, freq="min")
    return pd.DataFrame(
        {
            "market_id": ["crypto"] * 6,
            "asset_id": ["asset_a"] * 3 + ["asset_b"] * 3,
            "timestamp": [*timestamps, *timestamps],
        }
    )


def _panel_frame(*, start: int, minutes: int) -> pd.DataFrame:
    timestamps = pd.date_range(
        "2026-07-06 00:00:00",
        periods=start + minutes,
        freq="min",
    )[start:]
    return pd.DataFrame(
        {
            "market_id": ["crypto"] * minutes,
            "asset_id": ["btc_usdt"] * minutes,
            "timestamp": timestamps,
            "open": [100.0] * minutes,
            "high": [101.0] * minutes,
            "low": [99.0] * minutes,
            "close": [100.5] * minutes,
            "volume": [10.0] * minutes,
            "quote_volume": [1005.0] * minutes,
            "trade_count": [2.0] * minutes,
            "trade_flow_observed": [True] * minutes,
            "taker_buy_volume": [4.0] * minutes,
            "taker_sell_volume": [6.0] * minutes,
            "signed_volume": [1.0] * minutes,
            "best_bid": [100.0] * minutes,
            "best_ask": [100.2] * minutes,
            "quote_observed": [True] * minutes,
            "quote_age_seconds": [1.0] * minutes,
            "bid_quantity": [1.5] * minutes,
            "ask_quantity": [1.0] * minutes,
            "bid_depth": [1.5] * minutes,
            "ask_depth": [1.0] * minutes,
            "mid_price": [100.1] * minutes,
            "eligible": [True] * minutes,
        }
    )

"""Selected observations keep nominal HFT windows and shared causal context."""

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.markets.implementations.crypto.hft import (
    iter_factor_frames,
    iter_label_frames,
)
from quant_research_platform.markets.implementations.crypto.ids import (
    HFT_CROSS_ASSET_DATA_ID,
    HFT_FACTOR_DATA_ID,
    HFT_INFORMATION_STATE_DATA_ID,
    HFT_LABEL_DATA_ID,
    HFT_PANEL_DATA_ID,
)
from quant_research_platform.markets.implementations.crypto.processed import producer
from quant_research_platform.markets.prepare import prepare
from quant_research_platform.research.protocols.crypto_hft import factor_catalog
from quant_research_platform.storage.datasets.reading import read_dataset
from tests.markets.crypto.test_local_data_dependencies import _panel, _persist, _window


@pytest.mark.parametrize("selection", ["rows", "files"])
@pytest.mark.parametrize("data_id", [HFT_FACTOR_DATA_ID, HFT_LABEL_DATA_ID])
def test_selected_hole_equals_full_clock_unknowns(
    tmp_path, monkeypatch, selection, data_id
):
    panel = pd.concat(_panel(count=70), ignore_index=True)
    panel["selected"] = True
    panel.loc[10, "selected"] = False
    # One row per file permits an actual native member selection, with no price reread.
    from quant_research_platform.storage.datasets import writing

    with monkeypatch.context() as source_writer:
        if selection == "files":
            source_writer.setattr(writing, "DEFAULT_BATCH_ROWS", 1)
        root = _persist(tmp_path / "source", panel)
    chosen = panel.selected
    source = {"path": str(root), "years": [1970], "time_window": _window(200, 6500)}
    if selection == "rows":
        source["row_filters"] = [["selected", "==", True]]
    else:
        members = sorted(root.glob("data/**/*.parquet"))
        assert len(members) == len(panel)
        source["files"] = [
            str(path.relative_to(root))
            for index, path in enumerate(members)
            if index != 10
        ]
    actual_chunks = producer.iter_dataset_chunks
    monkeypatch.setattr(
        producer,
        "iter_dataset_chunks",
        lambda session, **kwargs: actual_chunks(session, batch_rows=3),
    )
    actual = read_dataset(
        prepare(
            {
                "market_id": "crypto",
                "data_id": data_id,
                "parameters": {"asset_ids": ["btc_usdt"]},
                "inputs": {HFT_PANEL_DATA_ID: source},
                "time_window": _window(200, 6500),
                "output_root": str(tmp_path / "output"),
            }
        )
    )
    oracle = panel.loc[
        panel.timestamp.between(
            pd.Timestamp(200, unit="ms"), pd.Timestamp(6500, unit="ms")
        )
    ].copy()
    dependencies = {
        column
        for item in factor_catalog().list()
        if item.family == "microstructure"
        for column in item.input_columns
    }
    dependencies.add("best_bid_qty")
    oracle.loc[~oracle.selected, list(dependencies)] = np.nan
    transform = (
        iter_factor_frames
        if data_id == HFT_FACTOR_DATA_ID
        else lambda frames: iter_label_frames(
            frames, horizons_ms=(100, 500, 1000, 5000)
        )
    )
    expected = pd.concat(transform([oracle]), ignore_index=True)
    selected_times = panel.loc[chosen, "timestamp"]
    expected = expected.loc[
        expected.timestamp.isin(selected_times), actual.columns
    ].reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False, check_exact=True)
    assert actual.timestamp.is_unique
    assert pd.Timestamp(1000, unit="ms") not in set(actual.timestamp)
    if data_id == HFT_FACTOR_DATA_ID:
        assert actual.spread.notna().all()
        assert (
            actual.loc[actual.timestamp.eq(pd.Timestamp(1400, unit="ms")), "ofi_500ms"]
            .isna()
            .all()
        )
        assert (
            actual.loc[actual.timestamp.eq(pd.Timestamp(1500, unit="ms")), "ofi_500ms"]
            .notna()
            .all()
        )


@pytest.mark.parametrize(
    "milliseconds,error",
    [
        ([0, 0], "non_monotonic"),
        ([100, 0], "non_monotonic"),
        ([0, 150], "time_invalid"),
    ],
)
@pytest.mark.parametrize(
    "transform",
    [iter_factor_frames, lambda frames: iter_label_frames(frames, horizons_ms=(100,))],
)
def test_clock_validation_keeps_alignment_order_and_unique_times(
    milliseconds, error, transform
):
    frame = pd.concat(_panel(count=1), ignore_index=True)
    frame["timestamp"] = pd.to_datetime(milliseconds, unit="ms")
    if error == "non_monotonic":
        frame.loc[1, "segment_id"] = "next-native-segment"
    with pytest.raises(ValueError, match=error):
        list(transform([frame.iloc[:1], frame.iloc[1:]]))


def test_shared_cross_context_matches_independent_targets_with_selections(
    tmp_path, monkeypatch
):
    btc = pd.concat(_panel(count=20), ignore_index=True)
    eth = btc.copy()
    eth["asset_id"] = "eth_usdt"
    target = pd.concat([btc, eth], ignore_index=True).sort_values(
        ["timestamp", "asset_id"]
    )
    target["selected"] = ~(
        (target.asset_id == "eth_usdt")
        & target.timestamp.eq(pd.Timestamp(700, unit="ms"))
    )
    state = pd.concat(iter_factor_frames([btc]), ignore_index=True)
    state["asset_id"] = "sol_usdt"
    state["selected"] = ~state.timestamp.eq(pd.Timestamp(1000, unit="ms"))
    targets = _persist(tmp_path / "targets", target)
    information = _persist(tmp_path / "information", state)
    columns = ["sol_return_100ms", "sol_lagged_return_100ms", "sol_tfi_500ms"]

    def compute(assets, name):
        return read_dataset(
            prepare(
                {
                    "market_id": "crypto",
                    "data_id": HFT_CROSS_ASSET_DATA_ID,
                    "factor_columns": columns,
                    "parameters": {
                        "target_assets": assets,
                        "information_assets": ["sol_usdt"],
                        "max_age_ms": 200,
                        "input_delay_ms": 100,
                    },
                    "inputs": {
                        key: {
                            "path": str(path),
                            "row_filters": [["selected", "==", True]],
                            "years": [1970],
                            "time_window": _window(200, 1700),
                        }
                        for key, path in [
                            (HFT_PANEL_DATA_ID, targets),
                            (HFT_INFORMATION_STATE_DATA_ID, information),
                        ]
                    },
                    "time_window": _window(400, 1800),
                    "output_root": str(tmp_path / name),
                }
            )
        )

    chunks = producer.iter_dataset_chunks
    monkeypatch.setattr(
        producer,
        "iter_dataset_chunks",
        lambda session, **kwargs: chunks(session, batch_rows=3),
    )
    combined = compute(["eth_usdt", "btc_usdt"], "combined")
    separate = pd.concat(
        [compute([asset], asset) for asset in ("btc_usdt", "eth_usdt")],
        ignore_index=True,
    ).sort_values(["timestamp", "asset_id"], ignore_index=True)
    pd.testing.assert_frame_equal(combined, separate, check_exact=True)
    assert set(combined.asset_id) == {"btc_usdt", "eth_usdt"}
    assert combined.timestamp.max() == pd.Timestamp(1700, unit="ms")
    assert (
        combined.loc[
            combined.timestamp.eq(pd.Timestamp(1100, unit="ms")), "sol_grid_at"
        ]
        .eq(pd.Timestamp(900, unit="ms"))
        .all()
    )
    assert (
        combined.loc[
            combined.timestamp.eq(pd.Timestamp(1200, unit="ms")),
            "sol_lagged_return_100ms",
        ]
        .isna()
        .all()
    )

"""Scientific native selection keeps production chronology and exact keys."""

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.contracts.data import DataSpec
from quant_research_platform.markets.implementations.crypto.ids import (
    HFT_CROSS_ASSET_DATA_ID,
    HFT_INFORMATION_PANEL_DATA_ID,
    HFT_INFORMATION_STATE_DATA_ID,
    HFT_PANEL_DATA_ID,
    HFT_SOL_PANEL_DATA_ID,
    HFT_SPOT_CONTEXT_DATA_ID,
    HFT_SPOT_PANEL_DATA_ID,
)
from quant_research_platform.markets.implementations.crypto.processed import producer
from quant_research_platform.markets.prepare import prepare
from quant_research_platform.storage.datasets.reading import read_dataset
from quant_research_platform.storage.datasets.writing import write_dataset
from tests.markets.crypto.test_local_data_dependencies import _panel

KEYS = ["market_id", "asset_id", "timestamp"]


def _persist(root, chunks):
    roles = dict.fromkeys(chunks[0].columns, "context")
    roles.update(market_id="key", asset_id="key", timestamp="timestamp")
    write_dataset(
        root,
        chunks,
        spec=DataSpec(
            data_id="chronology_fixture",
            frequency="tick",
            grain="asset_timestamp",
            primary_key=tuple(KEYS),
            partition_columns=("year",),
            column_roles=roles,
        ),
    )
    return root


def _selection(root, reverse):
    members = sorted(root.glob("data/year=*/part-*.parquet"))
    assert len(members) == 2
    return {
        "path": str(root),
        "files": [
            str(path.relative_to(root)) for path in members[:: -1 if reverse else 1]
        ],
    }


def _prepare(root, data_id, inputs, frame, **options):
    return prepare(
        {
            "market_id": "crypto",
            "data_id": data_id,
            "inputs": inputs,
            "time_window": {
                "start": frame.timestamp.min().isoformat(),
                "end": frame.timestamp.max().isoformat(),
            },
            "output_root": str(root),
            **options,
        }
    )


@pytest.mark.parametrize("kind", ["cross", "spot", "union"])
def test_public_prepare_restores_selected_native_member_chronology(
    tmp_path, monkeypatch, kind
):
    # Six points retain a valid current/lag pair and a wholly future second part.
    frame = pd.concat(_panel(count=5), ignore_index=True)
    frame["transaction_time_ms"] = pd.array(
        [None, 2**63 - 1, 2**53 + 1, None, 7, 8], dtype="Int64"
    )
    target = frame.iloc[:3].assign(mid=100.0)
    source = frame.assign(mid=200.0 + np.arange(6))
    monkeypatch.setattr(producer, "DEFAULT_BATCH_ROWS", 3)
    if kind == "cross":
        source["asset_id"] = "eth_usdt"
        panel_root = _persist(tmp_path / "panel", [source.iloc[:3], source.iloc[3:]])
        information = _prepare(
            tmp_path / "information",
            HFT_INFORMATION_STATE_DATA_ID,
            {HFT_PANEL_DATA_ID: str(panel_root)},
            frame,
            parameters={"asset_ids": ["eth_usdt"]},
        )
        data_id, source_id = HFT_CROSS_ASSET_DATA_ID, HFT_INFORMATION_STATE_DATA_ID
        options = {
            "factor_columns": ["eth_return_100ms", "eth_lagged_return_100ms"],
            "parameters": {
                "information_assets": ["eth_usdt"],
                "target_assets": ["btc_usdt"],
                "max_age_ms": 200,
            },
        }
    else:
        if kind == "union":
            target = pd.concat(
                [target, target.assign(asset_id="eth_usdt")], ignore_index=True
            ).sort_values(["timestamp", "asset_id"], ignore_index=True)
            source["asset_id"] = "sol_usdt"
            data_id, source_id = HFT_INFORMATION_PANEL_DATA_ID, HFT_SOL_PANEL_DATA_ID
            options = {}
        else:
            data_id, source_id = HFT_SPOT_CONTEXT_DATA_ID, HFT_SPOT_PANEL_DATA_ID
            options = {
                "factor_columns": ["spot_perp_basis"],
                "parameters": {"max_age_ms": 200},
            }
        information = _persist(
            tmp_path / "information", [source.iloc[:3], source.iloc[3:]]
        )
    target_root = _persist(tmp_path / "target", [target.iloc[:2], target.iloc[2:]])
    if kind == "spot":
        ordinary = read_dataset(
            target_root, files=_selection(target_root, True)["files"]
        )
        assert ordinary.timestamp.tolist() == target.timestamp.iloc[[2, 0, 1]].tolist()
    for target_reverse, source_reverse in ((False, False), (False, True), (True, True)):
        result = read_dataset(
            _prepare(
                tmp_path / f"output-{target_reverse}-{source_reverse}",
                data_id,
                {
                    HFT_PANEL_DATA_ID: _selection(target_root, target_reverse),
                    source_id: _selection(information, source_reverse),
                },
                frame,
                **options,
            )
        )
        expected = (
            pd.concat([target, source], ignore_index=True).sort_values(
                ["timestamp", "asset_id"], ignore_index=True
            )
            if kind == "union"
            else target
        )
        assert list(result[KEYS].itertuples(index=False, name=None)) == list(
            expected[KEYS].itertuples(index=False, name=None)
        )
        if kind == "cross":
            np.testing.assert_allclose(
                result.eth_return_100ms,
                [np.nan, 1 / 200, 1 / 201],
                rtol=2e-12,
                atol=1e-14,
                equal_nan=True,
            )
            np.testing.assert_allclose(
                result.eth_lagged_return_100ms,
                [np.nan, np.nan, 1 / 200],
                rtol=2e-12,
                atol=1e-14,
                equal_nan=True,
            )
            assert result.eth_grid_at.tolist() == target.timestamp.tolist()
            assert result.eth_grid_age_ms.tolist() == [0.0] * 3
        elif kind == "spot":
            np.testing.assert_allclose(
                result.spot_perp_basis, [1.0, 1.01, 1.02], rtol=2e-12, atol=1e-14
            )
            assert result.spot_grid_at.tolist() == target.timestamp.tolist()
        else:
            pd.testing.assert_series_equal(
                result.transaction_time_ms,
                expected.transaction_time_ms,
                check_exact=True,
            )
            # The union is a lossless copy, not a recalculated scientific oracle.
            pd.testing.assert_frame_equal(
                result, expected[result.columns], check_exact=True, check_dtype=False
            )


@pytest.mark.parametrize("kind", ["cross", "spot", "union"])
def test_public_prepare_rejects_read_nonmonotonic_context_before_publication(
    tmp_path, kind
):
    frame = pd.concat(_panel(count=2), ignore_index=True)
    target = _persist(tmp_path / "target", [frame])
    bad = frame.iloc[[1, 0, 2]].copy()
    if kind == "cross":
        bad = bad[KEYS].assign(asset_id="eth_usdt", return_100ms=0.25)
        data_id, source_id = HFT_CROSS_ASSET_DATA_ID, HFT_INFORMATION_STATE_DATA_ID
        options = {
            "factor_columns": ["eth_return_100ms"],
            "parameters": {
                "information_assets": ["eth_usdt"],
                "target_assets": ["btc_usdt"],
                "max_age_ms": 200,
            },
        }
    elif kind == "spot":
        data_id, source_id = HFT_SPOT_CONTEXT_DATA_ID, HFT_SPOT_PANEL_DATA_ID
        options = {
            "factor_columns": ["spot_perp_basis"],
            "parameters": {"max_age_ms": 200},
        }
    else:
        bad["asset_id"] = "sol_usdt"
        data_id, source_id = HFT_INFORMATION_PANEL_DATA_ID, HFT_SOL_PANEL_DATA_ID
        options = {}
    # Spot exercises the consumed batch boundary; the others the current batch.
    chunks = [bad.iloc[:1], bad.iloc[1:]] if kind == "spot" else [bad]
    source = _persist(tmp_path / "source", chunks)
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="hft_.*non_monotonic"):
        _prepare(
            output,
            data_id,
            {HFT_PANEL_DATA_ID: str(target), source_id: str(source)},
            frame,
            **options,
        )
    assert not output.exists()

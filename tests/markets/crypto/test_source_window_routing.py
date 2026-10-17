"""Archive routing clips the future while retaining book initialization."""

import json

import pandas as pd
import pytest

from quant_research_platform.markets.implementations.crypto.ids import HFT_PANEL_DATA_ID
from quant_research_platform.markets.prepare import prepare
from quant_research_platform.sources.tardis.raw import import_slices
from quant_research_platform.storage.datasets.reading import read_dataset
from tests.helpers.tardis import _line, _slices
from tests.sources.tardis.test_funding import _publish_funding_archive
from tests.sources.tardis.test_replay import _delta


def test_panel_routes_end_bound_without_losing_snapshot_prefix(tmp_path):
    native = _delta(1, 99, 101, 98) + _line(
        2,
        "depthSnapshot",
        {
            "lastUpdateId": 100,
            "bids": [[str(100 - i), "2"] for i in range(10)],
            "asks": [[str(101 + i), "2"] for i in range(10)],
        },
    )
    for second in (3, 4):
        native += _delta(second, 99 + second, 99 + second, 98 + second)
    import_slices(
        tmp_path / "market",
        date="2026-09-01",
        slices=_slices(tmp_path / "native", native),
    )
    future = _publish_funding_archive(tmp_path, "2026-09-02", include_funding=False)
    for member in json.loads((future / "manifest.json").read_text())["files"]:
        (future / member["path"]).unlink()

    def run(paths, output, end="2026-09-01T00:00:04"):
        return read_dataset(
            prepare(
                {
                    "market_id": "crypto",
                    "data_id": HFT_PANEL_DATA_ID,
                    "inputs": {},
                    "source": {"root": str(tmp_path), "archive_paths": paths},
                    "parameters": {"asset_ids": ["btc_usdt"]},
                    "time_window": {
                        "start": "2026-09-01T00:00:03",
                        "end": end,
                    },
                    "output_root": str(tmp_path / output),
                }
            )
        )

    expected = run(["market"], "expected")
    actual = run(["market", future.name], "actual")
    assert not actual.empty
    assert actual.timestamp.min() == pd.Timestamp("2026-09-01T00:00:03.1")
    assert actual.timestamp.max() == pd.Timestamp("2026-09-01T00:00:04")
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)

    with pytest.raises(ValueError, match="^tardis_raw_identity_changed$"):
        run(["market", future.name], "boundary", end="2026-09-02T00:00:00")

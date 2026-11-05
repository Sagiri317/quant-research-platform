"""Saved public pages retain units, missingness and explicit source clocks."""

import json

import pandas as pd
import pytest

from quant_research_platform.sources.macro.history import (
    merge_macro,
    parse_hkma_daily_page,
    prepare_hkma_history,
)
from quant_research_platform.sources.macro.interim import (
    _frame,
    interim_catalog,
    validate_observations,
)
from quant_research_platform.storage.datasets import read_dataset, write_dataset

_PAGE = """<html><body><table>
<tr><td>Date and Time</td><td>18:30, 19/09/2019</td></tr>
<tr><td>Base Rate</td><td>2.25%</td></tr>
<tr><td>Today's Overnight HIBOR</td><td>-</td></tr>
<tr><td>HKAB 1-Month HIBOR Fixing</td><td>1.99286</td></tr>
<tr><td>TWI</td><td>99</td></tr>
<tr><td>Opening Aggregate Balance</td><td>54,285 mn</td></tr>
<tr><td>Closing Aggregate Balance</td><td>0 mn</td></tr>
<tr><td>Change Attributable to</td><td>-</td></tr>
</table><script>9999年1月末社会融资规模存量为999万亿元</script></body></html>"""


def test_saved_hkma_prepare_and_official_source_merge(tmp_path):
    raw, pages, official, merged = (
        tmp_path / name for name in ("raw", "pages", "official", "merged")
    )
    raw.mkdir()
    (raw / "20190919.html").write_text(_PAGE, encoding="utf-8")
    manifest = [
        {
            "observation_date": "2019-09-19",
            "status": "parsed",
            "responses": [
                {
                    "file": "20190919.html",
                    "retrieved_at": "2026-11-05T00:00Z",
                    "url": "https://www.hkma.gov.hk/example",
                }
            ],
        },
        {
            "observation_date": "2019-09-20",
            "status": "HTTPStatusError",
            "responses": [],
        },
    ]
    (raw / "download.json").write_text(json.dumps(manifest), encoding="utf-8")
    prepare_hkma_history({"raw_root": str(raw), "output": str(pages)})
    with pytest.raises(ValueError, match="dataset_exists"):
        prepare_hkma_history({"raw_root": str(raw), "output": str(pages)})
    saved = read_dataset(pages).set_index("series_id")
    assert saved.loc["hk_hibor_1m", "value"] == pytest.approx(1.99286)
    assert pd.isna(saved.loc["hk_hibor_on", "value"])
    assert saved.loc["hk_balance_open", "value"] == 54285
    assert saved.loc["hk_balance_close", "value"] == 0
    assert saved.timestamp.eq(pd.Timestamp("2019-09-19T15:59:59")).all()
    assert saved.availability_basis.eq("inferred_snapshot_release").all()
    assert saved.published_at.isna().all()

    api = _frame(
        "hk_hibor_1m",
        [1.75],
        ["2019-09-19"],
        timezone="Asia/Hong_Kong",
        record={"retrieved_at": "2026-11-05T00:00Z", "url": "https://api.hkma.gov.hk/"},
    )
    write_dataset(
        official,
        validate_observations(api),
        spec=interim_catalog().get("macro_observations"),
    )
    merge_macro({"inputs": [str(pages), str(official)], "output": str(merged)})
    result = read_dataset(merged).set_index("series_id")
    assert len(result) == 5
    assert result.loc["hk_hibor_1m", "value"] == 1.75
    assert result.loc["hk_hibor_1m", "source_id"] == "hkma"
    assert pd.isna(result.loc["hk_hibor_on", "value"])
    assert read_dataset(pages).set_index("series_id").loc[
        "hk_hibor_1m", "value"
    ] == pytest.approx(1.99286)


def test_hkma_missing_page_does_not_invent_an_observation():
    assert parse_hkma_daily_page("<h1>Not found</h1>") == {
        "section_times": [],
        "values": {},
    }

"""Member ranks retain contract units, source versions and explicit clocks."""

import hashlib
import json

import pandas as pd

from quant_research_platform.sources.prepare import prepare_source
from quant_research_platform.storage.datasets.reading import read_dataset
from tests.helpers.public_sources import write_source_manifest


def test_member_csv_keeps_versions_unknown_quantities_and_existing_clock_bound(
    tmp_path,
):
    raw = tmp_path / "raw"
    raw.mkdir()
    first, revision = "a" * 64, "b" * 64
    rows = [
        (
            "IF1101                        ",
            "20110104",
            "0",
            "1",
            "0001      ",
            "IF      ",
            "35675",
            "-5498",
            first,
        ),
        (
            "IC1505                        ",
            "20150416",
            "0",
            "1",
            "0133      ",
            "IC      ",
            "15817",
            "15817",
            first,
        ),
        (
            "IH1505                        ",
            "20150416",
            "0",
            "1",
            "0133      ",
            "IH      ",
            "29394",
            "29394",
            first,
        ),
        ("IF2401", "20240102", "2", "1", "0018", "IF", "", "--", first),
        ("IF2401", "20240102", "2", "1", "0018", "IF", "105", "2", revision),
        ("IF2401", "20240102", "3", "1", "0088", "IF", "50", "4", revision),
        ("TF2403", "not-a-date", "2", "1", "0001", "TF", "invalid", "x", first),
        ("IF2403", "20240201", "2", "1", "0001", "IF", "invalid", "x", first),
    ]
    frame = pd.DataFrame(
        rows,
        columns=(
            "instrumentid",
            "tradingday",
            "datatypeid",
            "rank",
            "partyid",
            "productid",
            "volume",
            "varvolume",
            "source_sha256",
        ),
    )
    frame["source_url"] = frame.source_sha256.map(
        {
            first: "https://example.org/original",
            revision: "https://example.org/revision",
        }
    )
    table = raw / "positions.csv.gz"
    frame.to_csv(table, index=False)
    write_source_manifest(
        raw,
        {
            "records": [
                {
                    "source": "cffex_member_rankings",
                    "path": table.name,
                    "sha256": hashlib.sha256(table.read_bytes()).hexdigest(),
                }
            ]
        },
    )
    output = tmp_path / "interim"
    prepare_source(
        {
            "source_id": "public_research",
            "raw_root": str(raw),
            "start_date": "2011-01-04",
            "end_date": "2024-01-02",
            "outputs": {"public_futures_member_positions": str(output)},
        }
    )
    result = read_dataset(output).set_index(["asset_id", "source_sha256"])
    assert len(result) == 6
    for asset, quantity, change, member, available in (
        ("IF1101:0:1", 35675, -5498, "0001", "2011-01-04T16:00:00Z"),
        ("IC1505:0:1", 15817, 15817, "0133", "2015-04-16T16:00:00Z"),
        ("IH1505:0:1", 29394, 29394, "0133", "2015-04-16T16:00:00Z"),
    ):
        early = result.loc[(asset, first)]
        assert (early.quantity, early.quantity_change, early.member_id) == (
            quantity,
            change,
            member,
        )
        assert early.contract_id == asset.split(":")[0]
        assert early.available_at == pd.Timestamp(available)
        assert early.availability_basis == "publication_date_upper_bound"
    before, after = (
        result.loc[("IF2401:2:1", first)],
        result.loc[("IF2401:2:1", revision)],
    )
    assert pd.isna(before.quantity) and pd.isna(before.quantity_change)
    assert (after.quantity, after.quantity_change) == (105, 2)
    assert after.available_at == pd.Timestamp("2024-01-02T16:00:00Z")
    assert after.availability_basis == "publication_date_upper_bound"
    assert before.source_url == "https://example.org/original"
    assert after.source_url == "https://example.org/revision"
    assert result.published_at.isna().all()
    metadata = json.loads((output / "_manifest/manifest.json").read_bytes())
    assert metadata["producer_diagnostics"]["source_invalid_numeric_cells"] == {
        "rank": 0,
        "quantity": 0,
        "quantity_change": 0,
    }

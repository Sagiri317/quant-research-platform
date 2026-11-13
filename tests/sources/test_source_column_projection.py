"""External source extensions must not alter consumed observations."""

import io
import zipfile

import pandas as pd
import pytest

from quant_research_platform.sources.macro.interim import materialize_interim_data
from quant_research_platform.storage.datasets.reading import read_dataset
from tests.helpers.public_sources import write_source_manifest


def _publish(tmp_path, provider, columns, values, *, name):
    raw = tmp_path / name / "raw"
    source = raw / "input"
    source.mkdir(parents=True)
    csv = pd.DataFrame([values], columns=columns).to_csv(index=False).encode()
    series = "us_fed_assets" if provider == "alfred" else "cn_treasury_3y"
    if provider == "alfred":
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("obs._by_real-time_period.csv", csv)
        payload, filename = buffer.getvalue(), "vintages.zip"
    else:
        payload, filename = csv, "curve.csv"
    (source / filename).write_bytes(payload)
    write_source_manifest(
        raw,
        {
            "sources": {
                "input": {
                    "source": provider,
                    "series_ids": [series],
                    "responses": [
                        {
                            "file": f"input/{filename}",
                            "retrieved_at": "2026-11-05T00:00Z",
                            "url": "https://example.org/source",
                        }
                    ],
                }
            }
        },
        name="download.json",
    )
    output = tmp_path / name / "interim"
    materialize_interim_data(raw_root=raw, output_roots={"macro_observations": output})
    return read_dataset(output)


@pytest.mark.parametrize("provider", ["alfred", "chinabond"])
def test_irrelevant_source_column_and_column_order_preserve_interim(tmp_path, provider):
    if provider == "alfred":
        columns = [
            "period_start_date",
            "WALCL",
            "realtime_start_date",
            "realtime_end_date",
        ]
        values = ["2024-01-03", 100, "2024-01-04", "9999-12-31"]
    else:
        columns = [
            "曲线名称",
            "日期",
            "3月",
            "6月",
            "1年",
            "3年",
            "5年",
            "7年",
            "10年",
            "30年",
        ]
        values = ["中债国债收益率曲线", "2024-01-03", 1, 1, 2.1, 2.3, 2, 2, 2.7, 3]
    baseline = _publish(tmp_path, provider, columns, values, name="baseline")
    extended = _publish(
        tmp_path,
        provider,
        ["source_note", *reversed(columns)],
        ["not a numeric observation", *reversed(values)],
        name="extended",
    )
    pd.testing.assert_frame_equal(extended, baseline)
    assert baseline.value.iloc[0] == (100 if provider == "alfred" else 2.3)


def test_chinabond_only_requires_the_selected_tenor(tmp_path):
    actual = _publish(
        tmp_path,
        "chinabond",
        ["日期", "3年", "曲线名称"],
        ["2024-01-03", 2.3, "中债国债收益率曲线"],
        name="selected",
    )
    assert actual.value.tolist() == [2.3]
    assert actual.series_id.tolist() == ["cn_treasury_3y"]
    assert actual.unit.tolist() == ["percent"]
    assert actual.published_at.isna().all()


@pytest.mark.parametrize("provider", ["alfred", "chinabond"])
def test_missing_consumed_source_column_remains_invalid(tmp_path, provider):
    columns = (
        ["period_start_date", "realtime_start_date", "realtime_end_date"]
        if provider == "alfred"
        else ["日期", "曲线名称"]
    )
    values = (
        ["2024-01-03", "2024-01-04", "9999-12-31"]
        if provider == "alfred"
        else ["2024-01-03", "中债国债收益率曲线"]
    )
    with pytest.raises(ValueError, match="columns_invalid"):
        _publish(tmp_path, provider, columns, values, name="missing")


@pytest.mark.parametrize("provider", ["alfred", "chinabond"])
def test_duplicate_consumed_source_column_remains_invalid(tmp_path, provider):
    if provider == "alfred":
        columns = [
            "period_start_date",
            "WALCL",
            "WALCL",
            "realtime_start_date",
            "realtime_end_date",
        ]
        values = ["2024-01-03", 100, 110, "2024-01-04", "9999-12-31"]
    else:
        columns = ["日期", "曲线名称", "3年", "3年"]
        values = ["2024-01-03", "中债国债收益率曲线", 2.3, 2.4]
    with pytest.raises(ValueError, match="columns_invalid"):
        _publish(tmp_path, provider, columns, values, name="duplicate")

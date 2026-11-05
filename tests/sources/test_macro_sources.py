"""Public macro imports preserve vintages, missing values, and source clocks."""

import io
import json
import zipfile

import httpx
import pandas as pd
import pytest

from quant_research_platform.sources.macro.download import fetch_macro
from quant_research_platform.sources.macro.interim import (
    _normalize_response,
    date_end_available,
    materialize_interim_data,
    normalize_alfred,
    read_release_csv,
    resolve_source_conflicts,
)
from quant_research_platform.sources.macro.publication import (
    estimate_initial_archive_times,
    estimated_release_at,
)
from quant_research_platform.storage.datasets.reading import read_dataset
from quant_research_platform.storage.errors import OutputAlreadyExistsError


def _vintage_zip():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(
            "obs._by_real-time_period.csv",
            "period_start_date,WALCL,realtime_start_date,realtime_end_date\n"
            "2014-01-01,100,2014-01-02,2014-02-05\n"
            "2014-01-01,110,2014-02-06,9999-12-31\n"
            "2014-01-08,.,2014-01-09,9999-12-31\n",
        )
        archive.writestr("README.txt", "Source: Federal Reserve; Units: Millions USD")
    return buffer.getvalue()


def test_archive_versions_are_not_backdated_or_zero_filled():
    frame = normalize_alfred(
        _vintage_zip(),
        "us_fed_assets",
        {
            "retrieved_at": "2026-11-05T00:00:00Z",
            "url": "https://alfred.stlouisfed.org/",
        },
    )
    assert frame.value.iloc[:2].tolist() == [100, 110]
    assert pd.isna(frame.value.iloc[2])
    assert frame.published_at.isna().all()
    assert frame.timestamp.iloc[0] == pd.Timestamp("2014-01-03T06:00:00Z")
    assert frame.timestamp.iloc[1] == pd.Timestamp("2014-02-07T06:00:00Z")
    assert frame.observation_date.iloc[0] == frame.observation_date.iloc[1]


def test_date_end_upper_bound_uses_source_dst():
    dates = pd.Series(["2024-03-09", "2024-03-10", "2024-11-03"])
    result = date_end_available(dates, "America/Chicago")
    assert result.tolist() == [
        pd.Timestamp(item)
        for item in ("2024-03-10T06:00Z", "2024-03-11T05:00Z", "2024-11-04T06:00Z")
    ]


def test_policy_surprise_keeps_units_and_declares_historical_snapshot_assumption():
    payload = b"Date,Surprise,Orthogonalized Surprise\n2023-12-13,-8.17,-5.32\n"
    result = _normalize_response(
        payload,
        ["us_fomc_surprise"],
        {"retrieved_at": "2026-11-05T00:00:00Z", "url": "https://www.frbsf.org/"},
    )[0]
    assert result.value.iloc[0] == pytest.approx(-8.17)
    assert result.unit.iloc[0] == "bp"
    assert result.timestamp.iloc[0] == pd.Timestamp("2023-12-13T23:00:00Z")
    assert result.availability_basis.iloc[0] == "inferred_snapshot_release"
    assert result.published_at.isna().all()


def test_public_download_and_native_source_roundtrip(tmp_path, monkeypatch):
    requests = []

    def respond(request):
        requests.append(request)
        if request.method == "GET":
            return httpx.Response(
                200,
                text=(
                    '<input id="form_obs_start_date" value="2002-12-18">'
                    '<input id="form_obs_end_date" value="2026-11-04">'
                    '<select id="form_selected_vintage_dates">'
                    '<option value="2014-01-02">2014-01-02</option>'
                    '<option value="2014-02-06">2014-02-06</option></select>'
                ),
            )
        return httpx.Response(200, content=_vintage_zip())

    client = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: client(transport=httpx.MockTransport(respond), **kwargs),
    )
    raw, output = tmp_path / "raw", tmp_path / "interim"
    fetch_macro(
        {
            "raw_root": str(raw),
            "start_date": "2010-01-01",
            "end_date": "2025-12-31",
            "series_ids": ["us_fed_assets", "cn_dr007"],
            "workers": 2,
        }
    )
    manifest = json.loads((raw / "download.json").read_text(encoding="utf-8"))
    assert manifest["sources"]["us_fed_assets"]["status"] == "downloaded"
    assert manifest["sources"]["release_csv"]["status"] == "unavailable"
    assert len(requests) == 2
    assert b"form%5Bfile_type%5D=1" in requests[1].content
    result = materialize_interim_data(
        raw_root=raw, output_roots={"macro_observations": output}
    )
    assert result.skipped == ("cn_dr007",)
    frame = read_dataset(output)
    assert len(frame) == 3
    assert frame.value.isna().sum() == 1
    assert set(frame.value.dropna()) == {100, 110}
    assert frame.timestamp.min() == pd.Timestamp("2014-01-02T22:30:00")
    with pytest.raises(OutputAlreadyExistsError):
        fetch_macro(
            {
                "raw_root": str(raw),
                "start_date": "2010-01-01",
                "end_date": "2025-12-31",
                "series_ids": ["us_fed_assets"],
                "workers": 1,
            }
        )


def test_explicit_original_release_timestamp_and_timezone_requirement(tmp_path):
    path = tmp_path / "release.csv"
    row = {
        "series_id": "cn_dr007",
        "observation_date": "2024-01-04",
        "value": 1.8,
        "unit": "percent",
        "frequency": "D",
        "source_id": "cfets",
        "source_series": "DR007",
        "source_timezone": "Asia/Shanghai",
        "source_url": "https://www.chinamoney.com.cn/",
        "retrieved_at": "2026-11-05T00:00Z",
        "published_at": "2024-01-04T18:00:00+08:00",
    }
    pd.DataFrame([row]).to_csv(path, index=False)
    data = read_release_csv(path)
    assert data.timestamp.iloc[0] == pd.Timestamp("2024-01-04T10:00Z")
    assert data.published_at.iloc[0] == data.timestamp.iloc[0]
    row["published_at"] = "2024-01-04T18:00:00"
    pd.DataFrame([row]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="timezone_required"):
        read_release_csv(path)


def test_overlapping_archive_blocks_are_resolved_before_backdating(tmp_path):
    raw, output = tmp_path / "raw", tmp_path / "interim"
    source = raw / "us_fed_assets"
    source.mkdir(parents=True)
    records = []
    for day in (4, 5):
        filename = f"vintages-{day}.zip"
        (source / filename).write_bytes(_vintage_zip())
        records.append(
            {
                "file": filename,
                "retrieved_at": f"2026-11-0{day}T00:00:00Z",
                "url": "https://alfred.stlouisfed.org/",
            }
        )
    (raw / "download.json").write_text(
        json.dumps(
            {
                "sources": {
                    "us_fed_assets": {
                        "status": "downloaded",
                        "source": "alfred",
                        "series_ids": ["us_fed_assets"],
                        "responses": records,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    materialize_interim_data(raw_root=raw, output_roots={"macro_observations": output})
    frame = read_dataset(output).sort_values(["observation_date", "timestamp"])
    assert len(frame) == 3
    assert frame.value.iloc[:2].tolist() == [100, 110]
    assert pd.isna(frame.value.iloc[2])
    assert frame.timestamp.iloc[:2].tolist() == [
        pd.Timestamp("2014-01-02T22:30"),
        pd.Timestamp("2014-02-07T06:00"),
    ]
    assert frame.retrieved_at.eq(pd.Timestamp("2026-11-05", tz="UTC")).all()


@pytest.mark.parametrize(
    "name,observation,expected",
    [
        ("us_fed_assets", "2024-07-03", "2024-07-05T21:30Z"),
        ("us_broad_dollar", "2024-05-24", "2024-05-28T21:15Z"),
        ("us_m2", "2024-01-01", "2024-02-27T23:00Z"),
        ("eu_ecb_assets", "2015-04-03", "2015-04-10T16:00Z"),
        ("jp_boj_assets", "2024-01-01", "2024-02-10T09:00Z"),
        ("cn_m2", "2024-01-01", "2024-02-20T10:00Z"),
        ("global_usd_credit", "2024-01-01", "2024-07-31T21:59:59Z"),
    ],
)
def test_buffered_publication_rules_use_history_holidays_and_local_clocks(
    name, observation, expected
):
    assert estimated_release_at(name, pd.Series([observation])).iloc[0] == pd.Timestamp(
        expected
    )


def test_initial_archive_estimate_preserves_later_revision_and_missing_value():
    frame = normalize_alfred(
        _vintage_zip(),
        "us_fed_assets",
        {
            "retrieved_at": "2026-11-05T00:00Z",
            "url": "https://alfred.stlouisfed.org/",
        },
    )
    actual = estimate_initial_archive_times(frame)
    assert actual.timestamp.iloc[0] == pd.Timestamp("2014-01-02T22:30Z")
    assert actual.timestamp.iloc[1] == frame.timestamp.iloc[1]
    assert actual.value.iloc[1] == 110
    assert pd.isna(actual.value.iloc[2])
    pd.testing.assert_series_equal(actual.vintage_date, frame.vintage_date)


def test_source_priority_is_per_observation_and_keeps_selected_revisions():
    frame = pd.DataFrame(
        {
            "series_id": ["cn_m2"] * 4 + ["hk_hibor_1m"] * 2,
            "observation_date": pd.to_datetime(
                ["2020-01-01"] * 3 + ["2020-02-01"] + ["2020-01-02"] * 2
            ),
            "timestamp": pd.to_datetime(
                [
                    "2020-02-20",
                    "2020-02-15",
                    "2020-02-21",
                    "2020-03-20",
                    "2020-01-03",
                    "2020-01-03",
                ],
                utc=True,
            ),
            "retrieved_at": pd.Timestamp("2026-11-05", tz="UTC"),
            "source_id": [
                "eastmoney",
                "pbc_government_reprint",
                "pbc_government_reprint",
                "eastmoney",
                "hkma",
                "hkma_historical_page",
            ],
            "value": [999.0, 1.0, 2.0, 3.0, 4.0, 999.0],
        }
    )
    actual = resolve_source_conflicts(frame)
    assert sorted(actual.value.tolist()) == [1.0, 2.0, 3.0, 4.0]


def test_official_shibor_and_chinabond_parsers_preserve_curve_and_nulls():
    record = {"retrieved_at": "2026-11-05T00:00Z", "url": "https://example.org/"}
    shibor = _normalize_response(
        json.dumps(
            {"records": [{"showDateCN": "2024-01-02", "1W": "1.8", "3M": None}]}
        ).encode(),
        ["cn_shibor_1w", "cn_shibor_3m"],
        record,
    )
    assert shibor[0].value.astype(float).iloc[0] == 1.8
    assert pd.isna(shibor[1].value.iloc[0])
    rows = [
        ["曲线名称", "日期", "3月", "6月", "1年", "3年", "5年", "7年", "10年", "30年"],
        [
            "中债国债收益率曲线",
            "2024-01-02",
            "1",
            "1",
            "2.1",
            "2.3",
            "2",
            "2",
            "2.7",
            "3",
        ],
        [
            "中债商业银行普通债收益率曲线(AAA)",
            "2024-01-02",
            "9",
            "9",
            "9",
            "9",
            "9",
            "9",
            "9",
            "9",
        ],
        [
            "中债中短期票据收益率曲线(AAA)",
            "2024-01-02",
            "3",
            "3",
            "3",
            "3.4",
            "3",
            "3",
            "3",
            "3",
        ],
    ]
    payload = (
        "<table>"
        + "".join(
            "<tr>" + "".join("<td>" + v + "</td>" for v in row) + "</tr>"
            for row in rows
        )
        + "</table>"
    ).encode()
    names = ["cn_treasury_1y", "cn_treasury_3y", "cn_treasury_10y", "cn_credit_aaa_3y"]
    actual = _normalize_response(payload, names, record)
    assert [float(f.value.iloc[0]) for f in actual] == [2.1, 2.3, 2.7, 3.4]


def test_hkma_recorded_api_failure_preserves_valid_pages_and_other_sources(
    tmp_path, monkeypatch
):
    def respond(request):
        if request.url.host == "api.hkma.gov.hk":
            if request.url.params["from"] == "2014-01-01":
                return httpx.Response(
                    200,
                    json={
                        "header": {"success": True},
                        "result": {
                            "records": [
                                {"end_of_date": "2014-01-02", "hibor_overnight": 0.1}
                            ]
                        },
                    },
                )
            return httpx.Response(200, json={"header": {"success": False}})
        if request.method == "GET":
            return httpx.Response(
                200,
                text=(
                    '<input id="form_obs_start_date" value="2002-12-18">'
                    '<input id="form_obs_end_date" value="2026-11-04">'
                    '<select id="form_selected_vintage_dates">'
                    '<option value="2014-01-02">2014-01-02</option></select>'
                ),
            )
        return httpx.Response(200, content=_vintage_zip())

    client = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kw: client(transport=httpx.MockTransport(respond), **kw),
    )
    raw, output = tmp_path / "raw", tmp_path / "output"
    fetch_macro(
        {
            "raw_root": str(raw),
            "start_date": "2014-01-01",
            "end_date": "2015-12-31",
            "series_ids": ["hk_hibor_on", "us_fed_assets"],
            "workers": 1,
        }
    )
    materialize_interim_data(raw_root=raw, output_roots={"macro_observations": output})
    result = read_dataset(output)
    assert set(result.series_id) == {"hk_hibor_on", "us_fed_assets"}
    assert result.loc[result.series_id.eq("hk_hibor_on"), "value"].tolist() == [0.1]
    from quant_research_platform.storage.datasets import inspect_dataset

    assert inspect_dataset(output).manifest["producer_diagnostics"]["incomplete"] == [
        "hk_hibor_on"
    ]


def test_cboe_download_parse_roundtrip_retains_null_and_uses_exchange_date(
    tmp_path, monkeypatch
):
    def respond(request):
        if "VIX9D" in request.url.path:
            return httpx.Response(
                200, text="DATE,OPEN,HIGH,LOW,CLOSE\n11/11/2024,1,3,1,2\n"
            )
        return httpx.Response(200, text="DATE,VVIX\n03/08/2024,80\n03/11/2024,.\n")

    client = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kw: client(transport=httpx.MockTransport(respond), **kw),
    )
    raw, output = tmp_path / "raw", tmp_path / "output"
    fetch_macro(
        {
            "raw_root": str(raw),
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
            "series_ids": ["us_vvix", "us_vix9d"],
            "workers": 1,
        }
    )
    materialize_interim_data(raw_root=raw, output_roots={"macro_observations": output})
    frame = read_dataset(output).sort_values(["series_id", "observation_date"])
    vvix = frame.loc[frame.series_id.eq("us_vvix")]
    assert vvix.value.iloc[0] == 80
    assert pd.isna(vvix.value.iloc[1])
    assert vvix.timestamp.tolist() == [
        pd.Timestamp("2024-03-08T23:00"),
        pd.Timestamp("2024-03-11T22:00"),
    ]
    # Veterans Day is a federal holiday but an observed equity-options session.
    vix9d = frame.loc[frame.series_id.eq("us_vix9d")].iloc[0]
    assert vix9d.value == 2
    assert vix9d.timestamp == pd.Timestamp("2024-11-11T23:00")
    assert frame.published_at.isna().all()
    assert frame.availability_basis.eq("inferred_snapshot_release").all()

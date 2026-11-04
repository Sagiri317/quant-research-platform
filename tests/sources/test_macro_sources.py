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


def test_policy_surprise_units_do_not_imply_historical_availability():
    payload = b"Date,Surprise,Orthogonalized Surprise\n2023-12-13,-8.17,-5.32\n"
    result = _normalize_response(
        payload,
        ["us_fomc_surprise"],
        {"retrieved_at": "2026-11-05T00:00:00Z", "url": "https://www.frbsf.org/"},
    )[0]
    assert result.value.iloc[0] == pytest.approx(-8.17)
    assert result.unit.iloc[0] == "bp"
    assert result.timestamp.iloc[0] == pd.Timestamp("2026-11-05T00:00:00Z")
    assert result.availability_basis.iloc[0] == "retrieval_only"


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
    assert frame.timestamp.min() == pd.Timestamp("2014-01-03T06:00:00")
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

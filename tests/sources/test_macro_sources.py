"""Public macro imports preserve vintages, missing values, and source clocks."""

import hashlib
import io
import json
import zipfile

import pandas as pd
import pytest

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


def test_explicit_snapshot_clock_does_not_claim_publication_or_vintage(tmp_path):
    path = tmp_path / "snapshot.csv"
    row = dict(
        series_id="us_sofr",
        observation_date="2024-01-02",
        value=5.3,
        unit="percent",
        frequency="D",
        source_id="nyfed",
        source_series="SOFR",
        source_timezone="America/New_York",
        source_url="https://markets.newyorkfed.org/",
        retrieved_at="2026-11-11T00:00:00Z",
        estimated_available_at="2024-01-09T23:59:59-05:00",
    )
    pd.DataFrame([row]).to_csv(path, index=False)
    data = read_release_csv(path)
    assert data.timestamp.iloc[0] == pd.Timestamp("2024-01-10T04:59:59Z")
    assert data.published_at.isna().all()
    assert data.vintage_date.isna().all()
    assert data.availability_basis.iloc[0] == "inferred_snapshot_release"
    assert data.value.iloc[0] == 5.3
    row["published_at"] = row["estimated_available_at"]
    pd.DataFrame([row]).to_csv(path, index=False)
    actual = read_release_csv(path)
    assert actual.published_at.iloc[0] == actual.timestamp.iloc[0]
    assert actual.availability_basis.iloc[0] == "release_timestamp"


def test_merged_releases_select_clocks_per_row_without_inventing_publication(tmp_path):
    row = dict(
        series_id="cn_dr007",
        observation_date="2024-01-02",
        value="1.8",
        unit="percent",
        frequency="D",
        source_id="cfets",
        source_series="DR007",
        source_timezone="Asia/Shanghai",
        source_url="https://www.chinamoney.com.cn/",
        retrieved_at="2026-11-11T00:00Z",
    )
    rows = [
        dict(
            row,
            published_at="2024-01-02T18:00+08:00",
            estimated_available_at="2024-01-04T00:00+08:00",
        ),
        dict(row, release_date="2024-01-03"),
        dict(row, estimated_available_at="2024-01-05T00:00+08:00"),
    ]
    path = tmp_path / "observations.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    actual = read_release_csv(path).reset_index(drop=True)
    assert actual.timestamp.tolist() == [
        pd.Timestamp(v)
        for v in (
            "2024-01-02T10:00Z",
            "2024-01-03T16:00Z",
            "2024-01-04T16:00Z",
        )
    ]
    assert actual.availability_basis.tolist() == [
        "release_timestamp",
        "release_date_end",
        "inferred_snapshot_release",
    ]
    assert actual.published_at.notna().tolist() == [True, False, False]
    assert actual.vintage_date.notna().tolist() == [True, True, False]
    pd.DataFrame([row]).assign(estimated_available_at=None).to_csv(path, index=False)
    with pytest.raises(ValueError, match="macro_release_clock_missing"):
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
                "sha256": hashlib.sha256((source / filename).read_bytes()).hexdigest(),
                "bytes": (source / filename).stat().st_size,
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


def test_official_shibor_and_structured_chinabond_preserve_curve_and_nulls():
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
    payload = pd.DataFrame(rows[1:], columns=rows[0]).to_csv(index=False).encode()
    names = ["cn_treasury_1y", "cn_treasury_3y", "cn_treasury_10y", "cn_credit_aaa_3y"]
    actual = _normalize_response(payload, names, record)
    assert [float(f.value.iloc[0]) for f in actual] == [2.1, 2.3, 2.7, 3.4]


@pytest.mark.parametrize(
    "bad_field,bad_value",
    [
        ("value", "invalid"),
        ("unit", "wrong_unit"),
        ("published_at", "2024-02-04T18:00:00"),
    ],
)
def test_macro_release_checks_only_selected_observation_window(
    tmp_path, bad_field, bad_value
):
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
    later = row | {
        "observation_date": "2024-02-04",
        "published_at": "2024-02-04T18:00:00+08:00",
        bad_field: bad_value,
    }
    pd.DataFrame([row, later]).to_csv(path, index=False)
    output = tmp_path / "selected"
    materialize_interim_data(
        raw_root=tmp_path,
        output_roots={"macro_observations": output},
        release_csv=path,
        start_date="2024-01-01",
        end_date="2024-01-31",
    )
    result = read_dataset(output)
    assert result.observation_date.tolist() == [pd.Timestamp("2024-01-04")]
    assert result.timestamp.tolist() == [pd.Timestamp("2024-01-04T10:00:00")]
    assert result.value.tolist() == [1.8]
    invalid = tmp_path / "invalid"
    with pytest.raises(ValueError):
        materialize_interim_data(
            raw_root=tmp_path,
            output_roots={"macro_observations": invalid},
            release_csv=path,
            start_date="2024-02-01",
            end_date="2024-02-29",
        )
    assert not invalid.exists()


@pytest.mark.parametrize("provider", ["cboe", "alfred"])
def test_downloaded_macro_window_ignores_other_observations_invalid_facts(
    tmp_path, provider
):
    raw = tmp_path / "raw"
    source = raw / "observations"
    source.mkdir(parents=True)
    filename, series = "daily.csv", "us_vvix"
    payload = b"DATE,VVIX\n01/02/2024,20.0\n02/02/2024,invalid\n"
    if provider == "alfred":
        filename, series = "vintages.zip", "us_fed_assets"
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr(
                "obs._by_real-time_period.csv",
                "period_start_date,WALCL,realtime_start_date,realtime_end_date\n"
                "2024-01-02,20.0,2024-01-04,9999-12-31\n"
                "2024-02-02,30.0,invalid,9999-12-31\n",
            )
        payload = buffer.getvalue()
    (source / filename).write_bytes(payload)
    (raw / "download.json").write_text(
        json.dumps(
            {
                "sources": {
                    "observations": {
                        "status": "downloaded",
                        "source": provider,
                        "series_ids": [series],
                        "responses": [
                            {
                                "file": filename,
                                "retrieved_at": "2026-11-05T00:00:00Z",
                                "url": "https://example.org/vix",
                                "bytes": len(payload),
                                "sha256": hashlib.sha256(payload).hexdigest(),
                            }
                        ],
                    }
                }
            }
        )
    )
    output = tmp_path / "selected"
    materialize_interim_data(
        raw_root=raw,
        output_roots={"macro_observations": output},
        start_date="2024-01-01",
        end_date="2024-01-31",
    )
    result = read_dataset(output)
    assert result.observation_date.tolist() == [pd.Timestamp("2024-01-02")]
    assert result.value.tolist() == [20.0]
    assert result.timestamp.tolist() == [
        pd.Timestamp(
            "2024-01-03T22:30:00" if provider == "alfred" else "2024-01-02T23:00:00"
        )
    ]
    assert result.published_at.isna().all()
    if provider == "alfred":
        assert result.vintage_date.tolist() == [pd.Timestamp("2024-01-04")]
    with pytest.raises(ValueError):
        materialize_interim_data(
            raw_root=raw,
            output_roots={"macro_observations": tmp_path / "invalid"},
            start_date="2024-02-01",
            end_date="2024-02-29",
        )
    assert not (tmp_path / "invalid").exists()

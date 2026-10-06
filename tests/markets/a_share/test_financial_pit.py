from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.markets.implementations.a_share.fundamentals import (
    FINANCIAL_VALUE_COLUMNS,
    construct_ttm_events,
    financial_events_from_interim,
    materialize_pit_fundamentals,
)
from quant_research_platform.sources.choice.fields import FIRST_DISCLOSURE_COLUMNS
from quant_research_platform.sources.choice.interim import (
    first_disclosure_spec,
    normalize_first_disclosure_frame,
)


def test_financial_versions_use_next_session_and_preserve_old_event() -> None:
    records = (
        {
            "asset": "asset",
            "period": "2019-12-31",
            "corrected": False,
            "correction_date": pd.NaT,
            "version": "original",
            "value": 100.0,
        },
        {
            "asset": "asset",
            "period": "2019-12-31",
            "corrected": True,
            "correction_date": pd.Timestamp("2020-02-15"),
            "version": "correction@2020-02-15",
            "value": 110.0,
        },
    )
    interim = {
        data_id: _financial_frame(data_id, records)
        for data_id in FINANCIAL_VALUE_COLUMNS
    }
    disclosures = pd.DataFrame(
        {
            "asset_id": ["asset"],
            "report_period": [pd.Timestamp("2019-12-31")],
            "first_disclosure_date": [pd.Timestamp("2020-01-31")],
        }
    )
    sessions = pd.DatetimeIndex(
        pd.to_datetime(["2020-02-03", "2020-03-02", "2020-04-01"])
    )

    events = financial_events_from_interim(
        interim,
        disclosures,
        trading_sessions=sessions,
    )
    total_assets = events.loc[events["metric_id"].eq("total_assets")].sort_values(
        "available_at"
    )

    assert total_assets["version_identity"].tolist() == [
        "original",
        "correction@2020-02-15",
    ]
    assert total_assets["available_at"].tolist() == [sessions[0], sessions[1]]

    observations = pd.DataFrame(
        {
            "asset_id": ["asset", "asset"],
            "timestamp": [sessions[0], sessions[1]],
        }
    )
    pit = materialize_pit_fundamentals(
        observations,
        total_assets.loc[
            :,
            [
                "asset_id",
                "metric_id",
                "report_period",
                "available_at",
                "is_corrected",
                "version_identity",
                "value",
            ],
        ],
    )

    assert pit["value"].tolist() == [100.0, 110.0]
    assert pit["version_identity"].tolist() == [
        "original",
        "correction@2020-02-15",
    ]


def test_weekend_revisions_use_the_latest_public_version_on_monday() -> None:
    records = tuple(
        {
            "asset": "asset",
            "period": "2019-12-31",
            "corrected": day is not None,
            "correction_date": pd.Timestamp(day) if day else pd.NaT,
            "version": f"correction@{day}" if day else "original",
            "value": value,
        }
        for day, value in ((None, 100.0), ("2020-02-01", 110.0), ("2020-02-02", 120.0))
    )
    interim = {
        data_id: _financial_frame(data_id, records)
        for data_id in FINANCIAL_VALUE_COLUMNS
    }
    disclosures = pd.DataFrame(
        {
            "asset_id": ["asset"],
            "report_period": pd.to_datetime(["2019-12-31"]),
            "first_disclosure_date": pd.to_datetime(["2020-01-30"]),
        }
    )
    sessions = pd.DatetimeIndex(["2020-01-31", "2020-02-03", "2020-02-04"])
    events = financial_events_from_interim(
        interim, disclosures, trading_sessions=sessions
    )
    ttm = construct_ttm_events(events)
    cashflow = ttm.loc[ttm.metric_id.eq("operating_cash_flow")]
    assert cashflow.available_at.tolist() == list(sessions[:2])
    assert cashflow.ttm_value.tolist() == [100.0, 120.0]
    assert cashflow.version_identity.tolist() == ["original", "correction@2020-02-02"]
    assert interim["financial_cashflow"].operating_cash_flow.tolist() == [
        100.0,
        110.0,
        120.0,
    ]


def test_correction_after_observation_calendar_does_not_rewrite_past() -> None:
    records = (
        {
            "asset": "asset",
            "period": "2019-12-31",
            "corrected": False,
            "correction_date": pd.NaT,
            "version": "original",
            "value": 100.0,
        },
        {
            "asset": "asset",
            "period": "2019-12-31",
            "corrected": True,
            "correction_date": pd.Timestamp("2021-01-15"),
            "version": "correction@2021-01-15",
            "value": 110.0,
        },
    )
    interim = {
        data_id: _financial_frame(data_id, records)
        for data_id in FINANCIAL_VALUE_COLUMNS
    }
    disclosures = pd.DataFrame(
        {
            "asset_id": ["asset"],
            "report_period": [pd.Timestamp("2019-12-31")],
            "first_disclosure_date": [pd.Timestamp("2020-01-31")],
        }
    )

    events = financial_events_from_interim(
        interim,
        disclosures,
        trading_sessions=pd.DatetimeIndex(pd.to_datetime(["2020-02-03", "2020-03-02"])),
    )

    assert set(events["version_identity"]) == {"original"}


def test_disclosure_on_last_calendar_session_is_not_yet_available() -> None:
    records = (
        {
            "asset": "asset",
            "period": "2019-12-31",
            "corrected": False,
            "correction_date": pd.NaT,
            "version": "original",
            "value": 100.0,
        },
    )
    interim = {
        data_id: _financial_frame(data_id, records)
        for data_id in FINANCIAL_VALUE_COLUMNS
    }
    disclosures = pd.DataFrame(
        {
            "asset_id": ["asset"],
            "report_period": [pd.Timestamp("2019-12-31")],
            "first_disclosure_date": [pd.Timestamp("2020-02-03")],
        }
    )

    result = financial_events_from_interim(
        interim,
        disclosures,
        trading_sessions=pd.DatetimeIndex([pd.Timestamp("2020-02-03")]),
    )
    assert result.empty


def test_choice_disclosure_owner_keeps_the_canonical_three_field_key() -> None:
    assert first_disclosure_spec().primary_key == FIRST_DISCLOSURE_COLUMNS
    with pytest.raises(ValueError, match="duplicate_choice_first_disclosure_key"):
        normalize_first_disclosure_frame(
            pd.DataFrame(
                {
                    "asset_id": ["asset", "asset"],
                    "report_period": pd.to_datetime(["2019-12-31", "2019-12-31"]),
                    "first_disclosure_date": pd.to_datetime(
                        ["2020-01-31", "2020-02-01"]
                    ),
                }
            )
        )


def test_pit_exposes_beginning_quarter_and_prior_year_same_period() -> None:
    report_periods = pd.to_datetime(
        [
            "2019-03-31",
            "2019-06-30",
            "2019-12-31",
            "2020-03-31",
            "2020-06-30",
        ]
    )
    events = pd.DataFrame(
        {
            "asset_id": ["asset"] * len(report_periods),
            "metric_id": ["total_assets"] * len(report_periods),
            "report_period": report_periods,
            "available_at": pd.to_datetime(
                ["2019-04-01", "2019-07-01", "2020-01-02", "2020-04-01", "2020-07-01"]
            ),
            "is_corrected": False,
            "version_identity": "original",
            "value": [10.0, 20.0, 40.0, 50.0, 60.0],
        }
    )
    result = materialize_pit_fundamentals(
        pd.DataFrame(
            {
                "asset_id": ["asset"],
                "timestamp": [pd.Timestamp("2020-07-02")],
            }
        ),
        events,
    )

    row = result.iloc[0]
    assert row["value"] == 60.0
    assert row["beginning_report_period"] == pd.Timestamp("2019-06-30")
    assert row["beginning_value"] == 20.0
    assert row["prior_year_report_period"] == pd.Timestamp("2019-06-30")
    assert row["prior_year_value"] == 20.0
    assert row["prior_fy_report_period"] == pd.Timestamp("2019-12-31")
    assert row["prior_fy_value"] == 40.0


def test_old_period_revisions_update_dependencies_without_current_rollback() -> None:
    events = pd.DataFrame(
        {
            "asset_id": ["asset"] * 6,
            "metric_id": ["total_assets"] * 6,
            "report_period": pd.to_datetime(
                [
                    "2019-06-30",
                    "2019-12-31",
                    "2020-06-30",
                    "2019-12-31",
                    "2019-06-30",
                    "2020-06-30",
                ]
            ),
            "available_at": pd.to_datetime(
                [
                    "2020-01-02",
                    "2020-01-02",
                    "2020-08-03",
                    "2020-09-01",
                    "2020-10-01",
                    "2020-11-02",
                ]
            ),
            "is_corrected": [False, False, False, True, True, True],
            "version_identity": [
                "original",
                "original",
                "original",
                "correction@2020-09-01",
                "correction@2020-10-01",
                "correction@2020-11-02",
            ],
            "value": [50.0, 120.0, 80.0, 140.0, np.nan, np.nan],
        }
    )
    observations = pd.DataFrame(
        {
            "asset_id": ["asset"] * 6,
            "timestamp": pd.to_datetime(
                [
                    "2020-01-01",
                    "2020-08-03",
                    "2020-08-31",
                    "2020-09-01",
                    "2020-10-01",
                    "2020-11-02",
                ]
            ),
        }
    )
    before_late_revisions = materialize_pit_fundamentals(
        observations.iloc[:3], events.iloc[:3]
    )
    result = materialize_pit_fundamentals(observations, events)
    pd.testing.assert_frame_equal(
        result.iloc[:3], before_late_revisions, check_dtype=False
    )
    assert pd.isna(result.iloc[0].report_period)
    assert result.report_period.iloc[1:].eq(pd.Timestamp("2020-06-30")).all()
    assert result.available_at.iloc[1:5].eq(pd.Timestamp("2020-08-03")).all()
    assert result.version_identity.iloc[1:5].eq("original").all()
    np.testing.assert_allclose(
        result.value.to_numpy(dtype=float, na_value=np.nan),
        [np.nan, 80, 80, 80, 80, np.nan],
    )
    np.testing.assert_allclose(
        result.prior_year_value.to_numpy(dtype=float, na_value=np.nan),
        [np.nan, 50, 50, 50, np.nan, np.nan],
    )
    pd.testing.assert_series_equal(
        result.beginning_value, result.prior_year_value, check_names=False
    )
    assert result.prior_year_report_period.iloc[1:].eq(pd.Timestamp("2019-06-30")).all()
    np.testing.assert_allclose(
        result.prior_fy_value.to_numpy(dtype=float, na_value=np.nan),
        [np.nan, 120, 120, 140, 140, 140],
    )
    assert result.iloc[-1].version_identity == "correction@2020-11-02"


def test_financial_versions_fail_without_legal_disclosure_or_unique_identity() -> None:
    records = (
        {
            "asset": "asset",
            "period": "2019-12-31",
            "corrected": False,
            "correction_date": pd.NaT,
            "version": "original",
            "value": 100.0,
        },
    )
    interim = {
        data_id: _financial_frame(data_id, records)
        for data_id in FINANCIAL_VALUE_COLUMNS
    }
    missing_disclosure = pd.DataFrame(
        columns=["asset_id", "report_period", "first_disclosure_date"]
    )
    with pytest.raises(ValueError, match="financial_first_disclosure_missing"):
        financial_events_from_interim(
            interim,
            missing_disclosure,
            trading_sessions=pd.DatetimeIndex([pd.Timestamp("2020-02-03")]),
        )

    corrected = {
        **records[0],
        "corrected": True,
        "correction_date": pd.NaT,
        "version": "correction@2020-02-15",
    }
    corrected_interim = {
        data_id: _financial_frame(data_id, (corrected,))
        for data_id in FINANCIAL_VALUE_COLUMNS
    }
    disclosure = pd.DataFrame(
        {
            "asset_id": ["asset"],
            "report_period": [pd.Timestamp("2019-12-31")],
            "first_disclosure_date": [pd.Timestamp("2020-01-31")],
        }
    )
    with pytest.raises(ValueError, match="financial_correction_disclosure_missing"):
        financial_events_from_interim(
            corrected_interim,
            disclosure,
            trading_sessions=pd.DatetimeIndex([pd.Timestamp("2020-02-03")]),
        )

    corrected_with_first_date = {
        **records[0],
        "corrected": True,
        "correction_date": pd.Timestamp("2020-02-15"),
        "version": "correction@2020-02-15",
    }
    corrected_with_first_interim = {
        data_id: _financial_frame(data_id, (corrected_with_first_date,))
        for data_id in FINANCIAL_VALUE_COLUMNS
    }
    with pytest.raises(ValueError, match="financial_first_disclosure_missing"):
        financial_events_from_interim(
            corrected_with_first_interim,
            missing_disclosure,
            trading_sessions=pd.DatetimeIndex([pd.Timestamp("2020-02-03")]),
        )

    with pytest.raises(
        ValueError, match="financial_correction_before_first_disclosure"
    ):
        financial_events_from_interim(
            corrected_with_first_interim,
            disclosure.assign(first_disclosure_date=pd.Timestamp("2020-03-01")),
            trading_sessions=pd.DatetimeIndex(
                pd.to_datetime(["2020-02-03", "2020-03-02"])
            ),
        )

    with pytest.raises(
        ValueError, match="financial_correction_before_first_disclosure"
    ):
        financial_events_from_interim(
            corrected_with_first_interim,
            disclosure.assign(first_disclosure_date=pd.Timestamp("2020-02-15")),
            trading_sessions=pd.DatetimeIndex(
                pd.to_datetime(["2020-02-03", "2020-03-02"])
            ),
        )

    duplicate = {
        data_id: _financial_frame(data_id, (*records, records[0]))
        for data_id in FINANCIAL_VALUE_COLUMNS
    }
    with pytest.raises(ValueError, match="financial_version_key_duplicate"):
        financial_events_from_interim(
            duplicate,
            disclosure,
            trading_sessions=pd.DatetimeIndex([pd.Timestamp("2020-02-03")]),
        )


def test_ttm_uses_same_period_and_legal_correction_cutoff() -> None:
    events = pd.DataFrame(
        {
            "asset_id": ["asset"] * 5,
            "metric_id": ["operating_revenue"] * 5,
            "report_period": pd.to_datetime(
                [
                    "2019-06-30",
                    "2019-12-31",
                    "2019-12-31",
                    "2020-06-30",
                    "2020-06-30",
                ]
            ),
            "available_at": pd.to_datetime(
                [
                    "2020-01-02",
                    "2020-01-02",
                    "2020-09-01",
                    "2020-08-01",
                    "2020-10-01",
                ]
            ),
            "is_corrected": [False, False, True, False, True],
            "version_identity": [
                "original",
                "original",
                "correction@2020-09-01",
                "original",
                "correction@2020-10-01",
            ],
            "value": [50.0, 120.0, 140.0, 80.0, 90.0],
        }
    )

    result = construct_ttm_events(events)
    current = result.loc[
        result["report_period"].eq(pd.Timestamp("2020-06-30"))
    ].sort_values("available_at")

    assert current["version_identity"].tolist() == [
        "original",
        "derived@2020-09-01",
        "correction@2020-10-01",
    ]
    assert current["ttm_value"].tolist() == [150.0, 170.0, 180.0]

    without_same_period = events.loc[
        ~events["report_period"].eq(pd.Timestamp("2019-06-30"))
    ]
    assert pd.isna(
        construct_ttm_events(without_same_period)
        .loc[
            lambda frame: frame["report_period"].eq(pd.Timestamp("2020-06-30")),
            "ttm_value",
        ]
        .iloc[0]
    )


def test_latest_null_correction_does_not_fallback_to_an_older_value() -> None:
    events = pd.DataFrame(
        {
            "asset_id": ["asset"] * 5,
            "metric_id": ["operating_revenue"] * 5,
            "report_period": pd.to_datetime(
                [
                    "2019-06-30",
                    "2019-12-31",
                    "2020-06-30",
                    "2020-06-30",
                    "2019-06-30",
                ]
            ),
            "available_at": pd.to_datetime(
                [
                    "2020-01-02",
                    "2020-01-02",
                    "2020-08-01",
                    "2020-10-01",
                    "2020-09-01",
                ]
            ),
            "is_corrected": [False, False, False, True, True],
            "version_identity": [
                "original",
                "original",
                "original",
                "correction@2020-10-01",
                "correction@2020-09-01",
            ],
            "value": [50.0, 120.0, 80.0, 80.0, np.nan],
        }
    )

    result = construct_ttm_events(events)
    current = result.loc[result["report_period"].eq(pd.Timestamp("2020-06-30"))]

    assert current["ttm_value"].iloc[0] == 150.0
    assert pd.isna(current["ttm_value"].iloc[1])


def test_ttm_merges_dependency_corrections_at_one_cutoff() -> None:
    events = pd.DataFrame(
        {
            "asset_id": ["asset"] * 6,
            "metric_id": ["operating_revenue"] * 6,
            "report_period": pd.to_datetime(
                [
                    "2019-06-30",
                    "2019-12-31",
                    "2020-06-30",
                    "2020-06-30",
                    "2019-06-30",
                    "2019-12-31",
                ]
            ),
            "available_at": pd.to_datetime(
                [
                    "2020-01-02",
                    "2020-01-02",
                    "2020-08-01",
                    "2020-09-01",
                    "2020-10-01",
                    "2020-10-01",
                ]
            ),
            "is_corrected": [False, False, False, True, True, True],
            "version_identity": [
                "original",
                "original",
                "original",
                "correction@2020-09-01",
                "correction@2020-10-01",
                "correction@2020-10-01",
            ],
            "value": [50.0, 120.0, 80.0, 90.0, 60.0, 140.0],
        }
    )

    current = construct_ttm_events(events).loc[
        lambda frame: frame["report_period"].eq(pd.Timestamp("2020-06-30"))
    ]

    assert current["version_identity"].tolist() == [
        "original",
        "correction@2020-09-01",
        "derived@2020-10-01",
    ]
    assert current["ttm_value"].tolist() == [150.0, 160.0, 170.0]


def test_pit_rejects_same_availability_cross_version_ties() -> None:
    events = pd.DataFrame(
        {
            "asset_id": ["asset", "asset"],
            "metric_id": ["total_assets", "total_assets"],
            "report_period": pd.to_datetime(["2020-06-30", "2020-06-30"]),
            "available_at": pd.to_datetime(["2020-08-03", "2020-08-03"]),
            "is_corrected": [False, True],
            "version_identity": ["original", "correction@2020-08-03"],
            "value": [1.0, 2.0],
        }
    )

    with pytest.raises(ValueError, match="pit_revision_order_ambiguous"):
        materialize_pit_fundamentals(
            pd.DataFrame(
                {"asset_id": ["asset"], "timestamp": [pd.Timestamp("2020-08-04")]}
            ),
            events,
        )


def _financial_frame(
    data_id: str, records: tuple[dict[str, object], ...]
) -> pd.DataFrame:
    values = FINANCIAL_VALUE_COLUMNS[data_id]
    rows = []
    for record in records:
        rows.append(
            {
                "source_asset_id": record["asset"],
                "report_period": pd.Timestamp(record["period"]),
                "report_type": "A",
                "is_corrected": record["corrected"],
                "correction_disclosure_date": record["correction_date"],
                "version_identity": record["version"],
                **{column: record["value"] for column in values},
            }
        )
    return pd.DataFrame(rows)


@pytest.mark.parametrize("boundary", ("choice", "financial_events"))
def test_first_disclosure_before_report_period_is_rejected_before_pit(boundary):
    disclosure = pd.DataFrame(
        {
            "asset_id": ["asset"],
            "report_period": pd.to_datetime(["2021-12-31"]),
            "first_disclosure_date": pd.to_datetime(["2021-01-04"]),
        }
    )
    records = (
        {
            "asset": "asset",
            "period": "2021-12-31",
            "corrected": False,
            "correction_date": pd.NaT,
            "version": "original",
            "value": 123.0,
        },
    )
    with pytest.raises(ValueError, match="financial_disclosure_before_report_period"):
        if boundary == "choice":
            normalize_first_disclosure_frame(disclosure)
        else:
            financial_events_from_interim(
                {
                    data_id: _financial_frame(data_id, records)
                    for data_id in FINANCIAL_VALUE_COLUMNS
                },
                disclosure,
                trading_sessions=pd.DatetimeIndex(["2021-01-04", "2021-01-05"]),
            )
    disclosure["first_disclosure_date"] = disclosure["report_period"]
    normalized = normalize_first_disclosure_frame(disclosure)
    events = financial_events_from_interim(
        {
            data_id: _financial_frame(data_id, records)
            for data_id in FINANCIAL_VALUE_COLUMNS
        },
        normalized,
        trading_sessions=pd.DatetimeIndex(["2021-12-31", "2022-01-04", "2022-01-05"]),
    )
    assert events["available_at"].eq(pd.Timestamp("2022-01-04")).all()
    assert len(events) == sum(map(len, FINANCIAL_VALUE_COLUMNS.values()))


def test_late_original_dependency_recomputes_newer_ttm_without_lookahead():
    events = pd.DataFrame(
        {
            "asset_id": ["asset"] * 4,
            "metric_id": ["operating_revenue"] * 4,
            "report_period": pd.to_datetime(
                ["2019-03-31", "2020-03-31", "2019-12-31", "2019-03-31"]
            ),
            "available_at": pd.to_datetime(
                ["2019-04-25", "2020-04-21", "2020-04-27", "2020-05-04"]
            ),
            "is_corrected": [False, False, False, True],
            "version_identity": [
                "original",
                "original",
                "original",
                "correction@2020-05-04",
            ],
            "value": [10.0, 30.0, 100.0, 12.0],
        }
    )
    ttm = construct_ttm_events(events)
    ttm["value"] = ttm["ttm_value"]
    observed = materialize_pit_fundamentals(
        pd.DataFrame(
            {
                "asset_id": ["asset"] * 3,
                "timestamp": pd.to_datetime(["2020-04-22", "2020-04-28", "2020-05-05"]),
            }
        ),
        ttm,
    )
    assert pd.isna(observed.loc[0, "value"])
    assert observed.loc[1:, "value"].tolist() == [120.0, 118.0]
    assert observed.loc[1:, "report_period"].eq(pd.Timestamp("2020-03-31")).all()


@pytest.mark.parametrize("revision", (False, True))
@pytest.mark.parametrize(
    "disclosed",
    ("2020-01-22", "2020-01-23", "2020-02-01", "2020-02-03", "2020-02-04"),
)
def test_financial_event_window_clipping_and_calendar_extension(revision, disclosed):
    original = dict(
        asset="asset",
        period="2019-12-31",
        corrected=False,
        correction_date=pd.NaT,
        version="original",
        value=100.0,
    )
    records = (original,)
    if revision:
        records += (
            dict(
                asset="asset",
                period="2019-12-31",
                corrected=True,
                correction_date=pd.Timestamp(disclosed),
                version=f"correction@{disclosed}",
                value=110.0,
            ),
        )
    interim = {
        name: _financial_frame(name, records) for name in FINANCIAL_VALUE_COLUMNS
    }
    before = {name: frame.copy(deep=True) for name, frame in interim.items()}
    disclosure = pd.DataFrame(
        {
            "asset_id": ["asset"],
            "report_period": pd.to_datetime(["2019-12-31"]),
            "first_disclosure_date": pd.to_datetime(
                ["2020-01-21" if revision else disclosed]
            ),
        }
    )
    # The gap includes the Lunar New Year holiday; February 1 is a Saturday.
    calendar = pd.DatetimeIndex(
        ["2020-01-23", "2020-02-03", "2020-02-04", "2020-02-05"]
    )
    short = financial_events_from_interim(
        interim, disclosure, trading_sessions=calendar[:2]
    )
    extended = financial_events_from_interim(
        interim, disclosure, trading_sessions=calendar
    )
    expected = extended.loc[extended.available_at.le(calendar[1])].reset_index(
        drop=True
    )
    if expected.empty:
        assert short.empty
    else:
        pd.testing.assert_frame_equal(short, expected)
        short_ttm = construct_ttm_events(short)
        extended_ttm = construct_ttm_events(extended)
        pd.testing.assert_frame_equal(
            short_ttm,
            extended_ttm.loc[extended_ttm.available_at.le(calendar[1])].reset_index(
                drop=True
            ),
        )
    next_session = calendar[calendar > pd.Timestamp(disclosed)][0]
    selected = extended.loc[
        extended.version_identity.eq(
            f"correction@{disclosed}" if revision else "original"
        )
    ]
    assert not selected.empty and selected.available_at.eq(next_session).all()
    for name in before:
        pd.testing.assert_frame_equal(interim[name], before[name])


def test_last_session_revision_preserves_processed_pit_ttm_and_factor_prefix():
    from quant_research_platform.markets.implementations.a_share.processed import (
        producer,
    )
    from quant_research_platform.profiles.research import research_profile
    from quant_research_platform.research.factors.fundamental.calculations import (
        calculate_fundamental_factors,
    )

    records = (
        dict(
            asset="asset",
            period="2019-12-31",
            corrected=False,
            correction_date=pd.NaT,
            version="original",
            value=100.0,
        ),
        dict(
            asset="asset",
            period="2019-12-31",
            corrected=True,
            correction_date=pd.Timestamp("2020-02-03"),
            version="correction@2020-02-03",
            value=110.0,
        ),
    )
    calendar = pd.DatetimeIndex(["2020-01-31", "2020-02-03", "2020-02-04"])
    panel = pd.DataFrame(
        {
            "market_id": ["a_share"] * 3,
            "asset_id": ["asset"] * 3,
            "timestamp": calendar,
        }
    )
    spec = research_profile().processed.get("fundamental_pit")

    def produce(history, sessions):
        source = {
            name: _financial_frame(name, history) for name in FINANCIAL_VALUE_COLUMNS
        }
        source["financial_first_disclosure"] = pd.DataFrame(
            {
                "asset_id": ["asset"],
                "report_period": pd.to_datetime(["2019-12-31"]),
                "first_disclosure_date": pd.to_datetime(["2020-01-30"]),
            }
        )
        source["trading_calendar"] = pd.DataFrame(
            {
                "calendar_date": sessions,
                "source_market_type": 4,
                "is_trading_day": True,
            }
        )
        events = producer._financial_pit_events(source)
        facts = producer._fundamental_pit_frame(
            spec,
            source=source,
            panel=panel.loc[panel.timestamp.isin(sessions)],
            events=events,
        )
        facts["total_market_cap"] = 1000.0
        return events, facts, calculate_fundamental_factors(facts)

    original = produce(records[:1], calendar[:2])
    clipped = produce(records, calendar[:2])
    extended = produce(records, calendar)
    for expected, actual in zip(original, clipped, strict=True):
        pd.testing.assert_frame_equal(actual, expected)
    pd.testing.assert_frame_equal(
        extended[0]
        .loc[extended[0].available_at.le(calendar[1])]
        .reset_index(drop=True),
        clipped[0],
    )
    for full, prefix in zip(extended[1:], clipped[1:], strict=True):
        pd.testing.assert_frame_equal(full.iloc[:2].reset_index(drop=True), prefix)
    assert clipped[2].book_to_market.tolist() == [0.1, 0.1]
    assert extended[2].book_to_market.tolist() == [0.1, 0.1, 0.11]


@pytest.mark.parametrize("scale", [1.0, 1e308, 5e-324])
def test_financial_ttm_and_stock_mean_preserve_finite_results(scale) -> None:
    from quant_research_platform.markets.implementations.a_share.processed import (
        producer,
    )
    from quant_research_platform.profiles.research import research_profile

    events = pd.DataFrame(
        dict(
            asset_id="asset",
            metric_id="operating_revenue",
            report_period=pd.to_datetime(["2023-03-31", "2023-12-31", "2024-03-31"]),
            available_at=pd.to_datetime(["2023-05-01", "2024-03-01", "2024-05-01"]),
            is_corrected=False,
            version_identity="original",
            value=scale,
        )
    )
    ttm = construct_ttm_events(events)
    assert ttm.iloc[-1].ttm_value == scale
    observations = pd.DataFrame(
        dict(asset_id=["asset"], timestamp=pd.to_datetime(["2024-05-02"]))
    )
    materialized = materialize_pit_fundamentals(
        observations, ttm.assign(value=ttm.ttm_value)
    )
    assert materialized.iloc[0].value == scale
    stocks = pd.concat(
        [
            events.assign(metric_id=metric)
            for metric in ("total_assets", "parent_equity")
        ],
        ignore_index=True,
    )
    facts = producer._fundamental_pit_frame(
        research_profile().processed.get("fundamental_pit"),
        source={},
        panel=observations.assign(market_id="a_share"),
        events=stocks,
    )
    assert facts["average_total_assets"].iloc[0] == scale
    assert facts["average_parent_equity"].iloc[0] == scale


def test_financial_ttm_rejects_unrepresentable_final_value() -> None:
    events = pd.DataFrame(
        dict(
            asset_id="asset",
            metric_id="operating_revenue",
            report_period=pd.to_datetime(["2023-03-31", "2023-12-31", "2024-03-31"]),
            available_at=pd.to_datetime(["2023-05-01", "2024-03-01", "2024-05-01"]),
            is_corrected=False,
            version_identity="original",
            value=[-1e308, 1e308, 1e308],
        )
    )
    with pytest.raises(ValueError, match="invalid_ttm_value"):
        construct_ttm_events(events)


def test_empty_legal_financial_event_domain_persists_unknown_panel_rows(tmp_path):
    from quant_research_platform.markets.implementations.a_share.processed import (
        producer,
    )
    from quant_research_platform.profiles.research import research_profile
    from quant_research_platform.storage.datasets import read_dataset, write_dataset

    events = pd.DataFrame(
        columns=(
            "asset_id",
            "metric_id",
            "report_period",
            "available_at",
            "is_corrected",
            "version_identity",
            "value",
        )
    )
    panel = pd.DataFrame(
        {
            "market_id": "a_share",
            "asset_id": [asset for _ in range(3) for asset in ("a", "b", "c")],
            "timestamp": np.repeat(
                pd.bdate_range("2020-01-02", periods=3).as_unit("ns"), 3
            ),
        }
    )
    spec = research_profile().processed.get("fundamental_pit")
    actual = producer._fundamental_pit_frame(
        spec, source={}, panel=panel, events=construct_ttm_events(events)
    )
    from quant_research_platform.storage.layout import StorageLayout

    root = StorageLayout(tmp_path).processed_data_root(
        spec.market_id, spec.kind, spec.data_id
    )
    write_dataset(root, actual, spec=spec)
    restored = read_dataset(root)
    pd.testing.assert_frame_equal(restored[list(panel.columns)], panel)
    assert not restored.has_legal_fundamental_pit.any()
    assert (
        restored.drop(columns=[*panel.columns, "has_legal_fundamental_pit"])
        .isna()
        .all()
        .all()
    )


@pytest.mark.parametrize("size", [64, 128])
def test_ttm_annual_revisions_use_constant_time_known_period_lookup(size, monkeypatch):
    from quant_research_platform.markets.implementations.a_share import fundamentals

    events = pd.DataFrame(
        {
            "asset_id": "asset",
            "metric_id": "operating_revenue",
            "report_period": pd.Timestamp("2025-12-31"),
            "available_at": pd.date_range("2026-01-01", periods=size, freq="D"),
            "is_corrected": True,
            "version_identity": [f"revision-{number}" for number in range(size)],
            "value": [float(number) for number in range(size)],
        }
    )
    original = fundamentals._ttm_value
    sizes = []

    def counted(known, period):
        sizes.append(len(known))
        return original(known, period)

    monkeypatch.setattr(fundamentals, "_ttm_value", counted)
    result = construct_ttm_events(events)
    assert result.ttm_value.tolist() == events.value.tolist()
    assert sizes == [1] * size

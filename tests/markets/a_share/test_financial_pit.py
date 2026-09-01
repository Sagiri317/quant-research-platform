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


def test_disclosure_on_last_calendar_session_fails_with_availability_reason() -> None:
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

    with pytest.raises(ValueError, match="financial_availability_next_session_missing"):
        financial_events_from_interim(
            interim,
            disclosures,
            trading_sessions=pd.DatetimeIndex([pd.Timestamp("2020-02-03")]),
        )


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

"""Declared historical publication estimates for research snapshots."""

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from quant_research_platform.sources.macro.catalog import SERIES


def _us_business_day(dates, *, following=False):
    holidays = USFederalHolidayCalendar().holidays(
        dates.min() - pd.Timedelta(days=10), dates.max() + pd.Timedelta(days=20)
    )
    days = dates.to_numpy(dtype="datetime64[D]")
    if following:
        days = days + np.timedelta64(1, "D")
    return pd.Series(
        pd.to_datetime(
            np.busday_offset(
                days,
                0,
                roll="forward",
                holidays=holidays.to_numpy(dtype="datetime64[D]"),
            )
        ),
        index=dates.index,
    )


def estimated_release_at(name, observations):
    """Infer a buffered historical cutoff, never a retrieval-time cutoff.

    These are research assumptions, including for revised snapshots and ex-post
    policy factors. Explicit original releases take precedence at import.
    """
    dates = pd.to_datetime(observations).dt.normalize()
    if dates.empty:
        return pd.Series(pd.NaT, index=dates.index, dtype="datetime64[ns, UTC]")
    series = SERIES[name]
    zone, clock = "America/New_York", "18:00:00"
    if name in {
        "us_fed_assets",
        "us_tga_wed",
        "us_tga_weekly_mean",
        "us_reserves_wed",
        "us_reserves_weekly_mean",
    }:
        dates = _us_business_day(dates + pd.Timedelta(days=1))
        clock = "17:30:00"
    elif name in {
        "us_broad_dollar",
        "us_broad_dollar_legacy",
        "cn_usd_cny",
        "eu_usd_per_eur",
        "jp_jpy_per_usd",
    }:
        dates = _us_business_day(
            dates + pd.to_timedelta(7 - dates.dt.dayofweek, unit="D")
        )
        clock = "17:15:00"
    elif name in {"us_sofr", "us_effr"}:
        dates = _us_business_day(dates, following=True)
        clock = "15:30:00"
    elif name in {"us_fed_funds", "us_ig_oas", "us_hy_oas"}:
        dates = _us_business_day(dates, following=True)
    elif name in {"us_nfci", "us_anfci"}:
        dates = dates + pd.to_timedelta((2 - dates.dt.dayofweek) % 7, unit="D")
        holidays = USFederalHolidayCalendar().holidays(
            dates.min() - pd.Timedelta(days=3), dates.max()
        )
        shifted = np.logical_or.reduce(
            [(dates - pd.Timedelta(days=lag)).isin(holidays) for lag in (0, 1, 2)]
        )
        dates = dates + pd.to_timedelta(shifted.astype(int), unit="D")
        clock = "09:30:00"
    elif name == "us_m2":
        dates = dates + pd.offsets.MonthBegin(1)
        # Before the monthly H.6 schedule, use the fourth Thursday of next month.
        weekday = np.where(dates.ge("2021-02-01"), 1, 3)
        dates = _us_business_day(
            dates + pd.to_timedelta((weekday - dates.dt.dayofweek) % 7 + 21, unit="D")
        )
    elif name in {
        "us_vix",
        "us_vix3m",
        "us_vix9d",
        "us_vxn",
        "us_vxeem",
        "us_vvix",
        "us_skew",
        "us_ovx",
        "us_gvz",
    }:
        # Observed exchange sessions need not be federal business days.
        pass
    elif name.startswith("us_"):
        # Includes policy rates, daily Treasury yields, VIX and event-window shocks.
        if series.frequency != "event":
            dates = _us_business_day(dates)
    elif name in {"eu_ecb_assets", "eu_deposit_rate"}:
        zone = "Europe/Berlin"
        if name == "eu_ecb_assets":
            # Quarter-end/TARGET holidays can postpone the usual Tuesday release.
            dates = dates + pd.to_timedelta(11 - dates.dt.dayofweek, unit="D")
    elif name in {"jp_boj_assets", "jp_current_accounts"}:
        zone = "Asia/Tokyo"
        dates = (
            dates
            + pd.offsets.MonthBegin(1)
            + pd.Timedelta(days=9 if name == "jp_boj_assets" else 19)
        )
    elif name.startswith("hk_"):
        zone = "Asia/Hong_Kong"
        clock = "23:59:59"
    elif name == "global_usd_credit":
        zone = "Europe/Zurich"
        dates = dates + pd.offsets.MonthEnd(7)
        clock = "23:59:59"
    elif name in {"cn_m2_imf", "cn_interbank_3m_oecd", "cn_treasury_10y_oecd"}:
        dates = dates + pd.offsets.MonthEnd(3 if name == "cn_m2_imf" else 2)
    elif name.startswith("cn_"):
        zone = "Asia/Shanghai"
        if series.frequency == "M":
            dates = dates + pd.offsets.MonthBegin(1) + pd.Timedelta(days=19)
        elif name in {"cn_dr007", "cn_r007"}:
            clock = "21:00:00"
    else:
        raise ValueError("macro_publication_rule_missing")
    return (dates + pd.Timedelta(clock)).dt.tz_localize(zone).dt.tz_convert("UTC")


def estimate_initial_archive_times(frame):
    """Seed each observation before delayed archive enrollment; keep revisions.

    A backdated initial archived value is a revised-snapshot proxy, not a claim
    that its original vintage has been recovered.
    """
    frame = frame.copy()
    archived = frame.loc[frame.availability_basis.eq("archive_date_end")]
    first = archived.sort_values("timestamp", kind="stable").drop_duplicates(
        ["series_id", "observation_date"]
    )
    for name, rows in first.groupby("series_id", sort=False):
        estimates = estimated_release_at(name, rows.observation_date)
        earlier = estimates < rows.timestamp
        indices = rows.index[earlier]
        frame.loc[indices, "timestamp"] = estimates.loc[indices]
        frame.loc[indices, "availability_basis"] = "inferred_snapshot_release"
    return frame

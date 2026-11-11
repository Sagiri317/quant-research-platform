"""Point-in-time alignment of explicitly released, versioned observations."""

from collections import deque

import numpy as np
import pandas as pd


def rolling_available_at(clocks, window):
    """Latest dependency clock in a rolling numerical window, preserving NaT."""
    clocks = (
        pd.to_datetime(clocks, utc=True).dt.tz_localize(None).astype("datetime64[ns]")
    )
    values = clocks.to_numpy().view("int64")
    missing = np.isnat(clocks.to_numpy())
    result = np.full(len(values), np.datetime64("NaT"), dtype="datetime64[ns]")
    maxima = deque()
    unknown = 0
    for i, value in enumerate(values):
        unknown += missing[i]
        if i >= window:
            unknown -= missing[i - window]
        if not missing[i]:
            while maxima and values[maxima[-1]] <= value:
                maxima.pop()
            maxima.append(i)
        while maxima and maxima[0] <= i - window:
            maxima.popleft()
        if i + 1 >= window and not unknown:
            result[i] = clocks.iloc[maxima[0]].to_datetime64()
    return pd.Series(result, index=clocks.index)


def _visible_updates(observations, columns, by):
    keys = [*by, "timestamp", "available_at"]
    facts = observations.loc[:, [*keys, *columns]].copy()
    facts["available_at"] = (
        pd.to_datetime(facts.available_at, utc=True)
        .dt.tz_localize(None)
        .astype("datetime64[ns]")
    )
    facts["timestamp"] = pd.to_datetime(facts.timestamp).astype("datetime64[ns]")
    facts = facts.loc[facts.available_at.notna()]
    if by:
        facts = facts.loc[facts[list(by)].notna().all(axis=1)]
    if columns:
        grouped = facts.groupby(keys, sort=False, dropna=False)[list(columns)]
        facts = grouped.first().where(grouped.nunique(dropna=False).le(1)).reset_index()
    else:
        facts = facts.drop_duplicates(keys)
    facts = facts.sort_values(["available_at", "timestamp"], kind="stable")
    newest = (
        facts.groupby(list(by), sort=False, dropna=False).timestamp.cummax()
        if by
        else facts.timestamp.cummax()
    )
    return facts.loc[facts.timestamp.eq(newest)].drop_duplicates(
        [*by, "available_at"], keep="last"
    )


def latest_observations(observations, decisions, *, columns, by=("asset_id",)):
    """Select latest visible reports; older amendments never replace newer periods."""
    facts = _visible_updates(observations, columns, by)
    left = decisions.loc[:, [*by, "timestamp"]].copy()
    left["timestamp"] = (
        pd.to_datetime(left.timestamp, utc=True)
        .dt.tz_localize(None)
        .astype("datetime64[ns]")
    )
    left["_row"] = np.arange(len(left))
    for key in by:
        facts[key], left[key] = facts[key].astype("str"), left[key].astype("str")
    result = pd.DataFrame(
        index=decisions.index,
        columns=[*columns, "observation_date", "source_available_at"],
    )
    if facts.empty or left.empty:
        return result
    facts = facts.rename(
        columns={"timestamp": "observation_date", "available_at": "source_available_at"}
    )
    joined = pd.merge_asof(
        left.sort_values("timestamp", kind="stable"),
        facts.sort_values("source_available_at", kind="stable"),
        left_on="timestamp",
        right_on="source_available_at",
        by=list(by) or None,
        direction="backward",
    )
    return (
        joined.sort_values("_row")
        .loc[:, [*columns, "observation_date", "source_available_at"]]
        .set_axis(decisions.index)
    )


def iter_latest_report_keys(observations, cutoffs, *, entity):
    """Yield known report keys at sorted cutoffs without an entity-date product."""
    facts = _visible_updates(observations, (), (entity,))
    events = iter(
        facts[[entity, "timestamp", "available_at"]].itertuples(index=False, name=None)
    )
    event = next(events, None)
    state = {}
    for cutoff in pd.to_datetime(cutoffs, utc=True).tz_localize(None):
        while event is not None and event[2] <= cutoff:
            state[event[0]] = event[1:]
            event = next(events, None)
        frame = pd.DataFrame(
            [(key, *value) for key, value in state.items()],
            columns=[entity, "_report", "_available"],
        )
        frame["_report"] = pd.to_datetime(frame._report).astype("datetime64[ns]")
        frame["_available"] = pd.to_datetime(frame._available).astype("datetime64[ns]")
        frame["timestamp"] = cutoff
        yield frame

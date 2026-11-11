"""Point-in-time alignment of explicitly released, versioned observations."""

from collections import deque

import numpy as np
import pandas as pd


def rolling_available_at(clocks, window):
    """Latest dependency clock in a rolling numerical window, preserving NaT."""
    clocks = (
        pd.to_datetime(clocks, utc=True).dt.tz_localize(None).astype("datetime64[ns]")
    )
    timestamps = clocks.to_numpy()
    values = timestamps.view("int64")
    missing = np.isnat(timestamps)
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
            result[i] = timestamps[maxima[0]]
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


def observation_versions(frame, *, keys, columns):
    """Resolve simultaneous source conflicts, preserving real publication changes.

    Equal later copies do not postpone the first known value. Unknown release
    clocks remain unavailable and cannot revise a known observation.
    """
    coordinates = [*keys, "available_at"]
    grouped = frame.groupby(coordinates, sort=False, dropna=False)[list(columns)]
    versions = grouped.first().where(grouped.nunique(dropna=False).le(1)).reset_index()
    versions = versions.sort_values([*keys, "available_at"], kind="stable")
    previous = versions.groupby(list(keys), sort=False, dropna=False)[
        list(columns)
    ].shift()
    same = (
        versions[list(columns)].eq(previous)
        | (versions[list(columns)].isna() & previous.isna())
    ).all(axis=1)
    first = ~versions.duplicated(list(keys))
    return versions.loc[first | (~same & versions.available_at.notna())].reset_index(
        drop=True
    )


def revision_snapshots(inputs):
    """Yield the earliest observations once, then only genuine revision snapshots.

    Future first observations retain their own release clocks. Numerical callers
    still gate every dependency and decide the finite recalculation window.
    """
    snapshots, changes = {}, []
    for name, (frame, keys, columns) in inputs.items():
        versions = observation_versions(frame, keys=keys, columns=columns)
        first = ~versions.duplicated(list(keys))
        snapshots[name] = versions.loc[first].set_index(list(keys))
        revised = versions.loc[~first].copy()
        revised["_input"] = name
        changes.append(revised)
    yield None, {name: frame.reset_index() for name, frame in snapshots.items()}, {}
    revisions = pd.concat(changes, ignore_index=True) if changes else pd.DataFrame()
    if revisions.empty:
        return
    for released, batch in revisions.groupby("available_at", sort=True):
        updated = {}
        for name, block in batch.groupby("_input", sort=False):
            keys, columns = inputs[name][1:]
            rows = block.loc[:, [*keys, "available_at", *columns]].set_index(list(keys))
            snapshots[name].loc[rows.index, rows.columns] = rows
            updated[name] = rows.reset_index()
        yield (
            released,
            {name: frame.reset_index() for name, frame in snapshots.items()},
            updated,
        )


def combine_derived_versions(versions, *, columns, keys=("asset_id", "timestamp")):
    """Publish scalar dependency clocks, replacing computed ties at the producer."""
    pieces = []
    for column in columns:
        batches = []
        clock = f"{column}_available_at"
        prior_clocks = None
        for released, frame in versions:
            values = frame.loc[:, [*keys, column]].copy()
            values[clock] = frame.get(clock, frame.available_at)
            if released is not None:
                previous = prior_clocks.reindex(
                    pd.MultiIndex.from_frame(values[list(keys)])
                ).set_axis(values.index)
                withdrawn = values[clock].isna() & previous.notna()
                values.loc[withdrawn, column] = np.nan
                values[clock] = values[clock].fillna(previous)
                known = values[clock].notna()
                values.loc[known, clock] = values.loc[known, clock].clip(lower=released)
            batches.append(values)
            clocks = values.set_index(list(keys))[clock]
            prior_clocks = (
                clocks
                if prior_clocks is None
                else pd.concat([prior_clocks, clocks]).loc[
                    lambda x: ~x.index.duplicated(keep="last")
                ]
            )
        values = pd.concat(batches, ignore_index=True)
        values = values.drop_duplicates([*keys, clock], keep="last")
        values["_release"] = values[clock]
        pieces.append(values.set_index([*keys, "_release"]))
    result = pd.concat(pieces, axis=1).reset_index()
    result["available_at"] = result.pop("_release")
    return result.sort_values([*keys, "available_at"], kind="stable").reset_index(
        drop=True
    )

"""Align published daily risk facts to intraday decisions without changing raw X."""

import numpy as np
import pandas as pd

from quant_research_platform.contracts.time import (
    TimeWindow,
    time_window_timestamp_interval,
)
from quant_research_platform.contracts.validation import strict_timestamp_series
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.research.risk import FactorRiskState
from quant_research_platform.storage.datasets.reading import iter_dataset_chunks

_LIFE_KEYS = ["market_id", "asset_id", "listing_date"]


def risk_exposure_columns(model):
    return tuple(
        dict.fromkeys(
            (
                "industry_standard_id",
                "industry_level1_code",
                model.config.weight_column,
                *model.style_columns,
            )
        )
    )


def _risk_source(exposures, *, model, available_at_column, additional_columns=()):
    source = exposures.reindex(
        columns=list(
            dict.fromkeys(
                (
                    *_LIFE_KEYS,
                    "timestamp",
                    available_at_column,
                    *risk_exposure_columns(model),
                    *additional_columns,
                )
            )
        )
    ).copy()
    for column, target, error in (
        (available_at_column, "_available_at", "risk_exposure_availability_invalid"),
        ("timestamp", "_observed_at", "risk_exposure_timestamp_invalid"),
        ("listing_date", "listing_date", "risk_exposure_listing_invalid"),
    ):
        source[target] = strict_timestamp_series(
            source[column], error=error, allow_null=True
        )
    source = source.loc[
        source[_LIFE_KEYS].notna().all(axis=1)
        & source._available_at.notna()
        & source._observed_at.notna()
        & source._available_at.ge(source._observed_at)
        & source._observed_at.ge(source.listing_date)
    ].copy()
    for column in ("market_id", "asset_id"):
        source[column] = source[column].astype("string")
    # A date-only daily observation first becomes usable on the following date.
    next_day = source._observed_at.dt.normalize() + pd.Timedelta(days=1)
    source["_active_at"] = source._available_at.where(
        source._available_at.ge(next_day), next_day
    )
    return source


def _risk_frontier(source):
    source = source.sort_values(
        ["_active_at", "_observed_at", "_available_at"], kind="stable"
    )
    # Late publication of an older observation cannot rewind the known state.
    frontier = source.groupby(_LIFE_KEYS, sort=False)._observed_at.cummax()
    return source.loc[source._observed_at.eq(frontier)].drop_duplicates(
        [*_LIFE_KEYS, "_active_at"], keep="last"
    )


def read_asof_risk_facts(
    session, *, model, available_at_column, time_window, additional_columns=()
):
    """Keep each life's initial state and published updates in the decision window."""
    start, stop, closed = time_window_timestamp_interval(time_window)
    start, stop = pd.Timestamp(start), pd.Timestamp(stop)
    if not closed:
        stop -= pd.Timedelta(1, "ns")
    # Native date bounds prune future observation rows without imposing a lookback.
    first_date = session.manifest.get("date_min")
    if first_date is not None:
        last_observation = stop.normalize() - pd.Timedelta(1, "ns")
        session = session.filtered_view(
            TimeWindow(
                min(pd.Timestamp(first_date), last_observation), last_observation
            )
        )
    baseline = None
    updates = []
    with closing_iterator(iter_dataset_chunks(session)) as chunks:
        for frame in chunks:
            source = _risk_source(
                frame,
                model=model,
                available_at_column=available_at_column,
                additional_columns=additional_columns,
            )
            prior = source.loc[source._active_at.le(start)]
            baseline = pd.concat(
                (() if baseline is None else (baseline,)) + (prior,), ignore_index=True
            )
            # Physical partitions need not be time ordered. Choose the greatest
            # observation, then its last revision, across all consumed batches.
            baseline = baseline.sort_values(
                ["_observed_at", "_available_at"], kind="stable"
            ).drop_duplicates(_LIFE_KEYS, keep="last")
            updates.append(
                _risk_frontier(
                    source.loc[source._active_at.gt(start) & source._active_at.le(stop)]
                )
            )
    compact = _risk_frontier(pd.concat((baseline, *updates), ignore_index=True))
    compact[available_at_column] = compact._available_at
    return compact.loc[
        :,
        list(
            dict.fromkeys(
                (
                    *_LIFE_KEYS,
                    "timestamp",
                    available_at_column,
                    *risk_exposure_columns(model),
                    *additional_columns,
                )
            )
        ),
    ]


def asof_risk_frame(
    frame, exposures, *, model, available_at_column, additional_columns=(), cache=None
):
    """Align raw published facts, before fitting the decision's complete domain."""
    columns = tuple(dict.fromkeys((*risk_exposure_columns(model), *additional_columns)))
    cache = {} if cache is None else cache
    definition = (model, available_at_column, tuple(additional_columns))
    if cache.get("exposures") is not exposures or cache.get("definition") != definition:
        cache.clear()
        source = _risk_frontier(
            _risk_source(
                exposures,
                model=model,
                available_at_column=available_at_column,
                additional_columns=additional_columns,
            )
        )
        cache.update(exposures=exposures, definition=definition, source=source)
    source = cache["source"]
    left = frame.reindex(columns=[*_LIFE_KEYS, "timestamp", "eligible"]).copy()
    left["listing_date"] = strict_timestamp_series(
        left.listing_date, error="risk_exposure_listing_invalid", allow_null=True
    )
    left["timestamp"] = strict_timestamp_series(
        left.timestamp, error="risk_exposure_decision_invalid", allow_null=True
    )
    left["_row"] = np.arange(len(left))
    aligned = pd.DataFrame(index=np.arange(len(left)), columns=list(columns))
    for key in ("market_id", "asset_id"):
        left[key] = left[key].astype("string")
    decisions = left.loc[
        left[_LIFE_KEYS].notna().all(axis=1)
        & left.timestamp.notna()
        & left.timestamp.ge(left.listing_date)
    ]
    if not source.empty and not decisions.empty:
        joined = pd.merge_asof(
            decisions.sort_values("timestamp", kind="stable"),
            source.loc[:, [*_LIFE_KEYS, "_active_at", *columns]],
            left_on="timestamp",
            right_on="_active_at",
            by=_LIFE_KEYS,
            direction="backward",
            allow_exact_matches=True,
        )
        aligned.loc[joined._row, list(columns)] = joined.loc[
            :, list(columns)
        ].to_numpy()
    for column in (model.config.weight_column, *model.style_columns):
        aligned[column] = pd.to_numeric(aligned[column], errors="raise")
    aligned.index = frame.index
    left.index = frame.index
    return pd.concat((left.drop(columns="_row"), aligned), axis=1)


def prepare_asof_risk_sessions(
    frame,
    exposures,
    *,
    model,
    available_at_column,
    include_ineligible=False,
    cache=None,
):
    """Use only prior-day facts published by the decision, in the same stock life."""
    cache = {} if cache is None else cache
    cross = asof_risk_frame(
        frame,
        exposures,
        model=model,
        available_at_column=available_at_column,
        cache=cache,
    )
    state = FactorRiskState(style_columns=model.style_columns, config=model.config)
    sessions = {}
    for timestamp, group in cross.groupby("timestamp", sort=False):
        snapshot = (
            group.drop(columns="timestamp")
            .sort_values("asset_id")
            .reset_index(drop=True)
        )
        if (
            "snapshot" not in cache
            or cache["include_ineligible"] != include_ineligible
            or not cache["snapshot"].equals(snapshot)
        ):
            cache.update(
                snapshot=snapshot,
                include_ineligible=include_ineligible,
                prepared=state.prepare_exposure_session(
                    group, include_ineligible=include_ineligible
                ),
            )
        sessions[pd.Timestamp(timestamp)] = cache["prepared"]
    return sessions

"""Sample observed A-share quotes on a common, explicit decision clock."""

from pathlib import Path

import numpy as np
import pandas as pd

from quant_research_platform.contracts.datasets import ResearchDatasetSpec
from quant_research_platform.contracts.features import inherited_feature_transforms
from quant_research_platform.contracts.json_codec import _strict_config
from quant_research_platform.contracts.time import (
    TimeWindow,
    time_window_timestamp_interval,
)
from quant_research_platform.markets.implementations.a_share.calendar import (
    continuous_session,
)
from quant_research_platform.markets.implementations.a_share.stock_pool import (
    load_stock_pool,
    stock_pool_rows,
)
from quant_research_platform.research.labels.quote_returns import QuoteReturnLookup
from quant_research_platform.research.protocols import a_share_hft as protocol
from quant_research_platform.runtime.resources import DEFAULT_BATCH_ROWS
from quant_research_platform.storage.atomic import reject_if_exists
from quant_research_platform.storage.datasets import write_dataset
from quant_research_platform.storage.datasets.reading import (
    _dataset_input_options,
    iter_dataset_chunks,
    materialize_dataset_session,
    open_dataset_session,
)


def _sample_block(
    frame, domains, grid, roles, max_age, lookup, *, book_status, visibility_delay
):
    output, grids = [], {}
    groups = dict(tuple(frame.groupby("asset_id", sort=False)))
    for asset, segments in domains.items():
        quotes = groups.get(asset, frame.iloc[:0]).sort_values(
            "timestamp", kind="stable"
        )
        quotes = quotes.rename(columns={"timestamp": "quote_timestamp"}).copy()
        quotes["quote_timestamp"] = quotes.quote_timestamp.astype("datetime64[ns]")
        quotes["_quote_visible_at"] = quotes.quote_timestamp + visibility_delay
        ordered = sorted(segments.items(), key=lambda item: item[1][0])
        exchange, instrument = ordered[0][1][2:]
        key = (exchange, instrument)
        if key not in grids:
            grids[key] = (
                continuous_session(
                    pd.Series(grid),
                    pd.Series(exchange, index=range(len(grid))),
                    pd.Series(instrument, index=range(len(grid))),
                )
                .ne("")
                .to_numpy()
            )
        observed = np.zeros(len(grid), dtype=bool)
        for i, (_, (start, end, _, _)) in enumerate(ordered):
            # A known segment break does not establish observations in its gap.
            if i == len(ordered) - 1:
                end += pd.Timedelta(seconds=max_age)
            else:
                end += visibility_delay
            observed |= (grid >= start + visibility_delay) & (grid <= end)
        times = grid[observed & grids[key]]
        if not len(times):
            continue
        block = pd.merge_asof(
            pd.DataFrame({"timestamp": times}),
            quotes,
            left_on="timestamp",
            right_on="_quote_visible_at",
            direction="backward",
            allow_exact_matches=True,
        ).drop(columns="_quote_visible_at")
        block["market_id"] = "a_share"
        block["asset_id"] = asset
        block["session_date"] = times[0].normalize()
        block["exchange"] = exchange
        block["instrument_type"] = instrument
        # Retain stale source clocks even when their wide payload was not read.
        for segment, (start, _end, _, _) in reversed(ordered):
            times = lookup.observed_times[(asset, segment)]
            missing = block.quote_timestamp.isna() & block.timestamp.ge(
                start + visibility_delay
            )
            positions = (
                np.searchsorted(
                    times,
                    (block.loc[missing, "timestamp"] - visibility_delay).to_numpy(
                        dtype="datetime64[ns]"
                    ),
                    side="right",
                )
                - 1
            )
            rows = block.index[missing][positions >= 0]
            block.loc[rows, "quote_timestamp"] = times[positions[positions >= 0]]
            block.loc[rows, "segment_id"] = segment
        block["quote_age_seconds"] = (
            block.timestamp - block.quote_timestamp
        ).dt.total_seconds()
        fresh = block.quote_age_seconds.le(max_age) & block.segment_id.notna()
        payload = [
            name
            for name, role in roles.items()
            if role in {"factor", "feature"} or name == "mid"
        ]
        block.loc[~fresh, payload] = np.nan
        # The legacy panel flag is book status, never research membership.
        if book_status:
            block["eligible"] = True
        for name, role in roles.items():
            if role in {"universe", "mask"}:
                block[name] = block[name].astype("boolean")
        output.append(block)
    return (
        pd.concat(output, ignore_index=True)
        .sort_values(["timestamp", "asset_id"], kind="stable")
        .reset_index(drop=True)
        if output
        else pd.DataFrame()
    )


def _with_stock_pool(frame, membership):
    other = frame.loc[frame.instrument_type.ne("stock")].copy()
    stocks = frame.loc[frame.instrument_type.eq("stock")].drop(
        columns=["eligible", "listing_date"], errors="ignore"
    )
    aligned = stocks.merge(
        membership,
        on=list(protocol.KEY),
        how="outer",
        suffixes=("", "_pool"),
        indicator=True,
        validate="one_to_one",
    )
    aligned = aligned.loc[
        aligned._merge.ne("right_only") | aligned.eligible.fillna(False)
    ]
    for name in ("session_date", "exchange", "instrument_type"):
        aligned[name] = aligned[name].combine_first(aligned.pop(name + "_pool"))
    return (
        pd.concat((aligned.drop(columns="_merge"), other), ignore_index=True)
        .sort_values(["timestamp", "asset_id"], kind="stable")
        .reset_index(drop=True)
    )


def sample_hft(config):
    """Resample quote-axis inputs; recompute labels from each decision time."""
    config = _strict_config(
        config,
        {"inputs", "output", "dataset_id", "time_window"},
        {
            "interval_seconds",
            "label_tolerance_seconds",
            "quote_visibility_delay_ms",
            "stock_pool",
            "max_quote_age_seconds",
        },
    )
    interval = config.get("interval_seconds", 3)
    if type(interval) is not int or interval not in {3, 30}:
        raise ValueError("hft_sampling_interval_invalid")
    max_age = config.get("max_quote_age_seconds", 6)
    tolerance = config.get("label_tolerance_seconds", 6)
    delay_ms = config.get("quote_visibility_delay_ms", 1)
    for name, value in (
        ("quote_age", max_age),
        ("label_tolerance", tolerance),
        ("quote_visibility_delay", delay_ms),
    ):
        if value is None and name == "label_tolerance":
            continue
        if type(value) not in {int, float} or not np.isfinite(value) or value < 0:
            raise ValueError(f"hft_sampling_{name}_invalid")
    visibility_delay = pd.Timedelta(milliseconds=delay_ms)
    window = TimeWindow(
        **_strict_config(config["time_window"], {"start", "end"}, set())
    )
    lower, upper, closed = time_window_timestamp_interval(window)
    output = Path(config["output"])
    reject_if_exists(output)
    source = open_dataset_session(**_dataset_input_options(config["inputs"]))
    declared = source.spec
    required = {
        *protocol.KEY,
        "session_date",
        "exchange",
        "instrument_type",
        "segment_id",
        "eligible",
        "mid",
    }
    if (
        declared["market_id"] != "a_share"
        or declared["frequency"] != "tick"
        or not required.issubset(source.logical_columns)
        or "quote_timestamp" in source.logical_columns
    ):
        raise ValueError("hft_sampling_requires_quote_axis")
    roles = declared["column_roles"]
    if inherited_feature_transforms(declared, source.manifest, source.logical_columns):
        raise ValueError("hft_sampling_requires_untransformed_features")
    old_labels = {name for name, role in roles.items() if role == "label"} | set(
        declared.get("label_availability_columns", {}).values()
    )
    roles = {
        name: roles[name] for name in source.logical_columns if name not in old_labels
    }
    book_status = roles["eligible"] == "context"
    roles["eligible"] = "universe"
    inputs = tuple(roles)
    pool = (
        load_stock_pool(config["stock_pool"])
        if config.get("stock_pool") is not None
        else None
    )
    if pool is not None:
        roles["listing_date"] = "context"
    definitions = protocol.label_catalog().list()
    maturity = {item.output_column: item.available_at_column for item in definitions}
    roles.update({"quote_timestamp": "context", "quote_age_seconds": "context"})
    units = {
        name: unit
        for name, unit in declared.get("column_units", {}).items()
        if name in roles
    }
    units["quote_age_seconds"] = "seconds"
    for item in definitions:
        roles[item.output_column] = "label"
        roles[item.available_at_column] = "context"
        roles[item.output_column + "_lateness_seconds"] = "context"
        units[item.output_column] = (
            "direction" if item.family == "future_mid_direction" else "decimal_return"
        )
        units[item.output_column + "_lateness_seconds"] = "seconds"
    spec = ResearchDatasetSpec(
        data_id=config["dataset_id"],
        market_id="a_share",
        frequency="tick",
        grain="research_row",
        column_roles=roles,
        primary_key=protocol.KEY,
        partition_columns=tuple(declared["partition_columns"]),
        column_units=units,
        label_availability_columns=maturity,
        input_data_ids=(declared["data_id"],),
        base_input_data_id=declared["data_id"],
        factor_columns=tuple(name for name, role in roles.items() if role == "factor"),
        feature_columns=tuple(
            name for name, role in roles.items() if role == "feature"
        ),
        label_columns=tuple(maturity),
        filter_columns=tuple(
            name for name, role in roles.items() if role in {"mask", "universe"}
        ),
        context_columns=tuple(
            name for name, role in roles.items() if role == "context"
        ),
        target_transform="raw",
    )

    def days():
        last_day = pd.Timestamp(upper) - (
            pd.Timedelta(0) if closed else pd.Timedelta(1, "ns")
        )
        price_columns = (
            *protocol.KEY,
            "segment_id",
            "mid",
            "exchange",
            "instrument_type",
        )
        for day in pd.date_range(
            pd.Timestamp(lower).normalize(), last_day.normalize(), freq="D"
        ):
            for begin, end in (
                (
                    day + pd.Timedelta(hours=1, minutes=30),
                    day + pd.Timedelta(hours=3, minutes=30),
                ),
                (day + pd.Timedelta(hours=5), day + pd.Timedelta(hours=7)),
            ):
                if end <= lower or begin > upper or (begin == upper and not closed):
                    continue
                session = source.filtered_view(
                    TimeWindow(begin, end - pd.Timedelta(1, "ns"))
                )
                domains = {}

                def prices(session=session, domains=domains):
                    for panel in iter_dataset_chunks(
                        session.projected_view(price_columns)
                    ):
                        panel = panel.loc[
                            continuous_session(
                                panel.timestamp, panel.exchange, panel.instrument_type
                            ).ne("")
                        ]
                        for (asset, segment), group in panel.groupby(
                            ["asset_id", "segment_id"], sort=False
                        ):
                            bounds = domains.setdefault(asset, {})
                            first, last = group.timestamp.min(), group.timestamp.max()
                            exchange, instrument = group.iloc[0][
                                ["exchange", "instrument_type"]
                            ]
                            if segment in bounds:
                                previous = bounds[segment]
                                first, last = (
                                    min(first, previous[0]),
                                    max(last, previous[1]),
                                )
                            bounds[segment] = (first, last, exchange, instrument)
                        yield panel

                lookup = QuoteReturnLookup(prices(), definitions)
                if not domains and pool is None:
                    continue
                # Reuse storage batch sizing; this batches work without limiting rows.
                block_duration = pd.Timedelta(
                    seconds=interval
                    * max(
                        1,
                        DEFAULT_BATCH_ROWS
                        // max(
                            1,
                            len(domains),
                            pool["stocks"].asset_id.nunique()
                            if pool is not None
                            else 0,
                        ),
                    )
                )
                for block_start in pd.date_range(
                    begin, end, freq=block_duration, inclusive="left"
                ):
                    block_end = min(block_start + block_duration, end)
                    grid = pd.date_range(
                        block_start, block_end, freq=f"{interval}s", inclusive="left"
                    ).as_unit("ns")
                    grid = grid[
                        (grid >= lower)
                        & ((grid <= upper) if closed else (grid < upper))
                    ]
                    if not len(grid):
                        continue
                    active = {
                        asset: segments
                        for asset, segments in domains.items()
                        if any(
                            first + visibility_delay <= grid[-1]
                            and last
                            + (
                                pd.Timedelta(seconds=max_age)
                                if last == max(item[1] for item in segments.values())
                                else visibility_delay
                            )
                            >= grid[0]
                            for first, last, _, _ in segments.values()
                        )
                    }
                    if not active and pool is None:
                        continue
                    # Read only the wide inputs needed by this block's backward as-of.
                    read_start = max(
                        begin,
                        grid[0] - max(pd.Timedelta(seconds=max_age), visibility_delay),
                    )
                    read_end = grid[-1] - visibility_delay
                    quotes = materialize_dataset_session(
                        session.filtered_view(
                            TimeWindow(min(read_start, read_end), read_end)
                        ).projected_view(inputs)
                    )
                    frame = _sample_block(
                        quotes,
                        active,
                        grid,
                        roles,
                        max_age,
                        lookup,
                        book_status=book_status,
                        visibility_delay=visibility_delay,
                    )
                    if pool is not None:
                        if frame.empty:
                            frame = (
                                quotes.iloc[:0]
                                .rename(columns={"timestamp": "quote_timestamp"})
                                .assign(
                                    timestamp=pd.Series(dtype="datetime64[ns]"),
                                    quote_age_seconds=pd.Series(dtype=float),
                                )
                            )
                        frame = _with_stock_pool(frame, stock_pool_rows(grid, **pool))
                    if frame.empty:
                        continue
                    labels = lookup.labels(
                        frame,
                        sampled=True,
                        max_lateness_seconds=tolerance,
                        visibility_delay=visibility_delay,
                    )
                    yield frame.merge(
                        labels, on=list(protocol.KEY), validate="one_to_one"
                    ).loc[:, spec.required_columns]

    write_dataset(
        output,
        days(),
        spec=spec,
        metadata={
            "inputs": {
                "quotes": config["inputs"],
                **({"stock_pool": config["stock_pool"]} if pool is not None else {}),
            },
            "parameters": {
                "interval_seconds": interval,
                "max_quote_age_seconds": max_age,
                "quote_visibility_delay_ms": delay_ms,
                "label_tolerance_seconds": tolerance,
                "time_window": config["time_window"],
            },
            "producer_diagnostics": {
                **source.manifest.get("producer_diagnostics", {}),
                "timestamp_semantics": "common_decision_time",
                "sampling": {
                    "quote_clock": "vendor_event_time_no_receive_clock",
                    "availability_assumption": (
                        "quote_event_time_plus_configured_visibility_delay_at_or_before_decision"
                    ),
                    "alignment": "backward_same_asset_continuous_session_and_segment",
                    "unobserved_or_stale_inputs": "missing",
                    "observation_domain": (
                        "qualified_pool_and_raw_observations"
                        if pool is not None
                        else "observed_segments_and_final_quote_age_within_session"
                    ),
                    "asset_universe": "point_in_time_stock_pool_with_raw_observations"
                    if pool is not None
                    else "unfiltered_observations"
                    if book_status
                    else "inherited_research_membership",
                    "primary_comparison_horizon_seconds": 30,
                },
                "label_semantics": {
                    item.output_column: {
                        **item.to_manifest(),
                        "entry_rule": "fresh_asof_mid_at_common_decision_time",
                        "horizon_origin": "common_decision_time",
                        "maturity_rule": (
                            "target_quote_event_time_plus_visibility_delay"
                        ),
                        "quote_visibility_delay_ms": delay_ms,
                        "max_lateness_seconds": tolerance,
                        "late_quote_policy": (
                            "missing_label_retain_actual_maturity_and_lateness"
                        ),
                        "max_quote_age_seconds": max_age,
                        "decision_interval_seconds": interval,
                    }
                    for item in definitions
                },
            },
        },
    )
    return output

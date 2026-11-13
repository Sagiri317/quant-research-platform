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
from quant_research_platform.storage.atomic import reject_if_exists
from quant_research_platform.storage.datasets import write_dataset
from quant_research_platform.storage.datasets.reading import (
    _dataset_input_options,
    iter_frame_chunks,
    materialize_dataset_session,
    open_dataset_session,
)
from quant_research_platform.storage.parquet import DEFAULT_BATCH_ROWS


def _sample_block(
    frame, domains, grid, roles, max_age, *, book_status, visibility_delay
):
    """Align the shared grid in one grouped as-of, retaining each asset's domain."""
    observations, grids = [], {}
    for asset, segments in domains.items():
        ordered = sorted(segments.values(), key=lambda item: item[0])
        exchange, instrument = ordered[0][2:]
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
        for i, (start, end, _, _) in enumerate(ordered):
            end += (
                pd.Timedelta(seconds=max_age)
                if i == len(ordered) - 1
                else visibility_delay
            )
            observed |= (grid >= start + visibility_delay) & (grid <= end)
        times = grid[observed & grids[key]]
        if len(times):
            observations.append((asset, times))
    if not observations:
        return pd.DataFrame()
    quotes = frame.rename(columns={"timestamp": "quote_timestamp"}).copy()
    quotes["quote_timestamp"] = quotes.quote_timestamp.astype("datetime64[ns]")
    quotes["_quote_visible_at"] = quotes.quote_timestamp + visibility_delay
    decisions = pd.DataFrame(
        {
            "timestamp": np.concatenate(
                [times.to_numpy() for _, times in observations]
            ),
            "asset_id": pd.Series(
                np.repeat(
                    [asset for asset, _ in observations],
                    [len(times) for _, times in observations],
                ),
                dtype=frame.asset_id.dtype,
            ),
        }
    )
    block = pd.merge_asof(
        decisions.sort_values("timestamp", kind="stable"),
        quotes.sort_values("_quote_visible_at", kind="stable"),
        left_on="timestamp",
        right_on="_quote_visible_at",
        by="asset_id",
        direction="backward",
    ).drop(columns="_quote_visible_at")
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
    if book_status:
        block["eligible"] = True
    for name, role in roles.items():
        if role in {"universe", "mask"}:
            block[name] = block[name].astype("boolean")
    return block.sort_values(["timestamp", "asset_id"], kind="stable").reset_index(
        drop=True
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
                # Native members may be unordered. Read the half-session once,
                # sharing its payload with the price lookup and grouped as-of.
                quotes = materialize_dataset_session(session.projected_view(inputs))
                quotes = quotes.loc[
                    continuous_session(
                        quotes.timestamp, quotes.exchange, quotes.instrument_type
                    ).ne("")
                ]
                domains = {}
                bounds = quotes.groupby(["asset_id", "segment_id"], sort=False).agg(
                    first=("timestamp", "min"),
                    last=("timestamp", "max"),
                    exchange=("exchange", "first"),
                    instrument_type=("instrument_type", "first"),
                )
                for (
                    asset,
                    segment,
                ), first, last, exchange, instrument in bounds.itertuples(name=None):
                    domains.setdefault(asset, {})[segment] = (
                        first,
                        last,
                        exchange,
                        instrument,
                    )
                if not domains and pool is None:
                    continue
                lookup = QuoteReturnLookup(
                    (quotes.loc[:, ["asset_id", "segment_id", "timestamp", "mid"]],),
                    definitions,
                )
                grid = pd.date_range(
                    begin, end, freq=f"{interval}s", inclusive="left"
                ).as_unit("ns")
                grid = grid[
                    (grid >= lower) & ((grid <= upper) if closed else (grid < upper))
                ]
                if not len(grid):
                    continue
                frame = _sample_block(
                    quotes,
                    domains,
                    grid,
                    roles,
                    max_age,
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
                completed = frame.merge(
                    labels, on=list(protocol.KEY), validate="one_to_one"
                ).loc[:, spec.required_columns]
                yield from iter_frame_chunks(completed, batch_rows=DEFAULT_BATCH_ROWS)

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

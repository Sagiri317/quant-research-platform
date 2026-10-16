"""Native maker calibration scientific tape and publication contracts."""

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from quant_research_platform.artifacts import statistics
from quant_research_platform.storage.atomic import atomic_write_directory
from quant_research_platform.storage.identity import (
    filesystem_exists,
    native_filesystem_path,
)
from quant_research_platform.storage.manifests import write_manifest

MAKER_MODEL_IDS = ("price_model", "proxy_model")


def _write_unchecked(root, metadata, write):
    def produce(staging):
        write(staging)
        write_manifest(
            staging / "manifest.json",
            {
                "schema_id": "qrp-statistics",
                "table_group": "crypto_maker_calibration",
                "metadata": metadata,
                "members": list(
                    statistics.STATISTICS_TABLE_GROUPS["crypto_maker_calibration"]
                ),
            },
        )

    atomic_write_directory(root, produce)
    return root


def _bytes(root):
    return {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def _producer_config(metadata):
    return {
        "panel": "panel",
        "source": "source",
        "asset_id": metadata["asset_id"],
        "symbol": metadata["symbol"],
        "chronology": metadata["chronology"],
        "reference": metadata["reference"],
        "models": [
            {
                "fold_id": fold["fold_id"],
                "inputs": "inputs",
                "price_predictions": "price",
                "proxy_predictions": "proxy",
            }
            for fold in metadata["chronology"]["folds"]
        ],
    }


@pytest.fixture
def maker_calibration_tables():
    from quant_research_platform.markets.implementations.crypto.execution import (
        replay_maker_calibration,
    )
    from tests.helpers.binance_hft import _quote, hft_research_plan
    from tests.markets.crypto.calibration_test_support import maker_event_rows
    from tests.markets.crypto.test_hft_execution import _state, _trade

    plan = hft_research_plan()
    plan["folds"][0]["validation"] = {
        "start": "1970-01-01T00:00:02.100000",
        "end": "1970-01-01T00:00:03.500000",
    }
    plan["folds"] = plan["folds"][:1]
    reference = {"order_quantity": 1.0, "latency_ms": 10, "min_calibration_samples": 2}
    samples, events = ([], [])

    def on_sample(**sample):
        samples.append(
            {
                "fold_id": "fold_1",
                "side": sample["side"],
                "sample_id": sample["sample_id"],
                "decision_at": sample["decision_ns"],
                "prediction_end": sample["prediction_end_ns"],
                "label_available_at": sample["available_ns"],
                "price_prediction": sample["prediction"],
                "observed_return": sample["observed_return"],
                "forecast_error": sample["observed_return"] - sample["prediction"],
                "filled_quantity": sample["filled"],
                "quote_quantity": sample["quote_quantity"],
                "terminal_reason": sample["terminal_reason"],
                "book_segment_id": sample["book_segment_id"],
                "trade_observation_id": sample["trade_observation_id"],
                "initial_book_event_id": sample["initial_book_event_id"],
                "initial_bid_index": sample["initial_bid_index"],
                "initial_ask_index": sample["initial_ask_index"],
                "event_start": sample["event_start"],
                "event_end": sample["event_end"],
            }
        )

    def on_event(*, event_id, event):
        events.extend(
            {"fold_id": "fold_1", **row} for row in maker_event_rows(event_id, event)
        )

    replay_maker_calibration(
        [_state(stamp) for stamp in (2100, 2150, 2650, 2700)],
        [
            _trade(stamp, price="99.5", quantity="3", maker_buyer=True)
            for stamp in (2180, 2400)
        ],
        [
            {
                **_quote(stamp, book_segment_id="s1"),
                "price_prediction": 0.0,
                "price_label": 0.0,
                "proxy_prediction": 1.0,
                "label_available_at_ns": (stamp + 500) * 1000000,
            }
            for stamp in (2100, 2200)
        ],
        order_quantity=reference["order_quantity"],
        latency_ms=reference["latency_ms"],
        observed_coverage={"s1": (2100000000, 2800000000)},
        trade_coverage={"trade-connection-1": (2100000000, 2800000000)},
        cutoff_ns=3500000000,
        on_sample=on_sample,
        on_event=on_event,
    )
    frames = {
        "crypto_maker_samples": pd.DataFrame(
            samples,
            columns=statistics.STATISTICS_TABLE_COLUMNS["crypto_maker_samples.parquet"],
        ),
        "crypto_maker_events": pd.DataFrame(
            events,
            columns=statistics.STATISTICS_TABLE_COLUMNS["crypto_maker_events.parquet"],
        ),
        "crypto_maker_calibration": pd.DataFrame(
            [
                {
                    "fold_id": "fold_1",
                    "side": side,
                    "price_unit": "0.1",
                    "sample_count": 2,
                    "zero_fill_count": 0 if side == "buy" else 2,
                    "post_only_rejection_count": 0,
                    "excluded_decisions": 0,
                    "minimum_samples": 2,
                    "status": "DEFINED",
                    "reason": "",
                }
                for side in ("buy", "sell")
            ],
            columns=statistics.STATISTICS_TABLE_COLUMNS[
                "crypto_maker_calibration.parquet"
            ],
        ),
    }
    for column in ("decision_at", "prediction_end", "label_available_at"):
        frames["crypto_maker_samples"][column] = pd.to_datetime(
            frames["crypto_maker_samples"][column], unit="ns"
        )
    frames["crypto_maker_events"]["event_at"] = pd.to_datetime(
        frames["crypto_maker_events"]["event_at"], unit="ns"
    )
    assert frames["crypto_maker_events"].event_id.nunique() == 5
    assert len(frames["crypto_maker_samples"]) == 4
    assert frames["crypto_maker_samples"].loc[
        lambda frame: frame.side.eq("buy"), "filled_quantity"
    ].tolist() == [1.0, 1.0]
    envelope = {
        "window": plan["folds"][0]["validation"],
        "sample_role": "validation",
        "chronology": {"folds": plan["folds"], "test": plan["test"]},
        "reference": reference,
        "model_ids": {"fold_1": list(MAKER_MODEL_IDS)},
        "asset_id": "btc_usdt",
        "symbol": "BTCUSDT",
    }
    return (frames, envelope)


@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "truncate",
        "immature",
        "coordinate",
        "book_zero",
        "level_order",
        "reference",
        "pair",
        "unit_invalid",
        "unit_mixed",
        "tail",
        "tail_after_cutoff",
        "empty",
    ],
)
def test_mature_shared_maker_tape_round_trip_and_rejects_corruption(
    tmp_path, maker_calibration_tables, mutation
):
    tables, envelope = maker_calibration_tables

    def produce(on_table):
        for name, frame in tables.items():
            on_table(name, frame)

    from quant_research_platform.markets.implementations.crypto.portfolio import (
        validate_maker_calibration,
    )

    root = statistics.write_statistics_artifact(
        tmp_path / "valid",
        metadata=envelope,
        tables=produce,
        table_group="crypto_maker_calibration",
        validate_tables=validate_maker_calibration,
    )
    original = _bytes(root)
    artifact = statistics.read_statistics_artifact(
        root,
        validate_tables=validate_maker_calibration,
    )
    for member in statistics.STATISTICS_TABLE_GROUPS["crypto_maker_calibration"]:
        with artifact.table(member) as actual:
            pd.testing.assert_frame_equal(
                actual, tables[member.removesuffix(".parquet")]
            )
    assert "crypto_maker_fills.parquet" not in artifact._members
    if mutation is None:
        for missing_validator in (None, object()):
            with pytest.raises(
                ValueError, match="statistics_business_validator_required"
            ):
                statistics.write_statistics_artifact(
                    tmp_path / "missing-validator",
                    metadata=envelope,
                    tables=produce,
                    table_group="crypto_maker_calibration",
                    validate_tables=missing_validator,
                )
            with pytest.raises(
                ValueError, match="statistics_business_validator_required"
            ):
                statistics.read_statistics_artifact(
                    root,
                    validate_tables=missing_validator,
                )
        assert not filesystem_exists(tmp_path / "missing-validator")
        return
    if mutation == "truncate":
        tables["crypto_maker_events"] = (
            tables["crypto_maker_events"].loc[lambda frame: frame.event_id.lt(4)].copy()
        )
    elif mutation in {"tail", "tail_after_cutoff"}:
        extra = (
            tables["crypto_maker_events"].loc[lambda frame: frame.event_id.eq(4)].copy()
        )
        extra.event_id = 5
        extra.event_at += pd.Timedelta(milliseconds=100)
        if mutation == "tail_after_cutoff":
            extra.event_at = pd.Timestamp("1970-01-01T00:00:04")
        tables["crypto_maker_events"] = pd.concat(
            [tables["crypto_maker_events"], extra], ignore_index=True
        )
    elif mutation == "empty":
        tables["crypto_maker_samples"] = tables["crypto_maker_samples"].iloc[:0]
    elif mutation == "immature":
        tables["crypto_maker_samples"].loc[0, "label_available_at"] = pd.Timestamp(
            "1970-01-01T00:00:04"
        )
    elif mutation == "coordinate":
        tables["crypto_maker_samples"].loc[0, "initial_book_event_id"] = 2
    elif mutation == "book_zero":
        tables["crypto_maker_events"].loc[0, "quantity"] = 0.0
    elif mutation == "level_order":
        tables["crypto_maker_events"].loc[1, "price_index"] = 1000
    elif mutation == "reference":
        tables["crypto_maker_samples"].loc[0, "filled_quantity"] = 0.5
    elif mutation == "unit_invalid":
        tables["crypto_maker_calibration"]["price_unit"] = "invalid"
    elif mutation == "unit_mixed":
        tables["crypto_maker_calibration"].loc[0, "price_unit"] = "0.01"
    else:
        tables["crypto_maker_samples"].loc[1, "price_prediction"] = 0.001
        tables["crypto_maker_samples"].loc[1, "forecast_error"] = -0.001
    if mutation == "tail":
        # A valid unused tail is retained source evidence, not a corrupt sample.
        saved = statistics.write_statistics_artifact(
            tmp_path / "unused-tail",
            metadata=envelope,
            tables=produce,
            table_group="crypto_maker_calibration",
            validate_tables=validate_maker_calibration,
        )
        statistics.read_statistics_artifact(
            saved, validate_tables=validate_maker_calibration
        )
        assert _bytes(root) == original
        return
    with pytest.raises(
        ValueError,
        match="order_book_price_unit_invalid"
        if mutation == "unit_invalid"
        else "crypto_calibration_",
    ):
        statistics.write_statistics_artifact(
            tmp_path / "rejected",
            metadata=envelope,
            tables=produce,
            table_group="crypto_maker_calibration",
            validate_tables=validate_maker_calibration,
        )
    assert not (tmp_path / "rejected").exists()

    def write_corruption(staging):
        for name, frame in tables.items():
            pq.write_table(
                pa.Table.from_pandas(
                    frame,
                    schema=statistics._ARROW_SCHEMAS[f"{name}.parquet"],
                    preserve_index=False,
                ),
                native_filesystem_path(staging / f"{name}.parquet"),
            )

    bad = _write_unchecked(tmp_path / "corrupted", envelope, write_corruption)
    with pytest.raises(
        ValueError,
        match="order_book_price_unit_invalid"
        if mutation == "unit_invalid"
        else "crypto_calibration_",
    ):
        statistics.read_statistics_artifact(
            bad,
            validate_tables=validate_maker_calibration,
        )
    assert _bytes(root) == original


@pytest.mark.parametrize(
    "finish", ["normal", "writer_failure", "trade_gap", "book_gap", "visible_gap"]
)
def test_maker_producer_physical_batches_preserve_tables_and_rebuild(
    tmp_path, monkeypatch, maker_calibration_tables, finish
):
    from contextlib import nullcontext
    from types import SimpleNamespace

    from quant_research_platform.markets.implementations.crypto import portfolio
    from tests.markets.crypto.test_hft_execution import _state, _trade

    expected, envelope = maker_calibration_tables
    config = _producer_config(envelope)
    source_identity = "d" * 64
    coverage = {"s1": (2100000000, 2800000000)}
    run = SimpleNamespace(
        source_identity=source_identity,
        market_metadata=lambda *_: {"price_unit": "0.1", "snapshot_limit": 10},
        observation_coverage=lambda **kwargs: {
            "trade-connection-1": (2100000000, 2800000000)
        },
    )
    monkeypatch.setattr(
        portfolio,
        "read_dataset_manifest_evidence",
        lambda *_: {
            "source_evidence": {
                "market_metadata": {
                    "BTCUSDT": {"price_unit": "0.1", "snapshot_limit": 10}
                }
            },
            "producer_diagnostics": {
                "reconstruction": {"BTCUSDT": {"coverage_segments": coverage}}
            },
        },
    )

    def validation_chunks(*, fold_id, model_id):
        assert fold_id == "fold_1"
        try:
            yield pd.DataFrame(
                {
                    "asset_id": ["btc_usdt"] * 2,
                    "timestamp": pd.to_datetime([2100, 2200], unit="ms"),
                    "label_available_at": pd.to_datetime([2600, 2700], unit="ms"),
                    "book_segment_id": ["s1"] * 2,
                    "trade_observation_id": ["trade-connection-1"] * 2,
                    "segment_id": ["segment"] * 2,
                    "prediction_score": [0.0 if model_id == MAKER_MODEL_IDS[0] else 1.0]
                    * 2,
                    "label": [0.0] * 2,
                }
            )
        finally:
            closed.append(model_id)

    def prediction_rows(*, evidence, fold_id, **kwargs):
        model_path = evidence
        with portfolio.closing_iterator(
            validation_chunks(fold_id=fold_id, model_id=model_path)
        ) as chunks:
            for frame in chunks:
                yield from frame.itertuples(index=False)

    monkeypatch.setattr(portfolio, "_prediction_rows", prediction_rows)
    monkeypatch.setattr(
        portfolio, "open_market_source", lambda *_, **_kwargs: nullcontext(run)
    )

    primary = KeyboardInterrupt("synthetic_maker_writer_failure")
    roots, group_counts = ([], [])
    for build in (1, 2) if finish == "normal" else (1,):
        closed, emitted, frames = ([], [], [])

        def books(*_args, closed=closed, **_kwargs):
            _kwargs["stats"].coverage_segments.update(coverage)
            try:
                if finish == "book_gap":
                    from tests.markets.crypto.test_hft import _boundary

                    yield _state(2100)
                    yield _boundary(2150)
                    yield from (_state(stamp) for stamp in (2200, 2650, 2700))
                else:
                    yield from (_state(stamp) for stamp in (2100, 2150, 2650, 2700))
            finally:
                closed.append("books")

        def trades(*_args, closed=closed, **_kwargs):
            try:
                if finish == "trade_gap":
                    from quant_research_platform.contracts.order_book import (
                        TradeBoundary,
                    )

                    event = _trade(2150, price="99.5", quantity="3", maker_buyer=True)
                    yield TradeBoundary(
                        **{
                            name: getattr(event, name)
                            for name in TradeBoundary.__dataclass_fields__
                            if name != "reason"
                        },
                        reason="invalid_numeric",
                    )
                if finish == "visible_gap":
                    from dataclasses import replace

                    yield replace(
                        _trade(2150, price="99.5", quantity="3", maker_buyer=True),
                        visible_quantity=None,
                    )
                yield from (
                    _trade(stamp, price="99.5", quantity="3", maker_buyer=True)
                    for stamp in (2180, 2400)
                )
            finally:
                closed.append("trades")

        def produce(on_table, emitted=emitted, frames=frames):

            def append(table_name, output_frame):
                emitted.append((table_name, len(output_frame)))
                if finish == "writer_failure" and len(emitted) == 2:
                    raise primary
                on_table(table_name, output_frame)

            monkeypatch.setattr(portfolio, "iter_reconstructed_depth", books)
            monkeypatch.setattr(portfolio, "iter_agg_trades", trades)
            portfolio._produce_maker_calibration(
                on_table=append,
                config=config,
                model_evidence={
                    "fold_1": {"price": MAKER_MODEL_IDS[0], "proxy": MAKER_MODEL_IDS[1]}
                },
            )

        logical = tmp_path / f"build_{build}"

        def write(logical=logical):
            return statistics.write_statistics_artifact(
                logical,
                metadata=envelope,
                tables=produce,
                table_group="crypto_maker_calibration",
                validate_tables=portfolio.validate_maker_calibration,
            )

        if finish == "writer_failure":
            with pytest.raises(KeyboardInterrupt) as caught:
                write()
            assert caught.value is primary and len(emitted) == 2
            assert sorted(closed) == sorted(["books", "trades", *MAKER_MODEL_IDS])
            assert not logical.exists()
            return
        root = write()
        artifact = statistics.read_statistics_artifact(
            root,
            validate_tables=portfolio.validate_maker_calibration,
        )
        if finish in {"trade_gap", "book_gap", "visible_gap"}:
            with artifact.table("crypto_maker_samples.parquet") as samples:
                assert len(samples) == 2
                assert samples.sample_id.eq(0).all()
                assert samples.decision_at.eq(pd.Timestamp(2200, unit="ms")).all()
                assert samples.initial_book_event_id.gt(0).all()
            with artifact.table("crypto_maker_calibration.parquet") as summary:
                assert summary.sample_count.eq(1).all()
            for corruption in ("referenced_trade", "coordinates"):
                changed = {}
                for member in expected:
                    with artifact.table(f"{member}.parquet") as actual:
                        changed[member] = actual.copy()
                if corruption == "referenced_trade":
                    tape = changed["crypto_maker_events"]
                    tape.loc[
                        tape.event_kind.eq("trade")
                        & tape.event_at.eq(pd.Timestamp(2400, unit="ms")),
                        "quantity",
                    ] = 0.0
                else:
                    changed["crypto_maker_samples"]["initial_book_event_id"] = 0
                with pytest.raises(
                    ValueError,
                    match="crypto_calibration_(reference_response|event_coordinates)_invalid",
                ):
                    statistics.write_statistics_artifact(
                        tmp_path / corruption,
                        metadata=envelope,
                        tables=lambda on_table, changed=changed: [
                            on_table(name, frame) for name, frame in changed.items()
                        ],
                        table_group="crypto_maker_calibration",
                        validate_tables=portfolio.validate_maker_calibration,
                    )
            assert sorted(closed) == sorted(["books", "trades", *MAKER_MODEL_IDS])
            return
        for name, frame in expected.items():
            with artifact.table(f"{name}.parquet") as actual:
                pd.testing.assert_frame_equal(actual, frame)
            with pq.ParquetFile(
                native_filesystem_path(root / f"{name}.parquet")
            ) as parquet:
                assert parquet.schema_arrow.equals(
                    statistics._ARROW_SCHEMAS[f"{name}.parquet"]
                )
        assert sorted(closed) == sorted(["books", "trades", *MAKER_MODEL_IDS])
        roots.append(root)
        with pq.ParquetFile(
            native_filesystem_path(root / "crypto_maker_events.parquet")
        ) as parquet:
            group_counts.append(parquet.num_row_groups)
    assert group_counts[0] == group_counts[1]
    assert _bytes(roots[0]) == _bytes(roots[1])


@pytest.mark.parametrize("failure", ("disk_full", "interrupted", "writer_failed"))
def test_incremental_maker_publication_preserves_old_good_and_primary(
    tmp_path, monkeypatch, maker_calibration_tables, failure
):
    import errno

    from quant_research_platform.markets.implementations.crypto import portfolio

    tables, envelope = maker_calibration_tables

    def produce(on_table):
        for name, frame in tables.items():
            on_table(name, frame)

    primary = {
        "disk_full": OSError(errno.ENOSPC, "synthetic_disk_full"),
        "interrupted": KeyboardInterrupt("synthetic_interruption"),
        "writer_failed": RuntimeError("synthetic_writer_failure"),
    }[failure]
    good = statistics.write_statistics_artifact(
        tmp_path / "good",
        metadata=envelope,
        tables=produce,
        table_group="crypto_maker_calibration",
        validate_tables=portfolio.validate_maker_calibration,
    )
    before = _bytes(good)
    native = pq.ParquetWriter.write_table
    written = 0

    def fail_after_progress(writer, *args, **kwargs):
        nonlocal written
        written += 1
        if written == 2:
            raise primary
        return native(writer, *args, **kwargs)

    monkeypatch.setattr(pq.ParquetWriter, "write_table", fail_after_progress)
    with pytest.raises(type(primary)) as caught:
        statistics.write_statistics_artifact(
            tmp_path / "candidate",
            metadata=envelope,
            tables=produce,
            table_group="crypto_maker_calibration",
            validate_tables=portfolio.validate_maker_calibration,
        )
    assert caught.value is primary and written == 2
    assert _bytes(good) == before
    assert not (tmp_path / "candidate").exists()
    statistics.read_statistics_artifact(
        good,
        validate_tables=portfolio.validate_maker_calibration,
    )


def test_maker_calibration_streams_many_events_without_full_table_reads(
    tmp_path, monkeypatch, record_testsuite_property
):
    from contextlib import nullcontext
    from types import SimpleNamespace

    from quant_research_platform.markets.implementations.crypto import portfolio
    from tests.helpers.binance_hft import hft_research_plan
    from tests.markets.crypto.test_hft_execution import _state

    count, depth = (129, 512)
    first, last = (2100, 2100 + (count - 1) * 1000)

    def iso(stamp):
        return pd.Timestamp(stamp, unit="ms").isoformat()

    plan = hft_research_plan()
    fold = plan["folds"][0]
    fold["validation"] = {"start": iso(first), "end": iso(last + 600)}
    plan["folds"] = [fold]
    plan["test"] = {"start": iso(last + 2100), "end": iso(last + 3000)}
    window = fold["validation"]
    envelope = {
        "window": window,
        "sample_role": "validation",
        "asset_id": "btc_usdt",
        "symbol": "BTCUSDT",
        "chronology": {"folds": plan["folds"], "test": plan["test"]},
        "reference": {
            "order_quantity": 1.0,
            "latency_ms": 10,
            "min_calibration_samples": 2,
        },
        "model_ids": {"fold_1": list(MAKER_MODEL_IDS)},
    }
    config = _producer_config(envelope)
    coverage = {"s1": (first * 1000000, (last + 600) * 1000000)}
    run = SimpleNamespace(
        source_identity="d" * 64,
        market_metadata=lambda *_: {"price_unit": "0.1", "snapshot_limit": 10},
        observation_coverage=lambda **kwargs: {
            "trade-connection-1": (first * 1000000, (last + 600) * 1000000)
        },
    )
    monkeypatch.setattr(
        portfolio,
        "read_dataset_manifest_evidence",
        lambda *_: {
            "source_evidence": {
                "market_metadata": {
                    "BTCUSDT": {"price_unit": "0.1", "snapshot_limit": 10}
                }
            },
            "producer_diagnostics": {
                "reconstruction": {"BTCUSDT": {"coverage_segments": coverage}}
            },
        },
    )

    def validation_chunks(*, fold_id, model_id):
        assert fold_id == "fold_1"
        for start in range(0, count, 64):
            stamps = [
                first + index * 1000 for index in range(start, min(count, start + 64))
            ]
            yield pd.DataFrame(
                {
                    "asset_id": ["btc_usdt"] * len(stamps),
                    "timestamp": pd.to_datetime(stamps, unit="ms"),
                    "label_available_at": pd.to_datetime(
                        [stamp + 500 for stamp in stamps], unit="ms"
                    ),
                    "book_segment_id": ["s1"] * len(stamps),
                    "trade_observation_id": ["trade-connection-1"] * len(stamps),
                    "segment_id": ["segment"] * len(stamps),
                    "prediction_score": [0.0 if model_id == MAKER_MODEL_IDS[0] else 1.0]
                    * len(stamps),
                    "label": [0.0] * len(stamps),
                }
            )

    def prediction_rows(*, evidence, fold_id, **kwargs):
        model_path = evidence
        with portfolio.closing_iterator(
            validation_chunks(fold_id=fold_id, model_id=model_path)
        ) as chunks:
            for frame in chunks:
                yield from frame.itertuples(index=False)

    monkeypatch.setattr(portfolio, "_prediction_rows", prediction_rows)
    monkeypatch.setattr(
        portfolio, "open_market_source", lambda *_, **_kwargs: nullcontext(run)
    )

    bids = tuple((995 - index, 1.0) for index in range(depth))
    asks = tuple((1005 + index, 1.0) for index in range(depth))
    table_read = statistics.StatisticsArtifact.table

    def summary_only(artifact, member):
        assert member == "crypto_maker_calibration.parquet"
        return table_read(artifact, member)

    monkeypatch.setattr(statistics.StatisticsArtifact, "table", summary_only)
    emitted = dict.fromkeys(
        ("crypto_maker_samples", "crypto_maker_events", "crypto_maker_calibration"), 0
    )

    event_batch_rows = []

    def produce(on_table):

        def append(name, frame):
            emitted[name] += len(frame)
            if name == "crypto_maker_events":
                event_batch_rows.append(len(frame))
            on_table(name, frame)

        def books(*_args, stats, **_kwargs):
            stats.coverage_segments.update(coverage)
            yield from (
                _state(first + index * 1000, bid_levels=bids, ask_levels=asks)
                for index in range(count)
            )

        monkeypatch.setattr(portfolio, "iter_reconstructed_depth", books)
        monkeypatch.setattr(
            portfolio, "iter_agg_trades", lambda *_args, **_kwargs: iter(())
        )
        portfolio._produce_maker_calibration(
            on_table=append,
            config=config,
            model_evidence={
                "fold_1": {"price": MAKER_MODEL_IDS[0], "proxy": MAKER_MODEL_IDS[1]}
            },
        )

    root = statistics.write_statistics_artifact(
        tmp_path / "bounded",
        metadata=envelope,
        tables=produce,
        table_group="crypto_maker_calibration",
        validate_tables=portfolio.validate_maker_calibration,
    )
    artifact = statistics.read_statistics_artifact(
        root,
        validate_tables=portfolio.validate_maker_calibration,
    )
    with artifact.table("crypto_maker_calibration.parquet") as summary:
        assert summary.sample_count.tolist() == [count, count]
        assert summary.zero_fill_count.tolist() == [count, count]
    assert (
        sum(len(frame) for frame in artifact._batches("crypto_maker_samples.parquet"))
        == count * 2
    )
    assert (
        sum(len(frame) for frame in artifact._batches("crypto_maker_events.parquet"))
        == count * depth * 2
    )
    assert emitted == {
        "crypto_maker_samples": count * 2,
        "crypto_maker_events": count * depth * 2,
        "crypto_maker_calibration": 2,
    }
    assert event_batch_rows == [65_536, 65_536, 1_024]
    record_testsuite_property("maker_event_rows", emitted["crypto_maker_events"])


def test_maker_calibration_uses_declared_asset_without_btc_limits(
    tmp_path, maker_calibration_tables
):
    from quant_research_platform.markets.implementations.crypto.portfolio import (
        validate_maker_calibration,
    )

    tables, original = maker_calibration_tables
    envelope = {**original, "asset_id": "eth_usdt", "symbol": "ETHUSDT"}

    def produce(on_table):
        for name, frame in tables.items():
            on_table(name, frame)

    root = statistics.write_statistics_artifact(
        tmp_path / "eth",
        metadata=envelope,
        tables=produce,
        table_group="crypto_maker_calibration",
        validate_tables=validate_maker_calibration,
    )
    statistics.read_statistics_artifact(
        root,
        validate_tables=validate_maker_calibration,
    )


@pytest.mark.parametrize("boundary", [False, True])
def test_maker_validator_reuses_overlapping_and_adjacent_native_paths(
    tmp_path, monkeypatch, maker_calibration_tables, boundary
):
    from collections import deque

    from quant_research_platform.markets.implementations.crypto import portfolio

    tables, envelope = maker_calibration_tables
    if boundary:
        from tests.markets.crypto.calibration_test_support import maker_event_rows
        from tests.markets.crypto.test_hft_execution import _boundary

        events = tables["crypto_maker_events"]
        rows = pd.DataFrame(
            [
                {"fold_id": "fold_1", **row}
                for row in maker_event_rows(3, _boundary(2250))
            ],
            columns=events.columns,
        )
        rows.event_at = pd.to_datetime(rows.event_at, unit="ns")
        events.loc[events.event_id.ge(3), "event_id"] += 1
        tables["crypto_maker_events"] = pd.concat(
            [
                events.loc[events.event_id.lt(3)],
                rows,
                events.loc[events.event_id.gt(3)],
            ],
            ignore_index=True,
        )
        samples = tables["crypto_maker_samples"]
        samples.event_end += 1
        samples.loc[
            samples.sample_id.eq(1) & samples.side.eq("buy"), "filled_quantity"
        ] = 0.0
        samples.loc[
            samples.sample_id.eq(1) | samples.side.eq("sell"), "terminal_reason"
        ] = "book_boundary"
        summary = tables["crypto_maker_calibration"]
        summary.loc[summary.side.eq("buy"), "zero_fill_count"] += 1
    event_count = tables["crypto_maker_events"].event_id.nunique()
    for name in ("crypto_maker_samples", "crypto_maker_events"):
        frame = tables[name]
        later = frame.copy()
        for column in (
            "decision_at",
            "prediction_end",
            "label_available_at",
            "event_at",
        ):
            if column in later:
                later[column] += pd.Timedelta(milliseconds=600)
        for column in ("initial_book_event_id", "event_start", "event_end", "event_id"):
            if column in later:
                later[column] += event_count
        if "sample_id" in later:
            later.sample_id += 2
        tables[name] = pd.concat([frame, later], ignore_index=True)
    summary = tables["crypto_maker_calibration"]
    for column in ("sample_count", "zero_fill_count", "post_only_rejection_count"):
        summary[column] *= 2

    paths, decoded, closed, maxima = [], [], [], []
    native_events = portfolio._calibration_events
    native_response = portfolio.MakerCalibration._response

    def events(*args):
        count = 0
        try:
            for event in native_events(*args):
                count += 1
                yield event
        finally:
            decoded.append(count)
            closed.append(True)

    def response(model, sample, **kwargs):
        path = tuple(kwargs.pop("events"))
        paths.append((sample["sample_id"], sample["side"], path))
        return (yield from native_response(model, sample, events=iter(path), **kwargs))

    class Window(deque):
        def append(self, event):
            super().append(event)
            maxima.append(len(self))

    def no_full_history(_model):
        pytest.fail("validator materialized a full fold history")

    monkeypatch.setattr(portfolio, "deque", Window, raising=False)
    monkeypatch.setattr(portfolio, "_calibration_events", events)
    monkeypatch.setattr(portfolio.MakerCalibration, "_response", response)
    monkeypatch.setattr(portfolio.MakerCalibration, "_load", no_full_history)

    def produce(on_table):
        for name, frame in tables.items():
            on_table(name, frame)

    root = statistics.write_statistics_artifact(
        tmp_path / "paths",
        metadata=envelope,
        tables=produce,
        table_group="crypto_maker_calibration",
        validate_tables=portfolio.validate_maker_calibration,
    )
    statistics.read_statistics_artifact(
        root,
        validate_tables=portfolio.validate_maker_calibration,
    )
    assert decoded == [event_count * 2] * 2 and len(closed) == 2
    assert max(maxima) < event_count * 2
    assert len(paths) == 16
    for first in (0, 8):
        for index in range(first, first + 8, 2):
            buy, sell = paths[index : index + 2]
            assert (buy[1], sell[1]) == ("buy", "sell")
            assert len(buy[2]) == len(sell[2])
            assert all(
                left is right for left, right in zip(buy[2], sell[2], strict=True)
            )
        previous = {event.event_id: event for event in paths[first][2]}
        overlap = [event for event in paths[first + 2][2] if event.event_id in previous]
        assert overlap and all(event is previous[event.event_id] for event in overlap)
        assert paths[first + 4][2][0].event_id > paths[first + 2][2][-1].event_id


@pytest.mark.parametrize("finish", ["close_failed", "cancel"])
def test_maker_validator_closes_native_stream_and_preserves_primary(
    tmp_path, monkeypatch, maker_calibration_tables, finish
):
    from quant_research_platform.markets.implementations.crypto import portfolio

    tables, envelope = maker_calibration_tables

    def produce(on_table):
        for name, frame in tables.items():
            on_table(name, frame)

    root = statistics.write_statistics_artifact(
        tmp_path / "good",
        metadata=envelope,
        tables=produce,
        table_group="crypto_maker_calibration",
        validate_tables=portfolio.validate_maker_calibration,
    )
    before = {
        path.name: Path(native_filesystem_path(path)).read_bytes()
        for path in root.iterdir()
        if path.is_file()
    }
    native_events = portfolio._calibration_events
    primary = KeyboardInterrupt("synthetic_reference_cancel")
    cleanup = RuntimeError("synthetic_native_close_failure")
    closed, opened = [], 0

    def events(*args):
        nonlocal opened
        opened += 1
        number = opened
        try:
            yield from native_events(*args)
        finally:
            closed.append(number)
            if number == 1:
                raise cleanup

    def cancel(*_args, **_kwargs):
        raise primary

    with monkeypatch.context() as patch:
        patch.setattr(portfolio, "_calibration_events", events)
        if finish == "cancel":
            patch.setattr(portfolio.MakerCalibration, "_response", cancel)
        with pytest.raises(
            KeyboardInterrupt if finish == "cancel" else RuntimeError
        ) as caught:
            statistics.read_statistics_artifact(
                root,
                validate_tables=portfolio.validate_maker_calibration,
            )
        assert caught.value is (primary if finish == "cancel" else cleanup)
    assert closed == [1]
    if finish == "cancel":
        assert any("RuntimeError" in note for note in primary.__notes__)
    assert {
        path.name: Path(native_filesystem_path(path)).read_bytes()
        for path in root.iterdir()
        if path.is_file()
    } == before
    statistics.read_statistics_artifact(
        root,
        validate_tables=portfolio.validate_maker_calibration,
    )


@pytest.mark.parametrize(
    "damage",
    [
        "side",
        "events_side",
        "ordinal",
        "generic_before_ordinal",
        "book_cross_batch",
        "trade_cross_batch",
    ],
)
def test_maker_validator_rejects_compound_damage_without_error_priority(
    tmp_path, monkeypatch, maker_calibration_tables, damage
):
    from functools import partial

    from quant_research_platform.markets.implementations.crypto import portfolio

    tables, envelope = maker_calibration_tables
    tables["crypto_maker_samples"].loc[0, "filled_quantity"] = 0.5
    if damage in ("side", "events_side", "generic_before_ordinal"):
        extra = tables["crypto_maker_samples"].iloc[[0]].copy()
        extra.side = "events" if damage == "events_side" else "unknown"
        if damage == "generic_before_ordinal":
            extra.sample_id = -1
        tables["crypto_maker_samples"] = pd.concat(
            [tables["crypto_maker_samples"], extra], ignore_index=True
        )
    if damage in ("ordinal", "generic_before_ordinal"):
        extra = (
            tables["crypto_maker_events"].loc[lambda frame: frame.event_id.eq(4)].copy()
        )
        extra.event_id = 6
        extra.event_at += pd.Timedelta(milliseconds=500)
        tables["crypto_maker_events"] = pd.concat(
            [tables["crypto_maker_events"], extra], ignore_index=True
        )

    if damage in ("book_cross_batch", "trade_cross_batch"):
        events = tables["crypto_maker_events"]
        extra = events.loc[
            events.event_id.eq(4 if damage == "book_cross_batch" else 2)
        ].copy()
        if damage == "trade_cross_batch":
            extra = pd.concat([extra, extra], ignore_index=True)
        extra.event_id = 6
        extra.event_at = pd.Timestamp("1970-01-01T00:00:03.150000")
        extra.iloc[-1, extra.columns.get_loc("quantity")] = float("inf")
        tables["crypto_maker_events"] = pd.concat([events, extra], ignore_index=True)
        monkeypatch.setattr(
            statistics,
            "iter_declared_parquet_member_batches",
            partial(statistics.iter_declared_parquet_member_batches, batch_rows=1),
        )

    def write(staging):
        for name, frame in tables.items():
            pq.write_table(
                pa.Table.from_pandas(
                    frame,
                    schema=statistics._ARROW_SCHEMAS[f"{name}.parquet"],
                    preserve_index=False,
                ),
                native_filesystem_path(staging / f"{name}.parquet"),
            )

    root = _write_unchecked(tmp_path / "compound", envelope, write)
    before = {path.name: path.read_bytes() for path in root.iterdir() if path.is_file()}
    with pytest.raises(ValueError):
        statistics.read_statistics_artifact(
            root,
            validate_tables=portfolio.validate_maker_calibration,
        )
    assert {
        path.name: path.read_bytes() for path in root.iterdir() if path.is_file()
    } == before

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pytest

from quant_research_platform.contracts._json import canonical_json_hash
from quant_research_platform.contracts.order_book import (
    BookBoundary,
    BookState,
    price_to_tick,
    tick_to_price,
)
from quant_research_platform.runtime.resources import runtime_resource_policy_scope
from quant_research_platform.sources.binance.ids import (
    FUTURES_MARKET_TYPE,
    SPOT_MARKET_TYPE,
)
from quant_research_platform.sources.binance.replay import (
    iter_reconstructed_depth,
    open_recorder_run,
)
from tests.helpers.runtime_resources import capture_test_policy


def _raw(
    capture_seq: int,
    payload: dict[str, object],
    *,
    segment_id: str,
    connection_id: str = "connection-1",
) -> dict[str, object]:
    return {
        "capture_seq": capture_seq,
        "receive_time": "2026-08-27T00:00:00.123456Z",
        "receive_monotonic_ns": capture_seq * 10,
        "connection_id": connection_id,
        "segment_id": segment_id,
        "payload": payload,
    }


def _snapshot(
    capture_seq: int,
    *,
    market_type: str,
    segment_id: str,
    last_update_id: int,
    bids: list[list[str]] | None = None,
    asks: list[list[str]] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "lastUpdateId": last_update_id,
        "bids": bids or [["100.0", "2"], ["99.0", "3"]],
        "asks": asks or [["101.0", "2"], ["102.0", "3"]],
    }
    if market_type == FUTURES_MARKET_TYPE:
        payload.update({"E": 90, "T": 89})
    return _raw(capture_seq, payload, segment_id=segment_id)


def _futures_delta(
    capture_seq: int,
    first: int,
    last: int,
    *,
    previous: int,
    transaction_time: int,
    segment_id: str,
    bids: list[list[str]] | None = None,
    asks: list[list[str]] | None = None,
) -> dict[str, object]:
    return _raw(
        capture_seq,
        {
            "e": "depthUpdate",
            "E": transaction_time + 2,
            "T": transaction_time,
            "s": "BTCUSDT",
            "ps": "BTCUSDT",
            "st": 1,
            "U": first,
            "u": last,
            "pu": previous,
            "b": bids or [],
            "a": asks or [],
        },
        segment_id=segment_id,
    )


def _spot_delta(
    capture_seq: int,
    first: int,
    last: int,
    *,
    event_time: int,
    segment_id: str,
) -> dict[str, object]:
    return _raw(
        capture_seq,
        {
            "e": "depthUpdate",
            "E": event_time,
            "s": "BTCUSDT",
            "U": first,
            "u": last,
            "b": [],
            "a": [],
        },
        segment_id=segment_id,
    )


def _write_file(
    root: Path,
    *,
    market_type: str,
    data_kind: str,
    stream: str,
    rows: list[dict[str, object]],
) -> dict[str, object]:
    path = root / "date=2026-08-27" / "hour=00" / f"{market_type}_{data_kind}.jsonl.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    max_record_bytes = 0
    with gzip.open(path, "wb") as handle:
        for row in rows:
            line = (
                json.dumps(
                    row,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
                + b"\n"
            )
            max_record_bytes = max(max_record_bytes, len(line))
            handle.write(line)
    raw = path.read_bytes()
    return {
        "path": path.relative_to(root).as_posix(),
        "venue": "binance",
        "symbol": "BTCUSDT",
        "market": market_type,
        "stream": stream,
        "data_kind": data_kind,
        "record_count": len(rows),
        "capture_seq_min": min(row["capture_seq"] for row in rows),
        "capture_seq_max": max(row["capture_seq"] for row in rows),
        "max_record_bytes": max_record_bytes,
        "compressed_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _write_run(
    root: Path,
    files: list[tuple[str, str, str, list[dict[str, object]]]],
) -> dict[str, object]:
    root.mkdir(parents=True, exist_ok=True)
    file_manifests = [
        _write_file(
            root,
            market_type=market,
            data_kind=kind,
            stream=stream,
            rows=rows,
        )
        for market, kind, stream, rows in files
    ]
    manifest: dict[str, object] = {
        "schema_id": "qrp-binance-recorder",
        "run_id": "run-1",
        "status": "PASS",
        "market_data_status": "PASS",
        "venue": "binance",
        "symbol": "BTCUSDT",
        "market_types": sorted({market for market, *_ in files}),
        "market_metadata": {
            market: {"tick_size": "0.1", "snapshot_limit": 1000} for market, *_ in files
        },
        "files": file_manifests,
    }
    (root / "run.json").write_text(
        json.dumps(manifest, separators=(",", ":")),
        encoding="utf-8",
    )
    return manifest


def _futures_run(root: Path, depth_rows: list[dict[str, object]]) -> None:
    segments = dict.fromkeys(str(row["segment_id"]) for row in depth_rows)
    snapshots = [
        _snapshot(
            100 + index,
            market_type=FUTURES_MARKET_TYPE,
            segment_id=segment,
            last_update_id=100 if index == 0 else 200,
        )
        for index, segment in enumerate(segments)
    ]
    _write_run(
        root,
        [
            (
                FUTURES_MARKET_TYPE,
                "depth",
                "btcusdt@depth@100ms",
                depth_rows,
            ),
            (
                FUTURES_MARKET_TYPE,
                "depth_snapshot",
                "REST:/fapi/v1/depth",
                snapshots,
            ),
        ],
    )


def test_run_identity_binds_exact_manifest_files_and_capture_order(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    manifest = _write_run(
        root,
        [
            (
                FUTURES_MARKET_TYPE,
                "depth",
                "btcusdt@depth@100ms",
                [
                    _futures_delta(
                        1, 90, 100, previous=0, transaction_time=10, segment_id="s1"
                    ),
                    _futures_delta(
                        3, 101, 101, previous=100, transaction_time=30, segment_id="s1"
                    ),
                ],
            ),
            (
                FUTURES_MARKET_TYPE,
                "depth_snapshot",
                "REST:/fapi/v1/depth",
                [
                    _snapshot(
                        2,
                        market_type=FUTURES_MARKET_TYPE,
                        segment_id="s1",
                        last_update_id=100,
                    )
                ],
            ),
        ],
    )
    policy = capture_test_policy(available_memory_bytes=64 * 1024**2)

    with runtime_resource_policy_scope(policy), open_recorder_run(root, policy) as run:
        records = list(run.iter_records())
        assert run.source_identity == {
            "schema_id": "qrp-binance-recorder",
            "artifact_id": "run-1",
            "manifest_hash": canonical_json_hash(manifest),
        }

    assert [record["capture_seq"] for record in records] == [1, 2, 3]


def test_run_rejects_unlisted_or_hash_mismatched_file(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _write_run(
        root,
        [
            (
                FUTURES_MARKET_TYPE,
                "depth",
                "btcusdt@depth@100ms",
                [
                    _futures_delta(
                        1, 90, 100, previous=0, transaction_time=10, segment_id="s1"
                    )
                ],
            )
        ],
    )
    extra = root / "extra.jsonl.gz"
    with gzip.open(extra, "wt", encoding="utf-8") as handle:
        handle.write("{}\n")
    policy = capture_test_policy(available_memory_bytes=64 * 1024**2)

    with runtime_resource_policy_scope(policy):
        with pytest.raises(ValueError, match="file_manifest_mismatch"):
            with open_recorder_run(root, policy):
                pass

    extra.unlink()
    declared = next(root.rglob("*.jsonl.gz"))
    declared.write_bytes(declared.read_bytes() + b"tampered")
    policy = capture_test_policy(available_memory_bytes=64 * 1024**2)
    with runtime_resource_policy_scope(policy):
        with pytest.raises(ValueError, match="file_identity_mismatch"):
            with open_recorder_run(root, policy):
                pass


def test_futures_bootstrap_keeps_all_clocks_and_exact_tick_identity(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    _futures_run(
        root,
        [_futures_delta(1, 90, 100, previous=0, transaction_time=10, segment_id="s1")],
    )
    policy = capture_test_policy(available_memory_bytes=64 * 1024**2)

    with runtime_resource_policy_scope(policy), open_recorder_run(root, policy) as run:
        events = list(
            iter_reconstructed_depth(run, FUTURES_MARKET_TYPE, required_levels=1)
        )

    state = events[0]
    assert isinstance(state, BookState)
    assert state.transaction_time_ms == 10
    assert state.event_time_ms == 12
    assert state.receive_time == "2026-08-27T00:00:00.123456Z"
    assert state.receive_monotonic_ns == 10
    assert state.capture_seq == 1
    assert state.best_bid == (1000, 2.0)
    assert price_to_tick("100.0", "0.1", error="bad") == 1000
    assert tick_to_price(1000, "0.1") == 100.0
    with pytest.raises(ValueError, match="bad"):
        price_to_tick("100.05", "0.1", error="bad")


def test_futures_zero_transaction_time_is_not_replaced_by_event_time(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    _futures_run(
        root,
        [
            _futures_delta(
                1,
                90,
                100,
                previous=0,
                transaction_time=0,
                segment_id="s1",
            )
        ],
    )
    policy = capture_test_policy(available_memory_bytes=64 * 1024**2)

    with runtime_resource_policy_scope(policy), open_recorder_run(root, policy) as run:
        state = next(
            event
            for event in iter_reconstructed_depth(
                run,
                FUTURES_MARKET_TYPE,
                required_levels=1,
            )
            if isinstance(event, BookState)
        )

    assert state.research_time_ms == 0


def test_spot_uses_snapshot_plus_one_bootstrap(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _write_run(
        root,
        [
            (
                SPOT_MARKET_TYPE,
                "depth",
                "btcusdt@depth@100ms",
                [
                    _spot_delta(1, 10, 11, event_time=10, segment_id="s1"),
                    _spot_delta(2, 12, 12, event_time=20, segment_id="s1"),
                ],
            ),
            (
                SPOT_MARKET_TYPE,
                "depth_snapshot",
                "REST:/api/v3/depth",
                [
                    _snapshot(
                        3,
                        market_type=SPOT_MARKET_TYPE,
                        segment_id="s1",
                        last_update_id=10,
                    )
                ],
            ),
        ],
    )
    policy = capture_test_policy(available_memory_bytes=64 * 1024**2)

    with runtime_resource_policy_scope(policy), open_recorder_run(root, policy) as run:
        states = [
            event
            for event in iter_reconstructed_depth(
                run, SPOT_MARKET_TYPE, required_levels=1
            )
            if isinstance(event, BookState)
        ]

    assert [state.update_id for state in states] == [11, 12]
    assert all(state.transaction_time_ms is None for state in states)
    assert [state.event_time_ms for state in states] == [10, 20]


def test_gap_emits_boundary_and_recovery_never_reuses_old_book(tmp_path: Path) -> None:
    root = tmp_path / "run"
    depth_rows = [
        _futures_delta(1, 90, 100, previous=0, transaction_time=10, segment_id="s1"),
        _futures_delta(2, 105, 110, previous=99, transaction_time=20, segment_id="s1"),
        _futures_delta(3, 190, 200, previous=0, transaction_time=30, segment_id="s2"),
    ]
    _futures_run(root, depth_rows)
    policy = capture_test_policy(available_memory_bytes=64 * 1024**2)

    with runtime_resource_policy_scope(policy), open_recorder_run(root, policy) as run:
        events = list(
            iter_reconstructed_depth(run, FUTURES_MARKET_TYPE, required_levels=1)
        )

    assert [type(event) for event in events] == [BookState, BookBoundary, BookState]
    assert events[1].reason == "sequence_gap"
    assert events[1].transaction_time_ms == 20
    assert events[2].segment_id == "s2"


@pytest.mark.parametrize(
    ("bids", "asks", "reason"),
    (
        ([["100.0", "0"], ["99.0", "0"]], [], "one_sided_book"),
        ([["101.0", "1"]], [], "crossed_book"),
    ),
)
def test_invalid_book_emits_honest_boundary(
    tmp_path: Path,
    bids: list[list[str]],
    asks: list[list[str]],
    reason: str,
) -> None:
    root = tmp_path / reason
    _futures_run(
        root,
        [
            _futures_delta(
                1,
                90,
                100,
                previous=0,
                transaction_time=10,
                segment_id="s1",
                bids=bids,
                asks=asks,
            )
        ],
    )
    policy = capture_test_policy(available_memory_bytes=64 * 1024**2)

    with runtime_resource_policy_scope(policy), open_recorder_run(root, policy) as run:
        events = list(
            iter_reconstructed_depth(run, FUTURES_MARKET_TYPE, required_levels=1)
        )

    assert len(events) == 1
    assert isinstance(events[0], BookBoundary)
    assert events[0].reason == reason


def test_raw_decode_is_admitted_before_json_allocation(tmp_path: Path) -> None:
    root = tmp_path / "run"
    manifest = _write_run(
        root,
        [
            (
                FUTURES_MARKET_TYPE,
                "depth",
                "btcusdt@depth@100ms",
                [
                    _futures_delta(
                        1, 90, 100, previous=0, transaction_time=10, segment_id="s1"
                    )
                ],
            )
        ],
    )
    manifest["files"][0]["max_record_bytes"] = 1_000_000
    (root / "run.json").write_text(json.dumps(manifest), encoding="utf-8")
    policy = capture_test_policy(available_memory_bytes=256 * 1024)

    with runtime_resource_policy_scope(policy), open_recorder_run(root, policy) as run:
        with pytest.raises(MemoryError, match="runtime_memory_budget_exceeded"):
            next(run.iter_records())


def test_same_run_replay_is_deterministic(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _futures_run(
        root,
        [
            _futures_delta(
                1, 90, 100, previous=0, transaction_time=10, segment_id="s1"
            ),
            _futures_delta(
                2,
                101,
                101,
                previous=100,
                transaction_time=20,
                segment_id="s1",
                bids=[["100.0", "1.5"]],
            ),
        ],
    )
    policy = capture_test_policy(available_memory_bytes=64 * 1024**2)

    with runtime_resource_policy_scope(policy), open_recorder_run(root, policy) as run:
        first = list(
            iter_reconstructed_depth(run, FUTURES_MARKET_TYPE, required_levels=1)
        )
        second = list(
            iter_reconstructed_depth(run, FUTURES_MARKET_TYPE, required_levels=1)
        )

    assert first == second

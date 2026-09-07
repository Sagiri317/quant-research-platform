from __future__ import annotations

import warnings
from datetime import timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

import quant_research_platform.runtime.dataframes as memory_owner
from quant_research_platform.runtime.dataframes import (
    iter_complete_session_frames,
    iter_complete_time_sized_frames,
)
from quant_research_platform.runtime.resources import (
    RuntimeResourcePolicy,
    current_runtime_resource_policy,
    pandas_frame_payload_bytes,
    runtime_resource_policy_scope,
)
from tests.helpers.runtime_resources import capture_test_policy


def _single_time_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01"], utc=True),
            "value": [1],
        }
    )


def test_complete_session_frames_merge_boundary_and_emit_one_session() -> None:
    frames = iter(
        (
            pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(
                        ["2024-01-01", "2024-01-01", "2024-01-02"]
                    ),
                    "value": [1, 2, 3],
                }
            ),
            pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                    "value": [4, 5],
                }
            ),
        )
    )
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024 * 1024,
        explicit_cap_bytes=1024 * 1024,
    )

    chunks = list(
        iter_complete_session_frames(
            frames,
            timestamp_column="timestamp",
            resource_policy=policy,
        )
    )

    assert [len(chunk) for chunk in chunks] == [2, 2, 1]
    assert [chunk["timestamp"].nunique() for chunk in chunks] == [1, 1, 1]


@pytest.mark.parametrize("explicit_mode", ("none", "same"))
def test_lazy_memory_factory_binds_outer_for_active_consumption(
    explicit_mode: str,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024 * 1024,
        explicit_cap_bytes=1 * 1024 * 1024,
    )
    frame = _single_time_frame()
    with runtime_resource_policy_scope(policy):
        chunks = iter_complete_time_sized_frames(
            iter((frame,)),
            timestamp_column="timestamp",
            resource_policy=policy if explicit_mode == "same" else None,
        )
        assert len(list(chunks)) == 1


@pytest.mark.parametrize("explicit_mode", ("none", "same"))
def test_lazy_memory_factory_rejects_delayed_consumption_after_outer_exit(
    explicit_mode: str,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024 * 1024,
        explicit_cap_bytes=1 * 1024 * 1024,
    )
    frame = _single_time_frame()
    with runtime_resource_policy_scope(policy):
        chunks = iter_complete_time_sized_frames(
            iter((frame,)),
            timestamp_column="timestamp",
            resource_policy=policy if explicit_mode == "same" else None,
        )

    with pytest.raises(RuntimeError, match="runtime_resource_policy_not_active"):
        next(chunks)


def test_lazy_memory_factory_rejects_different_explicit_policy_at_activation() -> None:
    active = capture_test_policy(
        available_memory_bytes=16 * 1024 * 1024,
        explicit_cap_bytes=1 * 1024 * 1024,
    )
    different = capture_test_policy(
        available_memory_bytes=16 * 1024 * 1024,
        explicit_cap_bytes=2 * 1024 * 1024,
    )
    frame = _single_time_frame()
    with runtime_resource_policy_scope(active):
        chunks = iter_complete_time_sized_frames(
            iter((frame,)),
            timestamp_column="timestamp",
            resource_policy=different,
        )
        with pytest.raises(RuntimeError, match="runtime_resource_policy_conflict"):
            next(chunks)


def test_complete_time_sized_frames_preserves_all_empty_chunk_schema() -> None:
    first = pd.DataFrame(
        {
            "timestamp": pd.Series(dtype="datetime64[ns]"),
            "asset_id": pd.Series(dtype="object"),
            "factor": pd.Series(dtype="float64"),
            "eligible": pd.Series(dtype="bool"),
        }
    )
    second = first.iloc[0:0].copy()

    chunks = list(
        iter_complete_time_sized_frames(
            iter((first, second)),
            timestamp_column="timestamp",
        )
    )

    assert len(chunks) == 1
    assert chunks[0].empty
    assert list(chunks[0].columns) == list(first.columns)
    assert chunks[0].dtypes.to_dict() == first.dtypes.to_dict()
    assert chunks[0].loc[:, ["asset_id", "factor", "eligible"]].empty


def test_complete_frames_accept_equivalent_timezone_representations() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": [
                pd.Timestamp(
                    "2024-01-15T12:00:00",
                    tz=timezone(timedelta(hours=1)),
                ),
                pd.Timestamp(
                    "2024-01-15T12:01:00",
                    tz=ZoneInfo("Europe/Paris"),
                ),
            ],
            "asset_id": ["asset", "asset"],
        }
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        chunks = list(
            iter_complete_time_sized_frames(
                iter((frame,)),
                timestamp_column="timestamp",
            )
        )

    assert sum(len(chunk) for chunk in chunks) == 2


def test_complete_time_sized_frames_rejects_single_timestamp_over_budget() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2026-01-01")] * 400,
            "value": ["x" * 20] * 400,
        }
    )
    payload_bytes = pandas_frame_payload_bytes(frame)
    policy = capture_test_policy(
        available_memory_bytes=payload_bytes * 2,
        explicit_cap_bytes=payload_bytes,
    )

    with pytest.raises(
        ValueError,
        match="research_indivisible_time_unit_exceeds_task_memory_budget",
    ):
        list(
            iter_complete_time_sized_frames(
                iter((frame,)),
                timestamp_column="timestamp",
                resource_policy=policy,
            )
        )


def test_complete_time_frames_allow_soft_overrun_without_explicit_cap() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "value": ["first", "second"],
        }
    )
    policy = capture_test_policy(available_memory_bytes=1_000_000)

    with runtime_resource_policy_scope(policy):
        with policy.lease("test.time.soft.blocker", policy.memory_budget_bytes):
            chunks = list(
                iter_complete_time_sized_frames(
                    iter((frame,)),
                    timestamp_column="timestamp",
                    resource_policy=policy,
                )
            )

    assert [chunk["value"].tolist() for chunk in chunks] == [["first"], ["second"]]
    assert policy.used_memory_bytes == 0


def test_complete_time_sized_frames_honor_live_root_reservations() -> None:
    frame = _single_time_frame()
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024 * 1024,
        explicit_cap_bytes=1 * 1024 * 1024,
    )

    with runtime_resource_policy_scope(policy):
        with policy.lease("outer_owner", policy.memory_budget_bytes):
            with pytest.raises(
                ValueError,
                match="research_indivisible_time_unit_exceeds_task_memory_budget",
            ):
                list(
                    iter_complete_time_sized_frames(
                        iter((frame,)),
                        timestamp_column="timestamp",
                    )
                )


def test_complete_time_frames_reserve_pending_before_next_input_pull() -> None:
    first = pd.DataFrame(
        {"timestamp": pd.to_datetime(["2026-01-01"]), "value": ["first"]}
    )
    second = pd.DataFrame(
        {"timestamp": pd.to_datetime(["2026-01-02"]), "value": ["second"]}
    )
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024 * 1024,
        explicit_cap_bytes=1024 * 1024,
    )

    def frames():
        yield first
        assert policy.used_memory_bytes >= pandas_frame_payload_bytes(first)
        yield second

    with runtime_resource_policy_scope(policy):
        assert (
            len(
                list(
                    iter_complete_time_sized_frames(
                        frames(),
                        timestamp_column="timestamp",
                    )
                )
            )
            == 2
        )


def test_pending_tail_admits_before_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2026-01-01")],
            "value": ["tail"],
        }
    )
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024 * 1024,
        explicit_cap_bytes=1024 * 1024,
    )
    events: list[str] = []
    real_lease = RuntimeResourcePolicy.lease
    real_copy = pd.DataFrame.copy

    def observe_lease(root, owner, reserved_bytes):  # noqa: ANN001
        if owner == "research.time.pending":
            events.append("lease")
        return real_lease(root, owner, reserved_bytes)

    def observe_copy(current, *args, **kwargs):  # noqa: ANN001
        if "lease" not in events:
            events.append("copy_before_lease")
        else:
            events.append("copy")
        return real_copy(current, *args, **kwargs)

    monkeypatch.setattr(RuntimeResourcePolicy, "lease", observe_lease)
    monkeypatch.setattr(pd.DataFrame, "copy", observe_copy)

    with runtime_resource_policy_scope(policy):
        list(
            iter_complete_time_sized_frames(
                iter((frame,)),
                timestamp_column="timestamp",
            )
        )

    assert events == ["lease", "copy"]
    assert policy.used_memory_bytes == 0


def test_time_chunk_admission_precedes_timestamp_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "value": ["x" * 20, "y" * 20],
        }
    )
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024 * 1024,
        explicit_cap_bytes=1,
    )

    def reject_conversion(*_args, **_kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("timestamp conversion ran before owner admission")

    monkeypatch.setattr(memory_owner, "strict_timestamp_series", reject_conversion)

    with pytest.raises(
        ValueError,
        match="research_indivisible_time_unit_exceeds_task_memory_budget",
    ):
        list(
            iter_complete_time_sized_frames(
                iter((frame,)),
                timestamp_column="timestamp",
                resource_policy=policy,
            )
        )


def test_time_chunk_refreshes_shared_headroom_before_next_session_copy() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2026-01-01")] * 750
            + [pd.Timestamp("2026-01-02")] * 750,
            "value": ["x" * 900] * 1_500,
        }
    )
    first_bytes = pandas_frame_payload_bytes(frame.iloc[:750].copy())
    second_bytes = pandas_frame_payload_bytes(frame.iloc[750:].copy())
    timestamp_bytes = int(
        frame["timestamp"].memory_usage(index=False, deep=True)
    ) + 40 * len(frame)
    hard_budget = max(first_bytes, second_bytes) + timestamp_bytes + 64 * 1024
    policy = capture_test_policy(
        available_memory_bytes=32 * 1024 * 1024,
        explicit_cap_bytes=hard_budget,
    )

    with runtime_resource_policy_scope(policy):
        chunks = iter_complete_time_sized_frames(
            iter((frame,)),
            timestamp_column="timestamp",
            resource_policy=policy,
        )
        first = next(chunks)
        assert first["timestamp"].nunique() == 1
        new_owner_bytes = hard_budget - timestamp_bytes - second_bytes + 1
        with policy.lease("outer_after_first_yield", new_owner_bytes):
            with pytest.raises(
                ValueError,
                match="research_indivisible_time_unit_exceeds_task_memory_budget",
            ):
                next(chunks)
        chunks.close()


def test_complete_time_frames_reject_boundary_peak_before_concat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2026-01-01")] * 100,
            "value": ["x" * 20] * 100,
        }
    )
    second = first.copy()
    pending_bytes = pandas_frame_payload_bytes(first)
    combined_bound = pending_bytes + pandas_frame_payload_bytes(second)
    inspection_bytes = int(
        second["timestamp"].memory_usage(index=False, deep=True)
    ) + 40 * len(second)
    hard_budget = pending_bytes + inspection_bytes + combined_bound - 1
    policy = capture_test_policy(
        available_memory_bytes=hard_budget * 2,
        explicit_cap_bytes=hard_budget,
    )

    def reject_concat(*_args, **_kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("concat allocated before peak budget preflight")

    monkeypatch.setattr(pd, "concat", reject_concat)

    chunks = iter_complete_time_sized_frames(
        iter((first, second)),
        timestamp_column="timestamp",
        resource_policy=policy,
    )
    with pytest.raises(
        ValueError,
        match="research_time_boundary_payloads_exceed_task_memory_budget",
    ):
        next(chunks)
    chunks.close()
    assert policy.used_memory_bytes == 0


def test_complete_time_frames_keep_boundary_payload_reserved_through_yield() -> None:
    first = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2026-01-01")] * 2,
            "value": [1, 2],
        }
    )
    second = first.copy()
    combined_bytes = pandas_frame_payload_bytes(
        pd.concat([first, second], ignore_index=True)
    )
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024 * 1024,
        explicit_cap_bytes=1 * 1024 * 1024,
    )
    chunks = iter_complete_time_sized_frames(
        iter((first, second)),
        timestamp_column="timestamp",
        resource_policy=policy,
    )

    combined = next(chunks)
    assert len(combined) == 4
    assert policy.used_memory_bytes >= combined_bytes
    chunks.close()
    assert policy.used_memory_bytes == 0


@pytest.mark.parametrize("exit_kind", ("normal", "body_error", "consumer_close"))
def test_complete_time_frames_close_upstream_after_activation(
    exit_kind: str,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024 * 1024,
        explicit_cap_bytes=1 * 1024 * 1024,
    )
    close_calls = 0

    class Frames:
        def __init__(self) -> None:
            first = pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(["2026-01-01"]),
                    "value": [1],
                }
            )
            second = pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(["2026-01-02"]),
                    "value": [2],
                }
            )
            if exit_kind == "body_error":
                first = first.rename(columns={"timestamp": "wrong"})
            self._frames = iter((first, second))

        def __iter__(self) -> Frames:
            return self

        def __next__(self) -> pd.DataFrame:
            return next(self._frames)

        def close(self) -> None:
            nonlocal close_calls
            close_calls += 1

    with runtime_resource_policy_scope(policy):
        chunks = iter_complete_time_sized_frames(
            Frames(),
            timestamp_column="timestamp",
        )
        if exit_kind == "normal":
            assert len(list(chunks)) == 2
        elif exit_kind == "body_error":
            with pytest.raises(
                ValueError,
                match="missing_time_chunk_timestamp_column",
            ):
                next(chunks)
        else:
            next(chunks)
            chunks.close()

    assert close_calls == 1


@pytest.mark.parametrize(
    "primary_type",
    (RuntimeError, MemoryError, KeyboardInterrupt, SystemExit, GeneratorExit),
)
def test_complete_time_frames_close_upstream_inside_policy_scope(
    primary_type: type[BaseException],
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024 * 1024,
        explicit_cap_bytes=1 * 1024 * 1024,
    )
    primary = primary_type("upstream_primary")
    closed_under: list[RuntimeResourcePolicy | None] = []

    class Frames:
        def __iter__(self) -> Frames:
            return self

        def __next__(self) -> pd.DataFrame:
            raise primary

        def close(self) -> None:
            closed_under.append(current_runtime_resource_policy())
            raise OSError("close_secondary")

    chunks = iter_complete_time_sized_frames(
        Frames(),
        timestamp_column="timestamp",
        resource_policy=policy,
    )
    with pytest.raises(primary_type) as raised:
        next(chunks)

    assert raised.value is primary
    assert closed_under == [policy]

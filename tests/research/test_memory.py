from __future__ import annotations

import gc
import sys
import warnings
from contextlib import contextmanager
from datetime import timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

import quant_research_platform.research._memory as memory_owner
from quant_research_platform.research._memory import (
    _concat_time_boundary_frames,
    iter_complete_time_sized_frames,
)
from quant_research_platform.runtime.resources import (
    RuntimeResourcePolicy,
    current_runtime_resource_policy,
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

    with pytest.raises(RuntimeError, match="runtime_resource_policy_retired"):
        next(chunks)


def test_lazy_memory_factory_rejects_different_explicit_policy_at_creation() -> None:
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
        with pytest.raises(RuntimeError, match="runtime_resource_policy_conflict"):
            iter_complete_time_sized_frames(
                iter((frame,)),
                timestamp_column="timestamp",
                resource_policy=different,
            )


@pytest.mark.parametrize("close_fails", (False, True))
def test_complete_memory_factory_closes_once_on_creation_policy_conflict(
    close_fails: bool,
) -> None:
    active = capture_test_policy(
        available_memory_bytes=16 * 1024 * 1024,
        explicit_cap_bytes=1 * 1024 * 1024,
    )
    different = capture_test_policy(
        available_memory_bytes=16 * 1024 * 1024,
        explicit_cap_bytes=2 * 1024 * 1024,
    )
    close_calls = 0

    class Frames:
        def __iter__(self) -> Frames:
            return self

        def __next__(self) -> pd.DataFrame:
            pytest.fail("frames consumed after creation-time conflict")

        def close(self) -> None:
            nonlocal close_calls
            close_calls += 1
            if close_fails:
                raise OSError("close_secondary")

    with runtime_resource_policy_scope(active):
        with pytest.raises(RuntimeError, match="runtime_resource_policy_conflict"):
            iter_complete_time_sized_frames(
                Frames(),
                timestamp_column="timestamp",
                resource_policy=different,
            )

    assert close_calls == 1


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
    policy = capture_test_policy(
        available_memory_bytes=1_000,
    )

    with pytest.raises(
        ValueError,
        match="research_indivisible_time_unit_exceeds_payload_target",
    ):
        list(
            iter_complete_time_sized_frames(
                iter((frame,)),
                timestamp_column="timestamp",
                resource_policy=policy,
            )
        )


def test_complete_time_sized_frames_honor_live_root_reservations() -> None:
    frame = _single_time_frame()
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024 * 1024,
        explicit_cap_bytes=1 * 1024 * 1024,
    )

    with runtime_resource_policy_scope(policy):
        with policy.reserve("outer_owner", policy.task_memory_budget_bytes):
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


def test_time_boundary_concat_rejects_peak_before_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2026-01-01")] * 200,
            "value": ["x" * 20] * 200,
        }
    )
    policy = capture_test_policy(
        available_memory_bytes=1_000,
    )

    def reject_concat(*_args, **_kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("concat allocated before peak budget preflight")

    monkeypatch.setattr(pd, "concat", reject_concat)

    with pytest.raises(
        ValueError,
        match="research_time_boundary_payloads_exceed_task_memory_budget",
    ):
        _concat_time_boundary_frames(
            frame.iloc[:100].copy(),
            frame.iloc[100:].copy(),
            resource_policy=policy,
        )


@pytest.mark.parametrize("exit_kind", ("normal", "body_error", "consumer_close"))
def test_complete_frames_transfer_captured_input_to_lazy_activation(
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


@pytest.mark.parametrize("policy_mode", ("explicit", "captured"))
@pytest.mark.parametrize("termination", ("double_close", "gc"))
@pytest.mark.parametrize("close_fails", (False, True))
def test_unactivated_complete_frames_close_once_without_unraisable(
    monkeypatch: pytest.MonkeyPatch,
    policy_mode: str,
    termination: str,
    close_fails: bool,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024 * 1024,
        explicit_cap_bytes=1 * 1024 * 1024,
    )
    next_calls = 0
    close_calls = 0
    unraisable: list[object] = []

    class Frames:
        def __iter__(self) -> Frames:
            return self

        def __next__(self) -> pd.DataFrame:
            nonlocal next_calls
            next_calls += 1
            raise AssertionError("unactivated frames were consumed")

        def close(self) -> None:
            nonlocal close_calls
            close_calls += 1
            if close_fails:
                raise OSError("close_secondary")

    frames = Frames()
    monkeypatch.setattr(sys, "unraisablehook", unraisable.append)
    if policy_mode == "captured":
        with runtime_resource_policy_scope(policy):
            chunks = iter_complete_time_sized_frames(
                frames,
                timestamp_column="timestamp",
            )
    else:
        chunks = iter_complete_time_sized_frames(
            frames,
            timestamp_column="timestamp",
            resource_policy=policy,
        )

    if termination == "double_close":
        chunks.close()
        assert close_calls == 1
        chunks.close()
    else:
        del chunks
        gc.collect()

    assert next_calls == 0
    assert close_calls == 1
    assert unraisable == []


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


@pytest.mark.parametrize(
    "entry_error",
    (
        RuntimeError("runtime_resource_policy_retired"),
        RuntimeError("runtime_resource_policy_conflict"),
        RuntimeError("runtime_resource_policy_concurrent_root"),
    ),
)
@pytest.mark.parametrize("close_fails", (False, True))
def test_complete_time_frames_close_once_when_policy_entry_fails(
    entry_error: RuntimeError,
    close_fails: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024 * 1024,
        explicit_cap_bytes=1 * 1024 * 1024,
    )
    close_calls = 0

    class Frames:
        def __iter__(self) -> Frames:
            return self

        def __next__(self) -> pd.DataFrame:
            pytest.fail("frames consumed after policy entry failure")

        def close(self) -> None:
            nonlocal close_calls
            close_calls += 1
            if close_fails:
                raise OSError("close_secondary")

    @contextmanager
    def failing_scope(_policy):  # noqa: ANN001, ANN202
        raise entry_error
        yield

    monkeypatch.setattr(memory_owner, "runtime_resource_policy_scope", failing_scope)
    chunks = iter_complete_time_sized_frames(
        Frames(),
        timestamp_column="timestamp",
        resource_policy=policy,
    )

    with pytest.raises(RuntimeError) as raised:
        next(chunks)

    assert raised.value is entry_error
    assert close_calls == 1


@pytest.mark.parametrize("close_fails", (False, True))
def test_complete_time_frames_close_once_when_policy_resolution_fails(
    close_fails: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolution_error = RuntimeError("policy_factory_failed")
    close_calls = 0

    class Frames:
        def __iter__(self) -> Frames:
            return self

        def __next__(self) -> pd.DataFrame:
            pytest.fail("frames consumed after policy resolution failure")

        def close(self) -> None:
            nonlocal close_calls
            close_calls += 1
            if close_fails:
                raise OSError("close_secondary")

    def fail_resolution() -> RuntimeResourcePolicy:
        raise resolution_error

    monkeypatch.setattr(
        memory_owner,
        "active_runtime_resource_policy",
        fail_resolution,
    )
    chunks = iter_complete_time_sized_frames(
        Frames(),
        timestamp_column="timestamp",
    )

    with pytest.raises(BaseException) as raised:
        next(chunks)

    assert raised.value is resolution_error
    assert close_calls == 1

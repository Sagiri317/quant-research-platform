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


@pytest.mark.parametrize("available", (65_536, 1_000_000))
def test_indivisible_borrowed_frame_admits_new_copy_before_allocation(
    monkeypatch, available
):
    frame = pd.DataFrame(
        {"timestamp": [pd.Timestamp("2026-01-01")], "value": ["x" * 700_000]}
    )
    before = frame.copy(deep=True)
    copies = []
    actual_copy = pd.DataFrame.copy

    def copy(value, *args, **kwargs):
        copies.append(len(value))
        return actual_copy(value, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "copy", copy)
    policy = capture_test_policy(available_memory_bytes=available)
    with runtime_resource_policy_scope(policy):
        chunks = iter_complete_time_sized_frames(
            iter((frame,)), timestamp_column="timestamp", resource_policy=policy
        )
        if available == 65_536:
            with pytest.raises(
                ValueError,
                match="research_indivisible_time_unit_exceeds_task_memory_budget",
            ):
                list(chunks)
            assert not copies
        else:
            output = next(chunks)
            assert policy.memory_budget_bytes < policy.used_memory_bytes < available
            pd.testing.assert_frame_equal(output, frame, check_exact=True)
            assert list(chunks) == []
            assert copies == [1]
        assert policy.used_memory_bytes == 0
    pd.testing.assert_frame_equal(frame, before, check_exact=True)


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


@pytest.mark.parametrize(
    ("session_chunks", "failure"),
    (
        (False, "upstream"),
        (True, "upstream"),
        (False, "research.time.inspect"),
        (True, "research.time.inspect"),
        (False, "research.time.pending"),
        (True, "research.time.pending"),
        (True, "research.session.inspect"),
    ),
)
@pytest.mark.parametrize("consumer_primary", (False, True))
def test_complete_time_frames_early_close_reports_cleanup_failures(
    monkeypatch, session_chunks, failure, consumer_primary
):
    from quant_research_platform.policy.lifecycle import closing_iterator
    from quant_research_platform.runtime.resources import MemoryLease

    policy = capture_test_policy(
        available_memory_bytes=16 * 1024**2, explicit_cap_bytes=1024**2
    )
    closed = []

    class Frames:
        def __init__(self):
            self.frames = iter(
                (
                    pd.DataFrame(
                        {
                            "timestamp": pd.date_range("2026-01-01", periods=3),
                            "value": [1, 2, 3],
                        }
                    ),
                )
            )

        def __iter__(self):
            return self

        def __next__(self):
            return next(self.frames)

        def close(self):
            closed.append(current_runtime_resource_policy())
            if failure == "upstream":
                raise OSError("upstream_close")

    factory = (
        iter_complete_session_frames
        if session_chunks
        else iter_complete_time_sized_frames
    )
    chunks = factory(Frames(), timestamp_column="timestamp", resource_policy=policy)
    next(chunks)
    if failure == "research.time.pending":
        if session_chunks:
            next(chunks)
        next(chunks)
    original_close = MemoryLease.close
    failed = []

    def fail_release(lease):
        original_close(lease)
        if lease.owner == failure:
            failed.append(lease.owner)
            raise OSError("time_lease_release")

    monkeypatch.setattr(MemoryLease, "close", fail_release)
    primary = KeyboardInterrupt("time_consumer_primary")
    with pytest.raises(KeyboardInterrupt if consumer_primary else OSError) as caught:
        if consumer_primary:
            with closing_iterator(chunks):
                raise primary
        else:
            chunks.close()
    if consumer_primary:
        assert caught.value is primary
        assert any("OSError" in note for note in getattr(primary, "__notes__", ()))
    assert closed == [policy]
    assert failed == ([] if failure == "upstream" else [failure])
    assert policy.used_memory_bytes == 0


@pytest.mark.parametrize("rows", (3, 30))
@pytest.mark.parametrize(
    "entry", (iter_complete_time_sized_frames, iter_complete_session_frames)
)
def test_session_concat_accounts_for_replacement_range_index(rows, entry):
    frames = [
        pd.DataFrame(
            {
                "timestamp": pd.Timestamp("2020-01-01"),
                "value": list(range(offset, offset + rows)),
            },
            index=pd.Index([i * i + offset for i in range(rows)]),
        )
        for offset in (0, rows)
    ]
    with runtime_resource_policy_scope(explicit_cap_bytes=64 * 1024 * 1024) as policy:
        chunks = entry(
            iter(frames), timestamp_column="timestamp", resource_policy=policy
        )
        try:
            values = [value for chunk in chunks for value in chunk["value"]]
        finally:
            chunks.close()
        assert values == list(range(2 * rows))
        assert policy.used_memory_bytes == 0


@pytest.mark.parametrize("chunk_rows", (1, 4, 8))
def test_model_training_concat_keeps_irregularly_selected_rows(chunk_rows):
    from quant_research_platform.contracts.models import ModelSpec
    from quant_research_platform.research.models.fitting import (
        _fit_model_result_owned_from_frames,
    )
    from quant_research_platform.research.models.supervised import (
        build_estimator_from_model_spec,
    )

    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2020-01-01"] * 4 + ["2020-01-02"] * 4),
            "label_available_at": pd.Timestamp("2020-01-03"),
            "x": [float(i) for i in range(8)],
            "label": [2.0 * i + 1.0 for i in range(8)],
            "eligible": [True, False, True, True] * 2,
        }
    )
    spec = ModelSpec(
        model_id="concat_index",
        model_algorithm="linear_regression",
        input_factor_columns=("x",),
        label_column="label",
    )
    estimator = build_estimator_from_model_spec(spec)
    with runtime_resource_policy_scope(explicit_cap_bytes=64 * 1024 * 1024) as policy:
        fitted = _fit_model_result_owned_from_frames(
            (frame.iloc[i : i + chunk_rows] for i in range(0, 8, chunk_rows)),
            spec,
            estimator=estimator,
            fitting_cutoff="2020-01-04",
        )
        try:
            assert fitted.value.metadata["rows_after_input_label_dropna"] == 6
            actual = fitted.value.prediction_model.predict(
                pd.DataFrame({"x": [0.0, 1.0, 4.0]})
            )
            assert actual.tolist() == pytest.approx([1.0, 3.0, 9.0])
        finally:
            fitted.release(owner="test.concat_index")
        assert policy.used_memory_bytes == 0


@pytest.mark.parametrize("kind", ("numeric", "categorical"))
def test_session_concat_accounts_for_promoted_column_representation(kind):
    if kind == "numeric":
        left = pd.Series(range(1000), dtype="float32")
        right = pd.Series(range(1000), dtype="float64")
        expected = [float(i) for i in range(1000)] * 2
    else:
        left = pd.Series(pd.Categorical(["synthetic_a"] * 1000))
        right = pd.Series(pd.Categorical(["synthetic_b"] * 1000))
        expected = ["synthetic_a"] * 1000 + ["synthetic_b"] * 1000
    frames = [
        pd.DataFrame({"timestamp": pd.Timestamp("2020-01-01"), "value": values})
        for values in (left, right)
    ]
    with runtime_resource_policy_scope(explicit_cap_bytes=64 * 1024 * 1024) as policy:
        stream = iter_complete_time_sized_frames(
            iter(frames), timestamp_column="timestamp", resource_policy=policy
        )
        try:
            actual = [value for frame in stream for value in frame.value]
        finally:
            stream.close()
        assert actual == expected
        assert policy.used_memory_bytes == 0


@pytest.mark.parametrize("chunk_count", (2, 20))
def test_session_concat_counts_shared_categories_once(monkeypatch, chunk_count):
    from quant_research_platform.runtime.resources import MemoryLease

    frames = [
        pd.DataFrame(
            {
                "timestamp": pd.Timestamp("2020-01-01"),
                "value": pd.Categorical(["a", "b"], categories=["a", "b"]),
            }
        )
        for _ in range(chunk_count)
    ]
    expected = [
        int(pd.concat(frames[:count], ignore_index=True).memory_usage(deep=True).sum())
        for count in range(2, chunk_count + 1)
    ]
    increments = []
    original = MemoryLease.grow

    def observed_grow(lease, byte_count):
        if lease.owner == "research.time.pending":
            increments.append(byte_count)
        return original(lease, byte_count)

    monkeypatch.setattr(MemoryLease, "grow", observed_grow)
    with runtime_resource_policy_scope(explicit_cap_bytes=64 * 1024 * 1024) as policy:
        stream = iter_complete_time_sized_frames(
            iter(frames), timestamp_column="timestamp", resource_policy=policy
        )
        try:
            actual = [value for frame in stream for value in frame.value]
        finally:
            stream.close()
        assert actual == ["a", "b"] * chunk_count
        assert increments == expected
        assert policy.used_memory_bytes == 0

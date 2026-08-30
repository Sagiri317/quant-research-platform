from __future__ import annotations

import weakref

import pandas as pd
import pytest

import quant_research_platform.research.filtering as filtering_owner
from quant_research_platform.research.filtering import (
    _filter_mode_scope,
    _filter_working_set_bound,
    apply_filter_mode,
)
from quant_research_platform.runtime.resources import runtime_resource_policy_scope
from tests.helpers.runtime_resources import capture_test_policy


def test_apply_filter_mode_keeps_existing_integer_mask_contract() -> None:
    frame = pd.DataFrame(
        {
            "eligible": [1, 0],
            "value": [10.0, 20.0],
        }
    )

    filtered, metadata = apply_filter_mode(
        frame,
        mode="eligible",
        metadata_key="filter_mode",
    )

    assert filtered["value"].tolist() == [10.0]
    assert metadata["rows_input"] == 2
    assert metadata["rows_after_filter_mode"] == 1


def test_apply_filter_mode_rejects_string_bool_masks() -> None:
    frame = pd.DataFrame(
        {
            "eligible": ["False", "True"],
            "value": [10.0, 20.0],
        }
    )

    with pytest.raises(ValueError, match="invalid_filter_column_dtype: eligible"):
        apply_filter_mode(frame, mode="eligible", metadata_key="filter_mode")


def test_filter_mode_scope_rejects_before_mask_allocation() -> None:
    frame = pd.DataFrame({"eligible": [True], "value": [10.0]})
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024**2,
        explicit_cap_bytes=64 * 1024,
    )
    with runtime_resource_policy_scope(policy):
        with policy.lease("test.filter.blocker", policy.memory_budget_bytes):
            with pytest.raises(
                ValueError,
                match="research_filter_working_set_exceeds_task_memory_budget",
            ):
                with _filter_mode_scope(
                    frame,
                    mode="eligible",
                    metadata_key="filter_mode",
                ):
                    pytest.fail("filter scope allocated under a full budget")
    assert policy.used_memory_bytes == 0


def test_filter_mode_scope_allows_soft_target_overrun_without_explicit_cap() -> None:
    frame = pd.DataFrame({"eligible": [True], "value": [10.0]})
    policy = capture_test_policy(available_memory_bytes=1_000_000)

    with runtime_resource_policy_scope(policy):
        with policy.lease("test.filter.soft.blocker", policy.memory_budget_bytes):
            with _filter_mode_scope(
                frame,
                mode="eligible",
                metadata_key="filter_mode",
            ) as (filtered, _metadata):
                assert filtered["value"].tolist() == [10.0]
    assert policy.used_memory_bytes == 0


def test_filter_mode_scope_owns_filtered_frame_until_exit() -> None:
    frame = pd.DataFrame(
        {
            "eligible": [True, False],
            "value": [10.0, 20.0],
        }
    )
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024**2,
        explicit_cap_bytes=1 * 1024**2,
    )
    with runtime_resource_policy_scope(policy):
        with _filter_mode_scope(
            frame,
            mode="eligible",
            metadata_key="filter_mode",
        ) as (filtered, metadata):
            assert filtered["value"].tolist() == [10.0]
            assert metadata["rows_after_filter_mode"] == 1
            assert policy.used_memory_bytes > 0
        assert policy.used_memory_bytes == 0


@pytest.mark.parametrize(
    ("index", "eligible", "expected_index_type"),
    (
        (pd.RangeIndex(4), [True, True, True, True], "RangeIndex"),
        (
            pd.RangeIndex(1000),
            [index % 3 != 2 for index in range(1000)],
            "Index",
        ),
        (
            pd.Index(range(1000), name="asset_row"),
            [index % 3 != 2 for index in range(1000)],
            "Index",
        ),
        (pd.RangeIndex(0), [], "RangeIndex"),
        (pd.RangeIndex(1000), [False] * 1000, "RangeIndex"),
    ),
)
def test_filter_mode_scope_uses_formal_bound_before_materialization(
    index: pd.Index,
    eligible: list[bool],
    expected_index_type: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame(
        {"eligible": pd.array(eligible, dtype="bool")},
        index=index,
    )
    bound = _filter_working_set_bound(frame, mode="eligible")

    accepted = capture_test_policy(
        available_memory_bytes=16 * 1024**2,
        explicit_cap_bytes=bound,
    )
    with runtime_resource_policy_scope(accepted):
        with _filter_mode_scope(
            frame,
            mode="eligible",
            metadata_key="filter_mode",
        ) as (filtered, _metadata):
            assert type(filtered.index).__name__ == expected_index_type
            assert filtering_owner.pandas_frame_payload_bytes(filtered) <= bound
    assert accepted.used_memory_bytes == 0

    rejected = capture_test_policy(
        available_memory_bytes=16 * 1024**2,
        explicit_cap_bytes=bound - 1,
    )
    called = False

    def materialize(*_args: object, **_kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("filter_materialized_before_bound_admission")

    monkeypatch.setattr(filtering_owner, "apply_filter_mode", materialize)
    with runtime_resource_policy_scope(rejected):
        with pytest.raises(
            ValueError,
            match="research_filter_working_set_exceeds_task_memory_budget",
        ):
            with _filter_mode_scope(
                frame,
                mode="eligible",
                metadata_key="filter_mode",
            ):
                pytest.fail("formal bound minus one was admitted")
    assert not called
    assert rejected.used_memory_bytes == 0


def test_filter_mode_scope_drops_its_filtered_alias_before_close() -> None:
    frame = pd.DataFrame(
        {
            "eligible": [True, False],
            "value": [10.0, 20.0],
        }
    )
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024**2,
        explicit_cap_bytes=1 * 1024**2,
    )
    with runtime_resource_policy_scope(policy):
        with _filter_mode_scope(
            frame,
            mode="eligible",
            metadata_key="filter_mode",
        ) as (filtered, _metadata):
            filtered_ref = weakref.ref(filtered)
            assert policy.used_memory_bytes > 0
            del filtered
        assert policy.used_memory_bytes == 0
        assert filtered_ref() is None


def test_filter_mode_scope_releases_empty_filtered_frame() -> None:
    frame = pd.DataFrame(
        {
            "eligible": pd.Series(dtype="bool"),
            "value": pd.Series(dtype="float64"),
        }
    )
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024**2,
        explicit_cap_bytes=1 * 1024**2,
    )
    with runtime_resource_policy_scope(policy):
        with _filter_mode_scope(
            frame,
            mode="eligible",
            metadata_key="filter_mode",
        ) as (filtered, metadata):
            assert filtered.empty
            assert metadata["rows_after_filter_mode"] == 0
        assert policy.used_memory_bytes == 0


def test_filter_mode_scope_closes_lease_if_metadata_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame(
        {
            "eligible": [True, False],
            "value": [10.0, 20.0],
        }
    )
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024**2,
        explicit_cap_bytes=1 * 1024**2,
    )
    primary = RuntimeError("filter_metadata_primary")
    monkeypatch.setattr(
        filtering_owner,
        "filter_definition",
        lambda _mode: (_ for _ in ()).throw(primary),
    )

    with runtime_resource_policy_scope(policy):
        with pytest.raises(RuntimeError) as raised:
            with _filter_mode_scope(
                frame,
                mode="eligible",
                metadata_key="filter_mode",
            ):
                pytest.fail("metadata construction unexpectedly succeeded")

    assert raised.value is primary
    assert policy.used_memory_bytes == 0

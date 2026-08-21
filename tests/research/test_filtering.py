from __future__ import annotations

import pandas as pd
import pytest

import quant_research_platform.research.filtering as filtering_owner
from quant_research_platform.research.filtering import (
    _filter_mode_scope,
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
        with policy.reserve("test.filter.blocker", policy.memory_budget_bytes):
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

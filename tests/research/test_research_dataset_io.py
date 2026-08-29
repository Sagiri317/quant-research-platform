from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

import quant_research_platform.research.datasets.io as research_dataset_io
import quant_research_platform.storage.datasets.reading as dataset_reading
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.runtime.resources import (
    RuntimeResourcePolicy,
    runtime_resource_policy_scope,
)
from quant_research_platform.storage.layout import StorageLayout
from tests.helpers.runtime_resources import (
    capture_test_policy,
)


def test_research_dataset_chunk_wrapper_captures_outer_at_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024 * 1024,
        explicit_cap_bytes=1 * 1024 * 1024,
    )
    observed: list[RuntimeResourcePolicy] = []

    @contextmanager
    def open_session(*_args: object, resource_policy=None, **_kwargs: object):  # type: ignore[no-untyped-def]
        observed.append(resource_policy)
        yield SimpleNamespace(resource_policy=resource_policy)

    monkeypatch.setattr(dataset_reading, "open_dataset_session", open_session)
    monkeypatch.setattr(
        dataset_reading,
        "iter_dataset_chunks",
        lambda *_a, **_k: iter(()),
    )

    with runtime_resource_policy_scope(policy):
        chunks = research_dataset_io.iter_research_dataset_chunks(
            StorageLayout(Path("unused")),
            market_id=MarketId("demo"),
            dataset_id="dataset",
        )
        assert list(chunks) == []

    assert observed == [policy]


def test_research_dataset_chunk_wrapper_rejects_delayed_consumption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024 * 1024,
        explicit_cap_bytes=1 * 1024 * 1024,
    )
    monkeypatch.setattr(
        dataset_reading,
        "open_dataset_session",
        lambda *_args, **_kwargs: pytest.fail("session_opened_for_retired_policy"),
    )

    with runtime_resource_policy_scope(policy):
        chunks = research_dataset_io.iter_research_dataset_chunks(
            StorageLayout(Path("unused")),
            market_id=MarketId("demo"),
            dataset_id="dataset",
        )

    with pytest.raises(RuntimeError, match="runtime_resource_policy_not_active"):
        next(chunks)

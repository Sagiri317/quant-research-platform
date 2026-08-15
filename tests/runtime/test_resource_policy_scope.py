from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

import quant_research_platform.storage._dataset_reading as dataset_reading
from quant_research_platform.runtime.resources import (
    RuntimeResourcePolicy,
    active_runtime_resource_policy,
    bind_runtime_resource_policy,
    current_runtime_resource_policy,
    runtime_resource_policy_scope,
)
from quant_research_platform.storage._dataset_types import DatasetReadMode
from tests.helpers.runtime_resources import capture_test_policy


def _policy(megabytes: int) -> RuntimeResourcePolicy:
    return capture_test_policy(
        available_memory_bytes=16 * 1024 * 1024,
        rss_bytes=1024,
        explicit_cap_bytes=megabytes * 1024 * 1024,
    )


def test_bind_runtime_resource_policy_truth_table() -> None:
    active = _policy(1)
    different = _policy(2)

    assert bind_runtime_resource_policy(None) is None
    assert bind_runtime_resource_policy(active) is active
    with runtime_resource_policy_scope(active):
        assert bind_runtime_resource_policy(None) is active
        assert bind_runtime_resource_policy(active) is active
        with pytest.raises(RuntimeError, match="runtime_resource_policy_conflict"):
            bind_runtime_resource_policy(different)


def test_scope_is_same_object_reentrant_and_rejects_other_root() -> None:
    outer = _policy(1)
    other = _policy(2)

    with runtime_resource_policy_scope(outer):
        with runtime_resource_policy_scope(outer):
            assert active_runtime_resource_policy() is outer
        with pytest.raises(RuntimeError, match="runtime_resource_policy_conflict"):
            with runtime_resource_policy_scope(other):
                pass

    assert current_runtime_resource_policy() is None


def test_retired_policy_cannot_start_a_second_root() -> None:
    retired = _policy(1)
    with runtime_resource_policy_scope(retired):
        assert current_runtime_resource_policy() is retired

    with pytest.raises(RuntimeError, match="runtime_resource_policy_retired"):
        with runtime_resource_policy_scope(retired):
            pass


@pytest.mark.parametrize(
    "primary",
    [RuntimeError("body"), MemoryError(), KeyboardInterrupt(), SystemExit()],
)
def test_scope_releases_after_every_primary(primary: BaseException) -> None:
    first = _policy(1)
    second = _policy(2)

    with pytest.raises(type(primary)):
        with runtime_resource_policy_scope(first):
            raise primary

    with runtime_resource_policy_scope(second):
        assert current_runtime_resource_policy() is second


def test_lazy_read_binds_creation_policy_and_rejects_other_consumer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created_under = _policy(1)
    consumed_under = _policy(2)
    observed: list[RuntimeResourcePolicy | None] = []

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

    with runtime_resource_policy_scope(created_under):
        lazy = dataset_reading.read_dataset(
            Path("unused"),
            mode=DatasetReadMode.CHUNKED,
            resource_policy=created_under,
        )
    with runtime_resource_policy_scope(consumed_under):
        with pytest.raises(RuntimeError, match="runtime_resource_policy_conflict"):
            list(lazy)

    assert observed == []


def test_lazy_read_without_outer_admits_one_fresh_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[RuntimeResourcePolicy] = []

    class Session:
        def __init__(self, resource_policy: RuntimeResourcePolicy) -> None:
            self.resource_policy = resource_policy

        def __enter__(self) -> Session:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    @contextmanager
    def open_session(*_args: object, resource_policy=None, **_kwargs: object):  # type: ignore[no-untyped-def]
        issued = resource_policy or active_runtime_resource_policy()
        observed.append(issued)
        with runtime_resource_policy_scope(issued):
            with Session(issued) as session:
                yield session

    monkeypatch.setattr(dataset_reading, "open_dataset_session", open_session)
    monkeypatch.setattr(
        dataset_reading,
        "iter_dataset_chunks",
        lambda *_a, **_k: iter(()),
    )

    lazy = dataset_reading.read_dataset(
        Path("unused"),
        mode=DatasetReadMode.CHUNKED,
        resource_policy=None,
    )

    assert list(lazy) == []
    assert len(observed) == 1


def test_suspended_public_read_rejects_a_second_independent_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    issued = _policy(1)

    @contextmanager
    def open_session(*_args: object, resource_policy=None, **_kwargs: object):  # type: ignore[no-untyped-def]
        policy = resource_policy or issued
        with runtime_resource_policy_scope(policy):
            yield SimpleNamespace(resource_policy=policy)

    monkeypatch.setattr(dataset_reading, "open_dataset_session", open_session)
    monkeypatch.setattr(
        dataset_reading,
        "iter_dataset_chunks",
        lambda *_a, **_k: iter((SimpleNamespace(),)),
    )

    first = dataset_reading.read_dataset(
        Path("first"),
        mode=DatasetReadMode.CHUNKED,
        resource_policy=None,
    )
    assert not isinstance(first, SimpleNamespace)
    next(first)
    try:
        with pytest.raises(
            RuntimeError,
            match="runtime_resource_policy_competing_root",
        ):
            dataset_reading.read_dataset(
                Path("second"),
                mode=DatasetReadMode.CHUNKED,
                resource_policy=None,
            )
    finally:
        first.close()  # type: ignore[union-attr]


@pytest.mark.parametrize(
    "root_name",
    (
        "csmar_interim_frames",
        "csmar_interim_materialize",
        "csmar_raw_files",
        "csmar_raw_dataset_files",
        "csmar_raw_inspect",
        "csmar_raw_tree",
        "csmar_raw_chunks",
        "binance_interim_frames",
        "binance_interim_materialize",
        "a_share_processed_materialize",
        "crypto_processed_materialize",
    ),
)
def test_source_market_public_roots_reject_ambient_policy(
    root_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from quant_research_platform.contracts.ids import SourceId
    from quant_research_platform.contracts.time import TimeWindow
    from quant_research_platform.markets.implementations.a_share.processed import (
        materializer as a_share_materializer,
    )
    from quant_research_platform.markets.implementations.crypto.processed import (
        materializer as crypto_materializer,
    )
    from quant_research_platform.sources.binance import interim as binance_interim
    from quant_research_platform.sources.csmar import interim as csmar_interim
    from quant_research_platform.sources.csmar import raw_exports as csmar_raw
    from quant_research_platform.sources.csmar.raw_schemas import RAW_TABLE_CONTRACTS
    from quant_research_platform.storage.layout import StorageLayout

    policy = _policy(8)
    layout = StorageLayout(project_root=tmp_path)
    monkeypatch.setattr(
        csmar_interim,
        "_materialize_interim_data",
        lambda **_: object(),
    )
    monkeypatch.setattr(
        a_share_materializer,
        "_materialize_processed_data_with_policy",
        lambda **_: {},
    )
    monkeypatch.setattr(
        crypto_materializer,
        "_materialize_processed_data_with_policy",
        lambda **_: {},
    )

    def invoke() -> None:
        if root_name == "csmar_interim_frames":
            csmar_interim.iter_interim_frames(
                data_id=csmar_interim.DEFAULT_INTERIM_DATA_IDS[0],
                input_files=(),
                start_date=None,
                end_date=None,
            )
        elif root_name == "csmar_interim_materialize":
            csmar_interim.materialize_interim_data(layout=layout, data_ids=())
        elif root_name == "csmar_raw_files":
            csmar_raw.iter_raw_export_files(tmp_path)
        elif root_name == "csmar_raw_dataset_files":
            csmar_raw.dataset_files_with_identity_by_id(tmp_path)
        elif root_name == "csmar_raw_inspect":
            csmar_raw.inspect_raw_export(tmp_path / "missing.csv")
        elif root_name == "csmar_raw_tree":
            csmar_raw.inspect_raw_export_tree(tmp_path)
        elif root_name == "csmar_raw_chunks":
            csmar_raw.iter_private_export_chunks(
                tmp_path / "missing.csv",
                dataset=next(iter(RAW_TABLE_CONTRACTS)),
                chunksize=1,
            )
        elif root_name == "binance_interim_frames":
            next(
                binance_interim.iter_interim_frames(
                    data_id=next(iter(binance_interim.INTERIM_COLUMNS_BY_DATA_ID)),
                    input_files=(),
                )
            )
        elif root_name == "binance_interim_materialize":
            binance_interim.materialize_interim_data(layout, data_ids=())
        elif root_name == "a_share_processed_materialize":
            a_share_materializer.materialize_processed_data(
                layout=layout,
                source_id=SourceId("csmar"),
                time_window=TimeWindow("2024-01-01", "2024-01-02"),
                data_ids=(),
            )
        else:
            crypto_materializer.materialize_processed_data(
                layout=layout,
                source_id=SourceId("binance"),
                time_window=TimeWindow("2024-01-01", "2024-01-02"),
                data_ids=(),
            )

    with runtime_resource_policy_scope(policy):
        with pytest.raises(
            RuntimeError,
            match="^runtime_resource_policy_competing_root$",
        ):
            invoke()


def _lazy_source_root(
    root_name: str,
    tmp_path: Path,
    *,
    resource_policy: RuntimeResourcePolicy | None = None,
):  # type: ignore[no-untyped-def]
    from quant_research_platform.sources.binance import raw_exports as binance_raw
    from quant_research_platform.sources.csmar import interim as csmar_interim
    from quant_research_platform.sources.csmar import raw_exports as csmar_raw
    from quant_research_platform.sources.csmar.raw_schemas import RAW_TABLE_CONTRACTS

    if root_name == "csmar_interim_frames":
        return csmar_interim.iter_interim_frames(
            data_id=csmar_interim.DEFAULT_INTERIM_DATA_IDS[0],
            input_files=(),
            start_date=None,
            end_date=None,
            resource_policy=resource_policy,
        )
    if root_name == "csmar_raw_files":
        return csmar_raw.iter_raw_export_files(
            tmp_path,
            resource_policy=resource_policy,
        )
    if root_name == "csmar_index_rows":
        return csmar_raw.iter_index_info_rows(
            tmp_path / "missing.csv",
            resource_policy=resource_policy,
        )
    if root_name == "csmar_raw_chunks":
        return csmar_raw.iter_private_export_chunks(
            tmp_path / "missing.csv",
            dataset=next(iter(RAW_TABLE_CONTRACTS)),
            chunksize=1,
            resource_policy=resource_policy,
        )
    if root_name == "csmar_dataset_files":
        return csmar_raw.dataset_files_with_identity_by_id(
            tmp_path,
            resource_policy=resource_policy,
        )
    if root_name == "csmar_inspect_tree":
        assert resource_policy is None
        return csmar_raw.inspect_raw_export_tree(tmp_path)
    if root_name == "binance_raw_envelopes":
        return binance_raw.iter_raw_envelopes(
            tmp_path / "missing.jsonl",
            resource_policy=resource_policy,
        )
    raise AssertionError(root_name)


@pytest.mark.parametrize(
    "root_name",
    (
        "csmar_interim_frames",
        "csmar_raw_files",
        "csmar_index_rows",
        "csmar_raw_chunks",
        "csmar_dataset_files",
        "csmar_inspect_tree",
        "binance_raw_envelopes",
    ),
)
def test_lazy_source_root_rejects_unrelated_consumer_policy(
    root_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created_policy = _policy(4)
    consumer_policy = _policy(8)
    monkeypatch.setattr(
        RuntimeResourcePolicy,
        "capture",
        classmethod(lambda _cls, **_kwargs: created_policy),
    )
    lazy = _lazy_source_root(root_name, tmp_path)

    with runtime_resource_policy_scope(consumer_policy):
        with pytest.raises(
            RuntimeError,
            match="^runtime_resource_policy_competing_root$",
        ):
            next(iter(lazy))


@pytest.mark.parametrize(
    "root_name",
    (
        "csmar_interim_frames",
        "csmar_raw_files",
        "csmar_index_rows",
        "csmar_raw_chunks",
        "csmar_dataset_files",
        "binance_raw_envelopes",
    ),
)
def test_lazy_source_root_keeps_explicit_policy_for_delayed_consumption(
    root_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from quant_research_platform.sources.binance import raw_exports as binance_raw
    from quant_research_platform.sources.csmar import interim as csmar_interim
    from quant_research_platform.sources.csmar import raw_exports as csmar_raw

    class ScopeObserved(Exception):
        pass

    observed: list[RuntimeResourcePolicy] = []

    @contextmanager
    def observe_scope(policy: RuntimeResourcePolicy):
        observed.append(policy)
        raise ScopeObserved
        yield  # pragma: no cover

    module = (
        csmar_interim
        if root_name == "csmar_interim_frames"
        else binance_raw
        if root_name == "binance_raw_envelopes"
        else csmar_raw
    )
    monkeypatch.setattr(module, "runtime_resource_policy_scope", observe_scope)
    policy = _policy(8)
    lazy = _lazy_source_root(root_name, tmp_path, resource_policy=policy)

    with pytest.raises(ScopeObserved):
        next(iter(lazy))

    assert observed == [policy]

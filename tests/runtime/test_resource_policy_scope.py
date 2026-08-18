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
        with pytest.raises(
            RuntimeError,
            match="^runtime_resource_policy_competing_root$",
        ):
            bind_runtime_resource_policy(None)
        assert (
            bind_runtime_resource_policy(None, allow_implicit_active=True) is active
        )
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


def test_scope_none_is_the_sole_new_root_opener(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[int | None] = []
    original_capture = RuntimeResourcePolicy.capture

    def capture(
        cls: type[RuntimeResourcePolicy],
        *,
        explicit_cap_bytes: int | None = None,
    ) -> RuntimeResourcePolicy:
        captured.append(explicit_cap_bytes)
        return original_capture(
            explicit_cap_bytes=1 * 1024 * 1024,
        )

    monkeypatch.setattr(RuntimeResourcePolicy, "capture", classmethod(capture))
    with runtime_resource_policy_scope(
        None,
        explicit_cap_bytes=1 * 1024 * 1024,
    ) as policy:
        assert current_runtime_resource_policy() is policy
    assert current_runtime_resource_policy() is None
    assert captured == [1 * 1024 * 1024]


def test_scope_none_rejects_active_root_without_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_capture(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("scope(None) must not capture under an active root")

    active = _policy(1)
    monkeypatch.setattr(RuntimeResourcePolicy, "capture", classmethod(fail_capture))
    with runtime_resource_policy_scope(active):
        with pytest.raises(
            RuntimeError,
            match="^runtime_resource_policy_competing_root$",
        ):
            with runtime_resource_policy_scope(None):
                pass


def test_explicit_scope_validates_without_capturing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_capture(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("explicit scope must not capture")

    policy = _policy(1)
    monkeypatch.setattr(RuntimeResourcePolicy, "capture", classmethod(fail_capture))
    with pytest.raises(ValueError, match="invalid_runtime_resource_explicit_cap"):
        with runtime_resource_policy_scope(
            policy,
            explicit_cap_bytes=1 * 1024 * 1024,
        ):
            pass
    with runtime_resource_policy_scope(policy) as scoped:
        assert scoped is policy


def test_retired_policy_cannot_start_a_second_root() -> None:
    retired = _policy(1)
    with runtime_resource_policy_scope(retired):
        assert current_runtime_resource_policy() is retired

    with pytest.raises(RuntimeError, match="runtime_resource_policy_not_active"):
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


@pytest.mark.parametrize(
    "root_name",
    (
        "research_dataset",
        "model_fit",
        "validation_summary",
        "schema_inventory",
    ),
)
def test_non_source_public_roots_reject_ambient_policy_before_work(
    root_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from quant_research_platform.research.datasets import assembler
    from quant_research_platform.research.models import fitting
    from quant_research_platform.workflows.diagnostics import (
        schema_inventory,
        validation_summary,
    )

    work: list[str] = []

    def invoke() -> None:
        if root_name == "research_dataset":
            monkeypatch.setattr(
                assembler,
                "_assemble_research_dataset_with_policy",
                lambda **_: work.append("assemble"),
            )
            assembler.assemble_research_dataset(
                layout=object(),  # type: ignore[arg-type]
                dataset_catalog=object(),  # type: ignore[arg-type]
                processed_catalog=object(),  # type: ignore[arg-type]
                dataset_id="dataset",
                time_window=object(),  # type: ignore[arg-type]
            )
        elif root_name == "model_fit":
            fitting.fit_model_result_from_frames(
                (),
                object(),  # type: ignore[arg-type]
                estimator=SimpleNamespace(
                    fit_model_result_from_frames=lambda *_: work.append("fit")
                ),
            )
        elif root_name == "validation_summary":
            monkeypatch.setattr(
                validation_summary,
                "_build_validation_summary",
                lambda **_: work.append("summary"),
            )
            validation_summary.build_validation_summary(
                run_id="run",
                dataset_roots=(),
                layer_summaries=(),
                validated_output_members=(),
                output_bundle_expectations=(),
            )
        else:
            monkeypatch.setattr(
                schema_inventory,
                "require_unaliased_path",
                lambda *_args, **_kwargs: work.append("path"),
            )
            with schema_inventory.open_schema_inventory_payload(tmp_path):
                pass

    with runtime_resource_policy_scope(_policy(8)):
        with pytest.raises(
            RuntimeError,
            match="^runtime_resource_policy_competing_root$",
        ):
            invoke()

    assert work == []


def _prediction_result_for_scope_test(
    monkeypatch: pytest.MonkeyPatch,
    observed: list[RuntimeResourcePolicy | None],
    *,
    resource_policy: RuntimeResourcePolicy | None = None,
    retained_model_bytes: int = 0,
):  # type: ignore[no-untyped-def]
    from quant_research_platform.research.predictions import outputs

    monkeypatch.setattr(
        outputs,
        "algorithm_facts",
        lambda _algorithm: SimpleNamespace(dependency_package=None),
    )
    monkeypatch.setattr(
        outputs,
        "_resolved_model_hyperparameters_match",
        lambda *_args: True,
    )

    def output_frames(*_args: object, **_kwargs: object):  # type: ignore[no-untyped-def]
        observed.append(current_runtime_resource_policy())
        yield object()

    monkeypatch.setattr(outputs, "_iter_prediction_output_frames", output_frames)
    spec = SimpleNamespace(
        primary_key=("timestamp", "asset_id"),
        prediction_column="prediction",
        train_end="2024-01-01",
        test_start="2024-01-02",
        test_end=None,
        sample_filter_mode="unfiltered",
    )
    return outputs.build_prediction_result_from_frames(
        (),
        spec,  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        ("feature",),
        model_metadata={
            "model_algorithm": "linear_regression",
            "hyperparameters": {},
            "resolved_hyperparameters": {},
            "dependency_status": {"package": "internal"},
        },
        retained_model_bytes=retained_model_bytes,
        resource_policy=resource_policy,
    )


def test_lazy_prediction_root_rejects_unrelated_consumer_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[RuntimeResourcePolicy | None] = []
    result = _prediction_result_for_scope_test(monkeypatch, observed)

    with runtime_resource_policy_scope(_policy(8)):
        with pytest.raises(
            RuntimeError,
            match="^runtime_resource_policy_competing_root$",
        ):
            next(iter(result.output_chunks))

    assert observed == []


def test_lazy_prediction_root_keeps_explicit_policy_and_abandonment_is_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[RuntimeResourcePolicy | None] = []
    captures: list[bool] = []
    policy = _policy(8)
    monkeypatch.setattr(
        RuntimeResourcePolicy,
        "capture",
        classmethod(lambda _cls, **_kwargs: captures.append(True)),
    )
    abandoned = _prediction_result_for_scope_test(monkeypatch, observed)
    abandoned.close()
    assert captures == []

    explicit = _prediction_result_for_scope_test(
        monkeypatch,
        observed,
        retained_model_bytes=1,
        resource_policy=policy,
    )
    assert len(list(explicit.output_chunks)) == 1
    assert observed == [policy]


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

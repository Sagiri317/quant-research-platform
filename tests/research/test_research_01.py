from __future__ import annotations

import json
import weakref
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import quant_research_platform.research._memory as research_memory
import quant_research_platform.research.datasets.assembler as assembler_module
import quant_research_platform.research.datasets.io as research_dataset_io
import quant_research_platform.research.evaluations.runner as evaluation_runner_module
import quant_research_platform.research.models.fitting as model_fitting
import quant_research_platform.research.predictions.runner as prediction_runner_module
import quant_research_platform.storage._dataset_reading as dataset_reading
from quant_research_platform.artifacts._prediction_outputs import (
    write_prediction_outputs,
)
from quant_research_platform.artifacts.outputs import write_evaluation_outputs
from quant_research_platform.catalog.datasets import ResearchDatasetCatalog
from quant_research_platform.catalog.evaluations import EvaluationCatalog
from quant_research_platform.catalog.fields import platform_entity_time_key
from quant_research_platform.catalog.models import ModelCatalog
from quant_research_platform.catalog.predictions import PredictionCatalog
from quant_research_platform.catalog.processed import ProcessedDataCatalog
from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.data import (
    DataFrequency,
    DataGrain,
    ProcessedDataKind,
    ProcessedDataSpec,
)
from quant_research_platform.contracts.datasets import ResearchDatasetSpec
from quant_research_platform.contracts.evaluations import EvaluationSpec
from quant_research_platform.contracts.features import FeatureTransformSpec
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.models import ModelSpec
from quant_research_platform.contracts.outputs import OutputScope, OutputSpec
from quant_research_platform.contracts.predictions import (
    PREDICTION_PRIMARY_KEY,
    PREDICTION_SCORE_COLUMN,
    PredictionSpec,
)
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.research.datasets.assembler import (
    assemble_research_dataset,
)
from quant_research_platform.research.datasets.io import iter_research_dataset_chunks
from quant_research_platform.research.datasets.validation import (
    validate_evaluation_input,
    validate_prediction_input,
)
from quant_research_platform.research.evaluations.engine import (
    evaluate_signal_frame_chunks,
)
from quant_research_platform.research.evaluations.runner import (
    build_evaluation_result_from_research_dataset,
)
from quant_research_platform.research.models.supervised import (
    build_estimator_from_model_spec,
)
from quant_research_platform.research.predictions.runner import (
    build_prediction_result_from_research_dataset,
)
from quant_research_platform.runtime.resources import (
    RuntimeResourcePolicy,
    _OwnedValue,
    current_runtime_resource_policy,
    runtime_resource_policy_scope,
)
from quant_research_platform.storage import atomic as storage_atomic
from quant_research_platform.storage.datasets import inspect_dataset, write_dataset
from quant_research_platform.storage.layout import StorageLayout
from tests.helpers.research_processed_inputs import (
    make_demo_processed_catalog,
    write_demo_processed_inputs,
)
from tests.helpers.research_specs import (
    make_research_dataset_catalog,
    make_signal_spec,
)
from tests.helpers.runtime_resources import (
    capture_storage_test_policy,
    capture_test_policy,
)
from tests.research.prediction_output_helpers import (
    write_prediction_output_and_read,
)


def test_research_indivisible_result_is_exact_and_pre_admitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = pd.DataFrame({"timestamp": [pd.Timestamp("2026-01-01")], "value": [1.0]})
    second = pd.DataFrame({"timestamp": [pd.Timestamp("2026-01-02")], "value": [2.0]})
    policy = capture_test_policy(available_memory_bytes=16 * 1024 * 1024)

    with runtime_resource_policy_scope(policy):
        with research_memory._ResearchFrameAccumulator(
            "research.test.exact",
            resource_policy=policy,
        ) as accumulator:
            accumulator.append(first)
            accumulator.append(second)
            result = accumulator.materialize()
    pd.testing.assert_frame_equal(
        result,
        pd.concat([first, second], ignore_index=True),
    )

    policy = capture_test_policy(available_memory_bytes=16 * 1024 * 1024)
    entered: list[bool] = []

    def reject_concat(*_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
        entered.append(True)
        raise AssertionError("concat entered before full-overlap admission")

    monkeypatch.setattr(research_memory.pd, "concat", reject_concat)
    with runtime_resource_policy_scope(policy):
        with research_memory._ResearchFrameAccumulator(
            "research.test.reject",
            resource_policy=policy,
        ) as accumulator:
            accumulator.append(first)
            with policy.reserve("outer", policy.available_memory_bytes):
                with pytest.raises(
                    MemoryError,
                    match="runtime_memory_budget_exceeded",
                ):
                    accumulator.materialize()
    assert entered == []


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


def _output_spec(
    *,
    output_definition_id: str,
    run_id: str,
    run_type: str,
    market_id: MarketId | None = None,
) -> OutputSpec:
    return OutputSpec(
        output_id=run_id,
        output_definition_id=output_definition_id,
        run_id="test_run",
        market_id=market_id or MarketId("a_share"),
        run_type=run_type,
        scope=OutputScope.LOCAL_ONLY,
    )


def _write_prediction_result(layout: StorageLayout, result) -> Path:
    return write_prediction_outputs(
        spec=result.spec,
        output_spec=_output_spec(
            output_definition_id="prediction_outputs",
            run_id=result.spec.prediction_id,
            run_type=result.spec.run_type,
        ),
        layout=layout,
        output_chunks=result.output_chunks,
        model_metadata=result.model_metadata,
        prediction_metadata_factory=result.prediction_metadata_factory,
    )


def _write_evaluation_result(layout: StorageLayout, result) -> Path:
    return write_evaluation_outputs(
        result.metrics,
        result.grouped_label_values,
        ic_summary=result.ic_summary,
        group_spread_summary=result.group_spread_summary,
        evaluation_diagnostics=result.diagnostics,
        spec=result.spec,
        output_spec=_output_spec(
            output_definition_id="evaluation_outputs",
            run_id=result.spec.run_id,
            market_id=result.spec.market_id,
            run_type=result.spec.run_type,
        ),
        layout=layout,
    )


def _research_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "market_id": ["a_share", "a_share", "a_share", "a_share"],
            "asset_id": ["000001", "000002", "000001", "000002"],
            "timestamp": pd.to_datetime(
                ["2014-01-02", "2014-01-02", "2014-01-03", "2014-01-03"]
            ),
            "mom_20d": [1.0, 2.0, 2.0, 1.0],
            "forward_return_20d": [0.01, 0.03, 0.02, 0.00],
            "eligible": [True, True, True, True],
            "can_enter": [True, True, True, True],
            "can_exit": [True, True, True, True],
            "market_type": [4, 4, 4, 4],
        }
    )


def _research_rows_for_dates(dates: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for date_index, timestamp in enumerate(pd.to_datetime(dates), start=1):
        for asset_index, asset_id in enumerate(("000001", "000002"), start=1):
            rows.append(
                {
                    "market_id": "a_share",
                    "asset_id": asset_id,
                    "timestamp": timestamp,
                    "mom_20d": float(date_index + asset_index),
                    "forward_return_20d": 0.01 * (date_index + asset_index),
                    "eligible": True,
                    "can_enter": True,
                    "can_exit": True,
                    "market_type": 4,
                }
            )
    return pd.DataFrame(rows)


def test_research_dataset_assembles_from_processed_inputs(
    tmp_path: Path,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    processed_catalog = make_demo_processed_catalog()
    dataset_catalog = make_research_dataset_catalog(
        filter_column_roles={"eligible": ColumnRole.UNIVERSE},
    )
    write_demo_processed_inputs(layout, processed_catalog)
    root = assemble_research_dataset(
        layout=layout,
        dataset_catalog=dataset_catalog,
        processed_catalog=processed_catalog,
        dataset_id="demo",
        time_window=TimeWindow("2014-01-02", "2014-01-03"),
    )
    result = pd.concat(
        iter_research_dataset_chunks(
            layout,
            market_id=MarketId("a_share"),
            dataset_id="demo",
            time_window=TimeWindow("2014-01-02", "2014-01-03"),
        ),
        ignore_index=True,
    )
    inspection = inspect_dataset(root)

    assert root == tmp_path / "data" / "research" / "a_share" / "demo"
    assert list(result.columns) == [
        "market_id",
        "asset_id",
        "timestamp",
        "mom_20d",
        "forward_return_20d",
        "eligible",
        "can_enter",
        "can_exit",
        "market_type",
    ]
    assert len(result) == 4
    assert set(inspection.manifest["input_dataset_identities"]) == {
        "demo_factors",
        "demo_labels",
        "demo_annotations",
    }
    validate_evaluation_input(
        result,
        dataset_catalog.get("demo"),
        signal_columns=("mom_20d",),
        label_columns=("forward_return_20d",),
    )
    validate_prediction_input(
        result,
        dataset_catalog.get("demo"),
        input_columns=("mom_20d",),
    )


def test_research_join_bound_admission_precedes_join_pull(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=64 * 1024,
        explicit_cap_bytes=32 * 1024,
    )
    time_key_pulls: list[bool] = []
    pulled: list[bool] = []
    monkeypatch.setattr(
        assembler_module,
        "_research_session_view",
        lambda current, **_kwargs: current,
    )
    def fake_iter_dataset_chunks(*_args, **_kwargs):  # noqa: ANN002, ANN003
        time_key_pulls.append(True)
        yield pd.DataFrame({"timestamp": [pd.Timestamp("2026-01-01")]})

    monkeypatch.setattr(
        assembler_module,
        "iter_dataset_chunks",
        fake_iter_dataset_chunks,
    )
    monkeypatch.setattr(
        assembler_module,
        "_iter_research_chunk_windows_from_base_input",
        lambda *_args, **_kwargs: iter((object(),)),
    )
    monkeypatch.setattr(
        assembler_module,
        "_join_processed_inputs",
        lambda **_kwargs: (pulled.append(True), pd.DataFrame({"id": [1]}))[1],
    )
    dataset_spec = SimpleNamespace(
        primary_key=("id",),
        feature_transforms=(),
        input_data_ids=("base",),
        base_input_data_id="base",
    )
    sessions = {
        "base": SimpleNamespace(
            retained_memory_bytes=0,
            path_selection=None,
            close=lambda: None,
        )
    }
    chunks = assembler_module._iter_joined_processed_chunks(
        dataset_spec=dataset_spec,
        processed_catalog=SimpleNamespace(get=lambda _key: object()),
        sessions=sessions,
        time_window=object(),
    )

    with runtime_resource_policy_scope(policy):
        blocker = policy.lease("test.research.blocker", policy.memory_budget_bytes)
        try:
            bound = next(chunks)
            with pytest.raises(MemoryError, match="runtime_memory_budget_exceeded"):
                policy.lease("test.research.consumer", bound)
            assert time_key_pulls == []
            assert pulled == []
        finally:
            blocker.close()


def test_research_time_window_driver_uses_one_window_per_physical_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_targets: list[int] = []
    timestamps = pd.date_range("2026-01-01", periods=20, freq="min")
    policy = capture_test_policy(
        available_memory_bytes=128 * 1024,
        explicit_cap_bytes=64 * 1024 + 768,
    )
    session = SimpleNamespace(path_selection=None, close=lambda: None)
    monkeypatch.setattr(
        assembler_module,
        "_research_session_view",
        lambda current, **_kwargs: current,
    )

    def fake_iter_dataset_chunks(context, *, target_chunk_bytes):  # noqa: ANN001
        observed_targets.append(target_chunk_bytes)
        yield pd.DataFrame({"timestamp": timestamps})

    monkeypatch.setattr(
        assembler_module,
        "iter_dataset_chunks",
        fake_iter_dataset_chunks,
        raising=False,
    )

    with runtime_resource_policy_scope(policy):
        result = list(
            assembler_module._iter_research_chunk_windows_from_base_input(
            session,
            time_window=TimeWindow(
                "2026-01-01T00:00:00",
                "2026-01-01T00:19:00",
            ),
            )
        )

    assert result == [
        TimeWindow(
            "2026-01-01T00:00:00",
            "2026-01-01T00:19:00",
        )
    ]
    assert observed_targets == [policy.available_memory_bytes]


def test_research_time_window_driver_preserves_single_timestamp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=128 * 1024,
        explicit_cap_bytes=64 * 1024 + 240,
    )
    session = SimpleNamespace(path_selection=None, close=lambda: None)
    monkeypatch.setattr(
        assembler_module,
        "_research_session_view",
        lambda current, **_kwargs: current,
    )
    monkeypatch.setattr(
        assembler_module,
        "iter_dataset_chunks",
        lambda *_args, **_kwargs: (
            frame
            for frame in (pd.DataFrame({"timestamp": [pd.Timestamp("2026-01-01")]}),)
        ),
    )

    with runtime_resource_policy_scope(policy):
        assert list(
            assembler_module._iter_research_chunk_windows_from_base_input(
                session,
                time_window=TimeWindow(
                    "2026-01-01T00:00:00",
                    "2026-01-01T00:00:00",
                ),
            )
        ) == [TimeWindow("2026-01-01T00:00:00", "2026-01-01T00:00:00")]


def test_research_time_window_driver_normalizes_and_deduplicates_dates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(available_memory_bytes=128 * 1024)
    session = SimpleNamespace(path_selection=None, close=lambda: None)
    monkeypatch.setattr(
        assembler_module,
        "_research_session_view",
        lambda current, **_kwargs: current,
    )
    monkeypatch.setattr(
        assembler_module,
        "iter_dataset_chunks",
        lambda *_args, **_kwargs: iter(
            (
                pd.DataFrame(
                    {
                        "timestamp": [
                            pd.Timestamp("2026-01-02 15:00"),
                            pd.Timestamp("2026-01-01 09:00"),
                        ]
                    }
                ),
                pd.DataFrame(
                    {
                        "timestamp": [
                            pd.Timestamp("2026-01-02 09:00"),
                            pd.Timestamp("2026-01-03 09:00"),
                        ]
                    }
                ),
            )
        ),
    )

    with runtime_resource_policy_scope(policy):
        result = list(
            assembler_module._iter_research_chunk_windows_from_base_input(
                session,
                time_window=TimeWindow("2026-01-01", "2026-01-02"),
            )
        )

    assert result == [
        TimeWindow("2026-01-01", "2026-01-02"),
        TimeWindow("2026-01-03", "2026-01-03"),
    ]


def test_research_time_window_driver_rejects_empty_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(available_memory_bytes=128 * 1024)
    session = SimpleNamespace(path_selection=None, close=lambda: None)
    monkeypatch.setattr(
        assembler_module,
        "_research_session_view",
        lambda current, **_kwargs: current,
    )
    monkeypatch.setattr(
        assembler_module,
        "iter_dataset_chunks",
        lambda *_args, **_kwargs: iter(
            (pd.DataFrame({"timestamp": pd.Series(dtype="datetime64[ns]")}),)
        ),
    )

    with runtime_resource_policy_scope(policy):
        with pytest.raises(ValueError, match="no_research_dataset_sessions"):
            list(
                assembler_module._iter_research_chunk_windows_from_base_input(
                    session,
                    time_window=TimeWindow("2026-01-01", "2026-01-01"),
                )
            )


def test_research_time_window_driver_rejects_non_monotonic_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(available_memory_bytes=128 * 1024)
    session = SimpleNamespace(path_selection=None, close=lambda: None)
    monkeypatch.setattr(
        assembler_module,
        "_research_session_view",
        lambda current, **_kwargs: current,
    )
    monkeypatch.setattr(
        assembler_module,
        "iter_dataset_chunks",
        lambda *_args, **_kwargs: iter(
            (
                pd.DataFrame({"timestamp": [pd.Timestamp("2026-01-02")] }),
                pd.DataFrame({"timestamp": [pd.Timestamp("2026-01-01")] }),
            )
        ),
    )

    with runtime_resource_policy_scope(policy):
        with pytest.raises(ValueError, match="non_monotonic_research_time_keys"):
            list(
                assembler_module._iter_research_chunk_windows_from_base_input(
                    session,
                    time_window=TimeWindow("2026-01-01", "2026-01-02"),
                )
            )


def test_research_time_window_driver_closes_iterator_before_view_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    policy = capture_test_policy(
        available_memory_bytes=128 * 1024,
        explicit_cap_bytes=64 * 1024 + 768,
    )

    class Inventory:
        def release_path_selection(self, selection: int) -> None:
            assert selection == 1
            events.append("selection_release")

    base_session = SimpleNamespace(
        inventory=Inventory(),
        path_selection=None,
    )
    view = SimpleNamespace(
        inventory=base_session.inventory,
        path_selection=1,
        close=lambda: events.append("view_close"),
    )
    monkeypatch.setattr(
        assembler_module,
        "_research_session_view",
        lambda *_args, **_kwargs: view,
    )

    def chunks():
        try:
            yield pd.DataFrame({"not_timestamp": [1]})
        finally:
            events.append("iterator_close")

    iterator = chunks()
    monkeypatch.setattr(
        assembler_module,
        "iter_dataset_chunks",
        lambda *_args, **_kwargs: iterator,
    )

    with runtime_resource_policy_scope(policy):
        with pytest.raises(KeyError, match="timestamp"):
            list(
                assembler_module._iter_research_chunk_windows_from_base_input(
                    base_session,
                    time_window=TimeWindow("2026-01-01", "2026-01-01"),
                )
            )
    assert events == ["iterator_close", "selection_release", "view_close"]


@pytest.mark.parametrize("site", ("time_windows", "materialize", "view_construction"))
@pytest.mark.parametrize(
    "primary_type",
    (RuntimeError, MemoryError, KeyboardInterrupt, SystemExit),
)
def test_research_view_cleanup_preserves_active_primary(
    monkeypatch: pytest.MonkeyPatch,
    site: str,
    primary_type: type[BaseException],
) -> None:
    primary = primary_type("body_primary")
    policy = capture_test_policy(
        available_memory_bytes=128 * 1024,
        explicit_cap_bytes=64 * 1024 + 768,
    )
    window = TimeWindow("2026-01-01", "2026-01-01")

    class Inventory:
        def new_path_selection(self) -> int:
            return 1

        def select_path(self, _selection: int, _path: Path) -> None:
            pass

        def release_path_selection(self, _selection: int) -> None:
            raise OSError("release_secondary")

    inventory = Inventory()
    base = SimpleNamespace(inventory=inventory, path_selection=None)
    view = SimpleNamespace(
        inventory=inventory,
        path_selection=1,
        close=lambda: (_ for _ in ()).throw(OSError("close_secondary")),
    )

    if site == "time_windows":

        def chunks():
            raise primary
            yield

        monkeypatch.setattr(
            assembler_module,
            "_research_session_view",
            lambda *_args, **_kwargs: view,
        )
        monkeypatch.setattr(
            assembler_module,
            "iter_dataset_chunks",
            lambda *_args, **_kwargs: chunks(),
        )

        def invoke() -> object:
            return list(
                assembler_module._iter_research_chunk_windows_from_base_input(
                    base,
                    time_window=window,
                )
            )
    elif site == "materialize":
        session = SimpleNamespace(
            inventory=inventory,
            path_selection=None,
            logical_columns=("value",),
        )
        monkeypatch.setattr(
            assembler_module,
            "_processed_input_columns",
            lambda *_args, **_kwargs: ("value",),
        )
        monkeypatch.setattr(
            assembler_module,
            "_research_session_view",
            lambda *_args, **_kwargs: view,
        )
        monkeypatch.setattr(
            assembler_module,
            "materialize_dataset_session_owned",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(primary),
        )

        def invoke() -> object:
            return assembler_module._read_processed_input(
                dataset_spec=SimpleNamespace(primary_key=(), required_columns=()),
                processed_spec=SimpleNamespace(required_columns=()),
                session=session,
                existing_columns=(),
                time_window=window,
            )
    else:
        session = SimpleNamespace(
            resource_policy=policy,
            timestamp_column="timestamp",
            spec={"partition_columns": ("year",)},
            inventory=inventory,
            data_root=Path("data"),
            read_paths=(Path("data/part.parquet"),),
            path_selection=None,
        )
        monkeypatch.setattr(
            assembler_module,
            "parquet_filters_for_time_window",
            lambda **_kwargs: None,
        )
        monkeypatch.setattr(
            assembler_module,
            "prune_paths_for_time_window",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(primary),
        )

        def invoke() -> object:
            return assembler_module._research_session_view(
                session,
                time_window=window,
            )

    with runtime_resource_policy_scope(policy):
        with pytest.raises(primary_type, match="body_primary") as caught:
            invoke()
    assert caught.value is primary


def test_research_view_cleanup_attempts_close_and_raises_first_failure() -> None:
    events: list[str] = []

    class Inventory:
        def release_path_selection(self, _selection: int) -> None:
            events.append("release")
            raise OSError("release_first")

    base = SimpleNamespace(inventory=Inventory(), path_selection=None)

    def close() -> None:
        events.append("close")
        raise RuntimeError("close_second")

    view = SimpleNamespace(path_selection=1, close=close)
    with pytest.raises(OSError, match="release_first"):
        assembler_module._close_research_view_preserving_primary(view, base)
    assert events == ["release", "close"]


def test_research_processed_input_uses_live_budget_for_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processed_catalog = make_demo_processed_catalog()
    dataset_spec = make_research_dataset_catalog().get("demo")
    processed_spec = processed_catalog.get("demo_factors")
    policy = capture_test_policy(
        available_memory_bytes=128 * 1024,
        explicit_cap_bytes=64 * 1024 + 9216,
    )
    observed: dict[str, object] = {}
    columns = assembler_module._processed_input_columns(
        dataset_spec,
        processed_spec,
        existing_columns=(),
    )
    context = SimpleNamespace(
        logical_columns=columns,
        path_selection=None,
        close=lambda: None,
    )
    monkeypatch.setattr(
        assembler_module,
        "_research_session_view",
        lambda current, **_kwargs: current,
    )

    def materialize(context, **kwargs):  # noqa: ANN001
        observed["context"] = context
        observed.update(kwargs)
        return _OwnedValue(
            pd.DataFrame(),
            policy.lease("test.research.input", 0),
        )

    monkeypatch.setattr(
        assembler_module,
        "materialize_dataset_session_owned",
        materialize,
    )

    with runtime_resource_policy_scope(policy):
        owned = assembler_module._read_processed_input(
            dataset_spec=dataset_spec,
            processed_spec=processed_spec,
            session=context,
            existing_columns=(),
            time_window=TimeWindow("2014-01-02", "2014-01-08"),
        )
        owned.lease.close()

    assert observed == {
        "context": context,
        "target_chunk_bytes": policy.available_memory_bytes,
    }


def test_research_join_preflights_merged_output_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=128 * 1024,
        explicit_cap_bytes=64 * 1024 + 750,
    )
    base = pd.DataFrame({"id": [1], "left": [1]})
    right = pd.DataFrame({"id": [1], "right": [1]})
    frames = iter((base, right))
    monkeypatch.setattr(
        assembler_module,
        "_ordered_input_data_ids",
        lambda *_args, **_kwargs: ("left", "right"),
    )
    monkeypatch.setattr(
        assembler_module,
        "_read_processed_input",
        lambda **_kwargs: _OwnedValue(
            next(frames),
            policy.lease("test.research.input", 0),
        ),
    )
    monkeypatch.setattr(
        assembler_module,
        "_frame_bytes",
        lambda frame: {
            ("id",): 100,
            ("id", "left"): 200,
            ("id", "right"): 151,
        }[tuple(frame.columns)],
    )

    with runtime_resource_policy_scope(policy):
        blocking_lease = policy.lease(
            "test.research.merge.capacity",
            policy.memory_budget_bytes - 700,
        )
        try:
            with pytest.raises(MemoryError, match="runtime_memory_budget_exceeded"):
                assembler_module._join_processed_inputs(
                    dataset_spec=SimpleNamespace(primary_key=("id",)),
                    processed_catalog=SimpleNamespace(get=lambda _key: object()),
                    sessions={
                        "left": SimpleNamespace(),
                        "right": SimpleNamespace(),
                    },
                    time_window=object(),
                )
        finally:
            blocking_lease.close()


def test_research_join_clears_transient_chunk_before_next_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(available_memory_bytes=16 * 1024 * 1024)
    holder = {
        "frame": pd.DataFrame({"id": [1], "value": [1.0]}),
    }
    frame_ref = weakref.ref(holder["frame"])
    monkeypatch.setattr(
        assembler_module,
        "_base_input_data_id",
        lambda *_args, **_kwargs: "one",
    )
    monkeypatch.setattr(
        assembler_module,
        "_iter_research_chunk_windows_from_base_input",
        lambda *_args, **_kwargs: iter((object(), object())),
    )
    monkeypatch.setattr(
        assembler_module,
        "_join_processed_inputs",
        lambda **_kwargs: holder["frame"],
    )
    monkeypatch.setattr(
        assembler_module,
        "validate_research_dataset",
        lambda *_args, **_kwargs: None,
    )

    chunks = assembler_module._iter_joined_processed_chunks(
        dataset_spec=SimpleNamespace(primary_key=("id",)),
        processed_catalog=SimpleNamespace(get=lambda _key: object()),
        sessions={"one": SimpleNamespace(retained_memory_bytes=0)},
        time_window=object(),
    )
    with runtime_resource_policy_scope(policy):
        bound = next(chunks)
        lease = policy.lease("test.research.consumer", bound)
        owned = chunks.send(lease)
        assert policy.used_memory_bytes == lease.reserved_bytes
        holder["frame"] = None
        del owned
        next_bound = chunks.send(None)
        assert type(next_bound) is int and next_bound > 0
        assert frame_ref() is None
        lease.close()
        chunks.close()


def test_research_dataset_assembles_intraday_timestamp_window(tmp_path: Path) -> None:
    layout = StorageLayout(project_root=tmp_path)
    primary_key = platform_entity_time_key()
    factor_spec = ProcessedDataSpec(
        data_id="minute_factors",
        market_id=MarketId("crypto"),
        kind=ProcessedDataKind.FACTOR,
        frequency=DataFrequency.MINUTE,
        grain=DataGrain.ASSET_TIMESTAMP,
        column_roles={
            "market_id": ColumnRole.KEY,
            "asset_id": ColumnRole.KEY,
            "timestamp": ColumnRole.TIMESTAMP,
            "minute_factor": ColumnRole.FACTOR,
        },
        primary_key=primary_key,
        partition_columns=("year",),
    )
    label_spec = ProcessedDataSpec(
        data_id="minute_labels",
        market_id=MarketId("crypto"),
        kind=ProcessedDataKind.LABEL,
        frequency=DataFrequency.MINUTE,
        grain=DataGrain.ASSET_TIMESTAMP,
        column_roles={
            "market_id": ColumnRole.KEY,
            "asset_id": ColumnRole.KEY,
            "timestamp": ColumnRole.TIMESTAMP,
            "minute_label": ColumnRole.LABEL,
        },
        primary_key=primary_key,
        partition_columns=("year",),
    )
    processed_catalog = ProcessedDataCatalog()
    processed_catalog.register(factor_spec)
    processed_catalog.register(label_spec)
    dataset_catalog = make_research_dataset_catalog(
        data_id="minute_research",
        market_id="crypto",
        input_data_ids=("minute_factors", "minute_labels"),
        base_input_data_id="minute_factors",
        factor_columns=("minute_factor",),
        label_columns=("minute_label",),
        filter_columns=(),
        context_columns=(),
        frequency=DataFrequency.MINUTE,
        grain=DataGrain.ASSET_TIMESTAMP,
    )
    timestamps = pd.to_datetime(
        [
            "2026-07-06T00:00:00",
            "2026-07-06T00:01:00",
            "2026-07-06T00:02:00",
        ]
    )
    keys = pd.DataFrame(
        [
            {
                "market_id": "crypto",
                "asset_id": asset_id,
                "timestamp": timestamp,
            }
            for timestamp in timestamps
            for asset_id in ("asset_a", "asset_b")
        ]
    )
    for spec, value_column in (
        (factor_spec, "minute_factor"),
        (label_spec, "minute_label"),
    ):
        frame = keys.assign(**{value_column: range(len(keys))})
        write_dataset(
            layout.processed_data_root(spec.market_id, spec.kind, spec.data_id),
            frame,
            spec=spec,
        )

    assemble_research_dataset(
        layout=layout,
        dataset_catalog=dataset_catalog,
        processed_catalog=processed_catalog,
        dataset_id="minute_research",
        time_window=TimeWindow(
            "2026-07-06T00:01:00",
            "2026-07-06T00:02:00",
        ),
    )
    result = pd.concat(
        iter_research_dataset_chunks(
            layout,
            market_id=MarketId("crypto"),
            dataset_id="minute_research",
            time_window=TimeWindow(
                "2026-07-06T00:01:00",
                "2026-07-06T00:02:00",
            ),
        ),
        ignore_index=True,
    )

    assert len(result) == 4
    assert sorted(result["timestamp"].unique()) == list(timestamps[1:])


def test_research_dataset_assembler_generates_feature_transforms_from_factors(
    tmp_path: Path,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    processed_catalog = make_demo_processed_catalog()
    dataset_catalog = ResearchDatasetCatalog()
    dataset_catalog.register(
        ResearchDatasetSpec(
            data_id="demo",
            market_id=MarketId("a_share"),
            input_data_ids=("demo_factors", "demo_labels", "demo_annotations"),
            base_input_data_id="demo_annotations",
            factor_columns=("mom_20d",),
            feature_columns=("mom_20d_zscore",),
            feature_transforms=(
                FeatureTransformSpec(
                    transform_id="mom_20d_zscore",
                    source_columns=("mom_20d",),
                    output_columns=("mom_20d_zscore",),
                    transform="cross_sectional_zscore",
                ),
            ),
            label_columns=("forward_return_20d",),
            filter_columns=("eligible", "can_enter", "can_exit"),
            context_columns=("market_type",),
            frequency=DataFrequency.DAILY,
            grain=DataGrain.RESEARCH_ROW,
            column_roles={
                "market_id": ColumnRole.KEY,
                "asset_id": ColumnRole.KEY,
                "timestamp": ColumnRole.TIMESTAMP,
                "mom_20d": ColumnRole.FACTOR,
                "mom_20d_zscore": ColumnRole.FEATURE,
                "forward_return_20d": ColumnRole.LABEL,
                "eligible": ColumnRole.UNIVERSE,
                "can_enter": ColumnRole.MASK,
                "can_exit": ColumnRole.MASK,
                "market_type": ColumnRole.CONTEXT,
            },
            primary_key=platform_entity_time_key(),
            partition_columns=("year",),
        )
    )
    write_demo_processed_inputs(layout, processed_catalog)

    assemble_research_dataset(
        layout=layout,
        dataset_catalog=dataset_catalog,
        processed_catalog=processed_catalog,
        dataset_id="demo",
        time_window=TimeWindow("2014-01-02", "2014-01-03"),
    )
    result = pd.concat(
        iter_research_dataset_chunks(
            layout,
            market_id=MarketId("a_share"),
            dataset_id="demo",
            time_window=TimeWindow("2014-01-02", "2014-01-03"),
        ),
        ignore_index=True,
    ).sort_values(["timestamp", "asset_id"])

    assert list(result.columns) == [
        "market_id",
        "asset_id",
        "timestamp",
        "mom_20d",
        "mom_20d_zscore",
        "forward_return_20d",
        "eligible",
        "can_enter",
        "can_exit",
        "market_type",
    ]
    assert result["mom_20d_zscore"].tolist() == pytest.approx([-1.0, 1.0, 1.0, -1.0])
    validate_prediction_input(
        result,
        dataset_catalog.get("demo"),
        input_columns=("mom_20d", "mom_20d_zscore"),
    )


def test_research_dataset_session_complete_chunks_preserve_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _research_rows_for_dates(
        [
            "2014-01-02",
            "2014-01-03",
            "2014-01-06",
            "2014-01-07",
            "2014-01-08",
        ]
    )
    default_layout = StorageLayout(project_root=tmp_path / "default")
    tiny_layout = StorageLayout(project_root=tmp_path / "tiny")
    processed_catalog = make_demo_processed_catalog()
    dataset_catalog = make_research_dataset_catalog(
        filter_column_roles={"eligible": ColumnRole.UNIVERSE},
    )
    write_demo_processed_inputs(default_layout, processed_catalog, frame=frame)
    write_demo_processed_inputs(tiny_layout, processed_catalog, frame=frame)

    default_root = assemble_research_dataset(
        layout=default_layout,
        dataset_catalog=dataset_catalog,
        processed_catalog=processed_catalog,
        dataset_id="demo",
        time_window=TimeWindow("2014-01-02", "2014-01-08"),
    )
    tiny_policy = capture_storage_test_policy(
        working_headroom_bytes=4 * 1024**2,
    )
    tiny_root = assemble_research_dataset(
        layout=tiny_layout,
        dataset_catalog=dataset_catalog,
        processed_catalog=processed_catalog,
        dataset_id="demo",
        time_window=TimeWindow("2014-01-02", "2014-01-08"),
        resource_policy=tiny_policy,
    )

    default = pd.concat(
        iter_research_dataset_chunks(
            default_layout,
            market_id=MarketId("a_share"),
            dataset_id="demo",
            time_window=TimeWindow("2014-01-02", "2014-01-08"),
        ),
        ignore_index=True,
    ).sort_values(["market_id", "asset_id", "timestamp"])
    tiny = pd.concat(
        iter_research_dataset_chunks(
            tiny_layout,
            market_id=MarketId("a_share"),
            dataset_id="demo",
            time_window=TimeWindow("2014-01-02", "2014-01-08"),
        ),
        ignore_index=True,
    ).sort_values(["market_id", "asset_id", "timestamp"])

    pd.testing.assert_frame_equal(
        default.reset_index(drop=True),
        tiny.reset_index(drop=True),
    )
    assert (
        inspect_dataset(tiny_root).file_count
        == inspect_dataset(default_root).file_count
    )


def test_research_chunk_view_prunes_top_level_multi_year_physical_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    processed_catalog = make_demo_processed_catalog()
    frame = _research_rows_for_dates(["2014-12-31", "2015-01-02"])
    write_demo_processed_inputs(layout, processed_catalog, frame=frame)
    processed_spec = processed_catalog.get("demo_factors")
    dataset_spec = make_research_dataset_catalog(
        filter_column_roles={"eligible": ColumnRole.UNIVERSE},
    ).get("demo")
    columns = assembler_module._processed_input_columns(
        dataset_spec,
        processed_spec,
        existing_columns=(),
    )
    policy = capture_test_policy(
        available_memory_bytes=1024**3,
    )
    with assembler_module.open_dataset_session(
        layout.processed_data_root(
            processed_spec.market_id,
            processed_spec.kind,
            processed_spec.data_id,
        ),
        columns=columns,
        time_window=TimeWindow("2014-12-31", "2015-01-02"),
        resource_policy=policy,
    ) as session:
        with runtime_resource_policy_scope(policy):
            view = assembler_module._research_session_view(
                session,
                time_window=TimeWindow("2015-01-02", "2015-01-02"),
            )

        assert len(session.read_paths) == 2
        assert len(view.read_paths) == 1
        assert view.read_paths[0].parent.name == "year=2015"
        assert view.path_selection is not None
        session.inventory.release_path_selection(view.path_selection)

        observed_selections: list[int] = []

        def materialize(current, **_kwargs):  # noqa: ANN001
            assert len(current.read_paths) == 1
            assert current.path_selection is not None
            observed_selections.append(current.path_selection)
            return _OwnedValue(
                pd.DataFrame(),
                policy.lease("test.research.input", 0),
            )

        monkeypatch.setattr(
            assembler_module,
            "materialize_dataset_session_owned",
            materialize,
        )
        with runtime_resource_policy_scope(policy):
            owned = assembler_module._read_processed_input(
                dataset_spec=dataset_spec,
                processed_spec=processed_spec,
                session=session,
                existing_columns=(),
                time_window=TimeWindow("2014-12-31", "2014-12-31"),
            )
            owned.lease.close()
        normal_selection = observed_selections.pop()
        assert (
            session.inventory.path_count(
                selected=True,
                path_selection=normal_selection,
            )
            == 0
        )

        def fail_materialization(current, **_kwargs):  # noqa: ANN001
            assert current.path_selection is not None
            observed_selections.append(current.path_selection)
            raise RuntimeError("research_test_materialization_failure")

        monkeypatch.setattr(
            assembler_module,
            "materialize_dataset_session_owned",
            fail_materialization,
        )
        with runtime_resource_policy_scope(policy):
            with pytest.raises(
                RuntimeError,
                match="research_test_materialization_failure",
            ):
                assembler_module._read_processed_input(
                    dataset_spec=dataset_spec,
                    processed_spec=processed_spec,
                    session=session,
                    existing_columns=(),
                    time_window=TimeWindow("2015-01-02", "2015-01-02"),
                )
        error_selection = observed_selections.pop()
        assert (
            session.inventory.path_count(
                selected=True,
                path_selection=error_selection,
            )
            == 0
        )


def test_research_assembly_rejects_dependency_replacement_after_identity_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = StorageLayout(project_root=tmp_path / "current")
    replacement_layout = StorageLayout(project_root=tmp_path / "replacement")
    processed_catalog = make_demo_processed_catalog()
    dataset_catalog = make_research_dataset_catalog(
        filter_column_roles={"eligible": ColumnRole.UNIVERSE},
    )
    write_demo_processed_inputs(layout, processed_catalog)
    write_demo_processed_inputs(
        replacement_layout,
        processed_catalog,
        frame=_research_rows_for_dates(["2014-01-02", "2014-01-03", "2014-01-04"]),
    )
    factor_spec = processed_catalog.get("demo_factors")
    current_root = layout.processed_data_root(
        factor_spec.market_id,
        factor_spec.kind,
        factor_spec.data_id,
    )
    replacement_root = replacement_layout.processed_data_root(
        factor_spec.market_id,
        factor_spec.kind,
        factor_spec.data_id,
    )
    backup_root = current_root.with_name(current_root.name + "-before-replacement")
    real_identity = assembler_module.dataset_identity_payload
    real_open = assembler_module.open_dataset_session
    opened_roots: list[Path] = []
    replaced = False

    def record_open(root: Path, **kwargs):
        opened_roots.append(Path(root))
        return real_open(root, **kwargs)

    def replace_after_identity(spec, manifest):  # noqa: ANN001
        nonlocal replaced
        identity = real_identity(spec, manifest)
        if spec["data_id"] == factor_spec.data_id and not replaced:
            current_root.rename(backup_root)
            replacement_root.rename(current_root)
            replaced = True
        return identity

    monkeypatch.setattr(assembler_module, "open_dataset_session", record_open)
    monkeypatch.setattr(
        assembler_module,
        "dataset_identity_payload",
        replace_after_identity,
    )

    with pytest.raises(
        ValueError,
        match="dataset_(spec|manifest)_identity_changed",
    ):
        assemble_research_dataset(
            layout=layout,
            dataset_catalog=dataset_catalog,
            processed_catalog=processed_catalog,
            dataset_id="demo",
            time_window=TimeWindow("2014-01-02", "2014-01-03"),
        )

    assert replaced
    assert len(opened_roots) == len(processed_catalog.list())
    assert len(opened_roots) == len(set(opened_roots))

    reopened_policy = capture_test_policy(available_memory_bytes=16 * 1024 * 1024)
    with runtime_resource_policy_scope(reopened_policy):
        assert current_runtime_resource_policy() is reopened_policy


def test_research_assembly_verifies_every_input_inside_staged_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    processed_catalog = make_demo_processed_catalog()
    dataset_catalog = make_research_dataset_catalog(
        filter_column_roles={"eligible": ColumnRole.UNIVERSE},
    )
    write_demo_processed_inputs(layout, processed_catalog)
    real_open = assembler_module.open_dataset_session
    real_write = assembler_module.write_dataset
    real_verify = assembler_module.DatasetSession.verify
    opened_roots: list[Path] = []
    opened_sessions: list[assembler_module.DatasetSession] = []
    staged_verified_roots: list[Path] = []
    post_publish_verified_roots: list[Path] = []
    inside_staged_validator = False
    writer_returned = False

    @contextmanager
    def record_open(root: Path, **kwargs):
        opened_roots.append(Path(root))
        with real_open(root, **kwargs) as session:
            opened_sessions.append(session)
            yield session

    def record_verify(session) -> None:  # noqa: ANN001
        if inside_staged_validator:
            staged_verified_roots.append(session.root)
        elif writer_returned and session.root in opened_roots:
            post_publish_verified_roots.append(session.root)
        real_verify(session)

    def record_write(root, data, *, staged_validator, **kwargs):  # noqa: ANN001
        nonlocal inside_staged_validator, writer_returned
        assert staged_validator is not None
        assert "retained_payload_bytes" not in kwargs

        def record_staged(receipt) -> None:  # noqa: ANN001
            nonlocal inside_staged_validator
            inside_staged_validator = True
            try:
                staged_validator(receipt)
            finally:
                inside_staged_validator = False

        result = real_write(
            root,
            data,
            staged_validator=record_staged,
            **kwargs,
        )
        writer_returned = True
        return result

    monkeypatch.setattr(assembler_module, "open_dataset_session", record_open)
    monkeypatch.setattr(assembler_module.DatasetSession, "verify", record_verify)
    monkeypatch.setattr(assembler_module, "write_dataset", record_write)

    artifact_root = assemble_research_dataset(
        layout=layout,
        dataset_catalog=dataset_catalog,
        processed_catalog=processed_catalog,
        dataset_id="demo",
        time_window=TimeWindow("2014-01-02", "2014-01-03"),
    )

    assert artifact_root.is_dir()
    assert len(opened_roots) == len(processed_catalog.list())
    assert len(opened_roots) == len(set(opened_roots))
    assert staged_verified_roots == opened_roots
    assert post_publish_verified_roots == []


def test_research_assembly_rejects_replacement_before_publish_and_cleans_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = StorageLayout(project_root=tmp_path / "current")
    replacement_layout = StorageLayout(project_root=tmp_path / "replacement")
    processed_catalog = make_demo_processed_catalog()
    dataset_catalog = make_research_dataset_catalog(
        filter_column_roles={"eligible": ColumnRole.UNIVERSE},
    )
    write_demo_processed_inputs(layout, processed_catalog)
    write_demo_processed_inputs(
        replacement_layout,
        processed_catalog,
        frame=_research_rows_for_dates(["2014-01-02", "2014-01-03", "2014-01-04"]),
    )
    factor_spec = processed_catalog.get("demo_factors")
    current_root = layout.processed_data_root(
        factor_spec.market_id,
        factor_spec.kind,
        factor_spec.data_id,
    )
    replacement_root = replacement_layout.processed_data_root(
        factor_spec.market_id,
        factor_spec.kind,
        factor_spec.data_id,
    )
    backup_root = current_root.with_name(current_root.name + "-before-replacement")
    artifact_root = layout.research_dataset_root(MarketId("a_share"), "demo")
    real_open = assembler_module.open_dataset_session
    real_joined_chunks = assembler_module._iter_joined_processed_chunks
    real_write = assembler_module.write_dataset
    opened_roots: list[Path] = []
    replaced = False
    staged_validator_supplied = False

    def record_open(root: Path, **kwargs):
        opened_roots.append(Path(root))
        return real_open(root, **kwargs)

    def replace_after_last_input(**kwargs):  # noqa: ANN003, ANN202
        nonlocal replaced
        yield from real_joined_chunks(**kwargs)
        current_root.rename(backup_root)
        replacement_root.rename(current_root)
        replaced = True

    def record_write(root, data, *, staged_validator, **kwargs):  # noqa: ANN001
        nonlocal staged_validator_supplied
        staged_validator_supplied = staged_validator is not None
        return real_write(
            root,
            data,
            staged_validator=staged_validator,
            **kwargs,
        )

    monkeypatch.setattr(assembler_module, "open_dataset_session", record_open)
    monkeypatch.setattr(
        assembler_module,
        "_iter_joined_processed_chunks",
        replace_after_last_input,
    )
    monkeypatch.setattr(assembler_module, "write_dataset", record_write)

    with pytest.raises(
        ValueError,
        match="dataset_(spec|manifest)_identity_changed",
    ):
        assemble_research_dataset(
            layout=layout,
            dataset_catalog=dataset_catalog,
            processed_catalog=processed_catalog,
            dataset_id="demo",
            time_window=TimeWindow("2014-01-02", "2014-01-03"),
        )

    assert replaced
    assert staged_validator_supplied
    assert len(opened_roots) == len(processed_catalog.list())
    assert len(opened_roots) == len(set(opened_roots))
    assert not artifact_root.exists()
    scratch_prefix = storage_atomic._atomic_directory_scratch_prefix(artifact_root)
    assert list(artifact_root.parent.glob(f"{scratch_prefix}*")) == []


def test_sparse_research_dataset_left_joins_missing_non_base_input_key(
    tmp_path: Path,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    processed_catalog = make_demo_processed_catalog()
    dataset_catalog = make_research_dataset_catalog(
        key_coverage={"demo_labels": "sparse"},
        filter_column_roles={"eligible": ColumnRole.UNIVERSE},
    )
    frame = _research_rows()
    labels = frame.drop(index=frame.index[-1]).reset_index(drop=True)
    write_demo_processed_inputs(layout, processed_catalog, label_frame=labels)

    assemble_research_dataset(
        layout=layout,
        dataset_catalog=dataset_catalog,
        processed_catalog=processed_catalog,
        dataset_id="demo",
        time_window=TimeWindow("2014-01-02", "2014-01-03"),
    )
    result = pd.concat(
        iter_research_dataset_chunks(
            layout,
            market_id=MarketId("a_share"),
            dataset_id="demo",
            time_window=TimeWindow("2014-01-02", "2014-01-03"),
        ),
        ignore_index=True,
    )
    missing_row = result.loc[
        result["asset_id"].eq("000002")
        & result["timestamp"].eq(pd.Timestamp("2014-01-03"))
    ].iloc[0]

    assert len(result) == 4
    assert pd.isna(missing_row["forward_return_20d"])


def test_research_dataset_role_validation_rejects_wrong_factor_source(
    tmp_path: Path,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    processed_catalog = make_demo_processed_catalog()
    dataset_catalog = ResearchDatasetCatalog()
    dataset_catalog.register(
        ResearchDatasetSpec(
            data_id="bad_demo",
            market_id=MarketId("a_share"),
            input_data_ids=("demo_labels", "demo_annotations"),
            base_input_data_id="demo_annotations",
            factor_columns=("market_type",),
            label_columns=("forward_return_20d",),
            frequency=DataFrequency.DAILY,
            grain=DataGrain.RESEARCH_ROW,
            column_roles={
                "market_id": ColumnRole.KEY,
                "asset_id": ColumnRole.KEY,
                "timestamp": ColumnRole.TIMESTAMP,
                "market_type": ColumnRole.FACTOR,
                "forward_return_20d": ColumnRole.LABEL,
            },
            primary_key=platform_entity_time_key(),
            partition_columns=("year",),
        )
    )
    write_demo_processed_inputs(layout, processed_catalog)

    with pytest.raises(ValueError, match="invalid_research_factor_columns"):
        assemble_research_dataset(
            layout=layout,
            dataset_catalog=dataset_catalog,
            processed_catalog=processed_catalog,
            dataset_id="bad_demo",
            time_window=TimeWindow("2014-01-02", "2014-01-03"),
        )


@pytest.mark.parametrize(
    ("field_name", "bad_column", "expected_error"),
    [
        ("label_columns", "mom_20d", "research_input_column_overlap"),
        ("filter_columns", "market_type", "research_semantic_column_role_invalid"),
        ("context_columns", "eligible", "research_semantic_column_role_invalid"),
    ],
)
def test_research_dataset_role_validation_rejects_wrong_label_filter_and_context(
    tmp_path: Path,
    field_name: str,
    bad_column: str,
    expected_error: str,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    processed_catalog = make_demo_processed_catalog()
    dataset_catalog = ResearchDatasetCatalog()
    spec_kwargs = {
        "factor_columns": ("mom_20d",),
        "label_columns": ("forward_return_20d",),
        "filter_columns": ("eligible", "can_enter", "can_exit"),
        "context_columns": ("market_type",),
    }
    spec_kwargs[field_name] = (bad_column,)
    with pytest.raises(ValueError, match=expected_error):
        dataset_catalog.register(
            ResearchDatasetSpec(
                data_id=f"bad_{field_name}",
                market_id=MarketId("a_share"),
                input_data_ids=("demo_factors", "demo_labels", "demo_annotations"),
                base_input_data_id="demo_annotations",
                **spec_kwargs,
                frequency=DataFrequency.DAILY,
                grain=DataGrain.RESEARCH_ROW,
                column_roles={
                    "market_id": ColumnRole.KEY,
                    "asset_id": ColumnRole.KEY,
                    "timestamp": ColumnRole.TIMESTAMP,
                    "mom_20d": ColumnRole.FACTOR,
                    "forward_return_20d": ColumnRole.LABEL,
                    "eligible": ColumnRole.UNIVERSE,
                    "can_enter": ColumnRole.MASK,
                    "can_exit": ColumnRole.MASK,
                    "market_type": ColumnRole.CONTEXT,
                },
                primary_key=platform_entity_time_key(),
                partition_columns=("year",),
            )
        )
        write_demo_processed_inputs(layout, processed_catalog)
        assemble_research_dataset(
            layout=layout,
            dataset_catalog=dataset_catalog,
            processed_catalog=processed_catalog,
            dataset_id=f"bad_{field_name}",
            time_window=TimeWindow("2014-01-02", "2014-01-03"),
        )


def test_evaluation_prediction_and_report_boundary(tmp_path: Path) -> None:
    frame = _research_rows()
    eval_spec = EvaluationSpec(
        run_id="eval_demo",
        market_id=MarketId("a_share"),
        research_dataset_id="demo",
        signals=(make_signal_spec("mom_20d"),),
        label_columns=("forward_return_20d",),
    )
    model_spec = ModelSpec(
        model_id="linear_demo",
        model_algorithm="linear_regression",
        input_factor_columns=("mom_20d",),
        label_column="forward_return_20d",
        max_train_rows=len(frame),
    )
    prediction_spec = PredictionSpec(
        prediction_id="pred_demo",
        model_id=model_spec.model_id,
        research_dataset_id="demo",
        train_end="2014-01-02",
        test_start="2014-01-03",
        test_end="2014-01-03",
    )

    evaluation = evaluate_signal_frame_chunks((frame,), eval_spec)
    model_result = model_fitting.fit_model_result_from_frames(
        (frame.loc[frame["timestamp"] <= pd.Timestamp(prediction_spec.train_end)],),
        model_spec,
        estimator=build_estimator_from_model_spec(model_spec),
    )
    prediction, output_bundle, _ = write_prediction_output_and_read(
        tmp_path,
        (frame,),
        prediction_spec,
        model_result.coefficients,
        model_spec.input_columns,
        model_metadata=model_result.metadata,
    )

    assert len(evaluation.metrics) == 2
    assert len(evaluation.ic_summary) == 1
    assert len(evaluation.group_spread_summary) == 1
    assert evaluation.diagnostics.rows_input == 4
    assert prediction_spec.prediction_column in prediction.columns
    assert prediction["timestamp"].min() >= pd.Timestamp("2014-01-03")
    assert prediction["timestamp"].max() <= pd.Timestamp("2014-01-03")
    assert output_bundle.as_posix().endswith("outputs/pred_demo")


@pytest.mark.parametrize(
    ("model_algorithm", "expected_package", "wrong_package"),
    (
        ("linear_regression", "internal", "sklearn"),
        ("ridge_regression", "sklearn", "internal"),
    ),
)
def test_prediction_builder_binds_backend_package_to_actual_model_owner(
    model_algorithm: str,
    expected_package: str,
    wrong_package: str,
) -> None:
    frame = _research_rows()
    model_spec = ModelSpec(
        model_id=f"{model_algorithm}_demo",
        model_algorithm=model_algorithm,
        input_factor_columns=("mom_20d",),
        label_column="forward_return_20d",
        max_train_rows=len(frame),
    )
    model_result = model_fitting.fit_model_result_from_frames(
        (frame,),
        model_spec,
        estimator=build_estimator_from_model_spec(model_spec),
    )
    prediction_spec = PredictionSpec(
        prediction_id=f"{model_algorithm}_prediction",
        model_id=model_spec.model_id,
        research_dataset_id="demo",
        train_end="2014-01-02",
        test_start="2014-01-03",
        test_end="2014-01-03",
    )
    result = prediction_runner_module.build_prediction_result_from_frames(
        (frame,),
        prediction_spec,
        model_result.coefficients,
        model_spec.input_columns,
        prediction_model=model_result.prediction_model,
        model_metadata=model_result.metadata,
    )

    assert result.model_metadata["dependency_status"]["package"] == expected_package
    wrong_metadata = {
        **model_result.metadata,
        "dependency_status": {
            **model_result.metadata["dependency_status"],
            "package": wrong_package,
        },
    }
    with pytest.raises(ValueError, match="invalid_model_metadata"):
        prediction_runner_module.build_prediction_result_from_frames(
            (frame,),
            prediction_spec,
            model_result.coefficients,
            model_spec.input_columns,
            prediction_model=model_result.prediction_model,
            model_metadata=wrong_metadata,
        )


def test_prediction_handoff_reads_research_dataset_and_writes_report(
    tmp_path: Path,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    dataset_catalog = make_research_dataset_catalog(
        filter_column_roles={"eligible": ColumnRole.UNIVERSE},
    )
    processed_catalog = make_demo_processed_catalog()
    write_demo_processed_inputs(layout, processed_catalog)
    assemble_research_dataset(
        layout=layout,
        dataset_catalog=dataset_catalog,
        processed_catalog=processed_catalog,
        dataset_id="demo",
        time_window=TimeWindow("2014-01-02", "2014-01-03"),
    )
    model_catalog = ModelCatalog()
    model_catalog.register(
        ModelSpec(
            model_id="linear_demo",
            model_algorithm="linear_regression",
            input_factor_columns=("mom_20d",),
            label_column="forward_return_20d",
        )
    )
    prediction_catalog = PredictionCatalog()
    prediction_catalog.register(
        PredictionSpec(
            prediction_id="pred_demo",
            model_id="linear_demo",
            research_dataset_id="demo",
            train_end="2014-01-02",
            test_start="2014-01-03",
            test_end="2014-01-03",
        )
    )

    with build_prediction_result_from_research_dataset(
        layout=layout,
        dataset_catalog=dataset_catalog,
        model_catalog=model_catalog,
        prediction_catalog=prediction_catalog,
        prediction_id="pred_demo",
    ) as result:
        output_bundle = _write_prediction_result(layout, result)

    assert output_bundle.as_posix().endswith("outputs/pred_demo")
    assert (output_bundle / "prediction.parquet").is_file()
    manifest = json.loads((output_bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["metadata"]["is_oos"] is True
    assert manifest["metadata"]["prediction_mode"] == "minimal_oos"
    assert manifest["metadata"]["prediction_column"] == PREDICTION_SCORE_COLUMN
    assert manifest["metadata"]["primary_key"] == list(PREDICTION_PRIMARY_KEY)
    assert manifest["metadata"]["model_metadata"]["input_count"] == 1
    assert manifest["metadata"]["model_metadata"]["coefficient_count"] == 2
    assert manifest["metadata"]["model_metadata"]["train_start"] == "2014-01-02"
    assert manifest["metadata"]["model_metadata"]["train_end"] == "2014-01-02"
    assert manifest["metadata"]["prediction_metadata"]["prediction_rows"] == 2
    assert (
        manifest["metadata"]["prediction_metadata"]["requested_test_end"]
        == "2014-01-03"
    )
    assert (
        manifest["metadata"]["prediction_metadata"]["effective_test_end"]
        == "2014-01-03"
    )


def test_prediction_handoff_writes_prediction_report_without_materialized_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    dataset_catalog = make_research_dataset_catalog(
        filter_column_roles={"eligible": ColumnRole.UNIVERSE},
    )
    processed_catalog = make_demo_processed_catalog()
    write_demo_processed_inputs(layout, processed_catalog)
    assemble_research_dataset(
        layout=layout,
        dataset_catalog=dataset_catalog,
        processed_catalog=processed_catalog,
        dataset_id="demo",
        time_window=TimeWindow("2014-01-02", "2014-01-03"),
    )
    model_catalog = ModelCatalog()
    model_catalog.register(
        ModelSpec(
            model_id="linear_demo",
            model_algorithm="linear_regression",
            input_factor_columns=("mom_20d",),
            label_column="forward_return_20d",
        )
    )
    prediction_catalog = PredictionCatalog()
    prediction_catalog.register(
        PredictionSpec(
            prediction_id="pred_demo",
            model_id="linear_demo",
            research_dataset_id="demo",
            train_end="2014-01-02",
            test_start="2014-01-03",
            test_end="2014-01-03",
        )
    )

    with build_prediction_result_from_research_dataset(
        layout=layout,
        dataset_catalog=dataset_catalog,
        model_catalog=model_catalog,
        prediction_catalog=prediction_catalog,
        prediction_id="pred_demo",
    ) as result:
        output_bundle = _write_prediction_result(layout, result)

    assert (output_bundle / "prediction.parquet").is_file()


def test_prediction_handoff_streams_research_chunks_for_model_and_prediction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _research_rows_for_dates(
        [
            "2014-01-02",
            "2014-01-03",
            "2014-01-06",
            "2014-01-07",
            "2014-01-08",
        ]
    )
    layout = StorageLayout(project_root=tmp_path)
    dataset_catalog = make_research_dataset_catalog(
        filter_column_roles={"eligible": ColumnRole.UNIVERSE},
    )
    processed_catalog = make_demo_processed_catalog()
    write_demo_processed_inputs(layout, processed_catalog, frame=frame)
    assembly_policy = capture_storage_test_policy(
        working_headroom_bytes=4 * 1024**2,
    )
    assemble_research_dataset(
        layout=layout,
        dataset_catalog=dataset_catalog,
        processed_catalog=processed_catalog,
        dataset_id="demo",
        time_window=TimeWindow("2014-01-02", "2014-01-08"),
        resource_policy=assembly_policy,
    )
    model_catalog = ModelCatalog()
    model_catalog.register(
        ModelSpec(
            model_id="linear_demo",
            model_algorithm="linear_regression",
            input_factor_columns=("mom_20d",),
            label_column="forward_return_20d",
        )
    )
    prediction_catalog = PredictionCatalog()
    prediction_catalog.register(
        PredictionSpec(
            prediction_id="pred_demo",
            model_id="linear_demo",
            research_dataset_id="demo",
            train_end="2014-01-03",
            test_start="2014-01-06",
            test_end="2014-01-08",
        )
    )

    original_iter = prediction_runner_module.iter_research_dataset_chunks
    original_fit = prediction_runner_module.fitting.fit_model_result_from_frames
    original_write = prediction_runner_module.build_prediction_result_from_frames
    read_chunk_rows: list[int] = []
    fit_chunk_rows: list[int] = []
    prediction_chunk_rows: list[int] = []
    read_columns_seen: list[tuple[str, ...]] = []

    def wrapped_iter(*args, **kwargs):
        read_columns_seen.append(tuple(kwargs["columns"]))
        for chunk in original_iter(*args, **kwargs):
            read_chunk_rows.append(len(chunk))
            yield chunk

    def wrapped_fit(frames, spec, *, estimator, resource_policy=None):
        chunks = list(frames)
        fit_chunk_rows.extend(len(chunk) for chunk in chunks)
        return original_fit(
            iter(chunks),
            spec,
            estimator=estimator,
            resource_policy=resource_policy,
        )

    def wrapped_write(frames, spec, coefficients, input_columns, **kwargs):
        def counted_frames():
            for chunk in frames:
                prediction_chunk_rows.append(len(chunk))
                yield chunk

        return original_write(
            counted_frames(),
            spec,
            coefficients,
            input_columns,
            **kwargs,
        )

    monkeypatch.setattr(
        prediction_runner_module,
        "iter_research_dataset_chunks",
        wrapped_iter,
    )
    monkeypatch.setattr(
        prediction_runner_module.fitting,
        "fit_model_result_from_frames",
        wrapped_fit,
    )
    monkeypatch.setattr(
        prediction_runner_module,
        "build_prediction_result_from_frames",
        wrapped_write,
    )

    prediction_policy = capture_storage_test_policy(
        working_headroom_bytes=4 * 1024**2,
    )
    with runtime_resource_policy_scope(prediction_policy):
        with build_prediction_result_from_research_dataset(
            layout=layout,
            dataset_catalog=dataset_catalog,
            model_catalog=model_catalog,
            prediction_catalog=prediction_catalog,
            prediction_id="pred_demo",
            resource_policy=prediction_policy,
        ) as result:
            output_bundle = _write_prediction_result(layout, result)

    assert (output_bundle / "prediction.parquet").is_file()
    assert read_chunk_rows == [len(frame), len(frame)]
    assert fit_chunk_rows
    assert max(fit_chunk_rows) < len(frame)
    assert prediction_chunk_rows == [len(frame)]
    assert any("forward_return_20d" in columns for columns in read_columns_seen)
    assert any("forward_return_20d" not in columns for columns in read_columns_seen)


def test_evaluation_runner_reads_research_dataset_and_writes_report(
    tmp_path: Path,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    dataset_catalog = make_research_dataset_catalog(
        filter_column_roles={"eligible": ColumnRole.UNIVERSE},
    )
    processed_catalog = make_demo_processed_catalog()
    write_demo_processed_inputs(layout, processed_catalog)
    assemble_research_dataset(
        layout=layout,
        dataset_catalog=dataset_catalog,
        processed_catalog=processed_catalog,
        dataset_id="demo",
        time_window=TimeWindow("2014-01-02", "2014-01-03"),
    )
    evaluation_catalog = EvaluationCatalog()
    evaluation_catalog.register(
        EvaluationSpec(
            run_id="eval_demo",
            market_id=MarketId("a_share"),
            research_dataset_id="demo",
            signals=(make_signal_spec("mom_20d"),),
            label_columns=("forward_return_20d",),
        )
    )

    result = build_evaluation_result_from_research_dataset(
        layout=layout,
        dataset_catalog=dataset_catalog,
        evaluation_catalog=evaluation_catalog,
        run_id="eval_demo",
    )
    output_bundle = _write_evaluation_result(layout, result)

    assert output_bundle.as_posix().endswith("outputs/eval_demo")
    assert (output_bundle / "metrics.parquet").is_file()
    assert (output_bundle / "grouped_label_values.parquet").is_file()
    assert (output_bundle / "ic_summary.parquet").is_file()
    assert (output_bundle / "group_spread_summary.parquet").is_file()
    assert (output_bundle / "summary.md").is_file()
    assert (output_bundle / "manifest.json").is_file()
    manifest = json.loads((output_bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["files"] == [
        "metrics.parquet",
        "grouped_label_values.parquet",
        "ic_summary.parquet",
        "group_spread_summary.parquet",
        "summary.md",
    ]


def test_evaluation_runner_streams_research_chunks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _research_rows_for_dates(
        [
            "2014-01-02",
            "2014-01-03",
            "2014-01-06",
            "2014-01-07",
            "2014-01-08",
        ]
    )
    layout = StorageLayout(project_root=tmp_path)
    dataset_catalog = make_research_dataset_catalog(
        filter_column_roles={"eligible": ColumnRole.UNIVERSE},
    )
    processed_catalog = make_demo_processed_catalog()
    write_demo_processed_inputs(layout, processed_catalog, frame=frame)
    assembly_policy = capture_storage_test_policy(
        working_headroom_bytes=4 * 1024**2,
    )
    assemble_research_dataset(
        layout=layout,
        dataset_catalog=dataset_catalog,
        processed_catalog=processed_catalog,
        dataset_id="demo",
        time_window=TimeWindow("2014-01-02", "2014-01-08"),
        resource_policy=assembly_policy,
    )
    evaluation_catalog = EvaluationCatalog()
    evaluation_catalog.register(
        EvaluationSpec(
            run_id="eval_demo",
            market_id=MarketId("a_share"),
            research_dataset_id="demo",
            signals=(make_signal_spec("mom_20d"),),
            label_columns=("forward_return_20d",),
        )
    )
    original_iter = evaluation_runner_module.iter_research_dataset_chunks
    read_chunk_rows: list[int] = []

    def wrapped_iter(*args, **kwargs):
        for chunk in original_iter(*args, **kwargs):
            read_chunk_rows.append(len(chunk))
            yield chunk

    monkeypatch.setattr(
        evaluation_runner_module,
        "iter_research_dataset_chunks",
        wrapped_iter,
    )

    evaluation_policy = capture_test_policy(
        available_memory_bytes=2 * 1024**2,
        explicit_cap_bytes=1024**2,
    )
    with runtime_resource_policy_scope(evaluation_policy):
        result = build_evaluation_result_from_research_dataset(
            layout=layout,
            dataset_catalog=dataset_catalog,
            evaluation_catalog=evaluation_catalog,
            run_id="eval_demo",
            resource_policy=evaluation_policy,
        )
    output_bundle = _write_evaluation_result(layout, result)

    assert (output_bundle / "metrics.parquet").is_file()
    assert read_chunk_rows == [len(frame)]

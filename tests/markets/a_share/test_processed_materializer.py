from __future__ import annotations

import inspect
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from quant_research_platform.catalog.factors import FactorCatalog
from quant_research_platform.catalog.labels import LabelCatalog
from quant_research_platform.catalog.processed import ProcessedDataCatalog
from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.data import (
    DataFrequency,
    ProcessedDataKind,
)
from quant_research_platform.contracts.ids import SourceId
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.implementations.a_share.ids import (
    market_context_data_id,
    market_panel_data_id,
)
from quant_research_platform.markets.implementations.a_share.processed import (
    materialization_io,
    materialization_planning,
    materialize_annotations,
    materialize_factors,
    materialize_labels,
    materialize_panel,
    materializer,
)
from quant_research_platform.markets.implementations.a_share.processed.catalog import (
    processed_catalog,
)
from quant_research_platform.research.factors.price_volume import (
    price_volume_window_column,
)
from quant_research_platform.research.labels.forward_returns import (
    forward_return_label_column,
)
from quant_research_platform.runtime.resources import (
    RuntimeResourcePolicy,
    active_runtime_resource_policy,
)
from quant_research_platform.sources.csmar import ids as csmar_ids
from quant_research_platform.storage.datasets import (
    read_dataset,
    write_dataset,
)
from quant_research_platform.storage.layout import StorageLayout
from tests.helpers.processed_chunk_parity import (
    assert_forced_tiny_processed_chunks_match_default,
)
from tests.helpers.synthetic_a_share import (
    processed_a_share_root,
    write_synthetic_interim_inputs,
)

SYNTHETIC_SOURCE_ID = SourceId("synthetic_vendor_a")
MARKET_BARS_DATA_ID = csmar_ids.market_bars_data_id(DataFrequency.DAILY)
MARKET_PANEL_DATA_ID = market_panel_data_id()
MARKET_CONTEXT_DATA_ID = market_context_data_id()


def test_a_share_output_disk_guard_checks_each_chunk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[Path] = []
    monkeypatch.setattr(
        materialization_io,
        "require_free_disk_headroom",
        lambda path, **_kwargs: observed.append(Path(path)),
    )
    frames = (pd.DataFrame({"value": [1]}), pd.DataFrame({"value": [2]}))

    result = list(
        materialization_io._iter_disk_guarded_output_chunks(
            frames,
            scratch_parent=tmp_path,
        )
    )

    assert [len(frame) for frame in result] == [1, 1]
    assert observed == [tmp_path, tmp_path]
    source = inspect.getsource(materialization_io.write_and_inspect)
    assert "resource_policy" in source
    assert "_iter_disk_guarded_output_chunks(" in source


def test_a_share_output_wrappers_release_previous_chunk_before_next_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = processed_catalog().get(MARKET_PANEL_DATA_ID)
    frame = pd.DataFrame(
        {
            column: (
                pd.to_datetime(["2014-01-02"])
                if column == "timestamp"
                else ["a_share" if column == "market_id" else "value"]
            )
            for column in spec.required_columns
        }
    )
    observed: list[tuple[set[str], set[str]]] = []

    def source():  # noqa: ANN202
        for current in (frame, frame.iloc[:0].copy(), frame.copy()):
            yield current
            sorted_frame = inspect.currentframe().f_back
            assert sorted_frame is not None
            disk_frame = sorted_frame.f_back
            assert disk_frame is not None
            observed.append(
                (
                    {"chunk", "output"}.intersection(sorted_frame.f_locals),
                    {"chunk", "chunk_bytes"}.intersection(disk_frame.f_locals),
                )
            )

    monkeypatch.setattr(
        materialization_io,
        "require_free_disk_headroom",
        lambda *_args, **_kwargs: None,
    )
    chunks = materialization_io._iter_disk_guarded_output_chunks(
        materialization_io.iter_sorted_output_chunks(source(), spec),
        scratch_parent=tmp_path,
    )

    output = tuple(chunks)

    assert len(output) == 2
    assert observed[:2] == [(set(), set()), (set(), set())]


@pytest.mark.parametrize(
    "exit_mode",
    ("success", "runtime_error", "keyboard_interrupt", "system_exit", "close"),
)
def test_a_share_output_wrapper_close_cascades_to_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exit_mode: str,
) -> None:
    spec = processed_catalog().get(MARKET_PANEL_DATA_ID)
    frame = pd.DataFrame(
        {
            column: (
                pd.to_datetime(["2014-01-02"])
                if column == "timestamp"
                else ["a_share" if column == "market_id" else "value"]
            )
            for column in spec.required_columns
        }
    )
    closed = False

    def source():  # noqa: ANN202
        nonlocal closed
        try:
            yield frame.iloc[:0].copy()
            yield frame
            yield frame.copy()
        finally:
            closed = True

    failure_types = {
        "runtime_error": RuntimeError,
        "keyboard_interrupt": KeyboardInterrupt,
        "system_exit": SystemExit,
    }
    failure_type = failure_types.get(exit_mode)

    def disk_check(*_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
        if failure_type is not None:
            raise failure_type("disk_exit")

    monkeypatch.setattr(materialization_io, "require_free_disk_headroom", disk_check)
    source_chunks = source()
    sorted_chunks = materialization_io.iter_sorted_output_chunks(source_chunks, spec)
    chunks = materialization_io._iter_disk_guarded_output_chunks(
        sorted_chunks,
        scratch_parent=tmp_path,
    )
    if exit_mode == "success":
        assert len(tuple(chunks)) == 2
    elif exit_mode == "close":
        assert len(next(chunks)) == 1
        chunks.close()
    else:
        assert failure_type is not None
        with pytest.raises(failure_type, match="disk_exit"):
            next(chunks)

    assert closed


def test_write_and_inspect_activates_explicit_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = RuntimeResourcePolicy.from_environment(
        available_memory_bytes=16 * 1024 * 1024,
        environ={},
    )
    spec = processed_catalog().get(MARKET_PANEL_DATA_ID)
    inspection = object()
    observed: list[RuntimeResourcePolicy] = []

    def capture_write(*_args: object, **_kwargs: object) -> None:
        observed.append(active_runtime_resource_policy())

    def capture_inspection(*_args: object, **_kwargs: object) -> object:
        observed.append(active_runtime_resource_policy())
        return inspection

    monkeypatch.setattr(materialization_io, "write_dataset", capture_write)
    monkeypatch.setattr(materialization_io, "inspect_dataset", capture_inspection)
    monkeypatch.setattr(
        materialization_io,
        "iter_sorted_output_chunks",
        lambda *_args, **_kwargs: (),
    )
    written = {}

    result = materialization_io.write_and_inspect(
        spec,
        pd.DataFrame(),
        layout=StorageLayout(project_root=tmp_path),
        time_window=TimeWindow("2024-01-02", "2024-01-02"),
        written=written,
        resource_policy=policy,
    )

    assert observed == [policy, policy]
    assert result is inspection
    assert written == {spec.data_id: inspection}


FORWARD_RETURN_1D_COLUMN = forward_return_label_column(1)
MOMENTUM_20D_COLUMN = price_volume_window_column("mom", 20)


def test_materialize_processed_data_writes_requested_panel_and_dependencies(
    tmp_path,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    write_synthetic_interim_inputs(
        layout, SYNTHETIC_SOURCE_ID, market_bars_data_id=MARKET_BARS_DATA_ID
    )

    written = materializer.materialize_processed_data(
        layout=layout,
        source_id=SYNTHETIC_SOURCE_ID,
        time_window=TimeWindow("2014-01-02", "2014-01-03"),
        data_ids=(MARKET_PANEL_DATA_ID,),
    )

    assert set(written) == {MARKET_PANEL_DATA_ID}
    root = tmp_path / "data" / "processed" / "a_share" / "panel" / MARKET_PANEL_DATA_ID
    panel = read_dataset(root)
    assert len(panel) == 2
    assert panel["asset_id"].unique().tolist() == ["000001"]
    assert "source_asset_id" not in panel.columns


def test_panel_materializer_reuses_market_bar_inspection(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    write_synthetic_interim_inputs(
        layout, SYNTHETIC_SOURCE_ID, market_bars_data_id=MARKET_BARS_DATA_ID
    )
    market_bars_root = layout.interim_data_root(
        SYNTHETIC_SOURCE_ID,
        MARKET_BARS_DATA_ID,
    )
    calls = 0
    original = materialize_panel.inspect_dataset

    def spy_inspect_dataset(*args, **kwargs):
        nonlocal calls
        if Path(args[0]) == market_bars_root and "time_window" not in kwargs:
            calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(materialize_panel, "inspect_dataset", spy_inspect_dataset)

    materializer.materialize_processed_data(
        layout=layout,
        source_id=SYNTHETIC_SOURCE_ID,
        time_window=TimeWindow("2014-01-02", "2014-01-03"),
        data_ids=(MARKET_PANEL_DATA_ID,),
    )

    assert calls == 1


def test_panel_materializer_does_not_preload_full_market_bars(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    write_synthetic_interim_inputs(
        layout,
        SYNTHETIC_SOURCE_ID,
        market_bars_data_id=MARKET_BARS_DATA_ID,
        end_date="2014-01-10",
    )
    market_bars_root = layout.interim_data_root(
        SYNTHETIC_SOURCE_ID,
        MARKET_BARS_DATA_ID,
    )
    original = materialize_panel.materialization_io.read_materialized_dataset
    full_market_bars_reads = 0
    windowed_market_bars_reads = 0

    def spy_read_dataset(root, **kwargs):
        nonlocal full_market_bars_reads, windowed_market_bars_reads
        if Path(root) == market_bars_root:
            if kwargs.get("time_window") is None:
                full_market_bars_reads += 1
            else:
                windowed_market_bars_reads += 1
        return original(root, **kwargs)

    monkeypatch.setattr(
        materialize_panel.materialization_io,
        "read_materialized_dataset",
        spy_read_dataset,
    )

    materializer.materialize_processed_data(
        layout=layout,
        source_id=SYNTHETIC_SOURCE_ID,
        time_window=TimeWindow("2014-01-02", "2014-01-08"),
        data_ids=(MARKET_PANEL_DATA_ID,),
        resource_policy=RuntimeResourcePolicy.from_environment(
            available_memory_bytes=512 * 1024,
            environ={},
        ),
    )

    assert full_market_bars_reads == 0
    assert windowed_market_bars_reads > 0


def test_materialize_processed_data_rejects_existing_requested_root(tmp_path) -> None:
    layout = StorageLayout(project_root=tmp_path)
    write_synthetic_interim_inputs(
        layout, SYNTHETIC_SOURCE_ID, market_bars_data_id=MARKET_BARS_DATA_ID
    )
    kwargs = {
        "layout": layout,
        "source_id": SYNTHETIC_SOURCE_ID,
        "time_window": TimeWindow("2014-01-02", "2014-01-03"),
        "data_ids": (MARKET_PANEL_DATA_ID,),
    }

    materializer.materialize_processed_data(**kwargs)

    with pytest.raises(ValueError, match="dataset_exists"):
        materializer.materialize_processed_data(**kwargs)


def test_materialize_processed_data_requires_exactly_one_selector(tmp_path) -> None:
    common = {
        "layout": StorageLayout(project_root=tmp_path),
        "source_id": SYNTHETIC_SOURCE_ID,
        "time_window": TimeWindow("2014-01-02", "2014-01-03"),
    }

    with pytest.raises(ValueError, match="data_ids_xor_kinds_required"):
        materializer.materialize_processed_data(**common)
    with pytest.raises(ValueError, match="data_ids_xor_kinds_required"):
        materializer.materialize_processed_data(
            **common,
            data_ids=(MARKET_PANEL_DATA_ID,),
            kinds=frozenset({ProcessedDataKind.PANEL}),
        )


@pytest.mark.parametrize("frequency", [DataFrequency.MINUTE, DataFrequency.TICK])
def test_materialize_processed_data_rejects_non_daily_active_profile_before_io(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    frequency: DataFrequency,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    catalog = ProcessedDataCatalog()
    catalog.register(
        replace(
            processed_catalog().get(MARKET_PANEL_DATA_ID),
            frequency=frequency,
        )
    )

    def fail_if_sessions_are_read(
        *args: object,
        **kwargs: object,
    ) -> tuple[pd.Timestamp, ...]:
        raise AssertionError("frequency validation should run before source reads")

    monkeypatch.setattr(materializer, "processed_catalog", lambda: catalog)
    monkeypatch.setattr(
        materialization_planning,
        "available_trading_sessions",
        fail_if_sessions_are_read,
    )

    with pytest.raises(
        ValueError,
        match=(
            "unsupported_processed_frequency: "
            f"{MARKET_PANEL_DATA_ID}: {frequency.value}"
        ),
    ):
        materializer.materialize_processed_data(
            layout=layout,
            source_id=SYNTHETIC_SOURCE_ID,
            time_window=TimeWindow("2014-01-02", "2014-01-03"),
            data_ids=(MARKET_PANEL_DATA_ID,),
        )

    assert not (
        tmp_path / "data" / "processed" / "a_share" / "panel" / MARKET_PANEL_DATA_ID
    ).exists()


def test_materialize_processed_data_by_kind_writes_dependency_and_target(
    tmp_path,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    write_synthetic_interim_inputs(
        layout, SYNTHETIC_SOURCE_ID, market_bars_data_id=MARKET_BARS_DATA_ID
    )

    written = materializer.materialize_processed_data(
        layout=layout,
        source_id=SYNTHETIC_SOURCE_ID,
        time_window=TimeWindow("2014-01-02", "2014-01-03"),
        kinds=frozenset({ProcessedDataKind.FACTOR}),
    )

    assert set(written) == {MARKET_PANEL_DATA_ID, "price_volume"}
    assert (
        tmp_path / "data" / "processed" / "a_share" / "panel" / MARKET_PANEL_DATA_ID
    ).exists()
    assert (
        tmp_path / "data" / "processed" / "a_share" / "factor" / "price_volume"
    ).exists()


@pytest.mark.parametrize(
    "data_ids",
    [
        ("forward_returns",),
        (MARKET_PANEL_DATA_ID, "forward_returns"),
    ],
)
def test_label_materializer_expands_panel_dependency_for_future_halo(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    data_ids: tuple[str, ...],
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    write_synthetic_interim_inputs(
        layout,
        SYNTHETIC_SOURCE_ID,
        market_bars_data_id=MARKET_BARS_DATA_ID,
        start_date="2014-01-02",
        end_date="2014-01-10",
    )
    monkeypatch.setattr(
        materializer,
        "processed_catalog",
        lambda: _catalog_with_one_step_forward_return_only(),
    )

    materializer.materialize_processed_data(
        layout=layout,
        source_id=SYNTHETIC_SOURCE_ID,
        time_window=TimeWindow("2014-01-02", "2014-01-03"),
        data_ids=data_ids,
    )

    panel = read_dataset(
        processed_a_share_root(tmp_path, "panel", MARKET_PANEL_DATA_ID)
    )
    labels = read_dataset(processed_a_share_root(tmp_path, "label", "forward_returns"))
    assert pd.to_datetime(panel["timestamp"]).dt.date.max().isoformat() == "2014-01-07"
    jan3_label = labels.loc[
        pd.to_datetime(labels["timestamp"]).dt.date.astype(str).eq("2014-01-03"),
        FORWARD_RETURN_1D_COLUMN,
    ].iloc[0]
    assert pd.notna(jan3_label)


@pytest.mark.parametrize(
    "data_ids",
    [
        ("price_volume",),
        (MARKET_PANEL_DATA_ID, "price_volume"),
    ],
)
def test_factor_materializer_expands_panel_dependency_for_history_halo(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    data_ids: tuple[str, ...],
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    write_synthetic_interim_inputs(
        layout,
        SYNTHETIC_SOURCE_ID,
        market_bars_data_id=MARKET_BARS_DATA_ID,
        start_date="2014-01-01",
        end_date="2014-01-31",
    )
    monkeypatch.setattr(
        materializer,
        "processed_catalog",
        lambda: _catalog_with_momentum_20d_only(),
    )

    materializer.materialize_processed_data(
        layout=layout,
        source_id=SYNTHETIC_SOURCE_ID,
        time_window=TimeWindow("2014-01-31", "2014-01-31"),
        data_ids=data_ids,
    )

    panel = read_dataset(
        processed_a_share_root(tmp_path, "panel", MARKET_PANEL_DATA_ID)
    )
    factors = read_dataset(processed_a_share_root(tmp_path, "factor", "price_volume"))
    assert pd.to_datetime(panel["timestamp"]).dt.date.min().isoformat() == "2014-01-03"
    assert pd.notna(factors[MOMENTUM_20D_COLUMN].iloc[0])


def test_materializer_empty_target_window_still_fails_closed(tmp_path) -> None:
    layout = StorageLayout(project_root=tmp_path)
    write_synthetic_interim_inputs(
        layout, SYNTHETIC_SOURCE_ID, market_bars_data_id=MARKET_BARS_DATA_ID
    )

    with pytest.raises(ValueError, match="no_trading_sessions_in_time_window"):
        materializer.materialize_processed_data(
            layout=layout,
            source_id=SYNTHETIC_SOURCE_ID,
            time_window=TimeWindow("2009-01-01", "2009-01-31"),
            data_ids=(MARKET_PANEL_DATA_ID,),
        )


def test_factor_materializer_rejects_unexpected_declared_processed_input(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    write_synthetic_interim_inputs(
        layout, SYNTHETIC_SOURCE_ID, market_bars_data_id=MARKET_BARS_DATA_ID
    )
    catalog = _catalog_with_processed_inputs(
        "price_volume",
        (MARKET_PANEL_DATA_ID, MARKET_CONTEXT_DATA_ID),
    )
    monkeypatch.setattr(materializer, "processed_catalog", lambda: catalog)

    with pytest.raises(
        ValueError,
        match=(
            "unexpected_processed_input_for_factor: "
            f"price_volume: {MARKET_CONTEXT_DATA_ID}"
        ),
    ):
        materializer.materialize_processed_data(
            layout=layout,
            source_id=SYNTHETIC_SOURCE_ID,
            time_window=TimeWindow("2014-01-02", "2014-01-03"),
            data_ids=("price_volume",),
        )


def test_factor_materializer_requires_family_contract_input_declared_by_spec(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    write_synthetic_interim_inputs(
        layout, SYNTHETIC_SOURCE_ID, market_bars_data_id=MARKET_BARS_DATA_ID
    )
    catalog = _catalog_with_processed_inputs("price_volume", ())
    monkeypatch.setattr(materializer, "processed_catalog", lambda: catalog)

    with pytest.raises(
        ValueError,
        match=(
            f"missing_processed_input_for_factor: price_volume: {MARKET_PANEL_DATA_ID}"
        ),
    ):
        materializer.materialize_processed_data(
            layout=layout,
            source_id=SYNTHETIC_SOURCE_ID,
            time_window=TimeWindow("2014-01-02", "2014-01-03"),
            data_ids=("price_volume",),
        )


@pytest.mark.parametrize(
    (
        "definition_owner",
        "catalog_name",
        "families",
        "data_ids",
        "error_match",
    ),
    [
        (
            materialize_factors.factor_definitions,
            "factor_catalog",
            ("alternative_price_family",),
            ("price_volume",),
            "unknown_factor_family: alternative_price_family",
        ),
        (
            materialize_factors.factor_definitions,
            "factor_catalog",
            ("price_volume", "other_factor"),
            ("price_volume",),
            "mixed_factor_families: price_volume, other_factor",
        ),
        (
            materialize_labels.label_definitions,
            "label_catalog",
            ("alternative_label_family",),
            ("forward_returns",),
            "unknown_label_family: alternative_label_family",
        ),
        (
            materialize_labels.label_definitions,
            "label_catalog",
            ("forward_return", "other_label"),
            ("forward_returns",),
            "mixed_label_families: forward_return, other_label",
        ),
    ],
)
def test_materializer_family_contract_failures_are_closed(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    definition_owner,
    catalog_name: str,
    families: tuple[str, ...],
    data_ids: tuple[str, ...],
    error_match: str,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    write_synthetic_interim_inputs(
        layout, SYNTHETIC_SOURCE_ID, market_bars_data_id=MARKET_BARS_DATA_ID
    )
    catalog_builder = (
        _factor_catalog_with_families
        if catalog_name == "factor_catalog"
        else _label_catalog_with_families
    )
    monkeypatch.setattr(
        definition_owner,
        catalog_name,
        lambda: catalog_builder(families),
    )

    with pytest.raises(ValueError, match=error_match):
        materializer.materialize_processed_data(
            layout=layout,
            source_id=SYNTHETIC_SOURCE_ID,
            time_window=TimeWindow("2014-01-02", "2014-01-03"),
            data_ids=data_ids,
        )


def test_materializer_reuses_same_run_dependency_for_empty_window_tail(
    tmp_path,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    write_synthetic_interim_inputs(
        layout, SYNTHETIC_SOURCE_ID, market_bars_data_id=MARKET_BARS_DATA_ID
    )

    written = materializer.materialize_processed_data(
        layout=layout,
        source_id=SYNTHETIC_SOURCE_ID,
        time_window=TimeWindow("2014-01-02", "2014-01-04"),
        kinds=frozenset({ProcessedDataKind.ANNOTATION}),
    )

    assert {
        MARKET_PANEL_DATA_ID,
        MARKET_CONTEXT_DATA_ID,
        "default",
        "as_of",
        "execution",
        "evaluation",
    }.issubset(written)


def test_materialize_processed_data_existing_dependency_accepts_non_trading_end(
    tmp_path,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    write_synthetic_interim_inputs(
        layout, SYNTHETIC_SOURCE_ID, market_bars_data_id=MARKET_BARS_DATA_ID
    )
    materializer.materialize_processed_data(
        layout=layout,
        source_id=SYNTHETIC_SOURCE_ID,
        time_window=TimeWindow("2014-01-02", "2014-01-03"),
        data_ids=(MARKET_PANEL_DATA_ID,),
    )

    written = materializer.materialize_processed_data(
        layout=layout,
        source_id=SYNTHETIC_SOURCE_ID,
        time_window=TimeWindow("2014-01-02", "2014-01-04"),
        data_ids=("execution",),
    )

    assert "execution" in written


def test_materialize_processed_data_writes_only_core_trading_sessions(
    tmp_path,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    write_synthetic_interim_inputs(
        layout,
        SYNTHETIC_SOURCE_ID,
        market_bars_data_id=MARKET_BARS_DATA_ID,
        start_date="2017-04-28",
        end_date="2017-05-04",
        closed_dates=("2017-05-01",),
    )

    materializer.materialize_processed_data(
        layout=layout,
        source_id=SYNTHETIC_SOURCE_ID,
        time_window=TimeWindow("2017-04-28", "2017-05-03"),
        data_ids=("price_volume",),
    )

    panel = read_dataset(
        processed_a_share_root(tmp_path, "panel", MARKET_PANEL_DATA_ID)
    )
    factors = read_dataset(processed_a_share_root(tmp_path, "factor", "price_volume"))

    assert panel["timestamp"].dt.date.astype(str).tolist() == [
        "2017-04-28",
        "2017-05-02",
        "2017-05-03",
    ]
    assert factors["timestamp"].dt.date.astype(str).tolist() == [
        "2017-04-28",
        "2017-05-02",
        "2017-05-03",
    ]


def test_materialize_processed_data_existing_dependency_must_cover_trading_sessions(
    tmp_path,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    write_synthetic_interim_inputs(
        layout,
        SYNTHETIC_SOURCE_ID,
        market_bars_data_id=MARKET_BARS_DATA_ID,
        end_date="2014-01-06",
    )
    materializer.materialize_processed_data(
        layout=layout,
        source_id=SYNTHETIC_SOURCE_ID,
        time_window=TimeWindow("2014-01-02", "2014-01-03"),
        data_ids=(MARKET_PANEL_DATA_ID,),
    )

    with pytest.raises(ValueError, match="insufficient_dataset_coverage"):
        materializer.materialize_processed_data(
            layout=layout,
            source_id=SYNTHETIC_SOURCE_ID,
            time_window=TimeWindow("2014-01-02", "2014-01-06"),
            data_ids=("execution",),
        )


def test_existing_dependency_coverage_rejects_missing_middle_sessions(
    tmp_path,
) -> None:
    source_layout = StorageLayout(project_root=tmp_path / "source")
    write_synthetic_interim_inputs(
        source_layout,
        SYNTHETIC_SOURCE_ID,
        market_bars_data_id=MARKET_BARS_DATA_ID,
        start_date="2014-01-02",
        end_date="2016-01-04",
    )
    materializer.materialize_processed_data(
        layout=source_layout,
        source_id=SYNTHETIC_SOURCE_ID,
        time_window=TimeWindow("2014-01-02", "2016-01-04"),
        data_ids=(MARKET_PANEL_DATA_ID,),
    )
    source_panel_root = processed_a_share_root(
        tmp_path / "source",
        "panel",
        MARKET_PANEL_DATA_ID,
    )
    panel = read_dataset(source_panel_root)
    panel_without_2015 = panel.loc[pd.to_datetime(panel["timestamp"]).dt.year.ne(2015)]

    layout = StorageLayout(project_root=tmp_path / "coverage")
    write_synthetic_interim_inputs(
        layout,
        SYNTHETIC_SOURCE_ID,
        market_bars_data_id=MARKET_BARS_DATA_ID,
        start_date="2014-01-02",
        end_date="2016-01-04",
    )
    panel_root = processed_a_share_root(
        tmp_path / "coverage", "panel", MARKET_PANEL_DATA_ID
    )
    write_dataset(
        panel_root,
        panel_without_2015,
        spec=processed_catalog().get(MARKET_PANEL_DATA_ID),
    )

    with pytest.raises(
        ValueError,
        match=r"insufficient_dataset_coverage: missing_sessions=",
    ):
        materializer.materialize_processed_data(
            layout=layout,
            source_id=SYNTHETIC_SOURCE_ID,
            time_window=TimeWindow("2014-01-02", "2016-01-04"),
            data_ids=("execution",),
        )


def test_existing_dependency_coverage_accepts_non_trading_end_date(
    tmp_path,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    write_synthetic_interim_inputs(
        layout,
        SYNTHETIC_SOURCE_ID,
        market_bars_data_id=MARKET_BARS_DATA_ID,
        start_date="2016-12-29",
        end_date="2017-01-03",
    )
    materializer.materialize_processed_data(
        layout=layout,
        source_id=SYNTHETIC_SOURCE_ID,
        time_window=TimeWindow("2016-12-29", "2016-12-30"),
        data_ids=(MARKET_PANEL_DATA_ID,),
    )

    materializer.materialize_processed_data(
        layout=layout,
        source_id=SYNTHETIC_SOURCE_ID,
        time_window=TimeWindow("2016-12-29", "2016-12-31"),
        data_ids=("execution",),
    )

    annotation_root = processed_a_share_root(tmp_path, "annotation", "execution")
    annotation = read_dataset(annotation_root)
    assert annotation["timestamp"].dt.date.astype(str).tolist() == [
        "2016-12-29",
        "2016-12-30",
    ]


def test_annotation_materializer_uses_dependency_lookahead_session(tmp_path) -> None:
    layout = StorageLayout(project_root=tmp_path)
    write_synthetic_interim_inputs(
        layout,
        SYNTHETIC_SOURCE_ID,
        market_bars_data_id=MARKET_BARS_DATA_ID,
        start_date="2014-01-30",
        end_date="2014-02-07",
        closed_dates=(
            "2014-01-31",
            "2014-02-03",
            "2014-02-04",
            "2014-02-05",
            "2014-02-06",
        ),
    )
    materializer.materialize_processed_data(
        layout=layout,
        source_id=SYNTHETIC_SOURCE_ID,
        time_window=TimeWindow("2014-01-30", "2014-02-07"),
        data_ids=(MARKET_PANEL_DATA_ID,),
    )

    materializer.materialize_processed_data(
        layout=layout,
        source_id=SYNTHETIC_SOURCE_ID,
        time_window=TimeWindow("2014-01-30", "2014-01-30"),
        data_ids=("execution",),
    )

    execution = read_dataset(
        processed_a_share_root(tmp_path, "annotation", "execution")
    )
    assert execution["timestamp"].dt.date.astype(str).tolist() == ["2014-01-30"]
    assert execution["next_bar_is_trading"].tolist() == [True]
    assert execution["can_enter"].tolist() == [True]
    assert execution["can_exit"].tolist() == [True]


def test_materialize_processed_data_forced_tiny_chunks_equal_default_output(
    tmp_path,
) -> None:
    assert_forced_tiny_processed_chunks_match_default(
        tmp_path,
        data_id=MARKET_PANEL_DATA_ID,
        kind="panel",
        end_date="2014-01-08",
        time_window=TimeWindow("2014-01-02", "2014-01-07"),
    )


@pytest.mark.parametrize(
    ("data_id", "kind", "end_date", "time_window"),
    [
        (
            "execution",
            "annotation",
            "2014-01-08",
            TimeWindow("2014-01-02", "2014-01-07"),
        ),
        pytest.param(
            "price_volume",
            "factor",
            "2014-08-29",
            TimeWindow("2014-01-02", "2014-02-28"),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            "forward_returns",
            "label",
            "2014-08-29",
            TimeWindow("2014-01-02", "2014-02-28"),
            marks=pytest.mark.slow,
        ),
    ],
)
def test_materialize_processed_data_downstream_forced_tiny_chunks_equal_default_output(
    tmp_path,
    data_id: str,
    kind: str,
    end_date: str,
    time_window: TimeWindow,
) -> None:
    assert_forced_tiny_processed_chunks_match_default(
        tmp_path,
        data_id=data_id,
        kind=kind,
        end_date=end_date,
        time_window=time_window,
    )


@pytest.mark.parametrize(
    (
        "module",
        "yielded_name",
        "stale_names",
        "data_id",
        "kind",
        "end_date",
        "time_window",
    ),
    (
        (
            materialize_factors,
            "selected",
            {"processed_inputs", "panel", "factors", "output"},
            "price_volume",
            "factor",
            "2014-08-29",
            TimeWindow("2014-01-02", "2014-02-28"),
        ),
        (
            materialize_labels,
            "selected",
            {"processed_inputs", "panel", "labels", "output"},
            "forward_returns",
            "label",
            "2014-08-29",
            TimeWindow("2014-01-02", "2014-02-28"),
        ),
    ),
)
def test_a_share_window_pipeline_releases_previous_locals_before_next_compute(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    module: object,
    yielded_name: str,
    stale_names: set[str],
    data_id: str,
    kind: str,
    end_date: str,
    time_window: TimeWindow,
) -> None:
    original_disk_guard = materialization_io._iter_disk_guarded_output_chunks
    module_path = str(Path(module.__file__).resolve())
    observed: list[tuple[set[str], set[str], set[str], bool, bool, bool]] = []

    def inspect_disk_guard(chunks, *, scratch_parent):  # noqa: ANN001, ANN202
        sorted_chunks = iter(chunks)
        disk_chunks = original_disk_guard(
            sorted_chunks,
            scratch_parent=scratch_parent,
        )
        try:
            for yielded in disk_chunks:
                sorted_locals = inspect.getgeneratorlocals(sorted_chunks)
                producer = sorted_locals.get("frames")
                if (
                    inspect.isgenerator(producer)
                    and str(Path(producer.gi_code.co_filename).resolve()) == module_path
                ):
                    producer_locals = inspect.getgeneratorlocals(producer)
                    disk_locals = inspect.getgeneratorlocals(disk_chunks)
                    observed.append(
                        (
                            stale_names.intersection(producer_locals),
                            {"chunk"}.intersection(sorted_locals),
                            {"chunk_bytes"}.intersection(disk_locals),
                            isinstance(producer_locals.get(yielded_name), pd.DataFrame),
                            sorted_locals.get("output") is yielded,
                            disk_locals.get("chunk") is yielded,
                        )
                    )
                yield yielded
        finally:
            disk_chunks.close()

    monkeypatch.setattr(
        materialization_io,
        "_iter_disk_guarded_output_chunks",
        inspect_disk_guard,
    )
    assert_forced_tiny_processed_chunks_match_default(
        tmp_path,
        data_id=data_id,
        kind=kind,
        end_date=end_date,
        time_window=time_window,
    )

    assert observed
    assert observed == [(set(), set(), set(), True, True, True)] * len(observed)


def test_panel_window_producer_releases_previous_locals_before_next_compute(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = processed_catalog().get(MARKET_PANEL_DATA_ID)
    windows = tuple(
        SimpleNamespace(read=None, core_sessions=(pd.Timestamp(day),))
        for day in ("2014-01-02", "2014-01-03")
    )
    output = pd.DataFrame(
        {
            column: [pd.Timestamp("2014-01-02") if column == "timestamp" else 1]
            for column in spec.required_columns
        }
    )
    read_observed: list[set[str]] = []
    yield_observed: list[tuple[set[str], bool]] = []
    result = object()

    def read_market_bars(*_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
        producer = inspect.currentframe().f_back
        assert producer is not None
        read_observed.append(
            {"market_bars", "panel", "output", "selected"}.intersection(
                producer.f_locals
            )
        )
        return pd.DataFrame()

    def compute(_market_bars, **_kwargs):  # noqa: ANN001, ANN003, ANN202
        return output.copy()

    def consume(_spec, frames, **_kwargs):  # noqa: ANN001, ANN003
        iterator = iter(frames)
        for _ in windows:
            yielded = next(iterator)
            producer_locals = inspect.getgeneratorlocals(iterator)
            yield_observed.append(
                (
                    {"market_bars", "panel", "output"}.intersection(producer_locals),
                    producer_locals.get("selected") is yielded,
                )
            )
        with pytest.raises(StopIteration):
            next(iterator)
        return result

    source_inputs = {
        data_id: pd.DataFrame()
        for data_id in ("stock_master", "market_cap", "trading_calendar")
    }
    monkeypatch.setattr(
        materialize_panel.materialization_io,
        "read_source_inputs_for_spec",
        lambda *_args, **_kwargs: source_inputs,
    )
    monkeypatch.setattr(
        materialize_panel.materialization_validation,
        "require_source_inputs_for_spec",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        materialize_panel,
        "inspect_dataset",
        lambda _root: SimpleNamespace(columns=()),
    )
    monkeypatch.setattr(
        materialize_panel.materialization_validation,
        "require_source_input_columns_from_schema",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        materialize_panel.materialization_planning,
        "estimate_panel_bytes_per_session",
        lambda *_args, **_kwargs: 1,
    )
    monkeypatch.setattr(
        materialize_panel.materialization_planning,
        "planned_session_windows",
        lambda *_args, **_kwargs: windows,
    )
    monkeypatch.setattr(
        materialize_panel.materialization_io,
        "read_materialized_dataset",
        read_market_bars,
    )
    monkeypatch.setattr(
        materialize_panel.materialization_validation,
        "require_source_input_columns",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(materialize_panel.panels, "compute_market_panel", compute)
    monkeypatch.setattr(
        materialize_panel.materialization_planning,
        "filter_output_sessions",
        lambda current, _sessions: current,
    )
    monkeypatch.setattr(
        materialize_panel.materialization_planning,
        "require_known_materialization_payload_bound",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        materialize_panel.materialization_io, "write_and_inspect", consume
    )

    actual = materialize_panel.write_panel(
        spec,
        layout=StorageLayout(project_root=tmp_path),
        source_id=SourceId("csmar"),
        market_bars_data_id=MARKET_BARS_DATA_ID,
        time_window=TimeWindow("2014-01-02", "2014-01-03"),
        policy=active_runtime_resource_policy(),
        sessions=tuple(window.core_sessions[0] for window in windows),
        written={},
    )

    assert actual is result
    assert read_observed == [set(), set()]
    assert yield_observed == [(set(), True), (set(), True)]


def test_annotation_window_producer_releases_previous_locals_before_next_compute(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = processed_catalog().get("execution")
    contract = (
        materialize_annotations.materialization_validation.annotation_input_contract(
            spec
        )
    )
    panel = pd.DataFrame({"timestamp": pd.to_datetime(["2014-01-02"])})
    windows = tuple(
        SimpleNamespace(read=None, core_sessions=(pd.Timestamp(day),))
        for day in ("2014-01-02", "2014-01-03")
    )
    transition_observed: list[set[str]] = []
    yield_observed: list[tuple[set[str], bool]] = []
    result = object()

    def derive(current: pd.DataFrame, *, data_ids: tuple[str, ...]):
        producer = inspect.currentframe().f_back
        assert producer is not None
        transition_observed.append(
            {"outputs", "output"}.intersection(producer.f_locals)
        )
        return {data_ids[0]: current.copy()}

    def consume(_spec, frames, **_kwargs):  # noqa: ANN001, ANN003
        iterator = iter(frames)
        for _ in windows:
            yielded = next(iterator)
            producer_locals = inspect.getgeneratorlocals(iterator)
            yield_observed.append(
                (
                    {"processed_inputs", "panel", "outputs"}.intersection(
                        producer_locals
                    ),
                    producer_locals.get("output") is yielded,
                )
            )
        with pytest.raises(StopIteration):
            next(iterator)
        return result

    monkeypatch.setattr(
        materialize_annotations, "inspect_dataset", lambda _root: object()
    )
    monkeypatch.setattr(
        materialize_annotations.materialization_planning,
        "planned_session_windows",
        lambda *_args, **_kwargs: windows,
    )
    monkeypatch.setattr(
        materialize_annotations.materialization_io,
        "read_processed_inputs_for_spec",
        lambda *_args, **_kwargs: {contract.primary_input_data_id: panel.copy()},
    )
    monkeypatch.setattr(
        materialize_annotations.materialization_validation,
        "validate_processed_input_contract",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        materialize_annotations.annotations, "derive_annotations", derive
    )
    monkeypatch.setattr(
        materialize_annotations.materialization_planning,
        "filter_output_sessions",
        lambda output, _sessions: output,
    )
    monkeypatch.setattr(
        materialize_annotations.materialization_planning,
        "require_known_materialization_payload_bound",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        materialize_annotations.materialization_io,
        "write_and_inspect",
        consume,
    )

    actual = materialize_annotations.write_annotations(
        spec,
        layout=StorageLayout(project_root=tmp_path),
        time_window=TimeWindow("2014-01-02", "2014-01-03"),
        policy=active_runtime_resource_policy(),
        sessions=tuple(window.core_sessions[0] for window in windows),
        written={},
    )

    assert actual is result
    assert transition_observed == [set(), set()]
    assert yield_observed == [(set(), True), (set(), True)]


def _catalog_with_processed_inputs(
    target_data_id: str,
    processed_input_data_ids: tuple[str, ...],
) -> ProcessedDataCatalog:
    catalog = ProcessedDataCatalog()
    for spec in processed_catalog().list():
        if spec.data_id == target_data_id:
            spec = replace(
                spec,
                processed_input_data_ids=processed_input_data_ids,
            )
        catalog.register(spec)
    return catalog


def _catalog_with_one_step_forward_return_only() -> ProcessedDataCatalog:
    catalog = ProcessedDataCatalog()
    for spec in processed_catalog().list():
        if spec.data_id == "forward_returns":
            column_roles = {
                "market_id": ColumnRole.KEY,
                "asset_id": ColumnRole.KEY,
                "timestamp": ColumnRole.TIMESTAMP,
                FORWARD_RETURN_1D_COLUMN: ColumnRole.LABEL,
            }
            spec = replace(spec, column_roles=column_roles)
        catalog.register(spec)
    return catalog


def _catalog_with_momentum_20d_only() -> ProcessedDataCatalog:
    catalog = ProcessedDataCatalog()
    for spec in processed_catalog().list():
        if spec.data_id == "price_volume":
            column_roles = {
                "market_id": ColumnRole.KEY,
                "asset_id": ColumnRole.KEY,
                "timestamp": ColumnRole.TIMESTAMP,
                MOMENTUM_20D_COLUMN: ColumnRole.FACTOR,
            }
            spec = replace(spec, column_roles=column_roles)
        catalog.register(spec)
    return catalog


def _factor_catalog_with_families(families: tuple[str, ...]) -> FactorCatalog:
    catalog = FactorCatalog()
    definitions = materialize_factors.factor_definitions.factor_definitions()
    for index, definition in enumerate(definitions):
        family = families[min(index, len(families) - 1)]
        catalog.register(replace(definition, family=family))
    return catalog


def _label_catalog_with_families(families: tuple[str, ...]) -> LabelCatalog:
    catalog = LabelCatalog()
    definitions = materialize_labels.label_definitions.label_definitions()
    for index, definition in enumerate(definitions):
        family = families[min(index, len(families) - 1)]
        catalog.register(replace(definition, family=family))
    return catalog

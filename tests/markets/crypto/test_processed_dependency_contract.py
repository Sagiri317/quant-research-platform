from __future__ import annotations

import inspect
from contextlib import ExitStack, contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from quant_research_platform.contracts.ids import SourceId
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.implementations.crypto.processed import (
    materialization_resources,
    materialization_validation,
    materialize_factors,
    materialize_labels,
    materialize_panel,
    materializer,
)
from quant_research_platform.markets.implementations.crypto.processed.catalog import (
    processed_catalog,
)
from quant_research_platform.runtime.resources import (
    RuntimeResourcePolicy,
    active_runtime_resource_policy,
    runtime_resource_policy_scope,
)
from quant_research_platform.storage._dataset_key_tracking import ExactPrimaryKeyIndex
from quant_research_platform.storage._dataset_reading import open_dataset_session
from quant_research_platform.storage._dataset_types import DatasetInspection
from quant_research_platform.storage.datasets import write_dataset
from quant_research_platform.storage.layout import StorageLayout

RESOURCE_POLICY = RuntimeResourcePolicy.from_environment(
    available_memory_bytes=1024**3,
    environ={},
)


def test_input_dataset_identities_distinguish_source_and_manifest() -> None:
    first = _inspection(
        "market_bars_1m",
        spec_extra={"source_id": "source_a"},
        manifest_extra={"row_count": 3},
    )
    other_source = _inspection(
        "market_bars_1m",
        spec_extra={"source_id": "source_b"},
        manifest_extra={"row_count": 3},
    )
    changed_manifest = _inspection(
        "market_bars_1m",
        spec_extra={"source_id": "source_a"},
        manifest_extra={"row_count": 4},
    )

    first_identity = materialization_validation.input_dataset_identities(
        {"market_bars_1m": first}
    )
    other_source_identity = materialization_validation.input_dataset_identities(
        {"market_bars_1m": other_source}
    )
    changed_manifest_identity = materialization_validation.input_dataset_identities(
        {"market_bars_1m": changed_manifest}
    )

    assert first_identity != other_source_identity
    assert first_identity != changed_manifest_identity
    assert set(first_identity["market_bars_1m"]) == {
        "data_id",
        "spec_hash",
        "manifest_hash",
    }


@pytest.mark.parametrize(
    ("module", "writer_name", "data_id"),
    (
        (materialize_factors, "write_factors", "microstructure"),
        (materialize_labels, "write_labels", "forward_returns"),
    ),
)
def test_panel_dependent_writer_rejects_policy_mismatch_before_input_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    module,
    writer_name: str,
    data_id: str,
) -> None:
    session_policy = RuntimeResourcePolicy.from_environment(
        available_memory_bytes=1024**3,
        environ={},
    )
    explicit_policy = RuntimeResourcePolicy.from_environment(
        available_memory_bytes=1024**3,
        environ={},
    )
    panel_session = SimpleNamespace(resource_policy=session_policy)
    monkeypatch.setattr(
        materialization_validation,
        "input_dataset_identities",
        lambda *_args, **_kwargs: pytest.fail("input identity reached"),
    )
    monkeypatch.setattr(
        module,
        "write_dataset",
        lambda *_args, **_kwargs: pytest.fail("write reached"),
    )

    with pytest.raises(RuntimeError, match="^runtime_resource_policy_conflict$"):
        getattr(module, writer_name)(
            processed_catalog().get(data_id),
            layout=StorageLayout(project_root=tmp_path),
            panel_session=panel_session,
            time_window=TimeWindow("2026-07-06", "2026-07-06"),
            resource_policy=explicit_policy,
        )


@pytest.mark.parametrize(
    ("module", "iterator_name", "data_id"),
    (
        (materialize_factors, "_iter_factor_chunks", "microstructure"),
        (materialize_labels, "_iter_label_chunks", "forward_returns"),
    ),
)
def test_panel_dependent_iterator_rejects_policy_mismatch_before_selection(
    monkeypatch: pytest.MonkeyPatch,
    module,
    iterator_name: str,
    data_id: str,
) -> None:
    session_policy = RuntimeResourcePolicy.from_environment(
        available_memory_bytes=1024**3,
        environ={},
    )
    explicit_policy = RuntimeResourcePolicy.from_environment(
        available_memory_bytes=1024**3,
        environ={},
    )
    panel_session = SimpleNamespace(resource_policy=session_policy)
    monkeypatch.setattr(
        materialization_validation,
        "crypto_asset_session_views",
        lambda *_args, **_kwargs: pytest.fail("selection reached"),
    )

    iterator = getattr(module, iterator_name)(
        processed_catalog().get(data_id),
        panel_session=panel_session,
        time_window=TimeWindow("2026-07-06", "2026-07-06"),
        resource_policy=explicit_policy,
    )
    with pytest.raises(RuntimeError, match="^runtime_resource_policy_conflict$"):
        next(iterator)


def test_panel_iterator_rejects_any_session_policy_mismatch_before_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = processed_catalog().get("market_panel_1m")
    explicit_policy = RuntimeResourcePolicy.from_environment(
        available_memory_bytes=1024**3,
        environ={},
    )
    other_policy = RuntimeResourcePolicy.from_environment(
        available_memory_bytes=1024**3,
        environ={},
    )
    data_ids = (
        materialize_panel.source_ids.market_bars_source_data_id(spec),
        materialize_panel.source_ids.aggregate_trades_source_data_id(spec),
        materialize_panel.source_ids.depth_levels_source_data_id(spec),
    )
    sessions = {
        data_id: SimpleNamespace(
            resource_policy=other_policy if index == 1 else explicit_policy
        )
        for index, data_id in enumerate(data_ids)
    }
    monkeypatch.setattr(
        materialization_validation,
        "crypto_asset_session_views",
        lambda *_args, **_kwargs: pytest.fail("selection reached"),
    )

    iterator = materialize_panel._iter_panel_chunks(
        spec,
        input_sessions=sessions,
        time_window=TimeWindow("2026-07-06", "2026-07-06"),
        resource_policy=explicit_policy,
    )
    with pytest.raises(RuntimeError, match="^runtime_resource_policy_conflict$"):
        next(iterator)


@pytest.mark.parametrize(
    ("owner", "mismatch"),
    (
        ("panel", "receipt"),
        ("panel", "bar"),
        ("dependent", "receipt"),
        ("dependent", "panel"),
        ("existing", "panel"),
        ("existing", "bar"),
    ),
)
def test_validation_rejects_session_policy_mismatch_before_identity_or_traversal(
    monkeypatch: pytest.MonkeyPatch,
    owner: str,
    mismatch: str,
) -> None:
    policy = RuntimeResourcePolicy.from_environment(
        available_memory_bytes=1024**3,
        environ={},
    )
    other_policy = RuntimeResourcePolicy.from_environment(
        available_memory_bytes=1024**3,
        environ={},
    )

    def session(name: str) -> SimpleNamespace:
        return SimpleNamespace(
            name=name,
            resource_policy=other_policy if mismatch == name else policy,
            manifest={},
        )

    receipt = SimpleNamespace(session=session("receipt"))
    panel_session = session("panel")
    bar_session = session("bar")
    monkeypatch.setattr(
        materialization_validation,
        "validate_materialization_input_identities",
        lambda *_args, **_kwargs: pytest.fail("identity validation reached"),
    )
    monkeypatch.setattr(
        materialization_validation,
        "iter_dataset_chunks",
        lambda *_args, **_kwargs: pytest.fail("traversal reached"),
    )

    with pytest.raises(RuntimeError, match="^runtime_resource_policy_conflict$"):
        if owner == "panel":
            materialization_validation.validate_panel_materialization(
                receipt,
                bar_session=bar_session,
                spec=SimpleNamespace(primary_key=("value",)),
                expected_input_identities={},
                resource_policy=policy,
            )
        elif owner == "dependent":
            materialization_validation.validate_panel_dependent_materialization(
                receipt,
                panel_session=panel_session,
                spec=SimpleNamespace(primary_key=("value",)),
                time_window=TimeWindow("2026-07-06", "2026-07-06"),
                expected_input_identities={},
                resource_policy=policy,
            )
        else:
            materialization_validation.validate_existing_panel_materialization(
                panel_session,
                bar_session=bar_session,
                spec=SimpleNamespace(primary_key=("value",)),
                expected_input_identities={},
                resource_policy=policy,
            )
def test_materialization_input_identity_validation_is_exact() -> None:
    source = _inspection("market_bars_1m")
    expected = materialization_validation.input_dataset_identities(
        {"market_bars_1m": source}
    )
    target = _inspection(
        "market_panel_1m",
        manifest_extra={"input_dataset_identities": expected},
    )

    materialization_validation.validate_materialization_input_identities(
        target.manifest,
        expected=expected,
        error="crypto_panel_input_identity_mismatch",
    )

    with pytest.raises(ValueError, match="crypto_panel_input_identity_mismatch"):
        materialization_validation.validate_materialization_input_identities(
            _inspection("market_panel_1m").manifest,
            expected=expected,
            error="crypto_panel_input_identity_mismatch",
        )


@pytest.mark.parametrize(
    "mutation",
    ["missing_asset", "missing_middle", "extra", "market_mismatch", "duplicate"],
)
def test_exact_dependency_key_coverage_rejects_any_full_key_delta(
    mutation: str,
    tmp_path: Path,
) -> None:
    expected = _key_frame()
    actual = expected.copy()
    if mutation == "missing_asset":
        actual = actual.loc[actual["asset_id"].ne("asset_b")].copy()
    elif mutation == "missing_middle":
        actual = actual.loc[
            ~(
                actual["asset_id"].eq("asset_a")
                & actual["timestamp"].eq(pd.Timestamp("2026-07-06 00:01"))
            )
        ].copy()
    elif mutation == "extra":
        actual = pd.concat(
            [
                actual,
                actual.iloc[[0]].assign(timestamp=pd.Timestamp("2026-07-06 00:03")),
            ],
            ignore_index=True,
        )
    elif mutation == "market_mismatch":
        actual.loc[0, "market_id"] = "other_market"
    else:
        actual = pd.concat([actual, actual.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="invalid_crypto_dependency_key_coverage"):
        _validate_receipt_key_coverage(
            iter((expected.iloc[:2].copy(), expected.iloc[2:].copy())),
            iter((actual.iloc[:3].copy(), actual.iloc[3:].copy())),
            primary_key=("market_id", "asset_id", "timestamp"),
            error="invalid_crypto_dependency_key_coverage",
            resource_policy=RESOURCE_POLICY,
            scratch_parent=tmp_path,
        )


def test_exact_dependency_key_coverage_accepts_equal_bounded_streams(
    tmp_path: Path,
) -> None:
    expected = _key_frame()

    _validate_receipt_key_coverage(
        iter((expected.iloc[:1], expected.iloc[1:4], expected.iloc[4:])),
        iter((expected.iloc[:2], expected.iloc[2:5], expected.iloc[5:])),
        primary_key=("market_id", "asset_id", "timestamp"),
        error="invalid_crypto_dependency_key_coverage",
        resource_policy=RESOURCE_POLICY,
        scratch_parent=tmp_path,
    )


@pytest.mark.parametrize("chunk_size", [1, 2, 5])
def test_exact_dependency_key_coverage_is_order_and_chunk_independent(
    chunk_size: int,
    tmp_path: Path,
) -> None:
    expected = _key_frame()
    actual = expected.iloc[[5, 1, 3, 0, 4, 2]].reset_index(drop=True)

    _validate_receipt_key_coverage(
        (
            expected.iloc[start : start + chunk_size].copy()
            for start in range(0, len(expected), chunk_size)
        ),
        (
            actual.iloc[start : start + chunk_size].copy()
            for start in range(0, len(actual), chunk_size)
        ),
        primary_key=("market_id", "asset_id", "timestamp"),
        error="invalid_crypto_dependency_key_coverage",
        resource_policy=RESOURCE_POLICY,
        scratch_parent=tmp_path,
    )


def test_panel_key_coverage_accepts_non_order_preserving_asset_mapping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mapping = {"SOURCE_A": "asset_z", "SOURCE_B": "asset_a"}
    monkeypatch.setattr(
        materialization_validation,
        "canonical_crypto_asset_id",
        mapping.__getitem__,
    )
    bars = pd.DataFrame(
        {
            "source_asset_id": ["SOURCE_A", "SOURCE_B"],
            "timestamp": [pd.Timestamp("2026-07-06")] * 2,
        }
    )
    primary_key = ("market_id", "asset_id", "timestamp")
    expected = materialization_validation._panel_primary_keys(
        bars,
        primary_key=primary_key,
    )
    actual = expected.iloc[::-1].reset_index(drop=True)

    _validate_receipt_key_coverage(
        iter((expected,)),
        iter((actual,)),
        primary_key=primary_key,
        error="invalid_crypto_dependency_key_coverage",
        resource_policy=RESOURCE_POLICY,
        scratch_parent=tmp_path,
    )

    assert list(tmp_path.iterdir()) == []


def test_factor_carry_processing_is_linear_in_asset_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asset_count = 64
    rows_per_asset = 6
    chunks = tuple(
        _panel_frame(start=0, minutes=rows_per_asset).assign(
            asset_id=f"asset_{asset_index:04d}"
        )
        for asset_index in range(asset_count)
    )
    processed_rows = 0

    def record_rows(frame: pd.DataFrame, **_kwargs) -> pd.DataFrame:
        nonlocal processed_rows
        processed_rows += len(frame)
        output = frame.copy()
        for column in processed_catalog().get("microstructure").required_columns:
            if column not in output:
                output[column] = 0.0
        return output

    def read_asset_chunks(context, **_kwargs):
        filters = dict(
            (column, value)
            for column, operator, value in context.row_filters
            if operator == "=="
        )
        if not filters:
            return iter(chunks)
        return iter(
            tuple(
                chunk.loc[
                    chunk["market_id"].eq(filters["market_id"])
                    & chunk["asset_id"].eq(filters["asset_id"])
                ].copy()
                for chunk in chunks
                if chunk["asset_id"].eq(filters["asset_id"]).any()
            )
        )

    monkeypatch.setattr(
        materialize_factors,
        "iter_dataset_chunks",
        read_asset_chunks,
    )
    _patch_mock_group_plan(
        monkeypatch,
        tuple(("crypto", f"asset_{index:04d}") for index in range(asset_count)),
    )
    monkeypatch.setattr(
        materialize_factors,
        "calculate_microstructure_factors",
        record_rows,
    )
    spec = processed_catalog().get("microstructure")
    policy = RuntimeResourcePolicy.from_environment(
        available_memory_bytes=1024**3,
        environ={},
    )

    list(
        materialize_factors._iter_factor_chunks(
            spec,
            panel_session=SimpleNamespace(
                resource_policy=policy,
                verify=lambda: None,
            ),
            time_window=TimeWindow("2026-07-06 00:00", "2026-07-06 00:05"),
            resource_policy=policy,
        )
    )

    assert processed_rows <= asset_count * (
        rows_per_asset + materialize_factors.MICROSTRUCTURE_ROLLING_CARRY_BARS
    )


@pytest.mark.parametrize(
    ("module", "output_data_id", "minutes"),
    [
        (materialize_factors, "microstructure", 6),
        (materialize_labels, "forward_returns", 61),
    ],
)
def test_panel_dependents_reject_mixed_asset_physical_parts_before_repeated_reads(
    tmp_path: Path,
    module,
    output_data_id: str,
    minutes: int,
) -> None:
    catalog = processed_catalog()
    panel_spec = catalog.get("market_panel_1m")
    panel_root = StorageLayout(tmp_path).processed_data_root(
        panel_spec.market_id,
        panel_spec.kind,
        panel_spec.data_id,
    )
    mixed = pd.concat(
        tuple(
            _panel_frame(start=0, minutes=minutes).assign(
                asset_id=f"asset_{index:04d}"
            )
            for index in range(8)
        ),
        ignore_index=True,
    )
    write_dataset(panel_root, mixed, spec=panel_spec)
    iterator_name = (
        "_iter_factor_chunks"
        if output_data_id == "microstructure"
        else "_iter_label_chunks"
    )
    iterator = getattr(module, iterator_name)
    with open_dataset_session(
        panel_root,
        columns=None,
        time_window=None,
    ) as panel_session:
        with pytest.raises(
            ValueError,
            match="invalid_crypto_panel_physical_asset_grouping",
        ):
            list(
                iterator(
                    catalog.get(output_data_id),
                    panel_session=panel_session,
                    time_window=TimeWindow(
                        pd.Timestamp("2026-07-06 00:00:00"),
                        pd.Timestamp("2026-07-06 00:00:00")
                        + pd.Timedelta(minutes=minutes - 1),
                    ),
                    resource_policy=panel_session.resource_policy,
                )
            )


def test_canonical_asset_normalization_is_linear(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asset_count = 200
    work = {"len_calls": 0}

    class CountingString(str):
        def __len__(self) -> int:
            work["len_calls"] += 1
            return super().__len__()

    keys = tuple(
        (CountingString(f"source_{index:04d}"),)
        for index in range(asset_count)
    )
    monkeypatch.setattr(
        materialize_panel,
        "canonical_crypto_asset_id",
        lambda value: CountingString(f"canonical_{value}"),
    )

    observed = materialize_panel._canonical_source_assets(keys)

    assert len(observed) == asset_count
    assert work["len_calls"] <= asset_count * 5


@pytest.mark.parametrize(
    ("module", "iterator_name", "data_id"),
    [
        (materialize_factors, "_iter_factor_chunks", "microstructure"),
        (materialize_labels, "_iter_label_chunks", "forward_returns"),
    ],
)
def test_panel_dependencies_remain_asset_major_across_year_partitions(
    tmp_path: Path,
    module,
    iterator_name: str,
    data_id: str,
) -> None:
    timestamps = pd.date_range("2025-12-31 23:55", periods=10, freq="min")
    asset_a = _panel_frame(start=0, minutes=10).assign(
        asset_id="asset_a",
        timestamp=timestamps,
    )
    asset_b = _panel_frame(start=0, minutes=10).assign(
        asset_id="asset_b",
        timestamp=timestamps,
    )
    panel_root = StorageLayout(project_root=tmp_path).processed_data_root(
        "crypto",
        processed_catalog().get("market_panel_1m").kind,
        "market_panel_1m",
    )
    write_dataset(
        panel_root,
        iter((asset_a, asset_b)),
        spec=processed_catalog().get("market_panel_1m"),
    )

    with open_dataset_session(
        panel_root,
        columns=None,
        time_window=None,
    ) as panel_session, runtime_resource_policy_scope(panel_session.resource_policy):
        outputs = list(
            getattr(module, iterator_name)(
                processed_catalog().get(data_id),
                panel_session=panel_session,
                time_window=TimeWindow(
                    "2025-12-31 23:55",
                    "2026-01-01 00:04",
                ),
                resource_policy=panel_session.resource_policy,
            )
        )

    assert outputs
    observed = pd.concat(outputs, ignore_index=True).loc[
        :, ["market_id", "asset_id", "timestamp"]
    ]
    pd.testing.assert_frame_equal(
        observed,
        observed.sort_values(
            ["market_id", "asset_id", "timestamp"],
            kind="mergesort",
        ).reset_index(drop=True),
    )


@pytest.mark.parametrize("empty_side", ["expected", "actual"])
def test_exact_dependency_key_coverage_rejects_one_empty_stream(
    empty_side: str,
    tmp_path: Path,
) -> None:
    populated = iter((_key_frame(),))
    expected = iter(()) if empty_side == "expected" else populated
    actual = iter(()) if empty_side == "actual" else populated

    with pytest.raises(ValueError, match="invalid_crypto_dependency_key_coverage"):
        _validate_receipt_key_coverage(
            expected,
            actual,
            primary_key=("market_id", "asset_id", "timestamp"),
            error="invalid_crypto_dependency_key_coverage",
            resource_policy=RESOURCE_POLICY,
            scratch_parent=tmp_path,
        )


def test_exact_dependency_key_coverage_accepts_two_empty_streams(
    tmp_path: Path,
) -> None:
    _validate_receipt_key_coverage(
        iter(()),
        iter(()),
        primary_key=("market_id", "asset_id", "timestamp"),
        error="invalid_crypto_dependency_key_coverage",
        resource_policy=RESOURCE_POLICY,
        scratch_parent=tmp_path,
    )


@pytest.mark.parametrize("stream_name", ["expected", "actual"])
def test_exact_dependency_key_coverage_rejects_invalid_individual_streams(
    stream_name: str,
    tmp_path: Path,
) -> None:
    baseline = _key_frame()
    mutated = iter((baseline.iloc[:2], baseline.iloc[1:].copy()))
    expected = mutated if stream_name == "expected" else iter((baseline,))
    actual = mutated if stream_name == "actual" else iter((baseline,))

    with pytest.raises(ValueError, match="invalid_crypto_dependency_key_coverage"):
        _validate_receipt_key_coverage(
            expected,
            actual,
            primary_key=("market_id", "asset_id", "timestamp"),
            error="invalid_crypto_dependency_key_coverage",
            resource_policy=RESOURCE_POLICY,
            scratch_parent=tmp_path,
        )


def test_exact_dependency_validator_has_no_full_key_collectors() -> None:
    source = inspect.getsource(
        materialization_validation._validate_receipt_key_coverage
    )

    assert "set(" not in source
    assert "pd.concat" not in source


def test_coverage_scratch_follows_validation_lifecycle_owner() -> None:
    validation_source = inspect.getsource(
        materialization_validation.validate_panel_materialization
    ) + inspect.getsource(
        materialization_validation.validate_panel_dependent_materialization
    )
    writer_source = "".join(
        inspect.getsource(module)
        for module in (materialize_panel, materialize_factors, materialize_labels)
    )

    assert validation_source.count("receipt.session") >= 2
    assert validation_source.count("receipt.exact_primary_key_index") == 0
    assert writer_source.count(
        "def validate_staged_dataset(receipt: DatasetWriteReceipt)"
    ) == 3
    assert "inspect_dataset" not in writer_source


def test_dependency_chunk_order_rejects_unsorted_and_overlap() -> None:
    state = materialization_validation.OrderedDependencyState()
    first = _key_frame().iloc[[2, 0, 1, 5, 3, 4]].copy()

    with pytest.raises(ValueError, match="out_of_order_crypto_factor_dependency"):
        materialization_validation.validated_ordered_dependency_chunk(
            first,
            state=state,
            error="out_of_order_crypto_factor_dependency",
        )

    ordered = first.sort_values(
        ["market_id", "asset_id", "timestamp"], kind="mergesort"
    ).reset_index(drop=True)
    result = materialization_validation.validated_ordered_dependency_chunk(
        ordered,
        state=state,
        error="out_of_order_crypto_factor_dependency",
    )
    pd.testing.assert_frame_equal(result, ordered, check_dtype=False)
    assert state.last_key == (
        "crypto",
        "asset_b",
        pd.Timestamp("2026-07-06 00:02"),
    )

    with pytest.raises(ValueError, match="out_of_order_crypto_factor_dependency"):
        materialization_validation.validated_ordered_dependency_chunk(
            _key_frame().iloc[[2]].copy(),
            state=state,
            error="out_of_order_crypto_factor_dependency",
        )


def test_existing_dependency_reuse_passes_same_resource_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = RuntimeResourcePolicy.from_environment(
        available_memory_bytes=4096,
        environ={},
    )
    inspection = _inspection("market_panel_1m")
    observed: list[RuntimeResourcePolicy] = []
    monkeypatch.setattr(
        materializer,
        "open_dataset_session",
        lambda *_args, **_kwargs: nullcontext(
            SimpleNamespace(verify=lambda: None)
        ),
    )
    monkeypatch.setattr(
        materializer,
        "inspect_dataset_session",
        lambda *_args, **_kwargs: inspection,
    )
    monkeypatch.setattr(
        materialization_validation,
        "input_dataset_identities",
        lambda _sessions: {},
    )

    def record_validation(*_args, resource_policy, **_kwargs) -> None:
        observed.append(resource_policy)

    monkeypatch.setattr(
        materialization_validation,
        "validate_existing_panel_materialization",
        record_validation,
    )

    with ExitStack() as session_stack:
        result, panel_session = materializer._inspect_existing_dependency(
            StorageLayout(project_root=tmp_path),
            processed_catalog().get("market_panel_1m"),
            source_id=SourceId("synthetic_crypto"),
            time_window=TimeWindow("2026-07-06", "2026-07-06"),
            resource_policy=policy,
            session_stack=session_stack,
        )

        assert result is inspection
        panel_session.verify()
    assert observed == [policy]


def test_crypto_disk_guard_rechecks_between_chunks_and_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[tuple[Path, int]] = []
    monkeypatch.setattr(
        materialization_resources,
        "require_free_disk_headroom",
        lambda path, *, reserve_bytes: observed.append((Path(path), reserve_bytes)),
    )
    frames = (
        pd.DataFrame({"value": [1]}),
        pd.DataFrame({"value": [2, 3]}),
    )

    result = list(
        materialization_resources.iter_disk_guarded_chunks(
            frames,
            scratch_parent=tmp_path,
        )
    )

    assert [len(frame) for frame in result] == [1, 2]
    assert [path for path, _ in observed] == [tmp_path, tmp_path]
    assert [reserve for _, reserve in observed] == [140, 148]


def test_crypto_direct_dataset_writers_cannot_bypass_disk_guard() -> None:
    for module, function_name in (
        (materialize_panel, "write_panel"),
        (materialize_factors, "write_factors"),
        (materialize_labels, "write_labels"),
    ):
        source = inspect.getsource(getattr(module, function_name))
        assert "iter_disk_guarded_chunks(" in source


@pytest.mark.parametrize(
    ("module", "function_name", "iterator_name", "data_id"),
    [
        (
            materialize_panel,
            "write_panel",
            "_iter_panel_chunks",
            "market_panel_1m",
        ),
        (
            materialize_factors,
            "write_factors",
            "_iter_factor_chunks",
            "microstructure",
        ),
        (
            materialize_labels,
            "write_labels",
            "_iter_label_chunks",
            "forward_returns",
        ),
    ],
)
def test_crypto_direct_dataset_writers_keep_one_policy_through_inspection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    module,
    function_name: str,
    iterator_name: str,
    data_id: str,
) -> None:
    policy = RuntimeResourcePolicy.from_environment(
        available_memory_bytes=16 * 1024 * 1024,
        environ={},
    )
    inspection = _inspection(data_id)
    observed: list[RuntimeResourcePolicy] = []

    def capture_input_identities(_sessions):
        observed.append(active_runtime_resource_policy())
        return {}

    def capture_write(
        *_args,
        staged_validator,
        resource_policy,
        **_kwargs,
    ) -> DatasetInspection:
        assert resource_policy is policy
        observed.append(active_runtime_resource_policy())
        staged_validator(SimpleNamespace())
        return inspection

    monkeypatch.setattr(
        materialization_validation,
        "input_dataset_identities",
        capture_input_identities,
    )
    monkeypatch.setattr(
        materialization_validation,
        "validate_panel_materialization",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        materialization_validation,
        "validate_panel_dependent_materialization",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        materialization_resources,
        "iter_disk_guarded_chunks",
        lambda chunks, **_kwargs: chunks,
    )
    monkeypatch.setattr(module, iterator_name, lambda *_args, **_kwargs: iter(()))
    monkeypatch.setattr(module, "write_dataset", capture_write)
    kwargs = {
        "layout": StorageLayout(project_root=tmp_path),
        "time_window": TimeWindow("2026-07-06", "2026-07-06"),
        "resource_policy": policy,
    }
    if module is materialize_panel:
        kwargs["source_id"] = SourceId("synthetic_crypto")
        monkeypatch.setattr(
            module,
            "open_dataset_session",
            lambda *_args, **_kwargs: nullcontext(
                SimpleNamespace(
                    resource_policy=policy,
                    verify=lambda: None,
                )
            ),
        )
    else:
        kwargs["panel_session"] = SimpleNamespace(
            resource_policy=policy,
            verify=lambda: None,
        )

    result = getattr(module, function_name)(
        processed_catalog().get(data_id),
        **kwargs,
    )

    assert result is inspection
    assert observed == [policy, policy]


def test_crypto_materializers_use_storage_physical_iterator_without_split_wrapper(
) -> None:
    for module in (materialize_factors, materialize_labels):
        source = inspect.getsource(module)
        assert "iter_dataset_chunks(" in source
        assert "read_dataset(" not in source
        assert "def _frame_bytes(" not in source
        assert "pandas_frame_payload_bytes(" in source


@pytest.mark.parametrize(
    ("module_name", "error"),
    [
        ("factor", "out_of_order_crypto_factor_dependency"),
        ("label", "out_of_order_crypto_label_dependency"),
    ],
)
@pytest.mark.parametrize("second_start", [0, 4], ids=["reversed", "overlap"])
def test_factor_and_label_materializers_reject_non_monotonic_chunks(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    error: str,
    second_start: int,
) -> None:
    if second_start == 0:
        first = _panel_frame(start=5, minutes=5)
    else:
        first = _panel_frame(start=0, minutes=5)
    second = _panel_frame(start=second_start, minutes=5)
    spec = processed_catalog().get(
        "microstructure" if module_name == "factor" else "forward_returns"
    )
    module = materialize_factors if module_name == "factor" else materialize_labels
    monkeypatch.setattr(
        module,
        "iter_dataset_chunks",
        lambda *_args, **_kwargs: iter((first, second)),
    )
    _patch_mock_group_plan(monkeypatch, (("crypto", "btc_usdt"),))
    policy = RuntimeResourcePolicy.from_environment(
        available_memory_bytes=1024**3,
        environ={},
    )
    iterator = (
        module._iter_factor_chunks(
            spec,
            panel_session=SimpleNamespace(
                resource_policy=policy,
                verify=lambda: None,
            ),
            time_window=TimeWindow("2026-07-06 00:00", "2026-07-06 00:10"),
            resource_policy=policy,
        )
        if module_name == "factor"
        else module._iter_label_chunks(
            spec,
            panel_session=SimpleNamespace(
                resource_policy=policy,
                verify=lambda: None,
            ),
            time_window=TimeWindow("2026-07-06 00:00", "2026-07-06 00:10"),
            resource_policy=policy,
        )
    )

    with pytest.raises(ValueError, match=error):
        list(iterator)


@pytest.mark.parametrize(
    "module",
    [
        materialization_validation,
        materializer,
        materialize_factors,
        materialize_labels,
    ],
)
def test_crypto_materialization_hot_path_does_not_fingerprint_content(module) -> None:
    assert "fingerprint_dataset" not in inspect.getsource(module)


def _inspection(
    data_id: str,
    *,
    spec_extra: dict[str, object] | None = None,
    manifest_extra: dict[str, object] | None = None,
) -> DatasetInspection:
    spec = {"schema_id": "qrp-dataset-spec", "data_id": data_id}
    manifest = {"schema_id": "qrp-dataset-manifest", "data_id": data_id}
    spec.update(spec_extra or {})
    manifest.update(manifest_extra or {})
    return DatasetInspection(
        root=Path(data_id),
        data_root=Path(data_id) / "data",
        spec=spec,
        columns=("market_id", "asset_id", "timestamp"),
        schema_hash="schema",
        file_count=1,
        row_count=int(manifest.get("row_count", 0)),
        partitions=("year=2026",),
        partition_row_counts={"year=2026": int(manifest.get("row_count", 0))},
        date_min="2026-07-06",
        date_max="2026-07-06",
        manifest=manifest,
    )


def _key_frame() -> pd.DataFrame:
    timestamps = pd.date_range("2026-07-06", periods=3, freq="min")
    return pd.DataFrame(
        {
            "market_id": ["crypto"] * 6,
            "asset_id": ["asset_a"] * 3 + ["asset_b"] * 3,
            "timestamp": [*timestamps, *timestamps],
        }
    )


def _validate_receipt_key_coverage(
    expected,
    actual,
    *,
    primary_key: tuple[str, ...],
    error: str,
    resource_policy: RuntimeResourcePolicy,
    scratch_parent: Path,
) -> None:
    with ExactPrimaryKeyIndex(
        primary_key=primary_key,
        timestamp_columns=tuple(
            column for column in primary_key if column == "timestamp"
        ),
        scratch_parent=scratch_parent,
    ) as index:
        for frame in actual:
            duplicates = index.add_frame(
                frame.loc[:, list(primary_key)],
                namespace="dataset",
            )
            if duplicates:
                raise ValueError(error)
        receipt = SimpleNamespace(
            session=SimpleNamespace(verify=lambda: None),
            exact_primary_key_index=index,
        )
        materialization_validation._validate_receipt_key_coverage(
            expected,
            receipt,
            primary_key=primary_key,
            error=error,
            resource_policy=resource_policy,
        )


def _panel_frame(*, start: int, minutes: int) -> pd.DataFrame:
    timestamps = pd.date_range(
        "2026-07-06 00:00:00",
        periods=start + minutes,
        freq="min",
    )[start:]
    return pd.DataFrame(
        {
            "market_id": ["crypto"] * minutes,
            "asset_id": ["btc_usdt"] * minutes,
            "timestamp": timestamps,
            "open": [100.0] * minutes,
            "high": [101.0] * minutes,
            "low": [99.0] * minutes,
            "close": [100.5] * minutes,
            "volume": [10.0] * minutes,
            "quote_volume": [1005.0] * minutes,
            "trade_count": [2.0] * minutes,
            "trade_flow_observed": [True] * minutes,
            "taker_buy_volume": [4.0] * minutes,
            "taker_sell_volume": [6.0] * minutes,
            "signed_volume": [1.0] * minutes,
            "best_bid": [100.0] * minutes,
            "best_ask": [100.2] * minutes,
            "quote_observed": [True] * minutes,
            "quote_age_seconds": [1.0] * minutes,
            "bid_quantity": [1.5] * minutes,
            "ask_quantity": [1.0] * minutes,
            "bid_depth": [1.5] * minutes,
            "ask_depth": [1.0] * minutes,
            "mid_price": [100.1] * minutes,
            "eligible": [True] * minutes,
        }
    )


def _patch_mock_group_plan(
    monkeypatch: pytest.MonkeyPatch,
    keys: tuple[tuple[str, str], ...],
) -> None:
    monkeypatch.setattr(
        materialization_validation,
        "crypto_asset_session_views",
        lambda *_args, **_kwargs: _asset_views_context(
            tuple(
                (
                    key,
                    SimpleNamespace(
                        row_filters=(
                            ("market_id", "==", key[0]),
                            ("asset_id", "==", key[1]),
                        ),
                        verify=lambda: None,
                    ),
                )
                for key in keys
            )
        ),
    )


@contextmanager
def _asset_views_context(views):  # noqa: ANN001
    yield views

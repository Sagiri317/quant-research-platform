from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import quant_research_platform.research.datasets.assembler as assembler_module
from quant_research_platform.catalog.datasets import ResearchDatasetCatalog
from quant_research_platform.catalog.processed import ProcessedDataCatalog
from quant_research_platform.contracts.columns import (
    PLATFORM_ASSET_TIME_KEY,
    ColumnRole,
)
from quant_research_platform.contracts.data import (
    DataFrequency,
    DataGrain,
    ProcessedDataKind,
    ProcessedDataSpec,
)
from quant_research_platform.contracts.datasets import ResearchDatasetSpec
from quant_research_platform.contracts.features import FeatureTransformSpec
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.storage.datasets import (
    DatasetReadMode,
    inspect_dataset,
    read_dataset,
    write_dataset,
)
from tests.helpers.paths import StorageLayout
from tests.helpers.research_processed_inputs import (
    _demo_research_rows,
    make_demo_processed_catalog,
    write_demo_processed_inputs,
)
from tests.helpers.research_specs import (
    assemble_research_dataset,
    make_research_dataset_catalog,
)


def test_research_dataset_assembles_from_processed_inputs(tmp_path: Path) -> None:
    layout = StorageLayout(project_root=tmp_path)
    processed_catalog = make_demo_processed_catalog()
    dataset_catalog = make_research_dataset_catalog(
        filter_column_roles={"eligible": ColumnRole.UNIVERSE}
    )
    write_demo_processed_inputs(layout, processed_catalog)
    root = assemble_research_dataset(
        inputs={
            item.data_id: layout.processed_data_root(
                item.market_id, item.kind, item.data_id
            )
            for item in processed_catalog.list()
            if item.data_id in dataset_catalog.get("demo").input_data_ids
        },
        output_root=layout.research_dataset_root(
            dataset_catalog.get("demo").market_id, "demo"
        ),
        dataset_catalog=dataset_catalog,
        dataset_id="demo",
        time_window=TimeWindow("2014-01-02", "2014-01-03"),
    )
    result = pd.concat(
        read_dataset(
            layout.research_dataset_root(MarketId("a_share"), "demo"),
            time_window=TimeWindow("2014-01-02", "2014-01-03"),
            mode=DatasetReadMode.CHUNKED,
        ),
        ignore_index=True,
    )
    inspection = inspect_dataset(root)
    assert root == tmp_path / "data" / "research" / "a_share" / "demo"
    assert list(result.columns) == [
        "market_id",
        "asset_id",
        "timestamp",
        "momentum_20d",
        "forward_return_20d",
        "eligible",
        "can_enter",
        "can_exit",
        "market_type",
    ]
    assert len(result) == 4
    assert set(inspection.manifest["inputs"]) == {
        "demo_factors",
        "demo_labels",
        "demo_annotations",
    }


@pytest.mark.parametrize(
    "scenario", ("pv_all", "pv_expanded", "fusion_nf_core", "fusion_nf_expectations")
)
def test_non_a_share_named_scenario_keeps_input_semantics_and_requires_real_columns(
    tmp_path, scenario
):
    from dataclasses import replace

    from quant_research_platform.storage.datasets import read_dataset
    from tests.helpers.research_processed_inputs import (
        make_dummy_processed_catalog,
        write_dummy_processed_inputs,
    )
    from tests.helpers.research_specs import make_dummy_research_dataset_spec

    layout = StorageLayout(tmp_path)
    processed = make_dummy_processed_catalog()
    spec = make_dummy_research_dataset_spec(data_id=scenario)
    write_dummy_processed_inputs(layout, processed)
    root = assemble_research_dataset(
        inputs={
            item.data_id: layout.processed_data_root(
                item.market_id, item.kind, item.data_id
            )
            for item in processed.list()
            if item.data_id
            in ResearchDatasetCatalog((spec,)).get(scenario).input_data_ids
        },
        output_root=layout.research_dataset_root(
            ResearchDatasetCatalog((spec,)).get(scenario).market_id, scenario
        ),
        dataset_catalog=ResearchDatasetCatalog((spec,)),
        dataset_id=scenario,
        time_window=TimeWindow("2024-01-02", "2024-01-03"),
    )
    frame = read_dataset(root)
    assert frame.eligible.tolist() == [True, False]
    assert assembler_module._generated_semantic_columns(spec) == ()
    assert assembler_module._add_a_share_semantic_columns(frame, spec) is frame
    a_share = replace(
        spec, market_id=MarketId("a_share"), eligibility_recipe="a_share_all"
    )
    assert assembler_module._generated_semantic_columns(a_share) == ("eligible",)
    with pytest.raises(ValueError, match="^a_share_eligibility_facts_missing:"):
        assembler_module._add_a_share_semantic_columns(
            frame.assign(market_id="a_share"), a_share
        )
    missing = ProcessedDataCatalog(
        tuple(
            replace(
                item,
                column_roles={
                    key: role
                    for key, role in item.column_roles.items()
                    if key != "eligible"
                },
            )
            if str(item.data_id) == "filters"
            else item
            for item in processed.list()
        )
    )
    with pytest.raises(
        ValueError, match="^missing_research_required_columns: eligible$"
    ):
        assembler_module._validate_research_input_specs(
            spec, {item.data_id: item for item in missing.list()}
        )


@pytest.mark.parametrize("storage", ["native", "aware_base", "aware_auxiliary"])
def test_research_dataset_assembles_intraday_timestamp_window(tmp_path, storage):
    from quant_research_platform.contracts.data import DataSpec

    timestamps = pd.date_range("2026-07-06", periods=3, freq="min")
    inputs = {}
    for name, column, role in (
        ("base", "minute_factor", "factor"),
        ("auxiliary", "minute_label", "label"),
    ):
        frame = pd.DataFrame(
            {
                "asset_id": ["asset_a", "asset_b"] * 3,
                "timestamp": timestamps.repeat(2),
                column: range(6),
            }
        )
        if storage == "native" or storage == "aware_" + name:
            frame["timestamp"] = frame.timestamp.dt.tz_localize("UTC").dt.tz_convert(
                "Asia/Shanghai"
            )
        declaration = {
            "data_id": name,
            "frequency": "1m",
            "grain": "asset_timestamp",
            "primary_key": ("asset_id", "timestamp"),
            "partition_columns": ("year",),
            "column_roles": {"asset_id": "key", "timestamp": "timestamp", column: role},
        }
        if storage == "native":
            root = tmp_path / name
            write_dataset(root, frame, spec=DataSpec(**declaration))
            inputs[name] = str(root)
        else:
            root = tmp_path / (name + ".parquet")
            frame.to_parquet(root, index=False)
            inputs[name] = {"path": str(root), "spec": declaration}
    root = assembler_module.assemble(
        {
            "market_id": "crypto",
            "dataset_id": "minute_research",
            "inputs": inputs,
            "base_input_data_id": "base",
            "factor_columns": ["minute_factor"],
            "label_columns": ["minute_label"],
            "time_window": {
                "start": "2026-07-06T00:01:00",
                "end": "2026-07-06T00:02:00",
            },
            "output_root": str(tmp_path / "output"),
        }
    )
    result = read_dataset(root)
    expected = pd.DataFrame(
        {
            "asset_id": ["asset_a", "asset_b"] * 2,
            "timestamp": timestamps[1:].repeat(2).as_unit("ns"),
            "minute_factor": [2, 3, 4, 5],
            "minute_label": [2, 3, 4, 5],
        }
    )
    pd.testing.assert_frame_equal(result, expected)


@pytest.mark.parametrize("mapped", [False, True])
def test_assembled_label_semantics_survive_model_fit_without_maturity_mapping(
    tmp_path, mapped
):
    from quant_research_platform.contracts.data import DataSpec
    from quant_research_platform.contracts.models import (
        ModelSpec,
        _prediction_loss_reason,
        model_spec_payload,
    )
    from quant_research_platform.research.models.persisted import load_model
    from quant_research_platform.research.models.production import fit

    frame = pd.DataFrame(
        [
            dict(
                timestamp=time,
                asset_id=str(asset),
                x=float(asset),
                label=0.1 + 0.2 * asset,
                label_available_at=time,
                eligible=True,
                unused_label=0.4,
            )
            for time in pd.date_range("2024-01-02", periods=3, freq="min")
            for asset in range(4)
        ]
    )
    semantics = {
        "label": {"horizon_minutes": 1, "price_basis": "mid"},
        "unused_label": {"horizon_minutes": 2, "price_basis": "last"},
    }
    availability = {"label": "label_available_at"} if mapped else {}
    source = tmp_path / "source"
    write_dataset(
        source,
        frame,
        spec=DataSpec(
            data_id="source",
            frequency="1m",
            grain="asset_timestamp",
            primary_key=("asset_id", "timestamp"),
            partition_columns=("year",),
            column_roles={
                "asset_id": "key",
                "timestamp": "timestamp",
                "x": "factor",
                "label": "label",
                "label_available_at": "context",
                "eligible": "mask",
                "unused_label": "label",
            },
            column_units={"label": "decimal_return"},
            label_availability_columns=availability,
        ),
        metadata={"producer_diagnostics": {"label_semantics": semantics}},
    )
    window = {"start": "2024-01-02", "end": "2024-01-02"}
    output = assembler_module.assemble(
        {
            "market_id": "crypto",
            "dataset_id": "selected",
            "inputs": {"source": str(source)},
            "factor_columns": ["x"],
            "label_columns": ["label"],
            "context_columns": ["label_available_at"],
            "filter_columns": ["eligible"],
            "time_window": window,
            "output_root": str(tmp_path / "assembled"),
        }
    )
    saved = inspect_dataset(output)
    assert saved.spec["label_availability_columns"] == availability
    assert saved.manifest["producer_diagnostics"].get("label_semantics") == {
        "label": {"horizon_minutes": 1, "price_basis": "mid"}
    }
    assert "unused_label" not in read_dataset(output)
    model = fit(
        {
            "inputs": str(output),
            "output": str(tmp_path / "model"),
            "model": model_spec_payload(
                ModelSpec(
                    model_id="simple",
                    model_algorithm="linear_regression",
                    input_factor_columns=("x",),
                    target_column="label",
                )
            ),
            "market_id": "crypto",
            "fit_windows": [window],
            "fit_cutoff": "2024-01-03",
        }
    )
    _, facts = load_model(model)
    assert facts["target_definition"] == {
        "source_label": "label",
        "target_transform": "raw",
        "target_clip": "none",
        "label_semantics": {"horizon_minutes": 1, "price_basis": "mid"},
    }
    assert (
        _prediction_loss_reason(
            facts, "label", saved.spec, inspect_dataset(source).manifest
        )
        is None
    )


def test_assembly_keeps_nanosecond_windows_through_persisted_inputs(
    tmp_path: Path,
) -> None:
    layout = StorageLayout(tmp_path)
    roles = {
        "market_id": ColumnRole.KEY,
        "asset_id": ColumnRole.KEY,
        "timestamp": ColumnRole.TIMESTAMP,
        "factor": ColumnRole.FACTOR,
        "label": ColumnRole.LABEL,
    }
    source = ProcessedDataSpec(
        data_id="nanosecond_input",
        market_id=MarketId("crypto"),
        kind=ProcessedDataKind.PANEL,
        frequency=DataFrequency.TICK,
        grain=DataGrain.ASSET_TIMESTAMP,
        column_roles=roles,
        primary_key=PLATFORM_ASSET_TIME_KEY,
        partition_columns=("year",),
    )
    dataset = make_research_dataset_catalog(
        data_id="nanosecond_research",
        market_id="crypto",
        input_data_ids=(source.data_id,),
        base_input_data_id=source.data_id,
        factor_columns=("factor",),
        label_columns=("label",),
        filter_columns=(),
        context_columns=(),
        frequency=DataFrequency.TICK,
    )
    times = pd.Timestamp("2020-01-01") + pd.to_timedelta([1, 2], unit="ns")
    frame = pd.DataFrame(
        {
            "market_id": "crypto",
            "asset_id": "asset",
            "timestamp": times,
            "factor": [1.0, 2.0],
            "label": [3.0, 4.0],
        }
    )
    write_dataset(
        layout.processed_data_root(source.market_id, source.kind, source.data_id),
        frame,
        spec=source,
    )
    window = TimeWindow(times[0], times[-1])
    assemble_research_dataset(
        inputs={
            item.data_id: layout.processed_data_root(
                item.market_id, item.kind, item.data_id
            )
            for item in ProcessedDataCatalog((source,)).list()
            if item.data_id in dataset.get("nanosecond_research").input_data_ids
        },
        output_root=layout.research_dataset_root(
            dataset.get("nanosecond_research").market_id, "nanosecond_research"
        ),
        dataset_catalog=dataset,
        dataset_id="nanosecond_research",
        time_window=window,
    )
    result = pd.concat(
        tuple(
            read_dataset(
                layout.research_dataset_root(MarketId("crypto"), "nanosecond_research"),
                time_window=window,
                mode=DatasetReadMode.CHUNKED,
            )
        ),
        ignore_index=True,
    )
    pd.testing.assert_frame_equal(result, frame, check_exact=True)


@pytest.mark.parametrize("auxiliary_order", ["time", "asset"])
def test_research_dataset_session_complete_chunks_preserve_output(
    tmp_path: Path,
    monkeypatch,
    auxiliary_order,
) -> None:
    frame = _research_rows_for_dates(
        ["2014-01-02", "2014-01-03", "2014-01-06", "2014-01-07", "2014-01-08"]
    )
    default_layout = StorageLayout(project_root=tmp_path / "default")
    tiny_layout = StorageLayout(project_root=tmp_path / "tiny")
    processed_catalog = make_demo_processed_catalog()
    dataset_catalog = make_research_dataset_catalog(
        filter_column_roles={"eligible": ColumnRole.UNIVERSE}
    )
    write_demo_processed_inputs(default_layout, processed_catalog, frame=frame)
    for spec in processed_catalog.list():
        source = frame.loc[:, list(spec.column_roles)]
        if auxiliary_order == "asset" and spec.data_id != "demo_annotations":
            source = source.sort_values(["asset_id", "timestamp"], ignore_index=True)
        root = tiny_layout.processed_data_root(spec.market_id, spec.kind, spec.data_id)
        write_dataset(
            root,
            (source.iloc[position : position + 1] for position in range(len(source))),
            spec=spec,
        )
        assert inspect_dataset(root).file_count == len(source)
    base_spec = processed_catalog.get(dataset_catalog.get("demo").base_input_data_id)
    base_root = tiny_layout.processed_data_root(
        base_spec.market_id, base_spec.kind, base_spec.data_id
    )
    from quant_research_platform.storage.datasets.reading import (
        iter_dataset_chunks,
        open_dataset_session,
    )

    batches = list(
        iter_dataset_chunks(
            open_dataset_session(base_root, columns=None, time_window=None)
        )
    )
    assert len(batches) == len(frame)
    assert all(len(batch) == 1 for batch in batches)
    assert batches[0].timestamp.iloc[0] == batches[1].timestamp.iloc[0]
    default_root = assemble_research_dataset(
        inputs={
            item.data_id: default_layout.processed_data_root(
                item.market_id, item.kind, item.data_id
            )
            for item in processed_catalog.list()
            if item.data_id in dataset_catalog.get("demo").input_data_ids
        },
        output_root=default_layout.research_dataset_root(
            dataset_catalog.get("demo").market_id, "demo"
        ),
        dataset_catalog=dataset_catalog,
        dataset_id="demo",
        time_window=TimeWindow("2014-01-02", "2014-01-08"),
    )
    from collections import Counter

    from quant_research_platform.storage.datasets import reading

    opens = Counter()
    original = reading.iter_declared_parquet_member_batches

    def measured(path, **kwargs):
        opens[Path(path)] += 1
        yield from original(path, **kwargs)

    monkeypatch.setattr(reading, "iter_declared_parquet_member_batches", measured)
    tiny_root = assemble_research_dataset(
        inputs={
            item.data_id: tiny_layout.processed_data_root(
                item.market_id, item.kind, item.data_id
            )
            for item in processed_catalog.list()
            if item.data_id in dataset_catalog.get("demo").input_data_ids
        },
        output_root=tiny_layout.research_dataset_root(
            dataset_catalog.get("demo").market_id, "demo"
        ),
        dataset_catalog=dataset_catalog,
        dataset_id="demo",
        time_window=TimeWindow("2014-01-02", "2014-01-08"),
    )
    assert len(opens) == 3 * len(frame)
    assert set(opens.values()) == {1}
    monkeypatch.setattr(reading, "iter_declared_parquet_member_batches", original)
    default = pd.concat(
        read_dataset(
            default_layout.research_dataset_root(MarketId("a_share"), "demo"),
            time_window=TimeWindow("2014-01-02", "2014-01-08"),
            mode=DatasetReadMode.CHUNKED,
        ),
        ignore_index=True,
    ).sort_values(["market_id", "asset_id", "timestamp"])
    tiny = pd.concat(
        read_dataset(
            tiny_layout.research_dataset_root(MarketId("a_share"), "demo"),
            time_window=TimeWindow("2014-01-02", "2014-01-08"),
            mode=DatasetReadMode.CHUNKED,
        ),
        ignore_index=True,
    ).sort_values(["market_id", "asset_id", "timestamp"])
    expected = frame.sort_values(["market_id", "asset_id", "timestamp"]).reset_index(
        drop=True
    )
    expected["timestamp"] = expected.timestamp.dt.as_unit("ns")
    for actual in (default, tiny):
        pd.testing.assert_frame_equal(
            actual.reset_index(drop=True), expected, check_exact=True
        )
    assert (
        inspect_dataset(tiny_root).file_count > inspect_dataset(default_root).file_count
    )


def test_research_chunk_view_prunes_top_level_multi_year_physical_paths(
    tmp_path: Path,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    processed_catalog = make_demo_processed_catalog()
    frame = _research_rows_for_dates(["2014-12-31", "2015-01-02"])
    write_demo_processed_inputs(layout, processed_catalog, frame=frame)
    processed_spec = processed_catalog.get("demo_factors")
    dataset_spec = make_research_dataset_catalog(
        filter_column_roles={"eligible": ColumnRole.UNIVERSE}
    ).get("demo")
    columns = assembler_module._processed_input_columns(
        dataset_spec, processed_spec, existing_columns=()
    )
    session = assembler_module.open_dataset_session(
        layout.processed_data_root(
            processed_spec.market_id, processed_spec.kind, processed_spec.data_id
        ),
        columns=columns,
        time_window=TimeWindow("2014-12-31", "2015-01-02"),
    )
    view = session.filtered_view(TimeWindow("2015-01-02", "2015-01-02"))
    assert len(view.read_paths) == 1
    assert view.read_paths[0].parent.name == "year=2015"
    assert len(session.read_paths) == 2


def test_sparse_research_dataset_left_joins_missing_non_base_input_key(
    tmp_path: Path,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    processed_catalog = make_demo_processed_catalog()
    dataset_catalog = make_research_dataset_catalog(
        key_coverage={"demo_labels": "sparse"},
        filter_column_roles={"eligible": ColumnRole.UNIVERSE},
    )
    frame = _demo_research_rows()
    labels = frame.drop(index=frame.index[-1]).reset_index(drop=True)
    write_demo_processed_inputs(layout, processed_catalog, label_frame=labels)
    assemble_research_dataset(
        inputs={
            item.data_id: layout.processed_data_root(
                item.market_id, item.kind, item.data_id
            )
            for item in processed_catalog.list()
            if item.data_id in dataset_catalog.get("demo").input_data_ids
        },
        output_root=layout.research_dataset_root(
            dataset_catalog.get("demo").market_id, "demo"
        ),
        dataset_catalog=dataset_catalog,
        dataset_id="demo",
        time_window=TimeWindow("2014-01-02", "2014-01-03"),
    )
    result = pd.concat(
        read_dataset(
            layout.research_dataset_root(MarketId("a_share"), "demo"),
            time_window=TimeWindow("2014-01-02", "2014-01-03"),
            mode=DatasetReadMode.CHUNKED,
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
    dataset_catalog = ResearchDatasetCatalog(
        (
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
                primary_key=PLATFORM_ASSET_TIME_KEY,
                partition_columns=("year",),
            ),
        )
    )
    write_demo_processed_inputs(layout, processed_catalog)
    with pytest.raises(ValueError, match="assemble_factor_not_available"):
        assemble_research_dataset(
            inputs={
                item.data_id: layout.processed_data_root(
                    item.market_id, item.kind, item.data_id
                )
                for item in processed_catalog.list()
                if item.data_id in dataset_catalog.get("bad_demo").input_data_ids
            },
            output_root=layout.research_dataset_root(
                dataset_catalog.get("bad_demo").market_id, "bad_demo"
            ),
            dataset_catalog=dataset_catalog,
            dataset_id="bad_demo",
            time_window=TimeWindow("2014-01-02", "2014-01-03"),
        )


@pytest.mark.parametrize(
    "coverage,physical_rows,extra_date",
    [
        ("strict", 1, None),
        ("strict", 1, "2023-12-31"),
        ("strict", 2, "2024-01-02"),
        ("strict", 1, "2024-01-04"),
        ("strict", 2, "2024-01-06"),
        ("sparse", 2, None),
        ("sparse", 1, "2024-01-02"),
        ("sparse", 2, "2024-01-06"),
    ],
)
def test_full_request_key_domain_is_independent_of_base_file_boundaries(
    tmp_path, coverage, physical_rows, extra_date
):
    import hashlib

    from quant_research_platform.storage.datasets import read_dataset
    from tests.helpers.research_specs import (
        DEMO_FACTOR_COLUMN,
        DEMO_LABEL_COLUMN,
        make_demo_research_dataset_spec,
    )

    layout = StorageLayout(tmp_path)
    processed = make_demo_processed_catalog()
    spec = make_demo_research_dataset_spec(key_coverage={"demo_factors": coverage})
    rows = pd.DataFrame(
        {
            "market_id": "a_share",
            "asset_id": "asset_a",
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-03"]),
            DEMO_FACTOR_COLUMN: 1.0,
            DEMO_LABEL_COLUMN: 0.01,
            "eligible": True,
            "can_enter": True,
            "can_exit": True,
            "market_type": 1,
        }
    )
    factors = (
        rows
        if extra_date is None
        else pd.concat(
            (rows, rows.iloc[:1].assign(timestamp=pd.Timestamp(extra_date))),
            ignore_index=True,
        ).sort_values("timestamp")
    )
    roots = []
    for data_id, frame in (
        ("demo_annotations", rows),
        ("demo_labels", rows),
        ("demo_factors", factors),
    ):
        definition = processed.get(data_id)
        root = layout.processed_data_root(
            definition.market_id, definition.kind, data_id
        )
        selected = frame.loc[:, list(definition.required_columns)]
        chunks = tuple(
            selected.iloc[start : start + physical_rows]
            for start in range(0, len(selected), physical_rows)
        )
        write_dataset(root, chunks, spec=definition)
        roots.append(root)
    before = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for root in roots
        for path in root.rglob("*")
        if path.is_file()
    }

    def assemble():
        return assemble_research_dataset(
            inputs={
                item.data_id: layout.processed_data_root(
                    item.market_id, item.kind, item.data_id
                )
                for item in processed.list()
                if item.data_id
                in ResearchDatasetCatalog((spec,)).get(spec.data_id).input_data_ids
            },
            output_root=layout.research_dataset_root(
                ResearchDatasetCatalog((spec,)).get(spec.data_id).market_id,
                spec.data_id,
            ),
            dataset_catalog=ResearchDatasetCatalog((spec,)),
            dataset_id=spec.data_id,
            time_window=TimeWindow("2023-12-31", "2024-01-04"),
        )

    if extra_date in {"2023-12-31", "2024-01-02", "2024-01-04"}:
        with pytest.raises(
            ValueError, match="extra_research_input_keys: demo_factors: 1"
        ):
            assemble()
        assert not layout.research_dataset_root(spec.market_id, spec.data_id).exists()
    else:
        frame = read_dataset(assemble())
        assert frame.timestamp.tolist() == rows.timestamp.tolist()
    assert all(
        (
            hashlib.sha256(path.read_bytes()).hexdigest() == digest
            for path, digest in before.items()
        )
    )


def test_model_target_uses_last_participating_label_maturity_and_keeps_raw_clock(
    tmp_path,
):
    from dataclasses import replace

    from quant_research_platform.contracts.experiments import (
        legal_training_rows,
    )
    from quant_research_platform.research.protocols import a_share
    from quant_research_platform.storage.datasets import read_dataset

    base = a_share.research_dataset_catalog().get("pv_all")
    frame = pd.DataFrame(
        {
            "market_id": "a_share",
            "asset_id": ["a", "b", "ineligible"],
            "timestamp": pd.Timestamp("2020-01-02"),
            "forward_return_20d": [0.1, 0.3, 0.5],
            "forward_return_20d_available_at": pd.to_datetime(
                ["2020-02-01", "2020-02-04", "2020-02-10"]
            ),
            "is_a_share_scope": True,
            "is_trading": True,
            "is_st": [False, False, True],
            "is_pt": False,
            "listing_age_trading_days": 400,
            "has_252_trading_sessions": True,
            "close": 10.0,
            "adj_close": 10.0,
            "volume": 10.0,
            "amount": 100.0,
            "turnover": 0.1,
        }
    )
    output = assembler_module._add_a_share_semantic_columns(frame, base)
    output["momentum_20d"] = [1.0, 2.0, 3.0]
    pd.testing.assert_series_equal(
        output.forward_return_20d_available_at, frame.forward_return_20d_available_at
    )
    assert (
        output.model_target_available_at.iloc[:2].eq(pd.Timestamp("2020-02-04")).all()
    )
    assert pd.isna(output.model_target_available_at.iloc[2])
    columns = (
        "market_id",
        "asset_id",
        "timestamp",
        "forward_return_20d",
        "forward_return_20d_available_at",
        "model_target",
        "model_target_available_at",
        "eligible",
        "momentum_20d",
    )
    spec = replace(
        base,
        factor_columns=("momentum_20d",),
        feature_columns=(),
        feature_transforms=(),
        context_columns=(
            "model_target_available_at",
            "forward_return_20d_available_at",
        ),
        label_columns=(
            "forward_return_20d",
            "model_target",
        ),
        column_roles={key: base.column_roles[key] for key in columns},
        label_availability_columns={
            label: maturity
            for label, maturity in base.label_availability_columns.items()
            if label in columns and maturity in columns
        },
    )
    root = StorageLayout(tmp_path).research_dataset_root(spec.market_id, spec.data_id)
    write_dataset(root, output.loc[:, columns], spec=spec)
    saved = read_dataset(root)
    maturity = spec.label_availability_columns["model_target"]
    assert maturity == "model_target_available_at"
    assert legal_training_rows(
        saved,
        fitting_cutoff=pd.Timestamp("2020-02-02"),
        availability_column=maturity,
        target_column="model_target",
        feature_columns=(),
    ).empty
    assert (
        len(
            legal_training_rows(
                saved,
                fitting_cutoff=pd.Timestamp("2020-02-04"),
                availability_column=maturity,
                target_column="model_target",
                feature_columns=(),
            )
        )
        == 2
    )


@pytest.mark.parametrize("recipe", ("a_share_canonical", "a_share_quantile_1_99"))
def test_named_feature_recipe_assembles_complete_sessions_and_persists_identity(
    tmp_path, recipe
):
    import numpy as np

    from quant_research_platform.research.features.transforms import (
        a_share_feature_pipeline,
    )

    layout = StorageLayout(project_root=tmp_path)
    transform = FeatureTransformSpec(recipe, ("factor",))
    context = ("industry_standard_id", "industry_level1_code", "log_total_market_cap")
    frame = pd.DataFrame(
        {
            "market_id": "a_share",
            "asset_id": [f"asset_{i:03d}" for i in range(120)],
            "timestamp": pd.Timestamp("2020-01-03"),
            "factor": np.sin(np.arange(120)),
            "label": 0.0,
            "eligible": True,
            "industry_standard_id": "fixture",
            "industry_level1_code": "A",
            "log_total_market_cap": np.arange(120) * 0.01,
        }
    )
    frame.loc[119, "factor"] = 1000.0
    roles = {
        "market_id": ColumnRole.KEY,
        "asset_id": ColumnRole.KEY,
        "timestamp": ColumnRole.TIMESTAMP,
        "factor": ColumnRole.FACTOR,
        "label": ColumnRole.LABEL,
        "eligible": ColumnRole.UNIVERSE,
        **dict.fromkeys(context, ColumnRole.CONTEXT),
    }
    source = ProcessedDataSpec(
        data_id="recipe_inputs",
        market_id=MarketId("a_share"),
        kind=ProcessedDataKind.PANEL,
        frequency=DataFrequency.DAILY,
        grain=DataGrain.ASSET_TIMESTAMP,
        column_roles=roles,
        primary_key=PLATFORM_ASSET_TIME_KEY,
        partition_columns=("year",),
    )
    datasets = make_research_dataset_catalog(
        data_id="recipe_case",
        input_data_ids=(source.data_id,),
        base_input_data_id=source.data_id,
        factor_columns=("factor",),
        feature_columns=transform.output_columns,
        feature_transforms=(transform,),
        label_columns=("label",),
        filter_columns=("eligible",),
        filter_column_roles={"eligible": ColumnRole.UNIVERSE},
        context_columns=context,
    )
    write_dataset(
        layout.processed_data_root(source.market_id, source.kind, source.data_id),
        (frame.iloc[i : i + 25] for i in range(0, len(frame), 25)),
        spec=source,
    )
    window = TimeWindow("2020-01-03", "2020-01-03")
    output = assemble_research_dataset(
        inputs={
            item.data_id: layout.processed_data_root(
                item.market_id, item.kind, item.data_id
            )
            for item in ProcessedDataCatalog((source,)).list()
            if item.data_id in datasets.get("recipe_case").input_data_ids
        },
        output_root=layout.research_dataset_root(
            datasets.get("recipe_case").market_id, "recipe_case"
        ),
        dataset_catalog=datasets,
        dataset_id="recipe_case",
        time_window=window,
    )
    saved = pd.concat(
        read_dataset(
            output,
            time_window=window,
            mode=DatasetReadMode.CHUNKED,
        ),
        ignore_index=True,
    )
    expected = a_share_feature_pipeline(
        frame, factor_columns=("factor",), recipe_id=recipe
    )
    pd.testing.assert_series_equal(saved["factor"], frame["factor"], check_exact=True)
    for column in transform.output_columns:
        if column == "feature_factor":
            pd.testing.assert_series_equal(
                saved[column],
                expected[column],
                check_exact=False,
                rtol=1e-12,
                atol=1e-14,
            )
        else:
            pd.testing.assert_series_equal(
                saved[column], expected[column], check_exact=True
            )
    feature = saved.feature_factor.to_numpy()
    assert np.isfinite(feature).all()
    assert feature.mean() == pytest.approx(0.0, abs=1e-14)
    assert np.mean(feature**2) == pytest.approx(1.0, rel=1e-12)
    size = frame.log_total_market_cap.to_numpy()
    raw = frame.factor.to_numpy()
    if recipe == "a_share_quantile_1_99":
        size = np.clip(size, *np.quantile(size, [0.01, 0.99]))
        clipped = np.clip(raw, *np.quantile(raw, [0.01, 0.99]))
    else:
        center = np.median(raw)
        bound = 3 * 1.4826 * np.median(np.abs(raw - center))
        clipped = np.clip(raw, center - bound, center + bound)
    design = np.column_stack([np.ones(len(size)), size])
    residual = clipped - design @ np.linalg.lstsq(design, clipped, rcond=None)[0]
    independent = (residual - residual.mean()) / residual.std(ddof=0)
    np.testing.assert_allclose(feature, independent, rtol=1e-12, atol=1e-14)
    assert np.mean(feature * (size - size.mean())) == pytest.approx(0.0, abs=1e-14)
    assert inspect_dataset(output).spec["feature_transforms"] == [
        {
            "recipe_id": recipe,
            "source_columns": ["factor"],
            "output_groups": ["processed"],
            "clip": transform.clip,
            "industry_neutralize": True,
            "size_neutralize": True,
            "size_column": "log_total_market_cap",
            "center": True,
            "scale": True,
            "impute_missing": True,
        },
    ]


def _research_rows_for_dates(dates: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for date_index, timestamp in enumerate(pd.to_datetime(dates), start=1):
        for asset_index, asset_id in enumerate(("000001", "000002"), start=1):
            rows.append(
                {
                    "market_id": "a_share",
                    "asset_id": asset_id,
                    "timestamp": timestamp,
                    "momentum_20d": float(date_index + asset_index),
                    "forward_return_20d": 0.01 * (date_index + asset_index),
                    "eligible": True,
                    "can_enter": True,
                    "can_exit": True,
                    "market_type": 4,
                }
            )
    return pd.DataFrame(rows)


def test_join_closes_every_open_input_on_coverage_failure(tmp_path, monkeypatch):
    layout = StorageLayout(project_root=tmp_path)
    processed = make_demo_processed_catalog()
    dataset = make_research_dataset_catalog(
        filter_column_roles={"eligible": ColumnRole.UNIVERSE}
    )
    frame = _research_rows_for_dates(
        ["2014-01-02", "2014-01-03", "2014-01-06", "2014-01-07", "2014-01-08"]
    )
    for spec in processed.list():
        source = frame.loc[:, list(spec.column_roles)]
        if spec.data_id == "demo_labels":
            source = source.iloc[1:]
        write_dataset(
            layout.processed_data_root(spec.market_id, spec.kind, spec.data_id),
            (source.iloc[i : i + 1] for i in range(len(source))),
            spec=spec,
        )
    started, closed = [], []
    original = assembler_module.iter_dataset_chunks

    def observed(session, **kwargs):
        started.append(session.spec["data_id"])
        try:
            yield from original(session, **kwargs)
        finally:
            closed.append(session.spec["data_id"])

    from quant_research_platform.storage.datasets import reading

    monkeypatch.setattr(assembler_module, "iter_dataset_chunks", observed)
    monkeypatch.setattr(reading, "iter_dataset_chunks", observed)
    with pytest.raises(ValueError, match="missing_research_input_keys"):
        assemble_research_dataset(
            inputs={
                item.data_id: layout.processed_data_root(
                    item.market_id, item.kind, item.data_id
                )
                for item in processed.list()
            },
            output_root=tmp_path / "failed-output",
            dataset_catalog=dataset,
            dataset_id="demo",
            time_window=TimeWindow("2014-01-02", "2014-01-08"),
        )
    assert len(started) == 3
    assert sorted(started) == sorted(closed)
    assert not (tmp_path / "failed-output").exists()


@pytest.mark.parametrize("storage", ["native", "parquet"])
@pytest.mark.parametrize("dtype", ["datetime64[us]", "datetime64[ns]"])
@pytest.mark.parametrize(
    "end,count", [("2024-01-02", 3), ("2024-01-02T00:00:00.000000999", 1)]
)
def test_auxiliary_timestamp_units_preserve_nanosecond_window(
    tmp_path, dtype, end, count, storage
):
    from quant_research_platform.contracts.data import DataSpec

    times = (pd.Timestamp("2024-01-02") + pd.to_timedelta([0, 1, 2], unit="us")).astype(
        dtype
    )
    inputs = {}
    for data_id, column, role in (
        ("base", "factor", "factor"),
        ("aux", "label", "label"),
    ):
        frame = pd.DataFrame(
            {"asset_id": "a", "timestamp": times, column: [1.0, 2.0, 3.0]}
        )
        spec = DataSpec(
            data_id=data_id,
            frequency="tick",
            grain="asset_timestamp",
            primary_key=("asset_id", "timestamp"),
            partition_columns=("year",),
            column_roles={"asset_id": "key", "timestamp": "timestamp", column: role},
        )
        if storage == "native":
            root = tmp_path / data_id
            write_dataset(
                root, (frame.iloc[i : i + 1] for i in range(len(frame))), spec=spec
            )
            inputs[data_id] = str(root)
        else:
            root = tmp_path / (data_id + ".parquet")
            frame.to_parquet(root, index=False)
            inputs[data_id] = {
                "path": str(root),
                "spec": {
                    "data_id": data_id,
                    "frequency": "tick",
                    "grain": "asset_timestamp",
                    "primary_key": ["asset_id", "timestamp"],
                    "partition_columns": ["year"],
                    "column_roles": {
                        "asset_id": "key",
                        "timestamp": "timestamp",
                        column: role,
                    },
                },
            }
    root = assembler_module.assemble(
        {
            "market_id": "crypto",
            "dataset_id": "units",
            "inputs": inputs,
            "base_input_data_id": "base",
            "factor_columns": ["factor"],
            "label_columns": ["label"],
            "filter_columns": [],
            "time_window": {
                "start": "2024-01-02T00:00:00" if "T" in end else "2024-01-02",
                "end": end,
            },
            "output_root": str(tmp_path / "output"),
        }
    )
    result = read_dataset(root)
    assert len(result) == count
    pd.testing.assert_series_equal(
        result.timestamp,
        pd.Series(times[:count].astype("datetime64[ns]"), name="timestamp"),
    )
    assert result.label.tolist() == [1.0, 2.0, 3.0][:count]


@pytest.mark.parametrize(
    "base_id, auxiliary_id, clock_offset, sparse, matches",
    [
        (1, 1.0, 0, False, True),
        (True, 1, 0, False, True),
        (2**53 + 1, float(2**53), 0, False, False),
        (2**53 + 1, float(2**53), 0, True, False),
        ("1", 1, 0, False, False),
        (1, 1, 1, False, False),
    ],
)
def test_native_assembly_uses_exact_scalar_key_identity(
    tmp_path, base_id, auxiliary_id, clock_offset, sparse, matches
):
    from tests.storage.low_level_fixtures import make_processed_dataset_spec

    timestamp = pd.Timestamp("2024-01-02")
    for name, asset, column, offset in (
        ("base", base_id, "factor", 0),
        ("auxiliary", auxiliary_id, "other", clock_offset),
    ):
        write_dataset(
            tmp_path / name,
            pd.DataFrame(
                {
                    "market_id": ["custom"],
                    "asset_id": [asset],
                    "timestamp": [timestamp + pd.Timedelta(offset, "ns")],
                    column: [7.0],
                }
            ),
            spec=make_processed_dataset_spec(
                name, market_id="custom", value_column=column
            ),
        )
    config = {
        "market_id": "custom",
        "dataset_id": "joined",
        "inputs": {name: str(tmp_path / name) for name in ("base", "auxiliary")},
        "base_input_data_id": "base",
        "factor_columns": ["factor", "other"],
        "label_columns": [],
        "feature_recipe": None,
        "sparse_input_data_ids": ["auxiliary"] if sparse else [],
        "time_window": {"start": "2024-01-02", "end": "2024-01-02"},
        "output_root": str(tmp_path / "joined"),
    }
    if matches:
        result = read_dataset(assembler_module.assemble(config))
        pd.testing.assert_series_equal(
            result.asset_id, read_dataset(tmp_path / "base").asset_id
        )
        assert result.other.tolist() == [7.0]
    else:
        with pytest.raises(ValueError, match="extra_research_input_keys: auxiliary: 1"):
            assembler_module.assemble(config)
        assert not (tmp_path / "joined").exists()


def test_sparse_native_alignment_preserves_observed_discrete_payload(tmp_path):
    from dataclasses import replace

    from tests.storage.low_level_fixtures import make_processed_dataset_spec

    keys = pd.DataFrame(
        {
            "market_id": ["custom"] * 3,
            "asset_id": ["c", "a", "b"],
            "timestamp": [pd.Timestamp("2024-01-02")] * 3,
        }
    )
    base = make_processed_dataset_spec(
        "base", market_id="custom", value_column="factor"
    )
    write_dataset(tmp_path / "base", keys.assign(factor=1.0), spec=base)
    columns = {
        "signed": "int64",
        "unsigned": "uint64",
        "nullable": "Int64",
        "flag": "bool",
    }
    auxiliary = keys.iloc[[2, 0]].reset_index(drop=True)
    for name, dtype in columns.items():
        auxiliary[name] = pd.Series(
            [True, False] if name == "flag" else [2**53 + 1, 2**53 + 3], dtype=dtype
        )
    spec = replace(
        base,
        data_id="auxiliary",
        column_roles={
            **{key: base.column_roles[key] for key in base.primary_key},
            **{name: ColumnRole.CONTEXT for name in columns},
        },
    )
    write_dataset(tmp_path / "auxiliary", auxiliary, spec=spec)
    result = read_dataset(
        assembler_module.assemble(
            {
                "market_id": "custom",
                "dataset_id": "joined",
                "inputs": {
                    name: str(tmp_path / name) for name in ("base", "auxiliary")
                },
                "base_input_data_id": "base",
                "factor_columns": ["factor"],
                "label_columns": [],
                "context_columns": list(columns),
                "feature_recipe": None,
                "sparse_input_data_ids": ["auxiliary"],
                "time_window": {"start": "2024-01-02", "end": "2024-01-02"},
                "output_root": str(tmp_path / "joined"),
            }
        )
    ).set_index("asset_id")
    assert result.index.tolist() == ["a", "b", "c"]
    assert result.loc["a", list(columns)].isna().all()
    for name in ("signed", "unsigned", "nullable"):
        assert result[name].loc[["b", "c"]].tolist() == [2**53 + 1, 2**53 + 3]
    assert result.flag.loc[["b", "c"]].tolist() == [True, False]
    assert result[list(columns)].dtypes.astype(str).tolist() == [
        "Int64",
        "UInt64",
        "Int64",
        "boolean",
    ]

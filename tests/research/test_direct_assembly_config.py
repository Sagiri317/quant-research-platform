"""Actual declarations, minimal dependencies and no-label assembly output."""

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.contracts.columns import (
    PLATFORM_ASSET_TIME_KEY,
)
from quant_research_platform.contracts.data import DataSpec
from quant_research_platform.research.datasets import assembler
from quant_research_platform.research.features.transforms import (
    a_share_feature_pipeline,
)
from quant_research_platform.storage.datasets import (
    inspect_dataset,
    read_dataset,
    write_dataset,
)
from tests.helpers.research_processed_inputs import assembly_config as _config
from tests.helpers.research_processed_inputs import assembly_frame as _frame
from tests.helpers.research_processed_inputs import assembly_source as _source


def test_single_factor_canonical_uses_auxiliary_size_without_putting_it_in_model_x(
    tmp_path, monkeypatch
):
    frame = _frame().assign(
        eligible=True,
        industry_standard_id="test",
        industry_level1_code="A",
        log_total_market_cap=np.arange(120) / 10,
        unrelated_context="unused",
    )
    root, _ = _source(tmp_path, frame)
    calls = []
    original = assembler.iter_dataset_chunks

    def measured(session, **kwargs):
        calls.append((session.root, session.logical_columns, session.physical_columns))
        yield from original(session, **kwargs)

    monkeypatch.setattr(assembler, "iter_dataset_chunks", measured)
    output = assembler.assemble(
        _config(tmp_path, root, feature_recipe="a_share_canonical")
    )
    assert len(calls) == 1
    assert set(calls[0][1]) == set(PLATFORM_ASSET_TIME_KEY) | {
        "momentum_20d",
        "eligible",
        "industry_standard_id",
        "industry_level1_code",
        "log_total_market_cap",
    }
    assert "unrelated_context" not in calls[0][2]
    actual = read_dataset(output)
    spec = inspect_dataset(output).spec
    assert spec["label_columns"] == []
    assert spec["context_columns"] == []
    assert spec["factor_columns"] == ["momentum_20d"]
    assert "log_total_market_cap" not in actual
    assert "eligible" not in actual
    expected = a_share_feature_pipeline(frame, factor_columns=("momentum_20d",))
    np.testing.assert_allclose(
        actual.feature_momentum_20d,
        expected.feature_momentum_20d,
        rtol=1e-13,
        atol=1e-14,
    )
    assert actual.feature_observed_momentum_20d.eq(1).all()


def test_disabled_steps_accept_input_without_unneeded_context_and_persist_options(
    tmp_path,
):
    frame = _frame(3).assign(eligible=True)
    root, _ = _source(tmp_path, frame, units={"momentum_20d": "return"})
    options = {
        "clip": "none",
        "industry_neutralize": False,
        "size_neutralize": False,
        "center": False,
        "scale": False,
        "impute_missing": False,
    }
    output = assembler.assemble(
        _config(
            tmp_path,
            root,
            feature_recipe="a_share_canonical",
            feature_transform=options,
        )
    )
    actual = read_dataset(output)
    np.testing.assert_array_equal(actual.feature_momentum_20d, frame.momentum_20d)
    spec = inspect_dataset(output).spec
    assert all(
        spec["feature_transforms"][0][name] == value for name, value in options.items()
    )
    assert spec["column_units"]["feature_momentum_20d"] == "return"
    assert spec["column_units"]["feature_observed_momentum_20d"] == "dimensionless"


def test_declared_industry_pairs_have_independent_native_feature_and_target_residuals(
    tmp_path,
):
    group = np.repeat([0, 1], 60)
    residual = np.tile([-1.0, 1.0], 60)
    size = np.tile([-1.0, -1.0, 1.0, 1.0], 30)
    frame = _frame().assign(
        eligible=True,
        industry_standard_id=np.where(group == 0, "research:sector", "research"),
        industry_level1_code=np.where(group == 0, "A", "sector:A"),
        momentum_20d=10 + 10 * group + residual,
        log_total_market_cap=10 + size,
        forward_return_20d=0.1 + 0.1 * group + 0.01 * size + 0.001 * residual,
        matured_at=pd.Timestamp("2024-02-01"),
    )
    root, _ = _source(
        tmp_path, frame, availability={"forward_return_20d": "matured_at"}
    )
    actual = read_dataset(
        assembler.assemble(
            _config(
                tmp_path,
                root,
                feature_recipe="a_share_canonical",
                feature_transform={
                    "clip": "none",
                    "industry_neutralize": True,
                    "size_neutralize": False,
                    "center": False,
                    "scale": False,
                    "impute_missing": False,
                },
                label_columns=["model_target"],
                target_transform="industry_size_residual",
                target_clip="none",
            )
        )
    ).sort_values("asset_id")
    # Separate industry intercepts remove both offsets. The size pattern is
    # orthogonal to +/-1, and residual targets retain their return units.
    np.testing.assert_allclose(actual.feature_momentum_20d, residual, atol=1e-12)
    np.testing.assert_allclose(actual.model_target, 0.001 * residual, atol=1e-14)
    assert actual.feature_observed_momentum_20d.eq(1).all()
    assert actual.model_target_available_at.eq(pd.Timestamp("2024-02-01")).all()


def test_industry_pair_encoding_escapes_both_components_and_preserves_nulls():
    from quant_research_platform.research.features.transforms import (
        _industry_group_codes,
    )

    frame = pd.DataFrame(
        {
            "industry_standard_id": ["CSRC2012", "a:b", "a", "a\\b", "a", None, "a"],
            "industry_level1_code": ["A", "c", "b:c", "c", "b\\c", "c", None],
        }
    )
    actual = _industry_group_codes(frame)
    assert actual.iloc[:5].tolist() == [
        "CSRC2012:A",
        r"a\:b:c",
        r"a:b\:c",
        r"a\\b:c",
        r"a:b\\c",
    ]
    assert actual.iloc[5:].isna().all()
    assert actual.nunique() == 5
    assert isinstance(actual.dtype, pd.StringDtype)


def test_actual_label_clock_and_units_survive_selection(tmp_path):
    frame = _frame(3).assign(
        return_next=[0.1, 0.2, 0.3],
        matured_at=pd.Timestamp("2024-01-03"),
        unrelated_context="unused",
    )
    root, _ = _source(
        tmp_path,
        frame,
        units={"return_next": "fraction", "momentum_20d": "fraction"},
        availability={"return_next": "matured_at"},
    )
    output = assembler.assemble(_config(tmp_path, root, label_columns=["return_next"]))
    actual = read_dataset(output)
    spec = inspect_dataset(output).spec
    assert spec["label_availability_columns"] == {"return_next": "matured_at"}
    assert spec["context_columns"] == ["matured_at"]
    assert spec["column_units"] == {
        "return_next": "fraction",
        "momentum_20d": "fraction",
    }
    assert "unrelated_context" not in actual
    assert actual.matured_at.eq(pd.Timestamp("2024-01-03")).all()


def test_native_prepared_label_clock_remains_context_through_assembly(tmp_path):
    from tests.markets.a_share.test_calendar_dependency_scope import (
        _calendar,
        _panel,
        _prepare,
    )

    dates = pd.bdate_range("2024-01-02", periods=5).as_unit("ns")
    panel = _panel(dates)
    label = "forward_return_1d"
    clock = f"{label}_available_at"
    labels = _prepare(
        tmp_path / "prepared",
        panel,
        _calendar({1: dates}),
        "forward_returns",
        label_columns=[label],
    )
    label_root = tmp_path / "prepared" / "output"
    prepared = inspect_dataset(label_root).spec
    assert prepared["column_roles"][clock] == "context"
    factors, _ = _source(
        tmp_path,
        panel[list(PLATFORM_ASSET_TIME_KEY)].assign(momentum_20d=np.arange(5.0)),
    )
    output = assembler.assemble(
        _config(
            tmp_path,
            factors,
            inputs={factors.name: str(factors), "forward_returns": str(label_root)},
            label_columns=[label],
            time_window={"start": "2024-01-02", "end": "2024-01-08"},
        )
    )
    spec = inspect_dataset(output).spec
    assert spec["label_columns"] == [label]
    assert spec["context_columns"] == [clock]
    assert spec["column_roles"][clock] == "context"
    assert spec["column_units"][label] == "decimal_return"
    assert spec["label_availability_columns"] == {label: clock}
    actual = read_dataset(output)
    pd.testing.assert_frame_equal(actual[labels.columns], labels, check_exact=True)
    np.testing.assert_allclose(actual[label].iloc[:3], 1 / np.arange(101.0, 104.0))
    assert actual[label].iloc[-2:].isna().all()
    assert (
        actual[clock].iloc[:3].tolist() == (dates[2:] + pd.Timedelta(hours=7)).tolist()
    )
    assert actual[clock].iloc[-2:].isna().all()


def test_bare_input_uses_supplied_actual_semantics_without_catalog_membership(tmp_path):
    frame = _frame(3)
    path = tmp_path / "source.parquet"
    frame.to_parquet(path, index=False)
    spec = DataSpec(
        data_id="my_factor",
        frequency="1d",
        grain="asset_timestamp",
        primary_key=PLATFORM_ASSET_TIME_KEY,
        partition_columns=(),
        column_roles={
            "market_id": "key",
            "asset_id": "key",
            "timestamp": "timestamp",
            "momentum_20d": "factor",
        },
    )
    config = _config(tmp_path, path)
    config["inputs"] = {
        "my_factor": {
            "path": str(path),
            "spec": {
                "data_id": spec.data_id,
                "frequency": str(spec.frequency),
                "grain": str(spec.grain),
                "primary_key": list(spec.primary_key),
                "partition_columns": list(spec.partition_columns),
                "column_roles": {
                    key: str(role) for key, role in spec.column_roles.items()
                },
            },
        }
    }
    actual = read_dataset(assembler.assemble(config))
    pd.testing.assert_frame_equal(actual, frame.astype({"timestamp": "datetime64[ns]"}))


def test_base_cannot_be_sparse(tmp_path):
    root, _ = _source(tmp_path, _frame(3))
    with pytest.raises(ValueError, match="assemble_sparse_inputs_invalid"):
        assembler.assemble(_config(tmp_path, root, sparse_input_data_ids=[root.name]))


@pytest.mark.parametrize(
    "recipe,selected",
    [
        ("a_share_all", [True, True, True, True]),
        ("a_share_nonfinancial", [True, True, False, True]),
        ("a_share_fundamental", [True, True, False, False]),
    ],
)
def test_explicit_recipe_and_actual_clock_are_independent_of_output_name(
    tmp_path, recipe, selected
):
    frame = _frame(4).assign(
        is_a_share_scope=True,
        is_trading=True,
        is_st=False,
        is_pt=False,
        has_252_trading_sessions=True,
        close=10.0,
        adj_close=10.0,
        volume=100.0,
        amount=1000.0,
        turnover=0.1,
        is_nonfinancial_industry=[True, True, False, True],
        has_legal_fundamental_pit=[True, True, True, False],
        industry_standard_id="test",
        industry_level1_code="A",
        industry_level1_name="industry",
        forward_return_20d=[0.1, 0.2, 0.3, 0.4],
        matured_at=pd.date_range("2024-02-01", periods=4),
    )
    root, _ = _source(
        tmp_path, frame, availability={"forward_return_20d": "matured_at"}
    )
    outputs = []
    for name in ("first_name", "renamed_research"):
        config = _config(
            tmp_path,
            root,
            dataset_id=name,
            output_root=str(tmp_path / name),
            label_columns=["model_target"],
            filter_columns=["eligible"],
            eligibility_recipe=recipe,
        )
        output = assembler.assemble(config)
        outputs.append(read_dataset(output))
        spec = inspect_dataset(output).spec
        assert spec["eligibility_recipe"] == recipe
        assert spec["label_availability_columns"] == {
            "forward_return_20d": "matured_at",
            "model_target": "model_target_available_at",
        }
    pd.testing.assert_frame_equal(*outputs)
    actual = outputs[0]
    assert actual.eligible.tolist() == selected
    expected = frame.forward_return_20d.loc[selected]
    expected = (expected - expected.mean()) / expected.std(ddof=0)
    np.testing.assert_allclose(
        actual.loc[selected, "model_target"], expected, rtol=1e-13, atol=1e-14
    )
    assert actual.loc[~np.array(selected), "model_target"].isna().all()
    assert (
        actual.loc[selected, "model_target_available_at"]
        .eq(frame.matured_at.loc[selected].max())
        .all()
    )
    assert actual.matured_at.tolist() == frame.matured_at.tolist()
    assert "industry_level1_code" not in actual


def test_declared_lifecycle_key_is_kept_and_auxiliary_key_order_is_irrelevant(tmp_path):
    key_roles = {"asset_id": "key", "lifecycle": "key", "timestamp": "timestamp"}
    keys = pd.DataFrame(
        {
            "asset_id": ["same", "same"],
            "lifecycle": [1, 2],
            "timestamp": pd.Timestamp("2024-01-02"),
        }
    )
    roots = {}
    for name, column, values, primary_key in (
        ("base", "momentum_20d", [1.0, 2.0], ("asset_id", "lifecycle", "timestamp")),
        ("aux", "context_value", [20.0, 10.0], ("timestamp", "asset_id", "lifecycle")),
    ):
        frame = keys if name == "base" else keys.iloc[::-1]
        frame = frame.assign(**{column: values})
        spec = DataSpec(
            data_id=name,
            frequency="1d",
            grain="asset_timestamp",
            primary_key=primary_key,
            partition_columns=("year",),
            column_roles={
                **key_roles,
                column: "factor" if name == "base" else "context",
            },
        )
        root = tmp_path / name
        write_dataset(root, frame, spec=spec)
        roots[name] = str(root)
    config = _config(
        tmp_path,
        tmp_path / "base",
        inputs=roots,
        base_input_data_id="base",
        context_columns=["context_value"],
    )
    output = assembler.assemble(config)
    actual = read_dataset(output).sort_values("lifecycle")
    assert actual.momentum_20d.tolist() == [1.0, 2.0]
    assert actual.context_value.tolist() == [10.0, 20.0]
    assert inspect_dataset(output).spec["primary_key"] == [
        "asset_id",
        "lifecycle",
        "timestamp",
    ]


def test_actual_market_values_cannot_conflict_with_declared_market(tmp_path):
    root, _ = _source(tmp_path, _frame(3).assign(market_id="crypto"))
    with pytest.raises(ValueError, match="research_dataset_market_mismatch"):
        assembler.assemble(_config(tmp_path, root))
    assert not (tmp_path / "assembled").exists()


def test_input_column_selection_bounds_business_candidates_but_keeps_dependencies(
    tmp_path,
):
    frame = _frame().assign(
        eligible=True,
        industry_standard_id="test",
        industry_level1_code="A",
        log_total_market_cap=np.arange(120) / 10,
        return_next=0.1,
    )
    root, _ = _source(tmp_path, frame)
    config = _config(tmp_path, root, feature_recipe="a_share_canonical")
    config["inputs"] = {root.name: {"path": str(root), "columns": ["momentum_20d"]}}
    config.pop("factor_columns")
    config.pop("label_columns")
    output = assembler.assemble(config)
    spec = inspect_dataset(output).spec
    assert spec["factor_columns"] == ["momentum_20d"]
    assert spec["label_columns"] == []
    assert spec["context_columns"] == []
    assert read_dataset(output).feature_observed_momentum_20d.eq(1).all()
    config["factor_columns"] = ["log_total_market_cap"]
    config["output_root"] = str(tmp_path / "invalid_factor")
    with pytest.raises(ValueError, match="assemble_factor_not_selected"):
        assembler.assemble(config)
    config["factor_columns"] = ["momentum_20d"]
    config["label_columns"] = ["return_next"]
    with pytest.raises(ValueError, match="assemble_label_not_selected"):
        assembler.assemble(config)


def test_interleaved_file_order_cannot_silently_change_complete_sections(tmp_path):
    root = tmp_path / "external_parts"
    root.mkdir()
    frame = _frame(1)
    frame.to_parquet(root / "earlier.parquet", index=False)
    frame.assign(timestamp=pd.Timestamp("2024-01-03")).to_parquet(
        root / "later.parquet", index=False
    )
    spec = DataSpec(
        data_id="external_parts",
        frequency="1d",
        grain="asset_timestamp",
        primary_key=PLATFORM_ASSET_TIME_KEY,
        partition_columns=(),
        column_roles={
            "market_id": "key",
            "asset_id": "key",
            "timestamp": "timestamp",
            "momentum_20d": "factor",
        },
    )
    config = _config(
        tmp_path,
        root,
        inputs={
            root.name: {
                "path": str(root),
                "files": ["later.parquet", "earlier.parquet"],
                "spec": {
                    "data_id": spec.data_id,
                    "frequency": str(spec.frequency),
                    "grain": str(spec.grain),
                    "primary_key": list(spec.primary_key),
                    "partition_columns": list(spec.partition_columns),
                    "column_roles": {
                        key: str(role) for key, role in spec.column_roles.items()
                    },
                },
            }
        },
        time_window={"start": "2024-01-02", "end": "2024-01-03"},
    )
    with pytest.raises(ValueError, match="non_monotonic_time_chunk_order"):
        assembler.assemble(config)
    assert not (tmp_path / "assembled").exists()


def test_named_defaults_cannot_invent_missing_upstream_label_maturity(tmp_path):
    frame = _frame(3).assign(
        eligible=True,
        forward_return_20d=[0.1, 0.2, 0.3],
        forward_return_20d_available_at=pd.Timestamp("2024-02-01"),
    )
    root, _ = _source(tmp_path, frame)
    config = _config(
        tmp_path,
        root,
        dataset_id="pv_all",
        base_input_data_id=root.name,
        label_columns=["model_target"],
        eligibility_recipe=None,
    )
    with pytest.raises(ValueError, match="model_target_label_availability_required"):
        assembler.assemble(config)


@pytest.mark.parametrize(
    "clocks,expected",
    [(["2024-02-01", None], None), (["2024-10-01", "2024-2-01"], "2024-10-01")],
)
def test_derived_target_preserves_unknown_maturity_and_compares_actual_times(
    tmp_path, clocks, expected
):
    from quant_research_platform.contracts.data import DataSpec
    from quant_research_platform.research.datasets.assembler import assemble
    from quant_research_platform.storage.datasets import read_dataset, write_dataset

    frame = pd.DataFrame(
        {
            "asset_id": ["a", "b"],
            "timestamp": pd.Timestamp("2024-01-02"),
            "x": [1.0, 2.0],
            "eligible": True,
            "forward_return_20d": [0.1, 0.3],
            "clock": clocks,
        }
    )
    declared = DataSpec(
        data_id="source",
        frequency="1d",
        grain="asset_timestamp",
        primary_key=("asset_id", "timestamp"),
        partition_columns=("year",),
        column_roles={
            "asset_id": "key",
            "timestamp": "timestamp",
            "x": "factor",
            "eligible": "universe",
            "forward_return_20d": "label",
            "clock": "context",
        },
        label_availability_columns={"forward_return_20d": "clock"},
        column_units={"forward_return_20d": "decimal_return"},
    )
    write_dataset(tmp_path / "source", frame, spec=declared)
    output = assemble(
        {
            "market_id": "a_share",
            "dataset_id": "external",
            "inputs": {"source": str(tmp_path / "source")},
            "factor_columns": ["x"],
            "label_columns": ["model_target"],
            "feature_recipe": None,
            "time_window": {"start": "2024-01-02", "end": "2024-01-02"},
            "output_root": str(tmp_path / "result"),
        }
    )
    actual = read_dataset(output)
    assert actual.model_target.notna().all()
    if expected is None:
        assert actual.model_target_available_at.isna().all()
    else:
        assert actual.model_target_available_at.eq(pd.Timestamp(expected)).all()


@pytest.mark.parametrize(
    "roles",
    [
        {"asset_id": "key", "timestamp": "key", "asof": "timestamp", "x": "factor"},
    ],
)
def test_assemble_requires_actual_asset_and_timestamp_roles(tmp_path, roles):
    from quant_research_platform.contracts.data import DataSpec
    from quant_research_platform.research.datasets.assembler import assemble
    from quant_research_platform.storage.datasets import write_dataset

    source = tmp_path / "source"
    declared = DataSpec(
        data_id="source",
        frequency="1d",
        grain="asset_timestamp",
        primary_key=("asset_id", "timestamp", "asof"),
        partition_columns=("year",),
        column_roles=roles,
    )
    write_dataset(
        source,
        pd.DataFrame(
            {
                "asset_id": ["a"],
                "timestamp": pd.to_datetime(["2024-01-03"]),
                "asof": pd.to_datetime(["2024-01-02"]),
                "x": [1.0],
            }
        ),
        spec=declared,
    )
    with pytest.raises(ValueError, match="invalid_research_dataset_key_roles"):
        assemble(
            {
                "market_id": "a_share",
                "dataset_id": "external",
                "inputs": {"source": str(source)},
                "factor_columns": ["x"],
                "label_columns": [],
                "feature_recipe": None,
                "time_window": {"start": "2024-01-02", "end": "2024-01-03"},
                "output_root": str(tmp_path / "output"),
            }
        )
    assert not (tmp_path / "output").exists()


def test_assemble_json_configuration_rejects_runtime_objects_before_io(tmp_path):
    with pytest.raises(ValueError, match="qrp_json_unsupported_value"):
        assembler.assemble({"inputs": {"x": {"path": tmp_path / "missing"}}})
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "size_column", ["log_total_market_cap", "log_float_market_cap"]
)
def test_size_choice_projects_and_persists_exact_dependency(tmp_path, size_column):
    frame = _frame().assign(
        eligible=True,
        log_total_market_cap=np.tile([1.0, 2.0, 3.0, 4.0], 30),
        log_float_market_cap=np.tile([1.0, 3.0, 2.0, 4.0], 30),
    )
    root, _ = _source(tmp_path, frame)
    options = {"industry_neutralize": False, "clip": "none"}
    if size_column == "log_float_market_cap":
        options["size_column"] = size_column
    output = assembler.assemble(
        _config(
            tmp_path,
            root,
            feature_recipe="a_share_canonical",
            feature_transform=options,
        )
    )
    inspection = inspect_dataset(output)
    assert inspection.spec["feature_transforms"][0]["size_column"] == size_column
    parameters = inspection.manifest["parameters"]
    assert parameters["feature_transform"]["size_column"] == size_column
    assert set(parameters["input_columns"][root.name]) == {
        *PLATFORM_ASSET_TIME_KEY,
        "momentum_20d",
        "eligible",
        size_column,
    }
    expected = a_share_feature_pipeline(
        frame,
        factor_columns=("momentum_20d",),
        industry_neutralize=False,
        clip="none",
        size_column=size_column,
    )
    np.testing.assert_allclose(
        read_dataset(output).feature_momentum_20d, expected.feature_momentum_20d
    )

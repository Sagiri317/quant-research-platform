from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.data import (
    DataFrequency,
    DataGrain,
    InterimDataSpec,
    ProcessedDataKind,
    ProcessedDataSpec,
)
from quant_research_platform.contracts.datasets import ResearchDatasetSpec
from quant_research_platform.contracts.ids import MarketId, SourceId
from quant_research_platform.markets.implementations.a_share import (
    research_scenarios as a_share_research_scenarios,
)
from quant_research_platform.profiles.research import research_profile
from quant_research_platform.storage.datasets import inspect_dataset, write_dataset
from quant_research_platform.workflows import local_verification as module
from quant_research_platform.workflows.local_verification import (
    LocalVerificationRunConfig,
    run_local_verification,
)
from quant_research_platform.workflows.local_verification_layout import (
    LocalVerificationLayout,
)
from tests.helpers import frequency_ids

LEGACY_RSS_PEAK_FIELD = "rss_" + "peak_mb"
MIB = 1024 * 1024
GIB = 1024 * 1024 * 1024
SMOKE_PREDICTION_TRAIN_END = "2014-06-30"
SMOKE_PREDICTION_TEST_START = "2014-07-01"
SMOKE_PORTFOLIO_FREQUENCY = frequency_ids.DAILY_FREQUENCY_SLUG
SMOKE_PORTFOLIO_EXECUTION_LAG_STEPS = 1


def _verification_config(**overrides: object) -> LocalVerificationRunConfig:
    payload = {
        "prediction_train_end": SMOKE_PREDICTION_TRAIN_END,
        "prediction_test_start": SMOKE_PREDICTION_TEST_START,
        "portfolio_frequency": SMOKE_PORTFOLIO_FREQUENCY,
        "portfolio_execution_lag_steps": SMOKE_PORTFOLIO_EXECUTION_LAG_STEPS,
    }
    payload.update(overrides)
    return LocalVerificationRunConfig(**payload)


def _smoke_prediction_specs():
    return a_share_research_scenarios.build_prediction_catalog(
        train_end=SMOKE_PREDICTION_TRAIN_END,
        test_start=SMOKE_PREDICTION_TEST_START,
    ).list()


def _repo_run_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *parts: str,
) -> Path:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(exist_ok=True)
    (repo_root / ".git").write_text("gitdir: ../repo.git\n", encoding="utf-8")
    monkeypatch.chdir(repo_root)
    return repo_root.joinpath(*parts)


def _persist_fake_artifact(
    artifact_root: Path,
    dataset_id: str,
    logical_layer: str,
) -> None:
    frame = _interim_frame() if logical_layer == "interim" else _dataset_frame()
    write_dataset(
        artifact_root,
        frame,
        spec=_dataset_spec(dataset_id, logical_layer),
    )


def _persist_fake_research_root(**kwargs) -> Path:
    artifact_root = (
        kwargs["layout"].project_root
        / "data"
        / "research"
        / "a_share"
        / "price_volume_forward_returns"
    )
    _persist_fake_artifact(
        artifact_root,
        "price_volume_forward_returns",
        "research",
    )
    return artifact_root


def _empty_processed(
    layout,
    source_id,
    time_window,
    kinds,
) -> dict[str, object]:
    del layout, source_id, time_window, kinds
    return {}


def _profile_with_materializers(interim, processed):
    return replace(
        research_profile(),
        materialize_interim_data=interim,
        materialize_processed_data=processed,
    )


def _fake_evaluation_result(kwargs):
    return SimpleNamespace(
        spec=kwargs["evaluation_catalog"].get(kwargs["run_id"]),
        metrics=pd.DataFrame(),
        grouped_label_values=pd.DataFrame(),
        ic_summary=pd.DataFrame(),
        group_spread_summary=pd.DataFrame(),
        diagnostics={},
    )


def _fake_prediction_result(kwargs):
    return SimpleNamespace(
        spec=kwargs["prediction_catalog"].get(kwargs["prediction_id"]),
        output_chunks=(),
        model_metadata={},
        prediction_metadata_factory=lambda: {},
    )


def _fake_portfolio_result(kwargs):
    return SimpleNamespace(
        spec=kwargs["portfolio_catalog"].get(kwargs["portfolio_id"]),
        output_chunks=(),
        summary_factory=lambda: {},
    )


def _stub_output_writers(monkeypatch: pytest.MonkeyPatch) -> None:
    def write_stub(*args, **kwargs):
        del args
        output_spec = kwargs["output_spec"]
        root = kwargs["layout"].project_root / "outputs" / output_spec.output_id
        root.mkdir(parents=True)
        (root / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_id": "qrp-output-manifest",
                    "output_id": output_spec.output_id,
                    "output_definition_id": output_spec.output_definition_id,
                    "run_id": output_spec.run_id,
                    "metadata": {},
                    "files": [],
                }
            ),
            encoding="utf-8",
        )
        return root

    monkeypatch.setattr(
        module,
        "write_evaluation_outputs",
        write_stub,
    )
    monkeypatch.setattr(
        module,
        "write_prediction_outputs",
        write_stub,
    )
    monkeypatch.setattr(
        module,
        "write_portfolio_outputs",
        write_stub,
    )


def _persist_tiny_a_share_processed_inputs(
    layout,
    profile,
) -> dict[str, object]:
    dataset_spec = profile.research_datasets.get("price_volume_forward_returns")
    inspections: dict[str, object] = {}
    for data_id in dataset_spec.input_data_ids:
        spec = profile.processed.get(data_id)
        root = layout.processed_data_root(spec.market_id, spec.kind, spec.data_id)
        write_dataset(
            root,
            _tiny_a_share_processed_frame(spec),
            spec=spec,
        )
        inspections[data_id] = inspect_dataset(root)
    return inspections


def _tiny_a_share_processed_frame(spec: ProcessedDataSpec) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dates = pd.to_datetime(["2014-01-02", "2014-01-03", "2014-07-02", "2015-01-05"])
    for date_index, timestamp in enumerate(dates, start=1):
        for asset_index in range(1, 7):
            row: dict[str, object] = {
                "market_id": "a_share",
                "asset_id": f"asset_{asset_index:03d}",
                "timestamp": timestamp,
            }
            for column in spec.required_columns:
                if column in row:
                    continue
                row[column] = _tiny_processed_value(
                    column,
                    date_index=date_index,
                    asset_index=asset_index,
                )
            rows.append(row)
    return pd.DataFrame(rows, columns=spec.required_columns)


def _tiny_processed_value(
    column: str,
    *,
    date_index: int,
    asset_index: int,
) -> object:
    if column in {
        "eligible",
        "eligible_base",
        "eligible_after_listing_age",
        "eligible_liquidity",
        "can_enter",
        "can_exit",
        "next_bar_is_trading",
    }:
        return True
    if column in {"is_st", "is_pt"}:
        return False
    if column == "csrc_2012_industry_code":
        return "I"
    if column == "csrc_2012_industry_name":
        return "industry"
    if column in {"limit_status", "next_bar_limit_status"}:
        return 0
    if column == "market_type":
        return 1
    if column == "mom_20d":
        return float(asset_index + date_index / 10)
    if column == "rev_1d":
        return float((7 - asset_index) + date_index / 20)
    if column == "forward_return_20d":
        return float(asset_index + date_index) / 100
    return float(asset_index * 10 + date_index) / 1000


def _dataset_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "market_id": ["a_share"],
            "asset_id": ["asset_a"],
            "timestamp": pd.to_datetime(["2014-01-02"]),
            "value": [1.0],
            "label": [0.01],
        }
    )


def _interim_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_asset_id": ["asset_a"],
            "timestamp": pd.to_datetime(["2014-01-02"]),
            "value": [1.0],
        }
    )


def _dataset_spec(
    dataset_id: str,
    logical_layer: str,
) -> InterimDataSpec | ProcessedDataSpec | ResearchDatasetSpec:
    source_base = {
        "frequency": DataFrequency.DAILY,
        "grain": DataGrain.ASSET_TIMESTAMP,
        "column_roles": {
            "source_asset_id": ColumnRole.KEY,
            "timestamp": ColumnRole.TIMESTAMP,
            "value": ColumnRole.FACTOR,
        },
        "primary_key": ("source_asset_id", "timestamp"),
        "partition_columns": ("year",),
    }
    market_base = {
        "frequency": DataFrequency.DAILY,
        "grain": DataGrain.ASSET_TIMESTAMP,
        "column_roles": {
            "market_id": ColumnRole.KEY,
            "asset_id": ColumnRole.KEY,
            "timestamp": ColumnRole.TIMESTAMP,
            "value": ColumnRole.FACTOR,
            "label": ColumnRole.LABEL,
        },
        "primary_key": ("market_id", "asset_id", "timestamp"),
        "partition_columns": ("year",),
    }
    if logical_layer == "interim":
        return InterimDataSpec(
            **source_base,
            data_id=dataset_id,
            source_id=SourceId("csmar"),
        )
    if logical_layer == "processed":
        return ProcessedDataSpec(
            **market_base,
            data_id=dataset_id,
            market_id=MarketId("a_share"),
            kind=ProcessedDataKind.FACTOR,
        )
    if logical_layer == "research":
        return ResearchDatasetSpec(
            **market_base,
            data_id=dataset_id,
            market_id=MarketId("a_share"),
            input_data_ids=("price_volume",),
            base_input_data_id="price_volume",
            factor_columns=("value",),
            feature_columns=(),
            label_columns=("label",),
        )
    raise ValueError(f"unsupported_logical_layer: {logical_layer}")
def test_local_verification_workflow_runs_injected_generation_and_writes_artifacts(
    tmp_path,
    monkeypatch,
):
    calls: list[str] = []
    run_years: list[tuple[int, ...]] = []
    evaluation_run_ids: list[str] = []
    prediction_ids: list[str] = []
    portfolio_ids: list[str] = []

    def import_interim(
        layout,
        raw_root,
        start_date,
        end_date,
    ):
        del raw_root, start_date, end_date
        calls.append("interim")
        artifact_root = (
            layout.project_root
            / "data"
            / "interim"
            / "csmar"
            / frequency_ids.MARKET_BARS_DATA_ID
        )
        _persist_fake_artifact(
            artifact_root,
            frequency_ids.MARKET_BARS_DATA_ID,
            "interim",
        )
        return type(
            "Result",
            (),
            {"artifact_roots": {frequency_ids.MARKET_BARS_DATA_ID: artifact_root}},
        )()

    def build_processed(
        layout,
        source_id,
        time_window,
        kinds,
    ):
        del source_id, time_window, kinds
        calls.append("processed")
        artifact_root = (
            layout.project_root
            / "data"
            / "processed"
            / "a_share"
            / "factor"
            / "price_volume"
        )
        _persist_fake_artifact(artifact_root, "price_volume", "processed")
        inspection = inspect_dataset(artifact_root)
        return {"price_volume": inspection}

    def assemble_dataset(**kwargs):
        calls.append("research")
        artifact_root = (
            kwargs["layout"].project_root
            / "data"
            / "research"
            / "a_share"
            / "price_volume_forward_returns"
        )
        _persist_fake_artifact(
            artifact_root,
            "price_volume_forward_returns",
            "research",
        )
        return artifact_root

    monkeypatch.setattr(module, "assemble_research_dataset", assemble_dataset)
    _stub_output_writers(monkeypatch)
    monkeypatch.setattr(
        module,
        "build_evaluation_result_from_research_dataset",
        lambda **kwargs: (
            calls.append("evaluation")
            or run_years.append((kwargs["time_window"].start.year,))
            or evaluation_run_ids.append(kwargs["run_id"])
            or _fake_evaluation_result(kwargs)
        ),
    )
    monkeypatch.setattr(
        module,
        "build_prediction_result_from_research_dataset",
        lambda **kwargs: (
            calls.append("prediction")
            or run_years.append((kwargs["time_window"].start.year,))
            or prediction_ids.append(kwargs["prediction_id"])
            or _fake_prediction_result(kwargs)
        ),
    )
    monkeypatch.setattr(
        module,
        "build_portfolio_result_from_research_dataset",
        lambda **kwargs: (
            calls.append("portfolio")
            or run_years.append((kwargs["time_window"].start.year,))
            or portfolio_ids.append(kwargs["portfolio_id"])
            or _fake_portfolio_result(kwargs)
        ),
    )

    profile = _profile_with_materializers(import_interim, build_processed)
    expected_prediction_ids = [
        prediction_spec.prediction_id for prediction_spec in _smoke_prediction_specs()
    ]
    expected_prediction_count = len(expected_prediction_ids)
    run_root = _repo_run_root(tmp_path, monkeypatch, ".run", "s1")
    result = run_local_verification(
        _verification_config(
            run_root=run_root,
            start_date="2014-01-01",
            end_date="2016-12-31",
            profile=profile,
        )
    )

    assert calls == [
        "interim",
        "processed",
        "research",
        "evaluation",
        *(["prediction"] * expected_prediction_count),
        "portfolio",
    ]
    assert run_years == [(2014,)] * (expected_prediction_count + 2)
    assert evaluation_run_ids == ["signal_forward_return_20d"]
    assert prediction_ids == expected_prediction_ids
    assert portfolio_ids == ["signal_forward_return_20d_portfolio"]
    assert result.status == "NOT_RUN"
    assert result.run_id == "s1"
    assert result.run_root == "s1"
    layout = LocalVerificationLayout.from_run_root(run_root)
    run_payload = json.loads(layout.run_json_path.read_text(encoding="utf-8"))
    assert layout.summary_md_path.is_file()
    assert run_payload["schema_id"] == "qrp-run"
    assert run_payload["status"] == {
        "overall": "NOT_RUN",
        "validation": "PASS",
        "smoke": "PASS",
        "compare": "NOT_RUN",
        "share_safety": "PASS",
        "full_horizon": "NOT_RUN",
        "validation_matrix": "NOT_RUN",
    }
    assert run_payload["run_id"] == "s1"
    assert run_payload["root_ownership"]["replace_safe"] is True
    dataset_manifest_paths = sorted(
        (run_root / "data").rglob("_manifest/manifest.json")
    )
    expected_layers = {
        path.relative_to(run_root).parts[1] for path in dataset_manifest_paths
    }
    actual_layers = {
        layer["logical_layer"] for layer in run_payload["data"]["layers"]
    }
    represented_manifests = {
        dataset["manifest_path"]
        for layer in run_payload["data"]["layers"]
        for dataset in layer["datasets"]
    }
    expected_manifests = {
        path.relative_to(run_root).as_posix() for path in dataset_manifest_paths
    }
    assert actual_layers == expected_layers
    assert "evidence" not in actual_layers
    assert represented_manifests == expected_manifests
    assert len(run_payload["outputs"]) == expected_prediction_count + 2
    assert all(
        output["schema_id"] == "qrp-output-manifest"
        for output in run_payload["outputs"]
    )
    timing_steps = [
        record["step_name"] for record in run_payload["execution_metrics"]["records"]
    ]
    validation_record = next(
        record
        for record in run_payload["execution_metrics"]["records"]
        if record["step_name"] == "validation_summary"
    )
    assert "validation_summary" in timing_steps
    assert "rss_start_end_max_mb" in validation_record
    assert LEGACY_RSS_PEAK_FIELD not in validation_record
    assert {path.name for path in run_root.iterdir()} == {
        "data",
        "logs",
        "outputs",
        "run.json",
        "summary.md",
    }



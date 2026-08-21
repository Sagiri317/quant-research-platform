from __future__ import annotations

import json
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from quant_research_platform.artifacts._output_manifest import (
    _OUTPUT_DEFINITION_FILES,
    portfolio_output_metadata_from_row_counts,
    prediction_output_metadata,
)
from quant_research_platform.artifacts._portfolio_output_validation import (
    PORTFOLIO_ARTIFACTS,
)
from quant_research_platform.catalog.portfolios import PortfolioCatalog
from quant_research_platform.catalog.predictions import PredictionCatalog
from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.data import (
    DataFrequency,
    DataGrain,
    InterimDataSpec,
    ProcessedDataKind,
    ProcessedDataSpec,
)
from quant_research_platform.contracts.datasets import ResearchDatasetSpec
from quant_research_platform.contracts.filtering import (
    filter_definition,
    required_filter_columns,
)
from quant_research_platform.contracts.ids import MarketId, SourceId
from quant_research_platform.markets.implementations.a_share import (
    research_scenarios as a_share_research_scenarios,
)
from quant_research_platform.profiles.research import research_profile
from quant_research_platform.research.portfolios.runner import PortfolioResult
from quant_research_platform.research.predictions.outputs import PredictionResult
from quant_research_platform.runtime.resources import current_runtime_resource_policy
from quant_research_platform.storage._dataset_inspection import (
    inspect_dataset as inspect_dataset_with_policy,
)
from quant_research_platform.storage._dataset_writing import (
    write_dataset as write_dataset_with_policy,
)
from quant_research_platform.workflows import local_verification as module
from quant_research_platform.workflows.local_verification import (
    LocalVerificationRunConfig,
)
from tests.helpers import frequency_ids
from tests.helpers.report_output_frames import make_artifact_prediction_output_frame

MIB = 1024 * 1024
GIB = 1024 * 1024 * 1024
SMOKE_PREDICTION_TRAIN_END = "2014-06-30"
SMOKE_PREDICTION_TEST_START = "2014-07-01"
SMOKE_PORTFOLIO_FREQUENCY = frequency_ids.DAILY_FREQUENCY_SLUG
SMOKE_PORTFOLIO_EXECUTION_LAG_STEPS = 1


def make_local_verification_config(**overrides: object) -> LocalVerificationRunConfig:
    payload = {
        "prediction_train_end": SMOKE_PREDICTION_TRAIN_END,
        "prediction_test_start": SMOKE_PREDICTION_TEST_START,
        "portfolio_frequency": SMOKE_PORTFOLIO_FREQUENCY,
        "portfolio_execution_lag_steps": SMOKE_PORTFOLIO_EXECUTION_LAG_STEPS,
    }
    payload.update(overrides)
    return LocalVerificationRunConfig(**payload)


def smoke_prediction_specs():
    return a_share_research_scenarios.build_prediction_catalog(
        train_end=SMOKE_PREDICTION_TRAIN_END,
        test_start=SMOKE_PREDICTION_TEST_START,
    ).list()


def repo_run_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *parts: str,
) -> Path:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(exist_ok=True)
    (repo_root / ".git").write_text("gitdir: ../repo.git\n", encoding="utf-8")
    monkeypatch.chdir(repo_root)
    return repo_root.joinpath(*parts)


def persist_fake_artifact(
    artifact_root: Path,
    dataset_id: str,
    logical_layer: str,
) -> None:
    frame = _interim_frame() if logical_layer == "interim" else _dataset_frame()
    write_dataset_with_policy(
        artifact_root,
        frame,
        spec=_dataset_spec(dataset_id, logical_layer),
        resource_policy=current_runtime_resource_policy(),
        borrowed_input=True,
    )


def inspect_fake_artifact(artifact_root: Path):
    return inspect_dataset_with_policy(
        artifact_root,
        resource_policy=current_runtime_resource_policy(),
    )


def persist_fake_research_root(**kwargs) -> Path:
    artifact_root = (
        kwargs["layout"].project_root
        / "data"
        / "research"
        / "a_share"
        / "price_volume_forward_returns"
    )
    persist_fake_artifact(
        artifact_root,
        "price_volume_forward_returns",
        "research",
    )
    return artifact_root


def stub_interim_importer(
    *,
    on_call=None,
    on_artifact=None,
    missing_manifest: bool = False,
):
    def materialize_interim(
        layout,
        raw_root,
        start_date,
        end_date,
        resource_policy,
    ):
        del raw_root, start_date, end_date, resource_policy
        if on_call is not None:
            on_call()
        artifact_root = (
            layout.project_root
            / "data"
            / "interim"
            / "csmar"
            / frequency_ids.MARKET_BARS_DATA_ID
        )
        if missing_manifest:
            (artifact_root / "data").mkdir(parents=True)
        else:
            persist_fake_artifact(
                artifact_root,
                frequency_ids.MARKET_BARS_DATA_ID,
                "interim",
            )
            if on_artifact is not None:
                on_artifact(
                    artifact_root, inspect_fake_artifact(artifact_root).manifest
                )
        return SimpleNamespace(
            artifact_roots={frequency_ids.MARKET_BARS_DATA_ID: artifact_root}
        )

    return materialize_interim


def stub_processed_builder(*, on_call=None, materialize=None):
    def materialize_processed(
        layout,
        source_id,
        time_window,
        kinds,
        resource_policy,
    ):
        del source_id, time_window, resource_policy
        if on_call is not None:
            on_call()
        if materialize is not None:
            return materialize(layout, kinds)
        del kinds
        artifact_root = (
            layout.project_root / "data" / "processed" / "a_share" / "factor"
        ) / "price_volume"
        persist_fake_artifact(artifact_root, "price_volume", "processed")
        return {"price_volume": inspect_fake_artifact(artifact_root)}

    return materialize_processed


def empty_processed_inputs(
    layout,
    source_id,
    time_window,
    kinds,
    resource_policy,
) -> dict[str, object]:
    del layout, source_id, time_window, kinds, resource_policy
    return {}


def profile_with_materializers(interim, processed):
    return replace(
        research_profile(),
        materialize_interim_data=interim,
        materialize_processed_data=processed,
    )


def profile_with_empty_prediction_and_portfolio_builders(profile=None):
    return replace(
        profile or research_profile(),
        build_prediction_catalog=lambda **_: PredictionCatalog(),
        build_portfolio_catalog=lambda **_: PortfolioCatalog(),
    )


def fake_prediction_result(kwargs):
    return PredictionResult(
        spec=kwargs["prediction_catalog"].get(kwargs["prediction_id"]),
        output_chunks=(),
        model_metadata={},
        prediction_metadata_factory=lambda: {},
    )


def fake_portfolio_result(kwargs):
    return PortfolioResult(
        spec=kwargs["portfolio_catalog"].get(kwargs["portfolio_id"]),
        output_chunks=(),
        summary_factory=lambda: {},
    )


def stub_output_writers(monkeypatch: pytest.MonkeyPatch):
    def write_stub(*args, **kwargs):
        del args
        output_spec = kwargs["output_spec"]
        root = kwargs["layout"].project_root / "outputs" / output_spec.output_id
        root.mkdir(parents=True)
        files = list(_OUTPUT_DEFINITION_FILES[output_spec.output_definition_id])
        for name in files:
            (root / name).write_bytes(b"{}\n")
        (root / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_id": "qrp-output-manifest",
                    "output_id": output_spec.output_id,
                    "output_definition_id": output_spec.output_definition_id,
                    "run_id": output_spec.run_id,
                    "metadata": _stub_output_metadata(
                        output_spec.output_definition_id,
                        kwargs,
                    ),
                    "files": files,
                }
            ),
            encoding="utf-8",
        )
        return root

    def write_evaluation_stub(*args, **kwargs):
        del args
        stub_kwargs = dict(kwargs)
        stub_kwargs["spec"] = stub_kwargs["evaluation_catalog"].get(
            stub_kwargs["run_id"]
        )
        return write_stub(**stub_kwargs)

    monkeypatch.setattr(
        module,
        "_write_evaluation_outputs_from_research_dataset",
        write_evaluation_stub,
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
    monkeypatch.setattr(
        module,
        "_read_prediction_output_bundle",
        lambda *_args, **_kwargs: nullcontext(({"metadata": {}}, pd.DataFrame())),
    )
    return write_evaluation_stub


def _stub_output_metadata(
    output_definition_id: str,
    kwargs: dict[str, object],
) -> dict[str, object]:
    spec = kwargs["spec"]
    if output_definition_id == "evaluation_outputs":
        return {
            "run_id": spec.run_id,
            "run_type": spec.run_type,
            "market_id": str(spec.market_id),
            "research_dataset_id": spec.research_dataset_id,
            "signals": [
                {
                    "signal_id": signal.signal_id,
                    "source_kind": signal.source_kind,
                    "source_column": signal.source_column,
                    "direction": signal.direction,
                    "transform": signal.transform,
                }
                for signal in spec.signals
            ],
            "label_columns": list(spec.label_columns),
            "filter_mode": spec.filter_mode,
            "evaluation_diagnostics": {
                "filter_mode": spec.filter_mode,
                "filter_columns_used": list(required_filter_columns(spec.filter_mode)),
                "filter_definition": filter_definition(spec.filter_mode),
                "rows_input": 0,
                "rows_after_filter_mode": 0,
                "rows_after_signal_label_dropna": 0,
                "metric_rows": 0,
                "grouped_label_value_rows": 0,
                "ic_summary_rows": 0,
                "group_spread_summary_rows": 0,
                "timestamp_count": 0,
                "min_timestamp": None,
                "max_timestamp": None,
                "n_groups": spec.n_groups,
                "skipped_timestamp_count": 0,
            },
        }
    if output_definition_id == "prediction_outputs":
        frame = make_artifact_prediction_output_frame()
        effective_end = spec.test_end or spec.test_start
        model_metadata = {
            **frame.attrs["model_metadata"],
            "train_start": spec.train_end,
            "train_end": spec.train_end,
            "test_start": spec.test_start,
            "test_end": spec.test_end,
            "oos_start": spec.test_start,
            "oos_end": effective_end,
        }
        prediction_metadata = {
            **frame.attrs["prediction_metadata"],
            "prediction_mode": spec.prediction_mode,
            "sample_filter_mode": spec.sample_filter_mode,
            "train_end": spec.train_end,
            "test_start": spec.test_start,
            "requested_test_end": spec.test_end,
            "effective_test_end": effective_end,
            "is_oos": spec.is_oos,
            "is_final_oos_window": spec.is_final_oos_window,
            "prediction_column": spec.prediction_column,
        }
        return prediction_output_metadata(
            spec=spec,
            model_metadata=model_metadata,
            prediction_metadata=prediction_metadata,
        )
    if output_definition_id == "portfolio_outputs":
        return portfolio_output_metadata_from_row_counts(
            spec=spec,
            row_counts={name: 0 for name in PORTFOLIO_ARTIFACTS},
        )
    raise AssertionError(f"unexpected_output_definition:{output_definition_id}")


def persist_tiny_a_share_processed_inputs(
    layout,
    profile,
) -> dict[str, object]:
    dataset_spec = profile.research_datasets.get("price_volume_forward_returns")
    training_periods = max(model.purge_gap_steps for model in profile.models.list()) + 2
    inspections: dict[str, object] = {}
    for data_id in dataset_spec.input_data_ids:
        spec = profile.processed.get(data_id)
        root = layout.processed_data_root(spec.market_id, spec.kind, spec.data_id)
        write_dataset_with_policy(
            root,
            _tiny_a_share_processed_frame(spec, training_periods=training_periods),
            spec=spec,
            resource_policy=current_runtime_resource_policy(),
            borrowed_input=True,
        )
        inspections[data_id] = inspect_fake_artifact(root)
    return inspections


def _tiny_a_share_processed_frame(
    spec: ProcessedDataSpec,
    *,
    training_periods: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dates = pd.bdate_range("2014-01-02", periods=training_periods).append(
        pd.to_datetime(["2014-07-02", "2015-01-05"])
    )
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

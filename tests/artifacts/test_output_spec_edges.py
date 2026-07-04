from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from quant_research_platform.artifacts import (
    _output_common_validation as output_common_validation,
)
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.outputs import OutputScope, OutputSpec
from quant_research_platform.storage.layout import StorageLayout
from tests.artifacts.output_edge_fixtures import (
    make_output_edge_report_spec,
)


@pytest.mark.parametrize(
    "output_id",
    (
        "",
        "../escape",
        "nested/output",
        "nested\\output",
        "C:\\absolute",
        "CON",
        "aux.txt",
        "free text",
        "with.dot",
    ),
)
def test_output_spec_rejects_filesystem_unsafe_output_id(output_id: str) -> None:
    with pytest.raises(ValueError, match="invalid_output_id"):
        OutputSpec(
            output_id=output_id,
            output_definition_id="prediction_outputs",
            run_id="prediction_run",
            market_id=MarketId("a_share"),
            run_type="prediction",
            scope=OutputScope.LOCAL_ONLY,
        )


def test_output_spec_rejects_output_definition_id_as_output_id() -> None:
    with pytest.raises(ValueError, match="output_id_must_identify_instance"):
        OutputSpec(
            output_id="prediction_outputs",
            output_definition_id="prediction_outputs",
            run_id="prediction_run",
            market_id=MarketId("a_share"),
            run_type="prediction",
            scope=OutputScope.LOCAL_ONLY,
        )


def test_report_frame_validation_rejects_missing_and_extra_columns() -> None:
    with pytest.raises(ValueError, match="missing_evaluation_metrics_columns: ic"):
        output_common_validation.validate_output_frame(
            pd.DataFrame({"timestamp": [pd.Timestamp("2024-01-02")]}),
            required=("timestamp", "ic"),
            error_name="evaluation_metrics",
        )
    with pytest.raises(ValueError, match="unregistered_evaluation_metrics_columns"):
        output_common_validation.validate_output_frame(
            pd.DataFrame(
                {
                    "timestamp": [pd.Timestamp("2024-01-02")],
                    "ic": [0.1],
                    "extra": [1],
                }
            ),
            required=("timestamp", "ic"),
            error_name="evaluation_metrics",
        )


def test_output_bundle_root_rejects_mismatched_specs(tmp_path: Path) -> None:
    layout = StorageLayout(project_root=tmp_path)

    with pytest.raises(ValueError, match="output_spec_mismatch"):
        output_common_validation.derive_output_bundle_root(
            output_spec=make_output_edge_report_spec(
                "unknown",
                "run_demo",
                "prediction",
            ),
            layout=layout,
            expected_definition_id="prediction_outputs",
        )
    with pytest.raises(ValueError, match="output_spec_mismatch"):
        output_common_validation.derive_output_bundle_root(
            output_spec=make_output_edge_report_spec(
                "evaluation_outputs",
                "run_demo",
                "evaluation",
            ),
            layout=layout,
            expected_definition_id="prediction_outputs",
        )


@pytest.mark.parametrize(
    "data_layer",
    ["raw", "interim", "processed", "research"],
)
def test_output_bundle_root_rejects_dataset_roots(
    tmp_path: Path,
    data_layer: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        output_common_validation,
        "output_bundle_root",
        lambda _root, _output_id: tmp_path / "data" / data_layer / "report",
    )

    with pytest.raises(ValueError, match="output_bundle_must_not_be_dataset_root"):
        output_common_validation.derive_output_bundle_root(
            output_spec=make_output_edge_report_spec(
                "prediction_outputs",
                "prediction_demo",
                "prediction",
            ),
            layout=StorageLayout(project_root=tmp_path),
            expected_definition_id="prediction_outputs",
        )

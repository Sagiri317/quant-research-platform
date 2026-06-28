from __future__ import annotations

import inspect
from pathlib import Path

import pandas as pd

from quant_research_platform.artifacts import outputs
from quant_research_platform.artifacts.definitions import output_catalog
from quant_research_platform.artifacts.outputs import (
    write_stage_profile_outputs,
)
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.outputs import OutputScope, OutputSpec
from quant_research_platform.storage.layout import StorageLayout


def test_stage_profile_outputs_write_under_output_id_root(
    tmp_path: Path,
) -> None:
    spec = OutputSpec(
        output_id="stage_profile",
        output_definition_id="stage_profile_diagnostic",
        run_id="stage_profile_2026",
        market_id=MarketId("a_share"),
        run_type="diagnostics",
        scope=OutputScope.LOCAL_ONLY,
    )

    paths = write_stage_profile_outputs(
        pd.DataFrame([{"category": "x", "stage": "y", "seconds": 1.0}]),
        spec=spec,
        markdown_text="# Stage\n",
        layout=StorageLayout(project_root=tmp_path),
    )

    assert paths["markdown"].is_file()
    assert "outputs/stage_profile/stage_profile.md" in paths["markdown"].as_posix()


def test_exported_output_writers_do_not_expose_from_chunks_names() -> None:
    exported_names = set(outputs.__all__)

    assert not [name for name in exported_names if "from_chunks" in name]
    assert "write_prediction_outputs" in exported_names
    assert "write_portfolio_outputs" in exported_names


def test_output_catalog_definitions_are_local_only() -> None:
    definitions = output_catalog().list()

    assert definitions
    assert {definition.scope for definition in definitions} == {OutputScope.LOCAL_ONLY}


def test_writer_signature_uses_output_chunks_parameter() -> None:
    signature = inspect.signature(outputs.write_portfolio_outputs)

    assert "output_chunks" in signature.parameters

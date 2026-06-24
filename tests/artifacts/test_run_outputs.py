from __future__ import annotations

import inspect
from pathlib import Path

import pandas as pd

from quant_research_platform.artifacts import run_outputs
from quant_research_platform.artifacts.definitions import run_output_catalog
from quant_research_platform.artifacts.run_outputs import (
    write_stage_profile_run_outputs,
)
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.run_outputs import OutputScope, RunOutputSpec
from quant_research_platform.storage.layout import StorageLayout


def test_stage_profile_run_outputs_write_under_local_run_output_root(
    tmp_path: Path,
) -> None:
    spec = RunOutputSpec(
        output_id="stage_profile",
        output_definition_id="stage_profile_diagnostic",
        run_id="stage_profile_2026",
        market_id=MarketId("a_share"),
        run_type="diagnostics",
        scope=OutputScope.LOCAL_ONLY,
    )

    paths = write_stage_profile_run_outputs(
        pd.DataFrame([{"category": "x", "stage": "y", "seconds": 1.0}]),
        spec=spec,
        markdown_text="# Stage\n",
        layout=StorageLayout(project_root=tmp_path),
    )

    assert paths["markdown"].is_file()
    assert (
        "run_outputs/local/a_share/diagnostics/stage_profile_2026"
        in paths["markdown"].as_posix()
    )


def test_public_run_output_writers_do_not_expose_from_chunks_names() -> None:
    public_names = set(run_outputs.__all__)

    assert not [name for name in public_names if "from_chunks" in name]
    assert "write_prediction_run_outputs" in public_names
    assert "write_portfolio_run_outputs" in public_names


def test_run_output_catalog_definitions_are_local_only() -> None:
    definitions = run_output_catalog().list()

    assert definitions
    assert {definition.scope for definition in definitions} == {OutputScope.LOCAL_ONLY}


def test_writer_signature_uses_output_chunks_parameter() -> None:
    signature = inspect.signature(run_outputs.write_portfolio_run_outputs)

    assert "output_chunks" in signature.parameters

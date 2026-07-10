from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import quant_research_platform.artifacts._parquet_writers as parquet_writers
import quant_research_platform.artifacts._portfolio_outputs as portfolio_outputs
from quant_research_platform.storage.layout import StorageLayout
from tests.artifacts.output_edge_fixtures import (
    make_output_edge_portfolio_artifacts,
    make_output_edge_portfolio_spec,
    make_output_edge_report_spec,
    write_output_edge_portfolio_chunks,
)


def test_portfolio_outputs_clean_temp_bundle_directory_on_failure(
    monkeypatch,
    tmp_path: Path,
) -> None:
    spec = make_output_edge_portfolio_spec()
    layout = StorageLayout(project_root=tmp_path)
    report_spec = make_output_edge_report_spec(
        "portfolio_outputs",
        spec.portfolio_id,
        "portfolio",
    )
    run_parent = tmp_path / "outputs"

    def fail_write(frame, path, **kwargs):  # noqa: ANN001
        del frame, kwargs
        Path(path).write_text("partial", encoding="utf-8")
        raise RuntimeError("portfolio parquet failed")

    monkeypatch.setattr(portfolio_outputs, "write_parquet", fail_write)

    with pytest.raises(RuntimeError, match="portfolio parquet failed"):
        write_output_edge_portfolio_chunks(
            make_output_edge_portfolio_artifacts(),
            spec=spec,
            output_spec=report_spec,
            layout=layout,
        )
    assert not (run_parent / report_spec.output_id).exists()
    assert not list(run_parent.glob(".tmp-*"))


def test_report_parquet_writer_cleans_temp_on_failure(
    monkeypatch,
    tmp_path: Path,
) -> None:
    frame = pd.DataFrame({"value": [1]})

    def fail_to_parquet(self, path, **kwargs):  # noqa: ANN001
        Path(path).write_text("temporary", encoding="utf-8")
        raise RuntimeError("write failed")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail_to_parquet)

    with pytest.raises(RuntimeError, match="write failed"):
        parquet_writers.write_parquet(
            frame,
            tmp_path / "report.parquet",
        )
    assert not list(tmp_path.glob("*.tmp-*"))


def test_report_parquet_writer_does_not_overwrite_unclaimed_temp_collision(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    final_path = tmp_path / "report.parquet"
    live_temp = final_path.with_name(f".report.parquet.tmp-{'c' * 12}")
    live_temp.write_text("live\n", encoding="utf-8")
    monkeypatch.setattr(
        parquet_writers,
        "uuid4",
        lambda: type("FixedUuid", (), {"hex": "c" * 32})(),
    )

    with pytest.raises(FileExistsError):
        parquet_writers.write_parquet(pd.DataFrame({"value": [1]}), final_path)

    assert live_temp.read_text(encoding="utf-8") == "live\n"
    assert not final_path.exists()

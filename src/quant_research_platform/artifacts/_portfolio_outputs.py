"""Portfolio output writer."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from quant_research_platform.artifacts._output_common_validation import (
    derive_output_bundle_root,
    validate_output_frame,
)
from quant_research_platform.artifacts._output_manifest import (
    output_manifest_payload,
    portfolio_output_metadata_from_row_counts,
)
from quant_research_platform.artifacts._parquet_writers import write_parquet
from quant_research_platform.artifacts._portfolio_output_validation import (
    PORTFOLIO_ARTIFACTS,
    portfolio_required_columns,
    validate_portfolio_signal_column,
)
from quant_research_platform.contracts.outputs import OutputSpec
from quant_research_platform.contracts.portfolios import PortfolioSpec
from quant_research_platform.storage.atomic import atomic_write_directory
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.storage.manifests import write_manifest


def write_portfolio_outputs(
    *,
    output_chunks: Iterable[Mapping[str, pd.DataFrame]],
    summary_factory: Callable[[], Mapping[str, pd.DataFrame]],
    spec: PortfolioSpec,
    output_spec: OutputSpec,
    layout: StorageLayout,
) -> Path:
    """Write local portfolio outputs."""

    return _write_portfolio_output_chunks(
        output_chunks,
        summary_factory=summary_factory,
        spec=spec,
        output_spec=output_spec,
        layout=layout,
    )


def _write_portfolio_output_chunks(
    output_chunks: Iterable[Mapping[str, pd.DataFrame]],
    *,
    summary_factory: Callable[[], Mapping[str, pd.DataFrame]],
    spec: PortfolioSpec,
    output_spec: OutputSpec,
    layout: StorageLayout,
) -> Path:
    """Write local portfolio row outputs from bounded chunks."""

    output_bundle = derive_output_bundle_root(
        output_spec=output_spec,
        layout=layout,
        expected_definition_id="portfolio_outputs",
    )
    required_columns = portfolio_required_columns()
    row_artifacts = (
        "positions",
        "turnover",
        "cost_breakdown",
        "portfolio_returns",
        "diagnostics",
        "capacity_diagnostics",
        "numeric_exposure_diagnostics",
        "categorical_exposure_diagnostics",
    )
    summary_names = tuple(
        name for name in PORTFOLIO_ARTIFACTS if name not in row_artifacts
    )

    def _write(temp_root: Path) -> None:
        row_counts = _write_portfolio_row_artifact_chunks(
            output_chunks,
            temp_root=temp_root,
            row_artifacts=row_artifacts,
            required_columns=required_columns,
            spec=spec,
        )
        summary_outputs = summary_factory()
        for artifact_name in summary_names:
            validate_output_frame(
                summary_outputs[artifact_name],
                required=required_columns[artifact_name],
                error_name=f"portfolio_{artifact_name}",
            )
            validate_portfolio_signal_column(
                summary_outputs[artifact_name],
                artifact_name=artifact_name,
                spec=spec,
            )
            write_parquet(
                summary_outputs[artifact_name],
                temp_root / PORTFOLIO_ARTIFACTS[artifact_name],
                engine="pyarrow",
                compression="zstd",
                index=False,
            )
            row_counts[artifact_name] = int(len(summary_outputs[artifact_name]))
        write_manifest(
            temp_root / "manifest.json",
            output_manifest_payload(
                output_spec=output_spec,
                metadata=portfolio_output_metadata_from_row_counts(
                    spec=spec,
                    row_counts=row_counts,
                ),
                files=tuple(PORTFOLIO_ARTIFACTS.values()),
            ),
        )

    atomic_write_directory(output_bundle, _write)
    return output_bundle


def _write_portfolio_row_artifact_chunks(
    artifact_chunks: Iterable[Mapping[str, pd.DataFrame]],
    *,
    temp_root: Path,
    row_artifacts: tuple[str, ...],
    required_columns: Mapping[str, tuple[str, ...]],
    spec: PortfolioSpec,
) -> dict[str, int]:
    writers: dict[str, pq.ParquetWriter] = {}
    temp_paths = {
        artifact_name: temp_root / PORTFOLIO_ARTIFACTS[artifact_name]
        for artifact_name in row_artifacts
    }
    row_counts = {artifact_name: 0 for artifact_name in PORTFOLIO_ARTIFACTS}
    try:
        for artifact_chunk in artifact_chunks:
            for artifact_name in row_artifacts:
                frame = artifact_chunk[artifact_name]
                selected = _portfolio_output_chunk_frame(
                    frame,
                    artifact_name=artifact_name,
                    required=required_columns[artifact_name],
                    spec=spec,
                )
                if selected.empty:
                    continue
                table = pa.Table.from_pandas(selected, preserve_index=False)
                writer = writers.get(artifact_name)
                if writer is None:
                    writer = pq.ParquetWriter(
                        temp_paths[artifact_name],
                        table.schema,
                        compression="zstd",
                    )
                    writers[artifact_name] = writer
                writer.write_table(table)
                row_counts[artifact_name] += int(len(selected))
        for writer in writers.values():
            writer.close()
        writers.clear()
        for artifact_name in row_artifacts:
            if row_counts[artifact_name] == 0:
                write_parquet(
                    pd.DataFrame(columns=list(required_columns[artifact_name])),
                    temp_paths[artifact_name],
                    engine="pyarrow",
                    compression="zstd",
                    index=False,
                )
    except Exception:
        for writer in writers.values():
            writer.close()
        raise
    return row_counts


def _portfolio_output_chunk_frame(
    frame: pd.DataFrame,
    *,
    artifact_name: str,
    required: tuple[str, ...],
    spec: PortfolioSpec,
) -> pd.DataFrame:
    validate_output_frame(
        frame,
        required=required,
        error_name=f"portfolio_{artifact_name}",
    )
    if "signal" in frame.columns:
        signal_ids = {signal.signal_id for signal in spec.signals}
        artifact_signals = {
            str(value) for value in frame["signal"].dropna().unique().tolist()
        }
        if not artifact_signals.issubset(signal_ids):
            raise ValueError("portfolio_artifact_signal_mismatch: " + artifact_name)
    if artifact_name == "portfolio_returns":
        return_sources = {
            str(value)
            for value in frame["return_source_column"].dropna().unique().tolist()
        }
        if return_sources and return_sources != {spec.return_source_column}:
            raise ValueError("portfolio_return_source_column_mismatch")
    return frame.loc[:, list(required)]

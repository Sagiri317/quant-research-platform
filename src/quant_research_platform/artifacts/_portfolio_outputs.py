"""Portfolio output writer."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from contextlib import ExitStack, nullcontext
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
from quant_research_platform.artifacts._parquet_writers import (
    _artifact_frame_reservation,
    _artifact_table_write_state,
    write_parquet,
)
from quant_research_platform.artifacts._portfolio_output_validation import (
    PORTFOLIO_ARTIFACTS,
    portfolio_required_columns,
    validate_portfolio_artifact_mapping_keys,
    validate_portfolio_artifact_membership_columns,
)
from quant_research_platform.contracts.outputs import OutputSpec
from quant_research_platform.contracts.portfolios import PortfolioSpec
from quant_research_platform.runtime.resources import (
    current_runtime_resource_policy,
    pandas_frame_payload_bytes,
)
from quant_research_platform.storage._dataset_reading import closing_dataset_iterator
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
    _finalize_memory: bool = False,
) -> Path:
    """Write local portfolio outputs."""

    return _write_portfolio_output_chunks(
        output_chunks,
        summary_factory=summary_factory,
        spec=spec,
        output_spec=output_spec,
        layout=layout,
        finalize_memory=_finalize_memory,
    )


def _write_portfolio_output_chunks(
    output_chunks: Iterable[Mapping[str, pd.DataFrame]],
    *,
    summary_factory: Callable[[], Mapping[str, pd.DataFrame]],
    spec: PortfolioSpec,
    output_spec: OutputSpec,
    layout: StorageLayout,
    finalize_memory: bool = False,
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
        summary_outputs = validate_portfolio_artifact_mapping_keys(
            summary_outputs,
            expected_names=summary_names,
            error="portfolio_summary_artifacts_invalid",
        )
        policy = current_runtime_resource_policy()
        summary_bytes = sum(
            pandas_frame_payload_bytes(summary_outputs[name]) for name in summary_names
        )
        with (
            policy.reserve("artifacts.portfolio.summaries", summary_bytes)
            if policy is not None
            else nullcontext()
        ):
            for artifact_name in summary_names:
                validate_output_frame(
                    summary_outputs[artifact_name],
                    required=required_columns[artifact_name],
                    error_name=f"portfolio_{artifact_name}",
                )
                validate_portfolio_artifact_membership_columns(
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
                row_counts[artifact_name] = len(summary_outputs[artifact_name])
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

    atomic_write_directory(
        output_bundle,
        _write,
        finalize_memory=finalize_memory,
    )
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
    writer_states: dict[str, ExitStack] = {}
    writer_memory: dict[str, tuple[int, int]] = {}
    decoded_metadata: dict[str, dict[bytes, bytes]] = {}
    temp_paths = {
        artifact_name: temp_root / PORTFOLIO_ARTIFACTS[artifact_name]
        for artifact_name in row_artifacts
    }
    row_counts = {artifact_name: 0 for artifact_name in PORTFOLIO_ARTIFACTS}
    try:
        with closing_dataset_iterator(iter(artifact_chunks)) as active_chunks:
            for artifact_chunk in active_chunks:
                artifact_chunk = validate_portfolio_artifact_mapping_keys(
                    artifact_chunk,
                    expected_names=row_artifacts,
                    error="portfolio_row_artifacts_invalid",
                )
                for artifact_name in row_artifacts:
                    frame = artifact_chunk[artifact_name]
                    with _artifact_frame_reservation(frame):
                        selected = _portfolio_output_chunk_frame(
                            frame,
                            artifact_name=artifact_name,
                            required=required_columns[artifact_name],
                            spec=spec,
                        )
                        if selected.empty:
                            continue
                        table = pa.Table.from_pandas(selected, preserve_index=False)
                        (
                            persisted,
                            table_state,
                            batch_rows,
                            next_writer_memory,
                        ) = (
                            _artifact_table_write_state(
                                table,
                                previous_metadata=decoded_metadata.get(artifact_name),
                                retained_memory=writer_memory.get(
                                    artifact_name,
                                    (0, 0),
                                ),
                            )
                        )
                        writer_memory[artifact_name] = next_writer_memory
                        writer_state = writer_states.setdefault(
                            artifact_name,
                            ExitStack(),
                        )
                        writer_state.enter_context(table_state)
                        decoded_metadata[artifact_name] = dict(
                            persisted.schema.metadata or {}
                        )
                        writer = writers.get(artifact_name)
                        if writer is None:
                            try:
                                writer = pq.ParquetWriter(
                                    temp_paths[artifact_name],
                                    persisted.schema,
                                    compression="zstd",
                                    write_batch_size=batch_rows,
                                    max_rows_per_page=batch_rows,
                                    use_dictionary=False,
                                    write_statistics=False,
                                    write_page_index=False,
                                    write_page_checksum=False,
                                )
                            except BaseException:
                                writer_states.pop(artifact_name).close()
                                writer_memory.pop(artifact_name)
                                raise
                            writers[artifact_name] = writer
                        writer.write_table(
                            persisted,
                            row_group_size=batch_rows,
                        )
                        row_counts[artifact_name] += len(selected)
            for artifact_name in tuple(writers):
                writer = writers[artifact_name]
                writer.add_key_value_metadata(decoded_metadata[artifact_name])
                active_writer = writers.pop(artifact_name)
                try:
                    active_writer.close()
                finally:
                    writer_states.pop(artifact_name).close()
            for artifact_name in row_artifacts:
                if row_counts[artifact_name] == 0:
                    write_parquet(
                        pd.DataFrame(columns=list(required_columns[artifact_name])),
                        temp_paths[artifact_name],
                        engine="pyarrow",
                        compression="zstd",
                        index=False,
                    )
    except BaseException:
        for artifact_name in tuple(writers):
            writer = writers.pop(artifact_name)
            try:
                writer.close()
            except BaseException:
                pass
            writer_states.pop(artifact_name).close()
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
    validate_portfolio_artifact_membership_columns(
        frame,
        artifact_name=artifact_name,
        spec=spec,
    )
    return frame.loc[:, list(required)]

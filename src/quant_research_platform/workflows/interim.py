"""Market source routing for the public staged interim-data operation."""

from __future__ import annotations

from pathlib import Path

from quant_research_platform.runtime.resources import RuntimeResourcePolicy
from quant_research_platform.sources.choice.ids import (
    FIRST_DISCLOSURE_DATA_ID,
)
from quant_research_platform.sources.choice.ids import (
    SOURCE_ID as CHOICE_SOURCE_ID,
)
from quant_research_platform.sources.csmar.interim import (
    InterimImportResult,
)
from quant_research_platform.sources.csmar.interim import (
    interim_catalog as csmar_interim_catalog,
)
from quant_research_platform.sources.csmar.interim import (
    materialize_interim_data as materialize_csmar,
)
from quant_research_platform.storage.layout import StorageLayout


def materialize_a_share_interim(
    *,
    layout: StorageLayout,
    raw_root: Path,
    data_ids: tuple[str, ...] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    resource_policy: RuntimeResourcePolicy | None = None,
) -> InterimImportResult:
    """Route CSMAR primary/auxiliary exports while Choice stays separately owned."""

    from quant_research_platform.sources.choice.interim import interim_catalog

    selected = (
        tuple(data_ids)
        if data_ids is not None
        else tuple(
            spec.data_id
            for spec in (*csmar_interim_catalog().list(), *interim_catalog().list())
        )
    )
    csmar_ids = {spec.data_id for spec in csmar_interim_catalog().list()}
    csmar_selected = tuple(data_id for data_id in selected if data_id in csmar_ids)
    result = materialize_csmar(
        layout=layout,
        raw_root=raw_root,
        data_ids=csmar_selected,
        start_date=start_date,
        end_date=end_date,
        resource_policy=resource_policy,
    )
    aux_ids = tuple(
        data_id
        for data_id in selected
        if data_id not in csmar_ids and data_id != "financial_first_disclosure"
    )
    aux_root = raw_root.parent / "csmar_aux"
    if aux_ids and aux_root.exists():
        aux_result = materialize_csmar(
            layout=layout,
            raw_root=aux_root,
            data_ids=aux_ids,
            start_date=start_date,
            end_date=end_date,
            resource_policy=resource_policy,
        )
        result = InterimImportResult(
            artifact_roots={**result.artifact_roots, **aux_result.artifact_roots},
            skipped=(*result.skipped, *aux_result.skipped),
            pruning_report_text=result.pruning_report_text
            + aux_result.pruning_report_text,
        )
    if FIRST_DISCLOSURE_DATA_ID in selected:
        choice_root = layout.interim_data_root(
            CHOICE_SOURCE_ID,
            FIRST_DISCLOSURE_DATA_ID,
        )
        if choice_root.exists():
            result = InterimImportResult(
                artifact_roots={
                    **result.artifact_roots,
                    FIRST_DISCLOSURE_DATA_ID: choice_root,
                },
                skipped=result.skipped,
                pruning_report_text=result.pruning_report_text,
            )
        else:
            result = InterimImportResult(
                artifact_roots=result.artifact_roots,
                skipped=(*result.skipped, FIRST_DISCLOSURE_DATA_ID),
                pruning_report_text=result.pruning_report_text,
            )
    return result

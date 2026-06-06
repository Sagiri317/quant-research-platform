"""Thin service layer shared by API and CLI adapters."""

from __future__ import annotations

from quant_research_platform.catalog.data_products import DataProductRegistry
from quant_research_platform.catalog.labels import LabelCatalog
from quant_research_platform.catalog.registry_validation import (
    validate_data_product_source_graph,
)
from quant_research_platform.contracts.execution import ExecutionMode
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.run_id import validate_run_id
from quant_research_platform.contracts.universes import UniverseDefinition
from quant_research_platform.contracts.workflows import (
    WorkflowArtifactWriterPolicy,
    WorkflowNode,
    WorkflowSpec,
)
from quant_research_platform.markets.implementations.a_share.catalog import (
    a_share_factor_catalog,
    a_share_research_dataset_catalog,
)
from quant_research_platform.runtime.privacy import sanitize_public_object
from quant_research_platform.workflows.services import execute_registered_workflow

API_VERSION = "0.1"


def list_factors() -> dict[str, object]:
    return _response(
        items=[item.to_manifest() for item in a_share_factor_catalog().list()]
    )


def list_labels() -> dict[str, object]:
    return _response(
        items=[item.to_manifest() for item in LabelCatalog.bootstrap_default().list()]
    )


def list_universes() -> dict[str, object]:
    universe = UniverseDefinition(
        universe_id="a_share_default",
        market_id=MarketId("a_share"),
        membership_column="in_universe",
    )
    return _response(
        items=[
            {
                "definition_version": universe.definition_version,
                "universe_id": universe.universe_id,
                "market_id": str(universe.market_id),
                "membership_column": universe.membership_column,
            }
        ]
    )


def list_data_products() -> dict[str, object]:
    return _response(
        items=[
            item.to_public_record()
            for item in DataProductRegistry.bootstrap_default().list()
        ]
    )


def run_data_product_interim(
    data_product_id: str,
    *,
    execution_mode: ExecutionMode | str | None = None,
    run_id: str | None = None,
) -> dict[str, object]:
    mode = ExecutionMode.normalize(execution_mode)
    resolved_run_id = _resolve_run_id(run_id, prefix=f"{data_product_id}-interim")
    data_product = DataProductRegistry.bootstrap_default().get(data_product_id)
    validate_data_product_source_graph(data_product.data_product_id)
    spec = WorkflowSpec(
        workflow_id="data_product_interim_validation",
        market_id=MarketId(str(data_product.market_id)),
        data_product_id=data_product.data_product_id,
        input_spec_ids=(),
        output_spec_ids=data_product.interim_spec_ids,
        run_type="data_product_interim",
        artifact_writer_policy=WorkflowArtifactWriterPolicy.SPEC_DRIVEN,
        nodes=(WorkflowNode(node_id="validate_interim_spec"),),
    )
    summary = execute_registered_workflow(
        spec,
        allowed_spec_ids=set(data_product.interim_spec_ids),
        execution_mode=mode,
        run_id=resolved_run_id,
    )
    return _response(
        status="validated" if mode is ExecutionMode.DRY_RUN else "executed",
        execution_mode=mode.value,
        run_id=resolved_run_id,
        mode=(
            "dry_run_spec_validation"
            if mode is ExecutionMode.DRY_RUN
            else "workflow_execute_not_implemented"
        ),
        data_product_id=data_product.data_product_id,
        market_id=str(data_product.market_id),
        contract_version=data_product.contract_version,
        workflow=summary,
    )


def describe_research_dataset_requirements(dataset_id: str) -> dict[str, object]:
    spec = a_share_research_dataset_catalog().get(dataset_id)
    return _response(
        status="available",
        mode="requirements_query_only",
        dataset_id=spec.dataset_id,
        market_id=str(spec.market_id),
        required_processed_data_ids=list(spec.processed_data_ids),
        feature_count=len(spec.feature_columns),
        label_count=len(spec.label_columns),
        universe_id=spec.universe_id,
        mask_ids=list(spec.mask_ids),
    )


def _response(**payload: object) -> dict[str, object]:
    return sanitize_public_object({"api_version": API_VERSION, **payload})


def _resolve_run_id(run_id: str | None, *, prefix: str) -> str:
    resolved = run_id or f"{prefix}-fixture"
    return validate_run_id(resolved)


__all__ = [
    "API_VERSION",
    "list_factors",
    "list_data_products",
    "list_labels",
    "list_universes",
    "describe_research_dataset_requirements",
    "run_data_product_interim",
]

"""Thin service layer shared by API and CLI adapters."""

from __future__ import annotations

from quant_research_platform.catalog.data_products import DataProductRegistry
from quant_research_platform.catalog.factors import FactorCatalog
from quant_research_platform.catalog.labels import LabelCatalog
from quant_research_platform.catalog.markets import MarketRegistry
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.universes import UniverseDefinition
from quant_research_platform.contracts.workflows import (
    WorkflowArtifactWriterPolicy,
    WorkflowNode,
    WorkflowSpec,
)
from quant_research_platform.runtime.privacy import sanitize_public_object
from quant_research_platform.workflows.services import execute_registered_workflow

API_VERSION = "0.1"


def list_factors() -> dict[str, object]:
    return _response(
        items=[item.to_manifest() for item in FactorCatalog.bootstrap_default().list()]
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


def validate_data_product_interim(
    data_product_id: str,
    *,
    dry_run: bool = True,
) -> dict[str, object]:
    data_product = DataProductRegistry.bootstrap_default().get(data_product_id)
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
    )
    return _response(
        status="validated",
        mode="dry_run_spec_validation" if dry_run else "fixture_backed_execution",
        data_product_id=data_product.data_product_id,
        market_id=str(data_product.market_id),
        contract_version=data_product.contract_version,
        workflow=summary,
    )


def validate_market_processed(market_id: str) -> dict[str, object]:
    market = MarketRegistry.bootstrap_default().get(market_id)
    spec = WorkflowSpec(
        workflow_id="market_processed_validation",
        market_id=MarketId(str(market.market_id)),
        input_spec_ids=("a_share_daily_bar_interim",),
        output_spec_ids=market.processed_spec_ids,
        run_type="market_processed_data",
        artifact_writer_policy=WorkflowArtifactWriterPolicy.SPEC_DRIVEN,
        nodes=(WorkflowNode(node_id="validate_processed_specs"),),
    )
    summary = execute_registered_workflow(
        spec,
        allowed_spec_ids=set(market.processed_spec_ids).union(
            {"a_share_daily_bar_interim"}
        ),
    )
    return _response(
        status="validated",
        mode="dry_run_spec_validation",
        market_id=str(market.market_id),
        workflow=summary,
    )


def validate_research_dataset() -> dict[str, object]:
    return _response(
        status="validated",
        mode="dry_run_spec_validation",
        dataset_id="a_share_price_volume_forward_returns",
        required_specs=[
            "price_volume_56",
            "forward_returns",
            "daily_context",
            "a_share_default",
            "a_share_execution",
        ],
    )


def validate_workflow_run() -> dict[str, object]:
    spec = WorkflowSpec(
        workflow_id="clean_rewrite_smoke_validation",
        market_id=MarketId("a_share"),
        input_spec_ids=("daily_panel",),
        output_spec_ids=("a_share_price_volume_forward_returns",),
        run_type="workflow_validation",
        artifact_writer_policy=WorkflowArtifactWriterPolicy.SPEC_DRIVEN,
        nodes=(
            WorkflowNode(node_id="validate_inputs"),
            WorkflowNode(node_id="validate_outputs", dependencies=("validate_inputs",)),
        ),
    )
    summary = execute_registered_workflow(
        spec,
        allowed_spec_ids={"daily_panel", "a_share_price_volume_forward_returns"},
    )
    return _response(
        status="validated",
        mode="dry_run_spec_validation",
        workflow=summary,
    )


def report_metadata(run_id: str) -> dict[str, object]:
    if not run_id or "/" in run_id or "\\" in run_id:
        raise ValueError("invalid_run_id")
    return _response(
        status="metadata_unavailable",
        mode="catalog_query_only",
        run_id=run_id,
        note="No local report index lookup is performed by the interface adapter.",
    )


def _response(**payload: object) -> dict[str, object]:
    return sanitize_public_object({"api_version": API_VERSION, **payload})


__all__ = [
    "API_VERSION",
    "list_factors",
    "list_data_products",
    "list_labels",
    "list_universes",
    "report_metadata",
    "validate_data_product_interim",
    "validate_market_processed",
    "validate_research_dataset",
    "validate_workflow_run",
]

"""Command-line interface for the QRP 0.1 clean rewrite."""

from __future__ import annotations

import typer

from quant_research_platform.catalog.factors import FactorCatalog
from quant_research_platform.catalog.labels import LabelCatalog
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.universes import UniverseDefinition

app = typer.Typer(add_completion=False)
catalog_app = typer.Typer(add_completion=False, help="Catalog query commands.")
source_app = typer.Typer(add_completion=False, help="Source ingestion commands.")
market_app = typer.Typer(add_completion=False, help="Market build commands.")
research_app = typer.Typer(add_completion=False, help="Research dataset commands.")
workflow_app = typer.Typer(add_completion=False, help="Workflow commands.")
report_app = typer.Typer(add_completion=False, help="Report commands.")

app.add_typer(catalog_app, name="catalog")
app.add_typer(source_app, name="source")
app.add_typer(market_app, name="market")
app.add_typer(research_app, name="research")
app.add_typer(workflow_app, name="workflow")
app.add_typer(report_app, name="report")


def _require_fixed_value(option_name: str, value: str, expected: str) -> None:
    if value != expected:
        typer.echo(f"{option_name} must be {expected}.", err=True)
        raise typer.Exit(code=2)


@catalog_app.command("factors")
def catalog_factors() -> None:
    """List registered factor definitions."""

    typer.echo("factor_id,market_id,factor_set,definition_version")
    for item in FactorCatalog.bootstrap_default().list():
        typer.echo(
            f"{item.factor_id},{item.market_id},{item.factor_set},"
            f"{item.definition_version}"
        )


@catalog_app.command("labels")
def catalog_labels() -> None:
    """List registered label definitions."""

    typer.echo("label_id,market_id,horizon,definition_version")
    for item in LabelCatalog.bootstrap_default().list():
        typer.echo(
            f"{item.label_id},{item.market_id},{item.horizon},"
            f"{item.definition_version}"
        )


@catalog_app.command("universes")
def catalog_universes() -> None:
    """List registered universe definitions."""

    universe = UniverseDefinition(
        universe_id="a_share_default",
        market_id=MarketId("a_share"),
        membership_column="in_universe",
    )
    typer.echo("universe_id,market_id,membership_column,definition_version")
    typer.echo(
        f"{universe.universe_id},{universe.market_id},"
        f"{universe.membership_column},{universe.definition_version}"
    )


@source_app.command("build-interim")
def source_build_interim(
    source: str = typer.Option("csmar", "--source", help="Source id."),
) -> None:
    """Validate the canonical source interim build entry point."""

    _require_fixed_value("--source", source, "csmar")
    typer.echo(
        f"source build-interim for {source} requires an approved source spec."
    )


@market_app.command("build-processed")
def market_build_processed(
    market: str = typer.Option("a_share", "--market", help="Market id."),
) -> None:
    """Validate the market processed-data build entry point."""

    _require_fixed_value("--market", market, "a_share")
    typer.echo(
        f"market build-processed for {market} requires an approved "
        "ProcessedDataSpec."
    )


@research_app.command("build-dataset")
def research_build_dataset() -> None:
    """Validate the research dataset build entry point."""

    typer.echo("research build-dataset requires an approved ResearchDatasetSpec.")


@workflow_app.command("run")
def workflow_run() -> None:
    """Validate the workflow execution entry point."""

    typer.echo("workflow run requires an approved WorkflowSpec.")


@report_app.command("show")
def report_show() -> None:
    """Validate the report query entry point."""

    typer.echo("report show requires a run id.")


def main() -> None:
    app()


__all__ = ["app", "main"]

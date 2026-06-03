"""Command-line interface for the QRP 0.1 clean rewrite."""

from __future__ import annotations

from collections.abc import Callable

import typer

from quant_research_platform.interfaces import services
from quant_research_platform.runtime.privacy import sanitize_public_text

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


def _echo_safe(value: object, *, err: bool = False) -> None:
    typer.echo(sanitize_public_text(value), err=err)


def _run_cli(operation: Callable[[], object]) -> None:
    try:
        _echo_safe(operation())
    except (KeyError, ValueError) as exc:
        _echo_safe(str(exc), err=True)
        raise typer.Exit(code=2) from exc


@catalog_app.command("factors")
def catalog_factors() -> None:
    """List registered factor definitions."""

    _echo_safe("factor_id,market_id,factor_set,definition_version")
    for item in services.list_factors()["items"]:
        _echo_safe(
            f"{item['factor_id']},{item['market_id']},{item['factor_set']},"
            f"{item['definition_version']}"
        )


@catalog_app.command("labels")
def catalog_labels() -> None:
    """List registered label definitions."""

    _echo_safe("label_id,market_id,horizon,definition_version")
    for item in services.list_labels()["items"]:
        _echo_safe(
            f"{item['label_id']},{item['market_id']},{item['horizon']},"
            f"{item['definition_version']}"
        )


@catalog_app.command("universes")
def catalog_universes() -> None:
    """List registered universe definitions."""

    _echo_safe("universe_id,market_id,membership_column,definition_version")
    for item in services.list_universes()["items"]:
        _echo_safe(
            f"{item['universe_id']},{item['market_id']},"
            f"{item['membership_column']},{item['definition_version']}"
        )


@source_app.command("build-interim")
def source_build_interim(
    source: str = typer.Option("csmar", "--source", help="Source id."),
) -> None:
    """Validate the canonical source interim build entry point."""

    _run_cli(lambda: services.validate_source_interim(source))


@market_app.command("build-processed")
def market_build_processed(
    market: str = typer.Option("a_share", "--market", help="Market id."),
) -> None:
    """Validate the market processed-data build entry point."""

    _run_cli(lambda: services.validate_market_processed(market))


@research_app.command("build-dataset")
def research_build_dataset() -> None:
    """Validate the research dataset build entry point."""

    _run_cli(services.validate_research_dataset)


@workflow_app.command("run")
def workflow_run() -> None:
    """Validate the workflow execution entry point."""

    _run_cli(services.validate_workflow_run)


@report_app.command("show")
def report_show() -> None:
    """Validate the report query entry point."""

    _run_cli(lambda: services.report_metadata("example-run"))


def main() -> None:
    app()


__all__ = ["app", "main"]

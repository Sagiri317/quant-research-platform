"""Command-line interface for the QRP 0.1 clean rewrite."""

from __future__ import annotations

from collections.abc import Callable

import typer

from quant_research_platform.contracts.execution import ExecutionMode
from quant_research_platform.interfaces import services
from quant_research_platform.runtime.privacy import sanitize_public_text
from quant_research_platform.storage.errors import WriteRejectedError

app = typer.Typer(add_completion=False)
catalog_app = typer.Typer(add_completion=False, help="Catalog query commands.")
data_products_app = typer.Typer(
    add_completion=False,
    help="Data product run commands.",
)
research_datasets_app = typer.Typer(
    add_completion=False,
    help="Research dataset query commands.",
)

app.add_typer(catalog_app, name="catalog")
app.add_typer(data_products_app, name="data-products")
app.add_typer(research_datasets_app, name="research-datasets")


def _echo_safe(value: object, *, err: bool = False) -> None:
    typer.echo(sanitize_public_text(value), err=err)


def _run_cli(operation: Callable[[], object]) -> None:
    try:
        _echo_safe(operation())
    except WriteRejectedError as exc:
        _echo_safe(exc.to_public_summary(), err=True)
        raise typer.Exit(code=2) from exc
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


@catalog_app.command("data-products")
def catalog_data_products() -> None:
    """List public data products."""

    _echo_safe("data_product_id,market_id,grain,contract_version")
    for item in services.list_data_products()["items"]:
        _echo_safe(
            f"{item['data_product_id']},{item['market_id']},{item['grain']},"
            f"{item['contract_version']}"
        )


@data_products_app.command("run-interim")
def run_data_product_interim(
    data_product_id: str = typer.Argument(..., help="Data product id."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate without writes."),
    run_id: str | None = typer.Option(None, "--run-id", help="Logical run id."),
) -> None:
    """Run the public data product interim entry point."""

    _run_data_product_interim(data_product_id, dry_run=dry_run, run_id=run_id)


def _run_data_product_interim(
    data_product_id: str,
    *,
    dry_run: bool,
    run_id: str | None,
) -> None:
    execution_mode = ExecutionMode.DRY_RUN if dry_run else ExecutionMode.EXECUTE

    _run_cli(
        lambda: services.run_data_product_interim(
            data_product_id,
            execution_mode=execution_mode,
            run_id=run_id,
        )
    )


@research_datasets_app.command("requirements")
def research_dataset_requirements(
    dataset_id: str = typer.Argument(..., help="Research dataset id."),
) -> None:
    """Describe public research dataset requirements."""

    _run_cli(lambda: services.describe_research_dataset_requirements(dataset_id))


def main() -> None:
    app()


__all__ = ["app", "main"]

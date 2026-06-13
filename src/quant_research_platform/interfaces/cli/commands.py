"""Command-line interface for the QRP 0.1 clean rewrite."""

from __future__ import annotations

from collections.abc import Callable

import typer

from quant_research_platform.catalog.profiles import research_profile
from quant_research_platform.interfaces import services
from quant_research_platform.runtime.privacy import sanitize_public_text

app = typer.Typer(add_completion=False)
catalog_app = typer.Typer(add_completion=False, help="Catalog query commands.")
research_datasets_app = typer.Typer(
    add_completion=False,
    help="Research dataset query commands.",
)

app.add_typer(catalog_app, name="catalog")
app.add_typer(research_datasets_app, name="research-datasets")


def _echo_safe(value: object, *, err: bool = False) -> None:
    typer.echo(sanitize_public_text(value), err=err)


def _run_cli(operation: Callable[[], object]) -> None:
    try:
        _echo_safe(operation())
    except (KeyError, ValueError) as exc:
        _echo_safe(str(exc), err=True)
        raise typer.Exit(code=2) from exc
    except Exception as exc:
        _echo_safe("internal_error", err=True)
        raise typer.Exit(code=2) from exc


@catalog_app.command("factors")
def catalog_factors() -> None:
    """List registered factor definitions."""

    _echo_safe("factor_id,market_id,factor_set,definition_version")
    for item in services.list_factors(research_profile())["items"]:
        _echo_safe(
            f"{item['factor_id']},{item['market_id']},{item['factor_set']},"
            f"{item['definition_version']}"
        )


@catalog_app.command("labels")
def catalog_labels() -> None:
    """List registered label definitions."""

    _echo_safe("label_id,market_id,horizon,definition_version")
    for item in services.list_labels(research_profile())["items"]:
        _echo_safe(
            f"{item['label_id']},{item['market_id']},{item['horizon']},"
            f"{item['definition_version']}"
        )


@research_datasets_app.command("requirements")
def research_dataset_requirements(
    dataset_id: str = typer.Argument(..., help="Research dataset id."),
) -> None:
    """Describe public research dataset requirements."""

    _run_cli(
        lambda: services.describe_research_dataset_requirements(
            dataset_id,
            research_profile(),
        )
    )


def main() -> None:
    app()


__all__ = ["app", "main"]


if __name__ == "__main__":
    main()

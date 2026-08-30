"""Command-line interface for the QRP local research platform."""

from __future__ import annotations

import json

import typer

from quant_research_platform.interfaces import queries
from quant_research_platform.profiles.research import research_profile

app = typer.Typer(add_completion=False)
catalog_app = typer.Typer(add_completion=False, help="Catalog query commands.")
research_datasets_app = typer.Typer(
    add_completion=False,
    help="Research dataset query commands.",
)

app.add_typer(catalog_app, name="catalog")
app.add_typer(research_datasets_app, name="research-datasets")


@catalog_app.command("factors")
def catalog_factors() -> None:
    """List registered factor definitions."""

    typer.echo("factor_id,market_id,factor_set,definition_version")
    for item in queries.list_factors(research_profile()).items:
        typer.echo(
            f"{item.factor_id},{item.market_id},{item.factor_set},"
            f"{item.definition_version}"
        )


@catalog_app.command("labels")
def catalog_labels() -> None:
    """List registered label definitions."""

    typer.echo("label_id,market_id,horizon,definition_version")
    for item in queries.list_labels(research_profile()).items:
        typer.echo(
            f"{item.label_id},{item.market_id},{item.horizon},{item.definition_version}"
        )


@research_datasets_app.command("requirements")
def research_dataset_requirements(
    dataset_id: str = typer.Argument(..., help="Research dataset id."),
) -> None:
    """Describe public research dataset requirements."""

    try:
        result = queries.describe_research_dataset_requirements(
            dataset_id,
            research_profile(),
        )
    except KeyError:
        typer.echo("unknown_dataset_id", err=True)
        raise typer.Exit(code=2) from None
    except ValueError:
        typer.echo("invalid_request", err=True)
        raise typer.Exit(code=2) from None
    except Exception as exc:
        typer.echo("internal_error", err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(json.dumps(result.model_dump(), ensure_ascii=False, sort_keys=True))


def main() -> None:
    app()


if __name__ == "__main__":
    main()

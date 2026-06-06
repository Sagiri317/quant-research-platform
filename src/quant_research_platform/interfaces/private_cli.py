"""Private local-data command-line interface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from quant_research_platform.reports.schema_inventory import build_schema_inventory
from quant_research_platform.storage.atomic import atomic_write_text

app = typer.Typer(add_completion=False)
PROJECT_ROOT_DEFAULT = Path(".")


@app.callback()
def _root() -> None:
    """Private local-data commands."""


@app.command("schema-inventory")
def schema_inventory(
    project_root: Annotated[
        Path,
        typer.Option(
            "--project-root",
            help="Project root containing data/interim and data/processed.",
        ),
    ] = PROJECT_ROOT_DEFAULT,
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            help="JSON output path. The file must not already exist.",
        ),
    ] = ...,
) -> None:
    """Write sanitized aggregate schema inventory JSON."""

    if output.exists():
        typer.echo("output_already_exists", err=True)
        raise typer.Exit(code=2)
    try:
        inventory = build_schema_inventory(project_root)
    except ValueError as exc:
        if str(exc) == "schema_inventory_no_datasets_found":
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=2) from exc
        raise
    atomic_write_text(
        output,
        json.dumps(inventory, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    app()


__all__ = ["app", "main"]


if __name__ == "__main__":
    main()

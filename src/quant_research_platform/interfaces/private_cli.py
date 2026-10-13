"""Private local-data diagnostics CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from quant_research_platform.policy.share_safety import sanitize_shared_text
from quant_research_platform.storage.identity import (
    filesystem_exists,
)
from quant_research_platform.storage.schema_inventory import (
    write_schema_inventory,
)

app = typer.Typer(add_completion=False, help="Private local-data diagnostics.")


@app.command("schema-inventory")
def schema_inventory(
    project_root: Annotated[
        Path,
        typer.Option(
            "--project-root",
            help="Project root containing local dataset roots.",
        ),
    ] = Path("."),
    output: Annotated[
        Path,
        typer.Option("--output", help="New JSON output path."),
    ] = ...,
) -> None:
    if filesystem_exists(output):
        typer.echo("output_already_exists", err=True)
        raise typer.Exit(code=2)
    try:
        write_schema_inventory(project_root, output)
    except Exception as exc:
        typer.echo(sanitize_shared_text(str(exc)), err=True)
        raise typer.Exit(code=2) from exc


def main() -> None:
    app()


if __name__ == "__main__":
    main()

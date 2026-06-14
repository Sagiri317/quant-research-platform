"""Private local-data command-line interface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from quant_research_platform.catalog.profiles import research_profile
from quant_research_platform.reports.schema_inventory import build_schema_inventory
from quant_research_platform.runtime.privacy import sanitize_public_text
from quant_research_platform.storage.atomic import atomic_write_text
from quant_research_platform.workflows.verification_evidence import (
    VerificationEvidenceRunConfig,
    run_verification_evidence,
)

app = typer.Typer(add_completion=False)
evidence_app = typer.Typer(
    add_completion=False,
    help="Verification evidence commands.",
)
PROJECT_ROOT_DEFAULT = Path(".")

app.add_typer(evidence_app, name="evidence")


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


@evidence_app.command("run")
def evidence_run(
    output_root: Annotated[
        Path,
        typer.Option("--output-root", help="Explicit verification output root."),
    ] = ...,
    start_date: Annotated[
        str,
        typer.Option("--start-date", help="Inclusive evidence start date."),
    ] = ...,
    end_date: Annotated[
        str,
        typer.Option("--end-date", help="Inclusive evidence end date."),
    ] = ...,
    evaluation_start_date: Annotated[
        str | None,
        typer.Option(
            "--evaluation-start-date",
            help="Inclusive evaluation window start date.",
        ),
    ] = None,
    evaluation_end_date: Annotated[
        str | None,
        typer.Option(
            "--evaluation-end-date",
            help="Inclusive evaluation window end date.",
        ),
    ] = None,
    prediction_train_end: Annotated[
        str | None,
        typer.Option(
            "--prediction-train-end",
            help="Prediction training window end date.",
        ),
    ] = None,
    prediction_test_start: Annotated[
        str | None,
        typer.Option(
            "--prediction-test-start",
            help="Prediction test window start date.",
        ),
    ] = None,
    prediction_test_end: Annotated[
        str | None,
        typer.Option(
            "--prediction-test-end",
            help="Prediction test window end date.",
        ),
    ] = None,
    replace_output: Annotated[
        bool,
        typer.Option(
            "--replace-output",
            help="Replace an owned verification evidence output root.",
        ),
    ] = False,
) -> None:
    """Run the verification evidence verification harness."""

    try:
        result = run_verification_evidence(
            VerificationEvidenceRunConfig(
                output_root=output_root,
                start_date=start_date,
                end_date=end_date,
                evaluation_start_date=evaluation_start_date,
                evaluation_end_date=evaluation_end_date,
                prediction_train_end=prediction_train_end,
                prediction_test_start=prediction_test_start,
                prediction_test_end=prediction_test_end,
                profile=research_profile(),
                replace_output=replace_output,
            )
        )
    except Exception as exc:
        typer.echo(sanitize_public_text(str(exc)), err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(
        sanitize_public_text(
            f"status={result.status} output_root={result.output_root} "
            f"run_id={result.run_id}"
        )
    )


def main() -> None:
    app()


__all__ = ["app", "main"]


if __name__ == "__main__":
    main()

"""Command-line interface package."""

from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:
    if name in {"app", "main"}:
        from quant_research_platform.interfaces.cli.commands import app, main

        return {"app": app, "main": main}[name]
    raise AttributeError(name)


__all__ = ["app", "main"]

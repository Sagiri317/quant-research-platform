from __future__ import annotations

from fastapi import APIRouter

from quant_research_platform.interfaces.api.routes import (
    artifacts,
    comparison,
    downloads,
    factor_factory,
    factor_templates,
    health,
    markets,
    pipeline,
    runs,
    uploads,
)


def test_api_route_modules_expose_domain_routers() -> None:
    for module in (
        artifacts,
        comparison,
        downloads,
        factor_factory,
        factor_templates,
        health,
        markets,
        pipeline,
        runs,
        uploads,
    ):
        assert isinstance(module.router, APIRouter)

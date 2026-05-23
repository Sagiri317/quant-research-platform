"""市场能力元数据路由。"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from quant_research_platform.interfaces.api.markets import (
    get_market_capabilities,
    list_market_capabilities,
)

router = APIRouter()


@router.get("/markets")
def markets() -> dict[str, object]:
    """返回本地 market capability 占位信息。"""

    return {"markets": list_market_capabilities()}


@router.get("/api/markets")
def api_markets() -> list[dict[str, object]]:
    """返回供 web app 使用的本地 market capability 占位信息。"""

    return list_market_capabilities()


@router.get("/markets/{market}/capabilities")
@router.get("/api/markets/{market}/capabilities")
def market_capabilities(market: str) -> dict[str, object]:
    """返回单个 market 的静态 capability metadata。"""

    capabilities = get_market_capabilities(market)
    if capabilities is None:
        raise HTTPException(status_code=404, detail="market not found")
    return capabilities


__all__ = ["router"]

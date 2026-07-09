from __future__ import annotations

import pytest

from quant_research_platform.markets.implementations.crypto.ids import (
    canonical_crypto_asset_id,
)


class _StringLikeCryptoSymbol:
    def __str__(self) -> str:
        return "BTCUSDT"


@pytest.mark.parametrize(
    "source_asset_id",
    [None, 1, 1.0, True, [], {}, set(), "", _StringLikeCryptoSymbol()],
)
def test_canonical_crypto_asset_id_requires_strict_string_input(
    source_asset_id: object,
) -> None:
    with pytest.raises(ValueError, match="unsupported_crypto_source_asset_id"):
        canonical_crypto_asset_id(source_asset_id)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("source_asset_id", "expected"),
    [
        ("BTCUSDT", "btc_usdt"),
        ("ethusdc", "eth_usdc"),
    ],
)
def test_canonical_crypto_asset_id_preserves_supported_symbol_semantics(
    source_asset_id: str,
    expected: str,
) -> None:
    assert canonical_crypto_asset_id(source_asset_id) == expected

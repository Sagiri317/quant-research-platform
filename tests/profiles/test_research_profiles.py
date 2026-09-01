from __future__ import annotations

from types import MappingProxyType

import pytest

from quant_research_platform.contracts.targets import ProducerBinding
from quant_research_platform.profiles.research import (
    A_SHARE_PROFILE_ID,
    CRYPTO_PROFILE_ID,
    research_profile,
)


def test_profiles_use_stable_ids_and_immutable_producer_bindings() -> None:
    a_share = research_profile(A_SHARE_PROFILE_ID)
    crypto = research_profile(CRYPTO_PROFILE_ID)

    assert a_share.profile_id == A_SHARE_PROFILE_ID
    assert crypto.profile_id == CRYPTO_PROFILE_ID
    assert a_share.protocol("pv_all").manifest()["protocol_id"] == (
        "a_share_low_frequency"
    )
    assert crypto.protocol().manifest()["protocol_id"] == "crypto_hft"
    assert isinstance(a_share.producer_bindings, MappingProxyType)
    assert isinstance(crypto.producer_bindings, MappingProxyType)
    assert all(
        isinstance(value, ProducerBinding)
        for value in a_share.producer_bindings.values()
    )
    assert all(
        isinstance(value, ProducerBinding)
        for value in crypto.producer_bindings.values()
    )
    assert a_share.producer_bindings["market_bars_1d"].source_identity is not None
    assert a_share.producer_bindings["market_bars_1d"].source_root_resolver is not None
    assert crypto.producer_bindings["market_bars_1m"].source_identity is None


def test_profile_selector_rejects_derived_identity() -> None:
    with pytest.raises(KeyError, match="unknown_research_profile"):
        research_profile("a_share_csmar_choice_1d")

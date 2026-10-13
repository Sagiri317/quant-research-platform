"""Profiles are immutable read-only market catalogs, with no operation defaults."""

from dataclasses import FrozenInstanceError

import pytest

from quant_research_platform.profiles.research import (
    A_SHARE_PROFILE_ID,
    CRYPTO_PROFILE_ID,
    research_profile,
)


@pytest.mark.parametrize(
    "profile_id,market",
    [(A_SHARE_PROFILE_ID, "a_share"), (CRYPTO_PROFILE_ID, "crypto")],
)
def test_profiles_expose_catalogs_only(profile_id, market):
    profile = research_profile(profile_id)
    assert profile.market_id == market
    assert profile.factors.list()
    assert profile.labels.list()
    assert profile.research_datasets.list()
    assert not hasattr(profile, "producer_bindings")
    assert not hasattr(profile, "protocol")
    with pytest.raises(FrozenInstanceError):
        profile.profile_id = "changed"


def test_profile_selector_does_not_guess_a_market():
    with pytest.raises(KeyError, match="unknown_research_profile"):
        research_profile("a_share_csmar_choice_1d")

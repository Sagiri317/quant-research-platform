from __future__ import annotations

from dataclasses import asdict
from types import MappingProxyType, SimpleNamespace

import pandas as pd
import pytest

from quant_research_platform.contracts.targets import ProducerBinding
from quant_research_platform.profiles.research import (
    A_SHARE_PROFILE_ID,
    CRYPTO_PROFILE_ID,
    research_profile,
)
from tests.helpers.research_configuration import CORE6_FACTORS, PV25_FACTORS


def test_profiles_use_stable_ids_and_immutable_producer_bindings() -> None:
    a_share = research_profile(A_SHARE_PROFILE_ID)
    crypto = research_profile(CRYPTO_PROFILE_ID)

    assert a_share.profile_id == A_SHARE_PROFILE_ID
    assert crypto.profile_id == CRYPTO_PROFILE_ID
    assert a_share.protocol("pv_all").manifest()["protocol_id"] == (
        "a_share_low_frequency"
    )
    assert crypto.protocol().manifest()["protocol_id"] == "crypto_hft"
    with pytest.raises(ValueError, match="unknown_crypto_scenario"):
        crypto.protocol("microstructure_forward_returns")
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
    assert tuple(crypto.interim.list()) == ()
    assert not {
        "market_bars_1m",
        "aggregate_trades_tick",
        "depth_levels_tick",
    }.intersection(crypto.producer_bindings)
    assert (
        crypto.producer_bindings["binance_futures_l2_panel_100ms"].source_identity
        is not None
    )


def test_profile_selector_rejects_derived_identity() -> None:
    with pytest.raises(KeyError, match="unknown_research_profile"):
        research_profile("a_share_csmar_choice_1d")


def test_expanded_pv_uses_existing_owners_and_keeps_formal_selections():
    from quant_research_platform.research.factors.price_volume.definitions import (
        PRICE_VOLUME_LIBRARY,
    )

    profile = research_profile(A_SHARE_PROFILE_ID)
    expanded = profile.protocol("pv_expanded")
    assert len(PRICE_VOLUME_LIBRARY) == 50
    assert expanded.factor_columns == (
        *PRICE_VOLUME_LIBRARY,
        "beta_252d",
        "residual_volatility_60d",
    )
    from quant_research_platform.profiles.research import bind_research_configuration
    from tests.helpers.research_configuration import research_configuration

    core = bind_research_configuration(profile, research_configuration(name="core6"))
    full = bind_research_configuration(profile, research_configuration(name="pv25"))
    assert core.protocol("pv_all").factor_columns == CORE6_FACTORS
    assert full.protocol("pv_all").factor_columns == PV25_FACTORS
    assert profile.protocol("pv_all").portfolio_candidates() == ()
    assert len(core.protocol("pv_all").portfolio_candidates()) == 10
    expanded_profile = bind_research_configuration(
        profile, research_configuration(name="pv_expanded")
    )
    expanded = expanded_profile.protocol("pv_expanded")
    assert len(expanded.portfolio_candidates()) == 6
    targets = {
        target.target_id: target for target in expanded.target_specs("research run")
    }
    for candidate in expanded.portfolio_candidates():
        assert candidate.risk_style_columns
        assert candidate.risk_context_target_ids == (
            "price_volume_facts",
            "market_model_factors",
        )
        ledger = targets[f"pv_expanded_{candidate.candidate_id}_ledger"]
        assert set(candidate.risk_context_target_ids) <= set(ledger.dependencies)
        assert set(candidate.risk_context_target_ids) <= set(
            targets["pv_expanded_portfolio_statistics"].dependencies
        )
    assert expanded.manifest()["portfolio_candidates"] == [
        candidate.candidate_id for candidate in expanded.portfolio_candidates()
    ]
    spec = profile.research_datasets.get("pv_expanded")
    assert spec.factor_columns == expanded.factor_columns
    assert "fundamental_factors" not in spec.input_data_ids
    assert "fundamental" not in expanded.factor_fdr_families
    catalog_columns = {
        definition.output_column for definition in profile.factors.list()
    }
    assert set(PRICE_VOLUME_LIBRARY) <= catalog_columns
    processed = profile.processed.get("price_volume_facts")
    assert (
        tuple(
            column
            for column in processed.required_columns
            if column in PRICE_VOLUME_LIBRARY
        )
        == PRICE_VOLUME_LIBRARY
    )
    for target in expanded.target_specs("research run"):
        assert "fundamental_factors" not in target.dependencies
        assert target.target_id in expanded_profile.producer_bindings


def test_holdout_freezes_alpha_for_each_prediction_source():
    from quant_research_platform.profiles.research import bind_research_configuration
    from quant_research_platform.research.portfolios.accounting import (
        calibrate_expected_alpha,
    )
    from tests.helpers.research_configuration import research_configuration

    configuration = research_configuration()
    configuration["models"] = [
        model
        for model in configuration["models"]
        if model["model_algorithm"]
        in {"equal_weight_rank_ensemble", "linear_regression"}
    ]
    selectors = [model["model_id"] for model in configuration["models"]]
    configuration["portfolios"] = [
        {
            "candidate_id": f"enhanced_{index}",
            "prediction_selector": selector,
            "top_quantile": None,
            "rebalance_schedule": "weekly-last-trading-day",
            "construction": "benchmark_enhanced",
            "benchmark_id": "csi500",
        }
        for index, selector in enumerate(selectors)
    ]
    configuration["portfolios"].append(
        {
            **configuration["portfolios"][0],
            "candidate_id": "same_prediction_other_benchmark",
            "benchmark_id": "csi1000",
        }
    )
    configuration["primary_portfolio_id"] = "enhanced_0"
    protocol = bind_research_configuration(research_profile(), configuration).protocol(
        configuration["scenario_id"]
    )
    cutoff = configuration["windows"]["development"]["end"]
    dates = pd.date_range("2023-11-10", periods=6, freq="W-FRI")
    intervals = pd.DataFrame(
        {
            "timestamp": dates,
            "entry_at": dates + pd.Timedelta(days=3),
            "available_at": dates + pd.Timedelta(days=10),
        }
    )
    calibrations = {}
    for index, selector in enumerate(selectors):
        frame = pd.DataFrame(
            [
                {
                    "timestamp": timestamp,
                    "alpha_score": score,
                    "holding_return": 0.01
                    + (0.04 + 0.03 * index + day * 0.001) * score,
                }
                for day, timestamp in enumerate(dates)
                for score in (-0.5, 0.0, 0.5)
            ]
        )
        calibrations[selector] = calibrate_expected_alpha(
            frame, as_of=pd.Timestamp(cutoff), scheduled_intervals=intervals
        )
    assert len({value.slope for value in calibrations.values()}) == 2
    assert all(value.status == "DEFINED" for value in calibrations.values())
    artifacts = {
        protocol.model_selection_target_id: SimpleNamespace(
            final_models=[
                {"selection_group": group, "family_id": selector}
                for group in ("global", "quarterly")
                for selector in selectors
            ]
        )
    }
    references = {}
    for candidate in protocol.portfolio_candidates():
        target_id = f"{protocol.scenario_id}_{candidate.candidate_id}_ledger"
        references[target_id] = SimpleNamespace(
            target_id=target_id,
            payload=lambda target_id=target_id: {"target_id": target_id},
        )
        payload = {
            "schema_id": "qrp-final-alpha-calibration",
            "selection_group": candidate.selection_group,
            "cutoff": cutoff,
            "upstream_identities": {},
            "calibration": asdict(calibrations[candidate.prediction_selector]),
        }
        artifacts[target_id] = lambda payload=payload: SimpleNamespace(
            final_alpha_calibration=payload
        )
    record = SimpleNamespace(research_window={"end": cutoff})
    frozen, blocker = protocol.freeze_development(record, references, {}, artifacts)
    assert blocker is None
    assert frozen.access_window == "holdout"
    assert dict(frozen.frozen_alpha) == {
        (group, selector, "weekly-last-trading-day"): calibration
        for group in ("global", "quarterly")
        for selector, calibration in calibrations.items()
    }
    payload["calibration"]["slope"] += 1.0
    with pytest.raises(ValueError, match="holdout_alpha_sources_disagree"):
        protocol.freeze_development(record, references, {}, artifacts)

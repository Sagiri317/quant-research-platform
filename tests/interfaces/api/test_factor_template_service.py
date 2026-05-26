from quant_research_platform.interfaces.api.services.factor_template_service import (
    factor_template_metadata,
    list_factor_template_payloads,
)


def test_factor_template_service_lists_configured_a_share_templates() -> None:
    payloads = list_factor_template_payloads("a_share")

    assert [payload["id"] for payload in payloads] == [
        "liquidity_20d",
        "momentum_60d",
        "reversal_20d",
    ]


def test_factor_template_metadata_resolves_controlled_template_source() -> None:
    template_id, source = factor_template_metadata("reversal_20d", market="a_share")

    assert template_id == "reversal_20d"
    assert source == "config/a_share/factor_templates/reversal_20d.yaml"

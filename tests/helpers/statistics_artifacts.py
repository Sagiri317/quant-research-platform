"""Small explicit scientific inputs shared by native statistics regression tests."""

from quant_research_platform.research.statistics_production import _selection_trial_rows
from tests.helpers.research_configuration import (
    A_SHARE_MODEL_ALGORITHMS,
    research_configuration,
)


def statistics_config(
    *, factors=None, sample_role="validation", selection_group="global"
):
    evaluation = research_configuration()["evaluation"]
    if factors is not None:
        evaluation["factors"] = list(factors)
    return {
        "sample_role": sample_role,
        "selection_group": selection_group,
        "root_seed": 42,
        "evaluation": evaluation,
        "statistics": {
            "factor_fdr_families": {"fixture": evaluation["factors"]},
            "fama_macbeth_core_factors": evaluation["factors"],
            "hac_lag": 19,
            "bootstrap_block": 20,
            "bootstrap_repetitions": 10_000,
            "fdr_primary": "BH_5pct",
            "fdr_robustness": "BY_5pct",
            "group_comparison": {
                "fdr_method": "BY",
                "alpha": 0.05,
                "seed": 42,
                "confidence": 0.95,
            },
        },
    }


def statistics_metadata(window, *, sample_role="validation", **values):
    return {
        "sample_role": sample_role,
        "window": {"start": str(window.start), "end": str(window.end)},
        **values,
    }


def model_statistics_inputs(
    spec, config, window, *, families=None, selected=None, selection=None
):
    """Declare selected families explicitly for tests of numeric consumers."""
    families = families or tuple(
        f"{spec.data_id}_{algorithm}" for algorithm in A_SHARE_MODEL_ALGORITHMS
    )
    if selection is None:
        records = []
        for family in families:
            metrics = {
                "model_id": family,
                "validation_rows": 12,
                "validation_ic": 0.1,
                "validation_rankic": 0.1,
                "validation_ic_reason": None,
                "validation_rankic_reason": None,
                "validation_mse": 1.0,
            }
            for prefix in ("ic", "rankic"):
                metrics.update(
                    {
                        f"validation_std_{prefix}": 0.0,
                        f"validation_{prefix}ir": None,
                        f"validation_positive_{prefix}_rate": 1.0,
                        f"validation_valid_{prefix}_count": 1,
                        f"validation_{prefix}ir_reason": "insufficient_observations",
                    }
                )
            records.append(
                {
                    "family_id": family,
                    "selected_config_id": family,
                    "trial_metrics": [metrics],
                }
            )
        selection = {
            "selected_family_id": selected or families[0],
            "selected_model_id": selected or families[0],
            "families": records,
        }
    else:
        selection = {
            **selection,
            "selected_model_id": next(
                f["selected_config_id"]
                for f in selection["families"]
                if f["family_id"] == selection["selected_family_id"]
            ),
        }
    config["comparison_model_id"] = families[0]
    config["evaluation"]["factors"] = list(spec.factor_columns)
    period = "explicit_period"
    entries = {
        (period, family["family_id"]): {
            "family_id": family["family_id"],
            "model_id": family["selected_config_id"],
            "model_metadata": {
                "input_columns": [spec.factor_columns[0]],
                "sample_filter_mode": "eligible",
                "model_algorithm": A_SHARE_MODEL_ALGORITHMS[
                    index % len(A_SHARE_MODEL_ALGORITHMS)
                ],
            },
        }
        for index, family in enumerate(selection["families"])
    }
    return {
        "model_entries": entries,
        "selections": {period: selection},
        "period_windows": {period: window},
        "trial_rows": _selection_trial_rows(
            {"final" if config["selection_group"] == "global" else period: selection},
            config["selection_group"],
        ),
    }


def portfolio_statistics_config(sample_role="validation", *, configuration=None):
    from dataclasses import asdict

    from quant_research_platform.contracts.portfolios import PortfolioCandidateSpec
    from quant_research_platform.research.portfolios.risk import FactorRiskConfig
    from quant_research_platform.research.protocols.a_share import (
        EXECUTION_ASSUMPTIONS,
        RISK_STYLE_COLUMNS,
    )

    configuration = configuration or research_configuration(name="pv25")
    candidates = tuple(
        PortfolioCandidateSpec(
            **{
                **item,
                "candidate_id": group + "_" + item["candidate_id"],
                "selection_group": group,
                "risk_style_columns": RISK_STYLE_COLUMNS
                if item["construction"] == "benchmark_enhanced"
                else RISK_STYLE_COLUMNS[:-1],
            }
        )
        for item in configuration["portfolios"]
        for group in ("global", "quarterly")
    )
    config = statistics_config(sample_role=sample_role)
    config.update(
        analyses=[
            "performance",
            "attribution",
            "capacity",
            "sharpe_inference",
            "cscv",
            "group_comparison",
        ],
        execution_assumptions=asdict(EXECUTION_ASSUMPTIONS),
        risk=asdict(FactorRiskConfig()),
        benchmark_total_return_indices={"csi500": "H00905", "csi1000": "H00852"},
    )
    config["statistics"]["cscv_blocks"] = 10
    config["statistics"]["group_comparison"].update(
        portfolio_ids=[c["candidate_id"] for c in configuration["portfolios"]],
        primary_portfolio_id=configuration["primary_portfolio_id"],
    )
    return candidates, config

"""Declare macro formulas independently of vendors and acquisition methods."""

from quant_research_platform.contracts.factors import FactorDefinition
from quant_research_platform.contracts.macro import MACRO_VARIABLES

# Components must refer to the same observation date; weights also convert units.
COMPOSITES = {
    "us_net_liquidity": (
        {"us_fed_assets": 1, "us_tga_wed": -1, "us_on_rrp": -1000},
        "USD_million",
        "W",
    ),
    "us_sofr_iorb": ({"us_sofr": 100, "us_iorb": -100}, "bp", "D"),
    "us_effr_iorb": ({"us_effr": 100, "us_iorb": -100}, "bp", "D"),
    "us_effr_ioer": ({"us_effr": 100, "us_ioer": -100}, "bp", "D"),
    "us_term_10y_2y": ({"us_treasury_10y": 100, "us_treasury_2y": -100}, "bp", "D"),
    "us_term_10y_3m": ({"us_treasury_10y": 100, "us_treasury_3m": -100}, "bp", "D"),
    "cn_r007_dr007": ({"cn_r007": 100, "cn_dr007": -100}, "bp", "D"),
    "cn_dr007_policy": ({"cn_dr007": 100, "cn_repo_7d_rate": -100}, "bp", "D"),
    "cn_term_10y_1y": ({"cn_treasury_10y": 100, "cn_treasury_1y": -100}, "bp", "D"),
    "cn_credit_spread": ({"cn_credit_aaa_3y": 100, "cn_treasury_3y": -100}, "bp", "D"),
    "cn_net_injection": (
        {"cn_reverse_repo": 1, "cn_repo_maturity": -1},
        "CNY_100million",
        "D",
    ),
}
VARIABLES = {
    **MACRO_VARIABLES,
    **{name: (unit, frequency) for name, (_, unit, frequency) in COMPOSITES.items()},
}
TRANSFORMS = (
    "level",
    "change",
    "acceleration",
    "yoy_change",
    "zscore",
    "percentile",
    "volatility",
    "ewma_deviation",
    "innovation",
    "innovation_zscore",
)
POSITIVE_VARIABLES = {
    name
    for name, (unit, _) in VARIABLES.items()
    if unit.startswith(("USD", "CNY", "EUR", "JPY", "HKD"))
    and name not in {"cn_net_injection", "cn_tsf_flow", "us_net_liquidity"}
} | {"us_vix", "us_broad_dollar", "us_broad_dollar_legacy"}
PRESSURE_COMPONENTS = {
    "us_funding_pressure": {
        "us_sofr_iorb_zscore": 0.25,
        "us_hy_oas_zscore": 0.25,
        "us_nfci_zscore": 0.25,
        "us_broad_dollar_zscore": 0.25,
    },
    "cn_funding_pressure": {"cn_r007_dr007_zscore": 0.5, "cn_dr007_policy_zscore": 0.5},
    "global_liquidity": {
        "us_fed_assets_zscore": 1 / 3,
        "eu_ecb_assets_zscore": 1 / 3,
        "jp_boj_assets_zscore": 1 / 3,
    },
}
STATE_FACTORS = (
    "us_net_liquidity_zscore",
    "us_funding_pressure",
    "cn_funding_pressure",
    "us_cn_pressure_difference",
    "global_liquidity",
)
EXPOSURES = (
    "amihud_60d",
    "volatility_60d",
    "log_total_market_cap",
    "beta_252d",
    "turnover_20d",
    "momentum_60d",
    "residual_volatility_60d",
)
BETA_VARIABLES = ("us_net_liquidity", "us_broad_dollar", "us_nfci", "cn_m2", "cn_dr007")


def recipes():
    output = {}
    for name, (_, frequency) in VARIABLES.items():
        transforms = tuple(
            item for item in TRANSFORMS if frequency != "event" or item != "yoy_change"
        )
        if name in {
            "cn_reverse_repo",
            "cn_repo_maturity",
            "cn_net_injection",
            "cn_tsf_flow",
        }:
            transforms += ("sum",)
        if name in POSITIVE_VARIABLES:
            transforms += ("log_level", "growth", "yoy_growth")
        for transform in transforms:
            output[f"ml_{name}_{transform}"] = ("series", name, transform)
    for name in (
        *PRESSURE_COMPONENTS,
        "us_cn_pressure_difference",
        "us_net_liquidity_latest",
        "us_reserve_asset_ratio",
        "us_tga_asset_ratio",
        "us_rrp_asset_ratio",
    ):
        output[f"ml_{name}"] = ("state", name)
    for state in STATE_FACTORS:
        for exposure in EXPOSURES:
            output[f"ml_{state}_x_{exposure}"] = ("interaction", state, exposure)
    for variable in BETA_VARIABLES:
        for kind in ("beta", "partial_beta"):
            output[f"ml_{variable}_{kind}"] = ("beta", variable, kind)
    return output


RECIPES = recipes()
PRESERVE_AMPLITUDE_FACTORS = frozenset(
    name for name, recipe in RECIPES.items() if recipe[0] != "beta"
)


def factor_definitions():
    return tuple(
        FactorDefinition(
            factor_id=name,
            market_id="a_share",
            factor_set="experimental",
            family="macro_liquidity",
            input_columns=("timestamp", "value"),
            output_column=name,
            required_input_data_id="macro_observations",
            timing_semantics="released_vintage_at_or_before_explicit_decision_time",
            null_policy="preserve_missing_components_and_insufficient_history",
            description="Declared macro history or exposure; preserve macro amplitude.",
        )
        for name in RECIPES
    )


def factor_unit(name):
    recipe = RECIPES[name]
    if recipe[0] == "series":
        _, variable, transform = recipe
        unit = VARIABLES[variable][0]
        if transform in {
            "zscore",
            "percentile",
            "innovation_zscore",
            "growth",
            "yoy_growth",
            "log_level",
        }:
            return "dimensionless"
        return "bp" if unit == "percent" and transform != "level" else unit
    if name == "ml_us_net_liquidity_latest":
        return "USD_million"
    return (
        "dimensionless"
        if recipe[0] in {"state", "interaction"}
        else "return_per_macro_unit"
    )

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
RATIOS = {
    f"{name}_vix_ratio": (name, "us_vix")
    for name in ("us_vix3m", "us_vix9d", "us_vxn", "us_vxeem")
}
VOLATILITY_INDICES = {
    "us_vix",
    "us_vix3m",
    "us_vix9d",
    "us_vxn",
    "us_vxeem",
    "us_vvix",
    "us_ovx",
    "us_gvz",
}
# Explicit public hypotheses, not another transform/interaction Cartesian library.
# (native table, exact instrument/scope, input fields, unit, frequency, transforms)
PUBLIC_SERIES = {
    "us_finra_margin_debit": (
        "public_finra_margin",
        "US_MARGIN_ACCOUNTS",
        ("margin_debit",),
        "USD_million",
        "M",
        ("level", "growth", "yoy_growth"),
    ),
}
for _index, _code in (("sp500", "13874A"), ("nasdaq", "209742")):
    for _group in ("leveraged", "asset_manager"):
        PUBLIC_SERIES[f"us_cftc_{_index}_{_group}_net_share"] = (
            "public_cftc_positions",
            f"{_code}:futures_only",
            (f"{_group}_long", f"{_group}_short", "open_interest"),
            "dimensionless",
            "W",
            ("level", "change", "percentile"),
        )
    PUBLIC_SERIES[f"us_cftc_{_index}_leveraged_net_contracts"] = (
        "public_cftc_positions",
        f"{_code}:futures_only",
        ("leveraged_long", "leveraged_short"),
        "contracts",
        "W",
        ("level", "change"),
    )
for _scope in ("equity", "index", "total"):
    PUBLIC_SERIES[f"us_cboe_{_scope}_put_call"] = (
        "public_option_put_call",
        f"{_scope.upper()} PUT/CALL RATIO",
        ("put_call_ratio",),
        "dimensionless",
        "D",
        ("level", "zscore", "percentile"),
    )

VARIABLES = {
    **MACRO_VARIABLES,
    **{name: (recipe[3], recipe[4]) for name, recipe in PUBLIC_SERIES.items()},
    **{name: (unit, frequency) for name, (_, unit, frequency) in COMPOSITES.items()},
    **dict.fromkeys(RATIOS, ("dimensionless", "D")),
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
POSITIVE_VARIABLES = (
    {
        name
        for name, (unit, _) in VARIABLES.items()
        if unit.startswith(("USD", "CNY", "EUR", "JPY", "HKD"))
        and name not in {"cn_net_injection", "cn_tsf_flow", "us_net_liquidity"}
    }
    | VOLATILITY_INDICES
    | {"us_skew", "us_broad_dollar", "us_broad_dollar_legacy"}
)
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
STATE_PRODUCTS = {
    "us_vix_skew_zscore_product": ("us_vix_zscore", "us_skew_zscore"),
    "us_vix_vvix_zscore_product": ("us_vix_zscore", "us_vvix_zscore"),
    "us_liquidity_vix_zscore_product": ("us_net_liquidity_zscore", "us_vix_zscore"),
}
STATE_FACTORS = (
    "us_net_liquidity_zscore",
    "us_funding_pressure",
    "cn_funding_pressure",
    "us_cn_pressure_difference",
    "global_liquidity",
    "us_vix_change",
    "us_vix_zscore",
    "us_vvix_change",
    "us_skew_zscore",
    "us_vix3m_vix_ratio_level",
    "us_vix9d_vix_ratio_level",
    "us_broad_dollar_change",
    "us_real_treasury_10y_change",
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
BETA_VARIABLES = (
    "us_net_liquidity",
    "us_broad_dollar",
    "us_nfci",
    "cn_m2",
    "cn_dr007",
    "us_vix",
    "us_vvix",
    "us_skew",
    "us_real_treasury_10y",
)
MATCHED_SHOCK_VARIABLES = (
    "us_vix",
    "us_vvix",
    "us_skew",
    "us_broad_dollar",
    "us_real_treasury_10y",
    "us_net_liquidity",
)


def recipes():
    output = {}
    for name, (_, frequency) in VARIABLES.items():
        if name in PUBLIC_SERIES:
            for transform in PUBLIC_SERIES[name][5]:
                output[f"ml_{name}_{transform}"] = ("series", name, transform)
            continue
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
        if name in VOLATILITY_INDICES:
            transforms += ("implied_variance",)
        if name == "us_skew":
            transforms += ("risk_neutral_skewness",)
        for transform in transforms:
            output[f"ml_{name}_{transform}"] = ("series", name, transform)
    for name in (
        *PRESSURE_COMPONENTS,
        *STATE_PRODUCTS,
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
        for exposure in ("volatility_60d", "amihud_60d", "beta_252d"):
            output[f"ml_{state}_x_industry_{exposure}"] = (
                "industry_interaction",
                state,
                exposure,
            )
    for variable in BETA_VARIABLES:
        for kind in ("beta", "partial_beta"):
            output[f"ml_{variable}_{kind}"] = ("beta", variable, kind)
    for variable in MATCHED_SHOCK_VARIABLES:
        output[f"ml_{variable}_shock_x_own_beta"] = ("matched_beta", variable)
    return output


RECIPES = recipes()
PRESERVE_AMPLITUDE_FACTORS = frozenset(
    name for name, recipe in RECIPES.items() if recipe[0] != "beta"
)


def _description(recipe):
    if recipe[0] == "matched_beta":
        return (
            f"Latest known {recipe[1]} observation change times the stock's own "
            "rolling sensitivity to changes in that same variable, estimated only "
            "from mature past A-share returns and contemporaneously known releases. "
            "Preserve shock amplitude; not an A-share market-beta substitution."
        )
    if recipe[0] == "series" and recipe[1] in PUBLIC_SERIES:
        table, identity, fields, _unit, _frequency, _transforms = PUBLIC_SERIES[
            recipe[1]
        ]
        formula = fields[0]
        if table == "public_cftc_positions":
            formula = f"({fields[0]} - {fields[1]})"
            if len(fields) == 3:
                formula += f" / {fields[2]}"
        return (
            f"Public market context: {recipe[2]} of {formula}, exact scope {identity}; "
            "versions enter the common release history only at available_at. "
            "Not a stock cross-sectional score or observed investor sentiment."
        )
    return "Declared macro history or exposure; preserve macro amplitude."


def factor_definitions():
    return tuple(
        FactorDefinition(
            factor_id=name,
            market_id="a_share",
            factor_set="experimental",
            family="macro_liquidity",
            input_columns=(*PUBLIC_SERIES[recipe[1]][2], "available_at")
            if recipe[0] == "series" and recipe[1] in PUBLIC_SERIES
            else ("timestamp", "value"),
            output_column=name,
            required_input_data_id=PUBLIC_SERIES[recipe[1]][0]
            if recipe[0] == "series" and recipe[1] in PUBLIC_SERIES
            else "macro_observations",
            timing_semantics="released_vintage_at_or_before_explicit_decision_time",
            null_policy="preserve_missing_components_and_insufficient_history",
            description=_description(recipe),
        )
        for name, recipe in RECIPES.items()
    )


def factor_unit(name):
    recipe = RECIPES[name]
    if recipe[0] == "matched_beta":
        return "decimal_return"
    if recipe[0] == "series":
        _, variable, transform = recipe
        unit = VARIABLES[variable][0]
        if transform == "implied_variance":
            return "annualized_variance"
        if transform in {
            "zscore",
            "percentile",
            "innovation_zscore",
            "growth",
            "yoy_growth",
            "log_level",
            "risk_neutral_skewness",
        }:
            return "dimensionless"
        return "bp" if unit == "percent" and transform != "level" else unit
    if name == "ml_us_net_liquidity_latest":
        return "USD_million"
    if recipe[0] in {"interaction", "industry_interaction"}:
        return factor_unit("ml_" + recipe[1])
    return (
        "dimensionless"
        if recipe[0] in {"state", "interaction"}
        else "return_per_macro_unit"
    )

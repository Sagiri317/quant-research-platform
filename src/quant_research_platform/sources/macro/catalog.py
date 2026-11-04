"""Map public vendor series onto distinct canonical macroeconomic variables."""

from dataclasses import dataclass

from quant_research_platform.contracts.macro import MACRO_VARIABLES


@dataclass(frozen=True)
class MacroSeries:
    source: str
    code: str
    unit: str
    frequency: str


SERIES = {
    name: MacroSeries(source, code, *MACRO_VARIABLES[name])
    for name, source, code in (
        ("us_fed_assets", "alfred", "WALCL"),
        ("us_tga_wed", "alfred", "WDTGAL"),
        ("us_tga_weekly_mean", "alfred", "WTREGEN"),
        ("us_reserves_wed", "alfred", "WRBWFRBL"),
        ("us_reserves_weekly_mean", "alfred", "WRESBAL"),
        ("us_on_rrp", "alfred", "RRPONTSYD"),
        ("us_sofr", "alfred", "SOFR"),
        ("us_iorb", "alfred", "IORB"),
        ("us_ioer", "alfred", "IOER"),
        ("us_effr", "alfred", "EFFR"),
        ("us_fed_funds", "alfred", "DFF"),
        ("us_target_lower", "alfred", "DFEDTARL"),
        ("us_target_upper", "alfred", "DFEDTARU"),
        ("us_treasury_3m", "alfred", "DGS3MO"),
        ("us_treasury_2y", "alfred", "DGS2"),
        ("us_treasury_5y", "alfred", "DGS5"),
        ("us_treasury_10y", "alfred", "DGS10"),
        ("us_treasury_30y", "alfred", "DGS30"),
        ("us_ig_oas", "alfred", "BAMLC0A0CM"),
        ("us_hy_oas", "alfred", "BAMLH0A0HYM2"),
        ("us_vix", "alfred", "VIXCLS"),
        ("us_nfci", "alfred", "NFCI"),
        ("us_anfci", "alfred", "ANFCI"),
        ("us_broad_dollar", "alfred", "DTWEXBGS"),
        ("us_broad_dollar_legacy", "alfred", "DTWEXB"),
        ("us_m2", "alfred", "M2SL"),
        ("cn_usd_cny", "alfred", "DEXCHUS"),
        ("cn_m2_imf", "alfred", "MYAGM2CNM189N"),
        ("cn_interbank_3m_oecd", "alfred", "IR3TIB01CNM156N"),
        ("cn_treasury_10y_oecd", "alfred", "IRLTLT01CNM156N"),
        ("eu_ecb_assets", "alfred", "ECBASSETSW"),
        ("eu_deposit_rate", "alfred", "ECBDFR"),
        ("eu_usd_per_eur", "alfred", "DEXUSEU"),
        ("jp_boj_assets", "alfred", "JPNASSETS"),
        ("jp_jpy_per_usd", "alfred", "DEXJPUS"),
        ("hk_hibor_on", "hkma", "hibor_overnight"),
        ("hk_hibor_1m", "hkma", "hibor_fixing_1m"),
        ("hk_balance_open", "hkma", "opening_balance"),
        ("hk_balance_close", "hkma", "closing_balance"),
        ("hk_base_rate", "hkma", "disc_win_base_rate"),
        ("cn_m2", "eastmoney", "BASIC_CURRENCY"),
        ("cn_m2_yoy", "eastmoney", "BASIC_CURRENCY_SAME"),
        ("cn_tsf_flow", "release_csv", "AFRE_FLOW"),
        ("global_usd_credit", "bis", "Q.USD.3P.N.A.I.B.USD"),
        ("cn_dr007", "release_csv", "DR007"),
        ("cn_r007", "release_csv", "R007"),
        ("cn_shibor_1w", "release_csv", "SHIBOR1W"),
        ("cn_shibor_3m", "release_csv", "SHIBOR3M"),
        ("cn_reverse_repo", "release_csv", "OMO_REVERSE_REPO"),
        ("cn_repo_maturity", "release_csv", "OMO_MATURITY"),
        ("cn_repo_7d_rate", "release_csv", "OMO7D"),
        ("cn_treasury_1y", "release_csv", "CGB1Y"),
        ("cn_treasury_10y", "release_csv", "CGB10Y"),
        ("cn_credit_aaa_3y", "release_csv", "AAA3Y"),
        ("cn_treasury_3y", "release_csv", "CGB3Y"),
        ("cn_tsf_stock", "release_csv", "AFRE_STOCK"),
        ("jp_current_accounts", "boj", "MACAB2201"),
        ("us_fomc_surprise", "sf_fed", "Surprise"),
    )
}

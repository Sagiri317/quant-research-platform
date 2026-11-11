"""Canonical price-volume definitions, inputs and session windows."""

from __future__ import annotations

from collections.abc import Iterable

from quant_research_platform.contracts.data import DataGrain
from quant_research_platform.contracts.factors import FactorDefinition, FactorSet
from quant_research_platform.contracts.ids import MarketId

PRICE_VOLUME_FACTOR_FAMILY = "price_volume"

# Each entry owns the base inputs and the required session windows.
_PRICE_VOLUME_INPUT_WINDOWS: dict[str, tuple[tuple[str, ...], tuple[int, ...]]] = {
    "reversal_1d": (("adj_close",), (1,)),
    "reversal_5d": (("adj_close",), (5,)),
    "reversal_20d": (("adj_close",), (20,)),
    "momentum_60_20": (("adj_close",), (60, 20)),
    "momentum_120_20": (("adj_close",), (120, 20)),
    "momentum_252_20": (("adj_close",), (252, 20)),
    "return_acceleration_6m6m": (("adj_close",), (1,)),
    "volatility_20d": (("period_return",), (20,)),
    "volatility_60d": (("period_return",), (60,)),
    "downside_volatility_20d": (("period_return",), (20,)),
    "downside_volatility_60d": (("period_return",), (60,)),
    "max_return_20d": (("period_return",), (20,)),
    "skewness_60d": (("period_return",), (60,)),
    "drawdown_60d": (("adj_close",), (60,)),
    "amplitude_20d": (("high", "low", "close"), (20,)),
    "turnover_20d": (("turnover",), (20,)),
    "turnover_60d": (("turnover",), (60,)),
    "turnover_volatility_20d": (("turnover",), (20,)),
    "amihud_20d": (("period_return", "amount"), (20,)),
    "amihud_60d": (("period_return", "amount"), (60,)),
    "amount_shock_20d": (("amount",), (20,)),
    "return_amount_corr_20d": (("period_return", "amount"), (20,)),
    "trend_efficiency_60d": (("period_return",), (60,)),
    "signed_amount_flow_20d": (("period_return", "amount"), (20,)),
    "log_float_market_cap": (("float_market_cap",), ()),
    "momentum_60d": (("adj_close",), (60,)),
    "upside_volatility_20d": (("period_return",), (20,)),
    "min_return_20d": (("period_return",), (20,)),
    "kurtosis_60d": (("period_return",), (60,)),
    "avg_amount_20d": (("amount",), (20,)),
    "avg_volume_20d": (("volume",), (20,)),
    "amount_change_20d": (("amount",), (20,)),
    "volume_change_20d": (("volume",), (20,)),
    "turnover_change_20d": (("turnover",), (20,)),
    "amount_volatility_20d": (("amount",), (20,)),
    "volume_shock_20d": (("volume",), (20,)),
    "return_volume_corr_20d": (("period_return", "volume"), (20,)),
    "return_turnover_corr_20d": (
        ("period_return", "turnover"),
        (20,),
    ),
    "return_amount_imbalance_20d": (("period_return", "amount"), (20,)),
    "zero_amount_ratio_20d": (("amount",), (20,)),
    "log_total_market_cap": (("total_market_cap",), ()),
    "share_price": (("close",), ()),
    "avg_amount_126d": (("amount",), (126,)),
    "amount_cv_126d": (("amount",), (126,)),
    "turnover_cv_126d": (("turnover",), (126,)),
    "amihud_126d": (("period_return", "amount"), (126,)),
    "skewness_20d": (("period_return",), (20,)),
    "top5_return_mean_20d": (("period_return",), (20,)),
    "turnover_250d": (("turnover",), (250,)),
    "abnormal_turnover_20_250": (("turnover",), (250, 20)),
    "alpha101_054": (("open", "high", "low", "close"), ()),
    "alpha191_046": (("adj_close",), (24, 12, 6, 3)),
    "qlib_rsqr_20d": (("adj_close",), (20,)),
    "qlib_resi_20d": (("adj_close",), (20,)),
    # Twenty consecutive close differences require twenty-one session prices.
    "qlib_sump_20d": (("adj_close",), (21,)),
    "volatility_compression_20_60": (("period_return",), (60, 20)),
    "compression_trend_20_60": (("period_return",), (60, 20)),
    "breakout_efficiency_20d": (("adj_close",), (21,)),
    "turnover_absorption_20d": (("turnover", "period_return"), (21,)),
    "price_impact_efficiency_20d": (("adj_close", "turnover"), (21,)),
    "drawdown_turnover_pressure_20d": (("adj_close", "turnover"), (21,)),
    "drawdown_rebound_20d": (
        ("adj_close", "turnover", "period_return"),
        (21,),
    ),
    "price_position_60d": (("adj_close",), (60,)),
    "price_position_252d": (("adj_close",), (252,)),
    "relative_turnover_20d": (("turnover",), (21,)),
    "return_shock_1_20": (("period_return",), (21,)),
    "opening_gap": (("adj_open", "adj_close"), (2,)),
    "intraday_return": (("adj_open", "adj_close"), (1,)),
    "down_gap_recovery": (("adj_open", "adj_close"), (2,)),
    "up_gap_reversal": (("adj_open", "adj_close"), (2,)),
}

for _window in (60, 252):
    for _side in ("low_position_turnover_upshock", "high_position_turnover_downshock"):
        _PRICE_VOLUME_INPUT_WINDOWS[f"{_side}_{_window}d"] = (
            ("adj_close", "turnover", "period_return"),
            (_window + 1, 21),
        )

PRICE_VOLUME_THEME_DESCRIPTIONS = {
    "volatility_compression_20_60": (
        "QRP candidate: sample std(return,20)/std(return,60). Lower means "
        "recent volatility compression; no assumed future-return direction."
    ),
    "compression_trend_20_60": (
        "QRP candidate: max(1-std(return,20)/std(return,60),0) * "
        "sum(return,20)/sum(abs(return),20). Signed past trend during compression."
    ),
    "breakout_efficiency_20d": (
        "QRP candidate: signed distance outside the preceding 20 adjusted closes' "
        "range divided by the last 20 absolute close changes. Positive is an "
        "upward break; a complete flat path is undefined."
    ),
    "turnover_absorption_20d": (
        "QRP candidate: turnover/current-prior-20 mean turnover * "
        "(1-abs(current return)/prior-20 mean absolute return). Higher means "
        "more turnover with less price movement, not observed investor identity."
    ),
    "price_impact_efficiency_20d": (
        "QRP candidate: adjusted-close return over 20 sessions divided by "
        "the last 20 turnover observations. Signed return per turnover unit; "
        "not a causal estimate of market impact."
    ),
    "drawdown_turnover_pressure_20d": (
        "QRP candidate: max(1-current close/prior-20 maximum close,0) * "
        "current turnover/prior-20 mean turnover. Higher is drawdown with "
        "unusual turnover, not proof of forced liquidation."
    ),
    "drawdown_rebound_20d": (
        "QRP candidate: drawdown_turnover_pressure_20d * current return. "
        "Positive is an observed rebound while still below the past peak; "
        "no future return enters this candidate."
    ),
    "relative_turnover_20d": (
        "QRP candidate: current turnover / mean of the preceding 20 turnover "
        "observations. Unitless; a zero baseline is undefined."
    ),
    "return_shock_1_20": (
        "QRP candidate: (current return - prior-20 mean return) / prior-20 "
        "sample return standard deviation. A zero scale is undefined."
    ),
    "opening_gap": (
        "QRP candidate: current adjusted open / preceding adjusted close - 1. "
        "Corporate-action-consistent overnight gap, released next session "
        "under the daily engine's conservative timing."
    ),
    "intraday_return": (
        "QRP candidate: current adjusted close / adjusted open - 1. "
        "Realized same-day movement, never an opening-time predictor."
    ),
    "down_gap_recovery": (
        "QRP candidate: max(-opening_gap,0) * intraday_return. Positive is "
        "realized recovery after a gap down; negative is further decline. "
        "Available only after the close, released next session."
    ),
    "up_gap_reversal": (
        "QRP candidate: max(opening_gap,0) * -intraday_return. Positive is "
        "realized reversal after a gap up, negative is continuation. "
        "Available only after the close, released next session."
    ),
}

for _window in (60, 252):
    PRICE_VOLUME_THEME_DESCRIPTIONS[f"price_position_{_window}d"] = (
        f"QRP candidate: (current adjusted close - trailing-{_window} minimum) / "
        f"trailing-{_window} adjusted-close range, including current close. "
        "A flat range is undefined; this is observable position, not a confirmed top."
    )
    for _side, _position, _shock in (
        ("low_position_turnover_upshock", "1-position", "return_shock"),
        ("high_position_turnover_downshock", "position", "-return_shock"),
    ):
        PRICE_VOLUME_THEME_DESCRIPTIONS[f"{_side}_{_window}d"] = (
            f"QRP candidate: ({_position}) * max(relative_turnover_20d-1,0) * "
            f"max({_shock}_1_20,0). Position uses the {_window} adjusted closes "
            "ending YESTERDAY; turnover and return baselines also exclude today. "
            "No future-confirmed bottom/top or assumed predictive direction."
        )

PRICE_VOLUME_LEVEL_DESCRIPTIONS = {}
for _window in (20, 60, 252):
    for _name, _description in {
        "peak_drawdown": (
            "max(1-current adjusted close/prior-window maximum adjusted close,0)"
        ),
        "peak_age": (
            "sessions since the most recent maximum adjusted close in the prior window"
        ),
        "up_leg_retracement": (
            "(prior-window peak-current adjusted close)/(peak-minimum strictly before "
            "the most recent peak within that window); undefined without an earlier low"
        ),
        "support_distance": (
            "current adjusted close/prior-window minimum adjusted low-1"
        ),
        "resistance_distance": (
            "current adjusted close/prior-window maximum adjusted high-1"
        ),
        "support_rejection": (
            "(current adjusted close-current adjusted low)/prior support when "
            "current low is in [support,1.01*support], otherwise zero"
        ),
        "resistance_rejection": (
            "(current adjusted high-current adjusted close)/prior resistance when "
            "current high is in [0.99*resistance,resistance], otherwise zero"
        ),
    }.items():
        _factor = f"{_name}_{_window}d"
        _fields = (
            ("adj_close", "low" if _name.startswith("support_") else "high", "close")
            if _name.startswith(("support_", "resistance_"))
            else ("adj_close",)
        )
        _PRICE_VOLUME_INPUT_WINDOWS[_factor] = (_fields, (_window + 1,))
        PRICE_VOLUME_LEVEL_DESCRIPTIONS[_factor] = (
            f"QRP candidate, {_window} prior sessions: {_description}. "
            "References exclude today; no future-confirmed extrema. Adjusted intraday "
            "extremes use each bar's adj_close/close, an analytical reference rather "
            "than a reconstructed historical executable quote. Current peak drawdown "
            "is distinct from maximum drawdown during a historical path."
        )

for _window in (20, 60, 252):
    for _side, _extreme in (("up", "high"), ("down", "low")):
        _name = f"failed_{_side}_breakout_{_window}d"
        _PRICE_VOLUME_INPUT_WINDOWS[_name] = (
            ("adj_close", _extreme, "close"),
            (_window + 1,),
        )
        PRICE_VOLUME_LEVEL_DESCRIPTIONS[_name] = (
            f"Same-session failed {_side} breach of the preceding "
            f"{_window} adjusted {_extreme} range. "
            "Nonnegative intraday excursion beyond the reference, conditional"
            " on closing back at or inside it; zero otherwise. "
            "Known after the close only. Adjusted extremes use the same bar's"
            " adj_close/close."
        )
        for _horizon in (3, 5):
            _name = f"failed_close_{_side}_breakout_{_horizon}_{_window}d"
            _PRICE_VOLUME_INPUT_WINDOWS[_name] = (
                ("adj_close", _extreme, "close"),
                (_window + 1,),
            )
            PRICE_VOLUME_LEVEL_DESCRIPTIONS[_name] = (
                "User hypothesis: first strict closing reentry within "
                f"{_horizon} sessions of a closing-confirmed {_side} breakout "
                f"of the preceding {_window} adjusted {_extreme} range. "
                "Freeze the original boundary until reentry; intraday retests"
                " and closes at the boundary do not trigger failure. "
                "Value is nonnegative closing penetration into the original "
                "range, zero off the event, and known only at reentry close."
            )
for _window in (20, 60, 252):
    for _side in ("up", "down"):
        _name = f"{_side}_break_retest_5_{_window}d"
        _PRICE_VOLUME_INPUT_WINDOWS[_name] = (
            ("adj_close", "high", "low", "close"),
            (_window + 6,),
        )
        PRICE_VOLUME_LEVEL_DESCRIPTIONS[_name] = (
            f"QRP candidate: most recent still-held {_side} close breakout in the "
            f"preceding five sessions; freeze its preceding-{_window} adjusted "
            "high/low reference. All later closes must remain on the breakout side. "
            "Current low/high must touch within 1% of that frozen level; output "
            "current close-to-touch recovery/rejection divided by that level, else "
            "zero. No candidate is zero; incomplete required history is undefined."
        )
PRICE_VOLUME_THEME_DESCRIPTIONS.update(PRICE_VOLUME_LEVEL_DESCRIPTIONS)

PRICE_VOLUME_PROFILE_DESCRIPTIONS = {
    "traded_amount_node_distance_60d": (
        "log(current adjusted close / geometric center of the largest amount bin)"
    ),
    "traded_amount_concentration_60d": "largest price-bin amount / total amount",
    "traded_amount_above_price_60d": (
        "fraction of historical amount allocated above the current adjusted close"
    ),
}
for _name, _formula in PRICE_VOLUME_PROFILE_DESCRIPTIONS.items():
    _PRICE_VOLUME_INPUT_WINDOWS[_name] = (
        ("adj_close", "high", "low", "close", "amount"),
        (61,),
    )
    PRICE_VOLUME_THEME_DESCRIPTIONS[_name] = (
        f"QRP daily amount-profile proxy: {_formula}. Allocate each of the PRIOR "
        "60 sessions' CNY amount uniformly over its adjusted low-high price interval "
        "into fixed log-price bins of width log(1.02), anchored at price 1; a flat "
        "bar allocates all amount to its bin. Adjustment uses adj_close/close. "
        "Requires complete nonnegative amounts and valid price intervals. Zero "
        "total amount is undefined. Bin totals conserve amount; this is neither "
        "observed intraday volume-at-price nor outstanding investors' cost basis."
    )

_PUBLISHED_PRICE_VOLUME_DESCRIPTIONS = {
    "return_acceleration_6m6m": (
        "Xiong-Ibbotson-inspired monthly acceleration: mean simple returns in "
        "the latest six fully completed calendar months minus the preceding six; "
        "last session prices preserve missing month-end prices, never skip months. "
        "The current unfinished month is excluded. This implements the described "
        "six-versus-six comparison, without the paper example threshold or "
        "portfolio rule. "
        "Source: "
        "https://insights.som.yale.edu/insights/should-investors-look-for-stocks-with-momentum"
    ),
    "alpha101_054": (
        "Alpha101 #54: -((low-close)*open^5)/((low-high)*close^5). "
        "Uses same-session raw OHLC; a common price adjustment cancels. "
        "Zero denominators preserve NULL. QRP releases at the next session, "
        "not the paper's delay-0 close execution. "
        "Source: https://arxiv.org/pdf/1601.00991v3, Appendix A, p. 11."
    ),
    "alpha191_046": (
        "Alpha191 #46: (Mean(C,3)+Mean(C,6)+Mean(C,12)+Mean(C,24))/(4*C), "
        "C=adj_close. Requires 24 complete own-market sessions; zero close "
        "preserves NULL. Source: Guotai Junan, 2017-06-15, Table 6, p. 12, "
        "https://guorn.com/static/upload/file/3/134065454575605.pdf."
    ),
    "qlib_rsqr_20d": (
        "Qlib Alpha158 RSQR20: R-squared of adj_close on an intercept and "
        "session positions 1..20. Requires 20 complete own-market sessions; "
        "sample standard deviation <= 2e-5 preserves NULL as in Qlib v0.9.7. "
        "Source: https://github.com/microsoft/qlib/blob/v0.9.7/"
        "qlib/contrib/data/loader.py; operator: qlib/data/ops.py."
    ),
    "qlib_resi_20d": (
        "Qlib Alpha158 RESI20: current OLS residual / current adj_close, "
        "regressing adj_close on an intercept and session positions 1..20. "
        "Requires 20 complete own-market sessions; zero current close "
        "preserves NULL. Source: https://github.com/microsoft/qlib/blob/"
        "v0.9.7/qlib/contrib/data/loader.py; operator: qlib/data/_libs/rolling.pyx."
    ),
    "qlib_sump_20d": (
        "Qlib Alpha158 SUMP20: Sum(max(delta(C,1),0),20) / "
        "(Sum(abs(delta(C,1)),20)+1e-12), C=adj_close. Requires 21 complete "
        "own-market session prices; a complete flat path is zero. "
        "Source: https://github.com/microsoft/qlib/blob/v0.9.7/"
        "qlib/contrib/data/loader.py. Unlike upstream partial rolling windows, "
        "QRP preserves NULL for incomplete history."
    ),
}

for _width in (20, 60, 252):
    for _kind, _description in (
        (
            "signed_breakout_duration",
            (
                "Signed age of an active close breakout; the preceding range "
                "boundary is frozen at the first breakout and the episode ends on"
                " reentry."
            ),
        ),
        (
            "signed_breakout_followthrough",
            (
                "Close return since the first active breakout, retaining its "
                "natural return sign; zero outside an active episode."
            ),
        ),
        (
            "breakout_activity_confirmation",
            (
                "Active breakout direction times log(current amount / "
                "preceding-window mean amount); zero outside an active episode."
            ),
        ),
    ):
        _column = f"{_kind}_{_width}d"
        _PRICE_VOLUME_INPUT_WINDOWS[_column] = (
            ("adj_close", "amount")
            if _kind == "breakout_activity_confirmation"
            else ("adj_close",),
            (_width + 1,),
        )
        PRICE_VOLUME_THEME_DESCRIPTIONS[_column] = _description

for _width in (20, 60):
    _name = f"breakout_activity_divergence_{_width}d"
    _PRICE_VOLUME_INPUT_WINDOWS[_name] = (("adj_close", "amount"), (_width + 1,))
    PRICE_VOLUME_THEME_DESCRIPTIONS[_name] = (
        "QRP price-volume hypothesis: signed log-price excess outside the "
        "preceding window's adjusted-close range times max(log(prior-window "
        "mean amount / current amount),0). Zero inside the known price range; "
        "an actual breakout requires complete nonnegative prior amounts and "
        "positive current and mean amounts. Positive is a weak-activity upward "
        "break; negative is a weak-activity downward break, not a forecast sign. "
        "The volume-information mechanism is motivated by Blume, Easley and "
        "O'Hara (1994), not their original factor formula: "
        "https://doi.org/10.1111/j.1540-6261.1994.tb04424.x."
    )
    for _side in ("up", "down"):
        _name = f"price_{_side}_acceleration_{_width}d"
        _PRICE_VOLUME_INPUT_WINDOWS[_name] = (("adj_close",), (_width,))
        PRICE_VOLUME_THEME_DESCRIPTIONS[_name] = (
            "QRP acceleration hypothesis: quadratic OLS of log adjusted prices "
            "on session time; twice quadratic coefficient / RMS log returns. "
            "Positive curvature with positive endpoint slope for up, negative "
            "curvature with negative endpoint slope for down; otherwise zero. "
            "Complete positive prices, flat paths undefined; not chart angles "
            "or the monthly Xiong-Ibbotson 6-versus-6 definition."
        )

PRICE_VOLUME_LIBRARY = tuple(_PRICE_VOLUME_INPUT_WINDOWS)
PRICE_VOLUME_PATH_FACTORS = (
    "positive_return_fraction_20d",
    "largest_daily_contribution_20d",
    "path_roughness_20d",
)
_PRICE_VOLUME_INPUT_WINDOWS.update(
    {column: (("period_return",), (20,)) for column in PRICE_VOLUME_PATH_FACTORS}
)

PRICE_VOLUME_UNITS = dict.fromkeys(_PRICE_VOLUME_INPUT_WINDOWS, "dimensionless")
PRICE_VOLUME_UNITS["return_acceleration_6m6m"] = "decimal_return"
PRICE_VOLUME_UNITS.update(
    {
        f"signed_breakout_duration_{width}d": "signed_trading_sessions"
        for width in (20, 60, 252)
    }
)
PRICE_VOLUME_UNITS.update(
    {
        f"signed_breakout_followthrough_{width}d": "decimal_return"
        for width in (20, 60, 252)
    }
)
for _name in PRICE_VOLUME_UNITS:
    if _name.startswith(
        (
            "reversal_",
            "momentum_",
            "volatility_",
            "downside_volatility_",
            "upside_volatility_",
            "peak_drawdown_",
        )
    ) or _name in {
        "max_return_20d",
        "min_return_20d",
        "top5_return_mean_20d",
        "drawdown_60d",
        "opening_gap",
        "intraday_return",
        "qlib_resi_20d",
    }:
        # Compression is a ratio of two volatilities, not return volatility.
        if _name != "volatility_compression_20_60":
            PRICE_VOLUME_UNITS[_name] = "decimal_return"
    elif _name.startswith("amihud_"):
        PRICE_VOLUME_UNITS[_name] = "decimal_return_per_CNY"
    elif _name.startswith("peak_age_"):
        PRICE_VOLUME_UNITS[_name] = "trading_sessions"
    elif _name.startswith("avg_amount_") or _name == "amount_volatility_20d":
        PRICE_VOLUME_UNITS[_name] = "CNY"
    elif _name == "avg_volume_20d":
        PRICE_VOLUME_UNITS[_name] = "shares"
    elif _name in {"share_price"}:
        PRICE_VOLUME_UNITS[_name] = "CNY_per_share"
    elif _name in {
        "log_float_market_cap",
        "log_total_market_cap",
    }:
        PRICE_VOLUME_UNITS[_name] = "log_CNY"
    elif _name in {
        "turnover_20d",
        "turnover_60d",
        "turnover_250d",
        "turnover_volatility_20d",
    }:
        PRICE_VOLUME_UNITS[_name] = "share_turnover_fraction"


# Only sample statistics admit partial observations within the nominal window.
PRICE_VOLUME_PARTIAL_STATISTICS = frozenset(
    factor
    for factor in _PRICE_VOLUME_INPUT_WINDOWS
    if factor not in PRICE_VOLUME_THEME_DESCRIPTIONS
    and (
        factor in {"amount_shock_20d", "volume_shock_20d"}
        or factor.startswith(
            (
                "avg_amount_",
                "avg_volume_",
                "volatility_",
                "downside_volatility_",
                "amihud_",
                "skewness_",
            )
        )
        or factor.endswith("_corr_20d")
        or factor
        in {
            "turnover_20d",
            "turnover_60d",
            "turnover_250d",
            "amount_volatility_20d",
            "turnover_volatility_20d",
            "upside_volatility_20d",
            "kurtosis_60d",
            "amount_cv_126d",
            "turnover_cv_126d",
            "abnormal_turnover_20_250",
            "amplitude_20d",
        }
    )
)


def price_volume_factor_definitions(
    *,
    market_id: MarketId,
    required_input_data_id: str,
    factor_columns: Iterable[str] | None = None,
) -> tuple[FactorDefinition, ...]:
    """Bind canonical definitions without changing their identity across selections."""
    selected = (
        PRICE_VOLUME_LIBRARY
        if factor_columns is None
        else tuple(dict.fromkeys(factor_columns))
    )
    definitions = []
    for column in selected:
        try:
            inputs, windows = _PRICE_VOLUME_INPUT_WINDOWS[column]
        except KeyError as exc:
            raise ValueError(f"unknown_price_volume_factor: {column}") from exc
        lookback = max(windows, default=None)
        definitions.append(
            FactorDefinition(
                factor_id=f"price_volume_{column}",
                market_id=market_id,
                factor_set=FactorSet.CORE
                if column
                in {
                    "momentum_252_20",
                    "reversal_5d",
                    "volatility_20d",
                    "amihud_20d",
                    "turnover_20d",
                    "log_float_market_cap",
                }
                else FactorSet.STANDARD,
                input_columns=(*inputs, "filling_flag")
                if column in PRICE_VOLUME_PARTIAL_STATISTICS
                else inputs,
                output_column=column,
                display_name=column,
                description=PRICE_VOLUME_THEME_DESCRIPTIONS.get(column)
                or _PUBLISHED_PRICE_VOLUME_DESCRIPTIONS.get(column)
                or (
                    (
                        "Share of nonpositive amount (amount <= 0) over 20 complete "
                        "sessions; missing amounts propagate."
                    )
                    if column == "zero_amount_ratio_20d"
                    else (
                        "Statistic over the full nominal own-market session window; "
                        "requires ceil(min_fresh_fraction * window) jointly valid "
                        "fresh "
                        "observations (default 0.75), plus mathematical sample minima. "
                        "Numeric denominators use actual valid samples; confirmed "
                        "suspension valuations do not increase the fresh count."
                    )
                    if column in PRICE_VOLUME_PARTIAL_STATISTICS
                    else (
                        "Price-volume factor using complete session windows or exact "
                        "endpoints; missing required observations and invalid "
                        "denominators preserve NULL."
                    )
                ),
                family=PRICE_VOLUME_FACTOR_FAMILY,
                required_windows=windows,
                lookback_window=lookback,
                full_history=column == "return_acceleration_6m6m"
                or column.startswith(
                    (
                        "signed_breakout_",
                        "breakout_activity_confirmation_",
                        "failed_close_",
                    )
                ),
                grain=DataGrain.ASSET_TIMESTAMP,
                timing_semantics="same_session_close_available_at_next_session",
                null_policy="preserve_nulls_after_min_periods",
                source_layer="processed",
                required_input_data_id=required_input_data_id,
            )
        )
    return tuple(definitions)

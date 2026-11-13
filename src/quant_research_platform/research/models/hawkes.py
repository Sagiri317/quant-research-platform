"""Bivariate exponential Hawkes likelihood on observed signed-trade segments.

Kernels are alpha[target, source] * decay * exp(-decay * age_seconds).
Coincident timestamps share their pre-jump intensity; no artificial jitter is added.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import solve_banded
from scipy.optimize import minimize

from quant_research_platform.contracts.ids import validate_input_market
from quant_research_platform.contracts.json_codec import (
    _strict_config,
    parse_qrp_json_object,
)
from quant_research_platform.contracts.time import (
    TimeWindow,
    time_window_timestamp_interval,
)
from quant_research_platform.contracts.validation import strict_finite_float
from quant_research_platform.markets.implementations.a_share.level2 import (
    first_unknown_event_time,
    observed_segments,
)
from quant_research_platform.storage.atomic import (
    atomic_write_directory,
    reject_if_exists,
)
from quant_research_platform.storage.datasets.reading import (
    _dataset_input_options,
    materialize_dataset_session,
    open_dataset_session,
)
from quant_research_platform.storage.manifests import write_qrp_json_file


def branching_radius(alpha):
    return (
        alpha[0, 0]
        + alpha[1, 1]
        + np.sqrt((alpha[0, 0] - alpha[1, 1]) ** 2 + 4 * alpha[0, 1] * alpha[1, 0])
    ) / 2


def hawkes_parameters(value):
    params = _strict_config(value, {"baseline", "branching", "decay"})
    mu = np.asarray(params["baseline"], dtype=float)
    alpha = np.asarray(params["branching"], dtype=float)
    beta = strict_finite_float(params["decay"], error="hawkes_parameters_invalid")
    if (
        mu.shape != (2,)
        or alpha.shape != (2, 2)
        or not np.isfinite(mu).all()
        or not np.isfinite(alpha).all()
        or (mu <= 0).any()
        or (alpha < 0).any()
        or beta <= 0
        or branching_radius(alpha) >= 1
    ):
        raise ValueError("hawkes_parameters_invalid")
    return mu, alpha, beta


def trade_events(trades, origin):
    selected = trades.loc[
        trades.event_kind.eq("trade") & trades.side.isin(["buy", "sell"])
    ]
    if selected.empty:
        return np.empty(0), np.empty((0, 2))
    counts = (
        selected.groupby(["timestamp", "side"], sort=True)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["buy", "sell"], fill_value=0)
    )
    times = (counts.index - origin).total_seconds().to_numpy()
    return times, counts.to_numpy(dtype=float)


def _excitation(times, counts, beta, *, derivative=False):
    """Solve the stable exponential recurrence in compiled O(n) banded algebra."""
    if not len(times):
        return counts.copy(), counts.copy() if derivative else None
    elapsed = np.diff(times)
    decay = np.exp(-beta * elapsed)
    band = np.zeros((2, len(times)))
    band[0] = 1.0
    band[1, :-1] = -decay
    rhs = np.zeros_like(counts)
    rhs[1:] = decay[:, None] * counts[:-1]
    state = solve_banded((1, 0), band, rhs, check_finite=False)
    gradient = None
    if derivative:
        rhs[1:] = -elapsed[:, None] * decay[:, None] * (state[:-1] + counts[:-1])
        gradient = solve_banded((1, 0), band, rhs, check_finite=False)
    return state, gradient


def negative_log_likelihood(theta, segments):
    """Return exact conditional negative log likelihood and its analytic gradient."""
    mu, alpha, beta = theta[:2], theta[2:6].reshape(2, 2), theta[6]
    loss, grad_mu, grad_alpha, grad_beta = 0.0, np.zeros(2), np.zeros((2, 2)), 0.0
    for times, counts, duration in segments:
        state, derivative = _excitation(times, counts, beta, derivative=True)
        intensity = mu + beta * state @ alpha.T
        weights = counts / intensity
        age = duration - times
        integrals = (counts * (-np.expm1(-beta * age))[:, None]).sum(axis=0)
        integral_derivative = (counts * (age * np.exp(-beta * age))[:, None]).sum(
            axis=0
        )
        loss += (
            duration * mu.sum()
            + alpha.sum(axis=0) @ integrals
            - (counts * np.log(intensity)).sum()
        )
        grad_mu += duration - weights.sum(axis=0)
        grad_alpha += integrals[None, :] - weights.T @ (beta * state)
        grad_beta += (
            alpha.sum(axis=0) @ integral_derivative
            - (weights * ((state + beta * derivative) @ alpha.T)).sum()
        )
    return float(loss), np.r_[grad_mu, grad_alpha.ravel(), grad_beta]


def intensities(times, counts, queries, params):
    mu, alpha, beta = params
    state, _ = _excitation(times, counts, beta)
    index = np.searchsorted(times, queries, side="left") - 1
    result = np.broadcast_to(mu, (len(queries), 2)).copy()
    valid = index >= 0
    chosen = index[valid]
    history = (state[chosen] + counts[chosen]) * np.exp(
        -beta * (queries[valid] - times[chosen])
    )[:, None]
    result[valid] += beta * history @ alpha.T
    return result


def fit_hawkes(config):
    """Fit across selected assets; reset observed history at each segment."""
    config = _strict_config(
        config,
        {
            "inputs",
            "asset_ids",
            "time_window",
            "fit_cutoff",
            "initial_parameters",
            "max_branching",
            "max_iterations",
            "tolerance",
            "output",
        },
    )
    inputs = _strict_config(config["inputs"], {"hft_panel", "hft_trades"})
    assets = config["asset_ids"]
    if (
        not isinstance(assets, list)
        or not assets
        or any(type(asset) is not str or not asset for asset in assets)
        or len(set(assets)) != len(assets)
    ):
        raise ValueError("hawkes_asset_ids_invalid")
    window = TimeWindow(**config["time_window"])
    lower, upper, _ = time_window_timestamp_interval(window)
    cutoff = pd.Timestamp(config["fit_cutoff"])
    if pd.isna(cutoff) or cutoff.tz is not None or pd.Timestamp(upper) > cutoff:
        raise ValueError("hawkes_training_after_cutoff")
    mu, alpha, beta = hawkes_parameters(config["initial_parameters"])
    maximum = strict_finite_float(
        config["max_branching"], error="hawkes_fit_parameter_invalid"
    )
    tolerance = strict_finite_float(
        config["tolerance"], error="hawkes_fit_parameter_invalid"
    )
    if (
        not 0 < maximum < 1
        or branching_radius(alpha) > maximum
        or tolerance <= 0
        or type(config["max_iterations"]) is not int
        or config["max_iterations"] < 1
    ):
        raise ValueError("hawkes_fit_parameter_invalid")
    output = Path(config["output"])
    reject_if_exists(output)
    columns = {
        "hft_panel": [
            "asset_id",
            "timestamp",
            "session_date",
            "segment_id",
        ],
        "hft_trades": ["asset_id", "timestamp", "source_row", "event_kind", "side"],
    }
    sources = {
        name: open_dataset_session(
            **_dataset_input_options(
                path,
                time_window=window,
                row_filters=(("asset_id", "in", tuple(assets)),),
            )
        ).projected_view(columns[name])
        for name, path in inputs.items()
    }
    if any(source.spec["data_id"] != name for name, source in sources.items()):
        raise ValueError("hawkes_input_identity_mismatch")
    for source in sources.values():
        validate_input_market(source.spec, "a_share")
    segments, observed_assets, unknown = [], set(), 0
    unknown_kind_events = truncated_segments = 0
    for day in pd.date_range(
        pd.Timestamp(lower).normalize(), pd.Timestamp(upper).normalize(), freq="D"
    ):
        day_window = TimeWindow(day.date(), day.date())
        data = {
            name: materialize_dataset_session(source.filtered_view(day_window))
            for name, source in sources.items()
        }
        panel = data["hft_panel"]
        for quotes, selected in observed_segments(panel, trades=data["hft_trades"]):
            if len(quotes) < 2:
                continue
            start, end = quotes.timestamp.iloc[0], quotes.timestamp.iloc[-1]
            trades = selected["trades"]
            trades = trades.loc[trades.timestamp.lt(end)]
            unknown_at = first_unknown_event_time(trades)
            unknown_kind_events += int(trades.event_kind.eq("unknown").sum())
            if pd.notna(unknown_at):
                truncated_segments += 1
                quotes = quotes.loc[quotes.timestamp.lt(unknown_at)]
                if len(quotes) < 2:
                    continue
                end = quotes.timestamp.iloc[-1]
                trades = trades.loc[trades.timestamp.lt(end)]
            unknown += int(
                (
                    trades.event_kind.eq("trade") & ~trades.side.isin(["buy", "sell"])
                ).sum()
            )
            times, counts = trade_events(trades, start)
            segments.append((times, counts, (end - start).total_seconds()))
            observed_assets.add(quotes.asset_id.iloc[0])
    totals = sum((counts.sum(axis=0) for _, counts, _ in segments), np.zeros(2))
    if set(assets) != observed_assets or (totals == 0).any():
        raise ValueError("hawkes_observed_history_insufficient")
    initial = np.r_[mu, alpha.ravel(), beta]
    event_count = float(totals.sum())

    def objective(theta):
        # Scale the objective and tolerance together to condition large samples.
        loss, gradient = negative_log_likelihood(theta, segments)
        return loss / event_count, gradient / event_count

    fitted = minimize(
        objective,
        initial,
        jac=True,
        method="SLSQP",
        bounds=[(1e-10, None)] * 2 + [(0, None)] * 4 + [(1e-8, None)],
        constraints=[
            {
                "type": "ineq",
                "fun": lambda x: maximum - branching_radius(x[2:6].reshape(2, 2)),
            }
        ],
        options={"maxiter": config["max_iterations"], "ftol": tolerance / event_count},
    )
    if (
        not fitted.success
        or not np.isfinite(fitted.fun)
        or branching_radius(fitted.x[2:6].reshape(2, 2)) > maximum + 1e-8
    ):
        raise ValueError(f"hawkes_fit_failed_status_{fitted.status}")
    parameters = {
        "baseline": fitted.x[:2].tolist(),
        "branching": fitted.x[2:6].reshape(2, 2).tolist(),
        "decay": float(fitted.x[6]),
    }
    facts = {
        "kind": "bivariate_exponential_hawkes",
        "config": config,
        "parameters": parameters,
        "fit_cutoff": cutoff.isoformat(),
        "observed_event_counts": totals.astype(int).tolist(),
        "unknown_side_trades_excluded": unknown,
        "unknown_kind_events": unknown_kind_events,
        "unknown_kind_truncated_segments": truncated_segments,
        "segments": len(segments),
        "negative_log_likelihood": float(fitted.fun * event_count),
        "iterations": int(fitted.nit),
        "semantics": (
            "shared_observed_signed_trades_seconds_"
            "zero_history_per_segment_pre_jump_ties"
        ),
    }
    atomic_write_directory(
        output, lambda staging: write_qrp_json_file(staging / "model.json", facts)
    )
    return output


def load_hawkes(path):
    facts = parse_qrp_json_object((Path(path) / "model.json").read_bytes())
    if facts["kind"] != "bivariate_exponential_hawkes":
        raise ValueError("hawkes_model_kind_invalid")
    parameters = facts["parameters"]
    return (
        np.asarray(parameters["baseline"], dtype=float),
        np.asarray(parameters["branching"], dtype=float),
        parameters["decay"],
    ), pd.Timestamp(facts["fit_cutoff"])

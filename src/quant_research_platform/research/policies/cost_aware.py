"""Differentiable long-only allocation, evaluated by native share accounting."""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from stable_baselines3.common.utils import get_device
from torch import nn

from quant_research_platform.artifacts.portfolio_ledger import (
    write_portfolio_ledger_artifact,
)
from quant_research_platform.contracts.json_codec import (
    _strict_config,
    parse_qrp_json_object,
)
from quant_research_platform.contracts.policies import PolicyDecision
from quant_research_platform.contracts.portfolios import PortfolioCapital
from quant_research_platform.contracts.time import (
    TimeWindow,
    time_window_timestamp_interval,
)
from quant_research_platform.research.datasets.learning import (
    mature_rows,
    research_frame,
)
from quant_research_platform.research.experiments import TrialRecorder
from quant_research_platform.research.models.scaler import ModelMatrixScaler
from quant_research_platform.storage.atomic import (
    atomic_write_directory,
    atomic_write_json,
    reject_if_exists,
)


class CostAwareAllocator(nn.Module):
    """Score each asset with shared weights and allocate the remainder to cash."""

    def __init__(self, feature_count, hidden_sizes):
        super().__init__()
        layers = []
        width = feature_count + 1
        for hidden in hidden_sizes:
            layers.extend((nn.Linear(width, hidden), nn.Tanh()))
            width = hidden
        layers.append(nn.Linear(width, 1))
        self.score = nn.Sequential(*layers)
        self.cash_score = nn.Parameter(torch.zeros(()))

    def forward(self, features, previous):
        logits = self.score(
            torch.cat((features, previous.unsqueeze(-1)), dim=-1)
        ).squeeze(-1)
        return torch.softmax(torch.cat((logits, self.cash_score.reshape(1))), dim=-1)[
            :-1
        ]


def differentiable_returns(model, features, returns, cost_rate):
    """Return sequential net returns after proportional turnover costs.

    Features have shape (time, asset, feature), returns (time, asset), and costs use
    decimal notional rates. Post-cost holdings drift before the next allocation.
    This training surrogate does not implement discrete shares or market execution."""
    previous = torch.zeros(
        features.shape[1], dtype=features.dtype, device=features.device
    )
    net, turnover = [], []
    for x, y in zip(features, returns, strict=True):
        weights = model(x, previous)
        traded = (weights - previous).abs().sum()
        gross = 1 + torch.dot(weights, y)
        net.append((1 - cost_rate * traded) * gross - 1)
        turnover.append(traded)
        previous = weights * (1 + y) / gross
    return torch.stack(net), torch.stack(turnover)


def fit_cost_model(config):
    """Fit a continuous-weight allocator on consecutive mature return intervals.

    Scaling is estimated on the supplied training panel. Save model weights,
    scaling moments, and the cost approximation for later native-account replay.
    """
    config = _strict_config(
        config,
        {
            "dataset",
            "feature_columns",
            "asset_ids",
            "label",
            "return_end_column",
            "window",
            "cutoff",
            "cost_bps",
            "risk_aversion",
            "epochs",
            "learning_rate",
            "hidden_sizes",
            "device",
            "threads",
            "seed",
            "output",
        },
        set(),
    )
    for name in ("epochs", "threads"):
        if type(config[name]) is not int or config[name] < 1:
            raise ValueError("cost_learning_budget_invalid")
    if any(type(n) is not int or n < 1 for n in config["hidden_sizes"]):
        raise ValueError("cost_learning_hidden_size_invalid")
    if (
        not 0 <= config["cost_bps"] < 5000
        or not np.isfinite(config["risk_aversion"])
        or config["risk_aversion"] < 0
        or not np.isfinite(config["learning_rate"])
        or config["learning_rate"] <= 0
    ):
        raise ValueError("cost_learning_parameters_invalid")
    assets = config["asset_ids"]
    if not assets or len(assets) != len(set(assets)):
        raise ValueError("cost_learning_assets_invalid")
    frame, session, availability = research_frame(
        config["dataset"],
        config["window"],
        config["feature_columns"],
        label=config["label"],
        context_columns=(config["return_end_column"],),
    )
    spec = session.spec
    if spec.get("column_units", {}).get(config["label"]) != "decimal_return":
        raise ValueError("cost_learning_decimal_return_required")
    source = frame.loc[frame.asset_id.isin(assets)]
    frame = mature_rows(
        source,
        label=config["label"],
        availability=availability,
        cutoff=config["cutoff"],
    )
    if len(frame) != len(source) or frame.empty:
        raise ValueError("cost_learning_complete_mature_panel_required")
    times = pd.DatetimeIndex(frame.timestamp.unique()).sort_values()
    expected = pd.MultiIndex.from_product(
        [times, assets], names=["timestamp", "asset_id"]
    )
    indexed = frame.set_index(["timestamp", "asset_id"])
    if len(indexed) != len(expected) or not expected.isin(indexed.index).all():
        raise ValueError("cost_learning_fixed_universe_required")
    indexed = indexed.reindex(expected)
    ends = (
        pd.to_datetime(indexed[config["return_end_column"]])
        .to_numpy()
        .reshape(len(times), len(assets))
    )
    if (
        pd.isna(ends).any()
        or not (ends == ends[:, :1]).all()
        or not (ends[:, 0] > times.to_numpy()).all()
        or (len(times) > 1 and not (ends[:-1, 0] == times[1:].to_numpy()).all())
        or pd.Timestamp(ends[-1, 0]) > pd.Timestamp(config["cutoff"])
    ):
        raise ValueError("cost_learning_nonoverlapping_consecutive_returns_required")
    scaler = ModelMatrixScaler.fit(indexed[config["feature_columns"]], missing="impute")
    x = (
        scaler.transform(indexed[config["feature_columns"]])
        .to_numpy(dtype=np.float64)
        .reshape(len(times), len(assets), -1)
    )
    y = (
        indexed[config["label"]]
        .to_numpy(dtype=np.float64)
        .reshape(len(times), len(assets))
    )
    if (y <= -1).any():
        raise ValueError("cost_learning_return_outside_domain")
    output = Path(config["output"])
    reject_if_exists(output)
    recorder = TrialRecorder(
        output.with_name(output.name + ".trial"),
        {"kind": "cost_aware_allocator", "config": config},
    )
    recorder.begin()
    old_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(config["threads"])
        torch.manual_seed(config["seed"])
        device = get_device(config["device"])
        model = CostAwareAllocator(x.shape[-1], config["hidden_sizes"]).to(device)
        features = torch.tensor(x, dtype=torch.float32, device=device)
        returns = torch.tensor(y, dtype=torch.float32, device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
        for _ in range(config["epochs"]):
            optimizer.zero_grad()
            net, _ = differentiable_returns(
                model, features, returns, config["cost_bps"] / 10000
            )
            loss = -net.mean() + config["risk_aversion"] * net.var(unbiased=False)
            if not torch.isfinite(loss):
                raise ValueError("cost_learning_nonfinite_objective")
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            net, turnover = differentiable_returns(
                model, features, returns, config["cost_bps"] / 10000
            )
        facts = {
            "kind": "cost_aware_allocator",
            "device": str(device),
            "config": config,
            "periods": len(times),
            "train_mean_net_return": float(net.mean()),
            "train_mean_turnover": float(turnover.mean()),
            "training_execution": (
                "continuous_weights_linear_two_sided_cost_"
                "no_lot_or_capacity_constraints"
            ),
        }

        def write(staging):
            torch.save(model.cpu().state_dict(), staging / "weights.pt")
            (staging / "feature_scaler.pkl").write_bytes(pickle.dumps(scaler))
            atomic_write_json(staging / "model.json", facts)

        atomic_write_directory(output, write)
        recorder.finish(
            result={"periods": len(times), "objective": float(loss.detach())}
        )
    except BaseException as error:
        recorder.fail(error)
        raise
    finally:
        torch.set_num_threads(old_threads)
    return output


def backtest_cost_model(config):
    """Replay a frozen allocator through daily shares, fees, and execution
    constraints.
    """
    from quant_research_platform.research.policies.accounts import _episode, _stream

    config = _strict_config(config, {"model", "episode", "output", "device"}, set())
    root = Path(config["model"])
    facts = parse_qrp_json_object((root / "model.json").read_bytes())
    # Only load artifacts produced by trusted local training.
    scaler = pickle.loads((root / "feature_scaler.pkl").read_bytes())
    episode = _episode(config["episode"])
    training = facts["config"]
    if (
        facts["kind"] != "cost_aware_allocator"
        or episode["kind"] != "daily"
        or episode["task"] != "portfolio"
        or episode["feature_columns"] != training["feature_columns"]
        or episode["asset_ids"] != training["asset_ids"]
    ):
        raise ValueError("cost_model_account_mismatch")
    lower = pd.Timestamp(
        time_window_timestamp_interval(TimeWindow(**episode["window"]))[0]
    )
    if pd.Timestamp(training["cutoff"]) >= lower:
        raise ValueError("cost_model_backtest_not_causal")
    device = get_device(config["device"])
    model = CostAwareAllocator(
        len(training["feature_columns"]), training["hidden_sizes"]
    ).to(device)
    model.load_state_dict(
        torch.load(root / "weights.pt", map_location=device, weights_only=True)
    )
    model.eval()
    assets = training["asset_ids"]

    def produce(emit):
        with _stream(episode) as stream:
            try:
                event = next(stream)
                while True:
                    if isinstance(event, PolicyDecision):
                        x = scaler.transform(event.features.reindex(assets)).to_numpy(
                            dtype=float
                        )
                        if (
                            not np.isfinite(x).all()
                            or event.nav <= 0
                            or set(event.features.index) != set(assets)
                        ):
                            raise ValueError("cost_model_account_observation_invalid")
                        previous = (
                            event.quantities.reindex(assets, fill_value=0)
                            * event.marks.reindex(assets)
                        ).fillna(0).to_numpy() / event.nav
                        with torch.no_grad():
                            weights = (
                                model(
                                    torch.tensor(
                                        x,
                                        dtype=torch.float32,
                                        device=device,
                                    ),
                                    torch.as_tensor(
                                        previous,
                                        dtype=torch.float32,
                                        device=device,
                                    ),
                                )
                                .cpu()
                                .numpy()
                            )
                        weights = weights.astype(float)
                        # Float32 softmax may round a fully invested sum just above one.
                        weights /= max(1.0, weights.sum())
                        event = stream.send(pd.Series(weights, index=assets))
                    else:
                        emit(event)
                        event = next(stream)
            except StopIteration:
                pass

    return write_portfolio_ledger_artifact(
        Path(config["output"]),
        capital=PortfolioCapital(
            "cost_aware_allocator",
            episode["account"]["assumptions"]["initial_nav"],
            "CNY",
            "cash_equity",
        ),
        metadata={
            "kind": "a_share",
            "market_id": "a_share",
            "window": episode["window"],
            "model": config["model"],
            "training_approximation": facts["training_execution"],
        },
        produce=produce,
    )

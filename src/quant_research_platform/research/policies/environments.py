"""Expose native trading accounts as Gymnasium decision environments.

Observations contain current features, holdings, cash, and time.
Actions request inventory or portfolio targets; fills, fees, and tradeability
remain the account engine's responsibility. Rewards use its realized ledger.
"""

from math import fsum

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from quant_research_platform.contracts.policies import PolicyDecision
from quant_research_platform.contracts.time import (
    TimeWindow,
    time_window_timestamp_interval,
)
from quant_research_platform.research.policies.accounts import _episode, _stream


class TradingEnv(gym.Env):
    """One action resumes the real ledger until its next causal decision point."""

    metadata = {"render_modes": []}

    def __init__(self, episodes, *, on_batch=None, feature_scaler=None):
        self.episodes = [_episode(value) for value in episodes]
        if not self.episodes:
            raise ValueError("policy_episodes_required")
        first = self.episodes[0]
        identity = (
            first["kind"],
            first["task"],
            first["asset_ids"],
            first["feature_columns"],
        )
        if any(
            (item["kind"], item["task"], item["asset_ids"], item["feature_columns"])
            != identity
            for item in self.episodes
        ):
            raise ValueError("policy_episode_spaces_mismatch")
        self.assets, self.columns = first["asset_ids"], first["feature_columns"]
        self.task = first["task"]
        n_actions = (
            1
            if self.task == "execution"
            else len(self.assets) + (self.task == "portfolio")
        )
        self.action_space = spaces.Box(0.0, 1.0, (n_actions,), dtype=np.float32)
        self.observation_space = spaces.Box(
            -np.inf,
            np.inf,
            (len(self.assets) * (len(self.columns) + 2) + 3,),
            dtype=np.float32,
        )
        self.on_batch = on_batch
        self.feature_scaler = feature_scaler
        self._context = None
        self.done = True

    def close(self):
        if self._context is not None:
            context, self._context = self._context, None
            context.__exit__(None, None, None)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options:
            raise ValueError("policy_reset_options_unused")
        self.close()
        self.episode = self.episodes[int(self.np_random.integers(len(self.episodes)))]
        self._context = _stream(self.episode)
        self.stream = self._context.__enter__()
        self.fees = self.filled = self.shortfall = 0.0
        self.arrival_price = None
        self.observed_marks = {}
        self.last_block_reasons = {}
        self.nav = None
        self.done = False
        self.decision = None
        self._advance(None)
        if self.done:
            if "prediction" not in self.episode:
                raise ValueError("policy_episode_no_decisions")
            self.initial_nav = self.nav
            return self._observation(), self._info()
        self.initial_nav = self.decision.nav
        self.nav = self.initial_nav
        if self.task == "execution":
            order = self.episode["order"]
            self.arrival_price = float(self.decision.marks[order["asset_id"]])
            if not np.isfinite(self.arrival_price) or self.arrival_price <= 0:
                raise ValueError("policy_arrival_price_missing")
            initial = float(self.decision.quantities[order["asset_id"]])
            maximum = self.episode["account"]["parameters"]["maximum_inventory"][
                order["asset_id"]
            ]
            if not 0 <= initial + order["quantity"] <= maximum:
                raise ValueError("policy_order_outside_inventory")
        return self._observation(), self._info()

    def _advance(self, target):
        try:
            event = next(self.stream) if target is None else self.stream.send(target)
            while not isinstance(event, PolicyDecision):
                if self.on_batch is not None:
                    self.on_batch(event)
                ends = [row for row in event.valuations if row.phase == "end"]
                self.nav = fsum(row.value for row in ends)
                self.fees += -fsum(
                    row.amount
                    for row in event.cashflows
                    if row.kind not in {"principal", "dividend", "recovery"}
                )
                if self.task == "execution":
                    end_marks = {row.asset_id: row.mark_price for row in ends}
                    for diagnostic in event.diagnostics:
                        if (
                            diagnostic.name == "mark_source"
                            and diagnostic.text == "observed_mid"
                        ):
                            self.observed_marks[diagnostic.asset_id] = end_marks[
                                diagnostic.asset_id
                            ]
                        elif diagnostic.name == "order_block_reason":
                            self.last_block_reasons[diagnostic.asset_id] = (
                                diagnostic.text
                            )
                if self.task == "execution" and self.arrival_price is not None:
                    order = self.episode["order"]
                    sign = np.sign(order["quantity"])
                    for trade in event.trades:
                        if (
                            trade.asset_id != order["asset_id"]
                            or trade.quantity * sign <= 0
                        ):
                            raise ValueError("policy_execution_off_mandate")
                        self.filled += abs(trade.quantity)
                        self.shortfall += (
                            trade.notional - trade.quantity * self.arrival_price
                        )
                event = next(self.stream)
            if set(event.features.index) - set(self.assets):
                raise ValueError("policy_observed_asset_outside_universe")
            self.decision = event
            self.nav = event.nav
        except StopIteration:
            self.done = True
            self.close()

    def _observation(self):
        if self.done:
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        decision = self.decision
        features = decision.features.reindex(self.assets)[self.columns]
        if self.feature_scaler is not None:
            features = self.feature_scaler.transform(features)
        values = features.to_numpy(dtype=np.float64)
        if np.isinf(values).any():
            raise ValueError("policy_feature_infinite")
        denominator = max(abs(decision.nav), 1.0)
        holding = decision.quantities.reindex(
            self.assets, fill_value=0
        ) * decision.marks.reindex(self.assets)
        sellable = decision.sellable.reindex(
            self.assets, fill_value=0
        ) * decision.marks.reindex(self.assets)
        if (
            holding.where(
                decision.quantities.reindex(self.assets, fill_value=0).ne(0), 0
            )
            .isna()
            .any()
        ):
            raise ValueError("policy_held_mark_missing")
        lower, upper, _ = time_window_timestamp_interval(
            TimeWindow(**self.episode["window"])
        )
        elapsed = float(
            (decision.timestamp - pd.Timestamp(lower))
            / (pd.Timestamp(upper) - pd.Timestamp(lower))
        )
        remaining = (
            1 - self.filled / abs(self.episode["order"]["quantity"])
            if self.task == "execution"
            else 0.0
        )
        result = np.concatenate(
            (
                values.ravel(),
                holding.fillna(0) / denominator,
                sellable.fillna(0) / denominator,
                [decision.cash / denominator, 1 - elapsed, remaining],
            )
        ).astype(np.float32)
        if np.isinf(result).any() or not np.isfinite(result[values.size :]).all():
            raise ValueError("policy_observation_nonfinite")
        return result

    def _info(self):
        result = {"nav": self.nav, "fees": self.fees}
        if self.task == "execution" and self.arrival_price is not None:
            order = self.episode["order"]
            quantity = abs(order["quantity"])
            remaining = max(0.0, quantity - self.filled)
            terminal_mark = self.observed_marks.get(order["asset_id"])
            opportunity = None
            if self.done:
                if remaining == 0:
                    opportunity = 0.0
                elif (
                    terminal_mark is not None
                    and np.isfinite(terminal_mark)
                    and terminal_mark > 0
                ):
                    opportunity = float(
                        np.sign(order["quantity"])
                        * remaining
                        * (terminal_mark - self.arrival_price)
                        / (quantity * self.arrival_price)
                        * 10000
                    )
            shortfall = (
                (self.shortfall + self.fees) / (quantity * self.arrival_price) * 10000
            )
            result.update(
                completion_rate=self.filled / quantity,
                filled_quantity=self.filled,
                implementation_shortfall_bps=shortfall,
                unfilled_quantity=remaining,
                unfilled_reason=(
                    None
                    if remaining == 0
                    else "execution_window_ended"
                    if self.done
                    else "pending"
                ),
                last_block_reason=self.last_block_reasons.get(order["asset_id"]),
                arrival_price=self.arrival_price,
                terminal_observed_mark=terminal_mark if self.done else None,
                opportunity_cost_bps=opportunity,
                opportunity_cost_reason=(
                    "episode_in_progress"
                    if not self.done
                    else "terminal_observed_mark_missing"
                    if opportunity is None
                    else None
                ),
                total_implementation_shortfall_bps=(
                    shortfall + opportunity if opportunity is not None else None
                ),
            )
        return result

    def step(self, action):
        if self.done:
            raise RuntimeError("policy_episode_finished")
        action = np.asarray(action, dtype=np.float64)
        if (
            action.shape != self.action_space.shape
            or not np.isfinite(action).all()
            or (action < 0).any()
            or (action > 1).any()
        ):
            raise ValueError("policy_action_invalid")
        decision = self.decision
        if self.task == "execution":
            order = self.episode["order"]
            remaining = max(0, abs(order["quantity"]) - self.filled)
            target = decision.quantities.copy()
            target[order["asset_id"]] += np.sign(order["quantity"]) * np.floor(
                action[0] * remaining
            )
        elif self.task == "inventory":
            maximum = pd.Series(
                self.episode["account"]["parameters"]["maximum_inventory"]
            )
            target = maximum.reindex(self.assets) * action
        else:
            weights = action[:-1] / max(float(action.sum()), 1.0e-30)
            target = pd.Series(weights, index=self.assets)
            if self.episode["kind"] == "hft":
                target = (
                    target * max(decision.nav, 0) / decision.marks.reindex(self.assets)
                )
                maximum = pd.Series(
                    self.episode["account"]["parameters"]["maximum_inventory"]
                )
                target = target.clip(upper=maximum).fillna(0)
        return self.step_targets(target)

    def step_targets(self, target):
        """Submit native targets; missing forecasts do not change existing orders."""
        if self.done:
            raise RuntimeError("policy_episode_finished")
        decision = self.decision
        before_nav, before_shortfall, before_fees = self.nav, self.shortfall, self.fees
        if self.task == "execution":
            order = self.episode["order"]
            remaining = max(0, abs(order["quantity"]) - self.filled)
        target = target.reindex(decision.features.index, fill_value=0)
        self._advance(target)
        if self.task == "execution":
            notional = abs(self.episode["order"]["quantity"]) * self.arrival_price
            reward = (
                -(self.shortfall - before_shortfall + self.fees - before_fees)
                / notional
            )
            lower, upper, _ = time_window_timestamp_interval(
                TimeWindow(**self.episode["window"])
            )
            end = pd.Timestamp(upper) if self.done else self.decision.timestamp
            elapsed = (end - decision.timestamp) / (
                pd.Timestamp(upper) - pd.Timestamp(lower)
            )
            reward -= (
                self.episode["risk_aversion"]
                * (remaining / abs(order["quantity"])) ** 2
                * elapsed
            )
            if self.done:
                reward -= (
                    self.episode["unfinished_penalty_bps"]
                    / 10000
                    * (1 - self._info()["completion_rate"])
                )
        else:
            change = (self.nav - before_nav) / self.initial_nav
            reward = change - self.episode["risk_aversion"] * change * change
        return self._observation(), float(reward), self.done, False, self._info()

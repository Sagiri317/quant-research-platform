"""Expose native trading accounts as Gymnasium decision environments.

Observations contain current features, holdings, cash, and time.
Actions request inventory or portfolio targets; fills, fees, and tradeability
remain the account engine's responsibility. Rewards use its realized ledger.
"""

from datetime import time
from math import fsum

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from quant_research_platform.contracts.policies import PolicyDecision, PolicyExecution
from quant_research_platform.contracts.time import (
    TimeWindow,
    time_window_timestamp_interval,
)
from quant_research_platform.research.policies.accounts import _episode, _stream
from quant_research_platform.research.policies.inventory_policy import inventory_targets


def _trading_seconds(start, end):
    """Integrate the two continuous trading segments, excluding lunch and overnight."""
    seconds = 0.0
    for day in dict.fromkeys((start.normalize(), end.normalize())):
        for lower, upper in ((1.5, 3.5), (5, 7)):
            seconds += max(
                0.0,
                (
                    min(end, day + pd.Timedelta(hours=upper))
                    - max(start, day + pd.Timedelta(hours=lower))
                ).total_seconds(),
            )
    return seconds


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
        self.action_space = (
            spaces.MultiDiscrete([8] * len(self.assets))
            if self.task == "dynamic_execution"
            else spaces.Box(0.0, 1.0, (n_actions,), dtype=np.float32)
        )
        self.observation_space = spaces.Box(
            -np.inf,
            np.inf,
            (
                len(self.assets)
                * (len(self.columns) + (11 if self.task == "dynamic_execution" else 2))
                + (5 if self.task == "dynamic_execution" else 3),
            ),
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
        self._last_valuations = None
        self.target = None
        self.reward_total = self.economic_reward_total = 0.0
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
                if self.task == "dynamic_execution" and self.decision is not None:
                    before = [row for row in event.valuations if row.phase == "before"]
                    self._integrate_exposure(before, pd.Timestamp(ends[0].timestamp_ns))
                self._last_valuations = ends
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
            if self.task == "dynamic_execution" and self.decision is not None:
                self._integrate_exposure(self._last_valuations, event.timestamp)
            self.decision = event
            self.nav = event.nav
            if self.task == "dynamic_execution":
                self.target = inventory_targets(
                    event,
                    self.episode["account"]["parameters"]["maximum_inventory"],
                    self.episode["inventory_parameters"],
                )
                restore_at = self.episode["account"]["parameters"]["restore_at"]
                if restore_at is not None and (
                    event.timestamp + pd.Timedelta(hours=8)
                ).time() >= time.fromisoformat(restore_at):
                    self.target = event.opening_quantities.clip(
                        upper=pd.Series(
                            self.episode["account"]["parameters"]["maximum_inventory"]
                        )
                    )
        except StopIteration:
            if self.task == "dynamic_execution" and self.decision is not None:
                _, end, _ = time_window_timestamp_interval(
                    TimeWindow(**self.episode["window"])
                )
                session_end = pd.Timestamp(
                    self._last_valuations[0].timestamp_ns
                ).normalize() + pd.Timedelta(hours=7)
                self._integrate_exposure(
                    self._last_valuations, min(pd.Timestamp(end), session_end)
                )
            self.done = True
            self.close()

    def _integrate_exposure(self, rows, end):
        start = max(pd.Timestamp(rows[0].timestamp_ns), self.decision.timestamp)
        elapsed = _trading_seconds(start, end) / 14400
        for row in rows:
            if row.asset_id not in self.assets:
                continue
            holding = row.quantity * row.mark_price / self.initial_nav
            self._inventory_integral += holding * holding * elapsed
            target = self.target[row.asset_id]
            if pd.notna(target):
                deviation = (row.quantity - target) * row.mark_price / self.initial_nav
                self._deviation_integral += deviation * deviation * elapsed

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
        if self.task == "dynamic_execution":
            maximum = pd.Series(
                self.episode["account"]["parameters"]["maximum_inventory"]
            ).reindex(self.assets)
            state = decision.order_state.reindex(self.assets)
            marks = decision.marks.reindex(self.assets)
            result = np.r_[
                result,
                (
                    self.target.fillna(decision.quantities).reindex(self.assets)
                    * marks
                    / denominator
                ).fillna(0),
                (
                    decision.pending_quantities.reindex(self.assets)
                    * marks
                    / denominator
                ).fillna(0),
                (state.quantity * marks / denominator).fillna(0),
                state.queue_ahead.divide(maximum.where(maximum > 0)).fillna(0),
                state.age_seconds / 14400,
                state.cancel_seconds,
                (
                    decision.pending_targets.reindex(self.assets).fillna(
                        decision.quantities
                    )
                    * marks
                    / denominator
                ).fillna(0),
                state["mode"],
                state.replacement_mode,
                decision.reserved_cash / denominator,
                (
                    (decision.timestamp + pd.Timedelta(hours=8)).hour * 3600
                    + decision.timestamp.minute * 60
                    + decision.timestamp.second
                )
                / 86400,
            ].astype(np.float32)
        if np.isinf(result).any() or not np.isfinite(result[values.size :]).all():
            raise ValueError("policy_observation_nonfinite")
        return result

    def _info(self):
        result = {"nav": self.nav, "fees": self.fees}
        if self.task == "dynamic_execution":
            result.update(
                reward_bps=self.reward_total,
                economic_reward_bps=self.economic_reward_total,
            )
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
        if self.task == "dynamic_execution":
            if (
                action.shape != self.action_space.shape
                or not np.isfinite(action).all()
                or not np.equal(action, np.floor(action)).all()
                or not self.action_space.contains(action.astype(np.int64))
            ):
                raise ValueError("policy_action_invalid")
            action = action.astype(np.int64)
            modes = pd.Series(
                np.array(
                    [
                        "wait",
                        "cancel",
                        "maker",
                        "maker",
                        "maker",
                        "taker",
                        "taker",
                        "taker",
                    ]
                )[action],
                index=self.assets,
            )
            modes.loc[self.target.reindex(self.assets).isna() & modes.ne("cancel")] = (
                "wait"
            )
            quantities = self.decision.quantities.reindex(self.assets)
            fraction = np.array([0, 0, 0.25, 0.5, 1, 0.25, 0.5, 1])[action]
            target = (
                quantities
                + (self.target.reindex(self.assets).fillna(quantities) - quantities)
                * fraction
            )
            return self.step_targets(PolicyExecution(target, modes, self.target))
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
        self._inventory_integral = self._deviation_integral = 0.0
        if self.task == "execution":
            order = self.episode["order"]
            remaining = max(0, abs(order["quantity"]) - self.filled)
        if self.task != "dynamic_execution":
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
        elif self.task == "dynamic_execution":
            economic = float(10000 * (self.nav - before_nav) / self.initial_nav)
            reward = (
                economic
                - self.episode["risk_aversion"] * self._inventory_integral
                - self.episode["target_deviation_bps"] * self._deviation_integral
            )
            self.economic_reward_total += economic
            self.reward_total += float(reward)
        else:
            change = (self.nav - before_nav) / self.initial_nav
            reward = change - self.episode["risk_aversion"] * change * change
        return self._observation(), float(reward), self.done, False, self._info()

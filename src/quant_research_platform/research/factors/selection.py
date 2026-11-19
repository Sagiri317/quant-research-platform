"""Select distinct exposures using observable historical training sessions."""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


def cluster_exposures(days, columns, *, axis, threshold, preferred=()):
    """Complete-link clusters require evidence for every pair, including sign flips."""
    columns = list(columns)
    coverages, movements, observations = [], [], []
    first, varying = np.full(len(columns), np.nan), np.zeros(len(columns), bool)
    previous, previous_week = None, None
    for day in days:
        if day.empty:
            continue
        day = day.sort_values(["asset_id", "timestamp"], kind="stable")
        values = day[columns]
        minimum, maximum = values.min().to_numpy(float), values.max().to_numpy(float)
        first = np.where(np.isnan(first), minimum, first)
        varying |= np.isfinite(maximum) & (
            (minimum != maximum)
            if axis == "cross_sectional"
            else ((minimum != first) | (maximum != first))
        )
        coverages.append(values.notna().mean().to_numpy())
        ranks = values.groupby(
            day.timestamp if axis == "cross_sectional" else day.asset_id, sort=False
        ).rank(pct=True)
        if axis == "cross_sectional":
            ranks.index = pd.Index(day.asset_id)
            if previous is not None:
                movements.append((ranks - previous).abs().mean().to_numpy())
            previous = ranks
        else:
            movement = ranks.groupby(day.asset_id, sort=False).diff()
            if "segment_id" in day:
                movement = movement.where(
                    day.segment_id.eq(day.segment_id.shift()), axis=0
                )
            movements.append(movement.abs().mean().to_numpy())
        # Stratify across the full ordered session; equal-sized days get equal weight.
        week = day.timestamp.iloc[0].to_period("W")
        if week == previous_week:
            continue
        previous_week = week
        positions = np.linspace(0, len(ranks) - 1, min(128, len(ranks)), dtype=int)
        sampled = ranks if axis == "cross_sectional" else values
        observations.append(
            sampled.iloc[positions].to_numpy(dtype=float, na_value=np.nan)
        )
    if not varying.any():
        raise ValueError("rolling_no_varying_inputs")
    names = np.asarray(columns)[varying]
    coverage = pd.DataFrame(coverages, columns=columns).quantile(0.1)
    turnover = pd.DataFrame(movements, columns=columns).mean().fillna(np.inf)
    joint_weeks = np.zeros((len(names), len(names)), dtype=int)
    for sample in observations:
        observed = np.isfinite(sample[:, varying]).astype(np.float32)
        joint_weeks += (observed.T @ observed) > 0
    samples = np.concatenate(observations)[:, varying]
    if axis == "single_asset_time_series":
        samples = pd.DataFrame(samples).rank(pct=True).to_numpy()
    # Pairwise Pearson on session ranks, allowing different missingness per factor.
    seen = np.isfinite(samples).astype(float)
    values = np.nan_to_num(samples, nan=0.0)
    count = seen.T @ seen
    sums = values.T @ seen
    cross = values.T @ values - sums * sums.T / np.maximum(count, 1)
    squares = (values**2).T @ seen - sums**2 / np.maximum(count, 1)
    scale = np.sqrt(np.maximum(squares, 0) * np.maximum(squares.T, 0))
    smaller_sample = np.minimum.outer(count.diagonal(), count.diagonal())
    enough = (count >= 24) & (count >= 0.8 * smaller_sample) & (joint_weeks >= 3)
    correlations = np.divide(
        cross, scale, out=np.zeros_like(cross), where=enough & (scale > 1e-12)
    )
    distance = 1 - np.clip(np.abs(correlations), 0, 1)
    np.fill_diagonal(distance, 0)
    labels = (
        fcluster(
            linkage(squareform(distance, checks=False), method="complete"),
            t=1 - threshold,
            criterion="distance",
        )
        if len(names) > 1
        else [1]
    )
    groups = {}
    for name, group in zip(names, labels, strict=True):
        groups.setdefault(group, []).append(str(name))
    preferred = set(preferred)
    order = {name: index for index, name in enumerate(columns)}
    retained, duplicates = set(), {}
    for members in groups.values():
        # Preserve a representative when its lower-decile coverage is comparable.
        best_coverage = max(coverage[c] for c in members)
        stable = [c for c in members if coverage[c] >= best_coverage - 0.02]
        winner = min(
            stable,
            key=lambda c: (c not in preferred, turnover[c], -coverage[c], order[c]),
        )
        retained.add(winner)
        if len(members) > 1:
            duplicates[winner] = [c for c in members if c != winner]
    return {
        "retained": [c for c in columns if c in retained],
        "all_missing": [
            c for c, value in zip(columns, first, strict=True) if np.isnan(value)
        ],
        "constant": [
            c
            for c, value, varies in zip(columns, first, varying, strict=True)
            if np.isfinite(value) and not varies
        ],
        "clusters": duplicates,
        "exposure_sessions": len(coverages),
        "absolute_rank_correlation": threshold,
    }

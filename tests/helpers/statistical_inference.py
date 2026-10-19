"""Direct-definition references for SPA, StepM, and the MCS R method."""

import numpy as np
import pandas as pd
from arch.bootstrap import StationaryBootstrap


def hansen_spa_stepm_reference(
    benchmark, models, *, block_size, repetitions, seed, alpha=0.05
):
    differences = np.asarray(benchmark)[:, None] - models.to_numpy()
    n, k = differences.shape
    means = differences.mean(axis=0)
    variance = []
    for column in range(k):
        centered = differences[:, column] - means[column]
        covariances = [
            sum(centered[t] * centered[t + lag] for t in range(n - lag)) / n
            for lag in range(n)
        ]
        variance.append(
            covariances[0]
            + 2
            * sum(
                (
                    (1 - lag / n) * (1 - 1 / block_size) ** lag
                    + lag / n * (1 - 1 / block_size) ** (n - lag)
                )
                * covariances[lag]
                for lag in range(1, n)
            )
        )
    standard_errors = np.sqrt(np.asarray(variance) / n)
    observed = means / standard_errors
    centers = [
        np.maximum(means, 0),
        np.where(observed >= -np.sqrt(2 * np.log(np.log(n))), means, 0),
        means,
    ]
    draws = []
    for positional, _ in StationaryBootstrap(
        block_size, np.arange(n), seed=seed
    ).bootstrap(repetitions):
        sample = differences[positional[0]]
        draw_mean = np.asarray([sum(sample[:, column]) / n for column in range(k)])
        draws.append([(draw_mean - center) / standard_errors for center in centers])
    simulated = np.asarray(draws)
    observed_max = max(0, max(observed))
    p_values = np.asarray(
        [
            sum(max(0, max(draw[variant])) > observed_max for draw in simulated)
            / repetitions
            for variant in range(3)
        ]
        if observed_max > 0
        else [1.0, 1.0, 1.0]
    )
    remaining = list(range(k))
    while remaining:
        maxima = [max(0, max(draw[1, remaining])) for draw in simulated]
        critical = np.quantile(maxima, 1 - alpha)
        retained = [column for column in remaining if observed[column] <= critical]
        if retained == remaining:
            break
        remaining = retained
    return p_values, tuple(
        name for column, name in enumerate(models.columns) if column not in remaining
    )


def mcs_r_reference(losses, *, block_size, repetitions, seed, alpha=0.05):
    """Evaluate ordered pairs and bootstrap draws separately from the R kernel."""
    values = losses.to_numpy()
    n, k = values.shape
    means = [sum(values[:, i]) / n for i in range(k)]
    draw_means = []
    for positional, _ in StationaryBootstrap(
        block_size, np.arange(n), seed=seed
    ).bootstrap(repetitions):
        sample = values[positional[0]]
        draw_means.append([sum(sample[:, i]) / n for i in range(k)])
    observed, simulated = {}, {}
    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            difference = means[i] - means[j]
            centered = [draw[i] - draw[j] - difference for draw in draw_means]
            standard_error = np.sqrt(sum(value**2 for value in centered) / repetitions)
            observed[i, j] = difference / standard_error
            simulated[i, j] = [value / standard_error for value in centered]
    remaining, eliminated = list(range(k)), []
    running_p = 0.0
    while len(remaining) > 1:
        pairs = [(i, j) for i in remaining for j in remaining if i != j]
        statistic = max(abs(observed[pair]) for pair in pairs)
        tail_count = sum(
            max(abs(simulated[pair][draw]) for pair in pairs) >= statistic
            for draw in range(repetitions)
        )
        running_p = max(running_p, tail_count / repetitions)
        worst = max(
            remaining,
            key=lambda i: max(observed[i, j] for j in remaining if j != i),
        )
        eliminated.append((losses.columns[worst], running_p))
        remaining.remove(worst)
    eliminated.append((losses.columns[remaining[0]], 1.0))
    return (
        pd.DataFrame(
            {"Pvalue": [p for _, p in eliminated]},
            index=pd.Index([name for name, _ in eliminated], name="Model name"),
        ),
        tuple(sorted(name for name, p in eliminated if p >= alpha)),
        tuple(sorted(name for name, p in eliminated if p < alpha)),
    )

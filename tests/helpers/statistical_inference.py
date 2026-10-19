"""Direct lag-sum references for the Hansen SPA and stepwise tests."""

import numpy as np
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

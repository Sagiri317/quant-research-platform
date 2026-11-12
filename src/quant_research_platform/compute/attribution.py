"""Numerical additivity and tolerances for frozen-model attributions."""

from math import fsum

import numpy as np


def attribution_relative_tolerance(method, feature_count):
    """Tree backends accumulate float32 paths; affine paths use float64."""
    return float(
        5e-5
        if method in {"native_tree", "tree_shap_path", "tree_shap_interventional"}
        else 64 * np.finfo(float).eps * max(1, feature_count)
    )


def attribution_additivity(base, contributions, output, relative):
    """Require both backward stability and useful precision at the output scale."""
    terms = [float(base), *map(float, contributions)]
    if not all(np.isfinite(terms)):
        return np.nan, np.nan, "nonrepresentable_contribution_or_baseline"
    magnitude = max(abs(output), *map(abs, terms))
    floor = np.nextafter(0.0, 1.0) * max(1, len(contributions))
    backward = (
        (relative * magnitude) * max(1.0, fsum(abs(term) / magnitude for term in terms))
        if magnitude
        else 0.0
    )
    backward += floor
    tolerance = min(
        backward, max(relative, np.sqrt(np.finfo(float).eps)) * abs(output) + floor
    )
    try:
        residual = fsum([*terms, -float(output)])
    except OverflowError:
        return np.nan, np.nan, "nonrepresentable_additivity"
    if not np.isfinite(residual) or not np.isfinite(tolerance):
        return np.nan, np.nan, "nonrepresentable_additivity"
    if abs(residual) > tolerance:
        reason = (
            "cancellation_precision_loss"
            if abs(residual) <= backward
            else "additivity_mismatch"
        )
        return np.nan, np.nan, reason
    return residual, tolerance, ""

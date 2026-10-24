"""Wasserstein-1 distance (Earth Mover's Distance) for continuous distributions."""

from __future__ import annotations

import math
from typing import Sequence, Tuple

from driftpulse.core.exceptions import StatisticalTestError
from driftpulse.core.types import DriftStatus, MetricResult, MetricType


def wasserstein_distance_1d(
    sample_ref: Sequence[float],
    sample_curr: Sequence[float],
) -> float:
    """Computes the 1D Wasserstein-1 (Earth Mover's) distance between two empirical distributions.

    Equal to the L1 norm between the two cumulative distribution functions:
    W1 = integral |F_ref(x) - F_curr(x)| dx.
    """
    clean_ref = sorted([float(x) for x in sample_ref if x is not None and not math.isnan(float(x))])
    clean_curr = sorted([float(x) for x in sample_curr if x is not None and not math.isnan(float(x))])

    n1 = len(clean_ref)
    n2 = len(clean_curr)

    if n1 == 0 or n2 == 0:
        raise StatisticalTestError("Both samples must contain at least 1 valid observation.")

    # Combine all distinct points
    all_values = sorted(list(set(clean_ref + clean_curr)))
    if len(all_values) <= 1:
        return 0.0

    # Step-wise integration of |CDF1 - CDF2|
    i1, i2 = 0, 0
    w_dist = 0.0

    for idx in range(len(all_values) - 1):
        x_left = all_values[idx]
        x_right = all_values[idx + 1]
        interval_width = x_right - x_left

        # Advance pointer 1 up to x_left
        while i1 < n1 and clean_ref[i1] <= x_left:
            i1 += 1
        cdf1 = i1 / n1

        # Advance pointer 2 up to x_left
        while i2 < n2 and clean_curr[i2] <= x_left:
            i2 += 1
        cdf2 = i2 / n2

        w_dist += abs(cdf1 - cdf2) * interval_width

    return round(w_dist, 6)


def evaluate_wasserstein_drift(
    sample_ref: Sequence[float],
    sample_curr: Sequence[float],
    threshold: float = 0.15,
    normalize: bool = True,
) -> MetricResult:
    """Evaluates Wasserstein drift, optionally normalized by reference standard deviation."""
    raw_dist = wasserstein_distance_1d(sample_ref, sample_curr)

    metric_val = raw_dist
    if normalize:
        clean_ref = [float(x) for x in sample_ref if x is not None and not math.isnan(float(x))]
        mean_r = sum(clean_ref) / len(clean_ref)
        var_r = sum((x - mean_r) ** 2 for x in clean_ref) / max(len(clean_ref) - 1, 1)
        std_r = math.sqrt(var_r)
        if std_r > 1e-6:
            metric_val = raw_dist / std_r

    has_drift = metric_val >= threshold
    if metric_val >= threshold * 2.0:
        status = DriftStatus.CRITICAL_DRIFT
    elif metric_val >= threshold:
        status = DriftStatus.MODERATE_DRIFT
    elif metric_val >= threshold * 0.5:
        status = DriftStatus.LOW_DRIFT
    else:
        status = DriftStatus.NO_DRIFT

    return MetricResult(
        metric_type=MetricType.WASSERSTEIN,
        value=metric_val,
        threshold=threshold,
        has_drift=has_drift,
        status=status,
        details={
            "raw_wasserstein": raw_dist,
            "normalized": normalize,
            "interpretation": f"Normalized Earth Mover's Distance is {round(metric_val, 4)} (threshold: {threshold})",
        },
    )

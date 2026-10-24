"""Jensen-Shannon Divergence and symmetric KL distance for probability distributions."""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

from driftpulse.core.exceptions import StatisticalTestError
from driftpulse.core.types import DriftStatus, MetricResult, MetricType


def kl_divergence(p: Sequence[float], q: Sequence[float], epsilon: float = 1e-9) -> float:
    """Computes Kullback-Leibler divergence D_KL(P || Q) with base 2."""
    div = 0.0
    for pi, qi in zip(p, q):
        p_smooth = max(pi, epsilon)
        q_smooth = max(qi, epsilon)
        div += p_smooth * math.log2(p_smooth / q_smooth)
    return max(0.0, div)


def jensen_shannon_divergence(
    dist_p: Sequence[float],
    dist_q: Sequence[float],
    epsilon: float = 1e-9,
) -> float:
    """Computes Jensen-Shannon Divergence (JSD) bounded between 0.0 and 1.0 (base 2)."""
    if len(dist_p) != len(dist_q) or len(dist_p) == 0:
        raise StatisticalTestError("Probability vectors must be non-empty and of identical length.")

    # Normalize vectors to sum to 1.0
    sum_p = sum(dist_p)
    sum_q = sum(dist_q)
    if sum_p <= 0 or sum_q <= 0:
        raise StatisticalTestError("Distribution sums must be strictly positive.")

    p_norm = [v / sum_p for v in dist_p]
    q_norm = [v / sum_q for v in dist_q]

    # Mixture distribution M = (P + Q) / 2
    m = [(pi + qi) / 2.0 for pi, qi in zip(p_norm, q_norm)]

    jsd = 0.5 * kl_divergence(p_norm, m, epsilon) + 0.5 * kl_divergence(q_norm, m, epsilon)
    return round(max(0.0, min(1.0, jsd)), 6)


def evaluate_jensen_shannon_drift(
    dist_ref: Sequence[float],
    dist_curr: Sequence[float],
    threshold: float = 0.10,
) -> MetricResult:
    """Evaluates distribution divergence using the Jensen-Shannon Distance (sqrt(JSD))."""
    jsd = jensen_shannon_divergence(dist_ref, dist_curr)
    js_distance = math.sqrt(jsd)  # True mathematical metric

    has_drift = js_distance >= threshold
    if js_distance >= threshold * 2.0:
        status = DriftStatus.CRITICAL_DRIFT
    elif js_distance >= threshold:
        status = DriftStatus.MODERATE_DRIFT
    elif js_distance >= threshold * 0.5:
        status = DriftStatus.LOW_DRIFT
    else:
        status = DriftStatus.NO_DRIFT

    return MetricResult(
        metric_type=MetricType.JENSEN_SHANNON,
        value=js_distance,
        threshold=threshold,
        has_drift=has_drift,
        status=status,
        details={
            "js_divergence": jsd,
            "js_distance": round(js_distance, 6),
            "interpretation": f"JS distance is {round(js_distance, 4)} (bounded [0, 1])",
        },
    )

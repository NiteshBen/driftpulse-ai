"""Two-sample Kolmogorov-Smirnov (KS) test for numerical distribution drift.

Compares empirical cumulative distribution functions (eCDF) to determine
whether baseline and production samples originate from the same continuous distribution.
"""

from __future__ import annotations

import bisect
import math
from typing import Sequence, Tuple

from driftpulse.core.exceptions import StatisticalTestError
from driftpulse.core.types import DriftStatus, MetricResult, MetricType


def kolmogorov_smirnov_2sample(
    sample_ref: Sequence[float],
    sample_curr: Sequence[float],
) -> Tuple[float, float]:
    """Computes the 2-sample Kolmogorov-Smirnov statistic D and asymptotic p-value.

    Parameters:
        sample_ref: Reference / baseline numerical sample.
        sample_curr: Production / current numerical sample.

    Returns:
        Tuple of (ks_statistic, p_value).
    """
    clean_ref = sorted([float(x) for x in sample_ref if x is not None and not math.isnan(float(x))])
    clean_curr = sorted([float(x) for x in sample_curr if x is not None and not math.isnan(float(x))])

    n1 = len(clean_ref)
    n2 = len(clean_curr)

    if n1 == 0 or n2 == 0:
        raise StatisticalTestError("Both samples must contain at least 1 valid numeric observation.")

    # All unique values across both samples
    unique_vals = sorted(list(set(clean_ref + clean_curr)))
    d_max = 0.0

    for val in unique_vals:
        cdf1 = bisect.bisect_right(clean_ref, val) / n1
        cdf2 = bisect.bisect_right(clean_curr, val) / n2
        diff = abs(cdf1 - cdf2)
        if diff > d_max:
            d_max = diff

    # If samples are completely identical, D is exactly 0.0
    if d_max == 0.0:
        return 0.0, 1.0

    # Effective sample size
    en = math.sqrt((n1 * n2) / (n1 + n2))
    lambda_val = (en + 0.12 + 0.11 / en) * d_max

    # Asymptotic approximation of Kolmogorov distribution survival function
    if lambda_val <= 0:
        p_val = 1.0
    elif lambda_val > 5.0:
        p_val = 0.0
    else:
        # Sum alternating series
        term_sum = 0.0
        for k in range(1, 101):
            term = 2.0 * ((-1) ** (k - 1)) * math.exp(-2.0 * (k ** 2) * (lambda_val ** 2))
            term_sum += term
            if abs(term) < 1e-12:
                break
        p_val = max(0.0, min(1.0, term_sum))

    return round(d_max, 6), round(p_val, 6)


def evaluate_ks_drift(
    sample_ref: Sequence[float],
    sample_curr: Sequence[float],
    alpha: float = 0.05,
) -> MetricResult:
    """Evaluates whether statistical drift has occurred via the 2-sample KS test."""
    stat, p_val = kolmogorov_smirnov_2sample(sample_ref, sample_curr)
    has_drift = p_val < alpha

    if p_val < alpha / 5.0:
        status = DriftStatus.CRITICAL_DRIFT
    elif p_val < alpha:
        status = DriftStatus.MODERATE_DRIFT
    elif p_val < alpha * 2.0:
        status = DriftStatus.LOW_DRIFT
    else:
        status = DriftStatus.NO_DRIFT

    return MetricResult(
        metric_type=MetricType.KS_TEST,
        value=stat,
        threshold=alpha,
        p_value=p_val,
        has_drift=has_drift,
        status=status,
        details={
            "alpha": alpha,
            "interpretation": f"{'Significant drift detected' if has_drift else 'Distributions are statistically indistinguishable'} (p={p_val} vs alpha={alpha})",
        },
    )

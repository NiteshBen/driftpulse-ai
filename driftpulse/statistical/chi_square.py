"""Chi-Square test and Total Variation Distance (TVD) for categorical features."""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

from driftpulse.core.exceptions import StatisticalTestError
from driftpulse.core.types import DriftStatus, MetricResult, MetricType


def _gamma_incomplete_upper(a: float, x: float) -> float:
    """Computes upper incomplete regularized gamma function Q(a, x) = 1 - P(a, x).

    Used to compute exact/asymptotic p-value for Chi-Square distribution where df = 2*a and chi2 = 2*x.
    """
    if x <= 0:
        return 1.0
    if x < a + 1.0:
        # Use series representation for P(a, x)
        term = 1.0 / a
        total = term
        for n in range(1, 100):
            term *= x / (a + n)
            total += term
            if abs(term / total) < 1e-12:
                break
        try:
            ln_gamma = math.lgamma(a)
            p = total * math.exp(-x + a * math.log(x) - ln_gamma)
            return max(0.0, min(1.0, 1.0 - p))
        except (ValueError, OverflowError):
            return 0.0
    else:
        # Use continued fraction approximation for Q(a, x)
        b = x + 1.0 - a
        c = 1.0 / 1e-30
        d = 1.0 / b
        h = d
        for i in range(1, 100):
            an = -i * (i - a)
            b += 2.0
            d = an * d + b
            if abs(d) < 1e-30:
                d = 1e-30
            c = b + an / c
            if abs(c) < 1e-30:
                c = 1e-30
            d = 1.0 / d
            delta = d * c
            h *= delta
            if abs(delta - 1.0) < 1e-12:
                break
        try:
            ln_gamma = math.lgamma(a)
            return max(0.0, min(1.0, math.exp(-x + a * math.log(x) - ln_gamma) * h))
        except (ValueError, OverflowError):
            return 0.0


def chi_square_goodness_of_fit(
    ref_counts: Dict[str, int],
    curr_counts: Dict[str, int],
    epsilon: float = 1e-4,
) -> Tuple[float, float, int]:
    """Computes Chi-Square statistic, asymptotic p-value, and degrees of freedom."""
    n_ref = sum(ref_counts.values())
    n_curr = sum(curr_counts.values())

    if n_ref == 0 or n_curr == 0:
        raise StatisticalTestError("Both count dictionaries must have strictly positive totals.")

    all_categories = sorted(list(set(ref_counts.keys()) | set(curr_counts.keys())))
    k = len(all_categories)
    if k <= 1:
        return 0.0, 1.0, 1

    chi2 = 0.0
    for cat in all_categories:
        observed = float(curr_counts.get(cat, 0))
        ref_prob = max(ref_counts.get(cat, 0) / n_ref, epsilon)
        expected = n_curr * ref_prob

        chi2 += ((observed - expected) ** 2) / expected

    df = k - 1
    # Chi-square p-value is Q(df/2, chi2/2)
    p_val = _gamma_incomplete_upper(df / 2.0, chi2 / 2.0)
    return round(chi2, 6), round(p_val, 6), df


def total_variation_distance(
    ref_counts: Dict[str, int],
    curr_counts: Dict[str, int],
) -> float:
    """Computes Total Variation Distance (TVD) = 0.5 * sum |p_i - q_i| in [0, 1]."""
    n_ref = sum(ref_counts.values())
    n_curr = sum(curr_counts.values())

    all_categories = sorted(list(set(ref_counts.keys()) | set(curr_counts.keys())))
    tvd_sum = 0.0

    for cat in all_categories:
        p_ref = ref_counts.get(cat, 0) / n_ref
        p_curr = curr_counts.get(cat, 0) / n_curr
        tvd_sum += abs(p_ref - p_curr)

    return round(0.5 * tvd_sum, 6)


def evaluate_chi_square_drift(
    ref_counts: Dict[str, int],
    curr_counts: Dict[str, int],
    alpha: float = 0.05,
) -> MetricResult:
    """Evaluates categorical drift using the Chi-Square test of independence."""
    chi2, p_val, df = chi_square_goodness_of_fit(ref_counts, curr_counts)
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
        metric_type=MetricType.CHI_SQUARE,
        value=chi2,
        threshold=alpha,
        p_value=p_val,
        has_drift=has_drift,
        status=status,
        details={
            "degrees_of_freedom": df,
            "alpha": alpha,
            "interpretation": f"Chi-Square test {'rejects' if has_drift else 'fails to reject'} distribution equality (p={p_val})",
        },
    )

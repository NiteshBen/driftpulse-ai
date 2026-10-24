"""Population Stability Index (PSI) for numerical and categorical distribution drift."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence

from driftpulse.core.exceptions import StatisticalTestError
from driftpulse.core.types import DriftStatus, MetricResult, MetricType


def compute_numerical_psi(
    sample_ref: Sequence[float],
    sample_curr: Sequence[float],
    num_bins: int = 10,
    epsilon: float = 1e-4,
) -> Dict[str, Any]:
    """Computes Population Stability Index (PSI) for continuous numerical variables.

    Uses quantile-based binning derived from the reference distribution to ensure
    equal-mass baseline bins.
    """
    clean_ref = sorted([float(x) for x in sample_ref if x is not None and not math.isnan(float(x))])
    clean_curr = sorted([float(x) for x in sample_curr if x is not None and not math.isnan(float(x))])

    n_ref = len(clean_ref)
    n_curr = len(clean_curr)

    if n_ref == 0 or n_curr == 0:
        raise StatisticalTestError("Both samples must contain at least 1 valid observation.")

    # Determine quantile bin boundaries from reference
    bin_edges = []
    for i in range(num_bins + 1):
        q = i / num_bins
        idx = int(q * (n_ref - 1))
        bin_edges.append(clean_ref[idx])

    # Deduplicate boundaries in case of low variance
    unique_edges = sorted(list(set(bin_edges)))
    if len(unique_edges) < 2:
        # Constant data, use tiny window
        unique_edges = [clean_ref[0] - 1.0, clean_ref[0] + 1.0]

    # Ensure boundaries cover current extremes
    unique_edges[0] = min(unique_edges[0], clean_curr[0] - 1.0)
    unique_edges[-1] = max(unique_edges[-1], clean_curr[-1] + 1.0)

    n_intervals = len(unique_edges) - 1
    ref_counts = [0] * n_intervals
    curr_counts = [0] * n_intervals

    for val in clean_ref:
        for b in range(n_intervals):
            if b == n_intervals - 1:
                if unique_edges[b] <= val <= unique_edges[b + 1]:
                    ref_counts[b] += 1
                    break
            else:
                if unique_edges[b] <= val < unique_edges[b + 1]:
                    ref_counts[b] += 1
                    break

    for val in clean_curr:
        for b in range(n_intervals):
            if b == n_intervals - 1:
                if unique_edges[b] <= val <= unique_edges[b + 1]:
                    curr_counts[b] += 1
                    break
            else:
                if unique_edges[b] <= val < unique_edges[b + 1]:
                    curr_counts[b] += 1
                    break

    # Calculate PSI per bin
    total_psi = 0.0
    bin_details = []

    for b in range(n_intervals):
        pct_ref = max(ref_counts[b] / n_ref, epsilon)
        pct_curr = max(curr_counts[b] / n_curr, epsilon)

        contribution = (pct_curr - pct_ref) * math.log(pct_curr / pct_ref)
        total_psi += contribution

        bin_details.append({
            "bin": f"[{round(unique_edges[b], 2)}, {round(unique_edges[b+1], 2)})",
            "ref_pct": round(pct_ref, 4),
            "curr_pct": round(pct_curr, 4),
            "psi_contribution": round(contribution, 6),
        })

    return {
        "psi": round(total_psi, 6),
        "bin_details": bin_details,
        "n_bins": n_intervals,
    }


def compute_categorical_psi(
    counts_ref: Dict[str, int],
    counts_curr: Dict[str, int],
    epsilon: float = 1e-4,
) -> Dict[str, Any]:
    """Computes PSI for discrete categorical frequency distributions."""
    total_ref = sum(counts_ref.values())
    total_curr = sum(counts_curr.values())

    if total_ref == 0 or total_curr == 0:
        raise StatisticalTestError("Both category sets must have non-zero total frequency count.")

    all_categories = sorted(list(set(counts_ref.keys()) | set(counts_curr.keys())))
    total_psi = 0.0
    category_details = []

    for cat in all_categories:
        pct_ref = max(counts_ref.get(cat, 0) / total_ref, epsilon)
        pct_curr = max(counts_curr.get(cat, 0) / total_curr, epsilon)

        contribution = (pct_curr - pct_ref) * math.log(pct_curr / pct_ref)
        total_psi += contribution

        category_details.append({
            "category": cat,
            "ref_pct": round(pct_ref, 4),
            "curr_pct": round(pct_curr, 4),
            "psi_contribution": round(contribution, 6),
        })

    return {
        "psi": round(total_psi, 6),
        "category_details": category_details,
    }


def evaluate_psi_drift(
    psi_value: float,
    warning_threshold: float = 0.10,
    critical_threshold: float = 0.25,
) -> MetricResult:
    """Classifies drift severity based on the canonical credit & ML PSI standards."""
    if psi_value >= critical_threshold:
        status = DriftStatus.CRITICAL_DRIFT
        has_drift = True
    elif psi_value >= warning_threshold:
        status = DriftStatus.MODERATE_DRIFT
        has_drift = True
    elif psi_value >= warning_threshold / 2.0:
        status = DriftStatus.LOW_DRIFT
        has_drift = False
    else:
        status = DriftStatus.NO_DRIFT
        has_drift = False

    return MetricResult(
        metric_type=MetricType.PSI,
        value=psi_value,
        threshold=warning_threshold,
        has_drift=has_drift,
        status=status,
        details={
            "warning_threshold": warning_threshold,
            "critical_threshold": critical_threshold,
            "rule": "PSI < 0.10 (Stable), 0.10-0.25 (Moderate), >= 0.25 (Critical)",
        },
    )

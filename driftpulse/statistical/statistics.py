"""Summary and heuristic statistical checks: Out-of-vocabulary, Z-score shift, and vector distance."""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Set, Tuple

from driftpulse.core.exceptions import StatisticalTestError
from driftpulse.core.types import DriftStatus, MetricResult, MetricType


def compute_oov_rate(ref_categories: Set[str], curr_samples: Sequence[str]) -> Tuple[float, List[str]]:
    """Calculates the Out-of-Vocabulary (OOV) rate for categorical features.

    Returns:
        (oov_rate, list_of_new_categories)
    """
    if not curr_samples:
        return 0.0, []

    new_cats = set()
    oov_count = 0
    for item in curr_samples:
        s = str(item)
        if s not in ref_categories:
            oov_count += 1
            new_cats.add(s)

    rate = oov_count / len(curr_samples)
    return round(rate, 6), sorted(list(new_cats))


def compute_z_score_shift(
    ref_mean: float,
    ref_std: float,
    curr_mean: float,
) -> float:
    """Computes standardized Z-score shift of the current distribution mean."""
    if ref_std < 1e-9:
        return 0.0 if abs(curr_mean - ref_mean) < 1e-9 else float("inf")
    return round(abs(curr_mean - ref_mean) / ref_std, 4)


def cosine_distance_vectors(vec1: Sequence[float], vec2: Sequence[float]) -> float:
    """Computes cosine distance = 1 - cosine_similarity between two vectors."""
    if len(vec1) != len(vec2) or len(vec1) == 0:
        raise StatisticalTestError("Vectors must be non-empty and of identical dimension.")

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))

    if norm1 < 1e-9 or norm2 < 1e-9:
        return 1.0  # Degenerate zero vector

    cos_sim = dot_product / (norm1 * norm2)
    cos_sim = max(-1.0, min(1.0, cos_sim))
    return round(1.0 - cos_sim, 6)

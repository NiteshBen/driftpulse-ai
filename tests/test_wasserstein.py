"""Tests for Wasserstein-1 distance."""

import random
import pytest
from driftpulse.core.exceptions import StatisticalTestError
from driftpulse.core.types import DriftStatus
from driftpulse.statistical.wasserstein import (
    evaluate_wasserstein_drift,
    wasserstein_distance_1d,
)


def test_wasserstein_identical():
    s = [1.0, 2.0, 3.0, 4.0, 5.0]
    dist = wasserstein_distance_1d(s, s)
    assert dist == 0.0


def test_wasserstein_constant_shift():
    s1 = [1.0, 2.0, 3.0]
    s2 = [2.0, 3.0, 4.0]  # Exact +1 shift
    dist = wasserstein_distance_1d(s1, s2)
    assert round(dist, 4) == 1.0


def test_wasserstein_evaluation():
    random.seed(42)
    s1 = [random.gauss(10, 2) for _ in range(150)]
    s2 = [random.gauss(14, 2) for _ in range(150)]  # +4 units = 2 standard deviations

    res = evaluate_wasserstein_drift(s1, s2, threshold=0.15)
    assert res.has_drift is True
    assert res.status in (DriftStatus.MODERATE_DRIFT, DriftStatus.CRITICAL_DRIFT)


def test_wasserstein_empty_error():
    with pytest.raises(StatisticalTestError):
        wasserstein_distance_1d([], [1.0])

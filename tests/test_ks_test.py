"""Tests for Kolmogorov-Smirnov 2-sample test."""

import random
import pytest
from driftpulse.core.exceptions import StatisticalTestError
from driftpulse.core.types import DriftStatus
from driftpulse.statistical.ks_test import evaluate_ks_drift, kolmogorov_smirnov_2sample


def test_ks_identical_samples():
    sample = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    stat, p_val = kolmogorov_smirnov_2sample(sample, sample)
    assert stat == 0.0
    assert p_val == 1.0


def test_ks_no_drift():
    random.seed(42)
    s1 = [random.gauss(0, 1) for _ in range(200)]
    s2 = [random.gauss(0, 1) for _ in range(200)]

    res = evaluate_ks_drift(s1, s2, alpha=0.05)
    assert res.has_drift is False
    assert res.status == DriftStatus.NO_DRIFT
    assert res.p_value > 0.05


def test_ks_significant_drift():
    random.seed(42)
    s1 = [random.gauss(0, 1) for _ in range(200)]
    s2 = [random.gauss(3, 1) for _ in range(200)]  # Severe shift

    res = evaluate_ks_drift(s1, s2, alpha=0.05)
    assert res.has_drift is True
    assert res.status in (DriftStatus.MODERATE_DRIFT, DriftStatus.CRITICAL_DRIFT)
    assert res.p_value < 0.01


def test_ks_empty_sample_error():
    with pytest.raises(StatisticalTestError):
        kolmogorov_smirnov_2sample([], [1.0, 2.0])

"""Tests for Population Stability Index (PSI)."""

import random
import pytest
from driftpulse.core.exceptions import StatisticalTestError
from driftpulse.core.types import DriftStatus
from driftpulse.statistical.psi import (
    compute_categorical_psi,
    compute_numerical_psi,
    evaluate_psi_drift,
)


def test_psi_identical_numerical():
    sample = list(range(100))
    res = compute_numerical_psi(sample, sample)
    assert res["psi"] < 0.01


def test_psi_numerical_drift():
    random.seed(42)
    s1 = [random.gauss(50, 10) for _ in range(300)]
    s2 = [random.gauss(75, 12) for _ in range(300)]  # Clear shift

    res = compute_numerical_psi(s1, s2)
    eval_res = evaluate_psi_drift(res["psi"])
    assert res["psi"] > 0.25
    assert eval_res.has_drift is True
    assert eval_res.status == DriftStatus.CRITICAL_DRIFT


def test_psi_categorical_stable():
    ref_counts = {"A": 50, "B": 30, "C": 20}
    curr_counts = {"A": 52, "B": 29, "C": 19}

    res = compute_categorical_psi(ref_counts, curr_counts)
    eval_res = evaluate_psi_drift(res["psi"])
    assert res["psi"] < 0.05
    assert eval_res.has_drift is False
    assert eval_res.status == DriftStatus.NO_DRIFT


def test_psi_categorical_critical_drift():
    ref_counts = {"A": 70, "B": 20, "C": 10}
    curr_counts = {"A": 10, "B": 20, "C": 70}  # Inverse distribution

    res = compute_categorical_psi(ref_counts, curr_counts)
    eval_res = evaluate_psi_drift(res["psi"])
    assert res["psi"] > 0.25
    assert eval_res.status == DriftStatus.CRITICAL_DRIFT


def test_psi_empty_error():
    with pytest.raises(StatisticalTestError):
        compute_numerical_psi([], [1.0])

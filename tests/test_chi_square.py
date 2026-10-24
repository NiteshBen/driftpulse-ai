"""Tests for Chi-Square test and Total Variation Distance."""

import pytest
from driftpulse.core.exceptions import StatisticalTestError
from driftpulse.core.types import DriftStatus
from driftpulse.statistical.chi_square import (
    chi_square_goodness_of_fit,
    evaluate_chi_square_drift,
    total_variation_distance,
)


def test_chi_square_identical():
    counts = {"A": 100, "B": 100, "C": 100}
    chi2, p_val, df = chi_square_goodness_of_fit(counts, counts)
    assert chi2 == 0.0
    assert p_val == 1.0
    assert df == 2


def test_chi_square_drift():
    ref = {"A": 100, "B": 100, "C": 100}
    curr = {"A": 20, "B": 40, "C": 240}  # Significant skew
    res = evaluate_chi_square_drift(ref, curr, alpha=0.05)
    assert res.has_drift is True
    assert res.p_value < 0.01


def test_tvd_identical():
    counts = {"A": 50, "B": 50}
    tvd = total_variation_distance(counts, counts)
    assert tvd == 0.0


def test_tvd_disjoint():
    ref = {"A": 100, "B": 0}
    curr = {"A": 0, "B": 100}
    tvd = total_variation_distance(ref, curr)
    assert round(tvd, 4) == 1.0


def test_chi_square_empty_error():
    with pytest.raises(StatisticalTestError):
        chi_square_goodness_of_fit({}, {"A": 10})

"""Tests for summary statistics, OOV rate, and vector distances."""

import pytest
from driftpulse.core.exceptions import StatisticalTestError
from driftpulse.statistical.statistics import (
    compute_oov_rate,
    compute_z_score_shift,
    cosine_distance_vectors,
)


def test_oov_rate():
    ref_cats = {"US", "UK", "CA"}
    curr = ["US", "UK", "FR", "DE", "CA"]  # 2 new out of 5
    rate, new_cats = compute_oov_rate(ref_cats, curr)
    assert rate == 0.4
    assert new_cats == ["DE", "FR"]


def test_z_score_shift():
    shift = compute_z_score_shift(ref_mean=100.0, ref_std=10.0, curr_mean=125.0)
    assert shift == 2.5


def test_cosine_distance():
    v1 = [1.0, 0.0]
    v2 = [1.0, 0.0]
    assert cosine_distance_vectors(v1, v2) == 0.0

    v3 = [0.0, 1.0]
    assert round(cosine_distance_vectors(v1, v3), 4) == 1.0


def test_cosine_distance_dim_mismatch():
    with pytest.raises(StatisticalTestError):
        cosine_distance_vectors([1.0], [1.0, 2.0])

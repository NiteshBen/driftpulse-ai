"""Tests for Jensen-Shannon and KL divergence."""

import pytest
from driftpulse.core.exceptions import StatisticalTestError
from driftpulse.core.types import DriftStatus
from driftpulse.statistical.jensen_shannon import (
    evaluate_jensen_shannon_drift,
    jensen_shannon_divergence,
    kl_divergence,
)


def test_jsd_identical():
    p = [0.25, 0.25, 0.25, 0.25]
    jsd = jensen_shannon_divergence(p, p)
    assert jsd == 0.0


def test_jsd_disjoint():
    p = [1.0, 0.0]
    q = [0.0, 1.0]
    jsd = jensen_shannon_divergence(p, q)
    # With base 2 log, maximum JSD for completely disjoint distributions is 1.0
    assert round(jsd, 2) == 1.0


def test_evaluate_jsd_drift():
    p = [0.7, 0.2, 0.1]
    q = [0.1, 0.2, 0.7]
    res = evaluate_jensen_shannon_drift(p, q, threshold=0.10)
    assert res.has_drift is True
    assert res.value > 0.10


def test_jsd_empty_error():
    with pytest.raises(StatisticalTestError):
        jensen_shannon_divergence([], [])

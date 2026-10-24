"""Tests for ConceptDriftDetector."""

import random
import pytest
from driftpulse.core.exceptions import StatisticalTestError
from driftpulse.core.types import DriftStatus
from driftpulse.detectors.concept_detector import ConceptDriftDetector


def test_concept_no_drift():
    random.seed(42)
    ref_preds = [0.85 + random.gauss(0, 0.02) for _ in range(300)]
    curr_preds = [0.85 + random.gauss(0, 0.02) for _ in range(300)]

    detector = ConceptDriftDetector()
    res = detector.compare_predictions(ref_preds, curr_preds)

    assert res.has_drift is False
    assert res.prediction_drift_status in (DriftStatus.NO_DRIFT, DriftStatus.LOW_DRIFT)


def test_concept_confidence_decay():
    random.seed(42)
    ref_preds = [random.betavariate(9, 1) for _ in range(200)]  # Mean ~0.90
    curr_preds = [random.betavariate(2, 2) for _ in range(200)]  # Mean ~0.50 (Uncertainty)

    detector = ConceptDriftDetector(confidence_decay_threshold=0.10)
    res = detector.compare_predictions(ref_preds, curr_preds)

    assert res.has_drift is True
    assert res.confidence_decay > 0.10
    assert res.entropy_increase > 0.0


def test_concept_empty_error():
    detector = ConceptDriftDetector()
    with pytest.raises(StatisticalTestError):
        detector.compare_predictions([], [0.5])

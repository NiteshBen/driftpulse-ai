"""Tests for TabularDriftDetector."""

import random
import pytest
from driftpulse.core.config import ThresholdConfig
from driftpulse.core.exceptions import IncompatibleSchemaError
from driftpulse.core.types import DriftStatus
from driftpulse.detectors.tabular_detector import TabularDriftDetector


def test_tabular_detector_no_drift():
    random.seed(42)
    ref = {
        "x": [random.gauss(10, 2) for _ in range(200)],
        "cat": random.choices(["A", "B"], k=200),
    }
    curr = {
        "x": [random.gauss(10.1, 1.95) for _ in range(200)],
        "cat": random.choices(["A", "B"], k=200),
    }

    detector = TabularDriftDetector()
    report = detector.compare_datasets(ref, curr, dataset_name="TestDataset")

    assert report.total_features == 2
    assert report.has_dataset_drift is False
    assert report.overall_status in (DriftStatus.NO_DRIFT, DriftStatus.LOW_DRIFT)


def test_tabular_detector_severe_drift():
    random.seed(42)
    ref = {
        "val": [random.gauss(0, 1) for _ in range(200)],
        "category": ["X"] * 100 + ["Y"] * 100,
    }
    curr = {
        "val": [random.gauss(5, 1) for _ in range(200)],  # Extreme shift
        "category": ["Z"] * 200,  # Completely new category
    }

    detector = TabularDriftDetector()
    report = detector.compare_datasets(ref, curr)

    assert report.has_dataset_drift is True
    assert report.drifted_features == 2
    assert report.drift_share == 1.0
    assert report.overall_status == DriftStatus.CRITICAL_DRIFT
    assert len(report.alerts) > 0


def test_schema_mismatch_error():
    ref = {"a": [1, 2, 3]}
    curr = {"b": [4, 5, 6]}
    detector = TabularDriftDetector()
    with pytest.raises(IncompatibleSchemaError):
        detector.compare_datasets(ref, curr)

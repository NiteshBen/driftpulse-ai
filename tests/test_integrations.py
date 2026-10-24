"""Tests for data adapters and Scikit-learn monitor wrapper."""

import os
import tempfile
from driftpulse.integrations.pandas_adapter import load_csv_as_dict, to_standard_dict
from driftpulse.integrations.sklearn_monitor import SklearnDriftMonitor


class DummyEstimator:
    """Mock ML model for testing inference wrapper."""
    def predict(self, X):
        n = len(X["feature_1"]) if isinstance(X, dict) and "feature_1" in X else len(X)
        return [1] * n

    def predict_proba(self, X):
        n = len(X["feature_1"]) if isinstance(X, dict) and "feature_1" in X else len(X)
        return [[0.1, 0.9]] * n


def test_to_standard_dict():
    # From dict of lists
    d = {"a": [1, 2], "b": [3, 4]}
    assert to_standard_dict(d) == d

    # From list of dicts
    ld = [{"x": 10, "y": 20}, {"x": 30, "y": 40}]
    std = to_standard_dict(ld)
    assert std == {"x": [10, 30], "y": [20, 40]}


def test_csv_loading():
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "test.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("col_int,col_float,col_str\n1,1.5,hello\n2,2.5,world\n")

        data = load_csv_as_dict(csv_path)
        assert data["col_int"] == [1, 2]
        assert data["col_float"] == [1.5, 2.5]
        assert data["col_str"] == ["hello", "world"]


def test_sklearn_drift_monitor():
    dummy = DummyEstimator()
    baseline = {"feature_1": [10, 11, 12, 13, 14] * 20}

    monitor = SklearnDriftMonitor(
        estimator=dummy,
        baseline_data=baseline,
        feature_names=["feature_1"],
        buffer_size=10,
    )

    preds = monitor.predict({"feature_1": [10, 11, 12]})
    assert preds == [1, 1, 1]

    # Force check
    report = monitor.check_drift()
    assert report is not None
    assert "feature_1" in report.feature_results

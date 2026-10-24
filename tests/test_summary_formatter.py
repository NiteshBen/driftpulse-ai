"""Tests for format_terminal_summary."""

from driftpulse.core.types import DatasetDriftReport, DriftStatus, FeatureDriftResult, FeatureType, MetricType
from driftpulse.reporting.summary_formatter import format_terminal_summary


def test_format_terminal_summary():
    feat_res = FeatureDriftResult(
        feature_name="user_age",
        feature_type=FeatureType.NUMERICAL,
        drift_status=DriftStatus.NO_DRIFT,
        primary_metric=MetricType.PSI,
        primary_value=0.02,
        p_value=0.88,
    )

    report = DatasetDriftReport(
        dataset_name="UserData_Batch",
        timestamp="2026-10-24T12:00:00",
        total_features=1,
        drifted_features=0,
        drift_share=0.0,
        overall_status=DriftStatus.NO_DRIFT,
        feature_results={"user_age": feat_res},
        alerts=[{"level": "INFO", "message": "Baseline stable"}],
    )

    summary = format_terminal_summary(report)
    assert "UserData_Batch" in summary
    assert "user_age" in summary
    assert "NO DRIFT" in summary
    assert "Baseline stable" in summary

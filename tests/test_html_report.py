"""Tests for HTML report generation."""

from driftpulse.core.types import DatasetDriftReport, DriftStatus, FeatureDriftResult, FeatureProfile, FeatureType, MetricResult, MetricType
from driftpulse.reporting.html_report import generate_drift_gauge_svg, generate_html_report, generate_mini_histogram_svg


def test_generate_drift_gauge_svg():
    svg_0 = generate_drift_gauge_svg(0.0)
    assert "<svg" in svg_0
    assert "0.0%" in svg_0

    svg_50 = generate_drift_gauge_svg(0.5)
    assert "50.0%" in svg_50


def test_generate_html_report():
    base_prof = FeatureProfile(
        name="income",
        feature_type=FeatureType.NUMERICAL,
        count=100,
        histogram={"bin_edges": [0, 50, 100], "counts": [60, 40]},
    )
    curr_prof = FeatureProfile(
        name="income",
        feature_type=FeatureType.NUMERICAL,
        count=100,
        histogram={"bin_edges": [0, 50, 100], "counts": [20, 80]},
    )

    feat_res = FeatureDriftResult(
        feature_name="income",
        feature_type=FeatureType.NUMERICAL,
        drift_status=DriftStatus.CRITICAL_DRIFT,
        primary_metric=MetricType.PSI,
        primary_value=0.35,
        all_metrics={
            "psi": MetricResult(metric_type=MetricType.PSI, value=0.35, threshold=0.10, has_drift=True),
        },
        baseline_profile=base_prof,
        current_profile=curr_prof,
        warnings=["Test warning"],
    )

    report = DatasetDriftReport(
        dataset_name="TestReport",
        timestamp="2026-10-24T12:00:00",
        total_features=1,
        drifted_features=1,
        drift_share=1.0,
        overall_status=DriftStatus.CRITICAL_DRIFT,
        feature_results={"income": feat_res},
        alerts=[{"level": "CRITICAL", "message": "Test Alert"}],
    )

    html = generate_html_report(report)
    assert "<!DOCTYPE html>" in html
    assert "TestReport" in html
    assert "income" in html
    assert "CRITICAL DRIFT" in html

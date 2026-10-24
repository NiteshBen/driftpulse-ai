"""Tests for MonitoringPipeline continuous evaluation."""

import random
from driftpulse.core.config import ThresholdConfig
from driftpulse.core.types import DriftStatus
from driftpulse.monitoring.pipeline import MonitoringPipeline


def test_monitoring_pipeline_batch_history():
    random.seed(42)
    baseline = {"val": [random.gauss(10, 2) for _ in range(300)]}

    alerts_caught = []
    cfg = ThresholdConfig(psi_warning=0.15, psi_critical=0.30)
    pipeline = MonitoringPipeline(
        baseline_data=baseline,
        config=cfg,
        on_drift_alert=lambda a: alerts_caught.append(a),
    )

    # Batch 1: Stable
    b1 = {"val": [random.gauss(10, 2) for _ in range(200)]}
    r1 = pipeline.evaluate_batch(b1, dataset_name="Batch_1")
    assert r1.has_dataset_drift is False
    assert len(alerts_caught) == 0

    # Batch 2: Severe drift
    b2 = {"val": [random.gauss(25, 3) for _ in range(200)]}
    r2 = pipeline.evaluate_batch(b2, dataset_name="Batch_2")
    assert r2.has_dataset_drift is True
    assert len(alerts_caught) == 1

    metrics = pipeline.get_summary_metrics()
    assert metrics["total_batches_evaluated"] == 2
    assert metrics["batches_with_drift"] == 1
    assert metrics["drift_rate"] == 0.5
    assert metrics["total_alerts_dispatched"] == 1


def test_monitoring_pipeline_empty_metrics():
    pipeline = MonitoringPipeline(baseline_data={"a": [1, 2, 3]})
    metrics = pipeline.get_summary_metrics()
    assert metrics["total_batches_evaluated"] == 0

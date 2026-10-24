"""Continuous monitoring pipeline for batch and stream evaluation."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from driftpulse.core.config import ThresholdConfig
from driftpulse.core.types import DatasetDriftReport, FeatureType
from driftpulse.detectors.tabular_detector import TabularDriftDetector
from driftpulse.monitoring.alerts import AlertDispatcher, AlertNotification
from driftpulse.monitoring.baseline import BaselineStore


class MonitoringPipeline:
    """End-to-end monitoring pipeline coordinating baseline comparison, alerting, and reporting."""

    def __init__(
        self,
        baseline_data: Any,
        config: Optional[ThresholdConfig] = None,
        feature_types: Optional[Dict[str, FeatureType]] = None,
        on_drift_alert: Optional[Callable[[AlertNotification], None]] = None,
    ):
        self.baseline_data = baseline_data
        self.config = config or ThresholdConfig()
        self.feature_types = feature_types
        self.detector = TabularDriftDetector(config=self.config)
        self.dispatcher = AlertDispatcher()
        if on_drift_alert:
            self.dispatcher.add_callback(on_drift_alert)

        self.reports_history: List[DatasetDriftReport] = []

    def evaluate_batch(
        self,
        current_data: Any,
        dataset_name: str = "Production_Batch",
    ) -> DatasetDriftReport:
        """Processes an incoming production batch against the baseline."""
        report = self.detector.compare_datasets(
            ref_data=self.baseline_data,
            curr_data=current_data,
            dataset_name=dataset_name,
            feature_types=self.feature_types,
        )

        self.dispatcher.dispatch_report(report)
        self.reports_history.append(report)
        return report

    def get_summary_metrics(self) -> Dict[str, Any]:
        """Returns aggregated telemetry across all historical evaluations."""
        if not self.reports_history:
            return {"total_batches_evaluated": 0, "batches_with_drift": 0}

        drift_count = sum(1 for r in self.reports_history if r.has_dataset_drift)
        return {
            "total_batches_evaluated": len(self.reports_history),
            "batches_with_drift": drift_count,
            "drift_rate": round(drift_count / len(self.reports_history), 4),
            "total_alerts_dispatched": len(self.dispatcher.history),
        }

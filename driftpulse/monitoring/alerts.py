"""Monitoring alert dispatching, webhooks, and notifications."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from driftpulse.core.types import AlertLevel, DatasetDriftReport, DriftStatus


@dataclass
class AlertNotification:
    """Standardized alert payload."""
    level: AlertLevel
    title: str
    message: str
    dataset_name: str
    drifted_features: int
    total_features: int
    drift_share: float
    timestamp: str

    def to_slack_payload(self) -> Dict[str, Any]:
        """Formats alert as a rich Slack webhook message block."""
        color = "#ef4444" if self.level == AlertLevel.CRITICAL else "#f59e0b"
        return {
            "text": f"*{self.title}*",
            "attachments": [
                {
                    "color": color,
                    "fields": [
                        {"title": "Dataset", "value": self.dataset_name, "short": True},
                        {"title": "Drift Share", "value": f"{round(self.drift_share*100, 1)}%", "short": True},
                        {"title": "Drifted Count", "value": f"{self.drifted_features}/{self.total_features}", "short": True},
                        {"title": "Timestamp", "value": self.timestamp, "short": True},
                        {"title": "Details", "value": self.message, "short": False},
                    ],
                }
            ],
        }


class AlertDispatcher:
    """Dispatches drift notifications to subscribers, webhooks, or log handlers."""

    def __init__(self):
        self._callbacks: List[Callable[[AlertNotification], None]] = []
        self._history: List[AlertNotification] = []

    def add_callback(self, callback: Callable[[AlertNotification], None]) -> None:
        self._callbacks.append(callback)

    @property
    def history(self) -> List[AlertNotification]:
        return list(self._history)

    def dispatch_report(self, report: DatasetDriftReport) -> List[AlertNotification]:
        """Evaluates a report and dispatches notifications if drift thresholds are triggered."""
        dispatched = []

        if report.overall_status.is_drifted:
            level = AlertLevel.CRITICAL if report.overall_status == DriftStatus.CRITICAL_DRIFT else AlertLevel.WARNING
            title = f"[{level.value}] Data Drift Alert: {report.dataset_name}"
            msg = (
                f"{report.drifted_features} of {report.total_features} features drifted "
                f"({round(report.drift_share * 100, 1)}%). Model retraining or input quarantine advised."
            )

            notification = AlertNotification(
                level=level,
                title=title,
                message=msg,
                dataset_name=report.dataset_name,
                drifted_features=report.drifted_features,
                total_features=report.total_features,
                drift_share=report.drift_share,
                timestamp=report.timestamp,
            )

            self._history.append(notification)
            dispatched.append(notification)

            for cb in self._callbacks:
                try:
                    cb(notification)
                except Exception:
                    pass

        return dispatched

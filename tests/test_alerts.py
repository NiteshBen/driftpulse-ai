"""Tests for AlertDispatcher and notifications."""

from driftpulse.core.types import DatasetDriftReport, DriftStatus
from driftpulse.monitoring.alerts import AlertDispatcher, AlertLevel, AlertNotification


def test_alert_dispatcher():
    dispatcher = AlertDispatcher()
    received = []

    dispatcher.add_callback(lambda n: received.append(n))

    # Test report without drift -> no alert
    r_stable = DatasetDriftReport(
        dataset_name="StableDS",
        timestamp="2026-10-24T12:00:00",
        total_features=5,
        drifted_features=0,
        drift_share=0.0,
        overall_status=DriftStatus.NO_DRIFT,
    )
    dispatched = dispatcher.dispatch_report(r_stable)
    assert len(dispatched) == 0
    assert len(received) == 0

    # Test report with critical drift -> triggers alert
    r_drift = DatasetDriftReport(
        dataset_name="DriftedDS",
        timestamp="2026-10-24T12:05:00",
        total_features=5,
        drifted_features=3,
        drift_share=0.6,
        overall_status=DriftStatus.CRITICAL_DRIFT,
    )
    dispatched = dispatcher.dispatch_report(r_drift)
    assert len(dispatched) == 1
    assert len(received) == 1
    assert dispatched[0].level == AlertLevel.CRITICAL

    slack_msg = dispatched[0].to_slack_payload()
    assert "attachments" in slack_msg

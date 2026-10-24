"""Tests for JSON report exporter."""

import json
import os
import tempfile
from driftpulse.core.types import DatasetDriftReport, DriftStatus
from driftpulse.reporting.json_exporter import export_report_to_json


def test_export_report_to_json():
    report = DatasetDriftReport(
        dataset_name="JSONTest",
        timestamp="2026-10-24T12:00:00",
        total_features=3,
        drifted_features=1,
        drift_share=0.3333,
        overall_status=DriftStatus.MODERATE_DRIFT,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "report.json")
        export_report_to_json(report, out_path)

        assert os.path.exists(out_path)
        with open(out_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["dataset_name"] == "JSONTest"
        assert data["overall_status"] == "moderate_drift"
        assert data["drifted_features"] == 1

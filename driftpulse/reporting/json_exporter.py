"""Standardized JSON export for CI/CD gates and machine-readable telemetry."""

from __future__ import annotations

import json
from typing import Any, Dict

from driftpulse.core.types import DatasetDriftReport


def export_report_to_json(report: DatasetDriftReport, filepath: str, indent: int = 2) -> str:
    """Exports a DatasetDriftReport as structured JSON to disk."""
    data = report.to_dict()
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)
    return filepath

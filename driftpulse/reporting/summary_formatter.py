"""Terminal ANSI table formatter for drift diagnostics."""

from __future__ import annotations

from driftpulse.core.types import DatasetDriftReport, DriftStatus


def format_terminal_summary(report: DatasetDriftReport) -> str:
    """Formats a DatasetDriftReport into a clear terminal summary table."""
    lines = []
    lines.append("=" * 78)
    lines.append(f"  DRIFTPULSE-AI SENTINEL REPORT: {report.dataset_name}")
    lines.append("=" * 78)
    lines.append(f"  Overall Status:   {report.overall_status.value.upper()}")
    lines.append(f"  Drifted Features: {report.drifted_features} / {report.total_features} ({round(report.drift_share * 100, 1)}%)")
    lines.append(f"  Timestamp:        {report.timestamp}")
    lines.append("-" * 78)
    lines.append(f"  {'FEATURE':<22} | {'TYPE':<11} | {'STATUS':<15} | {'METRIC':<10} | {'VAL / P-VAL'}")
    lines.append("-" * 78)

    for feat_name, res in report.feature_results.items():
        status_str = res.drift_status.value.replace("_", " ").upper()
        p_str = f"{round(res.primary_value, 4)}"
        if res.p_value is not None:
            p_str += f" (p={round(res.p_value, 4)})"

        lines.append(
            f"  {feat_name[:20]:<22} | {res.feature_type.value[:10]:<11} | {status_str[:14]:<15} | {res.primary_metric.value[:9]:<10} | {p_str}"
        )

    lines.append("=" * 78)
    if report.alerts:
        lines.append("  ALERTS & WARNINGS:")
        for alert in report.alerts:
            lines.append(f"   * [{alert.get('level', 'INFO')}] {alert.get('message')}")
        lines.append("=" * 78)

    return "\n".join(lines)

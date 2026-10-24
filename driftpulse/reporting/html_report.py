"""Standalone Power BI-style executive dark-mode HTML drift report generator with embedded SVGs."""

from __future__ import annotations

import math
from typing import Any, Dict, List

from driftpulse.core.types import DatasetDriftReport, DriftStatus, FeatureDriftResult, FeatureType


def generate_drift_gauge_svg(drift_share: float) -> str:
    """Renders a pixel-perfect semi-circular SVG gauge meter for dataset drift share."""
    pct = max(0.0, min(1.0, drift_share))
    angle = pct * 180.0  # 0 to 180 degrees
    # Semi-circle radius 80, center at (100, 90)
    rad = math.radians(180 - angle)
    x_needle = 100 + 70 * math.cos(rad)
    y_needle = 90 - 70 * math.sin(rad)

    color = "#10b981" if pct < 0.15 else ("#f59e0b" if pct < 0.33 else "#ef4444")

    svg = f"""<svg width="200" height="110" viewBox="0 0 200 110" xmlns="http://www.w3.org/2000/svg">
      <path d="M 20 90 A 80 80 0 0 1 180 90" fill="none" stroke="#1e293b" stroke-width="16" stroke-linecap="round"/>
      <path d="M 20 90 A 80 80 0 0 1 70 33" fill="none" stroke="#10b981" stroke-width="16" opacity="0.8"/>
      <path d="M 70 33 A 80 80 0 0 1 130 33" fill="none" stroke="#f59e0b" stroke-width="16" opacity="0.8"/>
      <path d="M 130 33 A 80 80 0 0 1 180 90" fill="none" stroke="#ef4444" stroke-width="16" opacity="0.8"/>
      <!-- Needle -->
      <line x1="100" y1="90" x2="{round(x_needle, 1)}" y2="{round(y_needle, 1)}" stroke="{color}" stroke-width="3" stroke-linecap="round"/>
      <circle cx="100" cy="90" r="6" fill="#f8fafc"/>
      <text x="100" y="105" text-anchor="middle" fill="#f8fafc" font-size="14" font-weight="bold" font-family="system-ui">{round(pct * 100, 1)}%</text>
    </svg>"""
    return svg


def generate_mini_histogram_svg(result: FeatureDriftResult) -> str:
    """Generates an embedded SVG dual-distribution histogram comparing baseline vs current."""
    if not result.baseline_profile or not result.current_profile:
        return '<span style="color:#64748b;font-size:12px;">No histogram</span>'

    base_hist = result.baseline_profile.histogram.get("counts", [])
    curr_hist = result.current_profile.histogram.get("counts", [])

    if not base_hist or not curr_hist:
        return '<span style="color:#64748b;font-size:12px;">Sparse distribution</span>'

    max_cnt = max(max(base_hist, default=1), max(curr_hist, default=1), 1)
    svg_w, svg_h = 160, 48
    bar_w = (svg_w - 10) / (len(base_hist) * 2)

    rects = []
    for i in range(min(len(base_hist), len(curr_hist))):
        h1 = (base_hist[i] / max_cnt) * (svg_h - 8)
        h2 = (curr_hist[i] / max_cnt) * (svg_h - 8)
        x1 = 5 + i * (bar_w * 2 + 2)
        x2 = x1 + bar_w
        y1 = svg_h - h1 - 4
        y2 = svg_h - h2 - 4
        rects.append(f'<rect x="{round(x1, 1)}" y="{round(y1, 1)}" width="{round(bar_w, 1)}" height="{round(h1, 1)}" fill="#38bdf8" opacity="0.6"/>')
        rects.append(f'<rect x="{round(x2, 1)}" y="{round(y2, 1)}" width="{round(bar_w, 1)}" height="{round(h2, 1)}" fill="#f43f5e" opacity="0.8"/>')

    svg = f"""<svg width="{svg_w}" height="{svg_h}" viewBox="0 0 {svg_w} {svg_h}" xmlns="http://www.w3.org/2000/svg">
      <line x1="5" y1="{svg_h - 4}" x2="{svg_w - 5}" y2="{svg_h - 4}" stroke="#475569" stroke-width="1"/>
      {"".join(rects)}
    </svg>"""
    return svg


def generate_html_report(report: DatasetDriftReport, title: str = "DriftPulse-AI Executive Dashboard") -> str:
    """Renders a self-contained, offline, executive dark-mode HTML dashboard."""
    gauge_svg = generate_drift_gauge_svg(report.drift_share)

    # Feature rows
    rows = []
    for feat_name, res in report.feature_results.items():
        hist_svg = generate_mini_histogram_svg(res)
        status_bg = res.drift_status.badge_color
        status_text = res.drift_status.value.replace("_", " ").upper()

        psi_val = res.all_metrics.get("psi")
        ks_val = res.all_metrics.get("ks_test")
        wass_val = res.all_metrics.get("wasserstein")

        psi_str = f"{round(psi_val.value, 4)}" if psi_val else "-"
        ks_str = f"{round(ks_val.p_value, 4)}" if (ks_val and ks_val.p_value is not None) else "-"
        wass_str = f"{round(wass_val.value, 4)}" if wass_val else "-"

        warn_badge = ""
        if res.warnings:
            warn_badge = f'<div style="font-size:11px;color:#f59e0b;margin-top:4px;">⚠️ {res.warnings[0]}</div>'

        row = f"""
        <tr>
            <td style="font-weight:600;color:#f8fafc;">
                {feat_name}
                <div style="font-size:11px;color:#94a3b8;font-weight:normal;">{res.feature_type.value}</div>
                {warn_badge}
            </td>
            <td>
                <span class="badge" style="background-color:{status_bg}22;color:{status_bg};border:1px solid {status_bg}66;">
                    {status_text}
                </span>
            </td>
            <td><code style="color:#38bdf8;">{psi_str}</code></td>
            <td><code style="color:#a78bfa;">{ks_str}</code></td>
            <td><code style="color:#34d399;">{wass_str}</code></td>
            <td>{hist_svg}</td>
        </tr>
        """
        rows.append(row)

    status_color = report.overall_status.badge_color

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background-color: #0b0f19;
      color: #e2e8f0;
      padding: 32px 24px;
      line-height: 1.5;
    }}
    .container {{
      max-width: 1200px;
      margin: 0 auto;
    }}
    .header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding-bottom: 24px;
      border-bottom: 1px solid #1e293b;
      margin-bottom: 32px;
    }}
    .logo-title {{
      display: flex;
      align-items: center;
      gap: 12px;
    }}
    .logo-badge {{
      background: linear-gradient(135deg, #0ea5e9, #6366f1);
      color: white;
      font-weight: 800;
      font-size: 18px;
      padding: 8px 14px;
      border-radius: 8px;
      letter-spacing: 0.5px;
    }}
    .title-text h1 {{
      font-size: 24px;
      color: #f8fafc;
      font-weight: 700;
    }}
    .title-text p {{
      font-size: 13px;
      color: #94a3b8;
    }}
    .grid-cards {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 20px;
      margin-bottom: 32px;
    }}
    .card {{
      background-color: #111827;
      border: 1px solid #1f2937;
      border-radius: 14px;
      padding: 20px;
      position: relative;
      overflow: hidden;
    }}
    .card-label {{
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: #94a3b8;
      margin-bottom: 8px;
    }}
    .card-value {{
      font-size: 28px;
      font-weight: 700;
      color: #f8fafc;
    }}
    .gauge-card {{
      grid-column: span 1;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      text-align: center;
    }}
    .section-title {{
      font-size: 18px;
      font-weight: 600;
      color: #f8fafc;
      margin-bottom: 16px;
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background-color: #111827;
      border-radius: 14px;
      overflow: hidden;
      border: 1px solid #1f2937;
      margin-bottom: 32px;
    }}
    th {{
      background-color: #1e293b;
      color: #cbd5e1;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      text-align: left;
      padding: 14px 18px;
    }}
    td {{
      padding: 14px 18px;
      border-bottom: 1px solid #1e293b;
      font-size: 14px;
      vertical-align: middle;
    }}
    tr:last-child td {{
      border-bottom: none;
    }}
    tr:hover td {{
      background-color: #162033;
    }}
    .badge {{
      display: inline-block;
      padding: 4px 10px;
      border-radius: 6px;
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.5px;
    }}
    .legend {{
      display: flex;
      gap: 16px;
      font-size: 12px;
      color: #94a3b8;
      margin-top: 8px;
    }}
    .legend-item {{
      display: flex;
      align-items: center;
      gap: 6px;
    }}
    .legend-box {{
      width: 12px;
      height: 12px;
      border-radius: 2px;
    }}
    .remediation-box {{
      background-color: #111827;
      border: 1px solid #334155;
      border-left: 4px solid {status_color};
      border-radius: 12px;
      padding: 20px;
      margin-bottom: 32px;
    }}
    .footer {{
      text-align: center;
      font-size: 12px;
      color: #64748b;
      padding-top: 24px;
      border-top: 1px solid #1e293b;
    }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <div class="logo-title">
        <div class="logo-badge">DP</div>
        <div class="title-text">
          <h1>{report.dataset_name} &bull; Drift Sentinel Report</h1>
          <p>Generated at: {report.timestamp} &bull; DriftPulse-AI v1.0</p>
        </div>
      </div>
      <div>
        <span class="badge" style="background-color:{status_color}22;color:{status_color};border:1px solid {status_color}66;font-size:14px;padding:8px 16px;">
          {report.overall_status.value.replace("_", " ").upper()}
        </span>
      </div>
    </div>

    <div class="grid-cards">
      <div class="card">
        <div class="card-label">Total Features</div>
        <div class="card-value">{report.total_features}</div>
        <div style="font-size:12px;color:#64748b;margin-top:6px;">Monitored dimensions</div>
      </div>
      <div class="card">
        <div class="card-label">Drifted Features</div>
        <div class="card-value" style="color:{status_color};">{report.drifted_features}</div>
        <div style="font-size:12px;color:#64748b;margin-top:6px;">{round(report.drift_share * 100, 1)}% of total</div>
      </div>
      <div class="card">
        <div class="card-label">Dataset Drift Share</div>
        <div class="card-value">{round(report.drift_share * 100, 1)}%</div>
        <div style="font-size:12px;color:#64748b;margin-top:6px;">Alert threshold: 33%</div>
      </div>
      <div class="card gauge-card">
        <div class="card-label" style="margin-bottom:0;">Drift Gauge</div>
        {gauge_svg}
      </div>
    </div>

    <div class="remediation-box">
      <h3 style="font-size:16px;color:#f8fafc;margin-bottom:8px;">Executive Root-Cause & Actionable Engineering Remediation</h3>
      <p style="font-size:13px;color:#cbd5e1;margin-bottom:12px;">
        {'CRITICAL ACTION REQUIRED: Model inference outputs are degraded due to severe input covariate shifts. Retraining or fallback to fallback rule-based system recommended.' if report.has_dataset_drift else 'SYSTEM HEALTHY: Feature distributions conform to reference baselines within standard statistical tolerances.'}
      </p>
      <ul style="font-size:13px;color:#94a3b8;padding-left:20px;line-height:1.8;">
        <li><strong>Continuous CI/CD Gate:</strong> Ensure pre-deploy test suites evaluate PSI < 0.10 before promoting candidate models.</li>
        <li><strong>Data Imputation:</strong> Monitor missing value spikes in input APIs.</li>
        <li><strong>Dual-Histogram Comparison:</strong> Cyan = Reference Distribution, Rose = Production Distribution.</li>
      </ul>
    </div>

    <div class="section-title">
      Feature Distribution Diagnostics & Statistical Tests
    </div>

    <table>
      <thead>
        <tr>
          <th>Feature</th>
          <th>Status</th>
          <th>PSI</th>
          <th>KS p-value</th>
          <th>Wasserstein</th>
          <th>Distribution Shift (Ref vs Curr)</th>
        </tr>
      </thead>
      <tbody>
        {"".join(rows)}
      </tbody>
    </table>

    <div class="legend">
      <div class="legend-item"><div class="legend-box" style="background-color:#38bdf8;"></div> Reference Baseline</div>
      <div class="legend-item"><div class="legend-box" style="background-color:#f43f5e;"></div> Production Window</div>
    </div>

    <div class="footer">
      Built with precision by Nitesh Beniwal &bull; DriftPulse-AI &bull; Production Statistical Sentinel for ML & LLM Pipelines
    </div>
  </div>
</body>
</html>
"""
    return html

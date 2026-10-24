"""Command-line interface (CLI) for DriftPulse-AI."""

from __future__ import annotations

import argparse
import os
import random
import sys
from typing import List, Optional

from driftpulse.core.config import ThresholdConfig
from driftpulse.core.profiler import DataProfiler
from driftpulse.detectors.tabular_detector import TabularDriftDetector
from driftpulse.integrations.pandas_adapter import load_csv_as_dict, to_standard_dict
from driftpulse.monitoring.baseline import BaselineStore
from driftpulse.reporting.html_report import generate_html_report
from driftpulse.reporting.json_exporter import export_report_to_json
from driftpulse.reporting.summary_formatter import format_terminal_summary

VERSION = "1.0.0"


def build_parser() -> argparse.ArgumentParser:
    """Constructs the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="driftpulse",
        description="DriftPulse-AI: Production Data & Concept Drift Sentinel for ML & LLM Pipelines",
    )
    parser.add_argument("--version", "-v", action="version", version=f"DriftPulse-AI v{VERSION}")

    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: profile
    p_profile = subparsers.add_parser("profile", help="Generate and save baseline statistical profile")
    p_profile.add_argument("data", help="Path to reference CSV file")
    p_profile.add_argument("--out", "-o", default="baseline_profile.json", help="Output path for baseline JSON")
    p_profile.add_argument("--name", default="Reference_Baseline", help="Baseline dataset label")

    # Command: compare
    p_compare = subparsers.add_parser("compare", help="Compare production data against baseline")
    p_compare.add_argument("reference", help="Reference CSV or baseline JSON file")
    p_compare.add_argument("current", help="Current production CSV file")
    p_compare.add_argument("--html", help="Path to generate standalone executive HTML dashboard")
    p_compare.add_argument("--json", help="Path to export machine-readable JSON telemetry")
    p_compare.add_argument("--name", default="Production_Inference_Window", help="Report name")

    # Command: quicktest
    subparsers.add_parser("quicktest", help="Run self-contained synthetic benchmark and demo report")

    return parser


def run_quicktest() -> int:
    """Executes a self-contained demonstration comparing normal baseline vs drifted production data."""
    print("Running DriftPulse-AI QuickTest Simulation...")
    random.seed(42)

    # Reference distribution (Normal: mean 50, std 10)
    ref_data = {
        "customer_age": [random.gauss(45, 8) for _ in range(500)],
        "annual_income": [random.gauss(65000, 15000) for _ in range(500)],
        "credit_score": [random.gauss(710, 40) for _ in range(500)],
        "account_type": random.choices(["CHECKING", "SAVINGS", "PREMIUM"], weights=[0.6, 0.3, 0.1], k=500),
    }

    # Drifted production distribution (Severe covariate shift in income & credit score)
    curr_data = {
        "customer_age": [random.gauss(46, 8.2) for _ in range(500)],  # Stable
        "annual_income": [random.gauss(88000, 22000) for _ in range(500)],  # Severe Drift
        "credit_score": [random.gauss(640, 55) for _ in range(500)],  # Severe Drift
        "account_type": random.choices(["CHECKING", "SAVINGS", "PREMIUM"], weights=[0.2, 0.2, 0.6], k=500),  # Categorical Drift
    }

    detector = TabularDriftDetector()
    report = detector.compare_datasets(ref_data, curr_data, dataset_name="QuickTest_FinTech_Stream")

    print("\n" + format_terminal_summary(report))

    html_out = "quicktest_drift_report.html"
    with open(html_out, "w", encoding="utf-8") as f:
        f.write(generate_html_report(report))
    print(f"\n[+] Standalone Executive HTML Dashboard generated: {os.path.abspath(html_out)}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    if args.command == "quicktest":
        return run_quicktest()

    if args.command == "profile":
        print(f"Profiling reference dataset: {args.data}...")
        raw_dict = to_standard_dict(args.data)
        store = BaselineStore()
        baseline = store.create_baseline(raw_dict, baseline_name=args.name)
        store.save(baseline, args.out)
        print(f"[+] Baseline profile successfully saved to {args.out}")
        return 0

    if args.command == "compare":
        print(f"Comparing current dataset ({args.current}) against reference ({args.reference})...")
        ref_dict = to_standard_dict(args.reference)
        curr_dict = to_standard_dict(args.current)

        detector = TabularDriftDetector()
        report = detector.compare_datasets(ref_dict, curr_dict, dataset_name=args.name)

        print("\n" + format_terminal_summary(report))

        if args.html:
            with open(args.html, "w", encoding="utf-8") as f:
                f.write(generate_html_report(report))
            print(f"[+] HTML Dashboard saved to: {args.html}")

        if args.json:
            export_report_to_json(report, args.json)
            print(f"[+] JSON report saved to: {args.json}")

        return 1 if report.has_dataset_drift else 0

    return 0


if __name__ == "__main__":
    sys.exit(main())

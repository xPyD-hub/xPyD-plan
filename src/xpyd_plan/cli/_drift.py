"""CLI drift command."""

from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console
from rich.table import Table

from xpyd_plan.bench_adapter import load_benchmark_auto
from xpyd_plan.drift import DriftDetector, DriftSeverity


def _cmd_drift(args: argparse.Namespace) -> None:
    """Handle the 'drift' subcommand."""
    console = Console()

    baseline = load_benchmark_auto(args.baseline)
    current = load_benchmark_auto(args.current)

    detector = DriftDetector()
    report = detector.detect(baseline, current)

    output_format = getattr(args, "output_format", "table")
    if output_format == "json":
        json.dump(report.model_dump(), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return

    # Table output
    table = Table(title="Distribution Drift Detection")
    table.add_column("Metric", justify="left")
    table.add_column("KS Statistic", justify="right")
    table.add_column("p-value", justify="right")
    table.add_column("Severity", justify="center")
    table.add_column("Baseline Mean", justify="right")
    table.add_column("Current Mean", justify="right")
    table.add_column("Shift", justify="right")

    severity_style = {
        DriftSeverity.NONE: "green",
        DriftSeverity.MINOR: "yellow",
        DriftSeverity.MODERATE: "dark_orange",
        DriftSeverity.MAJOR: "bold red",
    }

    for r in report.results:
        style = severity_style[r.severity]
        shift_str = f"{r.mean_shift_ms:+.1f}ms"
        table.add_row(
            r.metric,
            f"{r.ks_statistic:.4f}",
            f"{r.p_value:.4f}",
            f"[{style}]{r.severity.value}[/{style}]",
            f"{r.baseline_mean_ms:.1f}ms",
            f"{r.current_mean_ms:.1f}ms",
            shift_str,
        )

    console.print(table)
    console.print()

    overall_style = severity_style[report.overall_severity]
    console.print(
        f"Overall: [{overall_style}]{report.overall_severity.value}[/{overall_style}]"
    )
    if report.drifted_metrics:
        console.print(f"Drifted metrics: {', '.join(report.drifted_metrics)}")
    console.print()
    console.print(f"[bold]{report.recommendation}[/bold]")


def add_drift_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the drift subcommand parser."""
    parser = subparsers.add_parser(
        "drift",
        help="Detect distribution drift between two benchmark runs (KS test)",
    )
    parser.add_argument(
        "--baseline",
        required=True,
        help="Path to baseline benchmark JSON file",
    )
    parser.add_argument(
        "--current",
        required=True,
        help="Path to current benchmark JSON file",
    )
    parser.add_argument(
        "--output-format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )
    parser.set_defaults(func=_cmd_drift)

"""CLI outlier-impact command."""

from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console
from rich.table import Table

from xpyd_plan.bench_adapter import load_benchmark_auto
from xpyd_plan.outlier_impact import ImpactRecommendation, OutlierImpactAnalyzer


def _cmd_outlier_impact(args: argparse.Namespace) -> None:
    """Handle the 'outlier-impact' subcommand."""
    console = Console()

    data = load_benchmark_auto(args.benchmark)
    analyzer = OutlierImpactAnalyzer()
    report = analyzer.analyze(
        data,
        iqr_multiplier=args.iqr_multiplier,
        sla_ttft_ms=getattr(args, "sla_ttft", None),
        sla_tpot_ms=getattr(args, "sla_tpot", None),
        sla_total_ms=getattr(args, "sla_total", None),
    )

    output_format = getattr(args, "output_format", "table")
    if output_format == "json":
        json.dump(report.model_dump(), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return

    # Summary
    console.print("\n[bold]Outlier Impact Analysis[/bold]")
    console.print(
        f"Total requests: {report.total_requests}  |  "
        f"Outliers: {report.outlier_count} ({report.outlier_fraction:.1%})  |  "
        f"IQR multiplier: {report.iqr_multiplier}"
    )
    console.print()

    # Metric impacts table
    table = Table(title="Per-Metric Latency Impact")
    table.add_column("Metric", justify="left")
    table.add_column("P50 Before", justify="right")
    table.add_column("P50 After", justify="right")
    table.add_column("P50 Δ", justify="right")
    table.add_column("P95 Before", justify="right")
    table.add_column("P95 After", justify="right")
    table.add_column("P95 Δ", justify="right")
    table.add_column("P99 Before", justify="right")
    table.add_column("P99 After", justify="right")
    table.add_column("P99 Δ", justify="right")

    for mi in report.metric_impacts:
        table.add_row(
            mi.metric,
            f"{mi.p50_before_ms:.1f}",
            f"{mi.p50_after_ms:.1f}",
            f"{mi.p50_delta_ms:+.1f}",
            f"{mi.p95_before_ms:.1f}",
            f"{mi.p95_after_ms:.1f}",
            f"{mi.p95_delta_ms:+.1f}",
            f"{mi.p99_before_ms:.1f}",
            f"{mi.p99_after_ms:.1f}",
            f"{mi.p99_delta_ms:+.1f}",
        )
    console.print(table)
    console.print()

    # SLA comparison
    if report.sla_comparison is not None:
        sc = report.sla_comparison
        before_style = "green" if sc.pass_before else "red"
        after_style = "green" if sc.pass_after else "red"
        console.print("[bold]SLA Compliance[/bold]")
        console.print(
            f"  Before: [{before_style}]{'PASS' if sc.pass_before else 'FAIL'}[/]  "
            f"  After:  [{after_style}]{'PASS' if sc.pass_after else 'FAIL'}[/]  "
            f"  Flipped: {'Yes' if sc.flipped else 'No'}"
        )
        console.print()

    # Recommendation
    rec_style = (
        "bold yellow" if report.recommendation == ImpactRecommendation.FILTER
        else "green"
    )
    console.print(
        f"Recommendation: [{rec_style}]{report.recommendation.value.upper()}[/]"
    )
    console.print(f"[dim]{report.summary}[/dim]")


def add_outlier_impact_parser(
    subparsers: argparse._SubParsersAction,
) -> None:
    """Add the outlier-impact subcommand parser."""
    parser = subparsers.add_parser(
        "outlier-impact",
        help="Analyze how outliers affect latency percentiles and SLA compliance",
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        help="Path to benchmark JSON file",
    )
    parser.add_argument(
        "--iqr-multiplier",
        type=float,
        default=1.5,
        help="IQR multiplier for outlier detection (default: 1.5)",
    )
    parser.add_argument(
        "--sla-ttft",
        type=float,
        default=None,
        help="TTFT SLA threshold in ms",
    )
    parser.add_argument(
        "--sla-tpot",
        type=float,
        default=None,
        help="TPOT SLA threshold in ms",
    )
    parser.add_argument(
        "--sla-total",
        type=float,
        default=None,
        help="Total latency SLA threshold in ms",
    )
    parser.add_argument(
        "--output-format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )
    parser.set_defaults(func=_cmd_outlier_impact)

"""CLI convergence command."""

from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console
from rich.table import Table

from xpyd_plan.bench_adapter import load_benchmark_auto
from xpyd_plan.convergence import ConvergenceAnalyzer, StabilityStatus


def _cmd_convergence(args: argparse.Namespace) -> None:
    """Handle the 'convergence' subcommand."""
    console = Console()

    data = load_benchmark_auto(args.benchmark)
    analyzer = ConvergenceAnalyzer(data)
    report = analyzer.analyze(
        steps=args.steps,
        threshold=args.threshold,
    )

    output_format = getattr(args, "output_format", "table")
    if output_format == "json":
        json.dump(report.model_dump(), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return

    # Summary
    status_style = {
        StabilityStatus.STABLE: "bold green",
        StabilityStatus.MARGINAL: "bold yellow",
        StabilityStatus.UNSTABLE: "bold red",
    }

    console.print(f"Total requests: {report.total_requests}")
    style = status_style[report.overall_status]
    console.print(f"Overall stability: [{style}]{report.overall_status.value}[/{style}]")
    if report.recommended_min_requests is not None:
        console.print(f"Recommended min requests: {report.recommended_min_requests}")
    console.print()

    # Per-metric table
    table = Table(title="Percentile Convergence Analysis")
    table.add_column("Field", justify="left")
    table.add_column("CV(P95)", justify="right")
    table.add_column("CV(P99)", justify="right")
    table.add_column("Status", justify="center")
    table.add_column("Min Stable Size", justify="right")

    for m in report.metrics:
        style = status_style[m.status]
        table.add_row(
            m.field,
            f"{m.cv_p95:.4f}",
            f"{m.cv_p99:.4f}",
            f"[{style}]{m.status.value}[/{style}]",
            str(m.min_stable_sample_size) if m.min_stable_sample_size else "N/A",
        )

    console.print(table)


def add_convergence_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the convergence subcommand parser."""
    parser = subparsers.add_parser(
        "convergence",
        help="Analyze percentile convergence (are metrics stable with enough samples?)",
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        help="Path to benchmark JSON file",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="Number of cumulative windows (default: 10)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="CV threshold for STABLE classification (default: 0.05)",
    )
    parser.add_argument(
        "--output-format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )
    parser.set_defaults(func=_cmd_convergence)

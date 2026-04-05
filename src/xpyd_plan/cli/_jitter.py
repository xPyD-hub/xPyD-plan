"""CLI jitter command."""

from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console
from rich.table import Table

from xpyd_plan.bench_adapter import load_benchmark_auto
from xpyd_plan.jitter import JitterAnalyzer, JitterClassification


def _cmd_jitter(args: argparse.Namespace) -> None:
    """Handle the 'jitter' subcommand."""
    console = Console()

    data = load_benchmark_auto(args.benchmark)
    analyzer = JitterAnalyzer()
    report = analyzer.analyze(data)

    output_format = getattr(args, "output_format", "table")
    if output_format == "json":
        json.dump(report.model_dump(), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return

    # Table output
    table = Table(title="Latency Jitter Analysis")
    table.add_column("Metric", justify="left")
    table.add_column("Consecutive Mean", justify="right")
    table.add_column("Consecutive P95", justify="right")
    table.add_column("Std Dev", justify="right")
    table.add_column("IQR", justify="right")
    table.add_column("CV", justify="right")
    table.add_column("Classification", justify="center")

    for m in report.metrics:
        style = {
            JitterClassification.STABLE: "green",
            JitterClassification.MODERATE: "yellow",
            JitterClassification.HIGH: "bold red",
        }[m.stats.classification]

        table.add_row(
            m.metric,
            f"{m.stats.consecutive_mean:.2f}",
            f"{m.stats.consecutive_p95:.2f}",
            f"{m.stats.std:.2f}",
            f"{m.stats.iqr:.2f}",
            f"[{style}]{m.stats.cv:.4f}[/{style}]",
            f"[{style}]{m.stats.classification.value}[/{style}]",
        )

    console.print(table)
    console.print()
    console.print(f"[bold]{report.recommendation}[/bold]")


def add_jitter_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the jitter subcommand parser."""
    parser = subparsers.add_parser(
        "jitter",
        help="Analyze latency jitter (request-to-request variability)",
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        help="Path to benchmark JSON file",
    )
    parser.add_argument(
        "--output-format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )
    parser.set_defaults(func=_cmd_jitter)

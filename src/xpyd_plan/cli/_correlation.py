"""CLI correlation command."""

from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console
from rich.table import Table

from xpyd_plan.bench_adapter import load_benchmark_auto
from xpyd_plan.correlation import CorrelationAnalyzer, CorrelationStrength


def _cmd_correlation(args: argparse.Namespace) -> None:
    """Handle the 'correlation' subcommand."""
    console = Console()

    data = load_benchmark_auto(args.benchmark)
    analyzer = CorrelationAnalyzer()
    report = analyzer.analyze(data)

    output_format = getattr(args, "output_format", "table")
    if output_format == "json":
        json.dump(report.model_dump(), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return

    # Table output
    table = Table(title="Correlation Analysis")
    table.add_column("Request Metric", justify="left")
    table.add_column("Latency Metric", justify="left")
    table.add_column("Pearson r", justify="right")
    table.add_column("Strength", justify="center")

    for pair in report.pairs:
        style = {
            CorrelationStrength.STRONG: "bold red",
            CorrelationStrength.MODERATE: "yellow",
            CorrelationStrength.WEAK: "dim",
            CorrelationStrength.NEGLIGIBLE: "dim",
        }[pair.strength]

        table.add_row(
            pair.x_metric,
            pair.y_metric,
            f"[{style}]{pair.pearson_r:+.4f}[/{style}]",
            f"[{style}]{pair.strength.value}[/{style}]",
        )

    console.print(table)
    console.print()
    console.print(f"[bold]{report.recommendation}[/bold]")


def add_correlation_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the correlation subcommand parser."""
    parser = subparsers.add_parser(
        "correlation",
        help="Analyze correlations between request characteristics and latency",
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
    parser.set_defaults(func=_cmd_correlation)

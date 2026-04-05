"""CLI anomaly-classify command."""

from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console
from rich.table import Table

from xpyd_plan.anomaly_classifier import LatencyAnomalyClassifier
from xpyd_plan.bench_adapter import load_benchmark_auto


def _cmd_anomaly_classify(args: argparse.Namespace) -> None:
    """Handle the 'anomaly-classify' subcommand."""
    console = Console()

    data = load_benchmark_auto(args.benchmark)
    classifier = LatencyAnomalyClassifier()
    report = classifier.classify(
        data,
        slow_multiplier=args.slow_multiplier,
        outlier_multiplier=args.outlier_multiplier,
        timeout_ttft=args.timeout_ttft,
        timeout_tpot=args.timeout_tpot,
        timeout_total=args.timeout_total,
    )

    output_format = getattr(args, "output_format", "table")
    if output_format == "json":
        json.dump(report.model_dump(), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return

    # Distribution table
    table = Table(title="Anomaly Classification Distribution")
    table.add_column("Metric", justify="left")
    table.add_column("Normal", justify="right")
    table.add_column("Slow", justify="right")
    table.add_column("Outlier", justify="right")
    table.add_column("Timeout", justify="right")

    for d in report.distributions:
        table.add_row(
            d.metric,
            f"{d.normal_count} ({d.normal_pct:.1f}%)",
            f"[yellow]{d.slow_count} ({d.slow_pct:.1f}%)[/yellow]",
            f"[red]{d.outlier_count} ({d.outlier_pct:.1f}%)[/red]",
            f"[bold red]{d.timeout_count} ({d.timeout_pct:.1f}%)[/bold red]",
        )

    console.print(table)
    console.print()
    console.print(
        f"Total anomalous: {report.total_anomalous}/{len(report.labels)} "
        f"({report.anomaly_rate:.1f}%)"
    )
    console.print(f"[bold]{report.recommendation}[/bold]")


def add_anomaly_classify_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the anomaly-classify subcommand parser."""
    parser = subparsers.add_parser(
        "anomaly-classify",
        help="Classify requests as NORMAL, SLOW, OUTLIER, or TIMEOUT",
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        help="Path to benchmark JSON file",
    )
    parser.add_argument(
        "--slow-multiplier",
        type=float,
        default=1.0,
        help="IQR multiplier for SLOW threshold (default: 1.0)",
    )
    parser.add_argument(
        "--outlier-multiplier",
        type=float,
        default=3.0,
        help="IQR multiplier for OUTLIER threshold (default: 3.0)",
    )
    parser.add_argument(
        "--timeout-ttft",
        type=float,
        default=None,
        help="Absolute TTFT timeout threshold in ms",
    )
    parser.add_argument(
        "--timeout-tpot",
        type=float,
        default=None,
        help="Absolute TPOT timeout threshold in ms",
    )
    parser.add_argument(
        "--timeout-total",
        type=float,
        default=None,
        help="Absolute total latency timeout threshold in ms",
    )
    parser.add_argument(
        "--output-format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )
    parser.set_defaults(func=_cmd_anomaly_classify)
